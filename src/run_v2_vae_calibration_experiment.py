from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from run_v2_torch_vae_experiment import ConditionalVAE, PAIR_ORDER, RANDOM_SEED, SOURCE_PAIRS, TARGET_PAIRS, flatten, fit_standardisation, load_matrices, split_name, train_model, vectorise, mask_for_scenario
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'
SCENARIOS = ('random_20_percent', 'full_3m_tenor', 'full_10d_wings')
CALIBRATION_QUANTILE = 0.9

def train_transfer_vae(train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device) -> tuple[ConditionalVAE, dict[str, object]]:
    model = ConditionalVAE(pair_count=len(PAIR_ORDER)).to(device)
    pretrain_history = train_model(model, train_arrays, SOURCE_PAIRS, device, epochs=140, learning_rate=0.001, beta=0.002, batch_size=2048, random_fraction=0.2, wing_fraction=0.45)
    finetune_history = train_model(model, train_arrays, TARGET_PAIRS, device, epochs=100, learning_rate=0.0005, beta=0.002, batch_size=1024, random_fraction=0.2, wing_fraction=0.55)
    return (model, {'pretrain': pretrain_history, 'finetune': finetune_history})

def collect_prediction_records(model: ConditionalVAE, matrices: list[dict[str, object]], arrays: tuple[np.ndarray, np.ndarray, np.ndarray], standardisation, device: torch.device, scenario: str, samples: int=16) -> list[dict[str, object]]:
    values_np, observed_np, pair_np = arrays
    selected_rows = [row_index for row_index, matrix in enumerate(matrices) if str(matrix['pair']) in TARGET_PAIRS]
    eval_masks = np.vstack([mask_for_scenario(observed_np[row_index], scenario, seed=RANDOM_SEED + row_index) for row_index in selected_rows]).astype(np.float32)
    hidden_masks = (observed_np[selected_rows] > 0.5) & (eval_masks < 0.5)
    values_tensor = torch.tensor(values_np[selected_rows], dtype=torch.float32)
    mask_tensor = torch.tensor(eval_masks, dtype=torch.float32)
    pair_tensor = torch.tensor(pair_np[selected_rows], dtype=torch.float32)
    prediction_means = np.zeros((len(selected_rows), 25), dtype=np.float32)
    prediction_lows = np.zeros((len(selected_rows), 25), dtype=np.float32)
    prediction_highs = np.zeros((len(selected_rows), 25), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected_rows), 512):
            end = start + 512
            batch_values = values_tensor[start:end].to(device)
            batch_masks = mask_tensor[start:end].to(device)
            batch_pairs = pair_tensor[start:end].to(device)
            mu, logvar = model.encode(batch_values, batch_masks, batch_pairs)
            batch_predictions = []
            for _ in range(samples):
                latent = model.reparameterise(mu, logvar)
                batch_predictions.append(model.decode(latent, batch_pairs).cpu().numpy())
            sample_cube = np.stack(batch_predictions, axis=0)
            prediction_means[start:end] = sample_cube.mean(axis=0)
            prediction_lows[start:end] = np.quantile(sample_cube, 0.05, axis=0)
            prediction_highs[start:end] = np.quantile(sample_cube, 0.95, axis=0)
    records = []
    for selected_index, row_index in enumerate(selected_rows):
        matrix = matrices[row_index]
        pair = str(matrix['pair'])
        means = standardisation.pair_means[pair]
        stds = standardisation.pair_stds[pair]
        actual_values = np.array([float(value) if value is not None else np.nan for value in flatten(matrix['values'])], dtype=np.float32)
        spreads = np.array([float(spread) if spread is not None else np.nan for spread in flatten(matrix['spreads'])], dtype=np.float32)
        predicted_values = means + prediction_means[selected_index] * stds
        raw_low_values = means + prediction_lows[selected_index] * stds
        raw_high_values = means + prediction_highs[selected_index] * stds
        for cell_index in np.where(hidden_masks[selected_index])[0]:
            actual = float(actual_values[cell_index])
            prediction = float(predicted_values[cell_index])
            spread = float(spreads[cell_index])
            records.append({'date': str(matrix['date']), 'pair': pair, 'cell_index': int(cell_index), 'actual': actual, 'prediction': prediction, 'error': actual - prediction, 'absolute_error': abs(actual - prediction), 'spread': None if np.isnan(spread) else spread, 'raw_low': float(raw_low_values[cell_index]), 'raw_high': float(raw_high_values[cell_index])})
    return records

def fit_calibration(records: list[dict[str, object]]) -> dict[str, object]:
    absolute_errors = np.array([float(record['absolute_error']) for record in records])
    global_width = float(np.quantile(absolute_errors, CALIBRATION_QUANTILE))
    errors_by_cell: dict[int, list[float]] = defaultdict(list)
    scaled_errors = []
    for record in records:
        errors_by_cell[int(record['cell_index'])].append(float(record['absolute_error']))
        spread = record['spread']
        if spread is not None and float(spread) > 0:
            scaled_errors.append(float(record['absolute_error']) / float(spread))
    cell_widths = {str(cell_index): float(np.quantile(values, CALIBRATION_QUANTILE)) if len(values) >= 30 else global_width for cell_index, values in errors_by_cell.items()}
    spread_scale = float(np.quantile(np.array(scaled_errors), CALIBRATION_QUANTILE)) if scaled_errors else None
    return {'quantile': CALIBRATION_QUANTILE, 'global_width': global_width, 'cell_widths': cell_widths, 'spread_scale': spread_scale, 'validation_count': len(records)}

def evaluate_interval(records: list[dict[str, object]], lower_key: str, upper_key: str) -> dict[str, float]:
    widths = []
    hits = 0
    for record in records:
        lower = float(record[lower_key])
        upper = float(record[upper_key])
        actual = float(record['actual'])
        widths.append(upper - lower)
        hits += int(lower <= actual <= upper)
    return {'coverage': hits / len(records), 'average_width': sum(widths) / len(widths)}

def evaluate_calibrated_intervals(records: list[dict[str, object]], calibration: dict[str, object]) -> dict[str, object]:
    raw_hits = 0
    raw_widths = []
    global_hits = 0
    global_widths = []
    cell_hits = 0
    cell_widths = []
    spread_hits = 0
    spread_widths = []
    global_width = float(calibration['global_width'])
    spread_scale = calibration['spread_scale']
    for record in records:
        actual = float(record['actual'])
        prediction = float(record['prediction'])
        raw_low = float(record['raw_low'])
        raw_high = float(record['raw_high'])
        raw_hits += int(raw_low <= actual <= raw_high)
        raw_widths.append(raw_high - raw_low)
        lower = prediction - global_width
        upper = prediction + global_width
        global_hits += int(lower <= actual <= upper)
        global_widths.append(upper - lower)
        cell_width = float(calibration['cell_widths'].get(str(record['cell_index']), global_width))
        lower = prediction - cell_width
        upper = prediction + cell_width
        cell_hits += int(lower <= actual <= upper)
        cell_widths.append(upper - lower)
        spread = record['spread']
        if spread_scale is not None and spread is not None and (float(spread) > 0):
            interval_width = float(spread_scale) * float(spread)
        else:
            interval_width = global_width
        lower = prediction - interval_width
        upper = prediction + interval_width
        spread_hits += int(lower <= actual <= upper)
        spread_widths.append(upper - lower)
    absolute_errors = [float(record['absolute_error']) for record in records]
    return {'point_mae': sum(absolute_errors) / len(absolute_errors), 'point_rmse': math.sqrt(sum((error * error for error in absolute_errors)) / len(absolute_errors)), 'raw_vae_interval': {'coverage': raw_hits / len(records), 'average_width': sum(raw_widths) / len(raw_widths)}, 'global_residual_interval': {'coverage': global_hits / len(records), 'average_width': sum(global_widths) / len(global_widths), 'width_parameter': global_width}, 'cell_residual_interval': {'coverage': cell_hits / len(records), 'average_width': sum(cell_widths) / len(cell_widths)}, 'spread_scaled_interval': {'coverage': spread_hits / len(records), 'average_width': sum(spread_widths) / len(spread_widths), 'spread_scale': spread_scale}, 'count': len(records)}

def write_csv(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_vae_calibration_comparison.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['scenario', 'method', 'coverage', 'average_width', 'point_mae', 'count'])
        for scenario, scenario_results in results['scenarios'].items():
            for method in ('raw_vae_interval', 'global_residual_interval', 'cell_residual_interval', 'spread_scaled_interval'):
                metrics = scenario_results['test_metrics'][method]
                writer.writerow([scenario, method, metrics['coverage'], metrics['average_width'], scenario_results['test_metrics']['point_mae'], scenario_results['test_metrics']['count']])
    return path

def write_report(results: dict[str, object]) -> Path:
    rows = []
    for scenario, scenario_results in results['scenarios'].items():
        for label, method in (('Raw VAE', 'raw_vae_interval'), ('Global residual', 'global_residual_interval'), ('Cell residual', 'cell_residual_interval'), ('Spread-scaled', 'spread_scaled_interval')):
            metrics = scenario_results['test_metrics'][method]
            rows.append('| ' + ' | '.join([scenario, label, f"{metrics['coverage'] * 100:.1f}%", f"{metrics['average_width']:.4f}", f"{scenario_results['test_metrics']['point_mae']:.4f}", f"{scenario_results['test_metrics']['count']:,}"]) + ' |')
    report = f"# V2 VAE Calibration Experiment\n\n## Purpose\n\nUse validation-set errors to convert the VAE's overconfident raw uncertainty\nintervals into calibrated 90% reference intervals.\n\n## Method\n\nThe VAE is trained only on the training period. Calibration widths are fitted on\nthe validation period only. Final coverage is measured on the untouched test\nperiod.\n\nThree post-hoc calibration methods are tested:\n\n- Global residual: one 90th-percentile absolute-error width for all cells.\n- Cell residual: separate 90th-percentile width by matrix cell.\n- Spread-scaled: one 90th-percentile `absolute error / bid-ask spread` scale.\n\n## Test Results\n\n| Scenario | Interval Method | Test Coverage | Average Width | Point MAE | Hidden Quotes |\n|---|---|---:|---:|---:|---:|\n{chr(10).join(rows)}\n\n## Interpretation\n\nCalibration does not change the VAE point prediction. It only changes the\nreported uncertainty interval. A good method should move coverage closer to the\nnominal 90% target without making intervals unusably wide.\n\n## Reproducibility\n\n- Machine-readable results: `results/v2_vae_calibration_experiment.json`\n- CSV: `outputs/v2_vae_calibration_comparison.csv`\n- Runner: `src/run_v2_vae_calibration_experiment.py`\n- Random seed: `{RANDOM_SEED}`\n"
    path = RESULTS_DIR / 'v2_vae_calibration_experiment.md'
    path.write_text(report, encoding='utf-8')
    return path

def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrices = load_matrices()
    train = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'train']
    validation = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'validation']
    test = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'test']
    standardisation = fit_standardisation(train)
    train_arrays = vectorise(train, standardisation)
    validation_arrays = vectorise(validation, standardisation)
    test_arrays = vectorise(test, standardisation)
    model, history = train_transfer_vae(train_arrays, device)
    results: dict[str, object] = {'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'torch_version': torch.__version__, 'seed': RANDOM_SEED, 'calibration_quantile': CALIBRATION_QUANTILE, 'training_history': history, 'scenarios': {}}
    for scenario in SCENARIOS:
        validation_records = collect_prediction_records(model, validation, validation_arrays, standardisation, device, scenario, samples=16)
        calibration = fit_calibration(validation_records)
        test_records = collect_prediction_records(model, test, test_arrays, standardisation, device, scenario, samples=16)
        results['scenarios'][scenario] = {'calibration': calibration, 'validation_count': len(validation_records), 'test_metrics': evaluate_calibrated_intervals(test_records, calibration)}
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / 'v2_vae_calibration_experiment.json'
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    csv_path = write_csv(results)
    report_path = write_report(results)
    print(output_path)
    print(csv_path)
    print(report_path)
if __name__ == '__main__':
    main()
