from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from fxvol.experiment import DealerStickyBaseline
from run_v2_torch_vae_experiment import RANDOM_SEED, TARGET_PAIRS, fit_standardisation, flatten, load_matrices, mask_for_scenario, split_name, vectorise
from run_v2_vae_calibration_experiment import SCENARIOS, collect_prediction_records, fit_calibration, train_transfer_vae
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'
BLEND_CANDIDATES = tuple((i / 20 for i in range(21)))
INTERVAL_QUANTILE = 0.9

def blend_prediction(record: dict[str, object], blend_weight: float) -> float:
    sticky = float(record['sticky_prediction'])
    vae = float(record['vae_prediction'])
    return (1.0 - blend_weight) * sticky + blend_weight * vae

def fit_blend_weight(records: Sequence[dict[str, object]], candidates: Sequence[float]=BLEND_CANDIDATES) -> float:

    def mae(weight: float) -> float:
        return sum((abs(float(record['actual']) - blend_prediction(record, weight)) for record in records)) / len(records)
    return min(candidates, key=lambda weight: (mae(weight), weight))

def fit_symmetric_residual_width(records: Sequence[dict[str, object]], blend_weight: float, quantile: float=INTERVAL_QUANTILE) -> float:
    residuals = [abs(float(record['actual']) - blend_prediction(record, blend_weight)) for record in records]
    return float(np.quantile(np.array(residuals, dtype=np.float64), quantile))

def evaluate_hybrid_records(records: Sequence[dict[str, object]], blend_weight: float, interval_width: float) -> dict[str, float]:
    errors = []
    spread_hits = 0
    spread_count = 0
    interval_hits = 0
    for record in records:
        actual = float(record['actual'])
        prediction = blend_prediction(record, blend_weight)
        error = prediction - actual
        abs_error = abs(error)
        errors.append(error)
        interval_hits += int(prediction - interval_width <= actual <= prediction + interval_width)
        spread = record.get('spread')
        if spread is not None and float(spread) > 0:
            spread_count += 1
            spread_hits += int(abs_error <= float(spread))
    return {'blend_weight_vae': float(blend_weight), 'count': len(errors), 'point_mae': sum((abs(error) for error in errors)) / len(errors), 'point_rmse': math.sqrt(sum((error * error for error in errors)) / len(errors)), 'coverage': interval_hits / len(errors), 'average_width': 2.0 * interval_width, 'interval_half_width': interval_width, 'within_spread_rate': spread_hits / spread_count if spread_count else None}

def collect_sticky_prediction_records(baseline: DealerStickyBaseline, matrices: list[dict[str, object]], scenario: str) -> list[dict[str, object]]:
    records = []
    for row_index, matrix in sorted(enumerate(matrices), key=lambda item: (str(item[1]['pair']), str(item[1].get('date', '')))):
        if str(matrix['pair']) not in TARGET_PAIRS:
            continue
        original = matrix['values']
        observed = matrix['observed_mask']
        eval_mask = mask_for_scenario(np.array(observed, dtype=np.float32).reshape(-1), scenario, seed=RANDOM_SEED + row_index).reshape(5, 5)
        masked_values = [[value if eval_mask[r, c] > 0.5 else None for c, value in enumerate(row)] for r, row in enumerate(original)]
        reconstructed = baseline.reconstruct(str(matrix['pair']), masked_values)
        spreads = matrix['spreads']
        for cell_index, was_visible in enumerate(np.array(observed, dtype=bool).reshape(-1)):
            r, c = divmod(cell_index, 5)
            was_hidden = was_visible and eval_mask[r, c] < 0.5
            actual = original[r][c]
            prediction = reconstructed[r][c]
            if was_hidden and actual is not None and (prediction is not None):
                spread = spreads[r][c]
                records.append({'date': str(matrix['date']), 'pair': str(matrix['pair']), 'cell_index': cell_index, 'actual': float(actual), 'sticky_prediction': float(prediction), 'spread': None if spread is None else float(spread)})
        baseline.update(str(matrix['pair']), original)
    return records

def merge_prediction_records(sticky_records: Sequence[dict[str, object]], vae_records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    sticky_by_key = {(record['date'], record['pair'], record['cell_index']): record for record in sticky_records}
    merged = []
    for vae_record in vae_records:
        key = (vae_record['date'], vae_record['pair'], vae_record['cell_index'])
        sticky_record = sticky_by_key.get(key)
        if sticky_record is None:
            continue
        merged.append({**sticky_record, 'vae_prediction': float(vae_record['prediction']), 'raw_low': float(vae_record['raw_low']), 'raw_high': float(vae_record['raw_high'])})
    return merged

def evaluate_single_model_records(records: Sequence[dict[str, object]], prediction_key: str) -> dict[str, float | None]:
    adapted = [{'actual': record['actual'], 'sticky_prediction': record[prediction_key], 'vae_prediction': record[prediction_key], 'spread': record.get('spread')} for record in records]
    metrics = evaluate_hybrid_records(adapted, blend_weight=0.0, interval_width=0.0)
    metrics['coverage'] = None
    metrics['average_width'] = None
    metrics['interval_half_width'] = None
    return metrics

def evaluate_vae_spread_scaled_interval(records: Sequence[dict[str, object]], calibration: dict[str, object]) -> dict[str, float]:
    spread_scale = calibration.get('spread_scale')
    global_width = float(calibration['global_width'])
    hits = 0
    widths = []
    for record in records:
        spread = record.get('spread')
        half_width = float(spread_scale) * float(spread) if spread_scale is not None and spread is not None and (float(spread) > 0) else global_width
        prediction = float(record['vae_prediction'])
        actual = float(record['actual'])
        hits += int(prediction - half_width <= actual <= prediction + half_width)
        widths.append(2.0 * half_width)
    return {'coverage': hits / len(records), 'average_width': sum(widths) / len(widths)}

def write_csv(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_hybrid_sticky_vae_comparison.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['scenario', 'method', 'mae', 'rmse', 'coverage', 'average_width', 'within_spread_rate', 'blend_weight_vae', 'count'])
        for scenario, scenario_results in results['scenarios'].items():
            for method, metrics in scenario_results['test_metrics'].items():
                coverage = metrics.get('coverage')
                average_width = metrics.get('average_width')
                if method.endswith('_point'):
                    coverage = None
                    average_width = None
                writer.writerow([scenario, method, metrics.get('point_mae'), metrics.get('point_rmse'), coverage if coverage is not None else '', average_width if average_width is not None else '', metrics.get('within_spread_rate'), metrics.get('blend_weight_vae'), metrics.get('count')])
    return path

def _fmt(value: object, decimals: int=4, pct: bool=False) -> str:
    if value is None:
        return 'n/a'
    number = float(value)
    if pct:
        number *= 100
        return f'{number:.1f}%'
    return f'{number:.{decimals}f}'

def write_report(results: dict[str, object]) -> Path:
    rows = []
    for scenario, scenario_results in results['scenarios'].items():
        for method, metrics in scenario_results['test_metrics'].items():
            coverage = metrics.get('coverage')
            average_width = metrics.get('average_width')
            if method.endswith('_point'):
                coverage = None
                average_width = None
            rows.append('| ' + ' | '.join([scenario, method, _fmt(metrics.get('point_mae')), _fmt(coverage, pct=True), _fmt(average_width), _fmt(metrics.get('within_spread_rate'), pct=True), _fmt(metrics.get('blend_weight_vae'), decimals=2)]) + ' |')
    report = f'# V2 Hybrid Sticky + VAE Experiment\n\n## Purpose\n\nTest whether a practical reconstruction framework can combine the strongest finance-aware point baseline with VAE-implied surface information and validation-calibrated uncertainty.\n\nThe hybrid point forecast is:\n\n```text\nprediction = (1 - alpha) * sticky + alpha * VAE\n```\n\nThe blend weight `alpha` and hybrid interval width are fitted on the validation split only, then evaluated on the untouched test split.\n\n## Test Results\n\n| Scenario | Method | MAE | Coverage | Avg Width | Within Spread | VAE Weight |\n|---|---|---:|---:|---:|---:|---:|\n{chr(10).join(rows)}\n\n## Interpretation\n\nIf the fitted VAE weight is close to zero, validation selected sticky as the best point forecast and the VAE does not improve point MAE. The practical value is then uncertainty calibration rather than point blending.\n\n## Reproducibility\n\n- Machine-readable results: `results/v2_hybrid_sticky_vae_experiment.json`\n- CSV: `outputs/v2_hybrid_sticky_vae_comparison.csv`\n- Runner: `src/run_v2_hybrid_sticky_vae_experiment.py`\n- Blend candidates: `{BLEND_CANDIDATES}`\n- Interval quantile: `{INTERVAL_QUANTILE}`\n'
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / 'v2_hybrid_sticky_vae_experiment.md'
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
    results: dict[str, object] = {'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'seed': RANDOM_SEED, 'training_history': history, 'blend_candidates': BLEND_CANDIDATES, 'interval_quantile': INTERVAL_QUANTILE, 'scenarios': {}}
    for scenario in SCENARIOS:
        validation_vae = collect_prediction_records(model, validation, validation_arrays, standardisation, device, scenario, samples=16)
        test_vae = collect_prediction_records(model, test, test_arrays, standardisation, device, scenario, samples=16)
        calibration = fit_calibration(validation_vae)
        validation_sticky_model = DealerStickyBaseline()
        validation_sticky_model.fit(train)
        validation_sticky = collect_sticky_prediction_records(validation_sticky_model, validation, scenario)
        test_sticky_model = DealerStickyBaseline()
        test_sticky_model.fit(train + validation)
        test_sticky = collect_sticky_prediction_records(test_sticky_model, test, scenario)
        validation_records = merge_prediction_records(validation_sticky, validation_vae)
        test_records = merge_prediction_records(test_sticky, test_vae)
        blend_weight = fit_blend_weight(validation_records)
        interval_width = fit_symmetric_residual_width(validation_records, blend_weight)
        sticky_metrics = evaluate_single_model_records(test_records, 'sticky_prediction')
        vae_metrics = evaluate_single_model_records(test_records, 'vae_prediction')
        hybrid_metrics = evaluate_hybrid_records(test_records, blend_weight, interval_width)
        vae_interval = evaluate_vae_spread_scaled_interval(test_records, calibration)
        vae_metrics.update(vae_interval)
        results['scenarios'][scenario] = {'validation_count': len(validation_records), 'test_count': len(test_records), 'blend_weight_vae': blend_weight, 'hybrid_interval_half_width': interval_width, 'vae_calibration': calibration, 'test_metrics': {'sticky_point': sticky_metrics, 'vae_point_spread_scaled_interval': vae_metrics, 'validation_blend_hybrid_interval': hybrid_metrics}}
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / 'v2_hybrid_sticky_vae_experiment.json'
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    csv_path = write_csv(results)
    report_path = write_report(results)
    print(output_path)
    print(csv_path)
    print(report_path)
if __name__ == '__main__':
    main()
