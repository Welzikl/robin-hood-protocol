from __future__ import annotations
import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from fxvol.experiment import DealerStickyBaseline, PCABaseline, chronological_split
from run_v2_lagged_vae_experiment import LAG_STEPS, LaggedConditionalVAE, rows_to_arrays, train_lagged_transfer_vae
from run_v2_torch_vae_experiment import PAIR_ORDER, RANDOM_SEED, SOURCE_PAIRS, TARGET_PAIRS, ConditionalVAE, fit_standardisation, flatten, load_matrices, train_model, vectorise
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'
WING_COLUMNS = (3, 4)
GAP_LENGTHS = (1, 5, 10, 17)
BATCH_SIZE = 512

def set_seeds() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

def hide_wings(matrix: dict[str, object]) -> tuple[list[list[float | None]], list[list[bool]]]:
    values = [list(row) for row in matrix['values']]
    observed_mask = [list(row) for row in matrix['observed_mask']]
    for row_index in range(len(values)):
        for column_index in WING_COLUMNS:
            if observed_mask[row_index][column_index]:
                values[row_index][column_index] = None
                observed_mask[row_index][column_index] = False
    return (values, observed_mask)

def block_is_evaluable(block: Sequence[dict[str, object]]) -> bool:
    for matrix in block:
        for row_index in range(5):
            for column_index in WING_COLUMNS:
                if not matrix['observed_mask'][row_index][column_index]:
                    return False
    return True

def summarise(errors: Sequence[float], spread_scaled_errors: Sequence[float], within_spread_hits: int, within_half_spread_hits: int, evaluated_blocks: int) -> dict[str, object]:
    return {'evaluated_blocks': evaluated_blocks, 'count': len(errors), 'mae': sum((abs(error) for error in errors)) / len(errors), 'rmse': math.sqrt(sum((error * error for error in errors)) / len(errors)), 'spread_observations': len(spread_scaled_errors), 'mean_abs_error_to_spread': sum(spread_scaled_errors) / len(spread_scaled_errors) if spread_scaled_errors else None, 'within_spread_rate': within_spread_hits / len(spread_scaled_errors) if spread_scaled_errors else None, 'within_half_spread_rate': within_half_spread_hits / len(spread_scaled_errors) if spread_scaled_errors else None}

def update_error_state(predicted_values: np.ndarray, matrix: dict[str, object], errors: list[float], spread_scaled_errors: list[float], hits: dict[str, int]) -> None:
    actual_values = np.array([float(value) if value is not None else np.nan for value in flatten(matrix['values'])], dtype=np.float32)
    spreads = np.array([float(spread) if spread is not None else np.nan for spread in flatten(matrix['spreads'])], dtype=np.float32)
    for row_index in range(5):
        for column_index in WING_COLUMNS:
            cell_index = row_index * 5 + column_index
            actual = float(actual_values[cell_index])
            prediction = float(predicted_values[cell_index])
            error = prediction - actual
            errors.append(error)
            spread = spreads[cell_index]
            if not np.isnan(spread) and spread > 0:
                absolute_error = abs(error)
                spread_scaled_errors.append(absolute_error / float(spread))
                hits['within_spread'] += int(absolute_error <= spread)
                hits['within_half_spread'] += int(absolute_error <= spread / 2)

def collect_baseline_errors(baseline, block: Sequence[dict[str, object]]) -> tuple[list[float], list[float], int, int]:
    errors = []
    spread_scaled_errors = []
    hits = {'within_spread': 0, 'within_half_spread': 0}
    for matrix in block:
        pair = str(matrix['pair'])
        masked_values, _ = hide_wings(matrix)
        reconstructed = baseline.reconstruct(pair, masked_values)
        update_error_state(np.array(flatten(reconstructed), dtype=np.float32), matrix, errors, spread_scaled_errors, hits)
        update = getattr(baseline, 'update', None)
        if callable(update):
            update(pair, masked_values)
    return (errors, spread_scaled_errors, hits['within_spread'], hits['within_half_spread'])

def evaluate_baseline_long_gap(fitted_baseline, matrices_by_pair: dict[str, list[dict[str, object]]], gap_length: int) -> dict[str, object]:
    all_errors = []
    all_spread_scaled_errors = []
    all_within_spread_hits = 0
    all_within_half_spread_hits = 0
    evaluated_blocks = 0
    by_pair = {}
    for pair, pair_matrices in matrices_by_pair.items():
        pair_errors = []
        pair_spread_scaled_errors = []
        pair_within_spread_hits = 0
        pair_within_half_spread_hits = 0
        pair_blocks = 0
        for start_index in range(1, len(pair_matrices) - gap_length + 1):
            block = pair_matrices[start_index:start_index + gap_length]
            if not block_is_evaluable(block):
                continue
            baseline = copy.deepcopy(fitted_baseline)
            update = getattr(baseline, 'update', None)
            if callable(update):
                for warmup_matrix in pair_matrices[:start_index]:
                    update(pair, warmup_matrix['values'])
            errors, scaled_errors, within_spread, within_half_spread = collect_baseline_errors(baseline, block)
            pair_errors.extend(errors)
            pair_spread_scaled_errors.extend(scaled_errors)
            pair_within_spread_hits += within_spread
            pair_within_half_spread_hits += within_half_spread
            pair_blocks += 1
        if pair_errors:
            by_pair[pair] = summarise(pair_errors, pair_spread_scaled_errors, pair_within_spread_hits, pair_within_half_spread_hits, pair_blocks)
            all_errors.extend(pair_errors)
            all_spread_scaled_errors.extend(pair_spread_scaled_errors)
            all_within_spread_hits += pair_within_spread_hits
            all_within_half_spread_hits += pair_within_half_spread_hits
            evaluated_blocks += pair_blocks
    return {'gap_length_weekdays': gap_length, **summarise(all_errors, all_spread_scaled_errors, all_within_spread_hits, all_within_half_spread_hits, evaluated_blocks), 'by_pair': by_pair}

def train_transfer_vae(train_arrays, device: torch.device) -> ConditionalVAE:
    model = ConditionalVAE(pair_count=len(PAIR_ORDER)).to(device)
    train_model(model, train_arrays, SOURCE_PAIRS, device, epochs=140, learning_rate=0.001, beta=0.002, batch_size=2048, random_fraction=0.2, wing_fraction=0.45)
    train_model(model, train_arrays, TARGET_PAIRS, device, epochs=100, learning_rate=0.0005, beta=0.002, batch_size=1024, random_fraction=0.2, wing_fraction=0.55)
    return model

def build_lagged_training_rows(matrices: Sequence[dict[str, object]], standardisation) -> list[dict[str, object]]:
    by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for matrix in matrices:
        by_pair[str(matrix['pair'])].append(matrix)
    rows = []
    for pair, pair_matrices in by_pair.items():
        ordered = sorted(pair_matrices, key=lambda item: str(item['date']))
        means = standardisation.pair_means[pair]
        stds = standardisation.pair_stds[pair]
        for index, matrix in enumerate(ordered):
            if index < max(LAG_STEPS):
                continue
            current_flat = flatten(matrix['values'])
            current_values = np.array([float(value) if value is not None else means[cell_index] for cell_index, value in enumerate(current_flat)], dtype=np.float32)
            current_mask = np.array([value is not None for value in current_flat], dtype=np.float32)
            lag_vectors = []
            for lag_step in LAG_STEPS:
                lag_flat = flatten(ordered[index - lag_step]['values'])
                lag_values = np.array([float(value) if value is not None else means[cell_index] for cell_index, value in enumerate(lag_flat)], dtype=np.float32)
                lag_vectors.append((lag_values - means) / stds)
            pair_onehot = np.zeros(len(PAIR_ORDER), dtype=np.float32)
            pair_onehot[PAIR_ORDER.index(pair)] = 1.0
            rows.append({'matrix': matrix, 'values': (current_values - means) / stds, 'mask': current_mask, 'lags': np.concatenate(lag_vectors).astype(np.float32), 'pair_onehot': pair_onehot})
    return rows

def make_unlagged_example(matrix: dict[str, object], standardisation) -> dict[str, object]:
    pair = str(matrix['pair'])
    means = standardisation.pair_means[pair]
    stds = standardisation.pair_stds[pair]
    flat_values = flatten(matrix['values'])
    values = np.array([float(value) if value is not None else means[cell_index] for cell_index, value in enumerate(flat_values)], dtype=np.float32)
    mask = np.array([value is not None for value in flat_values], dtype=np.float32)
    for column_index in WING_COLUMNS:
        mask[column_index::5] = 0.0
    pair_onehot = np.zeros(len(PAIR_ORDER), dtype=np.float32)
    pair_onehot[PAIR_ORDER.index(pair)] = 1.0
    return {'matrix': matrix, 'values': (values - means) / stds, 'mask': mask, 'pair_onehot': pair_onehot}

def make_lagged_example(matrix: dict[str, object], all_by_pair: dict[str, list[dict[str, object]]], index_by_pair_date: dict[tuple[str, str], int], blocked_dates: set[str], standardisation, mask_blocked_wings: bool=True) -> dict[str, object]:
    pair = str(matrix['pair'])
    means = standardisation.pair_means[pair]
    stds = standardisation.pair_stds[pair]
    ordered = all_by_pair[pair]
    matrix_index = index_by_pair_date[pair, str(matrix['date'])]
    current_flat = flatten(matrix['values'])
    current_values = np.array([float(value) if value is not None else means[cell_index] for cell_index, value in enumerate(current_flat)], dtype=np.float32)
    current_mask = np.array([value is not None for value in current_flat], dtype=np.float32)
    for column_index in WING_COLUMNS:
        current_mask[column_index::5] = 0.0
    lag_vectors = []
    lag_masks = []
    for lag_step in LAG_STEPS:
        lag_index = matrix_index - lag_step
        lag_matrix = ordered[lag_index]
        lag_date = str(lag_matrix['date'])
        lag_flat = flatten(lag_matrix['values'])
        lag_values = []
        lag_mask = []
        for cell_index, value in enumerate(lag_flat):
            _, column_index = divmod(cell_index, 5)
            hidden_inside_gap = mask_blocked_wings and lag_date in blocked_dates and (column_index in WING_COLUMNS)
            available = value is not None and (not hidden_inside_gap)
            lag_mask.append(float(available))
            if not available:
                lag_values.append(float(means[cell_index]))
            else:
                lag_values.append(float(value))
        lag_array = np.array(lag_values, dtype=np.float32)
        lag_vectors.append((lag_array - means) / stds)
        lag_masks.append(np.array(lag_mask, dtype=np.float32))
    pair_onehot = np.zeros(len(PAIR_ORDER), dtype=np.float32)
    pair_onehot[PAIR_ORDER.index(pair)] = 1.0
    return {'matrix': matrix, 'values': (current_values - means) / stds, 'mask': current_mask, 'lags': np.concatenate(lag_vectors).astype(np.float32), 'lag_mask': np.concatenate(lag_masks).astype(np.float32), 'pair_onehot': pair_onehot}

def collect_gap_examples(test_by_pair: dict[str, list[dict[str, object]]], all_by_pair: dict[str, list[dict[str, object]]], index_by_pair_date: dict[tuple[str, str], int], standardisation, gap_length: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], int]:
    unlagged_rows = []
    strict_lagged_rows = []
    leaky_lagged_rows = []
    evaluated_blocks = 0
    for pair, pair_matrices in test_by_pair.items():
        for start_index in range(1, len(pair_matrices) - gap_length + 1):
            block = pair_matrices[start_index:start_index + gap_length]
            if not block_is_evaluable(block):
                continue
            blocked_dates = {str(matrix['date']) for matrix in block}
            for matrix in block:
                unlagged_rows.append(make_unlagged_example(matrix, standardisation))
                strict_lagged_rows.append(make_lagged_example(matrix, all_by_pair, index_by_pair_date, blocked_dates, standardisation, mask_blocked_wings=True))
                leaky_lagged_rows.append(make_lagged_example(matrix, all_by_pair, index_by_pair_date, blocked_dates, standardisation, mask_blocked_wings=False))
            evaluated_blocks += 1
    return (unlagged_rows, strict_lagged_rows, leaky_lagged_rows, evaluated_blocks)

def predict_unlagged(model: ConditionalVAE, rows: Sequence[dict[str, object]], standardisation, device: torch.device, evaluated_blocks: int) -> dict[str, object]:
    values = torch.tensor(np.vstack([row['values'] for row in rows]), dtype=torch.float32)
    masks = torch.tensor(np.vstack([row['mask'] for row in rows]), dtype=torch.float32)
    pairs = torch.tensor(np.vstack([row['pair_onehot'] for row in rows]), dtype=torch.float32)
    predictions = np.zeros((len(rows), 25), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_values = values[start:end].to(device)
            batch_masks = masks[start:end].to(device)
            batch_pairs = pairs[start:end].to(device)
            mu, _ = model.encode(batch_values, batch_masks, batch_pairs)
            predictions[start:end] = model.decode(mu, batch_pairs).cpu().numpy()
    return evaluate_prediction_rows(predictions, rows, standardisation, evaluated_blocks)

def predict_lagged(model: LaggedConditionalVAE, rows: Sequence[dict[str, object]], standardisation, device: torch.device, evaluated_blocks: int) -> dict[str, object]:
    values = torch.tensor(np.vstack([row['values'] for row in rows]), dtype=torch.float32)
    masks = torch.tensor(np.vstack([row['mask'] for row in rows]), dtype=torch.float32)
    lags = torch.tensor(np.vstack([row['lags'] for row in rows]), dtype=torch.float32)
    pairs = torch.tensor(np.vstack([row['pair_onehot'] for row in rows]), dtype=torch.float32)
    predictions = np.zeros((len(rows), 25), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_values = values[start:end].to(device)
            batch_masks = masks[start:end].to(device)
            batch_lags = lags[start:end].to(device)
            batch_pairs = pairs[start:end].to(device)
            mu, _ = model.encode(batch_values, batch_masks, batch_lags, batch_pairs)
            predictions[start:end] = model.decode(mu, batch_lags, batch_pairs).cpu().numpy()
    return evaluate_prediction_rows(predictions, rows, standardisation, evaluated_blocks)

def evaluate_prediction_rows(predictions: np.ndarray, rows: Sequence[dict[str, object]], standardisation, evaluated_blocks: int) -> dict[str, object]:
    errors = []
    spread_scaled_errors = []
    hits = {'within_spread': 0, 'within_half_spread': 0}
    by_pair_errors: dict[str, list[float]] = defaultdict(list)
    by_pair_scaled: dict[str, list[float]] = defaultdict(list)
    by_pair_hits: dict[str, dict[str, int]] = defaultdict(lambda: {'within_spread': 0, 'within_half_spread': 0})
    by_pair_counts: dict[str, int] = defaultdict(int)
    for row_index, row in enumerate(rows):
        matrix = row['matrix']
        pair = str(matrix['pair'])
        means = standardisation.pair_means[pair]
        stds = standardisation.pair_stds[pair]
        predicted_values = means + predictions[row_index] * stds
        before_error_count = len(errors)
        update_error_state(predicted_values, matrix, errors, spread_scaled_errors, hits)
        new_errors = errors[before_error_count:]
        by_pair_errors[pair].extend(new_errors)
        by_pair_counts[pair] += 1
        actual_values = np.array([float(value) if value is not None else np.nan for value in flatten(matrix['values'])], dtype=np.float32)
        spreads = np.array([float(spread) if spread is not None else np.nan for spread in flatten(matrix['spreads'])], dtype=np.float32)
        for cell_index in [row_index_2d * 5 + column for row_index_2d in range(5) for column in WING_COLUMNS]:
            spread = spreads[cell_index]
            if not np.isnan(spread) and spread > 0:
                error = float(predicted_values[cell_index]) - float(actual_values[cell_index])
                absolute_error = abs(error)
                by_pair_scaled[pair].append(absolute_error / float(spread))
                by_pair_hits[pair]['within_spread'] += int(absolute_error <= spread)
                by_pair_hits[pair]['within_half_spread'] += int(absolute_error <= spread / 2)
    return {**summarise(errors, spread_scaled_errors, hits['within_spread'], hits['within_half_spread'], evaluated_blocks), 'by_pair': {pair: {**summarise(pair_errors, by_pair_scaled[pair], by_pair_hits[pair]['within_spread'], by_pair_hits[pair]['within_half_spread'], by_pair_counts[pair])} for pair, pair_errors in sorted(by_pair_errors.items())}}

def write_csv(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_lagged_vs_unlagged_long_gap_comparison.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'gap_length_weekdays', 'mae', 'rmse', 'error_to_spread', 'within_spread_rate', 'within_half_spread_rate', 'evaluated_blocks', 'hidden_quotes'])
        for model, gap_results in results['experiments'].items():
            for gap_length in GAP_LENGTHS:
                metrics = gap_results[str(gap_length)]
                writer.writerow([model, gap_length, metrics['mae'], metrics['rmse'], metrics['mean_abs_error_to_spread'], metrics['within_spread_rate'], metrics['within_half_spread_rate'], metrics['evaluated_blocks'], metrics['count']])
    return path

def write_report(results: dict[str, object]) -> Path:
    rows = []
    labels = {'dealer_sticky': 'Dealer sticky', 'pca_5_components': 'PCA', 'unlagged_transfer_vae': 'Unlagged transfer VAE', 'strict_lagged_transfer_vae': 'Strict lagged transfer VAE', 'leaky_lagged_transfer_vae': 'Leaky lagged transfer VAE'}
    for gap_length in GAP_LENGTHS:
        for model_key in ('dealer_sticky', 'pca_5_components', 'unlagged_transfer_vae', 'strict_lagged_transfer_vae', 'leaky_lagged_transfer_vae'):
            metrics = results['experiments'][model_key][str(gap_length)]
            rows.append('| ' + ' | '.join([str(gap_length), labels[model_key], str(metrics['evaluated_blocks']), f"{metrics['mae']:.4f}", f"{metrics['rmse']:.4f}", f"{metrics['mean_abs_error_to_spread']:.4f}", f"{metrics['within_spread_rate'] * 100:.1f}%"]) + ' |')
    strict_gap_17 = results['experiments']['strict_lagged_transfer_vae']['17']['mae']
    leaky_gap_17 = results['experiments']['leaky_lagged_transfer_vae']['17']['mae']
    unlagged_17 = results['experiments']['unlagged_transfer_vae']['17']['mae']
    report = f"# V2 Lagged vs Unlagged Long-Gap Experiment\n\n## Purpose\n\nCompare the unlagged transfer VAE and lagged transfer VAE when complete 10D\nwing quotes are missing for multiple consecutive weekdays.\n\n## Design\n\n- Evaluation pairs: `{', '.join(sorted(TARGET_PAIRS))}`.\n- Evaluation period: unseen 2024-2026 target-pair test split.\n- Hidden cells: complete 10D RR and 10D BF columns across all tenors.\n- Gap lengths: `{', '.join((str(length) for length in GAP_LENGTHS))}` weekdays.\n- VAE predictions use deterministic encoder means.\n- Lagged VAE uses 1-day and 5-day prior surfaces.\n- Strict leak control: when a lagged prior date falls inside the same artificial\n  gap, its 10D wing inputs are replaced with training-period pair means rather\n  than true hidden wing values.\n- A leaky lagged VAE diagnostic is also reported to show how much performance\n  would be overstated if unavailable lagged wing inputs were allowed.\n\n## Results\n\n| Gap Length | Model | Blocks | MAE | RMSE | Error / Spread | Within Spread |\n|---:|---|---:|---:|---:|---:|---:|\n{chr(10).join(rows)}\n\n## Interpretation\n\nAt the 17-weekday gap length, unlagged transfer VAE MAE is `{unlagged_17:.4f}`, strict\nlagged transfer VAE MAE is `{strict_gap_17:.4f}`, and leaky lagged transfer VAE\nMAE is `{leaky_gap_17:.4f}`. The strict result is the defensible long-gap number;\nthe leaky result is included only as an information-leakage diagnostic.\n\nThe gap windows are overlapping, matching the earlier V2 long-gap diagnostic.\nThat means the result is a robustness diagnostic, not a count of independent\nmarket events.\n\n## Reproducibility\n\n- Machine-readable results:\n  `results/v2_lagged_vs_unlagged_long_gap_experiment.json`\n- CSV: `outputs/v2_lagged_vs_unlagged_long_gap_comparison.csv`\n- Runner: `src/run_v2_lagged_vs_unlagged_long_gap_experiment.py`\n- Random seed: `{RANDOM_SEED}`\n"
    path = RESULTS_DIR / 'v2_lagged_vs_unlagged_long_gap_experiment.md'
    path.write_text(report, encoding='utf-8')
    return path

def main() -> None:
    set_seeds()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrices = load_matrices()
    train = [matrix for matrix in matrices if chronological_split(matrix) == 'train']
    test = [matrix for matrix in matrices if chronological_split(matrix) == 'test' and str(matrix['pair']) in TARGET_PAIRS]
    all_by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    test_by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for matrix in matrices:
        all_by_pair[str(matrix['pair'])].append(matrix)
    for pair in all_by_pair:
        all_by_pair[pair].sort(key=lambda item: str(item['date']))
    for matrix in test:
        test_by_pair[str(matrix['pair'])].append(matrix)
    for pair in test_by_pair:
        test_by_pair[pair].sort(key=lambda item: str(item['date']))
    index_by_pair_date = {(pair, str(matrix['date'])): index for pair, pair_matrices in all_by_pair.items() for index, matrix in enumerate(pair_matrices)}
    dealer = DealerStickyBaseline()
    dealer.fit(train)
    pca = PCABaseline(n_components=5)
    pca.fit(train)
    standardisation = fit_standardisation(train)
    train_arrays = vectorise(train, standardisation)
    unlagged_model = train_transfer_vae(train_arrays, device)
    lagged_train_rows = [row for row in build_lagged_training_rows(matrices, standardisation) if chronological_split(row['matrix']) == 'train']
    lagged_train_arrays = rows_to_arrays(lagged_train_rows)
    lagged_model, lagged_history = train_lagged_transfer_vae(lagged_train_arrays, device)
    experiments: dict[str, dict[str, object]] = {'dealer_sticky': {}, 'pca_5_components': {}, 'unlagged_transfer_vae': {}, 'strict_lagged_transfer_vae': {}, 'leaky_lagged_transfer_vae': {}}
    for gap_length in GAP_LENGTHS:
        experiments['dealer_sticky'][str(gap_length)] = evaluate_baseline_long_gap(dealer, test_by_pair, gap_length)
        experiments['pca_5_components'][str(gap_length)] = evaluate_baseline_long_gap(pca, test_by_pair, gap_length)
        unlagged_rows, strict_lagged_rows, leaky_lagged_rows, evaluated_blocks = collect_gap_examples(test_by_pair, all_by_pair, index_by_pair_date, standardisation, gap_length)
        experiments['unlagged_transfer_vae'][str(gap_length)] = predict_unlagged(unlagged_model, unlagged_rows, standardisation, device, evaluated_blocks)
        experiments['strict_lagged_transfer_vae'][str(gap_length)] = predict_lagged(lagged_model, strict_lagged_rows, standardisation, device, evaluated_blocks)
        experiments['leaky_lagged_transfer_vae'][str(gap_length)] = predict_lagged(lagged_model, leaky_lagged_rows, standardisation, device, evaluated_blocks)
    results: dict[str, object] = {'description': 'V2 target-pair lagged versus unlagged VAE long-gap 10D wing reconstruction.', 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'target_pairs': sorted(TARGET_PAIRS), 'gap_lengths_weekdays': list(GAP_LENGTHS), 'lag_steps': list(LAG_STEPS), 'lagged_history': lagged_history, 'leak_control': 'Strict lagged result replaces 10D wing inputs inside the same artificial gap with training-period pair means; leaky lagged diagnostic keeps those true lagged wings.', 'experiments': experiments}
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / 'v2_lagged_vs_unlagged_long_gap_experiment.json'
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    csv_path = write_csv(results)
    report_path = write_report(results)
    print(output_path)
    print(csv_path)
    print(report_path)
if __name__ == '__main__':
    main()
