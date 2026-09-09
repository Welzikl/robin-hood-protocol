from __future__ import annotations
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import torch
from fxvol.experiment import CellMeanBaseline, DealerStickyBaseline, PCABaseline, PreviousSurfaceBaseline, wing_mask_function
from run_v2_lagged_vae_experiment import LAG_STEPS, RANDOM_SEED, TARGET_PAIRS, build_lagged_rows, flatten, rows_to_arrays, run_lagged_variant, train_lagged_transfer_vae
from run_v2_lagged_vae_stress_regime_experiment import REGIMES, classify_regime, enrich_matrices_with_support_features, load_support_rows, stress_thresholds
from run_v2_torch_vae_experiment import fit_standardisation, load_matrices, split_name
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'
METHODS = ('previous_surface', 'sticky', 'pca_5_components', 'lagged_target_only_vae', 'lagged_transfer_vae')
REGIME_PRIORITY = ('high_spot_move', 'wide_vol_spread', 'lower_liquidity_pair')
BUCKETS = (*REGIME_PRIORITY, 'normal')
INTERVAL_QUANTILE = 0.9

def assign_priority_bucket(regimes: Mapping[str, bool]) -> str:
    for regime in REGIME_PRIORITY:
        if bool(regimes.get(regime)):
            return regime
    return 'normal'

def record_bucket(record: Mapping[str, object]) -> str:
    bucket = record.get('bucket')
    if bucket is not None:
        return str(bucket)
    return assign_priority_bucket(record.get('regimes', {}))

def prediction_error(record: Mapping[str, object], method: str) -> float:
    predictions = record['predictions']
    return float(predictions[method]) - float(record['actual'])

def fit_regime_policy(validation_records: Sequence[dict[str, object]], candidate_methods: Sequence[str]=METHODS) -> dict[str, str]:
    by_bucket: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in validation_records:
        by_bucket[record_bucket(record)].append(record)
    policy: dict[str, str] = {}
    for bucket in BUCKETS:
        records = by_bucket.get(bucket, [])
        if not records:
            policy[bucket] = candidate_methods[0]
            continue
        policy[bucket] = min(candidate_methods, key=lambda method: (sum((abs(prediction_error(record, method)) for record in records)) / len(records), candidate_methods.index(method)))
    return policy

def fit_policy_interval_widths(validation_records: Sequence[dict[str, object]], policy: Mapping[str, str], quantile: float=INTERVAL_QUANTILE) -> dict[str, float]:
    widths = {}
    all_residuals = [abs(prediction_error(record, policy[record_bucket(record)])) for record in validation_records]
    fallback = float(np.quantile(np.array(all_residuals, dtype=np.float64), quantile)) if all_residuals else 0.0
    for bucket in BUCKETS:
        residuals = [abs(prediction_error(record, policy[bucket])) for record in validation_records if record_bucket(record) == bucket]
        widths[bucket] = float(np.quantile(np.array(residuals, dtype=np.float64), quantile)) if residuals else fallback
    return widths

def evaluate_policy_records(records: Sequence[dict[str, object]], policy: Mapping[str, str], interval_widths: Mapping[str, float] | None=None) -> dict[str, object]:
    errors: list[float] = []
    within_spread_hits = 0
    spread_count = 0
    interval_hits = 0
    widths: list[float] = []
    method_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    by_bucket: dict[str, list[float]] = defaultdict(list)
    by_method: dict[str, list[float]] = defaultdict(list)
    for record in records:
        bucket = record_bucket(record)
        method = policy[bucket]
        error = prediction_error(record, method)
        absolute_error = abs(error)
        errors.append(error)
        by_bucket[bucket].append(error)
        by_method[method].append(error)
        method_counts[method] += 1
        bucket_counts[bucket] += 1
        spread = record.get('spread')
        if spread is not None and float(spread) > 0:
            spread_count += 1
            within_spread_hits += int(absolute_error <= float(spread))
        if interval_widths is not None:
            half_width = float(interval_widths[bucket])
            widths.append(2.0 * half_width)
            interval_hits += int(absolute_error <= half_width)
    return {'count': len(errors), 'point_mae': sum((abs(error) for error in errors)) / len(errors) if errors else None, 'point_rmse': math.sqrt(sum((error * error for error in errors)) / len(errors)) if errors else None, 'within_spread_rate': within_spread_hits / spread_count if spread_count else None, 'coverage': interval_hits / len(errors) if interval_widths is not None and errors else None, 'average_width': sum(widths) / len(widths) if widths else None, 'method_counts': dict(method_counts), 'bucket_counts': dict(bucket_counts), 'by_bucket_mae': {bucket: sum((abs(error) for error in bucket_errors)) / len(bucket_errors) for bucket, bucket_errors in by_bucket.items()}, 'by_method_mae': {method: sum((abs(error) for error in method_errors)) / len(method_errors) for method, method_errors in by_method.items()}}

def regime_flags(matrix: dict[str, object], thresholds: dict[str, dict[str, float]]) -> dict[str, bool]:
    return {regime: classify_regime(matrix, thresholds, regime) == 'stressed' for regime in REGIMES}

def collect_baseline_records(baseline: CellMeanBaseline, matrices: Sequence[dict[str, object]], thresholds: dict[str, dict[str, float]], method: str) -> list[dict[str, object]]:
    records = []
    for matrix in sorted(matrices, key=lambda item: (str(item['pair']), str(item['date']))):
        if str(matrix['pair']) not in TARGET_PAIRS:
            continue
        original = matrix['values']
        observed = matrix['observed_mask']
        masked_values, masked_mask = wing_mask_function(original, observed)
        pair = str(matrix['pair'])
        reconstructed = baseline.reconstruct(pair, masked_values)
        regimes = regime_flags(matrix, thresholds)
        bucket = assign_priority_bucket(regimes)
        for row_index, row in enumerate(original):
            for column_index, actual in enumerate(row):
                hidden = observed[row_index][column_index] and (not masked_mask[row_index][column_index])
                predicted = reconstructed[row_index][column_index]
                if not hidden or actual is None or predicted is None:
                    continue
                spread = matrix['spreads'][row_index][column_index]
                records.append({'date': str(matrix['date']), 'pair': pair, 'cell_index': row_index * 5 + column_index, 'actual': float(actual), 'spread': None if spread is None else float(spread), 'regimes': regimes, 'bucket': bucket, 'predictions': {method: float(predicted)}})
        update = getattr(baseline, 'update', None)
        if callable(update):
            update(pair, original)
    return records

def collect_lagged_vae_records(model, rows: Sequence[dict[str, object]], arrays, standardisation, thresholds: dict[str, dict[str, float]], device: torch.device, method: str) -> list[dict[str, object]]:
    values_np, observed_np, lags_np, pair_np = arrays
    selected_indices = [index for index, row in enumerate(rows) if str(row['matrix']['pair']) in TARGET_PAIRS]
    eval_masks = observed_np[selected_indices].copy()
    for column_index in (3, 4):
        eval_masks[:, column_index::5] = 0.0
    hidden_masks = (observed_np[selected_indices] > 0.5) & (eval_masks < 0.5)
    values_tensor = torch.tensor(values_np[selected_indices], dtype=torch.float32)
    mask_tensor = torch.tensor(eval_masks, dtype=torch.float32)
    lag_tensor = torch.tensor(lags_np[selected_indices], dtype=torch.float32)
    pair_tensor = torch.tensor(pair_np[selected_indices], dtype=torch.float32)
    predictions = np.zeros((len(selected_indices), 25), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected_indices), 512):
            end = start + 512
            batch_values = values_tensor[start:end].to(device)
            batch_masks = mask_tensor[start:end].to(device)
            batch_lags = lag_tensor[start:end].to(device)
            batch_pairs = pair_tensor[start:end].to(device)
            mu, _ = model.encode(batch_values, batch_masks, batch_lags, batch_pairs)
            predictions[start:end] = model.decode(mu, batch_lags, batch_pairs).cpu().numpy()
    records = []
    selected_rows = [rows[index] for index in selected_indices]
    for selected_index, row in enumerate(selected_rows):
        matrix = row['matrix']
        pair = str(matrix['pair'])
        means = standardisation.pair_means[pair]
        stds = standardisation.pair_stds[pair]
        predicted_values = means + predictions[selected_index] * stds
        actual_values = np.array([float(value) if value is not None else np.nan for value in flatten(matrix['values'])], dtype=np.float32)
        spreads = np.array([float(spread) if spread is not None else np.nan for spread in flatten(matrix['spreads'])], dtype=np.float32)
        regimes = regime_flags(matrix, thresholds)
        bucket = assign_priority_bucket(regimes)
        for cell_index in np.where(hidden_masks[selected_index])[0]:
            actual = float(actual_values[cell_index])
            predicted = float(predicted_values[cell_index])
            if math.isnan(actual) or math.isnan(predicted):
                continue
            spread = float(spreads[cell_index]) if not math.isnan(float(spreads[cell_index])) else None
            records.append({'date': str(matrix['date']), 'pair': pair, 'cell_index': int(cell_index), 'actual': actual, 'spread': spread, 'regimes': regimes, 'bucket': bucket, 'predictions': {method: predicted}})
    return records

def merge_method_records(method_records: Sequence[Sequence[dict[str, object]]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, int], dict[str, object]] = {}
    methods_by_key: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    expected_methods = set(METHODS)
    for records in method_records:
        for record in records:
            key = (str(record['date']), str(record['pair']), int(record['cell_index']))
            if key not in merged:
                merged[key] = {**record, 'predictions': {}}
            merged[key]['predictions'].update(record['predictions'])
            methods_by_key[key].update(record['predictions'].keys())
    return [record for key, record in merged.items() if methods_by_key[key] == expected_methods]

def collect_records_for_split(matrices: Sequence[dict[str, object]], lagged_rows: Sequence[dict[str, object]], lagged_target_only_model, lagged_transfer_model, standardisation, thresholds: dict[str, dict[str, float]], device: torch.device, train_matrices: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    previous_surface = PreviousSurfaceBaseline()
    previous_surface.fit(train_matrices)
    sticky = DealerStickyBaseline()
    sticky.fit(train_matrices)
    pca = PCABaseline(n_components=5)
    pca.fit(train_matrices)
    lagged_arrays = rows_to_arrays(lagged_rows)
    return merge_method_records([collect_baseline_records(previous_surface, matrices, thresholds, 'previous_surface'), collect_baseline_records(sticky, matrices, thresholds, 'sticky'), collect_baseline_records(pca, matrices, thresholds, 'pca_5_components'), collect_lagged_vae_records(lagged_target_only_model, lagged_rows, lagged_arrays, standardisation, thresholds, device, 'lagged_target_only_vae'), collect_lagged_vae_records(lagged_transfer_model, lagged_rows, lagged_arrays, standardisation, thresholds, device, 'lagged_transfer_vae')])

def single_method_policy(method: str) -> dict[str, str]:
    return {bucket: method for bucket in BUCKETS}

def write_csv(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_regime_switching_framework.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['policy', 'mae', 'rmse', 'coverage', 'average_width', 'within_spread_rate', 'count', 'method_counts'])
        for policy, metrics in results['test_metrics'].items():
            writer.writerow([policy, metrics['point_mae'], metrics['point_rmse'], metrics['coverage'] if metrics['coverage'] is not None else '', metrics['average_width'] if metrics['average_width'] is not None else '', metrics['within_spread_rate'], metrics['count'], json.dumps(metrics['method_counts'], sort_keys=True)])
    return path

def fmt(value: object, decimals: int=4, pct: bool=False) -> str:
    if value is None:
        return 'n/a'
    number = float(value)
    if pct:
        return f'{number * 100:.1f}%'
    return f'{number:.{decimals}f}'

def describe_switcher_result(test_metrics: Mapping[str, Mapping[str, object]], learned_policy: Mapping[str, str]) -> str:
    switcher_mae = float(test_metrics['validation_regime_switcher']['point_mae'])
    global_candidates = {name: float(metrics['point_mae']) for name, metrics in test_metrics.items() if name != 'validation_regime_switcher' and metrics.get('point_mae') is not None}
    best_global_name, best_global_mae = min(global_candidates.items(), key=lambda item: item[1])
    improvement = best_global_mae - switcher_mae
    direction = 'beat' if improvement > 0 else 'did not beat'
    return f'The validation-selected regime switcher {direction} the strongest global candidate, `{best_global_name}`, on the full 10D-wing point task (`{switcher_mae:.4f}` versus `{best_global_mae:.4f}` MAE). The frozen policy was {json.dumps(dict(learned_policy), sort_keys=True)}. This is evidence about conditional routing in this chronological holdout, not a claim that a learned model is universally superior.'

def write_report(results: dict[str, object]) -> Path:
    rows = []
    for policy, metrics in results['test_metrics'].items():
        rows.append('| ' + ' | '.join([policy, fmt(metrics['point_mae']), fmt(metrics['coverage'], pct=True), fmt(metrics['average_width']), fmt(metrics['within_spread_rate'], pct=True), json.dumps(metrics['method_counts'], sort_keys=True)]) + ' |')
    learned_policy = results['policies']['validation_regime_switcher']
    bucket_rows = [f'| {bucket} | {method} |' for bucket, method in learned_policy.items()]
    candidate_methods = ', '.join((f'`{method}`' for method in results['candidate_methods']))
    interpretation = describe_switcher_result(results['test_metrics'], learned_policy)
    report = f'# V2 Regime-Switching Reconstruction Framework\n\n## Purpose\n\nTest whether a practical policy can choose the reconstruction method by market regime instead of forcing one global winner.\n\n## Design\n\n- Task: full 10D-wing reconstruction.\n- Candidate point methods: {candidate_methods}.\n- Regime buckets are assigned in priority order: high spot-move, wide volatility-spread, lower-liquidity pair, then normal.\n- Stress thresholds are fitted on the training split only, pair-by-pair.\n- The regime policy and interval widths are selected on the validation split only.\n- Final metrics are evaluated on the untouched test split.\n\n## Validation-selected policy\n\n| Bucket | Selected method |\n|---|---|\n{chr(10).join(bucket_rows)}\n\n## Test results\n\n| Policy | MAE | Coverage | Avg Width | Within Spread | Method Counts |\n|---|---:|---:|---:|---:|---|\n{chr(10).join(rows)}\n\n## Interpretation\n\n{interpretation}\n\n## Reproducibility\n\n- JSON: `results/v2_regime_switching_framework.json`\n- CSV: `outputs/v2_regime_switching_framework.csv`\n- Runner: `src/run_v2_regime_switching_framework.py`\n- Seed: `{RANDOM_SEED}`\n- Lag steps: `{list(LAG_STEPS)}`\n'
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / 'v2_regime_switching_framework.md'
    path.write_text(report, encoding='utf-8')
    return path

def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrices = enrich_matrices_with_support_features(load_matrices(), load_support_rows())
    train = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'train']
    validation = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'validation']
    test = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'test']
    thresholds = stress_thresholds(train)
    standardisation = fit_standardisation(train)
    lagged_rows = build_lagged_rows(matrices, standardisation)
    train_rows = [row for row in lagged_rows if split_name(str(row['matrix']['date'])) == 'train']
    validation_rows = [row for row in lagged_rows if split_name(str(row['matrix']['date'])) == 'validation']
    test_rows = [row for row in lagged_rows if split_name(str(row['matrix']['date'])) == 'test']
    lagged_target_only_model, target_only_history = run_lagged_variant(rows_to_arrays(train_rows), device, None, TARGET_PAIRS)
    lagged_transfer_model, transfer_history = train_lagged_transfer_vae(rows_to_arrays(train_rows), device)
    validation_records = collect_records_for_split(validation, validation_rows, lagged_target_only_model, lagged_transfer_model, standardisation, thresholds, device, train)
    test_records = collect_records_for_split(test, test_rows, lagged_target_only_model, lagged_transfer_model, standardisation, thresholds, device, train)
    learned_policy = fit_regime_policy(validation_records, METHODS)
    policies = {'previous_surface_only': single_method_policy('previous_surface'), 'sticky_only': single_method_policy('sticky'), 'pca_only': single_method_policy('pca_5_components'), 'lagged_target_only_vae_only': single_method_policy('lagged_target_only_vae'), 'lagged_transfer_vae_only': single_method_policy('lagged_transfer_vae'), 'validation_regime_switcher': learned_policy}
    interval_widths = {name: fit_policy_interval_widths(validation_records, policy) for name, policy in policies.items()}
    test_metrics = {name: evaluate_policy_records(test_records, policy, interval_widths[name]) for name, policy in policies.items()}
    validation_metrics = {name: evaluate_policy_records(validation_records, policy, interval_widths[name]) for name, policy in policies.items()}
    results: dict[str, object] = {'dataset': 'v2', 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'seed': RANDOM_SEED, 'target_pairs': sorted(TARGET_PAIRS), 'task': 'full_10d_wings', 'regime_priority': list(REGIME_PRIORITY), 'candidate_methods': list(METHODS), 'interval_quantile': INTERVAL_QUANTILE, 'thresholds': thresholds, 'training_history': {'lagged_target_only_vae': target_only_history, 'lagged_transfer_vae': transfer_history}, 'validation_count': len(validation_records), 'test_count': len(test_records), 'policies': policies, 'interval_widths': interval_widths, 'validation_metrics': validation_metrics, 'test_metrics': test_metrics}
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / 'v2_regime_switching_framework.json'
    json_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    csv_path = write_csv(results)
    report_path = write_report(results)
    print(json_path)
    print(csv_path)
    print(report_path)
if __name__ == '__main__':
    main()
