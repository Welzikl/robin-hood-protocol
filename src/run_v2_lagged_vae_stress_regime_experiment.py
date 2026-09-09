from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
from fxvol.experiment import DealerStickyBaseline, PCABaseline, CellMeanBaseline, chronological_split, wing_mask_function
from run_v2_lagged_vae_experiment import LAG_STEPS, OUTPUT_DIR, PAIR_ORDER, RANDOM_SEED, RESULTS_DIR, TARGET_PAIRS, build_lagged_rows, flatten, rows_to_arrays, train_lagged_transfer_vae
from run_v2_torch_vae_experiment import fit_standardisation, load_matrices, split_name
ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / 'data' / 'interim' / 'v2'
SUPPORT_PATH = V2_DIR / 'market_support_features.jsonl'
MODEL_FACTORIES = {'dealer_sticky': DealerStickyBaseline, 'pca_5_components': lambda: PCABaseline(n_components=5)}
REGIMES = ('high_spot_move', 'wide_vol_spread', 'lower_liquidity_pair')
LIQUIDITY_BUCKETS = {'USDCNH': 'higher_liquidity_target', 'USDMXN': 'higher_liquidity_target', 'USDZAR': 'higher_liquidity_target', 'USDHUF': 'lower_liquidity_target', 'USDPLN': 'lower_liquidity_target', 'USDTRY': 'lower_liquidity_target'}

def load_support_rows() -> dict[tuple[str, str], dict[str, object]]:
    if not SUPPORT_PATH.exists():
        return {}
    rows = [json.loads(line) for line in SUPPORT_PATH.read_text(encoding='utf-8').splitlines() if line.strip()]
    return {(str(row['pair']), str(row['date'])): row for row in rows}

def enrich_matrices_with_support_features(matrices: Sequence[dict[str, object]], support_by_key: dict[tuple[str, str], dict[str, object]]) -> list[dict[str, object]]:
    enriched = []
    for matrix in matrices:
        output = dict(matrix)
        support = support_by_key.get((str(matrix['pair']), str(matrix['date'])))
        if support:
            feature_names = list(support['feature_names'])
            features = list(support['features'])
            by_name = dict(zip(feature_names, features, strict=False))
            spot = dict(output.get('spot') or {})
            if by_name.get('spot_last') is not None:
                spot['last'] = float(by_name['spot_last'])
            if by_name.get('spot_log_return_1d') is not None:
                spot['log_return'] = float(by_name['spot_log_return_1d'])
            if by_name.get('spot_abs_log_return_1d') is not None:
                spot['absolute_log_return'] = float(by_name['spot_abs_log_return_1d'])
            if by_name.get('spot_relative_spread') is not None:
                spot['relative_spread'] = float(by_name['spot_relative_spread'])
            output['spot'] = spot
        enriched.append(output)
    return enriched

def percentile(values: Sequence[float], probability: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * probability))
    return sorted_values[index]

def average_surface_spread(matrix: dict[str, object]) -> float | None:
    values = [float(spread) for row in matrix.get('spreads', []) for spread in row if spread is not None and float(spread) > 0]
    return sum(values) / len(values) if values else None

def stress_thresholds(train: Sequence[dict[str, object]]) -> dict[str, dict[str, float]]:
    spot_moves: dict[str, list[float]] = defaultdict(list)
    surface_spreads: dict[str, list[float]] = defaultdict(list)
    for matrix in train:
        pair = str(matrix['pair'])
        spot = matrix.get('spot') or {}
        absolute_log_return = spot.get('absolute_log_return')
        if absolute_log_return is not None:
            spot_moves[pair].append(float(absolute_log_return))
        average_spread = average_surface_spread(matrix)
        if average_spread is not None:
            surface_spreads[pair].append(average_spread)
    return {pair: {'high_spot_move': percentile(spot_moves[pair], 0.75), 'wide_vol_spread': percentile(surface_spreads[pair], 0.75)} for pair in sorted(set(spot_moves) | set(surface_spreads))}

def classify_regime(matrix: dict[str, object], thresholds: dict[str, dict[str, float]], regime: str) -> str:
    pair = str(matrix['pair'])
    if regime == 'lower_liquidity_pair':
        return 'stressed' if LIQUIDITY_BUCKETS.get(pair) == 'lower_liquidity_target' else 'normal'
    if pair not in thresholds:
        return 'normal'
    if regime == 'high_spot_move':
        spot = matrix.get('spot') or {}
        absolute_log_return = spot.get('absolute_log_return')
        is_stressed = absolute_log_return is not None and float(absolute_log_return) >= thresholds[pair]['high_spot_move']
        return 'stressed' if is_stressed else 'normal'
    if regime == 'wide_vol_spread':
        average_spread = average_surface_spread(matrix)
        is_stressed = average_spread is not None and average_spread >= thresholds[pair]['wide_vol_spread']
        return 'stressed' if is_stressed else 'normal'
    raise ValueError(f'Unknown regime: {regime}')

def summarise(errors: Sequence[float], spread_scaled_errors: Sequence[float], within_spread_hits: int, within_half_spread_hits: int) -> dict[str, float | int | None]:
    spread_count = len(spread_scaled_errors)
    return {'count': len(errors), 'mae': sum((abs(error) for error in errors)) / len(errors) if errors else None, 'rmse': math.sqrt(sum((error * error for error in errors)) / len(errors)) if errors else None, 'spread_observations': spread_count, 'mean_abs_error_to_spread': sum(spread_scaled_errors) / spread_count if spread_count else None, 'within_spread_rate': within_spread_hits / spread_count if spread_count else None, 'within_half_spread_rate': within_half_spread_hits / spread_count if spread_count else None}

def empty_bucket() -> dict[str, object]:
    return {'errors': [], 'spread_scaled_errors': [], 'within_spread_hits': 0, 'within_half_spread_hits': 0, 'surfaces': set()}

def add_error(bucket: dict[str, object], surface_key: str, error: float, spread: float | None) -> None:
    bucket['surfaces'].add(surface_key)
    bucket['errors'].append(error)
    if spread is not None and spread > 0:
        absolute_error = abs(error)
        bucket['spread_scaled_errors'].append(absolute_error / spread)
        bucket['within_spread_hits'] += int(absolute_error <= spread)
        bucket['within_half_spread_hits'] += int(absolute_error <= spread / 2)

def finalize_buckets(buckets: dict[str, dict[str, object]]) -> dict[str, dict[str, float | int | None]]:
    return {label: {'included_surfaces': len(bucket['surfaces']), **summarise(bucket['errors'], bucket['spread_scaled_errors'], int(bucket['within_spread_hits']), int(bucket['within_half_spread_hits']))} for label, bucket in buckets.items()}

def evaluate_baseline_by_regime(baseline: CellMeanBaseline, matrices: Sequence[dict[str, object]], thresholds: dict[str, dict[str, float]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    buckets = {regime: {'normal': empty_bucket(), 'stressed': empty_bucket()} for regime in REGIMES}
    for matrix in sorted(matrices, key=lambda item: (str(item['pair']), str(item['date']))):
        original = matrix['values']
        original_mask = matrix['observed_mask']
        masked_values, masked_mask = wing_mask_function(original, original_mask)
        pair = str(matrix['pair'])
        reconstructed = baseline.reconstruct(pair, masked_values)
        surface_key = f"{pair}:{matrix['date']}"
        for row_index, row in enumerate(original):
            for column_index, actual in enumerate(row):
                was_hidden = original_mask[row_index][column_index] and (not masked_mask[row_index][column_index])
                predicted = reconstructed[row_index][column_index]
                if not was_hidden or actual is None or predicted is None:
                    continue
                error = float(predicted) - float(actual)
                spread = matrix['spreads'][row_index][column_index]
                for regime in REGIMES:
                    label = classify_regime(matrix, thresholds, regime)
                    add_error(buckets[regime][label], surface_key, error, spread)
        update = getattr(baseline, 'update', None)
        if callable(update):
            update(pair, original)
    return {regime: finalize_buckets(regime_buckets) for regime, regime_buckets in buckets.items()}

def evaluate_lagged_vae_by_regime(model, rows: Sequence[dict[str, object]], arrays, standardisation, thresholds: dict[str, dict[str, float]], device: torch.device) -> dict[str, dict[str, dict[str, float | int | None]]]:
    values_np, observed_np, lags_np, pair_np = arrays
    selected_indices = [index for index, row in enumerate(rows) if str(row['matrix']['pair']) in TARGET_PAIRS]
    if not selected_indices:
        raise ValueError('No target-pair rows selected for lagged VAE stress evaluation')
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
    buckets = {regime: {'normal': empty_bucket(), 'stressed': empty_bucket()} for regime in REGIMES}
    selected_rows = [rows[index] for index in selected_indices]
    for selected_index, row in enumerate(selected_rows):
        matrix = row['matrix']
        pair = str(matrix['pair'])
        means = standardisation.pair_means[pair]
        stds = standardisation.pair_stds[pair]
        predicted_values = means + predictions[selected_index] * stds
        actual_values = np.array([float(value) if value is not None else np.nan for value in flatten(matrix['values'])], dtype=np.float32)
        spreads = np.array([float(spread) if spread is not None else np.nan for spread in flatten(matrix['spreads'])], dtype=np.float32)
        surface_key = f"{pair}:{matrix['date']}"
        for cell_index in np.where(hidden_masks[selected_index])[0]:
            actual = float(actual_values[cell_index])
            predicted = float(predicted_values[cell_index])
            if math.isnan(actual) or math.isnan(predicted):
                continue
            spread = float(spreads[cell_index]) if not math.isnan(float(spreads[cell_index])) else None
            for regime in REGIMES:
                label = classify_regime(matrix, thresholds, regime)
                add_error(buckets[regime][label], surface_key, predicted - actual, spread)
    return {regime: finalize_buckets(regime_buckets) for regime, regime_buckets in buckets.items()}

def write_csv(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_lagged_vae_stress_regime_comparison.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['regime', 'model', 'bucket', 'surfaces', 'count', 'mae', 'rmse', 'error_to_spread', 'within_spread_rate'])
        for regime, model_results in results['regimes'].items():
            for model_key, buckets in model_results.items():
                for bucket, metrics in buckets.items():
                    writer.writerow([regime, model_key, bucket, metrics['included_surfaces'], metrics['count'], metrics['mae'], metrics['rmse'], metrics['mean_abs_error_to_spread'], metrics['within_spread_rate']])
    return path

def write_chart(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []
    for regime, model_results in results['regimes'].items():
        for model_key, buckets in model_results.items():
            rows.append((regime, model_key, buckets['normal']['mae'], buckets['stressed']['mae']))
    labels = [f'{regime}\n{model}' for regime, model, _, _ in rows]
    normal = [row[2] or 0.0 for row in rows]
    stressed = [row[3] or 0.0 for row in rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.85), 5.5))
    ax.bar(x - width / 2, normal, width, label='normal')
    ax.bar(x + width / 2, stressed, width, label='stressed / lower-liquidity')
    ax.set_ylabel('MAE on hidden 10D wing quotes')
    ax.set_title('V2 full-10D-wing reconstruction by stress/liquidity regime')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / 'v2_lagged_vae_stress_regime_mae.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

def format_metric(value: object, digits: int=4) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)

def write_report(results: dict[str, object]) -> Path:
    rows = []
    for regime, model_results in results['regimes'].items():
        for model_key, buckets in model_results.items():
            normal = buckets['normal']
            stressed = buckets['stressed']
            rows.append('| ' + ' | '.join([regime, model_key, format_metric(normal['mae']), format_metric(stressed['mae']), format_metric(normal['within_spread_rate'] * 100 if normal['within_spread_rate'] is not None else None, 1) + '%', format_metric(stressed['within_spread_rate'] * 100 if stressed['within_spread_rate'] is not None else None, 1) + '%', str(normal['included_surfaces']), str(stressed['included_surfaces'])]) + ' |')
    report = f"# V2 Lagged VAE Stress-Regime 10D Wing Evaluation\n\n## Purpose\n\nEvaluate whether the lagged conditional VAE is more useful in the regimes that matter for the dissertation: stressed days, wide volatility-spread days, and lower-liquidity target pairs.\n\n## Design\n\n- Dataset: V2 Bloomberg-derived daily FX volatility matrices.\n- Evaluation: unseen test split only.\n- Target pairs: `{', '.join(sorted(TARGET_PAIRS))}`.\n- Task: hide both 10D wing columns and reconstruct them.\n- Stress thresholds: fitted on the training split only, pair-by-pair.\n- Models: dealer-style sticky, PCA with 5 components, and lagged transfer VAE.\n- Lagged VAE conditioning lags: `{', '.join((str(step) for step in LAG_STEPS))}` prior pair-date surfaces.\n\n## Results\n\n| Regime | Model | Normal MAE | Stressed MAE | Normal Within Spread | Stressed Within Spread | Normal Surfaces | Stressed Surfaces |\n|---|---|---:|---:|---:|---:|---:|---:|\n{chr(10).join(rows)}\n\n## Interpretation\n\nThis experiment asks whether the VAE's temporal/generative structure matters most when simple reconstruction is hardest. If the lagged VAE narrows the gap to sticky in the stressed buckets, that supports the dissertation claim that transfer/generative methods can add value for sparse or lower-liquidity surface construction. If sticky remains dominant everywhere, the thesis should frame the VAE as a calibrated reference-surface and uncertainty model rather than a point-forecast replacement for dealer-style rules.\n\n## Reproducibility\n\n- JSON: `results/v2_lagged_vae_stress_regime_experiment.json`\n- CSV: `outputs/v2_lagged_vae_stress_regime_comparison.csv`\n- Chart: `outputs/v2_lagged_vae_stress_regime_mae.png`\n- Runner: `src/run_v2_lagged_vae_stress_regime_experiment.py`\n- Random seed: `{RANDOM_SEED}`\n"
    path = RESULTS_DIR / 'v2_lagged_vae_stress_regime_experiment.md'
    path.write_text(report, encoding='utf-8')
    return path

def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrices = enrich_matrices_with_support_features(load_matrices(), load_support_rows())
    train_matrices = [matrix for matrix in matrices if chronological_split(matrix) == 'train']
    test_matrices = [matrix for matrix in matrices if chronological_split(matrix) == 'test' and str(matrix['pair']) in TARGET_PAIRS]
    thresholds = stress_thresholds(train_matrices)
    regimes: dict[str, dict[str, object]] = {regime: {} for regime in REGIMES}
    for model_key, factory in MODEL_FACTORIES.items():
        baseline = factory()
        baseline.fit(train_matrices)
        model_results = evaluate_baseline_by_regime(baseline, test_matrices, thresholds)
        for regime in REGIMES:
            regimes[regime][model_key] = model_results[regime]
    standardisation = fit_standardisation(train_matrices)
    lagged_rows = build_lagged_rows(matrices, standardisation)
    train_rows = [row for row in lagged_rows if split_name(str(row['matrix']['date'])) == 'train']
    test_rows = [row for row in lagged_rows if split_name(str(row['matrix']['date'])) == 'test']
    model, history = train_lagged_transfer_vae(rows_to_arrays(train_rows), device)
    vae_results = evaluate_lagged_vae_by_regime(model, test_rows, rows_to_arrays(test_rows), standardisation, thresholds, device)
    for regime in REGIMES:
        regimes[regime]['lagged_transfer_vae'] = vae_results[regime]
    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict[str, object] = {'dataset': 'v2', 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'seed': RANDOM_SEED, 'target_pairs': sorted(TARGET_PAIRS), 'regime_definitions': {'high_spot_move': 'test dates whose absolute spot log return is at or above the pair-specific 75th percentile fitted on train', 'wide_vol_spread': 'test dates whose average volatility quote spread is at or above the pair-specific 75th percentile fitted on train', 'lower_liquidity_pair': 'USDHUF, USDPLN, and USDTRY versus USDCNH, USDMXN, and USDZAR'}, 'thresholds': thresholds, 'lag_steps': list(LAG_STEPS), 'training_history': history, 'regimes': regimes}
    json_path = RESULTS_DIR / 'v2_lagged_vae_stress_regime_experiment.json'
    json_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    csv_path = write_csv(results)
    chart_path = write_chart(results)
    report_path = write_report(results)
    print(json_path)
    print(csv_path)
    print(chart_path)
    print(report_path)
if __name__ == '__main__':
    main()
