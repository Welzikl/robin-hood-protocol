from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from run_v2_vae_calibration_experiment import CALIBRATION_QUANTILE, SCENARIOS, collect_prediction_records, fit_calibration, train_transfer_vae
from run_v2_torch_vae_experiment import RANDOM_SEED, fit_standardisation, load_matrices, split_name, vectorise
from run_v2_lagged_vae_stress_regime_experiment import REGIMES, classify_regime, enrich_matrices_with_support_features, load_support_rows, stress_thresholds
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'
INTERVAL_METHODS = ('global_residual_interval', 'cell_residual_interval', 'spread_scaled_interval')
METHOD_LABELS = {'global_residual_interval': 'Global residual', 'cell_residual_interval': 'Cell residual', 'spread_scaled_interval': 'Spread-scaled'}
REGIME_LABELS = {'high_spot_move': 'High spot move', 'wide_vol_spread': 'Wide vol spread', 'lower_liquidity_pair': 'Lower-liquidity pair'}

def empty_interval_bucket() -> dict[str, object]:
    return {'count': 0, 'absolute_errors': [], 'surfaces': set(), 'global_residual_interval': {'hits': 0, 'widths': []}, 'cell_residual_interval': {'hits': 0, 'widths': []}, 'spread_scaled_interval': {'hits': 0, 'widths': []}}

def interval_widths(record: dict[str, object], calibration: dict[str, object]) -> dict[str, float]:
    global_width = float(calibration['global_width'])
    cell_width = float(calibration['cell_widths'].get(str(record['cell_index']), global_width))
    spread = record.get('spread')
    spread_scale = calibration.get('spread_scale')
    if spread_scale is not None and spread is not None and (float(spread) > 0):
        spread_width = float(spread_scale) * float(spread)
    else:
        spread_width = global_width
    return {'global_residual_interval': global_width, 'cell_residual_interval': cell_width, 'spread_scaled_interval': spread_width}

def add_interval_record(bucket: dict[str, object], record: dict[str, object], calibration: dict[str, object]) -> None:
    actual = float(record['actual'])
    prediction = float(record['prediction'])
    bucket['count'] += 1
    bucket['absolute_errors'].append(abs(actual - prediction))
    bucket['surfaces'].add((str(record['pair']), str(record['date'])))
    for method, half_width in interval_widths(record, calibration).items():
        lower = prediction - half_width
        upper = prediction + half_width
        bucket[method]['hits'] += int(lower <= actual <= upper)
        bucket[method]['widths'].append(upper - lower)

def finalize_interval_bucket(bucket: dict[str, object]) -> dict[str, object]:
    count = int(bucket['count'])
    absolute_errors = [float(value) for value in bucket['absolute_errors']]
    result: dict[str, object] = {'count': count, 'included_surfaces': len(bucket['surfaces']), 'point_mae': sum(absolute_errors) / count if count else None, 'point_rmse': math.sqrt(sum((error * error for error in absolute_errors)) / count) if count else None}
    for method in INTERVAL_METHODS:
        method_bucket = bucket[method]
        widths = [float(value) for value in method_bucket['widths']]
        result[method] = {'coverage': int(method_bucket['hits']) / count if count else None, 'average_width': sum(widths) / len(widths) if widths else None}
    return result

def evaluate_records_by_regime(records: Sequence[dict[str, object]], matrices_by_key: dict[tuple[str, str], dict[str, object]], thresholds: dict[str, dict[str, float]], calibration: dict[str, object]) -> dict[str, dict[str, dict[str, object]]]:
    buckets = {regime: {'normal': empty_interval_bucket(), 'stressed': empty_interval_bucket()} for regime in REGIMES}
    for record in records:
        key = (str(record['pair']), str(record['date']))
        matrix = matrices_by_key[key]
        for regime in REGIMES:
            label = classify_regime(matrix, thresholds, regime)
            add_interval_record(buckets[regime][label], record, calibration)
    return {regime: {label: finalize_interval_bucket(bucket) for label, bucket in regime_buckets.items()} for regime, regime_buckets in buckets.items()}

def write_csv(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_heteroscedastic_calibration_by_regime.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['scenario', 'regime', 'bucket', 'method', 'coverage', 'average_width', 'point_mae', 'hidden_quotes', 'included_surfaces'])
        for scenario, scenario_results in results['scenarios'].items():
            for regime, regime_results in scenario_results['regime_metrics'].items():
                for bucket, metrics in regime_results.items():
                    for method in INTERVAL_METHODS:
                        interval = metrics[method]
                        writer.writerow([scenario, regime, bucket, method, interval['coverage'], interval['average_width'], metrics['point_mae'], metrics['count'], metrics['included_surfaces']])
    return path

def write_report(results: dict[str, object]) -> Path:
    rows = []
    for scenario, scenario_results in results['scenarios'].items():
        for regime, regime_results in scenario_results['regime_metrics'].items():
            for bucket in ('normal', 'stressed'):
                metrics = regime_results[bucket]
                for method in INTERVAL_METHODS:
                    interval = metrics[method]
                    coverage = interval['coverage']
                    average_width = interval['average_width']
                    rows.append('| ' + ' | '.join([scenario, REGIME_LABELS.get(regime, regime), bucket, METHOD_LABELS[method], f'{coverage * 100:.1f}%' if coverage is not None else 'n/a', f'{average_width:.4f}' if average_width is not None else 'n/a', f"{metrics['point_mae']:.4f}" if metrics['point_mae'] is not None else 'n/a', f"{metrics['count']:,}"]) + ' |')
    report = f"# V2 Heteroscedastic Calibration by Regime\n\n## Purpose\n\nTest whether calibrated VAE uncertainty bands remain usable in harder market\nconditions, and whether interval width changes across stress/liquidity regimes.\n\n## Design\n\n- VAE training split: training period only.\n- Calibration split: validation period only.\n- Evaluation split: untouched test period.\n- Scenarios: `{', '.join(SCENARIOS)}`.\n- Regimes: `{', '.join(REGIMES)}`.\n- Stress thresholds for spot moves and volatility spreads are fitted on the\n  training period only, pair by pair.\n- Reported interval methods: global residual, cell residual, and spread-scaled\n  validation calibration.\n\n## Results\n\n| Scenario | Regime | Bucket | Method | Coverage | Avg Width | Point MAE | Hidden Quotes |\n|---|---|---|---|---:|---:|---:|---:|\n{chr(10).join(rows)}\n\n## Interpretation\n\nThe target is not lower point MAE here. The target is calibrated uncertainty:\ncoverage close to the nominal 90% level, with wider intervals in harder regimes\nwhere reconstruction is less certain.\n\nThis experiment should be used as evidence for heteroscedastic/reference-band\nbehaviour, not as a claim that the VAE beats dealer-style sticky on point error.\n\n## Reproducibility\n\n- Machine-readable results: `results/v2_heteroscedastic_calibration_experiment.json`\n- CSV: `outputs/v2_heteroscedastic_calibration_by_regime.csv`\n- Chart: `outputs/v2_heteroscedastic_calibration_by_regime.svg`\n- Runner: `src/run_v2_heteroscedastic_calibration_experiment.py`\n- Random seed: `{RANDOM_SEED}`\n"
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / 'v2_heteroscedastic_calibration_experiment.md'
    path.write_text(report, encoding='utf-8')
    return path

def write_chart(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []
    scenario = 'full_10d_wings'
    if scenario not in results['scenarios']:
        scenario = next(iter(results['scenarios']))
    for regime, regime_results in results['scenarios'][scenario]['regime_metrics'].items():
        for bucket in ('normal', 'stressed'):
            metrics = regime_results[bucket]
            interval = metrics['cell_residual_interval']
            rows.append((REGIME_LABELS.get(regime, regime), bucket, interval['coverage'] or 0.0, interval['average_width'] or 0.0))
    width, height = (1180, 650)
    margin_left, margin_top = (90, 90)
    plot_width, plot_height = (1000, 430)
    max_width = max((row[3] for row in rows)) * 1.15 if rows else 1.0
    group_gap = plot_width / max(len(rows), 1)
    bar_width = 30
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#F8FAFC"/>', '<text x="55" y="45" font-family="Arial" font-size="26" font-weight="700" fill="#0F172A">V2 Heteroscedastic Calibration by Regime</text>', '<text x="55" y="72" font-family="Arial" font-size="15" fill="#475569">Full 10D-wing scenario, cell-residual interval: coverage labels with width bars.</text>', f'<line x1="{margin_left}" y1="{margin_top + plot_height * 0.1:.1f}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height * 0.1:.1f}" stroke="#DC2626" stroke-width="2" stroke-dasharray="7,6"/>', f'<text x="{margin_left + plot_width - 5}" y="{margin_top + plot_height * 0.1 - 8:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#DC2626">90% coverage target</text>']
    for index, (regime, bucket, coverage, avg_width) in enumerate(rows):
        center_x = margin_left + group_gap * index + group_gap / 2
        coverage_y = margin_top + plot_height - coverage * plot_height
        bar_height = avg_width / max_width * 160
        color = '#2563EB' if bucket == 'normal' else '#D97706'
        svg.append(f'<circle cx="{center_x:.1f}" cy="{coverage_y:.1f}" r="8" fill="{color}"/>')
        svg.append(f'<rect x="{center_x - bar_width / 2:.1f}" y="{margin_top + plot_height - bar_height:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}" opacity="0.35"/>')
        svg.append(f'<text x="{center_x:.1f}" y="{coverage_y - 13:.1f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700" fill="{color}">{coverage * 100:.1f}%</text>')
        label = f'{regime}\n{bucket}'
        svg.append(f'<text x="{center_x:.1f}" y="{margin_top + plot_height + 35}" text-anchor="middle" font-family="Arial" font-size="12" fill="#334155">{regime}</text>')
        svg.append(f'<text x="{center_x:.1f}" y="{margin_top + plot_height + 53}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700" fill="{color}">{bucket}</text>')
    for tick in range(0, 101, 25):
        y = margin_top + plot_height - tick / 100 * plot_height
        svg.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#CBD5E1" stroke-width="1" opacity="0.5"/>')
        svg.append(f'<text x="{margin_left - 15}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#475569">{tick}%</text>')
    svg.append(f'<text x="30" y="{margin_top + plot_height / 2}" transform="rotate(-90 30 {margin_top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="14" fill="#334155">Coverage</text>')
    svg.append('<text x="55" y="615" font-family="Arial" font-size="13" fill="#475569">Bar height indicates average interval width; dots indicate empirical coverage.</text>')
    svg.append('</svg>')
    path = OUTPUT_DIR / 'v2_heteroscedastic_calibration_by_regime.svg'
    path.write_text('\n'.join(svg), encoding='utf-8')
    return path

def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrices = enrich_matrices_with_support_features(load_matrices(), load_support_rows())
    train = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'train']
    validation = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'validation']
    test = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'test']
    thresholds = stress_thresholds(train)
    matrices_by_key = {(str(matrix['pair']), str(matrix['date'])): matrix for matrix in test}
    standardisation = fit_standardisation(train)
    train_arrays = vectorise(train, standardisation)
    validation_arrays = vectorise(validation, standardisation)
    test_arrays = vectorise(test, standardisation)
    model, history = train_transfer_vae(train_arrays, device)
    results: dict[str, object] = {'description': 'V2 heteroscedastic calibrated VAE interval evaluation by stress and liquidity regime.', 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'seed': RANDOM_SEED, 'calibration_quantile': CALIBRATION_QUANTILE, 'regimes': list(REGIMES), 'training_history': history, 'scenarios': {}}
    for scenario in SCENARIOS:
        validation_records = collect_prediction_records(model, validation, validation_arrays, standardisation, device, scenario, samples=16)
        calibration = fit_calibration(validation_records)
        test_records = collect_prediction_records(model, test, test_arrays, standardisation, device, scenario, samples=16)
        results['scenarios'][scenario] = {'validation_count': len(validation_records), 'test_count': len(test_records), 'calibration': calibration, 'regime_metrics': evaluate_records_by_regime(test_records, matrices_by_key, thresholds, calibration)}
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / 'v2_heteroscedastic_calibration_experiment.json'
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    csv_path = write_csv(results)
    report_path = write_report(results)
    chart_path = write_chart(results)
    print(output_path)
    print(csv_path)
    print(report_path)
    print(chart_path)
if __name__ == '__main__':
    main()
