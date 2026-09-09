from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from fxvol.experiment import CellMeanBaseline, DealerStickyBaseline, PCABaseline, PreviousSurfaceBaseline, TenorInterpolationBaseline, chronological_split, evaluate_baseline, middle_tenor_mask_function, random_mask_function, wing_mask_function
ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / 'data' / 'interim' / 'v2'
RESULTS_DIR = ROOT / 'results'
START_DATE = '2010-01-01'
RANDOM_SEED = 20260618
TARGET_PAIRS = {'USDZAR', 'USDTRY', 'USDMXN', 'USDCNH', 'USDPLN', 'USDHUF'}
BASELINES = {'cell_mean': CellMeanBaseline, 'tenor_interpolation': TenorInterpolationBaseline, 'dealer_sticky': DealerStickyBaseline, 'previous_surface': PreviousSurfaceBaseline, 'pca_5_components': lambda: PCABaseline(n_components=5)}

def load_matrices() -> list[dict[str, object]]:
    path = V2_DIR / 'fx_volatility_daily_matrices.jsonl'
    matrices = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    return [matrix for matrix in matrices if str(matrix['date']) >= START_DATE]

def write_report(results: dict[str, object]) -> Path:
    rows = []
    for baseline_key, scenarios in results['experiments'].items():
        for scenario_key, metrics in scenarios.items():
            rows.append('| ' + ' | '.join([baseline_key, scenario_key, f"{metrics['mae']:.4f}", f"{metrics['rmse']:.4f}", f"{metrics['mean_abs_error_to_spread']:.4f}", f"{metrics['within_spread_rate'] * 100:.1f}%"]) + ' |')
    report = f"# V2 Target-Pair Baseline Experiment\n\n## Purpose\n\nProvide a fair target-pair benchmark for the V2 VAE experiment.\n\nEvaluation pairs: `{', '.join(sorted(TARGET_PAIRS))}`.\n\n## Split Counts\n\n| Split | Daily Pair-Surfaces |\n|---|---:|\n| Train | {results['split_counts'].get('train', 0):,} |\n| Validation | {results['split_counts'].get('validation', 0):,} |\n| Test | {results['split_counts'].get('test', 0):,} |\n| Target test | {results['target_test_count']:,} |\n\n## Results\n\n| Baseline | Scenario | MAE | RMSE | Error / Spread | Within Spread |\n|---|---|---:|---:|---:|---:|\n{chr(10).join(rows)}\n\n## Reproducibility\n\n- Machine-readable results: `results/v2_target_baseline_experiment.json`\n- Runner: `src/run_v2_target_baseline_experiment.py`\n- Random seed: `{RANDOM_SEED}`\n"
    path = RESULTS_DIR / 'v2_target_baseline_experiment.md'
    path.write_text(report, encoding='utf-8')
    return path

def main() -> None:
    matrices = load_matrices()
    split_counts = Counter((chronological_split(matrix) for matrix in matrices))
    train = [matrix for matrix in matrices if chronological_split(matrix) == 'train']
    test = [matrix for matrix in matrices if chronological_split(matrix) == 'test' and str(matrix['pair']) in TARGET_PAIRS]
    results: dict[str, object] = {'dataset': 'v2', 'target_pairs': sorted(TARGET_PAIRS), 'target_test_count': len(test), 'split_counts': dict(split_counts), 'experiments': {}}
    for baseline_key, baseline_factory in BASELINES.items():
        baseline = baseline_factory()
        baseline.fit(train)
        results['experiments'][baseline_key] = {'random_20_percent': evaluate_baseline(baseline, test, random_mask_function(0.2, seed=RANDOM_SEED)), 'full_3m_tenor': evaluate_baseline(baseline, test, middle_tenor_mask_function), 'full_10d_wings': evaluate_baseline(baseline, test, wing_mask_function)}
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / 'v2_target_baseline_experiment.json'
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    report_path = write_report(results)
    print(output_path)
    print(report_path)
if __name__ == '__main__':
    main()
