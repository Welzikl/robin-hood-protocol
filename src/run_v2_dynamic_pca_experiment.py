from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
from fxvol.experiment import DealerStickyBaseline, PCABaseline, PreviousSurfaceBaseline, chronological_split
from run_v2_lagged_vs_unlagged_long_gap_experiment import GAP_LENGTHS, block_is_evaluable, evaluate_baseline_long_gap
from run_v2_torch_vae_experiment import TARGET_PAIRS, flatten, load_matrices
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'
WING_COLUMNS = (3, 4)
COMPONENT_OPTIONS = (2, 3, 5)
TRANSITION_OPTIONS = ('random_walk', 'ar1')

class DynamicPCAModel:

    def __init__(self, n_components: int, transition: str) -> None:
        if transition not in TRANSITION_OPTIONS:
            raise ValueError(f'Unsupported transition: {transition}')
        self.n_components = n_components
        self.transition = transition
        self.means: dict[str, np.ndarray] = {}
        self.stds: dict[str, np.ndarray] = {}
        self.components: dict[str, np.ndarray] = {}
        self.intercepts: dict[str, np.ndarray] = {}
        self.slopes: dict[str, np.ndarray] = {}

    @staticmethod
    def _complete_vector(matrix: dict[str, object]) -> np.ndarray | None:
        vector = flatten(matrix['values'])
        if any((value is None for value in vector)):
            return None
        return np.asarray(vector, dtype=float)

    def fit(self, matrices: Sequence[dict[str, object]]) -> None:
        by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
        for matrix in matrices:
            by_pair[str(matrix['pair'])].append(matrix)
        for pair, pair_matrices in by_pair.items():
            ordered = sorted(pair_matrices, key=lambda item: str(item.get('date', '')))
            vectors = [self._complete_vector(matrix) for matrix in ordered]
            complete = np.vstack([vector for vector in vectors if vector is not None])
            if len(complete) < 3:
                raise ValueError(f'Insufficient complete training surfaces for {pair}')
            means = complete.mean(axis=0)
            stds = complete.std(axis=0)
            stds[stds == 0] = 1.0
            standardised = (complete - means) / stds
            _, _, right_singular_vectors = np.linalg.svd(standardised, full_matrices=False)
            component_count = min(self.n_components, len(right_singular_vectors))
            components = right_singular_vectors[:component_count]
            scores = standardised @ components.T
            self.means[pair] = means
            self.stds[pair] = stds
            self.components[pair] = components
            if self.transition == 'random_walk':
                self.intercepts[pair] = np.zeros(component_count)
                self.slopes[pair] = np.ones(component_count)
            else:
                previous = scores[:-1]
                current = scores[1:]
                design = np.column_stack([np.ones(len(previous)), previous])
                coefficients = []
                for factor_index in range(component_count):
                    fitted, *_ = np.linalg.lstsq(design[:, [0, factor_index + 1]], current[:, factor_index], rcond=None)
                    coefficients.append(fitted)
                self.intercepts[pair] = np.asarray([coefficient[0] for coefficient in coefficients])
                self.slopes[pair] = np.asarray([coefficient[1] for coefficient in coefficients])

    def forecast(self, pair: str, last_values: Sequence[Sequence[float | None]], steps: int) -> list[list[float]]:
        vector = np.asarray(flatten(last_values), dtype=float)
        if np.isnan(vector).any():
            raise ValueError('The pre-outage surface must be complete')
        score = (vector - self.means[pair]) / self.stds[pair] @ self.components[pair].T
        forecasts = []
        for _ in range(steps):
            score = self.intercepts[pair] + self.slopes[pair] * score
            decoded = self.means[pair] + score @ self.components[pair] * self.stds[pair]
            forecasts.append(decoded.astype(float).tolist())
        return forecasts

def _summarise(errors: Sequence[float], scaled: Sequence[float], blocks: int) -> dict[str, object]:
    absolute = [abs(error) for error in errors]
    return {'evaluated_blocks': blocks, 'count': len(errors), 'mae': sum(absolute) / len(absolute), 'rmse': math.sqrt(sum((error * error for error in errors)) / len(errors)), 'spread_observations': len(scaled), 'mean_abs_error_to_spread': sum(scaled) / len(scaled) if scaled else None, 'within_spread_rate': sum((value <= 1.0 for value in scaled)) / len(scaled) if scaled else None, 'within_half_spread_rate': sum((value <= 0.5 for value in scaled)) / len(scaled) if scaled else None}

def evaluate_dynamic_pca_long_gap(model: DynamicPCAModel, matrices_by_pair: Mapping[str, Sequence[dict[str, object]]], gap_length: int) -> dict[str, object]:
    errors: list[float] = []
    scaled: list[float] = []
    predictions: list[dict[str, object]] = []
    by_pair_errors: dict[str, list[float]] = defaultdict(list)
    by_pair_scaled: dict[str, list[float]] = defaultdict(list)
    by_pair_blocks: dict[str, int] = defaultdict(int)
    evaluated_blocks = 0
    for pair, matrices in matrices_by_pair.items():
        ordered = list(matrices)
        for start_index in range(1, len(ordered) - gap_length + 1):
            block = ordered[start_index:start_index + gap_length]
            if not block_is_evaluable(block):
                continue
            prior = ordered[start_index - 1]
            if DynamicPCAModel._complete_vector(prior) is None:
                continue
            block_forecasts = model.forecast(pair, prior['values'], gap_length)
            for horizon, (matrix, forecast) in enumerate(zip(block, block_forecasts, strict=True), start=1):
                predictions.append({'pair': pair, 'date': str(matrix['date']), 'horizon': horizon, 'values': forecast})
                actual = np.asarray(flatten(matrix['values']), dtype=float)
                spreads = np.asarray([float(value) if value is not None else np.nan for value in flatten(matrix['spreads'])], dtype=float)
                for row in range(5):
                    for column in WING_COLUMNS:
                        index = row * 5 + column
                        error = forecast[index] - actual[index]
                        errors.append(error)
                        by_pair_errors[pair].append(error)
                        if not np.isnan(spreads[index]) and spreads[index] > 0:
                            ratio = abs(error) / spreads[index]
                            scaled.append(ratio)
                            by_pair_scaled[pair].append(ratio)
            evaluated_blocks += 1
            by_pair_blocks[pair] += 1
    return {'gap_length_weekdays': gap_length, **_summarise(errors, scaled, evaluated_blocks), 'by_pair': {pair: _summarise(by_pair_errors[pair], by_pair_scaled[pair], by_pair_blocks[pair]) for pair in sorted(by_pair_errors)}, 'predictions': predictions}

def select_dynamic_pca_spec(validation_scores: Mapping[tuple[int, str], float], test_scores: Mapping[tuple[int, str], float] | None=None) -> dict[str, object]:
    del test_scores
    n_components, transition = min(validation_scores, key=lambda spec: (validation_scores[spec], spec[0], spec[1]))
    return {'n_components': n_components, 'transition': transition}

def group_target_matrices(matrices: Sequence[dict[str, object]], split: str) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for matrix in matrices:
        pair = str(matrix['pair'])
        if pair in TARGET_PAIRS and chronological_split(matrix) == split:
            grouped[pair].append(matrix)
    for pair in grouped:
        grouped[pair].sort(key=lambda item: str(item['date']))
    return dict(grouped)

def evaluate_specification(model: DynamicPCAModel, matrices_by_pair: Mapping[str, Sequence[dict[str, object]]]) -> tuple[float, dict[str, dict[str, object]]]:
    evaluations = {str(gap): evaluate_dynamic_pca_long_gap(model, matrices_by_pair, gap) for gap in GAP_LENGTHS}
    score = float(np.mean([float(result['mae']) for result in evaluations.values()]))
    return (score, evaluations)

def strip_predictions(evaluations: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    return {gap: {key: value for key, value in metrics.items() if key != 'predictions'} for gap, metrics in evaluations.items()}

def comparison_rows(results: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for model_name, by_gap in results['test_experiments'].items():
        for gap, metrics in by_gap.items():
            rows.append({'model': model_name, 'gap_length_weekdays': int(gap), 'mae': metrics['mae'], 'rmse': metrics['rmse'], 'mean_abs_error_to_spread': metrics['mean_abs_error_to_spread'], 'within_spread_rate': metrics['within_spread_rate'], 'within_half_spread_rate': metrics['within_half_spread_rate'], 'evaluated_blocks': metrics['evaluated_blocks'], 'count': metrics['count']})
    return rows

def write_csv(results: Mapping[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_dynamic_pca_comparison.csv'
    rows = comparison_rows(results)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path

def write_report(results: Mapping[str, object]) -> Path:
    selected = results['selected_specification']
    validation = results['validation_candidates']
    experiments = results['test_experiments']
    lines = ['# V2 causal dynamic-PCA outage experiment', '', '## Frozen selection', '', f"Validation selected **{selected['n_components']} factors with {selected['transition']} score dynamics**.", '', 'Candidate ranking used mean validation MAE across the unchanged 1-, 5-, 10- and 17-weekday controlled 10-delta-wing outages. Test results were not used for selection.', '', '| Factors | Dynamics | Validation score |', '|---:|---|---:|']
    for candidate in sorted(validation, key=lambda row: float(row['selection_score'])):
        lines.append(f"| {candidate['n_components']} | {candidate['transition']} | {float(candidate['selection_score']):.6f} |")
    lines.extend(['', '## Frozen test comparison', '', '| Model | Gap | MAE | Error/spread | Within spread |', '|---|---:|---:|---:|---:|'])
    for model_name, by_gap in experiments.items():
        for gap, metrics in by_gap.items():
            lines.append(f"| {model_name} | {gap} | {float(metrics['mae']):.6f} | {float(metrics['mean_abs_error_to_spread']):.3f} | {float(metrics['within_spread_rate']):.1%} |")
    dynamic_17 = float(experiments['dynamic_pca']['17']['mae'])
    persistence_17 = float(experiments['previous_surface']['17']['mae'])
    change = (dynamic_17 / persistence_17 - 1.0) * 100.0
    direction = 'higher' if change >= 0 else 'lower'
    lines.extend(['', '## Primary interpretation', '', f'At 17 weekdays, dynamic PCA MAE was `{dynamic_17:.6f}` versus `{persistence_17:.6f}` for persistence: `{abs(change):.2f}%` {direction}.', '', 'This is a forecast-based operational reconstruction benchmark. PCA loadings, normalisation and score transitions were fitted on training-period target-market surfaces only. During each artificial outage, forecasts were generated recursively from the last complete pre-outage surface; hidden test wings were never fed back. The contemporaneous ATM/25-delta cells remain visible to sticky/static-PCA comparators under the established protocol, but are deliberately not used by this pure dynamic forecast.'])
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / 'v2_dynamic_pca_experiment.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path

def main() -> None:
    matrices = load_matrices()
    all_train = [matrix for matrix in matrices if chronological_split(matrix) == 'train']
    target_train = [matrix for matrix in all_train if str(matrix['pair']) in TARGET_PAIRS]
    validation_by_pair = group_target_matrices(matrices, 'validation')
    test_by_pair = group_target_matrices(matrices, 'test')
    validation_scores: dict[tuple[int, str], float] = {}
    validation_candidates = []
    for n_components in COMPONENT_OPTIONS:
        for transition in TRANSITION_OPTIONS:
            print(f'Validating dynamic PCA: {n_components} factors, {transition}', flush=True)
            candidate = DynamicPCAModel(n_components, transition)
            candidate.fit(target_train)
            score, evaluations = evaluate_specification(candidate, validation_by_pair)
            validation_scores[n_components, transition] = score
            validation_candidates.append({'n_components': n_components, 'transition': transition, 'selection_score': score, 'by_gap': strip_predictions(evaluations)})
    selected = select_dynamic_pca_spec(validation_scores)
    dynamic = DynamicPCAModel(int(selected['n_components']), str(selected['transition']))
    dynamic.fit(target_train)
    previous = PreviousSurfaceBaseline()
    previous.fit(all_train)
    dealer = DealerStickyBaseline()
    dealer.fit(all_train)
    static_pca = PCABaseline(n_components=5)
    static_pca.fit(all_train)
    dynamic_test = {}
    previous_test = {}
    dealer_test = {}
    static_test = {}
    for gap in GAP_LENGTHS:
        print(f'Testing frozen specification at gap {gap}', flush=True)
        dynamic_test[str(gap)] = evaluate_dynamic_pca_long_gap(dynamic, test_by_pair, gap)
        previous_test[str(gap)] = evaluate_baseline_long_gap(previous, test_by_pair, gap)
        dealer_test[str(gap)] = evaluate_baseline_long_gap(dealer, test_by_pair, gap)
        static_test[str(gap)] = evaluate_baseline_long_gap(static_pca, test_by_pair, gap)
    results: dict[str, object] = {'description': 'Validation-selected causal dynamic PCA under strict consecutive 10D-wing gaps.', 'target_pairs': sorted(TARGET_PAIRS), 'training_split': 'through 2021-12-31', 'validation_split': '2022-01-01 through 2023-12-31', 'test_split': 'from 2024-01-01', 'selection_rule': 'Lowest mean validation MAE across 1, 5, 10 and 17 weekday gaps.', 'selected_specification': selected, 'validation_candidates': validation_candidates, 'leak_control': 'PCA loadings, cell scaling and pair-specific score transitions use target-pair training surfaces only. Each gap is forecast recursively from the last complete pre-gap surface. Hidden gap wings are never used as model inputs or recursive updates.', 'test_experiments': {'previous_surface': previous_test, 'dealer_sticky': dealer_test, 'static_pca_5_components': static_test, 'dynamic_pca': strip_predictions(dynamic_test)}}
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / 'v2_dynamic_pca_experiment.json'
    json_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    csv_path = write_csv(results)
    report_path = write_report(results)
    print(json_path, flush=True)
    print(csv_path, flush=True)
    print(report_path, flush=True)
if __name__ == '__main__':
    main()
