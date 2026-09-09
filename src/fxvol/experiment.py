from __future__ import annotations
import math
from collections import defaultdict
from datetime import date
from typing import Callable, Sequence
import numpy as np
from .masking import apply_random_mask, apply_tenor_mask, apply_wing_mask
TRAIN_END = date(2021, 12, 31)
VALIDATION_END = date(2023, 12, 31)

def chronological_split(matrix: dict[str, object]) -> str:
    observation_date = date.fromisoformat(str(matrix['date']))
    if observation_date <= TRAIN_END:
        return 'train'
    if observation_date <= VALIDATION_END:
        return 'validation'
    return 'test'

class CellMeanBaseline:

    def __init__(self) -> None:
        self.means: dict[tuple[str, int, int], float] = {}
        self.global_means: dict[tuple[int, int], float] = {}

    def fit(self, matrices: Sequence[dict[str, object]]) -> None:
        values_by_cell: dict[tuple[str, int, int], list[float]] = defaultdict(list)
        global_values: dict[tuple[int, int], list[float]] = defaultdict(list)
        for matrix in matrices:
            pair = str(matrix['pair'])
            for row_index, row in enumerate(matrix['values']):
                for column_index, value in enumerate(row):
                    if value is not None:
                        values_by_cell[pair, row_index, column_index].append(value)
                        global_values[row_index, column_index].append(value)
        self.means = {key: sum(values) / len(values) for key, values in values_by_cell.items()}
        self.global_means = {key: sum(values) / len(values) for key, values in global_values.items()}

    def reconstruct(self, pair: str, values: Sequence[Sequence[float | None]]) -> list[list[float | None]]:
        output = [list(row) for row in values]
        for row_index, row in enumerate(output):
            for column_index, value in enumerate(row):
                if value is None:
                    row[column_index] = self.means.get((pair, row_index, column_index), self.global_means.get((row_index, column_index)))
        return output

class TenorInterpolationBaseline(CellMeanBaseline):

    def reconstruct(self, pair: str, values: Sequence[Sequence[float | None]]) -> list[list[float | None]]:
        output = [list(row) for row in values]
        for column_index in range(len(output[0])):
            observed = [(row_index, row[column_index]) for row_index, row in enumerate(output) if row[column_index] is not None]
            for row_index, row in enumerate(output):
                if row[column_index] is not None:
                    continue
                lower = [item for item in observed if item[0] < row_index]
                upper = [item for item in observed if item[0] > row_index]
                if lower and upper:
                    low_index, low_value = lower[-1]
                    high_index, high_value = upper[0]
                    weight = (row_index - low_index) / (high_index - low_index)
                    row[column_index] = low_value + weight * (high_value - low_value)
                elif lower:
                    row[column_index] = lower[-1][1]
                elif upper:
                    row[column_index] = upper[0][1]
        return super().reconstruct(pair, output)

class PCABaseline(CellMeanBaseline):

    def __init__(self, n_components: int=5) -> None:
        super().__init__()
        self.n_components = n_components
        self.pair_cell_means: dict[str, np.ndarray] = {}
        self.pair_cell_stds: dict[str, np.ndarray] = {}
        self.components: np.ndarray | None = None

    def fit(self, matrices: Sequence[dict[str, object]]) -> None:
        super().fit(matrices)
        vectors_by_pair: dict[str, list[list[float | None]]] = defaultdict(list)
        for matrix in matrices:
            vectors_by_pair[str(matrix['pair'])].append(self._flatten(matrix['values']))
        for pair, vectors in vectors_by_pair.items():
            filled = np.array([[value if value is not None else self._fallback_mean(pair, cell_index) for cell_index, value in enumerate(vector)] for vector in vectors], dtype=float)
            means = filled.mean(axis=0)
            stds = filled.std(axis=0)
            stds[stds == 0] = 1.0
            self.pair_cell_means[pair] = means
            self.pair_cell_stds[pair] = stds
        standardized_rows = []
        for matrix in matrices:
            pair = str(matrix['pair'])
            vector = self._flatten(matrix['values'])
            means = self.pair_cell_means[pair]
            stds = self.pair_cell_stds[pair]
            filled = np.array([value if value is not None else self._fallback_mean(pair, cell_index) for cell_index, value in enumerate(vector)], dtype=float)
            standardized_rows.append((filled - means) / stds)
        _, _, right_singular_vectors = np.linalg.svd(np.vstack(standardized_rows), full_matrices=False)
        self.components = right_singular_vectors[:self.n_components]

    def reconstruct(self, pair: str, values: Sequence[Sequence[float | None]]) -> list[list[float | None]]:
        if self.components is None:
            raise ValueError('PCABaseline must be fitted before reconstruction')
        vector = self._flatten(values)
        observed_indices = [index for index, value in enumerate(vector) if value is not None]
        if len(observed_indices) < self.n_components:
            return super().reconstruct(pair, values)
        means = self.pair_cell_means[pair]
        stds = self.pair_cell_stds[pair]
        observed_values = np.array([vector[index] for index in observed_indices], dtype=float)
        standardized_observed = (observed_values - means[observed_indices]) / stds[observed_indices]
        observed_components = self.components[:, observed_indices].T
        coefficients, *_ = np.linalg.lstsq(observed_components, standardized_observed, rcond=None)
        reconstructed_standardized = coefficients @ self.components
        reconstructed = means + reconstructed_standardized * stds
        output = [list(row) for row in values]
        for cell_index, value in enumerate(vector):
            if value is None:
                row_index, column_index = divmod(cell_index, len(output[0]))
                output[row_index][column_index] = float(reconstructed[cell_index])
        return super().reconstruct(pair, output)

    def _flatten(self, values: Sequence[Sequence[float | None]]) -> list[float | None]:
        return [value for row in values for value in row]

    def _fallback_mean(self, pair: str, cell_index: int) -> float:
        row_index, column_index = divmod(cell_index, 5)
        return self.means.get((pair, row_index, column_index), self.global_means[row_index, column_index])

class DealerStickyBaseline(CellMeanBaseline):

    def __init__(self) -> None:
        super().__init__()
        self.previous_by_pair: dict[str, list[list[float | None]]] = {}

    def fit(self, matrices: Sequence[dict[str, object]]) -> None:
        super().fit(matrices)
        for matrix in sorted(matrices, key=lambda item: (str(item['pair']), str(item.get('date', '')))):
            self.previous_by_pair[str(matrix['pair'])] = [list(row) for row in matrix['values']]

    def reconstruct(self, pair: str, values: Sequence[Sequence[float | None]]) -> list[list[float | None]]:
        previous = self.previous_by_pair.get(pair)
        if previous is None:
            return super().reconstruct(pair, values)
        visible_deltas = []
        column_deltas: dict[int, list[float]] = defaultdict(list)
        for row_index, row in enumerate(values):
            for column_index, current_value in enumerate(row):
                previous_value = previous[row_index][column_index]
                if current_value is not None and previous_value is not None:
                    delta = current_value - previous_value
                    visible_deltas.append(delta)
                    column_deltas[column_index].append(delta)
        global_delta = sum(visible_deltas) / len(visible_deltas) if visible_deltas else 0.0
        output = [list(row) for row in values]
        for row_index, row in enumerate(output):
            for column_index, current_value in enumerate(row):
                if current_value is not None:
                    continue
                previous_value = previous[row_index][column_index]
                if previous_value is None:
                    continue
                adjustment_values = column_deltas.get(column_index)
                adjustment = sum(adjustment_values) / len(adjustment_values) if adjustment_values else global_delta
                row[column_index] = previous_value + adjustment
        return super().reconstruct(pair, output)

    def update(self, pair: str, values: Sequence[Sequence[float | None]]) -> None:
        previous = self.previous_by_pair.get(pair)
        if previous is None:
            self.previous_by_pair[pair] = [list(row) for row in values]
            return
        updated = [list(row) for row in previous]
        for row_index, row in enumerate(values):
            for column_index, value in enumerate(row):
                if value is not None:
                    updated[row_index][column_index] = value
        self.previous_by_pair[pair] = updated

class PreviousSurfaceBaseline(DealerStickyBaseline):

    def reconstruct(self, pair: str, values: Sequence[Sequence[float | None]]) -> list[list[float | None]]:
        previous = self.previous_by_pair.get(pair)
        if previous is None:
            return CellMeanBaseline.reconstruct(self, pair, values)
        output = [list(row) for row in values]
        for row_index, row in enumerate(output):
            for column_index, current_value in enumerate(row):
                if current_value is None and previous[row_index][column_index] is not None:
                    row[column_index] = previous[row_index][column_index]
        return CellMeanBaseline.reconstruct(self, pair, output)

class DenoisingAutoencoderBaseline(CellMeanBaseline):

    def __init__(self, hidden_dim: int=32, latent_dim: int=8, epochs: int=300, learning_rate: float=0.01, corruption_fraction: float=0.2, wing_corruption_fraction: float=0.35, seed: int=20260615) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.corruption_fraction = corruption_fraction
        self.wing_corruption_fraction = wing_corruption_fraction
        self.seed = seed
        self.pair_cell_means: dict[str, np.ndarray] = {}
        self.pair_cell_stds: dict[str, np.ndarray] = {}
        self.weights_encoder_1: np.ndarray | None = None
        self.bias_encoder_1: np.ndarray | None = None
        self.weights_encoder_2: np.ndarray | None = None
        self.bias_encoder_2: np.ndarray | None = None
        self.weights_decoder_1: np.ndarray | None = None
        self.bias_decoder_1: np.ndarray | None = None
        self.weights_decoder_2: np.ndarray | None = None
        self.bias_decoder_2: np.ndarray | None = None

    def fit(self, matrices: Sequence[dict[str, object]]) -> None:
        super().fit(matrices)
        rng = np.random.default_rng(self.seed)
        vectors_by_pair: dict[str, list[list[float | None]]] = defaultdict(list)
        for matrix in matrices:
            vectors_by_pair[str(matrix['pair'])].append(self._flatten(matrix['values']))
        for pair, vectors in vectors_by_pair.items():
            filled = np.array([[value if value is not None else self._fallback_mean(pair, cell_index) for cell_index, value in enumerate(vector)] for vector in vectors], dtype=float)
            means = filled.mean(axis=0)
            stds = filled.std(axis=0)
            stds[stds == 0] = 1.0
            self.pair_cell_means[pair] = means
            self.pair_cell_stds[pair] = stds
        targets = []
        observed_masks = []
        for matrix in matrices:
            pair = str(matrix['pair'])
            vector = self._flatten(matrix['values'])
            observed = np.array([value is not None for value in vector], dtype=float)
            filled = np.array([value if value is not None else self._fallback_mean(pair, cell_index) for cell_index, value in enumerate(vector)], dtype=float)
            targets.append((filled - self.pair_cell_means[pair]) / self.pair_cell_stds[pair])
            observed_masks.append(observed)
        target_matrix = np.vstack(targets)
        observed_matrix = np.vstack(observed_masks)
        sample_count, output_dim = target_matrix.shape
        input_dim = output_dim * 2
        self._initialise_parameters(rng, input_dim, output_dim)
        for _ in range(self.epochs):
            keep_mask = (rng.random(target_matrix.shape) > self.corruption_fraction).astype(float)
            keep_mask *= observed_matrix
            wing_rows = rng.random(sample_count) < self.wing_corruption_fraction
            for column_index in (3, 4):
                keep_mask[wing_rows, column_index::5] = 0.0
            corrupted_values = target_matrix * keep_mask
            inputs = np.hstack([corrupted_values, keep_mask])
            prediction, cache = self._forward(inputs)
            residual = (prediction - target_matrix) * observed_matrix
            grad_prediction = 2.0 * residual / max(float(observed_matrix.sum()), 1.0)
            self._backward(inputs, cache, grad_prediction, sample_count)

    def reconstruct(self, pair: str, values: Sequence[Sequence[float | None]]) -> list[list[float | None]]:
        self._require_fitted()
        vector = self._flatten(values)
        means = self.pair_cell_means.get(pair)
        stds = self.pair_cell_stds.get(pair)
        if means is None or stds is None:
            return super().reconstruct(pair, values)
        observed = np.array([value is not None for value in vector], dtype=float)
        standardized = np.zeros(len(vector), dtype=float)
        for cell_index, value in enumerate(vector):
            if value is not None:
                standardized[cell_index] = (value - means[cell_index]) / stds[cell_index]
        model_input = np.hstack([standardized * observed, observed]).reshape(1, -1)
        prediction, _ = self._forward(model_input)
        reconstructed = means + prediction[0] * stds
        output = [list(row) for row in values]
        for cell_index, value in enumerate(vector):
            if value is None:
                row_index, column_index = divmod(cell_index, len(output[0]))
                output[row_index][column_index] = float(reconstructed[cell_index])
        return super().reconstruct(pair, output)

    def _initialise_parameters(self, rng: np.random.Generator, input_dim: int, output_dim: int) -> None:
        self.weights_encoder_1 = rng.normal(0, 0.08, size=(input_dim, self.hidden_dim))
        self.bias_encoder_1 = np.zeros(self.hidden_dim)
        self.weights_encoder_2 = rng.normal(0, 0.08, size=(self.hidden_dim, self.latent_dim))
        self.bias_encoder_2 = np.zeros(self.latent_dim)
        self.weights_decoder_1 = rng.normal(0, 0.08, size=(self.latent_dim, self.hidden_dim))
        self.bias_decoder_1 = np.zeros(self.hidden_dim)
        self.weights_decoder_2 = rng.normal(0, 0.08, size=(self.hidden_dim, output_dim))
        self.bias_decoder_2 = np.zeros(output_dim)

    def _forward(self, inputs: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        self._require_fitted()
        hidden_encoder = np.tanh(inputs @ self.weights_encoder_1 + self.bias_encoder_1)
        latent = np.tanh(hidden_encoder @ self.weights_encoder_2 + self.bias_encoder_2)
        hidden_decoder = np.tanh(latent @ self.weights_decoder_1 + self.bias_decoder_1)
        output = hidden_decoder @ self.weights_decoder_2 + self.bias_decoder_2
        return (output, (hidden_encoder, latent, hidden_decoder))

    def _backward(self, inputs: np.ndarray, cache: tuple[np.ndarray, np.ndarray, np.ndarray], grad_output: np.ndarray, sample_count: int) -> None:
        hidden_encoder, latent, hidden_decoder = cache
        grad_weights_decoder_2 = hidden_decoder.T @ grad_output
        grad_bias_decoder_2 = grad_output.sum(axis=0)
        grad_hidden_decoder = grad_output @ self.weights_decoder_2.T
        grad_hidden_decoder *= 1.0 - hidden_decoder ** 2
        grad_weights_decoder_1 = latent.T @ grad_hidden_decoder
        grad_bias_decoder_1 = grad_hidden_decoder.sum(axis=0)
        grad_latent = grad_hidden_decoder @ self.weights_decoder_1.T
        grad_latent *= 1.0 - latent ** 2
        grad_weights_encoder_2 = hidden_encoder.T @ grad_latent
        grad_bias_encoder_2 = grad_latent.sum(axis=0)
        grad_hidden_encoder = grad_latent @ self.weights_encoder_2.T
        grad_hidden_encoder *= 1.0 - hidden_encoder ** 2
        grad_weights_encoder_1 = inputs.T @ grad_hidden_encoder
        grad_bias_encoder_1 = grad_hidden_encoder.sum(axis=0)
        scale = self.learning_rate * min(sample_count, 10000) / max(sample_count, 1)
        self.weights_decoder_2 -= scale * grad_weights_decoder_2
        self.bias_decoder_2 -= scale * grad_bias_decoder_2
        self.weights_decoder_1 -= scale * grad_weights_decoder_1
        self.bias_decoder_1 -= scale * grad_bias_decoder_1
        self.weights_encoder_2 -= scale * grad_weights_encoder_2
        self.bias_encoder_2 -= scale * grad_bias_encoder_2
        self.weights_encoder_1 -= scale * grad_weights_encoder_1
        self.bias_encoder_1 -= scale * grad_bias_encoder_1

    def _flatten(self, values: Sequence[Sequence[float | None]]) -> list[float | None]:
        return [value for row in values for value in row]

    def _fallback_mean(self, pair: str, cell_index: int) -> float:
        row_index, column_index = divmod(cell_index, 5)
        return self.means.get((pair, row_index, column_index), self.global_means[row_index, column_index])

    def _require_fitted(self) -> None:
        if self.weights_encoder_1 is None:
            raise ValueError('DenoisingAutoencoderBaseline must be fitted first')

def evaluate_baseline(baseline: CellMeanBaseline, matrices: Sequence[dict[str, object]], mask_function: Callable[[Sequence[Sequence[float | None]], Sequence[Sequence[bool]]], tuple[list[list[float | None]], list[list[bool]]]]) -> dict[str, object]:
    errors: list[float] = []
    spread_scaled_errors: list[float] = []
    within_spread_hits = 0
    within_half_spread_hits = 0
    spread_observations = 0
    errors_by_pair: dict[str, list[float]] = defaultdict(list)
    spread_scaled_by_pair: dict[str, list[float]] = defaultdict(list)
    within_spread_by_pair: dict[str, list[bool]] = defaultdict(list)
    within_half_spread_by_pair: dict[str, list[bool]] = defaultdict(list)
    for matrix in sorted(matrices, key=lambda item: (str(item['pair']), str(item.get('date', '')))):
        original = matrix['values']
        original_mask = matrix['observed_mask']
        spreads = matrix['spreads']
        masked_values, masked_mask = mask_function(original, original_mask)
        pair = str(matrix['pair'])
        reconstructed = baseline.reconstruct(pair, masked_values)
        for row_index, row in enumerate(original):
            for column_index, actual in enumerate(row):
                was_hidden = original_mask[row_index][column_index] and (not masked_mask[row_index][column_index])
                predicted = reconstructed[row_index][column_index]
                if was_hidden and actual is not None and (predicted is not None):
                    error = predicted - actual
                    errors.append(error)
                    errors_by_pair[pair].append(error)
                    spread = spreads[row_index][column_index]
                    if spread is not None and spread > 0:
                        absolute_error = abs(error)
                        scaled_error = absolute_error / spread
                        within_spread = absolute_error <= spread
                        within_half_spread = absolute_error <= spread / 2
                        spread_observations += 1
                        spread_scaled_errors.append(scaled_error)
                        within_spread_hits += int(within_spread)
                        within_half_spread_hits += int(within_half_spread)
                        spread_scaled_by_pair[pair].append(scaled_error)
                        within_spread_by_pair[pair].append(within_spread)
                        within_half_spread_by_pair[pair].append(within_half_spread)
        update = getattr(baseline, 'update', None)
        if callable(update):
            update(pair, original)
    return {'count': len(errors), 'mae': sum((abs(error) for error in errors)) / len(errors), 'rmse': math.sqrt(sum((error * error for error in errors)) / len(errors)), 'spread_observations': spread_observations, 'mean_abs_error_to_spread': sum(spread_scaled_errors) / len(spread_scaled_errors) if spread_scaled_errors else None, 'within_spread_rate': within_spread_hits / spread_observations if spread_observations else None, 'within_half_spread_rate': within_half_spread_hits / spread_observations if spread_observations else None, 'by_pair': {pair: {'count': len(pair_errors), 'mae': sum((abs(error) for error in pair_errors)) / len(pair_errors), 'rmse': math.sqrt(sum((error * error for error in pair_errors)) / len(pair_errors)), 'spread_observations': len(spread_scaled_by_pair[pair]), 'mean_abs_error_to_spread': sum(spread_scaled_by_pair[pair]) / len(spread_scaled_by_pair[pair]) if spread_scaled_by_pair[pair] else None, 'within_spread_rate': sum(within_spread_by_pair[pair]) / len(within_spread_by_pair[pair]) if within_spread_by_pair[pair] else None, 'within_half_spread_rate': sum(within_half_spread_by_pair[pair]) / len(within_half_spread_by_pair[pair]) if within_half_spread_by_pair[pair] else None} for pair, pair_errors in sorted(errors_by_pair.items())}}

def random_mask_function(fraction: float, seed: int) -> Callable[[Sequence[Sequence[float | None]], Sequence[Sequence[bool]]], tuple[list[list[float | None]], list[list[bool]]]]:
    counter = 0

    def mask(values: Sequence[Sequence[float | None]], observed_mask: Sequence[Sequence[bool]]) -> tuple[list[list[float | None]], list[list[bool]]]:
        nonlocal counter
        output = apply_random_mask(values, observed_mask, fraction, seed + counter)
        counter += 1
        return output
    return mask

def wing_mask_function(values: Sequence[Sequence[float | None]], observed_mask: Sequence[Sequence[bool]]) -> tuple[list[list[float | None]], list[list[bool]]]:
    return apply_wing_mask(values, observed_mask)

def middle_tenor_mask_function(values: Sequence[Sequence[float | None]], observed_mask: Sequence[Sequence[bool]]) -> tuple[list[list[float | None]], list[list[bool]]]:
    return apply_tenor_mask(values, observed_mask, tenor_index=2)
