from __future__ import annotations
import random
from collections.abc import Sequence

def apply_random_mask(values: Sequence[Sequence[float | None]], observed_mask: Sequence[Sequence[bool]], fraction: float, seed: int) -> tuple[list[list[float | None]], list[list[bool]]]:
    if not 0 <= fraction <= 1:
        raise ValueError('fraction must be between 0 and 1')
    candidates = [(row_index, column_index) for row_index, row in enumerate(observed_mask) for column_index, observed in enumerate(row) if observed]
    count = round(len(candidates) * fraction)
    selected = set(random.Random(seed).sample(candidates, count))
    masked_values = [list(row) for row in values]
    masked_observations = [list(row) for row in observed_mask]
    for row_index, column_index in selected:
        masked_values[row_index][column_index] = None
        masked_observations[row_index][column_index] = False
    return (masked_values, masked_observations)

def apply_wing_mask(values: Sequence[Sequence[float | None]], observed_mask: Sequence[Sequence[bool]]) -> tuple[list[list[float | None]], list[list[bool]]]:
    masked_values = [list(row) for row in values]
    masked_observations = [list(row) for row in observed_mask]
    for row_index in range(len(masked_values)):
        for column_index in (3, 4):
            masked_values[row_index][column_index] = None
            masked_observations[row_index][column_index] = False
    return (masked_values, masked_observations)

def apply_tenor_mask(values: Sequence[Sequence[float | None]], observed_mask: Sequence[Sequence[bool]], tenor_index: int) -> tuple[list[list[float | None]], list[list[bool]]]:
    if not 0 <= tenor_index < len(values):
        raise ValueError('tenor_index is outside the matrix')
    masked_values = [list(row) for row in values]
    masked_observations = [list(row) for row in observed_mask]
    for column_index in range(len(masked_values[tenor_index])):
        masked_values[tenor_index][column_index] = None
        masked_observations[tenor_index][column_index] = False
    return (masked_values, masked_observations)
