from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from fxvol.experiment import DealerStickyBaseline, PCABaseline, PreviousSurfaceBaseline, chronological_split
from run_v2_lagged_vae_experiment import LAG_STEPS, LaggedConditionalVAE, rows_to_arrays, run_lagged_variant
from run_v2_lagged_vs_unlagged_long_gap_experiment import GAP_LENGTHS, block_is_evaluable, evaluate_baseline_long_gap, evaluate_prediction_rows, make_lagged_example
from run_v2_torch_vae_experiment import PAIR_ORDER, RANDOM_SEED, SOURCE_PAIRS, TARGET_PAIRS, fit_standardisation, flatten, load_matrices
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'
WING_COLUMNS = (3, 4)
MAX_TRAINING_OUTAGE = max(GAP_LENGTHS)
BATCH_SIZE = 512

class OutageAwareLaggedConditionalVAE(nn.Module):

    def __init__(self, pair_count: int, surface_dim: int=25, lag_count: int=2, hidden_dim: int=512, latent_dim: int=24) -> None:
        super().__init__()
        lag_dim = surface_dim * lag_count
        encoder_input_dim = surface_dim * 2 + lag_dim * 2 + pair_count + 1
        decoder_input_dim = latent_dim + lag_dim * 2 + pair_count + 1
        self.encoder = nn.Sequential(nn.Linear(encoder_input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU())
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(decoder_input_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, surface_dim))

    def encode(self, values: torch.Tensor, current_mask: torch.Tensor, lag_values: torch.Tensor, lag_mask: torch.Tensor, pair_onehot: torch.Tensor, outage_age: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(torch.cat([values * current_mask, current_mask, lag_values * lag_mask, lag_mask, pair_onehot, outage_age], dim=1))
        return (self.mu(encoded), self.logvar(encoded))

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor, lag_values: torch.Tensor, lag_mask: torch.Tensor, pair_onehot: torch.Tensor, outage_age: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([latent, lag_values * lag_mask, lag_mask, pair_onehot, outage_age], dim=1))

    def forward(self, values: torch.Tensor, current_mask: torch.Tensor, lag_values: torch.Tensor, lag_mask: torch.Tensor, pair_onehot: torch.Tensor, outage_age: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(values, current_mask, lag_values, lag_mask, pair_onehot, outage_age)
        latent = self.reparameterise(mu, logvar)
        return (self.decode(latent, lag_values, lag_mask, pair_onehot, outage_age), mu, logvar)

def set_seeds() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

def wing_cell_indices(surface_dim: int=25, quote_types: int=5) -> list[int]:
    return [cell_index for cell_index in range(surface_dim) if cell_index % quote_types in WING_COLUMNS]

def apply_consecutive_outage_masks(current_observed: torch.Tensor, lag_observed: torch.Tensor, outage_ages: torch.Tensor, lag_steps: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    current_mask = current_observed.clone()
    lag_mask = lag_observed.clone()
    surface_dim = current_mask.shape[1]
    wing_cells = wing_cell_indices(surface_dim)
    current_mask[:, wing_cells] = 0.0
    for lag_position, lag_step in enumerate(lag_steps):
        unavailable_rows = torch.where(outage_ages >= lag_step)[0]
        offset_cells = torch.tensor([lag_position * surface_dim + index for index in wing_cells], device=lag_mask.device)
        if unavailable_rows.numel():
            lag_mask[unavailable_rows[:, None], offset_cells[None, :]] = 0.0
    return (current_mask, lag_mask)

def build_outage_rows(matrices: Sequence[dict[str, object]], standardisation) -> list[dict[str, object]]:
    by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for matrix in matrices:
        by_pair[str(matrix['pair'])].append(matrix)
    rows: list[dict[str, object]] = []
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
            lag_vectors: list[np.ndarray] = []
            lag_masks: list[np.ndarray] = []
            for lag_step in LAG_STEPS:
                lag_flat = flatten(ordered[index - lag_step]['values'])
                lag_values = np.array([float(value) if value is not None else means[cell_index] for cell_index, value in enumerate(lag_flat)], dtype=np.float32)
                lag_vectors.append((lag_values - means) / stds)
                lag_masks.append(np.array([value is not None for value in lag_flat], dtype=np.float32))
            pair_onehot = np.zeros(len(PAIR_ORDER), dtype=np.float32)
            pair_onehot[PAIR_ORDER.index(pair)] = 1.0
            rows.append({'matrix': matrix, 'values': (current_values - means) / stds, 'mask': current_mask, 'lags': np.concatenate(lag_vectors).astype(np.float32), 'lag_mask': np.concatenate(lag_masks).astype(np.float32), 'pair_onehot': pair_onehot})
    return rows

def rows_to_outage_arrays(rows: Sequence[dict[str, object]]):
    return (np.vstack([row['values'] for row in rows]).astype(np.float32), np.vstack([row['mask'] for row in rows]).astype(np.float32), np.vstack([row['lags'] for row in rows]).astype(np.float32), np.vstack([row['lag_mask'] for row in rows]).astype(np.float32), np.vstack([row['pair_onehot'] for row in rows]).astype(np.float32))

def make_outage_training_inputs(observed: torch.Tensor, lag_observed: torch.Tensor, random_fraction: float, outage_fraction: float, max_outage: int=MAX_TRAINING_OUTAGE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_mask = (torch.rand_like(observed) > random_fraction).float() * observed
    selected = torch.rand(observed.shape[0], device=observed.device) < outage_fraction
    ages = torch.randint(0, max_outage, (observed.shape[0],), device=observed.device)
    outage_current, outage_lags = apply_consecutive_outage_masks(current_mask, lag_observed, ages, LAG_STEPS)
    current_mask = torch.where(selected[:, None], outage_current, current_mask)
    lag_mask = torch.where(selected[:, None], outage_lags, lag_observed)
    age_feature = torch.where(selected, (ages.float() + 1.0) / float(MAX_TRAINING_OUTAGE), torch.zeros_like(ages, dtype=torch.float32))[:, None]
    return (current_mask, lag_mask, age_feature)

def outage_variant_specs() -> dict[str, dict[str, object]]:
    return {'single_date_target_only_vae': {'pretrain_pairs': None, 'max_training_outage': 1}, 'single_date_transfer_vae': {'pretrain_pairs': SOURCE_PAIRS, 'max_training_outage': 1}, 'outage_curriculum_target_only_vae': {'pretrain_pairs': None, 'max_training_outage': MAX_TRAINING_OUTAGE}, 'outage_curriculum_transfer_vae': {'pretrain_pairs': SOURCE_PAIRS, 'max_training_outage': MAX_TRAINING_OUTAGE}}

def train_outage_model(model: OutageAwareLaggedConditionalVAE, train_arrays, allowed_pairs: set[str], device: torch.device, epochs: int, learning_rate: float, beta: float, batch_size: int, random_fraction: float, outage_fraction: float, max_training_outage: int) -> list[dict[str, float]]:
    values_np, observed_np, lags_np, lag_observed_np, pair_np = train_arrays
    allowed_indices = {PAIR_ORDER.index(pair) for pair in allowed_pairs}
    row_mask = np.array([int(np.argmax(row)) in allowed_indices for row in pair_np], dtype=bool)
    values = torch.tensor(values_np[row_mask], dtype=torch.float32)
    observed = torch.tensor(observed_np[row_mask], dtype=torch.float32)
    lags = torch.tensor(lags_np[row_mask], dtype=torch.float32)
    lag_observed = torch.tensor(lag_observed_np[row_mask], dtype=torch.float32)
    pairs = torch.tensor(pair_np[row_mask], dtype=torch.float32)
    loader = DataLoader(TensorDataset(values, observed, lags, lag_observed, pairs), batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=torch.cuda.is_available())
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    wing_weights = torch.ones(25, dtype=torch.float32, device=device)
    wing_weights[wing_cell_indices()] = 4.0
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_reconstruction = 0.0
        total_kl = 0.0
        batches = 0
        for batch_values, batch_observed, batch_lags, batch_lag_observed, batch_pairs in loader:
            batch_values = batch_values.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            batch_lags = batch_lags.to(device, non_blocking=True)
            batch_lag_observed = batch_lag_observed.to(device, non_blocking=True)
            batch_pairs = batch_pairs.to(device, non_blocking=True)
            current_mask, lag_mask, outage_age = make_outage_training_inputs(batch_observed, batch_lag_observed, random_fraction=random_fraction, outage_fraction=outage_fraction, max_outage=max_training_outage)
            prediction, mu, logvar = model(batch_values, current_mask, batch_lags, lag_mask, batch_pairs, outage_age)
            weights = batch_observed * wing_weights
            reconstruction = ((prediction - batch_values) ** 2 * weights).sum()
            reconstruction = reconstruction / weights.sum().clamp_min(1.0)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction + beta * kl
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            total_loss += float(loss.detach().cpu())
            total_reconstruction += float(reconstruction.detach().cpu())
            total_kl += float(kl.detach().cpu())
            batches += 1
        if epoch == 1 or epoch % max(epochs // 10, 1) == 0 or epoch == epochs:
            history.append({'epoch': epoch, 'loss': total_loss / batches, 'reconstruction': total_reconstruction / batches, 'kl': total_kl / batches})
    return history

def run_outage_variant(train_arrays, device: torch.device, pretrain_pairs: set[str] | None, finetune_pairs: set[str], max_training_outage: int) -> tuple[OutageAwareLaggedConditionalVAE, dict[str, list[dict[str, float]]]]:
    set_seeds()
    model = OutageAwareLaggedConditionalVAE(pair_count=len(PAIR_ORDER), lag_count=len(LAG_STEPS)).to(device)
    history: dict[str, list[dict[str, float]]] = {}
    if pretrain_pairs:
        history['pretrain'] = train_outage_model(model, train_arrays, pretrain_pairs, device, epochs=140, learning_rate=0.001, beta=0.002, batch_size=2048, random_fraction=0.2, outage_fraction=0.65, max_training_outage=max_training_outage)
    history['finetune'] = train_outage_model(model, train_arrays, finetune_pairs, device, epochs=100 if pretrain_pairs else 160, learning_rate=0.0005, beta=0.002, batch_size=1024, random_fraction=0.2, outage_fraction=0.7, max_training_outage=max_training_outage)
    return (model, history)

def train_outage_ablation(train_arrays, device: torch.device) -> tuple[dict[str, OutageAwareLaggedConditionalVAE], dict[str, dict[str, list[dict[str, float]]]]]:
    models: dict[str, OutageAwareLaggedConditionalVAE] = {}
    histories: dict[str, dict[str, list[dict[str, float]]]] = {}
    for name, spec in outage_variant_specs().items():
        print(f'Training {name}', flush=True)
        model, history = run_outage_variant(train_arrays=train_arrays, device=device, pretrain_pairs=spec['pretrain_pairs'], finetune_pairs=TARGET_PAIRS, max_training_outage=int(spec['max_training_outage']))
        models[name] = model
        histories[name] = history
    return (models, histories)

def collect_strict_gap_rows(test_by_pair: dict[str, list[dict[str, object]]], all_by_pair: dict[str, list[dict[str, object]]], index_by_pair_date: dict[tuple[str, str], int], standardisation, gap_length: int) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    evaluated_blocks = 0
    for pair_matrices in test_by_pair.values():
        for start_index in range(1, len(pair_matrices) - gap_length + 1):
            block = pair_matrices[start_index:start_index + gap_length]
            if not block_is_evaluable(block):
                continue
            blocked_dates = {str(matrix['date']) for matrix in block}
            for outage_age, matrix in enumerate(block):
                row = make_lagged_example(matrix, all_by_pair, index_by_pair_date, blocked_dates, standardisation, mask_blocked_wings=True)
                row['outage_age'] = np.array([(outage_age + 1.0) / MAX_TRAINING_OUTAGE], dtype=np.float32)
                rows.append(row)
            evaluated_blocks += 1
    return (rows, evaluated_blocks)

def predict_existing(model: LaggedConditionalVAE, rows: Sequence[dict[str, object]], standardisation, device: torch.device, evaluated_blocks: int) -> dict[str, object]:
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

def predict_outage_aware(model: OutageAwareLaggedConditionalVAE, rows: Sequence[dict[str, object]], standardisation, device: torch.device, evaluated_blocks: int) -> dict[str, object]:
    values = torch.tensor(np.vstack([row['values'] for row in rows]), dtype=torch.float32)
    masks = torch.tensor(np.vstack([row['mask'] for row in rows]), dtype=torch.float32)
    lags = torch.tensor(np.vstack([row['lags'] for row in rows]), dtype=torch.float32)
    lag_masks = torch.tensor(np.vstack([row['lag_mask'] for row in rows]), dtype=torch.float32)
    pairs = torch.tensor(np.vstack([row['pair_onehot'] for row in rows]), dtype=torch.float32)
    ages = torch.tensor(np.vstack([row['outage_age'] for row in rows]), dtype=torch.float32)
    predictions = np.zeros((len(rows), 25), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_values = values[start:end].to(device)
            batch_masks = masks[start:end].to(device)
            batch_lags = lags[start:end].to(device)
            batch_lag_masks = lag_masks[start:end].to(device)
            batch_pairs = pairs[start:end].to(device)
            batch_ages = ages[start:end].to(device)
            mu, _ = model.encode(batch_values, batch_masks, batch_lags, batch_lag_masks, batch_pairs, batch_ages)
            predictions[start:end] = model.decode(mu, batch_lags, batch_lag_masks, batch_pairs, batch_ages).cpu().numpy()
    return evaluate_prediction_rows(predictions, rows, standardisation, evaluated_blocks)

def write_csv(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_outage_aware_transfer_vae_comparison.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'gap_length_weekdays', 'mae', 'rmse', 'error_to_spread', 'within_spread_rate', 'within_half_spread_rate', 'evaluated_blocks', 'hidden_quotes'])
        for model, gap_results in results['experiments'].items():
            for gap_length in GAP_LENGTHS:
                metrics = gap_results[str(gap_length)]
                writer.writerow([model, gap_length, metrics['mae'], metrics['rmse'], metrics['mean_abs_error_to_spread'], metrics['within_spread_rate'], metrics['within_half_spread_rate'], metrics['evaluated_blocks'], metrics['count']])
    return path

def relative_description(candidate: float, reference: float) -> str:
    relative = (candidate - reference) / reference
    direction = 'lower' if relative < 0 else 'higher'
    return f'{abs(relative) * 100:.1f}% {direction}'

def write_report(results: dict[str, object]) -> Path:
    labels = {'previous_surface': 'Previous-surface persistence', 'dealer_sticky': 'Dealer-style sticky', 'pca_5_components': 'PCA', 'single_date_target_only_vae': 'Single-date target-only VAE', 'single_date_transfer_vae': 'Single-date transfer VAE', 'outage_curriculum_target_only_vae': 'Outage-curriculum target-only VAE', 'outage_curriculum_transfer_vae': 'Outage-curriculum transfer VAE'}
    rows: list[str] = []
    for gap_length in GAP_LENGTHS:
        for model, label in labels.items():
            metrics = results['experiments'][model][str(gap_length)]
            rows.append(f"| {gap_length} | {label} | {metrics['evaluated_blocks']} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | {metrics['mean_abs_error_to_spread']:.4f} | {metrics['within_spread_rate'] * 100:.1f}% |")
    gap_key = str(max(GAP_LENGTHS))
    single_target = results['experiments']['single_date_target_only_vae'][gap_key]['mae']
    single_transfer = results['experiments']['single_date_transfer_vae'][gap_key]['mae']
    outage_target = results['experiments']['outage_curriculum_target_only_vae'][gap_key]['mae']
    outage_transfer = results['experiments']['outage_curriculum_transfer_vae'][gap_key]['mae']
    persistence = results['experiments']['previous_surface'][gap_key]['mae']
    best_model = min(labels, key=lambda key: results['experiments'][key][gap_key]['mae'])
    report = f"# V2 Outage-Aware Transfer VAE Experiment\n\n## Purpose\n\nTest whether deliberately simulating consecutive G10 and EM quote outages during\ntraining teaches the lagged VAE to reconstruct EM 10D wings more accurately when\nthe current and lagged wing quotes are unavailable.\n\n## Design\n\n- Chronological training only; no validation/test surface is used for fitting.\n- Source pretraining pairs: `{', '.join(sorted(SOURCE_PAIRS))}`.\n- EM target pairs: `{', '.join(sorted(TARGET_PAIRS))}`.\n- Evaluation: untouched target-pair test split.\n- Four VAE arms use the identical architecture, explicit current/lag masks,\n  normalized outage age, wing-weighted loss, lags, seed, and epoch schedule.\n- The curriculum ablation changes only maximum simulated outage length:\n  single-date masking (`1`) versus consecutive masking (`1-{MAX_TRAINING_OUTAGE}`).\n- The transfer ablation changes only whether G10 pretraining occurs before the\n  same EM fine-tuning procedure.\n- All four VAE arms use the identical mask-aware architecture, loss, lags, seed,\n  epoch schedule, and evaluation windows. Only source pretraining and the maximum\n  simulated training-outage length vary.\n- The single-date controls always hide the current wings but leave lagged wings\n  available; the outage-curriculum arms simulate 1-{MAX_TRAINING_OUTAGE}-day gaps.\n- All models are compared on exactly the same overlapping strict test windows.\n- Persistence, dealer-style sticky, and PCA remain mandatory comparators.\n\n## Results\n\n| Gap | Model | Blocks | MAE | RMSE | Error / Spread | Within Spread |\n|---:|---|---:|---:|---:|---:|---:|\n{chr(10).join(rows)}\n\n## 17-Weekday Causal Ablation\n\n| Comparison | Result |\n|---|---:|\n| Curriculum vs single-date, target-only | {relative_description(outage_target, single_target)} MAE |\n| Curriculum vs single-date, transfer | {relative_description(outage_transfer, single_transfer)} MAE |\n| Transfer vs target-only, single-date | {relative_description(single_transfer, single_target)} MAE |\n| Transfer vs target-only, curriculum | {relative_description(outage_transfer, outage_target)} MAE |\n| Curriculum transfer vs persistence | {relative_description(outage_transfer, persistence)} MAE |\n\nThe lowest 17-weekday MAE is produced by **{labels[best_model]}**. This is a\ncontrolled reconstruction result, not proof of better option pricing or a count\nof independent market events, because the gap windows overlap.\n\n## Thesis-Safe Interpretation\n\nThe identical-architecture comparison separates two questions: whether matching\ntraining corruption to a consecutive outage helps, and whether G10 pretraining\nadds value under the same masking curriculum. A positive outage-curriculum claim\nrequires lower MAE than the corresponding single-date arm. A positive transfer\nclaim requires lower MAE than the corresponding target-only arm. Persistence\nremains the practical benchmark. Wide artificial gaps test missingness recovery;\nthey are not evidence that the dates are genuine market-stress events.\n\n## Reproducibility\n\n- JSON: `results/v2_outage_aware_transfer_vae_experiment.json`\n- CSV: `outputs/v2_outage_aware_transfer_vae_comparison.csv`\n- Runner: `src/run_v2_outage_aware_transfer_vae_experiment.py`\n- Random seed: `{RANDOM_SEED}`\n"
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / 'v2_outage_aware_transfer_vae_experiment.md'
    path.write_text(report, encoding='utf-8')
    return path

def main() -> None:
    set_seeds()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrices = load_matrices()
    train = [matrix for matrix in matrices if chronological_split(matrix) == 'train']
    test = [matrix for matrix in matrices if chronological_split(matrix) == 'test' and str(matrix['pair']) in TARGET_PAIRS]
    standardisation = fit_standardisation(train)
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
    outage_rows = [row for row in build_outage_rows(matrices, standardisation) if chronological_split(row['matrix']) == 'train']
    outage_arrays = rows_to_outage_arrays(outage_rows)
    outage_models, outage_histories = train_outage_ablation(outage_arrays, device)
    previous = PreviousSurfaceBaseline()
    previous.fit(train)
    dealer = DealerStickyBaseline()
    dealer.fit(train)
    pca = PCABaseline(n_components=5)
    pca.fit(train)
    experiments: dict[str, dict[str, object]] = {'previous_surface': {}, 'dealer_sticky': {}, 'pca_5_components': {}, **{name: {} for name in outage_variant_specs()}}
    for gap_length in GAP_LENGTHS:
        print(f'Evaluating gap {gap_length}', flush=True)
        gap_key = str(gap_length)
        experiments['previous_surface'][gap_key] = evaluate_baseline_long_gap(previous, test_by_pair, gap_length)
        experiments['dealer_sticky'][gap_key] = evaluate_baseline_long_gap(dealer, test_by_pair, gap_length)
        experiments['pca_5_components'][gap_key] = evaluate_baseline_long_gap(pca, test_by_pair, gap_length)
        rows, evaluated_blocks = collect_strict_gap_rows(test_by_pair, all_by_pair, index_by_pair_date, standardisation, gap_length)
        for variant, model in outage_models.items():
            experiments[variant][gap_key] = predict_outage_aware(model, rows, standardisation, device, evaluated_blocks)
    results: dict[str, object] = {'description': 'Outage-aware G10-to-EM lagged VAE ablation under strict consecutive 10D-wing gaps.', 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'source_pairs': sorted(SOURCE_PAIRS), 'target_pairs': sorted(TARGET_PAIRS), 'gap_lengths_weekdays': list(GAP_LENGTHS), 'lag_steps': list(LAG_STEPS), 'max_training_outage_weekdays': MAX_TRAINING_OUTAGE, 'training_rows': len(outage_rows), 'leak_control': 'Only chronological training rows are fitted. Current wings are hidden, and lagged wings inside each simulated/evaluation outage are mean-filled with explicit zero masks.', 'variant_specs': {name: {'pretraining': 'g10_to_em' if spec['pretrain_pairs'] else 'target_only', 'max_training_outage_weekdays': spec['max_training_outage']} for name, spec in outage_variant_specs().items()}, 'histories': outage_histories, 'experiments': experiments}
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / 'v2_outage_aware_transfer_vae_experiment.json'
    json_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    csv_path = write_csv(results)
    report_path = write_report(results)
    print(json_path, flush=True)
    print(csv_path, flush=True)
    print(report_path, flush=True)
if __name__ == '__main__':
    main()
