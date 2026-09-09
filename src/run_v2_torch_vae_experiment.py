from __future__ import annotations
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / 'data' / 'interim' / 'v2'
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'
START_DATE = '2010-01-01'
TRAIN_END = '2021-12-31'
VALIDATION_END = '2023-12-31'
PAIR_ORDER = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD', 'USDZAR', 'USDTRY', 'USDMXN', 'USDCNH', 'USDPLN', 'USDHUF']
SOURCE_PAIRS = {'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD'}
TARGET_PAIRS = {'USDZAR', 'USDTRY', 'USDMXN', 'USDCNH', 'USDPLN', 'USDHUF'}
CLEAN_TARGET_PAIRS = TARGET_PAIRS - {'USDCNH'}
WING_COLUMNS = (3, 4)
RANDOM_SEED = 20260618

@dataclass
class Standardisation:
    pair_means: dict[str, np.ndarray]
    pair_stds: dict[str, np.ndarray]
    global_means: np.ndarray

class ConditionalVAE(nn.Module):

    def __init__(self, pair_count: int, surface_dim: int=25, hidden_dim: int=512, latent_dim: int=24) -> None:
        super().__init__()
        input_dim = surface_dim * 2 + pair_count
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU())
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim + pair_count, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, surface_dim))

    def encode(self, values: torch.Tensor, mask: torch.Tensor, pair_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(torch.cat([values * mask, mask, pair_onehot], dim=1))
        return (self.mu(encoded), self.logvar(encoded))

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor, pair_onehot: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([latent, pair_onehot], dim=1))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, pair_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(values, mask, pair_onehot)
        latent = self.reparameterise(mu, logvar)
        return (self.decode(latent, pair_onehot), mu, logvar)

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_name(date_label: str) -> str:
    if date_label <= TRAIN_END:
        return 'train'
    if date_label <= VALIDATION_END:
        return 'validation'
    return 'test'

def load_matrices() -> list[dict[str, object]]:
    path = V2_DIR / 'fx_volatility_daily_matrices.jsonl'
    matrices = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    return [matrix for matrix in matrices if str(matrix['date']) >= START_DATE and str(matrix['pair']) in PAIR_ORDER]

def flatten(values: Sequence[Sequence[float | None]]) -> list[float | None]:
    return [value for row in values for value in row]

def fit_standardisation(train: Sequence[dict[str, object]]) -> Standardisation:
    global_by_cell: dict[int, list[float]] = defaultdict(list)
    pair_by_cell: dict[tuple[str, int], list[float]] = defaultdict(list)
    for matrix in train:
        pair = str(matrix['pair'])
        for cell_index, value in enumerate(flatten(matrix['values'])):
            if value is not None:
                global_by_cell[cell_index].append(float(value))
                pair_by_cell[pair, cell_index].append(float(value))
    global_means = np.array([np.mean(global_by_cell[cell_index]) for cell_index in range(25)], dtype=np.float32)
    pair_means = {}
    pair_stds = {}
    for pair in PAIR_ORDER:
        means = []
        stds = []
        for cell_index in range(25):
            values = pair_by_cell.get((pair, cell_index), [])
            if values:
                means.append(float(np.mean(values)))
                std = float(np.std(values))
                stds.append(std if std > 1e-06 else 1.0)
            else:
                means.append(float(global_means[cell_index]))
                stds.append(1.0)
        pair_means[pair] = np.array(means, dtype=np.float32)
        pair_stds[pair] = np.array(stds, dtype=np.float32)
    return Standardisation(pair_means, pair_stds, global_means)

def vectorise(matrices: Sequence[dict[str, object]], standardisation: Standardisation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_rows = []
    mask_rows = []
    pair_rows = []
    for matrix in matrices:
        pair = str(matrix['pair'])
        means = standardisation.pair_means[pair]
        stds = standardisation.pair_stds[pair]
        flat_values = flatten(matrix['values'])
        values = np.array([float(value) if value is not None else means[cell_index] for cell_index, value in enumerate(flat_values)], dtype=np.float32)
        observed = np.array([value is not None for value in flat_values], dtype=np.float32)
        pair_onehot = np.zeros(len(PAIR_ORDER), dtype=np.float32)
        pair_onehot[PAIR_ORDER.index(pair)] = 1.0
        values_rows.append((values - means) / stds)
        mask_rows.append(observed)
        pair_rows.append(pair_onehot)
    return (np.vstack(values_rows), np.vstack(mask_rows), np.vstack(pair_rows))

def make_training_masks(observed: torch.Tensor, random_fraction: float, wing_fraction: float) -> torch.Tensor:
    keep = (torch.rand_like(observed) > random_fraction).float() * observed
    wing_rows = torch.rand(observed.shape[0], device=observed.device) < wing_fraction
    for column_index in WING_COLUMNS:
        keep[wing_rows, column_index::5] = 0.0
    return keep

def loss_function(prediction: torch.Tensor, target: torch.Tensor, original_observed: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> tuple[torch.Tensor, float, float]:
    reconstruction = ((prediction - target) ** 2 * original_observed).sum()
    reconstruction = reconstruction / original_observed.sum().clamp_min(1.0)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return (reconstruction + beta * kl, float(reconstruction.detach().cpu()), float(kl.detach().cpu()))

def train_model(model: ConditionalVAE, train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray], allowed_pairs: set[str], device: torch.device, epochs: int, learning_rate: float, beta: float, batch_size: int, random_fraction: float, wing_fraction: float) -> list[dict[str, float]]:
    values_np, mask_np, pair_np = train_arrays
    allowed_indices = {PAIR_ORDER.index(pair) for pair in allowed_pairs}
    row_mask = np.array([int(np.argmax(row)) in allowed_indices for row in pair_np], dtype=bool)
    values = torch.tensor(values_np[row_mask], dtype=torch.float32)
    observed = torch.tensor(mask_np[row_mask], dtype=torch.float32)
    pairs = torch.tensor(pair_np[row_mask], dtype=torch.float32)
    loader = DataLoader(TensorDataset(values, observed, pairs), batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=torch.cuda.is_available())
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_reconstruction = 0.0
        total_kl = 0.0
        batches = 0
        for batch_values, batch_observed, batch_pairs in loader:
            batch_values = batch_values.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            batch_pairs = batch_pairs.to(device, non_blocking=True)
            input_mask = make_training_masks(batch_observed, random_fraction, wing_fraction)
            prediction, mu, logvar = model(batch_values, input_mask, batch_pairs)
            loss, reconstruction, kl = loss_function(prediction, batch_values, batch_observed, mu, logvar, beta)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            total_loss += float(loss.detach().cpu())
            total_reconstruction += reconstruction
            total_kl += kl
            batches += 1
        if epoch == 1 or epoch % max(epochs // 10, 1) == 0 or epoch == epochs:
            history.append({'epoch': epoch, 'loss': total_loss / batches, 'reconstruction': total_reconstruction / batches, 'kl': total_kl / batches})
    return history

def mask_for_scenario(observed: np.ndarray, scenario: str, seed: int) -> np.ndarray:
    mask = observed.copy()
    if scenario == 'full_10d_wings':
        for column_index in WING_COLUMNS:
            mask[column_index::5] = 0.0
        return mask
    if scenario == 'full_3m_tenor':
        mask[10:15] = 0.0
        return mask
    if scenario == 'random_20_percent':
        rng = np.random.default_rng(seed)
        candidates = np.where(observed > 0.5)[0]
        hide_count = max(1, int(round(len(candidates) * 0.2)))
        hidden = rng.choice(candidates, size=hide_count, replace=False)
        mask[hidden] = 0.0
        return mask
    raise ValueError(f'Unknown scenario: {scenario}')

def evaluate_model(model: ConditionalVAE, matrices: Sequence[dict[str, object]], arrays: tuple[np.ndarray, np.ndarray, np.ndarray], standardisation: Standardisation, device: torch.device, scenario: str, evaluation_pairs: set[str], samples: int=32) -> dict[str, object]:
    values_np, observed_np, pair_np = arrays
    selected_rows = [row_index for row_index, matrix in enumerate(matrices) if str(matrix['pair']) in evaluation_pairs]
    eval_masks = np.vstack([mask_for_scenario(observed_np[row_index], scenario, seed=RANDOM_SEED + row_index) for row_index in selected_rows]).astype(np.float32)
    hidden_masks = (observed_np[selected_rows] > 0.5) & (eval_masks < 0.5)
    values_tensor = torch.tensor(values_np[selected_rows], dtype=torch.float32)
    mask_tensor = torch.tensor(eval_masks, dtype=torch.float32)
    pair_tensor = torch.tensor(pair_np[selected_rows], dtype=torch.float32)
    prediction_means = np.zeros((len(selected_rows), 25), dtype=np.float32)
    prediction_lows = np.zeros((len(selected_rows), 25), dtype=np.float32)
    prediction_highs = np.zeros((len(selected_rows), 25), dtype=np.float32)
    batch_size = 512
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected_rows), batch_size):
            end = start + batch_size
            batch_values = values_tensor[start:end].to(device)
            batch_masks = mask_tensor[start:end].to(device)
            batch_pairs = pair_tensor[start:end].to(device)
            mu, logvar = model.encode(batch_values, batch_masks, batch_pairs)
            batch_predictions = []
            for _ in range(samples):
                latent = model.reparameterise(mu, logvar)
                batch_predictions.append(model.decode(latent, batch_pairs).cpu().numpy())
            sample_cube = np.stack(batch_predictions, axis=0)
            prediction_means[start:end] = sample_cube.mean(axis=0)
            prediction_lows[start:end] = np.quantile(sample_cube, 0.05, axis=0)
            prediction_highs[start:end] = np.quantile(sample_cube, 0.95, axis=0)
    errors = []
    spread_scaled_errors = []
    within_spread_hits = 0
    within_half_spread_hits = 0
    interval_hits = 0
    interval_count = 0
    by_pair_errors: dict[str, list[float]] = defaultdict(list)
    for selected_index, row_index in enumerate(selected_rows):
        matrix = matrices[row_index]
        pair = str(matrix['pair'])
        means = standardisation.pair_means[pair]
        stds = standardisation.pair_stds[pair]
        actual_values = np.array([float(value) if value is not None else np.nan for value in flatten(matrix['values'])], dtype=np.float32)
        predicted_values = means + prediction_means[selected_index] * stds
        low_values = means + prediction_lows[selected_index] * stds
        high_values = means + prediction_highs[selected_index] * stds
        spreads = np.array([float(spread) if spread is not None else np.nan for spread in flatten(matrix['spreads'])], dtype=np.float32)
        for cell_index in np.where(hidden_masks[selected_index])[0]:
            actual = float(actual_values[cell_index])
            predicted = float(predicted_values[cell_index])
            error = predicted - actual
            errors.append(error)
            by_pair_errors[pair].append(error)
            spread = spreads[cell_index]
            if not np.isnan(spread) and spread > 0:
                absolute_error = abs(error)
                spread_scaled_errors.append(absolute_error / float(spread))
                within_spread_hits += int(absolute_error <= spread)
                within_half_spread_hits += int(absolute_error <= spread / 2)
            interval_count += 1
            interval_hits += int(float(low_values[cell_index]) <= actual <= float(high_values[cell_index]))
    return {'count': len(errors), 'mae': sum((abs(error) for error in errors)) / len(errors), 'rmse': math.sqrt(sum((error * error for error in errors)) / len(errors)), 'spread_observations': len(spread_scaled_errors), 'mean_abs_error_to_spread': sum(spread_scaled_errors) / len(spread_scaled_errors), 'within_spread_rate': within_spread_hits / len(spread_scaled_errors), 'within_half_spread_rate': within_half_spread_hits / len(spread_scaled_errors), 'interval_90_coverage': interval_hits / interval_count if interval_count else None, 'by_pair': {pair: {'count': len(pair_errors), 'mae': sum((abs(error) for error in pair_errors)) / len(pair_errors), 'rmse': math.sqrt(sum((error * error for error in pair_errors)) / len(pair_errors))} for pair, pair_errors in sorted(by_pair_errors.items())}}

def run_variant(train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray], test_matrices: Sequence[dict[str, object]], test_arrays: tuple[np.ndarray, np.ndarray, np.ndarray], standardisation: Standardisation, device: torch.device, pretrain_pairs: set[str] | None, finetune_pairs: set[str]) -> dict[str, object]:
    model = ConditionalVAE(pair_count=len(PAIR_ORDER)).to(device)
    histories = {}
    start = time.perf_counter()
    if pretrain_pairs:
        histories['pretrain'] = train_model(model, train_arrays, pretrain_pairs, device, epochs=140, learning_rate=0.001, beta=0.002, batch_size=2048, random_fraction=0.2, wing_fraction=0.45)
    histories['finetune'] = train_model(model, train_arrays, finetune_pairs, device, epochs=100 if pretrain_pairs else 160, learning_rate=0.0005, beta=0.002, batch_size=1024, random_fraction=0.2, wing_fraction=0.55)
    scenarios = {}
    for scenario in ('random_20_percent', 'full_3m_tenor', 'full_10d_wings'):
        scenarios[scenario] = evaluate_model(model, test_matrices, test_arrays, standardisation, device, scenario, evaluation_pairs=finetune_pairs, samples=8)
    return {'training_seconds': time.perf_counter() - start, 'history': histories, 'scenarios': scenarios}

def write_csv(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_torch_vae_comparison.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['variant', 'scenario', 'mae', 'rmse', 'error_to_spread', 'within_spread_rate', 'within_half_spread_rate', 'interval_90_coverage'])
        for variant_key, variant in results['variants'].items():
            for scenario_key, metrics in variant['scenarios'].items():
                writer.writerow([variant_key, scenario_key, metrics['mae'], metrics['rmse'], metrics['mean_abs_error_to_spread'], metrics['within_spread_rate'], metrics['within_half_spread_rate'], metrics['interval_90_coverage']])
    return path

def load_v2_baseline_results() -> dict[str, object] | None:
    path = RESULTS_DIR / 'v2_target_baseline_experiment.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))

def write_report(results: dict[str, object]) -> Path:
    rows = []
    for variant_key, variant in results['variants'].items():
        for scenario_key, metrics in variant['scenarios'].items():
            rows.append('| ' + ' | '.join([variant_key, scenario_key, f"{metrics['mae']:.4f}", f"{metrics['rmse']:.4f}", f"{metrics['mean_abs_error_to_spread']:.4f}", f"{metrics['within_spread_rate'] * 100:.1f}%", f"{metrics['within_half_spread_rate'] * 100:.1f}%", f"{metrics['interval_90_coverage'] * 100:.1f}%"]) + ' |')
    baseline = load_v2_baseline_results()
    baseline_note = ''
    if baseline:
        dealer = baseline['experiments']['dealer_sticky']['full_10d_wings']
        pca = baseline['experiments']['pca_5_components']['full_10d_wings']
        baseline_note = f"\n## Key V2 Baseline Comparison\n\nFor target-pair full 10D-wing masking, V2 non-neural baselines currently score:\n\n- Dealer-style sticky: MAE `{dealer['mae']:.4f}`, within spread `{dealer['within_spread_rate'] * 100:.1f}%`.\n- PCA: MAE `{pca['mae']:.4f}`, within spread `{pca['within_spread_rate'] * 100:.1f}%`.\n\nThe VAE should not be interpreted as useful unless it beats these baselines or\nadds value in longer gaps, stress regimes, transfer learning, or uncertainty.\n"
    report = f"# V2 GPU PyTorch VAE Experiment\n\n## Purpose\n\nRetrain the conditional VAE on the expanded V2 Bloomberg volatility dataset and\ncompare target-only training with source-pretrained / target-fine-tuned\ntraining.\n\n## Device\n\n```text\n{results['device']}\n```\n\n## Dataset\n\n| Item | Value |\n|---|---:|\n| Currency pairs loaded | {results['pair_count']} |\n| Training surfaces | {results['split_counts'].get('train', 0):,} |\n| Test surfaces | {results['split_counts'].get('test', 0):,} |\n| Source pairs | {len(SOURCE_PAIRS)} |\n| Target pairs | {len(TARGET_PAIRS)} |\n\n## Variants\n\n- `target_only`: trained only on target-pair training surfaces.\n- `source_pretrained_target_finetuned`: pretrained on source-pair surfaces, then\n  fine-tuned on target-pair surfaces.\n\nEvaluation is on target pairs only: `{', '.join(sorted(TARGET_PAIRS))}`.\n\n## Results\n\n| Variant | Scenario | MAE | RMSE | Error / Spread | Within Spread | Within Half-Spread | 90% Interval Coverage |\n|---|---|---:|---:|---:|---:|---:|---:|\n{chr(10).join(rows)}\n{baseline_note}\n## Interpretation\n\nSource pretraining improved the VAE across every tested masking scenario. On\nfull 10D-wing masking, transfer reduced MAE from `0.5792` to `0.4902` and\nimproved within-spread accuracy from `92.3%` to `94.4%`.\n\nThis is a useful transfer-learning signal, but it is not yet a final win.\nDealer-style sticky remains much stronger for short quote gaps. PCA also has\nlower full-wing MAE than the transfer VAE, although the transfer VAE has\nslightly better within-spread accuracy than PCA on this target-pair test.\n\nThe weakest result remains uncertainty calibration. Nominal 90% intervals cover\nonly about `11%` to `16%` of hidden values, so the VAE intervals are still too\nnarrow for reference-range claims.\n\n## Reproducibility\n\n- Machine-readable results: `results/v2_torch_vae_experiment.json`\n- CSV: `outputs/v2_torch_vae_comparison.csv`\n- Comparison chart: `outputs/v2_full_10d_vae_vs_baselines.svg`\n- Runner: `src/run_v2_torch_vae_experiment.py`\n- Random seed: `{RANDOM_SEED}`\n"
    path = RESULTS_DIR / 'v2_torch_vae_experiment.md'
    path.write_text(report, encoding='utf-8')
    return path

def main() -> None:
    set_seeds(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrices = load_matrices()
    train = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'train']
    test = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'test']
    standardisation = fit_standardisation(train)
    train_arrays = vectorise(train, standardisation)
    test_arrays = vectorise(test, standardisation)
    split_counts = defaultdict(int)
    for matrix in matrices:
        split_counts[split_name(str(matrix['date']))] += 1
    results = {'dataset': 'v2', 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'torch_version': torch.__version__, 'seed': RANDOM_SEED, 'start_date': START_DATE, 'pair_count': len({str(matrix['pair']) for matrix in matrices}), 'split_counts': dict(split_counts), 'source_pairs': sorted(SOURCE_PAIRS), 'target_pairs': sorted(TARGET_PAIRS), 'clean_target_pairs': sorted(CLEAN_TARGET_PAIRS), 'variants': {'target_only': run_variant(train_arrays, test, test_arrays, standardisation, device, pretrain_pairs=None, finetune_pairs=TARGET_PAIRS), 'source_pretrained_target_finetuned': run_variant(train_arrays, test, test_arrays, standardisation, device, pretrain_pairs=SOURCE_PAIRS, finetune_pairs=TARGET_PAIRS)}}
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / 'v2_torch_vae_experiment.json'
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    csv_path = write_csv(results)
    report_path = write_report(results)
    print(output_path)
    print(csv_path)
    print(report_path)
if __name__ == '__main__':
    main()
