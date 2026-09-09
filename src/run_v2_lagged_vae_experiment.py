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
from run_v2_torch_vae_experiment import PAIR_ORDER, RANDOM_SEED, SOURCE_PAIRS, TARGET_PAIRS, fit_standardisation, flatten, load_matrices, mask_for_scenario, split_name
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'
SCENARIOS = ('random_20_percent', 'full_3m_tenor', 'full_10d_wings')
LAG_STEPS = (1, 5)

class LaggedConditionalVAE(nn.Module):

    def __init__(self, pair_count: int, surface_dim: int=25, lag_count: int=2, hidden_dim: int=512, latent_dim: int=24) -> None:
        super().__init__()
        lag_dim = surface_dim * lag_count
        encoder_input_dim = surface_dim * 2 + lag_dim + pair_count
        decoder_input_dim = latent_dim + lag_dim + pair_count
        self.encoder = nn.Sequential(nn.Linear(encoder_input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU())
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(decoder_input_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, surface_dim))

    def encode(self, values: torch.Tensor, mask: torch.Tensor, lag_values: torch.Tensor, pair_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(torch.cat([values * mask, mask, lag_values, pair_onehot], dim=1))
        return (self.mu(encoded), self.logvar(encoded))

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor, lag_values: torch.Tensor, pair_onehot: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([latent, lag_values, pair_onehot], dim=1))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, lag_values: torch.Tensor, pair_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(values, mask, lag_values, pair_onehot)
        latent = self.reparameterise(mu, logvar)
        return (self.decode(latent, lag_values, pair_onehot), mu, logvar)

def set_seeds() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

def build_lagged_rows(matrices: Sequence[dict[str, object]], standardisation) -> list[dict[str, object]]:
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

def rows_to_arrays(rows: Sequence[dict[str, object]]):
    return (np.vstack([row['values'] for row in rows]).astype(np.float32), np.vstack([row['mask'] for row in rows]).astype(np.float32), np.vstack([row['lags'] for row in rows]).astype(np.float32), np.vstack([row['pair_onehot'] for row in rows]).astype(np.float32))

def make_training_masks(observed: torch.Tensor, random_fraction: float, wing_fraction: float) -> torch.Tensor:
    keep = (torch.rand_like(observed) > random_fraction).float() * observed
    wing_rows = torch.rand(observed.shape[0], device=observed.device) < wing_fraction
    for column_index in (3, 4):
        keep[wing_rows, column_index::5] = 0.0
    return keep

def train_model(model: LaggedConditionalVAE, train_arrays, allowed_pairs: set[str], device: torch.device, epochs: int, learning_rate: float, beta: float, batch_size: int, random_fraction: float, wing_fraction: float) -> list[dict[str, float]]:
    values_np, mask_np, lags_np, pair_np = train_arrays
    allowed_indices = {PAIR_ORDER.index(pair) for pair in allowed_pairs}
    row_mask = np.array([int(np.argmax(row)) in allowed_indices for row in pair_np], dtype=bool)
    values = torch.tensor(values_np[row_mask], dtype=torch.float32)
    observed = torch.tensor(mask_np[row_mask], dtype=torch.float32)
    lags = torch.tensor(lags_np[row_mask], dtype=torch.float32)
    pairs = torch.tensor(pair_np[row_mask], dtype=torch.float32)
    loader = DataLoader(TensorDataset(values, observed, lags, pairs), batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=torch.cuda.is_available())
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_reconstruction = 0.0
        total_kl = 0.0
        batches = 0
        for batch_values, batch_observed, batch_lags, batch_pairs in loader:
            batch_values = batch_values.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            batch_lags = batch_lags.to(device, non_blocking=True)
            batch_pairs = batch_pairs.to(device, non_blocking=True)
            input_mask = make_training_masks(batch_observed, random_fraction, wing_fraction)
            prediction, mu, logvar = model(batch_values, input_mask, batch_lags, batch_pairs)
            reconstruction = ((prediction - batch_values) ** 2 * batch_observed).sum()
            reconstruction = reconstruction / batch_observed.sum().clamp_min(1.0)
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

def run_lagged_variant(train_arrays, device: torch.device, pretrain_pairs: set[str] | None, finetune_pairs: set[str]) -> tuple[LaggedConditionalVAE, dict[str, list[dict[str, float]]]]:
    model = LaggedConditionalVAE(pair_count=len(PAIR_ORDER), lag_count=len(LAG_STEPS)).to(device)
    history: dict[str, list[dict[str, float]]] = {}
    if pretrain_pairs:
        history['pretrain'] = train_model(model=model, train_arrays=train_arrays, allowed_pairs=pretrain_pairs, device=device, epochs=140, learning_rate=0.001, beta=0.002, batch_size=2048, random_fraction=0.2, wing_fraction=0.45)
    history['finetune'] = train_model(model=model, train_arrays=train_arrays, allowed_pairs=finetune_pairs, device=device, epochs=100 if pretrain_pairs else 160, learning_rate=0.0005, beta=0.002, batch_size=1024, random_fraction=0.2, wing_fraction=0.55)
    return (model, history)

def train_lagged_transfer_vae(train_arrays, device: torch.device):
    return run_lagged_variant(train_arrays, device, SOURCE_PAIRS, TARGET_PAIRS)

def evaluate_predictions(predictions: np.ndarray, selected_rows: list[dict[str, object]], hidden_masks: np.ndarray, standardisation) -> dict[str, object]:
    errors = []
    spread_scaled_errors = []
    within_spread_hits = 0
    within_half_spread_hits = 0
    by_pair_errors: dict[str, list[float]] = defaultdict(list)
    for selected_index, row in enumerate(selected_rows):
        matrix = row['matrix']
        pair = str(matrix['pair'])
        means = standardisation.pair_means[pair]
        stds = standardisation.pair_stds[pair]
        predicted_values = means + predictions[selected_index] * stds
        actual_values = np.array([float(value) if value is not None else np.nan for value in flatten(matrix['values'])], dtype=np.float32)
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
    return {'count': len(errors), 'mae': sum((abs(error) for error in errors)) / len(errors), 'rmse': math.sqrt(sum((error * error for error in errors)) / len(errors)), 'spread_observations': len(spread_scaled_errors), 'mean_abs_error_to_spread': sum(spread_scaled_errors) / len(spread_scaled_errors), 'within_spread_rate': within_spread_hits / len(spread_scaled_errors), 'within_half_spread_rate': within_half_spread_hits / len(spread_scaled_errors), 'by_pair': {pair: {'count': len(pair_errors), 'mae': sum((abs(error) for error in pair_errors)) / len(pair_errors), 'rmse': math.sqrt(sum((error * error for error in pair_errors)) / len(pair_errors))} for pair, pair_errors in sorted(by_pair_errors.items())}}

def evaluate_scenario(model: LaggedConditionalVAE, rows: Sequence[dict[str, object]], arrays, standardisation, device: torch.device, scenario: str) -> dict[str, object]:
    values_np, observed_np, lags_np, pair_np = arrays
    selected_indices = [index for index, row in enumerate(rows) if str(row['matrix']['pair']) in TARGET_PAIRS]
    selected_rows = [rows[index] for index in selected_indices]
    eval_masks = np.vstack([mask_for_scenario(observed_np[index], scenario, seed=RANDOM_SEED + index) for index in selected_indices]).astype(np.float32)
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
    return evaluate_predictions(predictions, selected_rows, hidden_masks, standardisation)

def write_csv(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_lagged_vae_comparison.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['variant', 'scenario', 'mae', 'rmse', 'error_to_spread', 'within_spread_rate', 'within_half_spread_rate', 'count'])
        for variant_key, variant in results['variants'].items():
            for scenario, metrics in variant['scenarios'].items():
                writer.writerow([variant_key, scenario, metrics['mae'], metrics['rmse'], metrics['mean_abs_error_to_spread'], metrics['within_spread_rate'], metrics['within_half_spread_rate'], metrics['count']])
    return path

def write_report(results: dict[str, object]) -> Path:
    rows = []
    for variant_key, variant in results['variants'].items():
        for scenario, metrics in variant['scenarios'].items():
            rows.append('| ' + ' | '.join([variant_key, scenario, f"{metrics['mae']:.4f}", f"{metrics['rmse']:.4f}", f"{metrics['mean_abs_error_to_spread']:.4f}", f"{metrics['within_spread_rate'] * 100:.1f}%", f"{metrics['within_half_spread_rate'] * 100:.1f}%", f"{metrics['count']:,}"]) + ' |')
    target_full = results['variants']['target_only']['scenarios']['full_10d_wings']
    transfer_full = results['variants']['source_pretrained_target_finetuned']['scenarios']['full_10d_wings']
    relative_change = (transfer_full['mae'] - target_full['mae']) / target_full['mae']
    if relative_change < 0:
        transfer_summary = f'{abs(relative_change) * 100:.1f}% lower MAE'
        transfer_meaning = 'Transfer improves the lagged VAE on the full 10D-wing task.'
    else:
        transfer_summary = f'{relative_change * 100:.1f}% higher MAE'
        transfer_meaning = 'Transfer does not improve the lagged VAE on the full 10D-wing task.'
    report = f"# V2 Lagged Conditional VAE Transfer-Learning Experiment\n\n## Purpose\n\nTest the user's question: if lagged conditioning made the VAE much stronger,\nshould transfer learning still help after adding lagged surface information?\n\n## Design\n\nTwo lagged conditional VAE variants are compared with the same architecture,\nlag inputs, masking scenarios, target-pair evaluation set, and test split.\n\n- `target_only`: trained only on target-pair training surfaces.\n- `source_pretrained_target_finetuned`: pretrained on source/liquid-pair training\n  surfaces, then fine-tuned on target-pair training surfaces.\n- Lag inputs: `{', '.join((str(step) for step in LAG_STEPS))}` previous pair-date surfaces.\n- Evaluation pairs: `{', '.join(sorted(TARGET_PAIRS))}`.\n- Evaluation period: unseen test split.\n- Hidden values are created by controlled masking.\n- Only prior surfaces are used as lag features; no future surfaces are used.\n\n## Results\n\n| Variant | Scenario | MAE | RMSE | Error / Spread | Within Spread | Within Half-Spread | Hidden Quotes |\n|---|---|---:|---:|---:|---:|---:|---:|\n{chr(10).join(rows)}\n\n## Key Full 10D-Wing Comparison\n\n| Comparison | MAE | Meaning |\n|---|---:|---|\n| Lagged target-only VAE | {target_full['mae']:.4f} | Lagged VAE trained only on target pairs. |\n| Lagged transfer VAE | {transfer_full['mae']:.4f} | Lagged VAE pretrained on source pairs then fine-tuned on target pairs. |\n| Transfer change | {transfer_summary} | {transfer_meaning} |\n\n## Interpretation\n\nThis is the fairer version of the transfer-learning question after adding the\nlagged conditional VAE. It asks whether the VAE benefits from liquid-market\npretraining even when it already receives recent target-pair surface history.\n\nImportant caveat: this is still a reconstruction experiment, not a true\nall-quotes-missing forecast. The current-day visible surface remains available\nunder the controlled masks.\n\n## Reproducibility\n\n- Machine-readable results: `results/v2_lagged_vae_experiment.json`\n- CSV: `outputs/v2_lagged_vae_comparison.csv`\n- Runner: `src/run_v2_lagged_vae_experiment.py`\n- Random seed: `{RANDOM_SEED}`\n"
    path = RESULTS_DIR / 'v2_lagged_vae_experiment.md'
    path.write_text(report, encoding='utf-8')
    return path

def main() -> None:
    set_seeds()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrices = load_matrices()
    train_matrices = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'train']
    standardisation = fit_standardisation(train_matrices)
    lagged_rows = build_lagged_rows(matrices, standardisation)
    train_rows = [row for row in lagged_rows if split_name(str(row['matrix']['date'])) == 'train']
    test_rows = [row for row in lagged_rows if split_name(str(row['matrix']['date'])) == 'test']
    train_arrays = rows_to_arrays(train_rows)
    test_arrays = rows_to_arrays(test_rows)
    variant_specs = {'target_only': {'pretrain_pairs': None, 'finetune_pairs': TARGET_PAIRS}, 'source_pretrained_target_finetuned': {'pretrain_pairs': SOURCE_PAIRS, 'finetune_pairs': TARGET_PAIRS}}
    results: dict[str, object] = {'dataset': 'v2', 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'torch_version': torch.__version__, 'seed': RANDOM_SEED, 'source_pairs': sorted(SOURCE_PAIRS), 'target_pairs': sorted(TARGET_PAIRS), 'lag_steps': list(LAG_STEPS), 'train_rows': len(train_rows), 'test_rows': len(test_rows), 'variants': {}}
    for variant_key, spec in variant_specs.items():
        set_seeds()
        model, history = run_lagged_variant(train_arrays=train_arrays, device=device, pretrain_pairs=spec['pretrain_pairs'], finetune_pairs=spec['finetune_pairs'])
        scenarios = {}
        for scenario in SCENARIOS:
            scenarios[scenario] = evaluate_scenario(model, test_rows, test_arrays, standardisation, device, scenario)
        results['variants'][variant_key] = {'training_history': history, 'scenarios': scenarios}
    results['training_history'] = results['variants']['source_pretrained_target_finetuned']['training_history']
    results['scenarios'] = results['variants']['source_pretrained_target_finetuned']['scenarios']
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / 'v2_lagged_vae_experiment.json'
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    csv_path = write_csv(results)
    report_path = write_report(results)
    print(output_path)
    print(csv_path)
    print(report_path)
if __name__ == '__main__':
    main()
