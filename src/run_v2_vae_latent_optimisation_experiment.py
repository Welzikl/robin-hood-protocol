from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from run_v2_torch_vae_experiment import ConditionalVAE, PAIR_ORDER, RANDOM_SEED, SOURCE_PAIRS, TARGET_PAIRS, fit_standardisation, flatten, load_matrices, mask_for_scenario, split_name, train_model, vectorise
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'
SCENARIOS = ('random_20_percent', 'full_3m_tenor', 'full_10d_wings')
LATENT_OPT_STEPS = 100
LATENT_OPT_LR = 0.05
LATENT_PRIOR_WEIGHT = 0.02
BATCH_SIZE = 384

def set_seeds() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

def train_transfer_vae(train_arrays, device: torch.device) -> ConditionalVAE:
    model = ConditionalVAE(pair_count=len(PAIR_ORDER)).to(device)
    train_model(model, train_arrays, SOURCE_PAIRS, device, epochs=140, learning_rate=0.001, beta=0.002, batch_size=2048, random_fraction=0.2, wing_fraction=0.45)
    train_model(model, train_arrays, TARGET_PAIRS, device, epochs=100, learning_rate=0.0005, beta=0.002, batch_size=1024, random_fraction=0.2, wing_fraction=0.55)
    return model

def evaluate_predictions(prediction_matrix: np.ndarray, selected_rows: list[int], hidden_masks: np.ndarray, matrices: list[dict[str, object]], standardisation) -> dict[str, object]:
    errors = []
    spread_scaled_errors = []
    within_spread_hits = 0
    within_half_spread_hits = 0
    by_pair_errors: dict[str, list[float]] = defaultdict(list)
    by_pair_scaled_errors: dict[str, list[float]] = defaultdict(list)
    for selected_index, row_index in enumerate(selected_rows):
        matrix = matrices[row_index]
        pair = str(matrix['pair'])
        means = standardisation.pair_means[pair]
        stds = standardisation.pair_stds[pair]
        predicted_values = means + prediction_matrix[selected_index] * stds
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
                scaled_error = absolute_error / float(spread)
                spread_scaled_errors.append(scaled_error)
                by_pair_scaled_errors[pair].append(scaled_error)
                within_spread_hits += int(absolute_error <= spread)
                within_half_spread_hits += int(absolute_error <= spread / 2)
    return {'count': len(errors), 'mae': sum((abs(error) for error in errors)) / len(errors), 'rmse': math.sqrt(sum((error * error for error in errors)) / len(errors)), 'spread_observations': len(spread_scaled_errors), 'mean_abs_error_to_spread': sum(spread_scaled_errors) / len(spread_scaled_errors) if spread_scaled_errors else None, 'within_spread_rate': within_spread_hits / len(spread_scaled_errors) if spread_scaled_errors else None, 'within_half_spread_rate': within_half_spread_hits / len(spread_scaled_errors) if spread_scaled_errors else None, 'by_pair': {pair: {'count': len(pair_errors), 'mae': sum((abs(error) for error in pair_errors)) / len(pair_errors), 'rmse': math.sqrt(sum((error * error for error in pair_errors)) / len(pair_errors)), 'mean_abs_error_to_spread': sum(by_pair_scaled_errors[pair]) / len(by_pair_scaled_errors[pair]) if by_pair_scaled_errors[pair] else None} for pair, pair_errors in sorted(by_pair_errors.items())}}

def optimise_latents_for_batch(model: ConditionalVAE, batch_values: torch.Tensor, batch_masks: torch.Tensor, batch_pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    with torch.no_grad():
        initial_mu, _ = model.encode(batch_values, batch_masks, batch_pairs)
        initial_prediction = model.decode(initial_mu, batch_pairs)
    latent = initial_mu.detach().clone().requires_grad_(True)
    optimiser = torch.optim.Adam([latent], lr=LATENT_OPT_LR)
    final_loss = 0.0
    for _ in range(LATENT_OPT_STEPS):
        prediction = model.decode(latent, batch_pairs)
        visible_residual = (prediction - batch_values) * batch_masks
        fit_loss = visible_residual.pow(2).sum() / batch_masks.sum().clamp_min(1.0)
        prior_loss = (latent - initial_mu.detach()).pow(2).mean()
        loss = fit_loss + LATENT_PRIOR_WEIGHT * prior_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        final_loss = float(loss.detach().cpu())
    with torch.no_grad():
        optimised_prediction = model.decode(latent, batch_pairs)
        initial_visible_loss = ((initial_prediction - batch_values) * batch_masks).pow(2).sum() / batch_masks.sum().clamp_min(1.0)
        optimised_visible_loss = ((optimised_prediction - batch_values) * batch_masks).pow(2).sum() / batch_masks.sum().clamp_min(1.0)
    return (initial_prediction.detach(), optimised_prediction.detach(), float(initial_visible_loss.cpu()), float(optimised_visible_loss.cpu()))

def evaluate_scenario(model: ConditionalVAE, matrices: list[dict[str, object]], arrays, standardisation, device: torch.device, scenario: str) -> dict[str, object]:
    values_np, observed_np, pair_np = arrays
    selected_rows = [row_index for row_index, matrix in enumerate(matrices) if str(matrix['pair']) in TARGET_PAIRS]
    eval_masks = np.vstack([mask_for_scenario(observed_np[row_index], scenario, seed=RANDOM_SEED + row_index) for row_index in selected_rows]).astype(np.float32)
    hidden_masks = (observed_np[selected_rows] > 0.5) & (eval_masks < 0.5)
    values_tensor = torch.tensor(values_np[selected_rows], dtype=torch.float32)
    mask_tensor = torch.tensor(eval_masks, dtype=torch.float32)
    pair_tensor = torch.tensor(pair_np[selected_rows], dtype=torch.float32)
    encoder_predictions = np.zeros((len(selected_rows), 25), dtype=np.float32)
    optimised_predictions = np.zeros((len(selected_rows), 25), dtype=np.float32)
    initial_visible_losses = []
    optimised_visible_losses = []
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    for start in range(0, len(selected_rows), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(selected_rows))
        batch_values = values_tensor[start:end].to(device)
        batch_masks = mask_tensor[start:end].to(device)
        batch_pairs = pair_tensor[start:end].to(device)
        initial_prediction, optimised_prediction, initial_loss, optimised_loss = optimise_latents_for_batch(model, batch_values, batch_masks, batch_pairs)
        encoder_predictions[start:end] = initial_prediction.cpu().numpy()
        optimised_predictions[start:end] = optimised_prediction.cpu().numpy()
        initial_visible_losses.append(initial_loss)
        optimised_visible_losses.append(optimised_loss)
    encoder_metrics = evaluate_predictions(encoder_predictions, selected_rows, hidden_masks, matrices, standardisation)
    optimised_metrics = evaluate_predictions(optimised_predictions, selected_rows, hidden_masks, matrices, standardisation)
    return {'encoder_mu': encoder_metrics, 'latent_optimised': optimised_metrics, 'visible_fit': {'encoder_visible_mse': sum(initial_visible_losses) / len(initial_visible_losses), 'optimised_visible_mse': sum(optimised_visible_losses) / len(optimised_visible_losses), 'optimisation_steps': LATENT_OPT_STEPS, 'learning_rate': LATENT_OPT_LR, 'prior_weight': LATENT_PRIOR_WEIGHT}}

def write_csv(results: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_vae_latent_optimisation_comparison.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['scenario', 'method', 'mae', 'rmse', 'error_to_spread', 'within_spread_rate', 'within_half_spread_rate', 'count'])
        for scenario, scenario_results in results['scenarios'].items():
            for method in ('encoder_mu', 'latent_optimised'):
                metrics = scenario_results[method]
                writer.writerow([scenario, method, metrics['mae'], metrics['rmse'], metrics['mean_abs_error_to_spread'], metrics['within_spread_rate'], metrics['within_half_spread_rate'], metrics['count']])
    return path

def write_report(results: dict[str, object]) -> Path:
    rows = []
    for scenario, scenario_results in results['scenarios'].items():
        for method in ('encoder_mu', 'latent_optimised'):
            metrics = scenario_results[method]
            rows.append('| ' + ' | '.join([scenario, method, f"{metrics['mae']:.4f}", f"{metrics['rmse']:.4f}", f"{metrics['mean_abs_error_to_spread']:.4f}", f"{metrics['within_spread_rate'] * 100:.1f}%", f"{metrics['within_half_spread_rate'] * 100:.1f}%"]) + ' |')
    full_wing = results['scenarios']['full_10d_wings']
    encoder_mae = full_wing['encoder_mu']['mae']
    optimised_mae = full_wing['latent_optimised']['mae']
    improvement = (encoder_mae - optimised_mae) / encoder_mae
    report = f"# V2 VAE Latent Optimisation Experiment\n\n## Purpose\n\nTest whether the trained transfer VAE decoder has learned a useful volatility\nsurface manifold. The decoder is frozen at inference time. For each masked test\nsurface, only the latent vector is optimised to fit the visible quotes.\n\n## Design\n\n- Dataset: V2 target-pair test split.\n- Evaluation pairs: `{', '.join(sorted(TARGET_PAIRS))}`.\n- Base model: source-pretrained / target-fine-tuned VAE.\n- Decoder weights: frozen during latent optimisation.\n- Hidden values are never used during optimisation.\n- Objective: visible-cell reconstruction loss plus latent prior penalty.\n- Optimisation steps: `{LATENT_OPT_STEPS}`.\n- Latent prior weight: `{LATENT_PRIOR_WEIGHT}`.\n\n## Results\n\n| Scenario | Method | MAE | RMSE | Error / Spread | Within Spread | Within Half-Spread |\n|---|---|---:|---:|---:|---:|---:|\n{chr(10).join(rows)}\n\n## Interpretation\n\nFull 10D-wing encoder-MAE was `{encoder_mae:.4f}`. Full 10D-wing latent-optimised\nMAE was `{optimised_mae:.4f}`, a relative change of `{improvement * 100:.1f}%`.\n\nIf latent optimisation improves hidden-cell MAE, the VAE decoder has learned a\nuseful surface manifold and the one-pass encoder was limiting inference. If it\ndoes not improve hidden-cell MAE, the current decoder manifold is probably not\nstrong enough for this quote-completion task.\n\nImportant caveat: latent optimisation fits visible quotes only. It can improve\nvisible fit while still failing to improve hidden quote estimates, so hidden\nMAE remains the main result.\n\n## Reproducibility\n\n- Machine-readable results: `results/v2_vae_latent_optimisation_experiment.json`\n- CSV: `outputs/v2_vae_latent_optimisation_comparison.csv`\n- Runner: `src/run_v2_vae_latent_optimisation_experiment.py`\n- Random seed: `{RANDOM_SEED}`\n"
    path = RESULTS_DIR / 'v2_vae_latent_optimisation_experiment.md'
    path.write_text(report, encoding='utf-8')
    return path

def main() -> None:
    set_seeds()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrices = load_matrices()
    train = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'train']
    test = [matrix for matrix in matrices if split_name(str(matrix['date'])) == 'test']
    standardisation = fit_standardisation(train)
    train_arrays = vectorise(train, standardisation)
    test_arrays = vectorise(test, standardisation)
    model = train_transfer_vae(train_arrays, device)
    results: dict[str, object] = {'dataset': 'v2', 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'torch_version': torch.__version__, 'seed': RANDOM_SEED, 'target_pairs': sorted(TARGET_PAIRS), 'latent_optimisation': {'steps': LATENT_OPT_STEPS, 'learning_rate': LATENT_OPT_LR, 'prior_weight': LATENT_PRIOR_WEIGHT, 'batch_size': BATCH_SIZE}, 'scenarios': {}}
    for scenario in SCENARIOS:
        results['scenarios'][scenario] = evaluate_scenario(model, test, test_arrays, standardisation, device, scenario)
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / 'v2_vae_latent_optimisation_experiment.json'
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    csv_path = write_csv(results)
    report_path = write_report(results)
    print(output_path)
    print(csv_path)
    print(report_path)
if __name__ == '__main__':
    main()
