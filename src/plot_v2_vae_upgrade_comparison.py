from __future__ import annotations
import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'

def load_results() -> list[dict[str, float | str]]:
    baseline = json.loads((RESULTS_DIR / 'v2_target_baseline_experiment.json').read_text(encoding='utf-8'))
    transfer = json.loads((RESULTS_DIR / 'v2_torch_vae_experiment.json').read_text(encoding='utf-8'))
    latent = json.loads((RESULTS_DIR / 'v2_vae_latent_optimisation_experiment.json').read_text(encoding='utf-8'))
    lagged = json.loads((RESULTS_DIR / 'v2_lagged_vae_experiment.json').read_text(encoding='utf-8'))
    scenario = 'full_10d_wings'
    return [{'method': 'Dealer sticky', 'mae': baseline['experiments']['dealer_sticky'][scenario]['mae'], 'within_spread': baseline['experiments']['dealer_sticky'][scenario]['within_spread_rate']}, {'method': 'PCA', 'mae': baseline['experiments']['pca_5_components'][scenario]['mae'], 'within_spread': baseline['experiments']['pca_5_components'][scenario]['within_spread_rate']}, {'method': 'Transfer VAE', 'mae': transfer['variants']['source_pretrained_target_finetuned']['scenarios'][scenario]['mae'], 'within_spread': transfer['variants']['source_pretrained_target_finetuned']['scenarios'][scenario]['within_spread_rate']}, {'method': 'Latent-opt VAE', 'mae': latent['scenarios'][scenario]['latent_optimised']['mae'], 'within_spread': latent['scenarios'][scenario]['latent_optimised']['within_spread_rate']}, {'method': 'Lagged transfer VAE', 'mae': lagged['scenarios'][scenario]['mae'], 'within_spread': lagged['scenarios'][scenario]['within_spread_rate']}]

def write_csv(rows: list[dict[str, float | str]]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'v2_vae_upgrade_full_10d_comparison.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['method', 'mae', 'within_spread'])
        writer.writeheader()
        writer.writerows(rows)
    return path

def write_svg(rows: list[dict[str, float | str]]) -> Path:
    path = OUTPUT_DIR / 'v2_vae_upgrade_full_10d_mae.svg'
    width = 820
    height = 430
    margin_left = 160
    margin_right = 40
    margin_top = 60
    bar_height = 42
    gap = 22
    max_value = max((float(row['mae']) for row in rows)) * 1.1
    plot_width = width - margin_left - margin_right
    colors = {'Dealer sticky': '#4C78A8', 'PCA': '#F58518', 'Transfer VAE': '#E45756', 'Latent-opt VAE': '#72B7B2', 'Lagged transfer VAE': '#54A24B'}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">V2 Full 10D-Wing MAE Comparison</text>', f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">Lower MAE is better; target-pair test split</text>']
    for index, row in enumerate(rows):
        y = margin_top + index * (bar_height + gap)
        method = str(row['method'])
        mae = float(row['mae'])
        bar_width = mae / max_value * plot_width
        lines.extend([f'<text x="{margin_left - 12}" y="{y + 27}" text-anchor="end" font-family="Arial" font-size="13">{method}</text>', f'<rect x="{margin_left}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{colors[method]}"/>', f'<text x="{margin_left + bar_width + 8}" y="{y + 27}" font-family="Arial" font-size="13">{mae:.4f}</text>'])
    axis_y = margin_top + len(rows) * (bar_height + gap) + 8
    lines.append(f'<line x1="{margin_left}" y1="{axis_y}" x2="{margin_left + plot_width}" y2="{axis_y}" stroke="#999"/>')
    for tick in [0.0, 0.2, 0.4, 0.6]:
        x = margin_left + tick / max_value * plot_width
        lines.extend([f'<line x1="{x}" y1="{axis_y}" x2="{x}" y2="{axis_y + 6}" stroke="#999"/>', f'<text x="{x}" y="{axis_y + 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="#555">{tick:.1f}</text>'])
    lines.append('</svg>')
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path

def main() -> None:
    rows = load_results()
    csv_path = write_csv(rows)
    svg_path = write_svg(rows)
    print(csv_path)
    print(svg_path)
if __name__ == '__main__':
    main()
