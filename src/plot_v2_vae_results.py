from __future__ import annotations
import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUTPUT_DIR = ROOT / 'outputs'
FULL_10D_MODELS = [('dealer_sticky', 'Dealer sticky', '#D97706', 'baseline'), ('pca_5_components', 'PCA', '#059669', 'baseline'), ('target_only', 'VAE target-only', '#7C3AED', 'vae'), ('source_pretrained_target_finetuned', 'VAE transfer', '#DB2777', 'vae')]

def load_rows() -> list[dict[str, object]]:
    baseline = json.loads((RESULTS_DIR / 'v2_target_baseline_experiment.json').read_text(encoding='utf-8'))
    vae = json.loads((RESULTS_DIR / 'v2_torch_vae_experiment.json').read_text(encoding='utf-8'))
    rows = []
    for key, label, color, source in FULL_10D_MODELS:
        if source == 'baseline':
            metrics = baseline['experiments'][key]['full_10d_wings']
        else:
            metrics = vae['variants'][key]['scenarios']['full_10d_wings']
        rows.append({'key': key, 'label': label, 'color': color, 'mae': float(metrics['mae']), 'within_spread': float(metrics['within_spread_rate']) * 100})
    return rows

def write_csv(rows: list[dict[str, object]]) -> Path:
    path = OUTPUT_DIR / 'v2_full_10d_vae_vs_baselines.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['key', 'label', 'mae', 'within_spread'])
        writer.writeheader()
        for row in rows:
            writer.writerow({'key': row['key'], 'label': row['label'], 'mae': row['mae'], 'within_spread': row['within_spread']})
    return path

def write_svg(rows: list[dict[str, object]]) -> Path:
    width, height = (1100, 620)
    margin_left, margin_right = (110, 50)
    margin_top, margin_bottom = (115, 115)
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_mae = max((float(row['mae']) for row in rows)) * 1.25
    bar_width = 115
    gap = 70
    start_x = margin_left + 65

    def y_position(value: float) -> float:
        return margin_top + plot_height - value / max_mae * plot_height
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#F8FAFC"/>', '<text x="55" y="48" font-family="Arial" font-size="28" font-weight="700" fill="#0F172A">V2 Full 10D-Wing Reconstruction</text>', '<text x="55" y="76" font-family="Arial" font-size="15" fill="#475569">Target-pair test set. Lower MAE is better; labels show within-spread rate.</text>']
    for tick in range(5):
        value = max_mae * tick / 4
        y = margin_top + plot_height - value / max_mae * plot_height
        svg.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#CBD5E1" stroke-width="1" opacity="0.65"/>')
        svg.append(f'<text x="{margin_left - 15}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#475569">{value:.2f}</text>')
    svg.append(f'<text x="27" y="{margin_top + plot_height / 2}" transform="rotate(-90 27 {margin_top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="14" fill="#334155">MAE</text>')
    for index, row in enumerate(rows):
        value = float(row['mae'])
        x = start_x + index * (bar_width + gap)
        y = y_position(value)
        bar_height = value / max_mae * plot_height
        svg.append(f'''<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{row['color']}" rx="4"/>''')
        svg.append(f'<text x="{x + bar_width / 2}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700" fill="#334155">{value:.3f}</text>')
        svg.append(f'''<text x="{x + bar_width / 2}" y="{margin_top + plot_height + 34}" text-anchor="middle" font-family="Arial" font-size="14" fill="#0F172A">{row['label']}</text>''')
        svg.append(f'''<text x="{x + bar_width / 2}" y="{margin_top + plot_height + 56}" text-anchor="middle" font-family="Arial" font-size="13" fill="#475569">{float(row['within_spread']):.1f}% within spread</text>''')
    svg.append('</svg>')
    path = OUTPUT_DIR / 'v2_full_10d_vae_vs_baselines.svg'
    path.write_text('\n'.join(svg), encoding='utf-8')
    return path

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = load_rows()
    print(write_csv(rows))
    print(write_svg(rows))
if __name__ == '__main__':
    main()
