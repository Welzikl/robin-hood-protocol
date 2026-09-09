from __future__ import annotations
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / 'outputs' / 'v2_lagged_vs_unlagged_long_gap_comparison.csv'
OUTPUT_DIR = ROOT / 'outputs'
MODELS = [('dealer_sticky', 'Dealer sticky', '#D97706'), ('pca_5_components', 'PCA', '#059669'), ('unlagged_transfer_vae', 'Unlagged VAE', '#7C3AED'), ('strict_lagged_transfer_vae', 'Strict lagged VAE', '#DC2626'), ('leaky_lagged_transfer_vae', 'Leaky lagged VAE', '#0284C7')]

def read_rows() -> list[dict[str, str]]:
    with INPUT_PATH.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def write_mae_chart(rows: list[dict[str, str]]) -> Path:
    width, height = (1180, 680)
    margin_left, margin_right = (100, 50)
    margin_top, margin_bottom = (110, 100)
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    gap_lengths = [1, 5, 10, 17]
    values = {model_key: {int(row['gap_length_weekdays']): float(row['mae']) for row in rows if row['model'] == model_key} for model_key, _, _ in MODELS}
    max_y = max((max(model_values.values()) for model_values in values.values())) * 1.15

    def x_position(gap_length: int) -> float:
        index = gap_lengths.index(gap_length)
        return margin_left + index * (plot_width / (len(gap_lengths) - 1))

    def y_position(value: float) -> float:
        return margin_top + plot_height - value / max_y * plot_height
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#F8FAFC"/>', '<text x="55" y="45" font-family="Arial" font-size="27" font-weight="700" fill="#0F172A">V2 Long-Gap 10D-Wing Reconstruction</text>', '<text x="55" y="74" font-family="Arial" font-size="15" fill="#475569">Lagged VAE helps one-day gaps but fails when the hidden wing stays missing across the lag window.</text>']
    for tick in range(6):
        value = max_y * tick / 5
        y = y_position(value)
        svg.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#CBD5E1" stroke-width="1" opacity="0.65"/>')
        svg.append(f'<text x="{margin_left - 15}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#475569">{value:.2f}</text>')
    for gap_length in gap_lengths:
        x = x_position(gap_length)
        svg.append(f'<text x="{x:.1f}" y="{margin_top + plot_height + 38}" text-anchor="middle" font-family="Arial" font-size="14" fill="#334155">{gap_length}</text>')
    svg.append(f'<text x="{margin_left + plot_width / 2}" y="{height - 27}" text-anchor="middle" font-family="Arial" font-size="14" fill="#334155">Artificial gap length in weekdays</text>')
    svg.append(f'<text x="28" y="{margin_top + plot_height / 2}" transform="rotate(-90 28 {margin_top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="14" fill="#334155">MAE on hidden 10D wing quotes</text>')
    for model_key, _, color in MODELS:
        points = [(x_position(gap_length), y_position(values[model_key][gap_length])) for gap_length in gap_lengths]
        path_data = ' '.join([('M' if index == 0 else 'L') + f'{x:.1f},{y:.1f}' for index, (x, y) in enumerate(points)])
        svg.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="4"/>')
        for gap_length, (x, y) in zip(gap_lengths, points):
            value = values[model_key][gap_length]
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
            svg.append(f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700" fill="{color}">{value:.3f}</text>')
    legend_x = 790
    for index, (_, model_label, color) in enumerate(MODELS):
        y = 35 + index * 24
        svg.append(f'<rect x="{legend_x}" y="{y}" width="16" height="16" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{y + 13}" font-family="Arial" font-size="14" fill="#334155">{model_label}</text>')
    svg.append('</svg>')
    output_path = OUTPUT_DIR / 'v2_lagged_vs_unlagged_long_gap_mae.svg'
    output_path.write_text('\n'.join(svg), encoding='utf-8')
    return output_path

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(write_mae_chart(read_rows()))
if __name__ == '__main__':
    main()
