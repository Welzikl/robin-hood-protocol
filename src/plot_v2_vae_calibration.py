from __future__ import annotations
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / 'outputs' / 'v2_vae_calibration_comparison.csv'
OUTPUT_DIR = ROOT / 'outputs'
METHODS = [('raw_vae_interval', 'Raw VAE', '#64748B'), ('global_residual_interval', 'Global residual', '#2563EB'), ('cell_residual_interval', 'Cell residual', '#059669'), ('spread_scaled_interval', 'Spread-scaled', '#D97706')]
SCENARIOS = [('random_20_percent', 'Random 20%'), ('full_3m_tenor', 'Full 3M'), ('full_10d_wings', 'Full 10D wings')]

def read_rows() -> dict[tuple[str, str], float]:
    with INPUT_PATH.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return {(row['scenario'], row['method']): float(row['coverage']) * 100.0 for row in reader}

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    coverages = read_rows()
    width, height = (1280, 740)
    margin_left, margin_right = (105, 45)
    margin_top, margin_bottom = (115, 145)
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    group_width = plot_width / len(SCENARIOS)
    bar_width = 48
    gap = 10

    def y_position(value: float) -> float:
        return margin_top + plot_height - value / 100.0 * plot_height
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#F8FAFC"/>', '<text x="55" y="48" font-family="Arial" font-size="28" font-weight="700" fill="#0F172A">V2 VAE Interval Calibration</text>', '<text x="55" y="76" font-family="Arial" font-size="15" fill="#475569">Coverage should be close to the 90% target line without making intervals too wide.</text>']
    for tick in range(0, 101, 25):
        y = y_position(float(tick))
        svg.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#CBD5E1" stroke-width="1" opacity="0.65"/>')
        svg.append(f'<text x="{margin_left - 15}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#475569">{tick}%</text>')
    target_y = y_position(90.0)
    svg.append(f'<line x1="{margin_left}" y1="{target_y:.1f}" x2="{width - margin_right}" y2="{target_y:.1f}" stroke="#DC2626" stroke-width="2" stroke-dasharray="7,6"/>')
    svg.append(f'<text x="{width - margin_right - 4}" y="{target_y - 8:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#DC2626">90% target</text>')
    legend_x = 630
    for index, (_, label, color) in enumerate(METHODS):
        x = legend_x + index % 2 * 250
        y = 43 + index // 2 * 25
        svg.append(f'<rect x="{x}" y="{y}" width="17" height="17" fill="{color}"/>')
        svg.append(f'<text x="{x + 24}" y="{y + 13}" font-family="Arial" font-size="14" fill="#334155">{label}</text>')
    for scenario_index, (scenario_key, scenario_label) in enumerate(SCENARIOS):
        center = margin_left + group_width * scenario_index + group_width / 2
        total_bar_width = len(METHODS) * bar_width + (len(METHODS) - 1) * gap
        start_x = center - total_bar_width / 2
        for method_index, (method_key, _, color) in enumerate(METHODS):
            value = coverages[scenario_key, method_key]
            x = start_x + method_index * (bar_width + gap)
            y = y_position(value)
            bar_height = value / 100.0 * plot_height
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}" rx="4"/>')
            svg.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700" fill="#334155">{value:.1f}%</text>')
        svg.append(f'<text x="{center:.1f}" y="{margin_top + plot_height + 42}" text-anchor="middle" font-family="Arial" font-size="15" fill="#0F172A">{scenario_label}</text>')
    svg.append('</svg>')
    output_path = OUTPUT_DIR / 'v2_vae_calibration_coverage.svg'
    output_path.write_text('\n'.join(svg), encoding='utf-8')
    print(output_path)
if __name__ == '__main__':
    main()
