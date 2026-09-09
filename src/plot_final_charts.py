from pathlib import Path
import json, csv, hashlib, shutil, zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1] / 'outputs'
ROOT.mkdir(exist_ok=True)
SRC = Path(__file__).resolve().parents[1] / 'results'
DATA = SRC
names = ['v2_regime_switching_framework', 'v2_dynamic_pca_experiment', 'v2_outage_aware_transfer_vae_experiment']
loaded = {}
manifest = {}
for n in names:
    p = SRC / (n + '.json')
    if not p.exists():
        p = DATA / (n + '.json')
    loaded[n] = json.loads(p.read_text(encoding='utf-8'))
    shutil.copyfile(p, DATA / p.name) if p.resolve() != (DATA / p.name).resolve() else None
    manifest[n] = {'source_path': 'results/' + p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
r, d, o = [loaded[n] for n in names]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': '#263341', 'text.color': '#263341', 'axes.titleweight': 'bold', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
C = {'persistence': '#24364B', 'sticky': '#D38327', 'pca': '#7D8792', 'dynamic': '#327B9D', 'target': '#9564A4', 'transfer': '#008A7A', 'router': '#B32945'}

def save(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(ROOT / (name + '.' + ext), dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def clean(ax):
    ax.grid(axis='y', color='#E5E8EB', lw=0.7)
    ax.set_axisbelow(True)
fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.35], hspace=0.55, top=0.73, bottom=0.12)
tenors = ['1W', '1M', '3M', '6M', '1Y']
cols = ['ATM', '25Δ\nRR', '25Δ\nBF', '10Δ\nRR', '10Δ\nBF']
random = {(0, 1), (1, 4), (2, 0), (3, 2), (4, 3)}
for k, title in enumerate(['A  Random missing quotes', 'B  Missing maturity (3M)', 'C  Missing 10-delta wings']):
    ax = fig.add_subplot(gs[0, k])
    ax.set_title(title, loc='left', fontsize=12, pad=18)
    for row in range(5):
        for col in range(5):
            hidden = (row, col) in random if k == 0 else row == 2 if k == 1 else col >= 3
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor='#F9E1D3' if hidden else '#E1EBEF', edgecolor='white', linewidth=2, hatch='///' if hidden else None))
    ax.set(xlim=(0, 5), ylim=(5, 0), xticks=np.arange(5) + 0.5, yticks=np.arange(5) + 0.5, xticklabels=cols, yticklabels=tenors)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=6)
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_visible(False)
ax = fig.add_subplot(gs[1, :])
ax.axis('off')
ax.set_title('D  Consecutive wing outage', loc='left', fontsize=12)
for i, label in enumerate(['Last complete\npre-gap surface', 'Gap day 1', 'Gap day 2', '…', 'Gap day h']):
    x = 0.02 + i * 0.19
    ax.add_patch(Rectangle((x, 0.28), 0.17, 0.42, transform=ax.transAxes, facecolor='#E1EBEF' if i == 0 else '#F9E1D3', edgecolor='white', hatch=None if i == 0 else '///'))
    ax.text(x + 0.085, 0.49, label, transform=ax.transAxes, ha='center', va='center', fontsize=10)
ax.text(0.02, 0.02, 'During the gap: both 10Δ columns are hidden. Unavailable lagged wings must not be supplied to the model.', transform=ax.transAxes, fontsize=10)
fig.legend(handles=[Patch(facecolor='#E1EBEF', label='Visible quote'), Patch(facecolor='#F9E1D3', hatch='///', label='Deliberately hidden quote')], loc='lower center', bbox_to_anchor=(0.5, -0.035), ncol=2, frameon=False)
fig.suptitle('What is being reconstructed?', x=0.08, ha='left', fontsize=18, y=1.02)
fig.text(0.08, 0.965, 'Illustrative masks on a 5 × 5 FX quote grid — not observed market data', fontsize=11)
save(fig, '01_missing_quote_schematic')
metrics = r['test_metrics']
keys = ['validation_regime_switcher', 'previous_surface_only', 'sticky_only', 'lagged_target_only_vae_only', 'lagged_transfer_vae_only', 'pca_only']
labels = ['Validation-selected router', 'Previous-surface persistence', 'Dealer-style sticky', 'Lagged target-only VAE', 'Lagged transfer VAE', 'Static PCA (5 factors)']
colors = [C[k] for k in ['router', 'persistence', 'sticky', 'target', 'transfer', 'pca']]
assert all((metrics[k]['count'] == 38460 for k in keys))
counts = metrics[keys[0]]['method_counts']
assert counts == {'lagged_transfer_vae': 5780, 'previous_surface': 32680}
values = [metrics[k]['point_mae'] for k in keys]
gain = 100 * (1 - values[0] / values[1])
fig, ax = plt.subplots(figsize=(10.5, 5.8))
fig.subplots_adjust(left=0.29, bottom=0.25, top=0.8, right=0.92)
y = np.arange(len(keys))
ax.barh(y, values, color=colors, height=0.6)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(0, max(values) * 1.18)
for i, v in enumerate(values):
    ax.text(v + 0.006, i, f'{v:.4f}', va='center', fontsize=11)
ax.set_xlabel('Hidden-quote MAE (volatility percentage points; lower is better)')
ax.grid(axis='x', color='#E5E8EB')
ax.set_axisbelow(True)
fig.text(0.06, 0.94, 'A small routing gain, not a universal VAE advantage', fontsize=17, weight='bold')
fig.text(0.06, 0.875, 'Full 10-delta wings • six target pairs • 38,460 hidden test values', fontsize=11)
fig.text(0.06, 0.105, f'Router MAE is {gain:.1f}% lower than persistence on this chronological holdout.\nStatistical superiority has not been established.', fontsize=11)
fig.text(0.06, 0.035, 'Route allocation: 5,780 values to the lagged transfer VAE; 32,680 to persistence.', fontsize=10)
save(fig, '02_final_router_comparison')
gaps = [1, 5, 10, 17]
ex = o['experiments']
dx = d['test_experiments']
for a, b in [('previous_surface', 'previous_surface'), ('dealer_sticky', 'dealer_sticky'), ('pca_5_components', 'static_pca_5_components')]:
    for g in gaps:
        for field in ['mae', 'count', 'evaluated_blocks']:
            assert ex[a][str(g)][field] == dx[b][str(g)][field], (a, g, field)
allseries = {**ex, 'dynamic_pca': dx['dynamic_pca']}
for g in gaps:
    assert len({v[str(g)]['count'] for v in allseries.values()}) == 1
    assert len({v[str(g)]['evaluated_blocks'] for v in allseries.values()}) == 1
    assert ex['previous_surface'][str(g)]['mae'] == min((v[str(g)]['mae'] for v in allseries.values()))
fig, axs = plt.subplots(1, 2, figsize=(13, 6.8))
fig.subplots_adjust(top=0.78, bottom=0.28, wspace=0.25)
series = [('previous_surface', 'Persistence', 'persistence', 'o', '-'), ('dealer_sticky', 'Sticky', 'sticky', 's', '-'), ('pca_5_components', 'Static PCA', 'pca', '^', '--'), ('dynamic_pca', 'Dynamic PCA', 'dynamic', 'D', '--'), ('outage_curriculum_target_only_vae', 'Outage-trained target-only VAE', 'target', 'v', '-'), ('outage_curriculum_transfer_vae', 'Outage-trained transfer VAE', 'transfer', 'P', '-')]
for key, label, col, marker, ls in series:
    axs[0].plot(gaps, [allseries[key][str(g)]['mae'] for g in gaps], label=label, color=C[col], marker=marker, ls=ls, lw=2, ms=6)
for key, label, col, marker, ls in [series[0], series[4], series[5], ('single_date_target_only_vae', 'Single-date target-only VAE', 'target', 'v', ':'), ('single_date_transfer_vae', 'Single-date transfer VAE', 'transfer', 'P', ':')]:
    axs[1].plot(gaps, [allseries[key][str(g)]['mae'] for g in gaps], label=label, color=C[col], marker=marker, ls=ls, lw=2, ms=6)
for ax, title, upper in zip(axs, ['A  Final benchmark comparison', 'B  Effect of outage-aware training'], [0.55, 1.65]):
    ax.set_title(title, loc='left', fontsize=12, pad=12)
    ax.set_xticks(gaps)
    ax.set_xlabel('Artificial outage length (weekdays)')
    ax.set_ylim(0, upper)
    ax.set_xlim(0.4, 17.6)
    clean(ax)
    ax.set_ylabel('Hidden-quote MAE (volatility percentage points)')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.19), fontsize=8.5, frameon=False, ncol=2)
fig.text(0.065, 0.955, 'Longer outages: training helps, persistence still leads', fontsize=17, weight='bold')
fig.text(0.065, 0.885, 'Controlled 10-delta-wing gaps • six target pairs • overlapping test blocks\nPanel scales differ; both start at zero. Single-date controls are not the earlier leaky diagnostic.', fontsize=10.5)
fig.text(0.065, 0.015, 'At 17 weekdays: persistence 0.2850; outage-trained transfer 0.4492; outage-trained target-only 0.4642.', fontsize=10.5)
save(fig, '03_outage_length_comparison')
with (ROOT / 'chart_values.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['chart', 'method', 'gap_weekdays', 'mae', 'hidden_values', 'evaluated_blocks'])
    for k in keys:
        w.writerow(['router', k, '', metrics[k]['point_mae'], metrics[k]['count'], ''])
    for k, v in allseries.items():
        for g in gaps:
            w.writerow(['outage', k, g, v[str(g)]['mae'], v[str(g)]['count'], v[str(g)]['evaluated_blocks']])
manifest['verification'] = {'chart_count': 3, 'router_methods': len(keys), 'outage_methods': len(allseries), 'gap_lengths': gaps, 'common_control_metrics_exactly_equal': True, 'persistence_lowest_at_every_gap': True, 'router_counts': counts, 'manuscripts_modified': False}
(ROOT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest['verification'], indent=2))
