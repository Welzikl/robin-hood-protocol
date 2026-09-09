from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / 'data' / 'interim' / 'v2' / 'fx_volatility_long.csv'
OUTPUT_DIR = ROOT / 'outputs'
RESULTS_DIR = ROOT / 'results'
COMMON_START = pd.Timestamp('2010-01-01')
VALIDATION_START = pd.Timestamp('2022-01-01')
TEST_START = pd.Timestamp('2024-01-01')
SOURCE_PAIRS = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
TARGET_PAIRS = ['USDCNH', 'USDHUF', 'USDMXN', 'USDPLN', 'USDTRY', 'USDZAR']
PAIR_ORDER = [*SOURCE_PAIRS, *TARGET_PAIRS]
QUOTE_ORDER = ['ATM', '25D_RR', '25D_BF', '10D_RR', '10D_BF']
ROLE_ORDER = ['source', 'target']
ROLE_COLORS = {'source': '#2563EB', 'target': '#D97706'}

def _normalise_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    mapping = {'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False}
    normalised = series.astype(str).str.strip().str.lower().map(mapping)
    if normalised.isna().any():
        unknown = sorted(series[normalised.isna()].astype(str).unique().tolist())
        raise ValueError(f'Unknown boolean values: {unknown}')
    return normalised.astype(bool)

def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['date'] = pd.to_datetime(result['date'], errors='raise')
    result['is_observed'] = _normalise_boolean(result['is_observed'])
    if 'midpoint_flag' in result:
        result['midpoint_flag'] = _normalise_boolean(result['midpoint_flag'])
    else:
        result['midpoint_flag'] = False
    if 'spread' not in result:
        result['spread'] = np.nan
    for column in ('last', 'spread'):
        result[column] = pd.to_numeric(result[column], errors='coerce')
    return result[result['date'] >= COMMON_START].copy()

def build_pair_summary(frame: pd.DataFrame, expected_cells: int=25) -> pd.DataFrame:
    data = prepare_frame(frame)
    matrix_quality = data.groupby(['pair', 'date'], observed=True).agg(observed_cells=('is_observed', 'sum'), cells=('is_observed', 'size')).reset_index()
    matrix_quality['complete'] = (matrix_quality['cells'] == expected_cells) & (matrix_quality['observed_cells'] == expected_cells)
    complete = matrix_quality.groupby('pair', observed=True)['complete'].agg(['sum', 'count'])
    rows = []
    for pair, group in data.groupby('pair', observed=True):
        available_spreads = group['spread'].dropna()
        observed_count = int(group['is_observed'].sum())
        total_count = int(len(group))
        rows.append({'pair': pair, 'role': str(group['role'].iloc[0]), 'date_start': group['date'].min().date().isoformat(), 'date_end': group['date'].max().date().isoformat(), 'daily_matrices': int(group['date'].nunique()), 'cell_records': total_count, 'observed_last_count': observed_count, 'missing_last_count': total_count - observed_count, 'missing_last_rate': (total_count - observed_count) / total_count, 'midpoint_flag_count': int(group['midpoint_flag'].sum()), 'midpoint_flag_rate': float(group['midpoint_flag'].mean()), 'complete_matrices': int(complete.loc[pair, 'sum']), 'complete_matrix_rate': float(complete.loc[pair, 'sum'] / complete.loc[pair, 'count']), 'median_spread': float(available_spreads.median()), 'p90_spread': float(available_spreads.quantile(0.9))})
    summary = pd.DataFrame(rows)
    role_rank = {role: index for index, role in enumerate(ROLE_ORDER)}
    pair_rank = {pair: index for index, pair in enumerate(PAIR_ORDER)}
    summary['_role_rank'] = summary['role'].map(role_rank)
    summary['_pair_rank'] = summary['pair'].map(pair_rank)
    return summary.sort_values(['_role_rank', '_pair_rank']).drop(columns=['_role_rank', '_pair_rank']).reset_index(drop=True)

def build_role_quote_summary(frame: pd.DataFrame) -> pd.DataFrame:
    data = prepare_frame(frame)
    rows = []
    for (role, quote_type), group in data.groupby(['role', 'quote_type'], observed=True):
        observed = group.loc[group['is_observed'] & group['last'].notna(), 'last']
        spreads = group['spread'].dropna()
        total = int(len(group))
        rows.append({'role': role, 'quote_type': quote_type, 'cell_records': total, 'observed_count': int(len(observed)), 'missing_rate': 1.0 - len(observed) / total, 'mean': float(observed.mean()), 'std': float(observed.std(ddof=1)), 'median': float(observed.median()), 'p10': float(observed.quantile(0.1)), 'p90': float(observed.quantile(0.9)), 'median_spread': float(spreads.median()), 'p90_spread': float(spreads.quantile(0.9))})
    result = pd.DataFrame(rows)
    result['role'] = pd.Categorical(result['role'], categories=ROLE_ORDER, ordered=True)
    result['quote_type'] = pd.Categorical(result['quote_type'], categories=QUOTE_ORDER, ordered=True)
    return result.sort_values(['role', 'quote_type']).reset_index(drop=True)

def build_atm_change_correlation(frame: pd.DataFrame, min_periods: int=250) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = prepare_frame(frame)
    atm = data[data['is_observed'] & (data['tenor'] == '3M') & (data['quote_type'] == 'ATM') & data['last'].notna()]
    pivot = atm.pivot_table(index='date', columns='pair', values='last', aggfunc='first').sort_index()
    ordered = [pair for pair in PAIR_ORDER if pair in pivot.columns]
    ordered.extend(sorted((pair for pair in pivot.columns if pair not in PAIR_ORDER)))
    pivot = pivot[ordered]
    changes = pivot.diff()
    correlation = changes.corr(method='pearson', min_periods=min_periods)
    return (correlation, changes)

def build_monthly_role_series(frame: pd.DataFrame) -> pd.DataFrame:
    data = prepare_frame(frame)
    atm = data[data['is_observed'] & (data['tenor'] == '3M') & (data['quote_type'] == 'ATM') & data['last'].notna()].copy()
    atm['month'] = atm['date'].dt.to_period('M').dt.to_timestamp()
    return atm.groupby(['month', 'role'], observed=True).agg(atm_median=('last', 'median'), spread_median=('spread', 'median'), observations=('last', 'size')).reset_index().sort_values(['month', 'role'])

def correlation_group_summary(correlation: pd.DataFrame) -> dict[str, float]:

    def mean_pairs(left: list[str], right: list[str], same_group: bool) -> float:
        values = []
        for i, first in enumerate(left):
            for j, second in enumerate(right):
                if first not in correlation.index or second not in correlation.columns:
                    continue
                if first == second:
                    continue
                if same_group and j <= i:
                    continue
                value = correlation.loc[first, second]
                if pd.notna(value):
                    values.append(float(value))
        return float(np.mean(values))
    return {'mean_source_source': mean_pairs(SOURCE_PAIRS, SOURCE_PAIRS, True), 'mean_target_target': mean_pairs(TARGET_PAIRS, TARGET_PAIRS, True), 'mean_source_target': mean_pairs(SOURCE_PAIRS, TARGET_PAIRS, False)}

def _write_markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    header = '| ' + ' | '.join(columns) + ' |'
    rule = '|' + '|'.join(['---'] * len(columns)) + '|'
    rows = ['| ' + ' | '.join((str(value) for value in row)) + ' |' for row in display.itertuples(index=False, name=None)]
    return '\n'.join([header, rule, *rows])

def plot_correlation_heatmap(correlation: pd.DataFrame) -> list[Path]:
    sns.set_theme(style='white', font_scale=0.9)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation, cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, annot=True, fmt='.2f', annot_kws={'size': 7}, linewidths=0.4, linecolor='white', cbar_kws={'label': 'Pearson correlation'}, ax=ax)
    boundary = len(SOURCE_PAIRS)
    ax.axhline(boundary, color='#111827', linewidth=2)
    ax.axvline(boundary, color='#111827', linewidth=2)
    ax.set_title('Correlation of Daily Changes in 3M ATM Volatility', fontsize=16, weight='bold', pad=34)
    ax.text(0.27, 1.015, 'Source pairs (7)', transform=ax.transAxes, ha='center', va='bottom', color='#2563EB', weight='bold')
    ax.text(0.77, 1.015, 'Target pairs (6)', transform=ax.transAxes, ha='center', va='bottom', color='#D97706', weight='bold')
    ax.set_xlabel('Currency pair (source pairs first, target pairs second)')
    ax.set_ylabel('Currency pair')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    paths = [OUTPUT_DIR / 'v2_data_3m_atm_change_correlation_heatmap.png', OUTPUT_DIR / 'v2_data_3m_atm_change_correlation_heatmap.svg']
    fig.savefig(paths[0], dpi=220, bbox_inches='tight')
    fig.savefig(paths[1], bbox_inches='tight')
    plt.close(fig)
    return paths

def plot_monthly_context(monthly: pd.DataFrame) -> list[Path]:
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for role in ROLE_ORDER:
        group = monthly[monthly['role'] == role]
        label = role.capitalize() + ' group median'
        axes[0].plot(group['month'], group['atm_median'], label=label, color=ROLE_COLORS[role], linewidth=1.8)
        axes[1].plot(group['month'], group['spread_median'], color=ROLE_COLORS[role], linewidth=1.8)
    for axis in axes:
        axis.axvline(VALIDATION_START, color='#64748B', linestyle='--', linewidth=1.2)
        axis.axvline(TEST_START, color='#111827', linestyle='--', linewidth=1.2)
    axes[0].legend(loc='upper left', frameon=True)
    fig.suptitle('Monthly 3M ATM Volatility and Quoted Spread by Pair Group', fontsize=16, weight='bold', y=0.995)
    fig.text(0.5, 0.96, 'Monthly medians across seven source pairs and six target pairs; dashed lines mark split boundaries.', ha='center', color='#475569')
    axes[0].set_ylabel('Median 3M ATM (volatility points)')
    axes[1].set_ylabel('Median 3M ATM spread')
    axes[1].set_xlabel('Month')
    axes[0].text(VALIDATION_START, 0.97, ' Validation start', transform=axes[0].get_xaxis_transform(), va='top', color='#64748B')
    axes[0].text(TEST_START, 0.9, ' Test start', transform=axes[0].get_xaxis_transform(), va='top', color='#111827')
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    paths = [OUTPUT_DIR / 'v2_data_monthly_3m_atm_context.png', OUTPUT_DIR / 'v2_data_monthly_3m_atm_context.svg']
    fig.savefig(paths[0], dpi=220, bbox_inches='tight')
    fig.savefig(paths[1], bbox_inches='tight')
    plt.close(fig)
    return paths

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    usecols = ['date', 'pair', 'role', 'tenor', 'quote_type', 'last', 'spread', 'is_observed', 'midpoint_flag']
    raw = pd.read_csv(INPUT_PATH, usecols=usecols, low_memory=False)
    data = prepare_frame(raw)
    if sorted(data['pair'].unique().tolist()) != sorted(PAIR_ORDER):
        raise RuntimeError('Unexpected V2 pair set')
    if sorted(data['role'].unique().tolist()) != ROLE_ORDER:
        raise RuntimeError('Unexpected source/target role set')
    if sorted(data['tenor'].unique().tolist()) != sorted(['1W', '1M', '3M', '6M', '1Y']):
        raise RuntimeError('Unexpected tenor set')
    if sorted(data['quote_type'].unique().tolist()) != sorted(QUOTE_ORDER):
        raise RuntimeError('Unexpected quote-type set')
    pair_summary = build_pair_summary(data)
    quote_summary = build_role_quote_summary(data)
    correlation, changes = build_atm_change_correlation(data)
    pairwise_counts = changes.notna().astype(int).T.dot(changes.notna().astype(int))
    monthly = build_monthly_role_series(data)
    correlation_summary = correlation_group_summary(correlation)
    pair_csv = OUTPUT_DIR / 'v2_data_pair_coverage_summary.csv'
    quote_csv = OUTPUT_DIR / 'v2_data_descriptive_statistics.csv'
    correlation_csv = OUTPUT_DIR / 'v2_data_3m_atm_change_correlations.csv'
    correlation_count_csv = OUTPUT_DIR / 'v2_data_3m_atm_change_pairwise_counts.csv'
    monthly_csv = OUTPUT_DIR / 'v2_data_monthly_3m_atm_context.csv'
    pair_summary.to_csv(pair_csv, index=False)
    quote_summary.to_csv(quote_csv, index=False)
    correlation.to_csv(correlation_csv)
    pairwise_counts.to_csv(correlation_count_csv)
    monthly.to_csv(monthly_csv, index=False)
    figures = [*plot_correlation_heatmap(correlation), *plot_monthly_context(monthly)]
    missing_total = int(pair_summary['missing_last_count'].sum())
    usdcnh_missing = int(pair_summary.loc[pair_summary['pair'] == 'USDCNH', 'missing_last_count'].iloc[0])
    usdcnh_share = usdcnh_missing / missing_total
    role_spreads = data.groupby('role', observed=True)['spread'].median().reindex(ROLE_ORDER).astype(float).to_dict()
    report_pair = pair_summary.copy()
    for column in ('missing_last_rate', 'midpoint_flag_rate'):
        report_pair[column] = report_pair[column].map(lambda value: f'{value:.3%}')
    report_pair['complete_matrix_rate'] = report_pair['complete_matrix_rate'].map(lambda value: f'{value:.2%}')
    for column in ('median_spread', 'p90_spread'):
        report_pair[column] = report_pair[column].map(lambda value: f'{value:.4f}')
    report_quote = quote_summary.copy()
    report_quote['role'] = report_quote['role'].astype(str)
    report_quote['quote_type'] = report_quote['quote_type'].astype(str)
    report_quote['missing_rate'] = report_quote['missing_rate'].map(lambda value: f'{value:.2%}')
    for column in ('mean', 'std', 'median', 'p10', 'p90', 'median_spread', 'p90_spread'):
        report_quote[column] = report_quote[column].map(lambda value: f'{value:.4f}')
    report = RESULTS_DIR / 'v2_data_evidence_pack.md'
    report.write_text('\n'.join(['# V2 Dissertation Data Evidence Pack', '', '## Purpose and confidentiality', '', 'This report contains aggregated structural and descriptive statistics only. It does not reproduce quote-level Bloomberg observations. Raw and interim licensed data remain local and must not be redistributed.', '', '## Design', '', '- Common modelling window: 2010-01-01 to 2026-06-15.', '- Pair groups: seven source pairs and six target pairs.', '- Descriptive statistics: aggregated by pair and by role/quote type.', '- Correlation: pairwise Pearson correlation of daily changes in observed 3M ATM volatility, using pairwise-complete dates.', '- Time series: monthly median 3M ATM level and spread, aggregated across source or target pairs.', '', '## Aggregate findings', '', f'- Common-window records: {len(data):,}.', f"- Common-window pair-date matrices: {data[['pair', 'date']].drop_duplicates().shape[0]:,}.", f'- Missing last values in the common window: {missing_total:,}.', f'- USDCNH share of common-window missing last values: {usdcnh_share:.1%}.', f"- Median spread across source-group cells: {role_spreads['source']:.4f}.", f"- Median spread across target-group cells: {role_spreads['target']:.4f}.", f"- Mean source-source correlation of daily 3M ATM changes: {correlation_summary['mean_source_source']:.3f}.", f"- Mean target-target correlation of daily 3M ATM changes: {correlation_summary['mean_target_target']:.3f}.", f"- Mean source-target correlation of daily 3M ATM changes: {correlation_summary['mean_source_target']:.3f}.", f'- Minimum pairwise daily-change observations used in the correlation matrix: {int(pairwise_counts.where(~np.eye(len(pairwise_counts), dtype=bool)).min().min()):,}.', '', 'The source/target labels are design groups. Aggregate spreads and missingness describe this sample but do not establish that every target observation is less liquid than every source observation.', '', '## Table 3.1 — Pair coverage and data quality', '', _write_markdown_table(report_pair, ['pair', 'role', 'date_start', 'date_end', 'daily_matrices', 'missing_last_count', 'missing_last_rate', 'midpoint_flag_count', 'midpoint_flag_rate', 'complete_matrix_rate', 'median_spread', 'p90_spread']), '', '## Table 3.2 — Descriptive statistics by role and quote type', '', _write_markdown_table(report_quote, ['role', 'quote_type', 'observed_count', 'missing_rate', 'mean', 'std', 'median', 'p10', 'p90', 'median_spread', 'p90_spread']), '', '## Generated artifacts', '', f'- `{pair_csv.relative_to(ROOT)}`', f'- `{quote_csv.relative_to(ROOT)}`', f'- `{correlation_csv.relative_to(ROOT)}`', f'- `{correlation_count_csv.relative_to(ROOT)}`', f'- `{monthly_csv.relative_to(ROOT)}`', *[f'- `{path.relative_to(ROOT)}`' for path in figures], '', '## Interpretation boundaries', '', '- Aggregate descriptive statistics do not reveal individual quote observations.', '- Correlation is descriptive and does not identify a causal transfer mechanism.', '- Correlation of daily changes is not the same as correlation of levels or full surfaces.', '- Monthly source/target medians can hide pair-level heterogeneity.', '- Full-sample descriptive figures are reporting artifacts only; they were not used to select or tune models.', '- Missingness may be informative, particularly for USDCNH, and should remain visible as a limitation.']), encoding='utf-8')
    metadata = {'common_start': COMMON_START.date().isoformat(), 'common_end': data['date'].max().date().isoformat(), 'records': int(len(data)), 'pair_date_matrices': int(data[['pair', 'date']].drop_duplicates().shape[0]), 'pairs': PAIR_ORDER, 'source_pairs': SOURCE_PAIRS, 'target_pairs': TARGET_PAIRS, 'missing_last_values': missing_total, 'usdcnh_missing_share': usdcnh_share, 'role_median_spreads': role_spreads, 'correlation_summary': correlation_summary, 'correlation_min_pairwise_observations': int(pairwise_counts.where(~np.eye(len(pairwise_counts), dtype=bool)).min().min()), 'artifacts': [str(path.relative_to(ROOT)) for path in [pair_csv, quote_csv, correlation_csv, correlation_count_csv, monthly_csv, *figures, report]]}
    metadata_path = RESULTS_DIR / 'v2_data_evidence_pack.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'REPORT={report}')
    print(f'METADATA={metadata_path}')
    print(f'PAIR_ROWS={len(pair_summary)}')
    print(f'DESCRIPTIVE_ROWS={len(quote_summary)}')
    print(f'CORRELATION_SHAPE={correlation.shape}')
    print(f'MONTHLY_ROWS={len(monthly)}')
    print(f'USDCNH_MISSING_SHARE={usdcnh_share:.6f}')
    print(f"SOURCE_MEDIAN_SPREAD={role_spreads['source']:.6f}")
    print(f"TARGET_MEDIAN_SPREAD={role_spreads['target']:.6f}")
    print(f"MEAN_SOURCE_TARGET_CORRELATION={correlation_summary['mean_source_target']:.6f}")
    for path in figures:
        print(f'FIGURE={path}')
if __name__ == '__main__':
    main()
