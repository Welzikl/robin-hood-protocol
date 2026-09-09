from __future__ import annotations
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from .bql import PanelRecord, read_bql_workbook
from .constants import PAIR_ROLES, QUOTE_TYPES, TENORS

def build_daily_matrices(records: list[PanelRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[tuple[str, str], PanelRecord]] = defaultdict(dict)
    for record in records:
        grouped[record.pair, record.date.isoformat()][record.tenor, record.quote_type] = record
    matrices: list[dict[str, object]] = []
    for (pair, date), entries in sorted(grouped.items()):
        values = []
        observed = []
        spreads = []
        for tenor in TENORS:
            value_row = []
            observed_row = []
            spread_row = []
            for quote_type in QUOTE_TYPES:
                record = entries.get((tenor, quote_type))
                value_row.append(record.last if record else None)
                observed_row.append(bool(record and record.is_observed))
                spread_row.append(record.spread if record else None)
            values.append(value_row)
            observed.append(observed_row)
            spreads.append(spread_row)
        matrices.append({'date': date, 'pair': pair, 'role': PAIR_ROLES[pair], 'values': values, 'observed_mask': observed, 'spreads': spreads})
    return matrices

def write_standard_outputs(input_paths: list[Path], output_dir: Path, midpoint_tolerance: float=0.01) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [record for path in input_paths for record in read_bql_workbook(path, weekdays_only=True)]
    long_path = output_dir / 'fx_volatility_long.csv'
    with long_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['date', 'pair', 'role', 'tenor', 'quote_type', 'last', 'bid', 'ask', 'spread', 'is_observed', 'midpoint_difference', 'midpoint_flag'])
        for record in records:
            difference = record.midpoint_difference
            writer.writerow([record.date.isoformat(), record.pair, PAIR_ROLES[record.pair], record.tenor, record.quote_type, record.last, record.bid, record.ask, record.spread, int(record.is_observed), difference, int(difference is not None and difference > midpoint_tolerance)])
    matrices = build_daily_matrices(records)
    matrix_path = output_dir / 'fx_volatility_daily_matrices.jsonl'
    with matrix_path.open('w', encoding='utf-8') as handle:
        for matrix in matrices:
            handle.write(json.dumps(matrix, separators=(',', ':')) + '\n')
    summary = {'input_panels': len(input_paths), 'long_records': len(records), 'daily_matrices': len(matrices), 'missing_last_values': sum((not record.is_observed for record in records)), 'midpoint_flags': sum((record.midpoint_difference is not None and record.midpoint_difference > midpoint_tolerance for record in records)), 'records_by_pair': dict(Counter((record.pair for record in records))), 'missing_last_by_pair': dict(Counter((record.pair for record in records if not record.is_observed))), 'midpoint_flags_by_pair': dict(Counter((record.pair for record in records if record.midpoint_difference is not None and record.midpoint_difference > midpoint_tolerance))), 'records_by_pair_quote_type': dict(Counter((f'{record.pair}|{record.quote_type}' for record in records)))}
    (output_dir / 'pipeline_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
