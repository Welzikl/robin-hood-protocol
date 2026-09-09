from __future__ import annotations
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from fxvol.bql import _date, _normalise_field, _number, read_nonempty_bql_rows
from fxvol.constants import PAIR_ROLES
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw' / 'v2_bloomberg_2010_2026'
FORWARD_DIR = RAW_DIR / 'forwards'
OUTPUT_DIR = ROOT / 'data' / 'interim' / 'v2'
SPOT_PATTERN = re.compile('^(?P<pair>[A-Z]{6}) BGN Curncy$')
FORWARD_PATTERN = re.compile('^(?P<pair>[A-Z]{6})(?P<tenor>1W|1M|3M|6M|12M) BGN Curncy$')
FORWARD_TENORS = ('1W', '1M', '3M', '6M', '12M')
PAIR_ORDER = list(PAIR_ROLES)
SPOT_FILE = RAW_DIR / 'Spot_fx_13pairs_bloomberg_2010_2026.ods'
FORWARD_FILES = [FORWARD_DIR / 'EURUSDForwards.xlsx', FORWARD_DIR / 'GBPUSDFORWARDS.xlsx', FORWARD_DIR / 'FX_Forwards_USDJPY_2010_2026.xlsx', FORWARD_DIR / 'FX_Forwards_AUDUSD_2010_2026_FIXED.ods', FORWARD_DIR / 'FX_Forwards_USDCHF_2010_2026.xlsx', FORWARD_DIR / 'FX_Forwards_USDCAD_2010_2026.xlsx', FORWARD_DIR / 'FX_Forwards_NZDUSD_2010_2026.xlsx', FORWARD_DIR / 'FX_Forwards_USDZAR_2010_2026.xlsx', FORWARD_DIR / 'FX_Forwards_USDTRY_2010_2026.xlsx', FORWARD_DIR / 'FX_Forwards_USDMXN_2010_2026.xlsx', FORWARD_DIR / 'FX_Forwards_USDCNH_2010_2026.xlsx', FORWARD_DIR / 'FX_Forwards_USDPLN_2010_2026.xlsx', FORWARD_DIR / 'FX_Forwards_USDHUF_2010_2026.xlsx']

@dataclass(frozen=True)
class MarketRecord:
    date: date
    pair: str
    tenor: str | None
    last: float | None
    bid: float | None
    ask: float | None

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

def _read_market_records(path: Path, pattern: re.Pattern[str]) -> list[MarketRecord]:
    rows = read_nonempty_bql_rows(path)
    if len(rows) < 3 or rows[1][0] != 'DATES':
        raise ValueError(f'Unexpected BQL workbook structure in {path}')
    securities = rows[0]
    fields = rows[1]
    column_map: dict[tuple[str, str | None], dict[str, int]] = {}
    for index in range(1, min(len(securities), len(fields))):
        match = pattern.match(securities[index])
        field = _normalise_field(fields[index])
        if not match or field is None:
            continue
        pair = match.group('pair')
        tenor = match.groupdict().get('tenor')
        if pair in PAIR_ORDER:
            column_map.setdefault((pair, tenor), {})[field] = index
    records = []
    for row in rows[2:]:
        try:
            observation_date = _date(row[0])
        except (ValueError, IndexError):
            continue
        if observation_date.weekday() >= 5:
            continue
        for (pair, tenor), indexes in column_map.items():
            values = {field: _number(row[index]) if index < len(row) else None for field, index in indexes.items()}
            records.append(MarketRecord(date=observation_date, pair=pair, tenor=tenor, last=values.get('last'), bid=values.get('bid'), ask=values.get('ask')))
    return records

def read_spot_records() -> list[MarketRecord]:
    return _read_market_records(SPOT_FILE, SPOT_PATTERN)

def read_forward_records() -> list[MarketRecord]:
    records = []
    missing = [str(path) for path in FORWARD_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing forward files:\n' + '\n'.join(missing))
    for path in FORWARD_FILES:
        records.extend(_read_market_records(path, FORWARD_PATTERN))
    return records

def write_long_csv(path: Path, records: list[MarketRecord]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['date', 'pair', 'tenor', 'last', 'bid', 'ask', 'spread'])
        for record in records:
            writer.writerow([record.date.isoformat(), record.pair, record.tenor or '', record.last, record.bid, record.ask, record.spread])

def _safe_relative_spread(record: MarketRecord | None) -> float | None:
    if not record or record.spread is None or record.last is None:
        return None
    if not math.isfinite(record.last) or abs(record.last) <= 1e-12:
        return None
    return record.spread / abs(record.last)

def build_support_features(spot_records: list[MarketRecord], forward_records: list[MarketRecord]) -> list[dict[str, object]]:
    spot_by_key = {(record.pair, record.date.isoformat()): record for record in spot_records}
    forward_by_key = {(record.pair, record.date.isoformat(), str(record.tenor)): record for record in forward_records}
    dates_by_pair = defaultdict(set)
    for record in spot_records:
        dates_by_pair[record.pair].add(record.date.isoformat())
    for record in forward_records:
        dates_by_pair[record.pair].add(record.date.isoformat())
    feature_names = ['spot_last', 'spot_log_return_1d', 'spot_abs_log_return_1d', 'spot_relative_spread']
    for tenor in FORWARD_TENORS:
        feature_names.extend([f'forward_{tenor}_last', f'forward_{tenor}_relative_spread', f'forward_{tenor}_minus_spot'])
    feature_names.extend(['forward_12m_minus_1w', 'forward_6m_minus_1m'])
    rows = []
    for pair in PAIR_ORDER:
        previous_spot_last = None
        for date_label in sorted(dates_by_pair[pair]):
            spot = spot_by_key.get((pair, date_label))
            spot_last = spot.last if spot else None
            log_return = None
            if spot_last is not None and previous_spot_last is not None and (spot_last > 0) and (previous_spot_last > 0):
                log_return = math.log(spot_last / previous_spot_last)
            if spot_last is not None and spot_last > 0:
                previous_spot_last = spot_last
            values: dict[str, float | None] = {'spot_last': spot_last, 'spot_log_return_1d': log_return, 'spot_abs_log_return_1d': abs(log_return) if log_return is not None else None, 'spot_relative_spread': _safe_relative_spread(spot)}
            forward_last_by_tenor = {}
            for tenor in FORWARD_TENORS:
                forward = forward_by_key.get((pair, date_label, tenor))
                forward_last = forward.last if forward else None
                forward_last_by_tenor[tenor] = forward_last
                values[f'forward_{tenor}_last'] = forward_last
                values[f'forward_{tenor}_relative_spread'] = _safe_relative_spread(forward)
                values[f'forward_{tenor}_minus_spot'] = forward_last - spot_last if forward_last is not None and spot_last is not None else None
            values['forward_12m_minus_1w'] = forward_last_by_tenor['12M'] - forward_last_by_tenor['1W'] if forward_last_by_tenor['12M'] is not None and forward_last_by_tenor['1W'] is not None else None
            values['forward_6m_minus_1m'] = forward_last_by_tenor['6M'] - forward_last_by_tenor['1M'] if forward_last_by_tenor['6M'] is not None and forward_last_by_tenor['1M'] is not None else None
            rows.append({'date': date_label, 'pair': pair, 'feature_names': feature_names, 'features': [values[name] for name in feature_names], 'feature_mask': [values[name] is not None for name in feature_names]})
    return rows

def write_support_outputs() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spot_records = read_spot_records()
    forward_records = read_forward_records()
    write_long_csv(OUTPUT_DIR / 'spot_fx_support_long.csv', spot_records)
    write_long_csv(OUTPUT_DIR / 'fx_forward_support_long.csv', forward_records)
    support_rows = build_support_features(spot_records, forward_records)
    support_path = OUTPUT_DIR / 'market_support_features.jsonl'
    with support_path.open('w', encoding='utf-8') as handle:
        for row in support_rows:
            handle.write(json.dumps(row, separators=(',', ':')) + '\n')
    feature_names = support_rows[0]['feature_names'] if support_rows else []
    summary = {'spot_records': len(spot_records), 'forward_records': len(forward_records), 'support_rows': len(support_rows), 'feature_count': len(feature_names), 'feature_names': feature_names, 'spot_pairs': sorted({record.pair for record in spot_records}), 'forward_pairs': sorted({record.pair for record in forward_records}), 'missing_support_features_by_pair': dict(Counter((row['pair'] for row in support_rows if not all((bool(value) for value in row['feature_mask'])))))}
    summary_path = OUTPUT_DIR / 'market_support_feature_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary

def main() -> None:
    summary = write_support_outputs()
    for key, value in summary.items():
        if key != 'feature_names':
            print(f'{key}: {value}')
if __name__ == '__main__':
    main()
