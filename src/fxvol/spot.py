from __future__ import annotations
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from .bql import _number, _read_nonempty_rows

@dataclass(frozen=True)
class SpotRecord:
    date: date
    pair: str
    last: float | None
    bid: float | None
    ask: float | None

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

def read_spot_bql_ods(path: str | Path, weekdays_only: bool=True) -> list[SpotRecord]:
    rows = _read_nonempty_rows(Path(path))
    if len(rows) < 3 or rows[1][0] != 'DATES':
        raise ValueError(f'Unexpected BQL workbook structure in {path}')
    securities = rows[0]
    fields = rows[1]
    column_map: dict[str, dict[str, int]] = {}
    for index in range(1, min(len(securities), len(fields))):
        security = securities[index]
        if not security.endswith(' Curncy') or fields[index] not in ('#last', '#bid', '#ask'):
            continue
        pair = security.removesuffix(' Curncy')
        if len(pair) == 6 and pair.isalpha():
            column_map.setdefault(pair, {})[fields[index]] = index
    records: list[SpotRecord] = []
    for row in rows[2:]:
        try:
            observation_date = datetime.fromisoformat(row[0]).date()
        except (ValueError, IndexError):
            continue
        if weekdays_only and observation_date.weekday() >= 5:
            continue
        for pair, indexes in column_map.items():
            values = {field: _number(row[index]) if index < len(row) else None for field, index in indexes.items()}
            records.append(SpotRecord(date=observation_date, pair=pair, last=values.get('#last'), bid=values.get('#bid'), ask=values.get('#ask')))
    return records

def add_log_returns(records: list[SpotRecord]) -> list[dict[str, object]]:
    previous: dict[str, float] = {}
    output: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: (item.pair, item.date)):
        log_return = None
        prior = previous.get(record.pair)
        if record.last is not None and record.last > 0:
            if prior is not None and prior > 0:
                log_return = math.log(record.last / prior)
            previous[record.pair] = record.last
        output.append({'date': record.date.isoformat(), 'pair': record.pair, 'last': record.last, 'bid': record.bid, 'ask': record.ask, 'spread': record.spread, 'log_return': log_return, 'absolute_log_return': abs(log_return) if log_return is not None else None})
    return output

def write_spot_outputs(input_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_spot_bql_ods(input_path)
    enriched = add_log_returns(records)
    with (output_dir / 'spot_fx_long.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
        writer.writeheader()
        writer.writerows(enriched)
    summary = {'records': len(records), 'pairs': sorted({record.pair for record in records}), 'records_by_pair': dict(Counter((record.pair for record in records))), 'missing_last_by_pair': dict(Counter((record.pair for record in records if record.last is None))), 'missing_bid_by_pair': dict(Counter((record.pair for record in records if record.bid is None))), 'missing_ask_by_pair': dict(Counter((record.pair for record in records if record.ask is None)))}
    (output_dir / 'spot_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary

def attach_spot_features(matrices: list[dict[str, object]], enriched_spot: list[dict[str, object]]) -> list[dict[str, object]]:
    spot_by_key = {(str(record['pair']), str(record['date'])): record for record in enriched_spot}
    output = []
    for matrix in matrices:
        enriched_matrix = dict(matrix)
        spot = spot_by_key.get((str(matrix['pair']), str(matrix['date'])))
        enriched_matrix['spot'] = {'last': spot['last'], 'bid': spot['bid'], 'ask': spot['ask'], 'spread': spot['spread'], 'log_return': spot['log_return'], 'absolute_log_return': spot['absolute_log_return']} if spot else None
        output.append(enriched_matrix)
    return output
