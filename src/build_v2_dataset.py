from __future__ import annotations
from pathlib import Path
from fxvol.pipeline import write_standard_outputs
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw' / 'v2_bloomberg_2010_2026'
OUTPUT_DIR = ROOT / 'data' / 'interim' / 'v2'
PANELS = ['1-FullEURUSD_BQL_2010-2026.xlsx', '10-FullGBPUSD_BQL_2006-2026.xlsx', '6-FullUSDJPY_BQL_2006-2026.xlsx', '18-FullAUDUSD_BQL_2010-2026.ods.xlsx', '13-FullUSDCHF_BQL_2010-2026_ATM_25D.ods.xlsx', '4-FullUSDCHF_BQL_2010-2026_10D.ods.xlsx', '8-USDZAR.xlsx', '16-FullUSDTRY_BQL_2010-2026.ods.xlsx', '12-FullUSDCAD_BQL_2010-2026.ods.xlsx', '3-FullNZDUSD_BQL_2010-2026_ATM_25D.ods.xlsx', '11-FullNZDUSD_BQL_2010-2026_10D.ods.xlsx', '15-FullUSDMXN_BQL_2010-2026.ods.xlsx', '5-FullUSDCNH_BQL_2010-2026.ods.xlsx', '7-FullUSDPLN_BQL_2010-2026.ods.xlsx', '14-FullUSDHUF_BQL_2010-2026.ods.xlsx']

def main() -> None:
    input_paths = [RAW_DIR / panel for panel in PANELS]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing V2 panels:\n' + '\n'.join(missing))
    summary = write_standard_outputs(input_paths, OUTPUT_DIR)
    for key, value in summary.items():
        print(f'{key}: {value}')
if __name__ == '__main__':
    main()
