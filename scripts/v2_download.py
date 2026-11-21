"""Download the v2 universe and write a data-quality report (data/raw2/quality_report.csv)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft2.data import RAW, download, quality_report  # noqa: E402
from aft2.universe import MARKET, SECTOR_ETFS, SECTOR_OF, STOCKS, VIX  # noqa: E402

spy = download(MARKET)
calendar = pd.DatetimeIndex(spy.Date)
rows = []
for t in [MARKET, VIX] + SECTOR_ETFS + STOCKS:
    try:
        df = download(t)
        rep = quality_report(df, calendar)
        rows.append({"ticker": t, "sector": SECTOR_OF.get(t, "market"), **rep})
        print(f"{t:6s} {rep['first_date']}..{rep['last_date']} rows {rep['rows']:5d} missing {rep['missing_days']:3d} "
              f"maxjump {rep['max_abs_logret']:.3f} ({rep['max_abs_logret_date']}) badOHLC {rep['bad_ohlc_rows']}", flush=True)
    except Exception as e:
        rows.append({"ticker": t, "sector": SECTOR_OF.get(t, "market"), "error": str(e)})
        print(f"{t:6s} ERROR {e}", flush=True)
pd.DataFrame(rows).to_csv(RAW / "quality_report.csv", index=False)
print("done")
