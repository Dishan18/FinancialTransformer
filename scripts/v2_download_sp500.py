"""Download all current S&P 500 constituents (data/sp500_constituents_2026-11-21.csv) and write a
quality report (data/raw2/quality_report_sp500.csv)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft2.data import RAW, download, quality_report  # noqa: E402

cons = pd.read_csv(ROOT / "data" / "sp500_constituents_2026-11-21.csv")
spy = download("SPY")
cal = pd.DatetimeIndex(spy.Date)
rows = []
for _, r in cons.iterrows():
    y = r.Symbol.replace(".", "-")
    try:
        rep = quality_report(download(y), cal)
        rows.append({"ticker": y, "sector": r["GICS Sector"], "added": r["Date added"], **rep})
        print(f"{y:6s} {rep['first_date']} rows {rep['rows']} maxjump {rep['max_abs_logret']:.3f} ({rep['max_abs_logret_date']})", flush=True)
    except Exception as e:
        rows.append({"ticker": y, "sector": r["GICS Sector"], "added": r["Date added"], "error": str(e)[:120]})
        print(f"{y:6s} ERROR {str(e)[:120]}", flush=True)
pd.DataFrame(rows).to_csv(RAW / "quality_report_sp500.csv", index=False)
print("done", len(rows))
