"""Download and cache daily auto-adjusted OHLCV for the multi-asset study (Yahoo Finance)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.data import build_features, download_ohlcv  # noqa: E402

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "JPM", "XOM", "JNJ", "WMT", "SPY"]

if __name__ == "__main__":
    for t in TICKERS:
        raw = download_ohlcv(t)
        feats = build_features(raw)
        print(f"{t:5s} raw {len(raw)} rows {raw.Date.iloc[0].date()}..{raw.Date.iloc[-1].date()} | "
              f"features {len(feats)} rows {feats.Date.iloc[0].date()}..{feats.Date.iloc[-1].date()}", flush=True)
