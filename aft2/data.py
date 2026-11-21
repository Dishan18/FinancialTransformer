"""Download, cache and quality-check daily OHLCV for the panel."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import END, START

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw2"


def download(ticker: str, start: str = START, end: str = END, retries: int = 3) -> pd.DataFrame:
    """Auto-adjusted (splits and dividends) daily OHLCV from Yahoo Finance, cached as CSV."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{ticker.replace('^', 'IDX_')}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["Date"])
    import yfinance as yf
    last = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False,
                             group_by="column", threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns.name = None
            if len(df) == 0:
                raise RuntimeError("empty")
            df.to_csv(path, index=False)
            return pd.read_csv(path, parse_dates=["Date"])
        except Exception as e:  # network hiccups / rate limits
            last = e
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"download failed for {ticker}: {last}")


def quality_report(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    """Simple data-quality checks against the market (SPY) trading calendar."""
    d = df.set_index("Date").sort_index()
    first = d.index[0]
    cal = calendar[calendar >= first]
    missing = cal.difference(d.index)
    lr = np.log(d["Close"]).diff()
    bad_ohlc = ((d["High"] < d[["Open", "Close"]].max(axis=1) - 1e-6) |
                (d["Low"] > d[["Open", "Close"]].min(axis=1) + 1e-6)).sum()
    big = lr.abs().nlargest(3)
    return {
        "first_date": first.date().isoformat(), "last_date": d.index[-1].date().isoformat(),
        "rows": int(len(d)), "missing_days": int(len(missing)),
        "nonpositive_price": int((d[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()),
        "zero_volume_days": int((d["Volume"] <= 0).sum()),
        "bad_ohlc_rows": int(bad_ohlc),
        "max_abs_logret": float(lr.abs().max()),
        "max_abs_logret_date": lr.abs().idxmax().date().isoformat(),
        "top3_jumps": "; ".join(f"{i.date()}:{v:+.3f}" for i, v in lr.loc[big.index].items()),
    }
