"""Investment universe: large, liquid US stocks across the 11 GICS sectors, listed before 2013.

Selection rule (fixed before looking at any model result): well-known large-cap constituents of the
S&P 100 / S&P 500 with continuous Yahoo Finance history over the study period and no ticker change.
This is a current-constituent list, so the universe is subject to survivorship bias; we evaluate
mainly *relative* (cross-sectional) returns and state the bias as a limitation.
"""
from __future__ import annotations

SECTORS: dict[str, list[str]] = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "ORCL", "CSCO", "INTC", "IBM", "ADBE", "CRM",
                               "QCOM", "TXN", "AVGO", "AMD", "MU", "AMAT", "ACN", "INTU"],
    "Communication Services": ["GOOG", "META", "NFLX", "DIS", "CMCSA", "VZ", "T"],
    "Consumer Discretionary": ["AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "BKNG", "TJX", "F"],
    "Consumer Staples": ["WMT", "PG", "KO", "PEP", "COST", "MO", "PM", "CL", "MDLZ", "KMB"],
    "Health Care": ["JNJ", "UNH", "PFE", "MRK", "ABT", "TMO", "AMGN", "BMY", "LLY", "MDT", "GILD",
                    "CVS", "DHR", "ABBV"],
    # BK was in the initial list but Yahoo Finance returned no data for it (dropped before any modelling).
    "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SCHW", "USB", "MET", "AIG"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "OXY", "EOG"],
    "Industrials": ["BA", "CAT", "HON", "UNP", "LMT", "UPS", "DE", "FDX", "EMR", "GD"],
    "Materials": ["APD", "NEM", "SHW", "ECL", "FCX"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "Real Estate": ["AMT", "SPG", "PLD", "PSA"],
}
STOCKS: list[str] = [t for ts in SECTORS.values() for t in ts]
SECTOR_OF: dict[str, str] = {t: s for s, ts in SECTORS.items() for t in ts}
MARKET = "SPY"
VIX = "^VIX"
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
START, END = "2010-01-01", "2025-12-31"

# Data repairs found by the quality report (scripts/v2_download.py):
#   DHR: Yahoo's split/dividend adjustment around the Fortive spin-off is wrong -- the adjusted close
#        jumps from 41.94 (2016-07-01) to 67.61 (2016-07-05), an artificial +48% log return.  DHR is
#        therefore used only from 2016-07-05 onwards.
#   S&P 500 universe (scripts/v2_download_sp500.py + manual check of every |log return| > 0.40 and of
#   known spin-off dates): conservatively start these series at the corporate-action date, because
#   the adjusted close moves by 12-41 % on that day, which looks like an incomplete adjustment:
#   EXPE 2011-12-21 (TripAdvisor spin-off, -40.9 %), HPQ 2015-11-02 (HPE separation, -12.2 %),
#   DD 2019-06-03 (Corteva spin-off, -16.4 %), RTX 2020-04-03 (Carrier/Otis spin-offs; pre-2020 history
#   is UTX, -14.2 % on 2020-04-06), IFF 2021-02-01 (N&B combination, -14.7 %).
FIRST_VALID_DATE: dict[str, str] = {"DHR": "2016-07-05", "EXPE": "2011-12-21", "HPQ": "2015-11-02",
                                    "DD": "2019-06-03", "RTX": "2020-04-06", "IFF": "2021-02-01"}

assert len(STOCKS) == len(set(STOCKS)) == 100
