"""Build a point-in-time panel in the memory-lean format (stock features + market features stored
separately, targets for h = 5 and h = 21) and run the look-ahead test on it.

Usage:  python scripts/v2_build_panel_pit.py sp500|top100pit
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft2.data import download  # noqa: E402
from aft2.features import MARKET_FEATURES, STOCK_FEATURES, compute_panel_ext  # noqa: E402
from aft2.panel import _cs_rank  # noqa: E402
from aft2.protocol import PANELS  # noqa: E402
from aft2.universe import SECTOR_OF, STOCKS  # noqa: E402

which = sys.argv[1]
HORIZONS = (5, 21)
if which == "sp500":
    cons = pd.read_csv(ROOT / "data" / "sp500_constituents_2026-11-21.csv")
    q = pd.read_csv(ROOT / "data" / "raw2" / "quality_report_sp500.csv")
    excl = pd.read_csv(ROOT / "data" / "sp500_exclusions.csv") if (ROOT / "data" / "sp500_exclusions.csv").exists() else None
    ok = set(q.loc[q.get("error").isna() if "error" in q else slice(None), "ticker"])
    cons["ticker"] = cons.Symbol.str.replace(".", "-", regex=False)
    cons = cons[cons.ticker.isin(ok)]
    if excl is not None:
        cons = cons[~cons.ticker.isin(set(excl.ticker))]
    tickers = cons.ticker.tolist()
    sector_of = dict(zip(cons.ticker, cons["GICS Sector"]))
    eligible_from = dict(zip(cons.ticker, cons["Date added"]))
else:
    tickers, sector_of, eligible_from = list(STOCKS), dict(SECTOR_OF), None
stocks = {t: download(t) for t in tickers}
spy, vix = download("SPY"), download("^VIX")
kw = dict(sector_of=sector_of, eligible_from=eligible_from, horizons=HORIZONS, replicate_market=False, as_float32=True)

# ---------------- look-ahead test (3 random dates; on a random subset of <= 60 stocks to bound memory --
# the test checks the causality of the code path, which is identical for any universe)
rng = np.random.default_rng(1)
sub = sorted(rng.choice(tickers, size=min(60, len(tickers)), replace=False))
sub_stocks = {t: stocks[t] for t in sub}
ref = compute_panel_ext(sub_stocks, spy, vix, **kw)
for d in sorted(rng.choice(ref["M"].index[300:-30], size=3, replace=False)):
    o2 = compute_panel_ext(sub_stocks, spy, vix, end=str(pd.Timestamp(d).date()), **kw)
    for f in STOCK_FEATURES:
        a, b = ref["F"][f].loc[d].to_numpy(float), o2["F"][f].loc[d].to_numpy(float)
        if not (np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b), initial=0) <= 1e-6 * (1 + np.nanmax(np.abs(a), initial=0))):
            raise SystemExit(f"LOOK-AHEAD FAILURE {f} {d}")
    for f in MARKET_FEATURES:
        a, b = ref["M"][f].loc[d], o2["M"][f].loc[d]
        if not ((np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-9 * (1 + abs(a))):
            raise SystemExit(f"LOOK-AHEAD FAILURE {f} {d}")
    del o2
    gc.collect()
    print(f"look-ahead test passed at {pd.Timestamp(d).date()} ({len(sub)} stocks)", flush=True)
del ref
gc.collect()

out = compute_panel_ext(stocks, spy, vix, **kw)
dates = out["M"].index

T, N = len(dates), len(tickers)
X = np.empty((T, N, len(STOCK_FEATURES)), dtype=np.float32)
for k, f in enumerate(STOCK_FEATURES):
    X[:, :, k] = out["F"][f].to_numpy(np.float32)
    out["F"][f] = None
eligible = out["eligible"].to_numpy()
save = dict(X_stock=X, X_mkt=out["M"][MARKET_FEATURES].to_numpy(np.float32), mkt_weight=eligible.sum(1).astype(np.int32),
            eligible=eligible, dates=dates.strftime("%Y-%m-%d").to_numpy(), tickers=np.array(tickers),
            sectors=np.array([sector_of[t] for t in tickers]), stock_features=np.array(STOCK_FEATURES),
            market_features=np.array(MARKET_FEATURES))
for h, (ya, yr) in out["targets"].items():
    save[f"y_abs_{h}"] = ya.to_numpy(np.float32)
    save[f"y_rel_{h}"] = yr.to_numpy(np.float32)
fname, rname = PANELS[which]
np.savez_compressed(ROOT / "data" / fname, **save)
valid = np.isfinite(X).all(-1) & np.isfinite(save["y_abs_5"])
print(f"{which}: T={T}, N={N}, valid stock-days {int(valid.sum()):,}; universe size per date "
      f"min {eligible.sum(1)[300:].min()} median {int(np.median(eligible.sum(1)))} max {eligible.sum(1).max()}", flush=True)
R = np.stack([_cs_rank(X[:, :, k]) for k in range(X.shape[-1])], axis=-1)
np.save(ROOT / "data" / rname, R)
print("ranks saved", R.shape, flush=True)
