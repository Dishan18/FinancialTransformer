"""Build the v2 panel (features + targets) and verify that no feature looks ahead.

Look-ahead test: for several random dates t, all inputs are truncated at t and the features are
recomputed; every feature value at t must equal the value computed on the full sample.
Output: data/panel_v2.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft2.data import download  # noqa: E402
from aft2.features import ALL_FEATURES, compute_panel  # noqa: E402
from aft2.universe import MARKET, STOCKS, VIX  # noqa: E402

stocks = {t: download(t) for t in STOCKS}
spy, vix = download(MARKET), download(VIX)
F, y_abs, y_rel = compute_panel(stocks, spy, vix)
dates = y_abs.index

# ---------------- look-ahead test
rng = np.random.default_rng(0)
cand = dates[300:-10]
test_dates = sorted(rng.choice(cand, size=6, replace=False))
worst = 0.0
for d in test_dates:
    Ft, _, _ = compute_panel(stocks, spy, vix, end=str(pd.Timestamp(d).date()))
    for f in ALL_FEATURES:
        a = F[f].loc[d].to_numpy(float)
        b = Ft[f].loc[d].to_numpy(float)
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        diff = np.nanmax(np.abs(a - b) / (1 + np.abs(a))) if np.isfinite(a).any() else 0.0
        if not same_nan or diff > 1e-9:
            raise SystemExit(f"LOOK-AHEAD FAILURE: feature {f} at {pd.Timestamp(d).date()} diff={diff} nan_eq={same_nan}")
        worst = max(worst, diff)
print(f"look-ahead test passed on {len(test_dates)} dates x {len(ALL_FEATURES)} features x {len(STOCKS)} stocks "
      f"(max rel. diff {worst:.2e}); dates: {[str(pd.Timestamp(d).date()) for d in test_dates]}")

# ---------------- save
X = np.stack([F[f].to_numpy(np.float32) for f in ALL_FEATURES], axis=-1)   # (T, N, K)
np.savez_compressed(ROOT / "data" / "panel_v2.npz", X=X, y_abs=y_abs.to_numpy(np.float32),
                    y_rel=y_rel.to_numpy(np.float32), dates=dates.strftime("%Y-%m-%d").to_numpy(),
                    tickers=np.array(STOCKS), features=np.array(ALL_FEATURES))
valid = np.isfinite(X).all(-1) & np.isfinite(y_abs.to_numpy())
print(f"panel: T={X.shape[0]} dates ({dates[0].date()}..{dates[-1].date()}), N={X.shape[1]} stocks, K={X.shape[2]} features")
print(f"valid stock-days: {int(valid.sum()):,}; first date with >=90 valid stocks: "
      f"{dates[np.argmax(valid.sum(1) >= 90)].date()}")
nan_share = pd.Series({f: float(np.isnan(F[f].to_numpy()).mean()) for f in ALL_FEATURES}).sort_values()
print("highest NaN shares:", nan_share.tail(5).round(3).to_dict())
