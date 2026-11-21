"""Panel with the ORIGINAL 95 v1 features (Notebook 01 replica, aft.data.build_features) for the same
100 stocks and dates, and the SAME targets as the v2 panel -- used to isolate the effect of the
feature redesign.  Output: data/panel_v1feat.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.data import ALL_FEATURES as V1_FEATURES, build_features  # noqa: E402
from aft2.data import download  # noqa: E402
from aft2.panel import Panel  # noqa: E402
from aft2.universe import FIRST_VALID_DATE  # noqa: E402

P = Panel.load()
cal = pd.DatetimeIndex(P.dates.astype("datetime64[ns]"))
X = np.full((len(cal), len(P.tickers), len(V1_FEATURES)), np.nan, dtype=np.float32)
for j, t in enumerate(P.tickers):
    raw = download(str(t))
    if str(t) in FIRST_VALID_DATE:
        raw = raw[raw.Date >= FIRST_VALID_DATE[str(t)]].reset_index(drop=True)
    f = build_features(raw).set_index("Date").reindex(cal)
    X[:, j, :] = f[V1_FEATURES].to_numpy(np.float32)
np.savez_compressed(ROOT / "data" / "panel_v1feat.npz", X=X, y_abs=P.y_abs, y_rel=P.y_rel,
                    dates=np.datetime_as_string(P.dates, unit="D"), tickers=P.tickers, features=np.array(V1_FEATURES))
print("v1-feature panel", X.shape, "finite share", float(np.isfinite(X).all(-1).mean()))
