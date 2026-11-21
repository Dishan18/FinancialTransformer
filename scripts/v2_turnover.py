"""Hypothesis H1 (development period): cost-aware signal smoothing raises net long-short Sharpe.

For each signal, the daily cross-sectional rank is smoothed per stock with a causal EMA of span s
(s = 1 means no smoothing).  Only information up to t is used.  Evaluated on OOS years 2016-2022.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft2.evaluate import evaluate  # noqa: E402
from aft2.meta import cs_rank_long, load_ensemble  # noqa: E402
from aft2.panel import Panel  # noqa: E402
from aft2.protocol import DEV_LAST_YEAR  # noqa: E402

P = Panel.load()
yrs = P.dates.astype("datetime64[Y]").astype(int) + 1970
signals = {}
for m in sys.argv[1:] or ["Ridge", "MLP", "XGBoost"]:
    d = load_ensemble(m)
    signals[m] = d[yrs[d.t] <= DEV_LAST_YEAR].copy()
base = next(iter(signals.values()))
k = P.features.index("ret_5")
signals["Factor: reversal 5d"] = base[["t", "i", "y", "y_abs"]].assign(y_hat=-P.X_raw[base.t, base.i, k])
rows = []
for name, d in signals.items():
    d = d.sort_values(["i", "t"]).copy()
    d["r"] = cs_rank_long(d, "y_hat")
    for span in (1, 5, 10, 21):
        s = d.groupby("i")["r"].transform(lambda x: x.ewm(span=span, adjust=False).mean()) if span > 1 else d.r
        res = evaluate(d.assign(y_hat=s), costs=(0.0, 5.0, 10.0))
        rows.append({"signal": name, "ema_span": span, "IC": res["IC"], "IC_t": res["IC_t"],
                     "SR@0": res["LS_sharpe@0"], "SR@5": res["LS_sharpe@5"], "SR@10": res["LS_sharpe@10"],
                     "turnover": res["LS_turnover@5"], "ann_ret@5": res["LS_ann_return@5"]})
        print(rows[-1], flush=True)
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results_v2" / "summary" / "dev_turnover_smoothing.csv", index=False)
print(out.round(4).to_string(index=False))
