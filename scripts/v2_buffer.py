"""H5 (development period): a hysteresis (buffer) portfolio rule raises net long-short Sharpe.
Compares the plain quintile long-short with buffer rules (enter top/bottom q_in, keep while within
q_out) for several signals on the point-in-time S&P 500, h = 5, OOS 2016-2022."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft2.evaluate import long_short, long_short_buffer  # noqa: E402
from aft2.meta import cs_rank_long, load_ensemble  # noqa: E402
from aft2.protocol import DEV_LAST_YEAR, load_panel, wf_dir  # noqa: E402

panel, h = (sys.argv[1] if len(sys.argv) > 1 else "sp500"), int(sys.argv[2]) if len(sys.argv) > 2 else 5
P = load_panel(panel, h)[0]
yrs = P.dates.astype("datetime64[Y]").astype(int) + 1970
wf = wf_dir(panel, h)
sig = {}
for m in ("XGBoost", "Ridge", "MLP"):
    d = load_ensemble(m, wf)
    sig[m] = d[yrs[d.t] <= DEV_LAST_YEAR].copy()
base = sig["XGBoost"][["t", "i", "y", "y_abs"]]
for m in list(sig):
    sig[m] = base.merge(sig[m][["t", "i", "y_hat"]], on=["t", "i"])
sig["Reversal 5d"] = base.assign(y_hat=-P.raw("ret_5", base.t.to_numpy(), base.i.to_numpy()))
sig["Reversal 21d"] = base.assign(y_hat=-P.raw("ret_21", base.t.to_numpy(), base.i.to_numpy()))
ranks = {m: cs_rank_long(d, "y_hat") for m, d in sig.items()}
sig["EW (models + reversals)"] = base.assign(y_hat=np.mean([r.to_numpy() for r in ranks.values()], axis=0))
rows = []
for name, d in sig.items():
    for rule, fn in (("quintile", lambda dd, c: long_short(dd, cost_bps=c, h=h)),
                     ("buffer 20/40", lambda dd, c: long_short_buffer(dd, cost_bps=c, q_in=0.2, q_out=0.4, h=h)),
                     ("buffer 20/50", lambda dd, c: long_short_buffer(dd, cost_bps=c, q_in=0.2, q_out=0.5, h=h))):
        r = {"signal": name, "rule": rule}
        for c in (0.0, 5.0, 10.0):
            x = fn(d, c)
            r[f"SR@{int(c)}"] = x["sharpe"]
            if c == 5.0:
                r["turnover"], r["ann_ret@5"], r["mdd@5"] = x["turnover"], x["ann_return"], x["mdd"]
        rows.append(r)
        print(r, flush=True)
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results_v2" / "summary" / f"{panel}_h{h}_dev_buffer.csv", index=False)
print(out.round(3).to_string(index=False))
