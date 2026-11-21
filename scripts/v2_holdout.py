"""Single, pre-registered evaluation of the sealed holdout (2023-2025).  See PREREGISTRATION.md.

Runs, in this order and nothing else:
  1. System A / full system / System B at h = 21 and h = 5 (scripts/v2_metasystem.py --unseal)
  2. all base models and factors at h = 21 and h = 5 (scripts/v2_evaluate.py --unseal)
  3. H1 (one-sided Newey-West test of the System A IC at h = 21) and S5 (regime dependence of 5-day
     reversal on the holdout)
Writes results_v2/summary/holdout_report.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable
SUM = ROOT / "results_v2" / "summary"


def run(args):
    print(">>", " ".join(args), flush=True)
    out = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    print("\n".join(l for l in out.stdout.splitlines() if "Warning" not in l)[-6000:], flush=True)
    if out.returncode != 0:
        print(out.stderr[-3000:], flush=True)


for h in (21, 5):
    run(["scripts/v2_metasystem.py", "--panel", "sp500", "--horizon", str(h), "--unseal"])
models21 = "XGBoost,Ridge,MLP,XGBoost-IC,Transformer_s5,AFT_s5,AFT-R_s5"
run(["scripts/v2_evaluate.py", "--panel", "sp500", "--horizon", "21", "--unseal", "--models", models21,
     "--ref", "Transformer_s5"])
run(["scripts/v2_evaluate.py", "--panel", "sp500", "--horizon", "5", "--unseal", "--models",
     "XGBoost,Ridge,MLP,XGBoost-IC", "--ref", "Ridge"])

# ---------------- H1 and S5
from aft.stats import hac_mean_test  # noqa: E402
from aft2.evaluate import daily_ic  # noqa: E402
from aft2.meta import MARKET_STATE, cs_rank_long, load_ensemble  # noqa: E402
from aft2.protocol import HOLDOUT_FIRST_YEAR, HOLDOUT_LAST_YEAR, load_panel, wf_dir  # noqa: E402

report = {}
for h in (21, 5):
    P = load_panel("sp500", h)[0]
    yrs = P.dates.astype("datetime64[Y]").astype(int) + 1970
    wf = wf_dir("sp500", h)
    base = None
    for m in ("XGBoost", "Ridge", "MLP"):
        d = load_ensemble(m, wf)
        d = d[(yrs[d.t] >= HOLDOUT_FIRST_YEAR) & (yrs[d.t] <= HOLDOUT_LAST_YEAR)]
        d = d.assign(**{"r_" + m: cs_rank_long(d, "y_hat")})[["t", "i", "y", "y_abs", "r_" + m]]
        base = d if base is None else base.merge(d[["t", "i", "r_" + m]], on=["t", "i"])
    for name, feat in (("rev5", "ret_5"), ("rev21", "ret_21")):
        base["tmp"] = -P.raw(feat, base.t.to_numpy(), base.i.to_numpy())
        base["r_" + name] = cs_rank_long(base, "tmp")
    rc = [c for c in base if c.startswith("r_")]
    sig = base[["t", "i", "y", "y_abs"]].assign(y_hat=base[rc].mean(axis=1))
    ic = daily_ic(sig)
    t = hac_mean_test(ic.to_numpy(), lag=h - 1)
    report[f"SystemA_h{h}"] = {"IC": t["mean"], "t": t["stat"], "p_one_sided": float(t["p"] / 2 if t["stat"] > 0 else 1 - t["p"] / 2),
                               "n_dates": int(len(ic)), "IC_by_year": {int(y): float(v) for y, v in
                               pd.Series(ic.to_numpy(), index=yrs[ic.index]).groupby(level=0).mean().items()}}
    if h == 5:   # S5: regime dependence of 5-day reversal on the holdout
        rev = base[["t", "i", "y", "y_abs"]].assign(y_hat=base["r_rev5"])
        ric = daily_ic(rev)
        st = pd.DataFrame({f: P.raw(f, ric.index.to_numpy(), np.zeros(len(ric), int)) for f in MARKET_STATE},
                          index=ric.index)
        s5 = {}
        for f in MARKET_STATE:
            q = pd.qcut(st[f], 3, labels=False)
            m = ric.groupby(q).mean()
            x = ((st[f] - st[f].mean()) / st[f].std()).to_numpy()
            y = ric.to_numpy()
            b = np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean())
            e = y - y.mean() - b * (x - x.mean())
            p = hac_mean_test((x - x.mean()) * e / np.mean((x - x.mean()) ** 2) + b, lag=4)["p"]
            s5[f] = {"tercile_IC": [float(v) for v in m.to_numpy()], "slope": float(b), "p": float(p)}
        report["S5_reversal_regimes"] = {"mean_IC": float(ric.mean()), "by_state": s5}
(SUM / "holdout_report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
