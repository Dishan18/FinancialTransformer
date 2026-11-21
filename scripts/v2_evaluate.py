"""Evaluate v2 walk-forward forecasts (development period only unless --unseal is given).

Usage:  python scripts/v2_evaluate.py [--unseal] [--models A,B,...] [--ref NAME]
Outputs results_v2/summary/{dev|holdout}_*.csv / .md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.stats import hac_mean_test, mean_ci  # noqa: E402
from aft2.evaluate import daily_ic, dm_vs, evaluate  # noqa: E402
from aft2.panel import Panel  # noqa: E402
from aft2.protocol import (DEV_FIRST_YEAR, DEV_LAST_YEAR, HOLDOUT_FIRST_YEAR, HOLDOUT_LAST_YEAR,  # noqa: E402
                           load_panel, wf_dir)

OUT = ROOT / "results_v2" / "summary"
OUT.mkdir(parents=True, exist_ok=True)
FACTORS = {"Factor: momentum 12-1": ("mom_12_1", +1), "Factor: reversal 5d": ("ret_5", -1),
           "Factor: reversal 21d": ("ret_21", -1), "Factor: low volatility": ("vol_63", -1),
           "Factor: MAX": ("max_ret_21", -1)}


def load_model(name: str, years: tuple[int, int], P: Panel, WF: Path) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    files = sorted((WF / name).glob("fold*_seed*.parquet"))
    per_seed = {}
    yrs = P.dates.astype("datetime64[Y]").astype(int) + 1970
    for f in files:
        seed = int(f.stem.split("seed")[1])
        d = pd.read_parquet(f)
        d = d[(yrs[d.t] >= years[0]) & (yrs[d.t] <= years[1])]
        per_seed.setdefault(seed, []).append(d)
    seeds = [pd.concat(v, ignore_index=True) for _, v in sorted(per_seed.items())]
    if not seeds:
        return None, []
    ens = pd.concat(seeds).groupby(["t", "i"], as_index=False).agg(y=("y", "first"), y_abs=("y_abs", "first"),
                                                                   y_hat=("y_hat", "mean"))
    return ens, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--ref", default="Ridge")
    ap.add_argument("--panel", default="top100")
    ap.add_argument("--horizon", type=int, default=5)
    a = ap.parse_args()
    WF = wf_dir(a.panel, a.horizon)
    h = a.horizon
    tagp = f"{a.panel}_h{h}_"
    period = "holdout" if a.unseal else "dev"
    years = (HOLDOUT_FIRST_YEAR, HOLDOUT_LAST_YEAR) if a.unseal else (DEV_FIRST_YEAR, DEV_LAST_YEAR)
    if a.unseal:
        with open(OUT / "UNSEAL_LOG.txt", "a") as fh:
            fh.write(f"holdout evaluated: models={a.models or 'all'}\n")
    P = load_panel(a.panel, a.horizon)[0]
    names = [m for m in (a.models.split(",") if a.models else sorted(p.name for p in WF.iterdir() if p.is_dir()))]
    ens, seedruns = {}, {}
    for n in names:
        e, s = load_model(n, years, P, WF)
        if e is not None:
            ens[n], seedruns[n] = e, s
    # factor baselines on the same (t, i) samples as the reference model
    base = ens.get(a.ref) if a.ref in ens else next(iter(ens.values()))
    for fname, (feat, sign) in FACTORS.items():
        ens[fname] = base[["t", "i", "y", "y_abs"]].assign(y_hat=sign * P.raw(feat, base.t.to_numpy(), base.i.to_numpy()))
        seedruns[fname] = [ens[fname]]
    rows, years_rows = [], []
    ic_series = {n: daily_ic(d) for n, d in ens.items()}
    for n, d in ens.items():
        r = {"model": n, "n_seeds": len(seedruns[n])}
        r.update(evaluate(d, h=h))
        if len(seedruns[n]) > 1:
            ics = [daily_ic(s).mean() for s in seedruns[n]]
            m, sd, half = mean_ci(ics)
            r.update({"seed_IC_mean": m, "seed_IC_ci": half})
        dm0 = dm_vs(d, h=h)
        r.update({"DM_zero_stat": dm0["stat"], "DM_zero_p": dm0["p"]})
        if n != a.ref and a.ref in ic_series:
            j = ic_series[n].index.intersection(ic_series[a.ref].index)
            diff = ic_series[n].loc[j] - ic_series[a.ref].loc[j]
            t = hac_mean_test(diff.to_numpy(), lag=h - 1)
            r.update({"dIC_vs_ref": float(diff.mean()), "dIC_p": t["p"]})
        rows.append(r)
        yr = P.dates[ic_series[n].index].astype("datetime64[Y]").astype(int) + 1970
        for y, v in pd.Series(ic_series[n].to_numpy(), index=yr).groupby(level=0).mean().items():
            years_rows.append({"model": n, "year": int(y), "IC": float(v)})
    res = pd.DataFrame(rows).sort_values("IC", ascending=False)
    res.to_csv(OUT / f"{tagp}{period}_models.csv", index=False)
    yr = pd.DataFrame(years_rows).pivot(index="model", columns="year", values="IC")
    yr.to_csv(OUT / f"{tagp}{period}_ic_by_year.csv")
    cols = ["model", "n_seeds", "IC", "IC_t", "ICIR", "IC_hit", "R2_OS", "DM_zero_p", "dIC_vs_ref", "dIC_p",
            "LS_sharpe@0", "LS_sharpe@5", "LS_sharpe@10", "LS_ann_return@5", "LS_mdd@5", "LS_turnover@5"]
    show = res[[c for c in cols if c in res]].copy()
    pd.set_option("display.width", 250)
    print(f"== {a.panel} h={h} {period} period {years[0]}-{years[1]} (ref = {a.ref})")
    print(show.round(4).to_string(index=False))
    print("\nIC by year:\n", yr.round(4).to_string())
    try:
        text = show.round(4).to_markdown(index=False)
    except ImportError:
        text = show.round(4).to_string(index=False)
    (OUT / f"{tagp}{period}_models.md").write_text(text)


if __name__ == "__main__":
    main()
