"""Layered meta-model system, evaluated incrementally (development period unless --unseal).

Layer 1  base forecasters: walk-forward OOS forecasts of the model zoo + classic factor signals
Layer 2  combination:      equal-weight average of cross-sectional ranks
Layer 3  exposure gate:    walk-forward ridge regression of the combined signal's realised daily IC on
                           the market state; exposure = 1 if the predicted IC > 0, else 0
                           (fitted only on dates whose IC was realised before each yearly refit)
Layer 4  execution:        hysteresis (buffer) long-short: enter top/bottom 20 %, keep within 50 %

Usage: python scripts/v2_metasystem.py --panel sp500 --horizon 5 --models XGBoost,Ridge,MLP[,...]
                                       [--factors rev5,rev21] [--unseal]
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

from aft2.evaluate import daily_ic, evaluate, long_short, long_short_buffer  # noqa: E402
from aft2.meta import MARKET_STATE, cs_rank_long, load_ensemble  # noqa: E402
from aft2.protocol import (DEV_FIRST_YEAR, DEV_LAST_YEAR, HOLDOUT_FIRST_YEAR, HOLDOUT_LAST_YEAR,  # noqa: E402
                           load_panel, wf_dir)

FACTORS = {"rev5": ("ret_5", -1), "rev21": ("ret_21", -1), "mom": ("mom_12_1", +1), "lowvol": ("vol_63", -1),
           "max": ("max_ret_21", -1)}


def gate(signal: pd.DataFrame, state: pd.DataFrame, years: list[int], year_of: pd.Series, h: int) -> pd.Series:
    from sklearn.linear_model import Ridge
    ic = daily_ic(signal).rename("ic")
    df = state.join(ic, how="inner")
    df["year"] = year_of.reindex(df.index)
    out = {}
    for y in years:
        test = df[df.year == y]
        if test.empty:
            continue
        t0 = test.index.min()
        train = df[df.index <= t0 - h].dropna()
        mu, sd = train[MARKET_STATE].mean(), train[MARKET_STATE].std().replace(0, 1)
        m = Ridge(alpha=10.0).fit(((train[MARKET_STATE] - mu) / sd).to_numpy(), train.ic.to_numpy())
        p = m.predict(((test[MARKET_STATE] - mu) / sd).fillna(0).to_numpy())
        out.update(dict(zip(test.index, (p > 0).astype(float))))
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="sp500")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--models", default="XGBoost,Ridge,MLP")
    ap.add_argument("--factors", default="rev5,rev21")
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    h = a.horizon
    P = load_panel(a.panel, h)[0]
    wf = wf_dir(a.panel, h)
    year_all = pd.Series(P.dates.astype("datetime64[Y]").astype(int) + 1970)
    if a.unseal:
        years_eval = list(range(HOLDOUT_FIRST_YEAR, HOLDOUT_LAST_YEAR + 1))
        with open(ROOT / "results_v2" / "summary" / "UNSEAL_LOG.txt", "a") as fh:
            fh.write(f"metasystem holdout evaluated: {vars(a)}\n")
    else:
        years_eval = list(range(DEV_FIRST_YEAR + 2, DEV_LAST_YEAR + 1))   # gate needs >= 1 year of history
    last_year = max(years_eval)
    # ---------------- layer 1: base signals on common samples
    sigs = {}
    for m in a.models.split(","):
        d = load_ensemble(m, wf)
        sigs[m] = d[year_all.to_numpy()[d.t] <= last_year]
    keys = None
    for d in sigs.values():
        keys = d[["t", "i", "y", "y_abs"]] if keys is None else keys.merge(d[["t", "i"]], on=["t", "i"])
    base = keys.copy()
    for m, d in sigs.items():
        base["r_" + m] = cs_rank_long(base.merge(d[["t", "i", "y_hat"]], on=["t", "i"]), "y_hat").to_numpy()
    for fct in [f for f in a.factors.split(",") if f]:
        feat, sign = FACTORS[fct]
        base["tmp"] = sign * P.raw(feat, base.t.to_numpy(), base.i.to_numpy())
        base["r_" + fct] = cs_rank_long(base, "tmp").to_numpy()
    base = base.drop(columns=["tmp"], errors="ignore")
    rcols = [c for c in base if c.startswith("r_")]
    # ---------------- layer 2: equal-weight combination
    base["combo"] = base[rcols].mean(axis=1)
    state = pd.DataFrame({f: P.raw(f, np.arange(P.T), np.zeros(P.T, int)) for f in MARKET_STATE})
    # ---------------- layer 3: gate (fitted walk-forward on all earlier OOS dates)
    combo = base[["t", "i", "y", "y_abs"]].assign(y_hat=base.combo)
    expo = gate(combo, state, years_eval, year_all, h)
    ev = base[year_all.to_numpy()[base.t] >= min(years_eval)]
    rows = []

    def add(name, d, exposure=None, buffer=False):
        r = {"system": name}
        ic = daily_ic(d)
        r["IC"] = float(ic.mean())
        for c in (0.0, 5.0, 10.0, 20.0):
            x = (long_short_buffer(d, cost_bps=c, q_in=0.2, q_out=0.5, h=h, exposure=exposure) if buffer
                 else long_short(d, cost_bps=c, h=h, exposure=exposure))
            r[f"SR@{int(c)}"] = x["sharpe"]
            if c == 5.0:
                r.update({"ann_ret@5": x["ann_return"], "mdd@5": x["mdd"], "turnover": x["turnover"]})
        r["exposure"] = float(np.mean(list(exposure.values))) if exposure is not None else 1.0
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

    for c in rcols:
        add(f"L1 base: {c[2:]}", ev[["t", "i", "y", "y_abs"]].assign(y_hat=ev[c]))
    dcombo = ev[["t", "i", "y", "y_abs"]].assign(y_hat=ev.combo)
    add("L2 combination (EW ranks)", dcombo)
    add("L2+L3 combination + regime gate", dcombo, exposure=expo)
    add("L2+L4 combination + buffer  [System A]", dcombo, buffer=True)
    add("L2+L3+L4 full system", dcombo, exposure=expo, buffer=True)
    # System B: reversal components are gated by their own walk-forward regime gate; model components
    # are always on.  When a reversal gate is off, that component contributes 0 (= neutral rank).
    rev = [c for c in rcols if c[2:] in ("rev5", "rev21")]
    if rev:
        gated = base[rcols].copy()
        for c in rev:
            sig_c = base[["t", "i", "y", "y_abs"]].assign(y_hat=base[c])
            e_c = gate(sig_c, state, years_eval, year_all, h)
            gated[c] = base[c] * base.t.map(e_c).fillna(0.0).to_numpy()
        base["comboB"] = gated.mean(axis=1)
        evB = base[year_all.to_numpy()[base.t] >= min(years_eval)]
        dB = evB[["t", "i", "y", "y_abs"]].assign(y_hat=evB.comboB)
        add("System B: regime-gated reversal components (no buffer)", dB)
        add("System B: regime-gated reversal components + buffer", dB, buffer=True)
    res = pd.DataFrame(rows)
    tag = f"{a.panel}_h{h}_{'holdout' if a.unseal else 'dev'}{('_' + a.tag) if a.tag else ''}"
    res.to_csv(ROOT / "results_v2" / "summary" / f"metasystem_{tag}.csv", index=False)
    print(f"\n== meta system {tag}, years {years_eval[0]}-{years_eval[-1]}, inputs: {rcols}")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
