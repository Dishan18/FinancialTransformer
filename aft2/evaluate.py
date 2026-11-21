"""Evaluation of cross-sectional forecasts on a panel.

Inputs are long-format predictions with columns: t (date index), i (stock index), y (relative
5-day return), y_abs (5-day return), y_hat.

* Rank IC: Spearman correlation between y_hat and y across stocks on each date; summarised by its
  mean, a Newey-West t-statistic (lag 4, because 5-day targets on consecutive dates overlap) and
  the information ratio.
* Long-short quintile portfolio: on each rebalancing date the top 20 % of stocks by y_hat are bought
  and the bottom 20 % sold short, equal weight, $1 per leg; positions are held for 5 days and
  rebalanced every 5 days.  Costs are charged per side on traded notional (c * sum |w' - w|).  All
  statistics are averaged over the 5 possible rebalancing phases.
* Forecast accuracy: MSE and R^2_OS against the zero forecast of the relative return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aft.stats import dm_from_differential, hac_mean_test

H = 5


def daily_ic(df: pd.DataFrame) -> pd.Series:
    def ic(g):
        if len(g) < 5 or g.y_hat.nunique() < 2:
            return np.nan
        return np.corrcoef(g.y_hat.rank(), g.y.rank())[0, 1]
    return df.groupby("t").apply(ic, include_groups=False).dropna()


def long_short(df: pd.DataFrame, cost_bps: float = 5.0, q: float = 0.2, h: int = H,
               exposure: pd.Series | None = None) -> dict:
    """``exposure`` (optional): gross scale in [0, 1] per date t applied to both legs (known at t)."""
    c = cost_bps / 1e4
    dates = np.sort(df.t.unique())
    by_t = {t: g for t, g in df.groupby("t")}
    runs = []
    for k in range(h):
        prev = {}
        rets, turns = [], []
        for t in dates[k::h]:
            g = by_t[t]
            n = len(g)
            m = max(1, int(np.floor(q * n)))
            order = g.y_hat.to_numpy().argsort(kind="stable")
            ids = g.i.to_numpy()
            ya = g.y_abs.to_numpy()
            long_ids, short_ids = ids[order[-m:]], ids[order[:m]]
            e = 1.0 if exposure is None else float(exposure.get(t, 0.0))
            w = {**{s: -e / m for s in short_ids}, **{l: e / m for l in long_ids}}
            w = {k_: v for k_, v in w.items() if v != 0.0}
            keys = set(w) | set(prev)
            turn = sum(abs(w.get(s, 0.0) - prev.get(s, 0.0)) for s in keys)
            gross = e * (ya[order[-m:]].mean() - ya[order[:m]].mean())
            rets.append(gross - c * turn)
            turns.append(turn)
            prev = w
        rets = np.array(rets)
        rets[-1] -= c * sum(abs(v) for v in prev.values())
        eq = np.concatenate([[1.0], np.cumprod(1 + rets)])
        sd = rets.std(ddof=1)
        runs.append({"sharpe": rets.mean() / sd * np.sqrt(252 / h) if sd > 0 else 0.0,
                     "ann_return": (eq[-1] ** (252 / (h * len(rets))) - 1),
                     "ann_vol": float(sd * np.sqrt(252 / h)),
                     "mdd": float((eq / np.maximum.accumulate(eq) - 1).min()),
                     "turnover": float(np.mean(turns)), "mean_ret": float(rets.mean())})
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


def long_short_buffer(df: pd.DataFrame, cost_bps: float = 5.0, q_in: float = 0.2, q_out: float = 0.4,
                      h: int = H, exposure: pd.Series | None = None) -> dict:
    """Long-short with hysteresis: a stock enters the long (short) leg when its forecast is in the top
    (bottom) ``q_in`` of the cross-section and is kept while it stays in the top (bottom) ``q_out``.
    Equal weights within each leg ($1 per leg, scaled by ``exposure``); other conventions as long_short."""
    c = cost_bps / 1e4
    dates = np.sort(df.t.unique())
    by_t = {t: g for t, g in df.groupby("t")}
    runs = []
    for k in range(h):
        prev, L, S = {}, set(), set()
        rets, turns = [], []
        for t in dates[k::h]:
            g = by_t[t]
            pct = g.y_hat.rank(pct=True).to_numpy()
            ids = g.i.to_numpy()
            ya = dict(zip(ids, g.y_abs.to_numpy()))
            top_in = set(ids[pct > 1 - q_in]); top_keep = set(ids[pct > 1 - q_out])
            bot_in = set(ids[pct <= q_in]); bot_keep = set(ids[pct <= q_out])
            L = (L & top_keep) | top_in
            S = ((S & bot_keep) | bot_in) - L
            e = 1.0 if exposure is None else float(exposure.get(t, 0.0))
            w = {}
            if L and S and e > 0:
                w.update({s: -e / len(S) for s in S})
                w.update({l: e / len(L) for l in L})
            keys = set(w) | set(prev)
            turn = sum(abs(w.get(s, 0.0) - prev.get(s, 0.0)) for s in keys)
            gross = sum(wt * ya[s] for s, wt in w.items())
            rets.append(gross - c * turn)
            turns.append(turn)
            prev = w
        rets = np.array(rets)
        rets[-1] -= c * sum(abs(v) for v in prev.values())
        eq = np.concatenate([[1.0], np.cumprod(1 + rets)])
        sd = rets.std(ddof=1)
        runs.append({"sharpe": rets.mean() / sd * np.sqrt(252 / h) if sd > 0 else 0.0,
                     "ann_return": (eq[-1] ** (252 / (h * len(rets))) - 1),
                     "mdd": float((eq / np.maximum.accumulate(eq) - 1).min()),
                     "turnover": float(np.mean(turns)), "rets": rets})
    out = {k: float(np.mean([r[k] for r in runs])) for k in ("sharpe", "ann_return", "mdd", "turnover")}
    return out


def evaluate(df: pd.DataFrame, costs=(0.0, 5.0, 10.0, 20.0), exposure: pd.Series | None = None,
             h: int = H) -> dict:
    ic = daily_ic(df)
    nw = hac_mean_test(ic.to_numpy(), lag=h - 1)
    out = {"IC": float(ic.mean()), "IC_t": nw["stat"], "IC_p": nw["p"], "ICIR": float(ic.mean() / ic.std() * np.sqrt(252)),
           "IC_hit": float((ic > 0).mean()), "n_dates": int(len(ic)), "n_obs": int(len(df))}
    e = df.y - df.y_hat
    out["MSE"] = float(np.mean(e ** 2))
    out["R2_OS"] = float(1 - np.sum(e ** 2) / np.sum(df.y ** 2))
    for cst in costs:
        ls = long_short(df, cost_bps=cst, exposure=exposure, h=h)
        for k, v in ls.items():
            out[f"LS_{k}@{int(cst)}"] = v
    return out


def dm_vs(df_a: pd.DataFrame, df_b: pd.DataFrame | None = None, power: int = 2, h: int = H) -> dict:
    """DM test of model a vs model b (or vs the zero forecast) on the cross-sectionally averaged loss."""
    a = df_a.assign(la=np.abs(df_a.y - df_a.y_hat) ** power)
    if df_b is None:
        a["lb"] = np.abs(a.y) ** power
        m = a
    else:
        b = df_b.assign(lb=np.abs(df_b.y - df_b.y_hat) ** power)[["t", "i", "lb"]]
        m = a.merge(b, on=["t", "i"])
    d = m.groupby("t").apply(lambda g: (g.la - g.lb).mean(), include_groups=False)
    return dm_from_differential(d.to_numpy(), h=h)
