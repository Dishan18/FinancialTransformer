"""Meta-models over the out-of-sample forecasts of the base-model zoo.

Every input of the meta level at date t is available at the close of t:
  * base forecasts: the seed-ensemble walk-forward OOS forecast of each base model, converted to its
    cross-sectional rank on date t (scale-free, comparable across models);
  * trailing skill: for each base model, the mean daily rank IC over the previous ``skill_window``
    dates whose targets are already realised, i.e. dates <= t - h (h = 5);
  * market state: the raw market features of the panel at t.
Meta-learners are refitted once per year on all earlier OOS dates (expanding window), with the last
h dates before the refit purged, and predict the following year.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .panel import Panel

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "results_v2" / "wf"
H = 5
MARKET_STATE = ["vix_log", "mkt_vol_21", "mkt_ret_21", "mkt_ret_63", "mkt_ma_gap_200", "cs_disp_21", "breadth_200"]


def cs_rank_long(df: pd.DataFrame, col: str) -> pd.Series:
    r = df.groupby("t")[col].rank(method="average")
    n = df.groupby("t")[col].transform("count")
    return (r - 0.5) / n * 2 - 1


def load_ensemble(name: str, wf: Path = WF) -> pd.DataFrame | None:
    files = sorted((wf / name).glob("fold*_seed*.parquet"))
    if not files:
        return None
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return d.groupby(["t", "i"], as_index=False).agg(y=("y", "first"), y_abs=("y_abs", "first"), y_hat=("y_hat", "mean"))


def daily_ic_series(df: pd.DataFrame, col: str = "y_hat") -> pd.Series:
    r = df[["t"]].copy()
    r["a"] = df.groupby("t")[col].rank()
    r["b"] = df.groupby("t")["y"].rank()
    g = r.groupby("t")
    return g.apply(lambda x: np.corrcoef(x.a, x.b)[0, 1] if len(x) > 4 else np.nan, include_groups=False)


@dataclass
class MetaData:
    frame: pd.DataFrame          # t, i, y, y_abs, rank_<model>..., skill_<model>..., market state
    models: list[str]
    rank_cols: list[str]
    skill_cols: list[str]
    state_cols: list[str]


def build_meta_data(models: list[str], P: Panel, skill_window: int = 63, factor_feats: dict | None = None,
                    wf: Path = WF, h: int = H) -> MetaData:
    ens = {m: load_ensemble(m, wf) for m in models}
    ens = {m: d for m, d in ens.items() if d is not None}
    models = list(ens)
    keys = None
    for d in ens.values():
        k = d[["t", "i"]]
        keys = k if keys is None else keys.merge(k, on=["t", "i"])
    base = keys.merge(next(iter(ens.values()))[["t", "i", "y", "y_abs"]], on=["t", "i"])
    rank_cols, skill_cols = [], []
    for m, d in ens.items():
        col = f"rank_{m}"
        d = d.merge(keys, on=["t", "i"])
        d[col] = cs_rank_long(d, "y_hat")
        base = base.merge(d[["t", "i", col]], on=["t", "i"])
        rank_cols.append(col)
    for fname, (feat, sign) in (factor_feats or {}).items():
        col = f"rank_{fname}"
        base[col] = sign * P.raw(feat, base.t.to_numpy(), base.i.to_numpy())
        base[col] = cs_rank_long(base, col)
        rank_cols.append(col)
    # trailing realised skill (lagged by h so that only realised targets are used); a short (63-day)
    # and a long (252-day) window
    for col in rank_cols:
        ic = daily_ic_series(base, col)
        trailing = ic.shift(h).rolling(skill_window, min_periods=20).mean()
        sc = "skill_" + col[5:]
        base[sc] = base.t.map(trailing)
        skill_cols.append(sc)
        base["skill252_" + col[5:]] = base.t.map(ic.shift(h).rolling(252, min_periods=60).mean())
    state_cols = []
    for f in MARKET_STATE:
        base["state_" + f] = P.raw(f, base.t.to_numpy(), base.i.to_numpy())
        state_cols.append("state_" + f)
    base["year"] = P.dates[base.t.to_numpy()].astype("datetime64[Y]").astype(int) + 1970
    return MetaData(base, models, rank_cols, skill_cols, state_cols)
