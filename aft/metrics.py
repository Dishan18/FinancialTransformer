"""Metrics computed from saved predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import backtest, directional_accuracy, long_only

COSTS = (0.0, 5.0, 10.0, 20.0)
PRIMARY_COST = 5.0


def regression_metrics(y, y_hat, train_mean: float) -> dict:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    e = y - y_hat
    sse_bench = np.sum((y - train_mean) ** 2)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R2_OS": float(1.0 - np.sum(e ** 2) / sse_bench)}


def run_metrics(y, y_hat, train_mean: float) -> dict:
    out = regression_metrics(y, y_hat, train_mean)
    out["DA"] = directional_accuracy(y, y_hat)
    for c in COSTS:
        bt = backtest(y, y_hat, cost_bps=c)
        tag = f"@{int(c)}"
        out["Sharpe" + tag] = bt["sharpe"]
        out["Return" + tag] = bt["total_return"]
        out["MDD" + tag] = bt["max_drawdown"]
        if c == PRIMARY_COST:
            out["Turnover"] = bt["turnover"]
            out["Sharpe_offsets"] = bt["sharpe_offsets"]
    out["Long share"] = float(np.mean(np.asarray(y_hat) > 0))
    return out


def per_run_table(preds: pd.DataFrame, train_means: dict[str, float], split: str = "test") -> pd.DataFrame:
    """One row per (asset, model, seed) with all metrics on ``split``."""
    rows = []
    for (asset, model, seed), g in preds[preds.split == split].groupby(["asset", "model", "seed"], sort=False):
        g = g.sort_values("row")
        m = run_metrics(g.y.to_numpy(), g.y_hat.to_numpy(), train_means[asset])
        rows.append({"asset": asset, "model": model, "seed": seed, **m})
    return pd.DataFrame(rows)


def benchmark_row(y, train_mean: float, name: str = "Long only (buy & hold)") -> dict:
    y = np.asarray(y, float)
    out = {"model": name, "DA": float(np.mean(y > 0))}
    for c in COSTS:
        bt = long_only(y, cost_bps=c)
        out[f"Sharpe@{int(c)}"] = bt["sharpe"]
        out[f"Return@{int(c)}"] = bt["total_return"]
        out[f"MDD@{int(c)}"] = bt["max_drawdown"]
        if c == PRIMARY_COST:
            out["Turnover"] = bt["turnover"]
    return out


def ensemble(preds: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Seed-averaged prediction per (asset, model, row)."""
    p = preds[preds.split == split]
    return (p.groupby(["asset", "model", "row"], sort=False)
             .agg(date=("date", "first"), y=("y", "first"), y_hat=("y_hat", "mean"), n_seeds=("seed", "nunique"))
             .reset_index())
