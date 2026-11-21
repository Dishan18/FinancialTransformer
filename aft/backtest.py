"""Cost-aware, non-overlapping backtest of a 5-day long/short sign strategy.

Convention
----------
* At a rebalancing date ``t`` the position is ``s_t = sign(y_hat_t)`` in {-1, 0, +1}
  (predictions with |y_hat| < 1e-6 give s_t = 0) and it is held from the close of ``t`` to the
  close of ``t+h``; the realised gross return is ``s_t * y_t``.
* Rebalancing happens every ``h`` trading days, so holding periods never overlap.  Because the
  choice of the first rebalancing day is arbitrary, the backtest is run for each of the ``h``
  possible offsets and every reported statistic is the average over the ``h`` offsets.
* Transaction costs are charged per side on traded notional: a trade that changes the position
  from s to s' costs ``c * |s' - s|`` (so a long-to-short flip costs 2c).  The initial entry and
  the final exit are charged.  ``c`` is given in basis points.
* Net returns are compounded; the Sharpe ratio is mean / standard deviation (ddof = 1) of the
  non-overlapping net ``h``-day returns, annualised by ``sqrt(252 / h)``.  Maximum drawdown is
  measured on the compounded equity curve starting at 1.  Turnover is the mean of |s' - s| per
  rebalancing (0 = never trades, 2 = flips every time).
"""
from __future__ import annotations

import numpy as np

H = 5
ZERO_THRESHOLD = 1e-6


def positions(y_hat: np.ndarray, threshold: float = ZERO_THRESHOLD) -> np.ndarray:
    s = np.sign(np.asarray(y_hat, dtype=np.float64))
    s[np.abs(y_hat) < threshold] = 0.0
    return s


def directional_accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.sign(y) == positions(y_hat)))


def _single_offset(s: np.ndarray, y: np.ndarray, c: float, h: int) -> dict:
    gross = s * y
    prev = np.concatenate([[0.0], s[:-1]])
    trade = np.abs(s - prev)
    net = gross - c * trade
    net[-1] -= c * abs(s[-1])                     # close the final position
    equity = np.concatenate([[1.0], np.cumprod(1.0 + net)])
    running_max = np.maximum.accumulate(equity)
    sd = net.std(ddof=1)
    sharpe = float(net.mean() / sd * np.sqrt(252.0 / h)) if sd > 0 else 0.0
    return {
        "sharpe": sharpe,
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": float(np.min(equity / running_max - 1.0)),
        "turnover": float(trade.mean()),
        "n_trades": int(len(s)),
        "mean_net": float(net.mean()),
    }


def backtest(y: np.ndarray, y_hat: np.ndarray, cost_bps: float = 5.0, h: int = H) -> dict:
    """Phase-averaged non-overlapping backtest (see module docstring)."""
    y = np.asarray(y, dtype=np.float64)
    s = positions(y_hat)
    c = cost_bps / 1e4
    runs = [_single_offset(s[k::h], y[k::h], c, h) for k in range(h)]
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    out["sharpe_offsets"] = [r["sharpe"] for r in runs]
    return out


def long_only(y: np.ndarray, cost_bps: float = 5.0, h: int = H) -> dict:
    """Buy-and-hold benchmark expressed in the same non-overlapping framework."""
    return backtest(y, np.ones_like(np.asarray(y, dtype=np.float64)), cost_bps, h)
