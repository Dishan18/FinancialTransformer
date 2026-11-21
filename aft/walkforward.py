"""Walk-forward (expanding-window) generation of out-of-sample base-model forecasts.

The sample is cut into consecutive out-of-sample blocks of ``block`` rows starting at row ``start``.
For block k with rows [s_k, s_{k+1}):
  * the pre-block history rows [0, s_k - h) are split chronologically 85/15 into an inner training
    part and an inner validation part (used only for early stopping / ridge penalty / XGBoost
    rounds); the last h origins of the inner training part are purged, and so are the last h
    origins before the block (their targets are realised inside the block);
  * features are standardised with the statistics of the inner training rows;
  * every base model is refitted and predicts all origins of the block.
Concatenating the blocks gives forecasts that are out-of-sample at every date, which is what a
meta-model must be trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import HORIZON, MAX_LOOKBACK, SplitData

WF_START = 1008      # first out-of-sample row (~March 2019 for the 2015-2025 sample)
WF_BLOCK = 252       # yearly refits


def blocks(n_rows: int, start: int = WF_START, block: int = WF_BLOCK) -> list[tuple[int, int]]:
    out, s = [], start
    while s < n_rows:
        out.append((s, min(s + block, n_rows)))
        s += block
    return out


def block_split(df: pd.DataFrame, features: list[str], s: int, e: int, h: int = HORIZON,
                inner_val: float = 0.15, max_lookback: int = MAX_LOOKBACK) -> SplitData:
    hist_end = s - h                          # first origin whose target is realised inside the block
    v = int(hist_end * (1 - inner_val))       # inner validation starts here
    X = df[features].to_numpy(np.float64)
    scaler = StandardScaler().fit(X[:v])
    Z = scaler.transform(X)
    y = df["Future_Return"].to_numpy(np.float64)
    train_idx = np.arange(max_lookback - 1, v - h)
    val_idx = np.arange(v, hist_end)
    test_idx = np.arange(s, e)
    return SplitData(df, list(features), Z, y, train_idx, val_idx, test_idx, v, hist_end, scaler)
