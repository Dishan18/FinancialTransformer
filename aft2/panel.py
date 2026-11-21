"""Panel tensors, walk-forward splits and GPU window gathering.

Normalisation (per refit, using training dates only where statistics are estimated):
  * stock-level features: cross-sectional rank on each date mapped to (-1, 1).  A rank uses only
    same-date information, so it is computed once for the whole panel without leakage;
  * market-state features (identical across stocks on a date): z-scored with the mean and standard
    deviation of the training dates of the refit, then clipped to [-5, 5];
  * target: cross-sectionally demeaned 5-day return, winsorised for *training only* at the 0.5 % /
    99.5 % quantiles of the training targets.
A sample (t, i) is usable with look-back L if all features of stock i are finite on dates
t-L+1 .. t and its target is finite.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .features import ALL_FEATURES, GROUPS, MARKET_FEATURES, STOCK_FEATURES

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 5


def _cs_rank(x: np.ndarray) -> np.ndarray:
    """x: (T, N) -> ranks mapped to (-1, 1) per row, NaN preserved."""
    out = np.full_like(x, np.nan, dtype=np.float32)
    for t in range(x.shape[0]):
        row = x[t]
        ok = np.isfinite(row)
        n = ok.sum()
        if n == 0:
            continue
        v = row[ok]
        order = v.argsort(kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1)
        # average ranks for ties
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        if (cnt > 1).any():
            sums = np.bincount(inv, weights=ranks)
            ranks = sums[inv] / cnt[inv]
        out[t, ok] = ((ranks - 0.5) / n * 2 - 1).astype(np.float32)
    return out


@dataclass
class Panel:
    X_raw: np.ndarray          # (T, N, K) raw stock-level features (old format: all features)
    y_abs: np.ndarray          # (T, N)
    y_rel: np.ndarray          # (T, N)
    dates: np.ndarray
    tickers: np.ndarray
    features: list[str]        # names of the columns of X_raw
    X_mkt: np.ndarray | None = None          # (T, Km) market features (new format only)
    market_features: list[str] | None = None
    mkt_weight: np.ndarray | None = None     # (T,) number of universe stocks per date (new format)
    horizon: int = HORIZON

    @staticmethod
    def load(path: Path | None = None, horizon: int = HORIZON) -> "Panel":
        z = np.load(path or ROOT / "data" / "panel_v2.npz", allow_pickle=True)
        dates = z["dates"].astype("datetime64[D]")
        if "X_stock" in z:                                   # new (point-in-time, multi-horizon) format
            return Panel(z["X_stock"], z[f"y_abs_{horizon}"], z[f"y_rel_{horizon}"], dates, z["tickers"],
                         [str(f) for f in z["stock_features"]], z["X_mkt"], [str(f) for f in z["market_features"]],
                         z["mkt_weight"], horizon)
        assert horizon == HORIZON, "old-format panels only contain the 5-day target"
        return Panel(z["X"], z["y_abs"], z["y_rel"], dates, z["tickers"], [str(f) for f in z["features"]])

    def raw(self, feature: str, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        """Raw value of a stock or market feature at (t, i)."""
        if feature in self.features:
            return self.X_raw[t, i, self.features.index(feature)]
        return self.X_mkt[t, self.market_features.index(feature)]

    @property
    def T(self):
        return self.X_raw.shape[0]

    @property
    def N(self):
        return self.X_raw.shape[1]

    def ranked_stock_features(self, cache: Path | None = None, stock_features: list[str] | None = None) -> np.ndarray:
        cache = cache or ROOT / "data" / "panel_v2_ranks.npy"
        if cache.exists():
            return np.load(cache)
        idx = [self.features.index(f) for f in (stock_features or STOCK_FEATURES)]
        R = np.stack([_cs_rank(self.X_raw[:, :, k]) for k in idx], axis=-1)
        np.save(cache, R)
        return R


def group_indices(features: list[str]) -> dict[str, list[int]]:
    return {g: [features.index(f) for f in fs if f in features] for g, fs in GROUPS.items()
            if any(f in features for f in fs)}


@dataclass
class Fold:
    """Dates (indices into the panel) of one walk-forward refit."""
    train_dates: np.ndarray
    val_dates: np.ndarray
    test_dates: np.ndarray


def walk_forward(dates: np.ndarray, first_test_year: int = 2016, last_year: int = 2025, block_years: int = 1,
                 val_days: int = 252, h: int = HORIZON) -> list[Fold]:
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    folds = []
    for y0 in range(first_test_year, last_year + 1, block_years):
        test = np.where((years >= y0) & (years < y0 + block_years))[0]
        if len(test) == 0:
            continue
        s = test[0]
        hist_end = s - h                     # last h origins before the block are purged
        v0 = hist_end - val_days
        folds.append(Fold(np.arange(0, v0 - h), np.arange(v0, hist_end), test))
    return folds


class PanelTensors:
    """Normalised feature tensor on the GPU plus helpers to enumerate and gather samples."""

    def __init__(self, panel: Panel, ranks: np.ndarray, train_dates: np.ndarray, device,
                 feature_mode: str = "rank", stock_features: list[str] | None = None,
                 market_features: list[str] | None = None):
        if panel.X_mkt is not None:
            self._init_split(panel, ranks, train_dates, device, feature_mode)
            return
        self.S = None
        stock_features = stock_features if stock_features is not None else STOCK_FEATURES
        market_features = market_features if market_features is not None else MARKET_FEATURES
        stock_idx = [panel.features.index(f) for f in stock_features]
        mkt_idx = [panel.features.index(f) for f in market_features]
        if feature_mode == "rank":
            S = ranks
        elif feature_mode == "zscore":   # pooled time-series z-score with training statistics (ablation)
            S = panel.X_raw[:, :, stock_idx].astype(np.float64)
            tr = S[train_dates]
            mu, sd = np.nanmean(tr, axis=(0, 1)), np.nanstd(tr, axis=(0, 1))
            S = np.clip((S - mu) / np.where(sd > 0, sd, 1), -5, 5).astype(np.float32)
        else:
            raise ValueError(feature_mode)
        if mkt_idx:
            Mk = panel.X_raw[:, :, mkt_idx].astype(np.float64)
            trm = Mk[train_dates]
            mu, sd = np.nanmean(trm, axis=(0, 1)), np.nanstd(trm, axis=(0, 1))
            Mk = np.clip((Mk - mu) / np.where(sd > 0, sd, 1), -5, 5).astype(np.float32)
            X = np.concatenate([S, Mk], axis=-1)
        else:
            X = S.astype(np.float32)
        self.features = list(stock_features) + list(market_features)
        self.finite = np.isfinite(X).all(-1)                        # (T, N)
        X = np.nan_to_num(X, nan=0.0)
        self.X = torch.as_tensor(X, device=device)
        self.y = panel.y_rel
        self.y_abs = panel.y_abs
        self.device = device
        # cumulative count of finite rows per stock for O(1) window validity checks
        self._cum = np.concatenate([np.zeros((1, panel.N), int), np.cumsum(self.finite, axis=0)], axis=0)

    def _init_split(self, panel: Panel, ranks: np.ndarray, train_dates: np.ndarray, device, feature_mode: str):
        """New format: stock part (T, N, Ks) and market part (T, Km) kept separately on the GPU."""
        if feature_mode == "rank":
            S = ranks
        elif feature_mode == "zscore":
            S = panel.X_raw
            tr = S[train_dates].astype(np.float64)
            mu, sd = np.nanmean(tr, axis=(0, 1)), np.nanstd(tr, axis=(0, 1))
            S = np.clip((S - mu.astype(np.float32)) / np.where(sd > 0, sd, 1).astype(np.float32), -5, 5)
        else:
            raise ValueError(feature_mode)
        Mk = panel.X_mkt.astype(np.float64)
        w = panel.mkt_weight.astype(np.float64)[train_dates]
        trm = Mk[train_dates]
        ok = np.isfinite(trm)
        ww = np.where(ok, w[:, None], 0.0)
        mu = np.nansum(np.where(ok, trm, 0) * ww, 0) / ww.sum(0)
        sd = np.sqrt(np.nansum(np.where(ok, (trm - mu) ** 2, 0) * ww, 0) / ww.sum(0))
        Mk = np.clip((Mk - mu) / np.where(sd > 0, sd, 1), -5, 5).astype(np.float32)
        self.features = list(panel.features) + list(panel.market_features)
        fin_s = np.isfinite(S).all(-1)
        fin_m = np.isfinite(Mk).all(-1)
        self.finite = fin_s & fin_m[:, None]
        self.S = torch.empty(S.shape, dtype=torch.float32, device=device)
        for a in range(0, S.shape[0], 256):                     # chunked copy keeps host memory low
            self.S[a:a + 256] = torch.as_tensor(np.nan_to_num(S[a:a + 256], nan=0.0), device=device)
        self.M = torch.as_tensor(np.nan_to_num(Mk, nan=0.0), device=device)
        self.X = None
        self.y = panel.y_rel
        self.y_abs = panel.y_abs
        self.device = device
        self._cum = np.concatenate([np.zeros((1, panel.N), int), np.cumsum(self.finite, axis=0)], axis=0)

    def samples(self, dates: np.ndarray, L: int) -> np.ndarray:
        """All usable (t, i) pairs with t in ``dates`` for look-back L; shape (n, 2)."""
        dates = dates[dates >= L - 1]
        win_ok = (self._cum[dates + 1] - self._cum[dates + 1 - L]) == L
        tgt_ok = np.isfinite(self.y[dates])
        t, i = np.nonzero(win_ok & tgt_ok)
        return np.stack([dates[t], i], axis=1)

    def gather(self, idx: torch.Tensor, L: int) -> torch.Tensor:
        """idx: (B, 2) long tensor of (t, i) -> windows (B, L, K)."""
        offs = torch.arange(-L + 1, 1, device=idx.device)
        tt = idx[:, :1] + offs
        if self.S is None:
            return self.X[tt, idx[:, 1:2]]
        return torch.cat([self.S[tt, idx[:, 1:2]], self.M[tt]], dim=-1)

    def current(self, idx: torch.Tensor) -> torch.Tensor:
        if self.S is None:
            return self.X[idx[:, 0], idx[:, 1]]
        return torch.cat([self.S[idx[:, 0], idx[:, 1]], self.M[idx[:, 0]]], dim=-1)

    @property
    def K(self) -> int:
        return self.X.shape[-1] if self.S is None else self.S.shape[-1] + self.M.shape[-1]
