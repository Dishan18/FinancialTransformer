"""Pooled training on the panel (one model for all stocks), deterministic on GPU.

Protocol: AdamW (betas 0.9/0.95), batch 1024, MSE on the winsorised relative return, at most
``max_epochs`` epochs, early stopping on the mean daily rank IC of the validation dates
(patience ``patience``), restoration of the best-validation weights, gradient clipping at 1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from aft.train import DEVICE, set_seed

from .panel import PanelTensors


@dataclass
class PanelTrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 15
    patience: int = 3
    clip: float = 1.0
    winsor: float = 0.005
    loss: str = "mse"            # "mse" (pooled) or "ic" (date-batched negative cross-sectional correlation)
    dates_per_batch: int = 10


def mean_rank_ic(t: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    df = pd.DataFrame({"t": t, "y": y, "p": p})
    r = df.groupby("t")[["y", "p"]].rank()
    df["ry"], df["rp"] = r.y, r.p
    g = df.groupby("t")
    cov = g.apply(lambda x: np.corrcoef(x.ry, x.rp)[0, 1] if len(x) > 4 else np.nan, include_groups=False)
    return float(np.nanmean(cov))


def predict(model: nn.Module, PT: PanelTensors, idx: np.ndarray, L: int, sequence: bool,
            batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    I = torch.as_tensor(idx, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for b in range(0, len(I), batch):
            xb = PT.gather(I[b:b + batch], L) if sequence else PT.current(I[b:b + batch])
            out.append(model(xb).float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def fit(model: nn.Module, PT: PanelTensors, tr_idx: np.ndarray, va_idx: np.ndarray, L: int, sequence: bool,
        cfg: PanelTrainConfig, seed: int):
    set_seed(seed)
    model = model.to(DEVICE)
    y_tr = PT.y[tr_idx[:, 0], tr_idx[:, 1]].astype(np.float64)
    lo, hi = np.quantile(y_tr, [cfg.winsor, 1 - cfg.winsor])
    y_t = torch.as_tensor(np.clip(y_tr, lo, hi), dtype=torch.float32, device=DEVICE)
    I = torch.as_tensor(tr_idx, dtype=torch.long, device=DEVICE)
    y_va = PT.y[va_idx[:, 0], va_idx[:, 1]]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, bad, hist = -np.inf, None, 0, 0, []
    t0 = time.perf_counter()
    if cfg.loss == "ic":
        # group training samples by date (they are sorted by date because samples() uses np.nonzero)
        tvals = tr_idx[:, 0]
        bounds = np.flatnonzero(np.diff(tvals)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(tvals)]])
    for ep in range(cfg.max_epochs):
        model.train()
        if cfg.loss == "mse":
            perm = torch.randperm(len(I), generator=gen).to(DEVICE)
            batches = (perm[b:b + cfg.batch_size] for b in range(0, len(I), cfg.batch_size))
        else:
            dperm = torch.randperm(len(starts), generator=gen).numpy()
            batches = []
            for b in range(0, len(dperm), cfg.dates_per_batch):
                ds = dperm[b:b + cfg.dates_per_batch]
                idx = np.concatenate([np.arange(starts[d], ends[d]) for d in ds])
                seg = np.concatenate([np.full(ends[d] - starts[d], k) for k, d in enumerate(ds)])
                batches.append((torch.as_tensor(idx, device=DEVICE), torch.as_tensor(seg, device=DEVICE), len(ds)))
        for j in batches:
            if cfg.loss == "mse":
                xb = PT.gather(I[j], L) if sequence else PT.current(I[j])
                opt.zero_grad(set_to_none=True)
                loss = torch.mean((model(xb) - y_t[j]) ** 2)
            else:
                j, seg, nd = j
                xb = PT.gather(I[j], L) if sequence else PT.current(I[j])
                opt.zero_grad(set_to_none=True)
                p, y = model(xb), y_t[j]
                cnt = torch.zeros(nd, device=DEVICE).index_add_(0, seg, torch.ones_like(p))
                pm = torch.zeros(nd, device=DEVICE).index_add_(0, seg, p) / cnt
                ym = torch.zeros(nd, device=DEVICE).index_add_(0, seg, y) / cnt
                pc, yc = p - pm[seg], y - ym[seg]
                cov = torch.zeros(nd, device=DEVICE).index_add_(0, seg, pc * yc)
                vp = torch.zeros(nd, device=DEVICE).index_add_(0, seg, pc * pc)
                vy = torch.zeros(nd, device=DEVICE).index_add_(0, seg, yc * yc)
                loss = -torch.mean(cov / torch.sqrt(vp * vy + 1e-12))
            loss.backward()
            if cfg.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
        p_va = predict(model, PT, va_idx, L, sequence)
        ic = mean_rank_ic(va_idx[:, 0], y_va, p_va)
        hist.append(ic)
        if ic > best:
            best, best_ep, bad = ic, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    torch.cuda.synchronize()
    model.load_state_dict(best_state)
    return model, {"val_ic": best, "best_epoch": best_ep + 1, "epochs": len(hist), "val_ic_hist": hist,
                   "train_time_s": time.perf_counter() - t0, "n_train": int(len(tr_idx))}
