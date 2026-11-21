"""Deterministic training loop shared by every neural model.

Protocol (identical for all neural models; matches Notebooks 12.1/13 except for the loss switch):
AdamW (betas 0.9/0.95), batch 64, at most ``max_epochs`` epochs, ReduceLROnPlateau(factor 0.5,
patience 3) on validation MSE, gradient-norm clipping at 1.0, early stopping after ``patience``
epochs without improvement, and restoration of the weights with the lowest validation MSE.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class CompositeLoss(nn.Module):
    """MSE + beta * (1 - Pearson correlation) + gamma * (-mean(sign(y) * tanh(s * y_hat))).

    Values alpha=1, beta=0.005, gamma=0.002, s=10 are those used in Notebooks 09/10.
    """

    def __init__(self, alpha=1.0, beta=0.005, gamma=0.002, scale=10.0):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.scale = alpha, beta, gamma, scale

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        pd_, td = pred - pred.mean(), target - target.mean()
        corr = (pd_ * td).mean() / ((pred.std(unbiased=False) + 1e-8) * (target.std(unbiased=False) + 1e-8))
        direction = -torch.mean(torch.sign(target) * torch.tanh(self.scale * pred))
        return self.alpha * mse + self.beta * (1.0 - corr) + self.gamma * direction


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 35
    patience: int = 8
    lr_patience: int = 3
    clip: float = 1.0
    loss: str = "mse"            # "mse" or "composite"


@dataclass
class TrainResult:
    model: nn.Module
    best_val_mse: float
    epochs_run: int
    best_epoch: int
    train_time: float


def _predict(model: nn.Module, X: torch.Tensor, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(X[i:i + batch]).float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    return _predict(model, torch.as_tensor(X, dtype=torch.float32, device=DEVICE))


def train_model(model: nn.Module, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
                cfg: TrainConfig, seed: int) -> TrainResult:
    """Train ``model`` (already constructed under ``set_seed(seed)``) and return the best-val model."""
    model = model.to(DEVICE)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=DEVICE)
    yva_np = np.asarray(yva, dtype=np.float64)
    criterion = CompositeLoss() if cfg.loss == "composite" else nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=cfg.lr_patience)
    gen = torch.Generator(device="cpu").manual_seed(seed)   # data order depends only on the seed
    n = len(Xtr_t)
    best, best_state, best_epoch, bad = float("inf"), None, -1, 0
    t0 = time.perf_counter()
    epoch = 0
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            if cfg.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
        val_mse = float(np.mean((_predict(model, Xva_t) - yva_np) ** 2))
        sched.step(val_mse)
        if val_mse < best:
            best, best_epoch, bad = val_mse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, best, epoch + 1, best_epoch + 1, elapsed)
