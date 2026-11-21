"""Experiment runners.  Every run stores its per-sample validation and test predictions, so all
reported metrics, tests and tables are computed afterwards from saved predictions only."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ALL_FEATURES, SplitData, group_indices, rf_top_features
from .models import (AdaptiveFinancialTransformer, DLinear, RNNRegressor, VanillaTransformer,
                     count_parameters)
from .train import TrainConfig, predict, set_seed, train_model

SEEDS = [42, 101, 2023, 777, 999]

# Baseline AFT configuration (Notebooks 10.1 / 13).
AFT_BASE = {"lr": 1e-3, "wd": 1e-4, "dropout": 0.15, "seq_len": 60, "d_model": 128, "ff_dim": 256,
            "num_heads": 8, "num_layers": 2, "use_top40": False}


def params_from_optuna(p: dict) -> dict:
    return {"lr": p["lr"], "wd": p["wd"], "dropout": p["dropout"], "seq_len": p["seq_len"],
            "d_model": p["d_model"], "ff_dim": p["ff_multiplier"] * p["d_model"],
            "num_heads": p["num_heads"], "num_layers": p["num_layers"], "use_top40": p["use_top40"]}


class Store:
    """One parquet file of predictions and one JSON record per (model, seed)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "preds").mkdir(parents=True, exist_ok=True)

    def _stem(self, model: str, seed: int) -> str:
        safe = model.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        return f"{safe}__{seed}"

    def has(self, model: str, seed: int) -> bool:
        return (self.root / "preds" / f"{self._stem(model, seed)}.parquet").exists()

    def save(self, model: str, seed: int, frame: pd.DataFrame, meta: dict) -> None:
        stem = self._stem(model, seed)
        frame.to_parquet(self.root / "preds" / f"{stem}.parquet", index=False)
        (self.root / "preds" / f"{stem}.json").write_text(json.dumps(meta, indent=1, default=float))

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        preds = [pd.read_parquet(p) for p in sorted((self.root / "preds").glob("*.parquet"))]
        metas = [json.loads(p.read_text()) for p in sorted((self.root / "preds").glob("*.json"))]
        return pd.concat(preds, ignore_index=True), pd.DataFrame(metas)


def _frame(model, seed, sd: SplitData, val_pred, test_pred, asset):
    parts = []
    for split, idx, pred in (("val", sd.val_idx, val_pred), ("test", sd.test_idx, test_pred)):
        parts.append(pd.DataFrame({"asset": asset, "model": model, "seed": seed, "split": split,
                                   "row": idx, "date": sd.dates.iloc[idx].to_numpy(),
                                   "y": sd.y[idx], "y_hat": np.asarray(pred, dtype=np.float64)}))
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- neural models

def build_neural(kind: str, n_features: int, feats: list[str], p: dict, variant: dict | None = None):
    L = p["seq_len"]
    if kind == "AFT":
        return AdaptiveFinancialTransformer(n_features, group_indices(feats), L, d_model=p["d_model"],
                                            num_heads=p["num_heads"], ff_dim=p["ff_dim"],
                                            num_layers=p["num_layers"], dropout=p["dropout"],
                                            **(variant or {}))
    if kind == "Transformer":
        return VanillaTransformer(n_features, d_model=p["d_model"], num_heads=p["num_heads"],
                                  ff_dim=p["ff_dim"], num_layers=p["num_layers"], dropout=p["dropout"])
    if kind in ("LSTM", "GRU"):
        return RNNRegressor(n_features, cell=kind.lower())
    if kind == "DLinear":
        return DLinear(n_features, L)
    raise ValueError(kind)


def run_neural(store: Store, name: str, kind: str, sd: SplitData, p: dict, seeds=SEEDS,
               variant: dict | None = None, loss: str = "mse", asset: str = "AAPL",
               top40: list[str] | None = None, max_epochs: int = 35, patience: int = 8, verbose=True):
    feats = top40 if p.get("use_top40") else list(sd.features)
    sdx = sd.subset(feats) if feats != list(sd.features) else sd
    L = p["seq_len"]
    Xtr, Xva, Xte = sdx.windows(sdx.train_idx, L), sdx.windows(sdx.val_idx, L), sdx.windows(sdx.test_idx, L)
    ytr, yva = sdx.y[sdx.train_idx], sdx.y[sdx.val_idx]
    cfg = TrainConfig(lr=p["lr"], weight_decay=p["wd"], loss=loss, max_epochs=max_epochs, patience=patience)
    for seed in seeds:
        if store.has(name, seed):
            continue
        set_seed(seed)
        model = build_neural(kind, len(feats), feats, p, variant)
        res = train_model(model, Xtr, ytr, Xva, yva, cfg, seed)
        t0 = time.perf_counter()
        test_pred = predict(res.model, Xte)
        infer = time.perf_counter() - t0
        val_pred = predict(res.model, Xva)
        meta = {"asset": asset, "model": name, "kind": kind, "seed": seed, "params": count_parameters(res.model),
                "n_features": len(feats), "seq_len": L, "train_time_s": res.train_time,
                "infer_time_s": infer, "epochs_run": res.epochs_run, "best_epoch": res.best_epoch,
                "best_val_mse": res.best_val_mse, "loss": loss, "variant": json.dumps(variant or {}),
                "hparams": json.dumps(p), "features": json.dumps(feats if len(feats) < 95 else "all95")}
        store.save(name, seed, _frame(name, seed, sdx, val_pred, test_pred, asset), meta)
        if kind == "AFT":
            import torch
            (store.root / "weights").mkdir(exist_ok=True)
            torch.save({k: v.cpu() for k, v in res.model.state_dict().items()},
                       store.root / "weights" / f"{store._stem(name, seed)}.pt")
        if verbose:
            te = sdx.y[sdx.test_idx]
            print(f"  [{asset}] {name:<22} seed {seed:<5} params {meta['params']:>7}  "
                  f"epochs {res.epochs_run:>2} ({res.train_time:5.1f}s)  test MAE {np.abs(test_pred - te).mean():.5f}  "
                  f"DA {np.mean(np.sign(test_pred) == np.sign(te)):.4f}", flush=True)
        del res, model


# ----------------------------------------------------------------------------- other baselines

def run_naive(store: Store, sd: SplitData, asset: str = "AAPL"):
    ytr = sd.y[sd.train_idx]
    mom_col = sd.df["ROC_5"].to_numpy()
    specs = {
        "Zero": (np.zeros(len(sd.val_idx)), np.zeros(len(sd.test_idx))),
        "Train mean": (np.full(len(sd.val_idx), ytr.mean()), np.full(len(sd.test_idx), ytr.mean())),
        "Momentum (5d)": (mom_col[sd.val_idx], mom_col[sd.test_idx]),
    }
    for name, (v, t) in specs.items():
        if store.has(name, 0):
            continue
        store.save(name, 0, _frame(name, 0, sd, v, t, asset),
                   {"asset": asset, "model": name, "kind": "naive", "seed": 0, "params": 0,
                    "n_features": 0, "seq_len": 1})


def run_tabular(store: Store, sd: SplitData, seeds=SEEDS, asset: str = "AAPL", which=("Ridge", "Random Forest", "XGBoost")):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    Xtr, Xva, Xte = sd.tabular(sd.train_idx), sd.tabular(sd.val_idx), sd.tabular(sd.test_idx)
    ytr, yva = sd.y[sd.train_idx], sd.y[sd.val_idx]
    if "Ridge" in which and not store.has("Ridge", 0):
        grid = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        scores = [np.mean((Ridge(alpha=a).fit(Xtr, ytr).predict(Xva) - yva) ** 2) for a in grid]
        alpha = grid[int(np.argmin(scores))]
        m = Ridge(alpha=alpha).fit(Xtr, ytr)
        store.save("Ridge", 0, _frame("Ridge", 0, sd, m.predict(Xva), m.predict(Xte), asset),
                   {"asset": asset, "model": "Ridge", "kind": "tabular", "seed": 0, "alpha": alpha,
                    "params": int(Xtr.shape[1] + 1), "n_features": Xtr.shape[1], "seq_len": 1})
    for seed in seeds:
        if "Random Forest" in which and not store.has("Random Forest", seed):
            t0 = time.perf_counter()
            m = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
            store.save("Random Forest", seed, _frame("Random Forest", seed, sd, m.predict(Xva), m.predict(Xte), asset),
                       {"asset": asset, "model": "Random Forest", "kind": "tabular", "seed": seed,
                        "train_time_s": time.perf_counter() - t0, "params": 0, "n_features": Xtr.shape[1], "seq_len": 1})
        if "XGBoost" in which and not store.has("XGBoost", seed):
            t0 = time.perf_counter()
            m = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed, objective="reg:squarederror",
                             early_stopping_rounds=50, n_jobs=8)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            store.save("XGBoost", seed, _frame("XGBoost", seed, sd, m.predict(Xva), m.predict(Xte), asset),
                       {"asset": asset, "model": "XGBoost", "kind": "tabular", "seed": seed,
                        "best_iteration": int(m.best_iteration), "train_time_s": time.perf_counter() - t0,
                        "params": 0, "n_features": Xtr.shape[1], "seq_len": 1})


def top40_for(sd: SplitData) -> list[str]:
    """Top-40 features by random-forest importance on the (purged) training samples only."""
    return rf_top_features(sd.Z, sd.y, sd.train_idx, list(sd.features), k=40, seed=42, n_estimators=50)
