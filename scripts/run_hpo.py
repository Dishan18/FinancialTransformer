"""Hyper-parameter search for the AFT with a leakage-controlled protocol.

Differences from Notebook 12.1 (all of which remove possible information leakage or make the
search reproducible):
  * windows end at the forecast origin t (aligned targets), training targets that overlap the
    validation block are purged;
  * the Top-40 feature subset is selected inside each fold from that fold's training rows only;
  * the composite objective uses only validation-fold quantities (no constants measured on the
    test set):  score = 0.40 * MAE / MAE_zero - 0.30 * DA / 0.5 - 0.30 * SR_net
    where MAE_zero is the MAE of the zero forecast on the same validation fold and SR_net is the
    cost-aware (5 bp per side) non-overlapping Sharpe ratio of the sign strategy;
  * the TPE sampler is seeded.
Search space, number of trials (50), folds (3 walk-forward folds inside the first 75 % of the
data, never touching the test block), pruner and per-fold training budget are unchanged.

Usage:  python scripts/run_hpo.py composite|mse
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.backtest import backtest, directional_accuracy  # noqa: E402
from aft.data import (ALL_FEATURES, HORIZON, MAX_LOOKBACK, SplitData, load_master_dataset,  # noqa: E402
                      rf_top_features, walk_forward_folds)
from aft.experiment import build_neural  # noqa: E402
from aft.train import TrainConfig, predict, set_seed, train_model  # noqa: E402

OBJECTIVE = sys.argv[1] if len(sys.argv) > 1 else "composite"
assert OBJECTIVE in ("composite", "mse")
N_TRIALS = 50
SEED = 42
OUT = ROOT / "results" / "hpo"
OUT.mkdir(parents=True, exist_ok=True)

df = load_master_dataset()
X_all = df[ALL_FEATURES].to_numpy(np.float64)
y_all = df["Future_Return"].to_numpy(np.float64)
FOLDS = []
for start, tr_end, va_end in walk_forward_folds(len(df)):
    scaler = StandardScaler().fit(X_all[start:tr_end])
    Z = scaler.transform(X_all[:va_end])
    tr_idx = np.arange(max(start, MAX_LOOKBACK - 1), tr_end - HORIZON)
    va_idx = np.arange(tr_end, va_end)
    sd = SplitData(df.iloc[:va_end], list(ALL_FEATURES), Z, y_all[:va_end], tr_idx, va_idx,
                   np.array([], dtype=int), tr_end, va_end, scaler)
    top40 = rf_top_features(Z, y_all[:va_end], tr_idx, list(ALL_FEATURES), k=40, seed=SEED)
    FOLDS.append((sd, top40))
    print(f"fold rows {start}-{tr_end}-{va_end}: train {len(tr_idx)} val {len(va_idx)} "
          f"({df.Date[tr_idx[0]].date()} .. {df.Date[va_idx[-1]].date()})", flush=True)


def evaluate_fold(p: dict, sd: SplitData, top40: list[str]) -> dict:
    feats = top40 if p["use_top40"] else list(ALL_FEATURES)
    sdx = sd.subset(feats) if p["use_top40"] else sd
    L = p["seq_len"]
    Xtr, Xva = sdx.windows(sdx.train_idx, L), sdx.windows(sdx.val_idx, L)
    ytr, yva = sdx.y[sdx.train_idx], sdx.y[sdx.val_idx]
    set_seed(SEED)
    model = build_neural("AFT", len(feats), feats, p)
    cfg = TrainConfig(lr=p["lr"], weight_decay=p["wd"], max_epochs=25, patience=5)
    res = train_model(model, Xtr, ytr, Xva, yva, cfg, SEED)
    pred = predict(res.model, Xva)
    mae = float(np.mean(np.abs(pred - yva)))
    da = directional_accuracy(yva, pred)
    sr = backtest(yva, pred, cost_bps=5.0)["sharpe"]
    return {"mae": mae, "mae_zero": float(np.mean(np.abs(yva))), "mse": float(np.mean((pred - yva) ** 2)),
            "da": da, "sharpe": sr,
            "score": 0.40 * mae / float(np.mean(np.abs(yva))) - 0.30 * da / 0.5 - 0.30 * sr}


def objective(trial: optuna.Trial) -> float:
    p = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "seq_len": trial.suggest_categorical("seq_len", [20, 40, 60, 80]),
        "d_model": trial.suggest_categorical("d_model", [64, 96, 128, 160, 192]),
    }
    ffm = trial.suggest_categorical("ff_multiplier", [2, 4])
    p["ff_dim"] = ffm * p["d_model"]
    p["num_heads"] = trial.suggest_categorical("num_heads", [4, 8])
    p["num_layers"] = trial.suggest_categorical("num_layers", [2, 3])
    p["use_top40"] = trial.suggest_categorical("use_top40", [False, True])
    results = []
    for k, (sd, top40) in enumerate(FOLDS):
        r = evaluate_fold(p, sd, top40)
        results.append(r)
        for key, val in r.items():
            trial.set_user_attr(f"fold{k}_{key}", val)
        running = float(np.mean([x["score" if OBJECTIVE == "composite" else "mse"] for x in results]))
        trial.report(running, k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    for key in ("mae", "da", "sharpe", "mse", "score"):
        trial.set_user_attr(f"mean_{key}", float(np.mean([x[key] for x in results])))
    return float(np.mean([x["score" if OBJECTIVE == "composite" else "mse"] for x in results]))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))

    def cb(st, tr):
        print(f"trial {tr.number:>2} {tr.state.name:<8} value={tr.value if tr.value is not None else float('nan'):.5f} "
              f"params={json.dumps(tr.params)}", flush=True)

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])
    study.trials_dataframe().to_csv(OUT / f"trials_{OBJECTIVE}.csv", index=False)
    best = study.best_trial
    (OUT / f"best_{OBJECTIVE}.json").write_text(json.dumps({"trial": best.number, "value": best.value,
                                                            "params": best.params,
                                                            "user_attrs": best.user_attrs}, indent=2))
    print("BEST", best.number, best.value, best.params)
