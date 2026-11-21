"""Does the hyper-parameter search select configurations that are better out of sample?

Every COMPLETE trial of a search is retrained on the main training split (seed 42, same protocol
as the main study) and evaluated on the main validation and test blocks.  The Spearman rank
correlation between the trial's search objective and its out-of-sample metrics measures whether
selection on the walk-forward folds transfers.  This is a post-hoc analysis: test results are
never used to choose a model.

Usage:  python scripts/hpo_transfer.py composite|mse
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.backtest import backtest, directional_accuracy  # noqa: E402
from aft.data import load_master_dataset, make_splits  # noqa: E402
from aft.experiment import build_neural, params_from_optuna  # noqa: E402
from aft.train import TrainConfig, predict, set_seed, train_model  # noqa: E402

obj = sys.argv[1] if len(sys.argv) > 1 else "composite"
OUT = ROOT / "results" / "hpo"
trials = pd.read_csv(OUT / f"trials_{obj}.csv")
trials = trials[trials.state == "COMPLETE"]
df = load_master_dataset()
sd = make_splits(df)
top40 = json.loads((ROOT / "results" / "aapl" / "top40.json").read_text())
out_path = OUT / f"transfer_{obj}.csv"
done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
rows = done.to_dict("records")
for _, tr in trials.iterrows():
    if len(done) and int(tr.number) in set(done.trial):
        continue
    p = params_from_optuna({k.replace("params_", ""): tr[k] for k in tr.index if k.startswith("params_")})
    p["seq_len"], p["d_model"], p["num_heads"], p["num_layers"] = (int(p["seq_len"]), int(p["d_model"]),
                                                                    int(p["num_heads"]), int(p["num_layers"]))
    p["ff_dim"] = int(p["ff_dim"])
    p["use_top40"] = bool(p["use_top40"])
    feats = top40 if p["use_top40"] else list(sd.features)
    sdx = sd.subset(feats) if p["use_top40"] else sd
    L = p["seq_len"]
    set_seed(42)
    model = build_neural("AFT", len(feats), feats, p)
    res = train_model(model, sdx.windows(sdx.train_idx, L), sdx.y[sdx.train_idx], sdx.windows(sdx.val_idx, L),
                      sdx.y[sdx.val_idx], TrainConfig(lr=p["lr"], weight_decay=p["wd"]), 42)
    row = {"trial": int(tr.number), "search_value": float(tr.value)}
    for split, idx in (("val", sdx.val_idx), ("test", sdx.test_idx)):
        pred = predict(res.model, sdx.windows(idx, L))
        y = sdx.y[idx]
        row[f"{split}_mae_ratio"] = float(np.mean(np.abs(pred - y)) / np.mean(np.abs(y)))
        row[f"{split}_da"] = directional_accuracy(y, pred)
        row[f"{split}_sharpe"] = backtest(y, pred, 5.0)["sharpe"]
        row[f"{split}_mse"] = float(np.mean((pred - y) ** 2))
    rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(row, flush=True)

R = pd.DataFrame(rows)
summary = {}
for col in ("val_sharpe", "test_sharpe", "test_mae_ratio", "test_da", "val_mse", "test_mse"):
    rho, p = spearmanr(R.search_value, R[col])
    summary[col] = {"rho": float(rho), "p": float(p)}
summary["n_trials"] = int(len(R))
(OUT / f"transfer_{obj}_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
