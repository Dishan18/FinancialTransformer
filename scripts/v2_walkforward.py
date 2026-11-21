"""Walk-forward (yearly refits, 2016-2025) out-of-sample forecasts of the v2 model zoo on the panel.

Usage:  python scripts/v2_walkforward.py MODEL [MODEL ...]
        MODEL in: Ridge XGBoost MLP LSTM GRU DLinear Transformer AFT AFT-R
Output: results_v2/wf/<MODEL>/fold<k>_seed<s>.parquet  (columns t, i, y, y_abs, y_hat)
        results_v2/wf/<MODEL>/fold<k>_seed<s>.json     (training metadata)
All hyper-parameters are fixed a priori (no tuning beyond early stopping on validation rank IC).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.train import DEVICE, set_seed  # noqa: E402
from aft2.models import SEQUENCE_KINDS, build, count_parameters  # noqa: E402
from aft2.panel import Panel, PanelTensors, group_indices, walk_forward  # noqa: E402
from aft2.protocol import DEV_LAST_YEAR, load_panel, wf_dir  # noqa: E402
from aft2.trainer import PanelTrainConfig, fit, mean_rank_ic, predict  # noqa: E402

L = 20
SEEDS = [42, 101, 2023]
OUT = ROOT / "results_v2" / "wf"
models = sys.argv[1:] or ["Ridge", "XGBoost", "MLP", "LSTM", "GRU", "DLinear", "Transformer", "AFT", "AFT-R"]
feature_mode, feature_set = "rank", "v2"
for a in list(models):
    if a.startswith("--mode="):
        feature_mode = a.split("=", 1)[1]
        models.remove(a)
    elif a.startswith("--features="):
        feature_set = a.split("=", 1)[1]
        models.remove(a)
loss, panel_name, horizon, stride = "mse", "top100", 5, 1
for a in list(models):
    if a.startswith("--stride="):
        stride = int(a.split("=", 1)[1])
        models.remove(a)
    elif a.startswith("--loss="):
        loss = a.split("=", 1)[1]
        models.remove(a)
    elif a.startswith("--panel="):
        panel_name = a.split("=", 1)[1]
        models.remove(a)
    elif a.startswith("--horizon="):
        horizon = int(a.split("=", 1)[1])
        models.remove(a)
tag = (("" if feature_mode == "rank" else f"_{feature_mode}") + ("" if feature_set == "v2" else f"_{feature_set}feat")
       + ("" if loss == "mse" else f"_{loss}") + ("" if stride == 1 else f"_s{stride}"))
OUT = wf_dir(panel_name, horizon)

if feature_set == "v1":
    from aft.data import ALL_FEATURES as V1_FEATURES, group_indices as v1_group_indices
    P = Panel.load(ROOT / "data" / "panel_v1feat.npz")
    R = P.ranked_stock_features(ROOT / "data" / "panel_v1feat_ranks.npy", stock_features=V1_FEATURES)
    TENSOR_KW = {"stock_features": V1_FEATURES, "market_features": []}
    group_indices = v1_group_indices  # noqa: F811
else:
    P, R = load_panel(panel_name, horizon)
    TENSOR_KW = {}
folds = walk_forward(P.dates, h=horizon)


def frame(PT, idx, pred):
    return pd.DataFrame({"t": idx[:, 0].astype(np.int32), "i": idx[:, 1].astype(np.int16),
                         "y": PT.y[idx[:, 0], idx[:, 1]], "y_abs": PT.y_abs[idx[:, 0], idx[:, 1]],
                         "y_hat": pred.astype(np.float32)})


for k, f in enumerate(folds):
    PT = PanelTensors(P, R, f.train_dates, DEVICE, feature_mode=feature_mode, **TENSOR_KW)
    tr, va, te = PT.samples(f.train_dates, L), PT.samples(f.val_dates, L), PT.samples(f.test_dates, L)
    if stride > 1:   # train on every stride-th date only (5-day targets on adjacent dates overlap heavily)
        tr = tr[tr[:, 0] % stride == 0]
    K = PT.K
    G = group_indices(PT.features)
    year = str(P.dates[f.test_dates[0]])[:4]
    for name in models:
        out_dir = OUT / (name + tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        seeds = [0] if name == "Ridge" else SEEDS
        for seed in seeds:
            stem = out_dir / f"fold{k}_seed{seed}"
            if stem.with_suffix(".parquet").exists():
                continue
            t0 = time.perf_counter()
            meta = {"model": name, "fold": k, "year": year, "seed": seed, "L": L, "feature_mode": feature_mode,
                    "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}
            if name == "Ridge":
                from sklearn.linear_model import Ridge
                Xtr = PT.current(torch.as_tensor(tr, device=DEVICE)).cpu().numpy(); ytr = PT.y[tr[:, 0], tr[:, 1]]
                Xva = PT.current(torch.as_tensor(va, device=DEVICE)).cpu().numpy(); yva = PT.y[va[:, 0], va[:, 1]]
                best = None
                for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
                    m = Ridge(alpha=alpha).fit(Xtr, ytr)
                    ic = mean_rank_ic(va[:, 0], yva, m.predict(Xva))
                    if best is None or ic > best[0]:
                        best = (ic, alpha, m)
                meta.update({"alpha": best[1], "val_ic": best[0], "params": K + 1})
                pred = best[2].predict(PT.current(torch.as_tensor(te, device=DEVICE)).cpu().numpy())
            elif name == "XGBoost":
                from xgboost import XGBRegressor
                Xtr = PT.current(torch.as_tensor(tr, device=DEVICE)).cpu().numpy(); ytr = PT.y[tr[:, 0], tr[:, 1]]
                lo, hi = np.quantile(ytr, [0.005, 0.995]); ytr = np.clip(ytr, lo, hi)
                Xva = PT.current(torch.as_tensor(va, device=DEVICE)).cpu().numpy(); yva = PT.y[va[:, 0], va[:, 1]]
                m = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=50, random_state=seed,
                                 tree_method="hist", device="cuda", early_stopping_rounds=50)
                m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                pred = m.predict(PT.current(torch.as_tensor(te, device=DEVICE)).cpu().numpy())
                meta.update({"best_iteration": int(m.best_iteration),
                             "val_ic": mean_rank_ic(va[:, 0], yva, m.predict(Xva))})
            elif name == "XGBoost-IC":
                # same booster, but the number of trees is chosen by validation rank IC (not MSE)
                from xgboost import XGBRegressor
                Xtr = PT.current(torch.as_tensor(tr, device=DEVICE)).cpu().numpy(); ytr = PT.y[tr[:, 0], tr[:, 1]]
                lo, hi = np.quantile(ytr, [0.005, 0.995]); ytr = np.clip(ytr, lo, hi)
                Xva = PT.current(torch.as_tensor(va, device=DEVICE)).cpu().numpy(); yva = PT.y[va[:, 0], va[:, 1]]
                m = XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=50, random_state=seed,
                                 tree_method="hist", device="cuda")
                m.fit(Xtr, ytr, verbose=False)
                grid = [25, 50, 100, 200, 300, 400]
                ics = [mean_rank_ic(va[:, 0], yva, m.predict(Xva, iteration_range=(0, n))) for n in grid]
                n_best = grid[int(np.argmax(ics))]
                pred = m.predict(PT.current(torch.as_tensor(te, device=DEVICE)).cpu().numpy(), iteration_range=(0, n_best))
                meta.update({"n_trees": n_best, "val_ic": float(max(ics)), "val_ic_grid": ics})
            else:
                set_seed(seed)
                model = build(name, K, G, L)
                seq = name in SEQUENCE_KINDS
                model, info = fit(model, PT, tr, va, L, seq, PanelTrainConfig(loss=loss), seed)
                pred = predict(model, PT, te, L, seq)
                info.pop("val_ic_hist", None)
                meta.update(info)
                meta["params"] = count_parameters(model)
                if name.startswith("AFT"):
                    import torch
                    torch.save({kk: v.cpu() for kk, v in model.state_dict().items()}, stem.with_suffix(".pt"))
            # Sealed holdout: test metrics of the holdout years are neither logged nor printed while
            # designs are being iterated (see aft2/protocol.py).
            dev = int(year) <= DEV_LAST_YEAR
            meta["test_ic"] = mean_rank_ic(te[:, 0], PT.y[te[:, 0], te[:, 1]], pred) if dev else None
            meta["wall_s"] = time.perf_counter() - t0
            frame(PT, te, pred).to_parquet(stem.with_suffix(".parquet"), index=False)
            stem.with_suffix(".json").write_text(json.dumps(meta, default=float, indent=1))
            shown = f"{meta['test_ic']:+.4f}" if dev else "sealed"
            print(f"{name:12s} fold {k} ({year}) seed {seed:<5} val IC {meta.get('val_ic', float('nan')):+.4f} "
                  f"OOS IC {shown}  {meta['wall_s']:6.1f}s", flush=True)
    del PT
print("done", flush=True)
