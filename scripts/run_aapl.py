"""AAPL main benchmark and ablation study (5 seeds, identical protocol for every model).

Usage:  python scripts/run_aapl.py base       # naive, tabular and neural baselines + AFT
        python scripts/run_aapl.py ablation   # AFT component ablations
        python scripts/run_aapl.py hpo        # AFT with re-tuned and with v1 hyper-parameters
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.data import load_master_dataset, make_splits  # noqa: E402
from aft.experiment import (AFT_BASE, Store, params_from_optuna, run_naive, run_neural,  # noqa: E402
                            run_tabular, top40_for)

what = sys.argv[1] if len(sys.argv) > 1 else "base"
store = Store(ROOT / "results" / "aapl")
df = load_master_dataset()
sd = make_splits(df)
top40_path = ROOT / "results" / "aapl" / "top40.json"
if top40_path.exists():
    top40 = json.loads(top40_path.read_text())
else:
    top40 = top40_for(sd)
    top40_path.write_text(json.dumps(top40, indent=1))
print(f"AAPL rows={len(df)} train={len(sd.train_idx)} val={len(sd.val_idx)} test={len(sd.test_idx)}", flush=True)

if what == "base":
    run_naive(store, sd)
    run_tabular(store, sd)
    for kind in ("LSTM", "GRU", "DLinear", "Transformer"):
        run_neural(store, kind, kind, sd, AFT_BASE)
    run_neural(store, "AFT", "AFT", sd, AFT_BASE)

elif what == "ablation":
    variants = {
        "AFT -regime/gate": ({"use_regime": False}, "mse", AFT_BASE),
        "AFT -fin. context": ({"use_financial_context": False}, "mse", AFT_BASE),
        "AFT -temporal bias": ({"use_temporal_bias": False}, "mse", AFT_BASE),
        "AFT -both biases": ({"use_financial_context": False, "use_temporal_bias": False}, "mse", AFT_BASE),
        "AFT +composite loss": ({}, "composite", AFT_BASE),
        "AFT top-40": ({}, "mse", {**AFT_BASE, "use_top40": True}),
        "AFT-R (renorm. gate)": ({"gate_renorm": True}, "mse", AFT_BASE),
    }
    for name, (variant, loss, p) in variants.items():
        run_neural(store, name, "AFT", sd, p, variant=variant, loss=loss, top40=top40)

elif what == "hpo":
    v1 = json.loads((ROOT / "experiments" / "best_hyperparams_financial.json").read_text())
    run_neural(store, "AFT (v1 tuned cfg)", "AFT", sd, params_from_optuna(v1), top40=top40)
    for obj, label in (("composite", "AFT-HPO (composite)"), ("mse", "AFT-HPO (MSE)")):
        path = ROOT / "results" / "hpo" / f"best_{obj}.json"
        if path.exists():
            best = json.loads(path.read_text())["params"]
            run_neural(store, label, "AFT", sd, params_from_optuna(best), top40=top40)
        else:
            print(f"missing {path}; skipped", flush=True)
else:
    raise SystemExit(f"unknown stage {what}")
print("done", what, flush=True)
