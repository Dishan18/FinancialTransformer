"""Multi-asset study: 11 assets (6 technology stocks, 4 stocks from other sectors, 1 ETF),
2015-2025, the same features, splits, training protocol, seeds and backtest as the AAPL study.
No per-asset tuning: every model uses the configuration it has in the AAPL study.

Usage:  python scripts/run_multiasset.py [TICKER ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.data import build_features, download_ohlcv, make_splits  # noqa: E402
from aft.experiment import (AFT_BASE, Store, params_from_optuna, run_naive, run_neural,  # noqa: E402
                            run_tabular, top40_for)

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "JPM", "XOM", "JNJ", "WMT", "SPY"]
tickers = sys.argv[1:] or TICKERS
hpo_path = ROOT / "results" / "hpo" / "best_composite.json"
hpo_params = params_from_optuna(json.loads(hpo_path.read_text())["params"]) if hpo_path.exists() else None

for t in tickers:
    store = Store(ROOT / "results" / "multiasset" / t)
    df = build_features(download_ohlcv(t))
    sd = make_splits(df)
    top40_file = store.root / "top40.json"
    if top40_file.exists():
        top40 = json.loads(top40_file.read_text())
    else:
        top40 = top40_for(sd)
        top40_file.write_text(json.dumps(top40, indent=1))
    print(f"== {t}: rows {len(df)}  test {df.Date.iloc[sd.test_idx[0]].date()}..{df.Date.iloc[sd.test_idx[-1]].date()}", flush=True)
    run_naive(store, sd, asset=t)
    run_tabular(store, sd, asset=t, which=("Ridge", "XGBoost"))
    for kind in ("LSTM", "DLinear", "Transformer"):
        run_neural(store, kind, kind, sd, AFT_BASE, asset=t)
    run_neural(store, "AFT", "AFT", sd, AFT_BASE, asset=t)
    if hpo_params is not None:
        run_neural(store, "AFT-HPO (composite)", "AFT", sd, hpo_params, asset=t, top40=top40)
print("done", flush=True)
