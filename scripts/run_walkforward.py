"""Generate walk-forward out-of-sample forecasts of every base model for every asset.

Usage:  python scripts/run_walkforward.py [TICKER ...]
Output: results/walkforward/<TICKER>/block<k>/preds/*.parquet   (split == "test" is the OOS block)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.data import ALL_FEATURES, build_features, download_ohlcv, load_master_dataset  # noqa: E402
from aft.experiment import AFT_BASE, Store, run_naive, run_neural, run_tabular  # noqa: E402
from aft.walkforward import block_split, blocks  # noqa: E402

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "JPM", "XOM", "JNJ", "WMT", "SPY"]
SEEDS = [42, 101, 2023]
NEURAL = ("LSTM", "GRU", "DLinear", "Transformer", "AFT")

tickers = sys.argv[1:] or TICKERS
for t in tickers:
    # AAPL uses the exact master dataset of the main study; other assets the identical pipeline.
    df = load_master_dataset() if t == "AAPL" else build_features(download_ohlcv(t))
    for k, (s, e) in enumerate(blocks(len(df))):
        t0 = time.perf_counter()
        store = Store(ROOT / "results" / "walkforward" / t / f"block{k}")
        sd = block_split(df, list(ALL_FEATURES), s, e)
        run_naive(store, sd, asset=t)
        run_tabular(store, sd, seeds=SEEDS, asset=t, which=("Ridge", "XGBoost"))
        for kind in NEURAL:
            run_neural(store, kind, kind, sd, AFT_BASE, seeds=SEEDS, asset=t, verbose=False)
        print(f"{t} block {k} rows {s}-{e} ({df.Date.iloc[s].date()}..{df.Date.iloc[e - 1].date()}) "
              f"train {len(sd.train_idx)} val {len(sd.val_idx)} test {len(sd.test_idx)}  "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
print("done", flush=True)
