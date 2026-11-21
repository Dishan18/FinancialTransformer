"""Data pipeline: feature engineering, purged chronological splits and windowing.

Alignment convention (used everywhere in this package)
------------------------------------------------------
Row ``t`` of the feature table contains only information available at the close of
trading day ``t``.  The target of row ``t`` is the forward 5-day simple return

    y_t = (C_{t+5} - C_t) / C_t .

A model with look-back ``L`` receives the window of rows ``t-L+1, ..., t`` (inclusive)
and predicts ``y_t``.  Windows may draw *features* from rows that belong to an earlier
split (those prices are already observed at time ``t``), so every model, whatever its
look-back, is evaluated on exactly the same forecast-origin dates.

Purging
-------
The last ``h`` samples of the training split and of the validation split are removed,
because their targets are realised with prices from the following split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 5
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
MAX_LOOKBACK = 80  # largest look-back in the search space; fixes the first usable sample

FEATURE_GROUPS: dict[str, list[str]] = {
    "price": ["Close", "High", "Low", "Open", "Previous_Close", "Gap", "Price_Change"],
    "returns": ["Daily_Return", "Return", "Log_Return", "ROC_5", "ROC_10"],
    "volatility": ["Rolling_Volatility", "ATR", "Historical_Volatility", "Parkinson_Volatility",
                   "Garman_Klass", "Rolling_STD", "Rolling_Variance", "Rolling_Return_STD"],
    "trend": ["EMA_9", "EMA_21", "EMA_50", "EMA_200", "EMA_12", "EMA_26", "MACD", "Signal",
              "MACD_Histogram"],
    "momentum": ["Momentum_5", "Momentum_10", "Momentum_20"],
    "volume": ["Volume", "Volume_MA_5", "Volume_MA_20", "Relative_Volume", "Volume_Change",
               "Volume_Momentum", "OBV", "VWAP"],
    "candlestick": ["Body", "Upper_Wick", "Lower_Wick", "Full_Range", "Body_Ratio",
                    "Upper_Wick_Ratio", "Lower_Wick_Ratio", "Body_to_Wick", "High_Low_Range",
                    "Open_Close_Range", "True_Range"],
    "statistics": ["Rolling_Mean", "Rolling_Min", "Rolling_Max", "Rolling_Median", "Rolling_Skew",
                   "Rolling_Kurtosis", "Rolling_Zscore", "Rolling_Max_Return", "Rolling_Min_Return"],
    "lags": ["Close_Lag_1", "Return_Lag_1", "Volume_Lag_1", "Close_Lag_2", "Return_Lag_2",
             "Volume_Lag_2", "Close_Lag_3", "Return_Lag_3", "Volume_Lag_3", "Close_Lag_5",
             "Return_Lag_5", "Volume_Lag_5", "Close_Lag_10", "Return_Lag_10", "Volume_Lag_10"],
    "breakout": ["Rolling_High_5", "Rolling_Low_5", "Rolling_High_10", "Rolling_Low_10",
                 "Rolling_High_20", "Rolling_Low_20", "Distance_From_High_5", "Distance_From_Low_5",
                 "Distance_From_High_10", "Distance_From_Low_10", "Distance_From_High_20",
                 "Distance_From_Low_20", "Range_Position", "Breakout_20", "Breakdown_20"],
    "calendar": ["Day", "Month", "Quarter", "DayOfWeek", "WeekOfYear"],
}
ALL_FEATURES: list[str] = [c for cols in FEATURE_GROUPS.values() for c in cols]
assert len(ALL_FEATURES) == 95 and len(set(ALL_FEATURES)) == 95

# Features whose scale follows the price or cumulative-volume level (non-stationary).
LEVEL_FEATURES: list[str] = [
    "Close", "High", "Low", "Open", "Previous_Close", "Gap", "Price_Change",
    "Rolling_Mean", "Rolling_Min", "Rolling_Max", "Rolling_Median", "Rolling_STD", "Rolling_Variance",
    "ATR", "EMA_9", "EMA_21", "EMA_50", "EMA_200", "EMA_12", "EMA_26", "MACD", "Signal",
    "MACD_Histogram", "Momentum_5", "Momentum_10", "Momentum_20", "OBV", "VWAP",
    "Body", "Upper_Wick", "Lower_Wick", "Full_Range", "High_Low_Range", "Open_Close_Range",
    "True_Range", "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5", "Close_Lag_10",
    "Rolling_High_5", "Rolling_Low_5", "Rolling_High_10", "Rolling_Low_10", "Rolling_High_20",
    "Rolling_Low_20", "Distance_From_High_20", "Distance_From_Low_20",
]


def build_features(raw: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Replicates the feature engineering of ``01_feature_engineering.ipynb`` exactly.

    ``raw`` must contain columns Date, Close, High, Low, Open, Volume (auto-adjusted prices),
    sorted by date with a default RangeIndex.
    """
    df = raw[["Date", "Close", "High", "Low", "Open", "Volume"]].copy().reset_index(drop=True)
    df["Daily_Return"] = df["Close"].pct_change()
    w = 20
    df["Rolling_Volatility"] = df["Daily_Return"].rolling(w).std()
    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Price_Change"] = df["Close"] - df["Open"]
    df["Previous_Close"] = df["Close"].shift(1)
    df["Gap"] = df["Open"] - df["Previous_Close"]
    df["High_Low_Range"] = df["High"] - df["Low"]
    df["Open_Close_Range"] = (df["Close"] - df["Open"]).abs()
    df["True_Range"] = np.maximum.reduce([
        (df["High"] - df["Low"]).to_numpy(),
        (df["High"] - df["Previous_Close"]).abs().to_numpy(),
        (df["Low"] - df["Previous_Close"]).abs().to_numpy(),
    ])
    df["Rolling_Mean"] = df["Close"].rolling(w).mean()
    df["Rolling_STD"] = df["Close"].rolling(w).std()
    df["Rolling_Min"] = df["Close"].rolling(w).min()
    df["Rolling_Max"] = df["Close"].rolling(w).max()
    df["Rolling_Median"] = df["Close"].rolling(w).median()
    df["Rolling_Variance"] = df["Close"].rolling(w).var()
    for k in (5, 10, 20):
        df[f"Momentum_{k}"] = df["Close"] - df["Close"].shift(k)
    df["ROC_5"] = df["Close"] / df["Close"].shift(5) - 1
    df["ROC_10"] = df["Close"] / df["Close"].shift(10) - 1
    df["Volume_MA_5"] = df["Volume"].rolling(5).mean()
    df["Volume_MA_20"] = df["Volume"].rolling(20).mean()
    df["Relative_Volume"] = df["Volume"] / df["Volume_MA_20"]
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_Momentum"] = df["Volume"] - df["Volume"].shift(5)
    # On-balance volume, identical to the explicit loop of Notebook 01.
    direction = np.sign(df["Close"].diff()).fillna(0.0)
    df["OBV"] = (direction * df["Volume"]).cumsum()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    eps = 1e-8
    df["Body"] = (df["Close"] - df["Open"]).abs()
    df["Upper_Wick"] = df["High"] - np.maximum(df["Open"], df["Close"])
    df["Lower_Wick"] = np.minimum(df["Open"], df["Close"]) - df["Low"]
    df["Full_Range"] = df["High"] - df["Low"]
    df["Body_Ratio"] = df["Body"] / (df["Full_Range"] + eps)
    df["Upper_Wick_Ratio"] = df["Upper_Wick"] / (df["Full_Range"] + eps)
    df["Lower_Wick_Ratio"] = df["Lower_Wick"] / (df["Full_Range"] + eps)
    df["Body_to_Wick"] = df["Body"] / (df["Upper_Wick"] + df["Lower_Wick"] + eps)
    df["ATR"] = df["True_Range"].rolling(14).mean()
    df["Historical_Volatility"] = df["Log_Return"].rolling(20).std() * np.sqrt(252)
    hl = np.log(df["High"] / df["Low"])
    df["Parkinson_Volatility"] = np.sqrt(hl.pow(2).rolling(20).mean() / (4 * np.log(2)))
    log_co = np.log(df["Close"] / df["Open"])
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    df["Garman_Klass"] = np.sqrt(gk.rolling(20).mean())
    for span in (9, 21, 50, 200):
        df[f"EMA_{span}"] = df["Close"].ewm(span=span).mean()
    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal"]
    df["Rolling_Skew"] = df["Return"].rolling(w).skew()
    df["Rolling_Kurtosis"] = df["Return"].rolling(w).kurt()
    df["Rolling_Zscore"] = (df["Close"] - df["Rolling_Mean"]) / (df["Rolling_STD"] + 1e-8)
    df["Rolling_Max_Return"] = df["Return"].rolling(w).max()
    df["Rolling_Min_Return"] = df["Return"].rolling(w).min()
    df["Rolling_Return_STD"] = df["Return"].rolling(w).std()
    df["Future_Close"] = df["Close"].shift(-horizon)
    df["Future_Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]
    df["Target_Direction"] = (df["Future_Return"] > 0).astype(int)
    df = df.dropna().reset_index(drop=True)
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    for lag in (1, 2, 3, 5, 10):
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)
        df[f"Volume_Lag_{lag}"] = df["Volume"].shift(lag)
    for win in (5, 10, 20):
        df[f"Rolling_High_{win}"] = df["High"].rolling(win).max()
        df[f"Rolling_Low_{win}"] = df["Low"].rolling(win).min()
        df[f"Distance_From_High_{win}"] = df["Close"] / df[f"Rolling_High_{win}"]
        df[f"Distance_From_Low_{win}"] = df["Close"] / df[f"Rolling_Low_{win}"]
    # Notebook 01 (cell 66) then overwrites the 20-day distances with differences.
    rh = df["High"].rolling(20).max()
    rl = df["Low"].rolling(20).min()
    df["Range_Position"] = (df["Close"] - rl) / (rh - rl + 1e-8)
    df["Distance_From_High_20"] = rh - df["Close"]
    df["Distance_From_Low_20"] = df["Close"] - rl
    df["Breakout_20"] = (df["Close"] > rh.shift(1)).astype(int)
    df["Breakdown_20"] = (df["Close"] < rl.shift(1)).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df


def load_master_dataset() -> pd.DataFrame:
    """The AAPL dataset produced by Notebook 01 (2015-03-02 .. 2025-12-22, 2,721 rows)."""
    df = pd.read_parquet(ROOT / "processed_data" / "master_dataset.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def download_ohlcv(ticker: str, start: str = "2015-01-01", end: str = "2025-12-31",
                   cache_dir: Path | None = None) -> pd.DataFrame:
    """Daily auto-adjusted OHLCV from Yahoo Finance, cached as CSV for reproducibility."""
    cache_dir = cache_dir or (ROOT / "data" / "raw")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ticker}_{start}_{end}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"])
        return df
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False,
                     group_by="column")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Close", "High", "Low", "Open", "Volume"]]
    df.columns.name = None
    if len(df) == 0:
        raise RuntimeError(f"No data downloaded for {ticker}")
    df.to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["Date"])


@dataclass
class SplitData:
    """Scaled feature matrix plus purged sample indices for train / val / test."""
    df: pd.DataFrame
    features: list[str]
    Z: np.ndarray                 # (n_rows, n_features) standardised with train statistics
    y: np.ndarray                 # (n_rows,) forward 5-day return
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_end: int
    val_end: int
    scaler: StandardScaler = field(repr=False)

    @property
    def dates(self) -> pd.Series:
        return self.df["Date"]

    def windows(self, idx: np.ndarray, lookback: int) -> np.ndarray:
        """Return windows of shape (len(idx), lookback, n_features) ending at each index."""
        if idx.min() < lookback - 1:
            raise ValueError("window would start before the first row")
        view = np.lib.stride_tricks.sliding_window_view(self.Z, lookback, axis=0)
        # view[i] covers rows i .. i+lookback-1 and has shape (n_features, lookback)
        return np.ascontiguousarray(view[idx - lookback + 1].transpose(0, 2, 1), dtype=np.float32)

    def tabular(self, idx: np.ndarray) -> np.ndarray:
        return self.Z[idx].astype(np.float32)

    def subset(self, features: list[str]) -> "SplitData":
        """Same samples and scaling statistics, restricted to a subset of features."""
        cols = [self.features.index(f) for f in features]
        sc = StandardScaler()
        sc.mean_ = self.scaler.mean_[cols]
        sc.scale_ = self.scaler.scale_[cols]
        sc.var_ = self.scaler.var_[cols]
        sc.n_features_in_ = len(cols)
        return SplitData(self.df, list(features), self.Z[:, cols], self.y, self.train_idx,
                         self.val_idx, self.test_idx, self.train_end, self.val_end, sc)


def make_splits(df: pd.DataFrame, features: list[str] | None = None, horizon: int = HORIZON,
                train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO,
                max_lookback: int = MAX_LOOKBACK) -> SplitData:
    """Chronological 70/15/15 split with purging and a common first sample for every model."""
    features = list(features or ALL_FEATURES)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    X = df[features].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("non-finite feature values")
    scaler = StandardScaler().fit(X[:train_end])
    Z = scaler.transform(X)
    y = df["Future_Return"].to_numpy(dtype=np.float64)
    train_idx = np.arange(max_lookback - 1, train_end - horizon)
    val_idx = np.arange(train_end, val_end - horizon)
    test_idx = np.arange(val_end, n)
    return SplitData(df, features, Z, y, train_idx, val_idx, test_idx, train_end, val_end, scaler)


def walk_forward_folds(n_rows: int, n_folds: int = 3, fold_frac: float = 0.25,
                       train_frac: float = 0.70, val_frac: float = 0.15) -> list[tuple[int, int, int]]:
    """Fold boundaries used for hyper-parameter search (identical to Notebook 12.1).

    Returns a list of (fold_start, train_end, val_end) row indices.  All folds end before
    the start of the main validation/test periods' test block (row int(0.85 n)).
    """
    fold_size = int(n_rows * fold_frac)
    out = []
    for k in range(n_folds):
        start = k * fold_size
        out.append((start, start + int(fold_size * train_frac), start + int(fold_size * (train_frac + val_frac))))
    return out


def rf_top_features(Z: np.ndarray, y: np.ndarray, idx: np.ndarray, features: list[str], k: int = 40,
                    seed: int = 42, n_estimators: int = 50) -> list[str]:
    """Top-k features by random-forest impurity importance, fitted on the given (training) rows only."""
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    rf.fit(Z[idx], y[idx])
    order = np.argsort(-rf.feature_importances_, kind="stable")
    return [features[i] for i in order[:k]]


def group_indices(features: list[str]) -> dict[str, list[int]]:
    gi = {g: [features.index(c) for c in cols if c in features] for g, cols in FEATURE_GROUPS.items()}
    return {g: v for g, v in gi.items() if v}
