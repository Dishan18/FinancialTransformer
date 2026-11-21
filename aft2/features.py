"""Feature engineering v2: stationary, scale-free panel features.

Design principles (motivated by the v1 diagnostics, where price-level features drifted far outside
their training range):
  * every feature is a return, a ratio of prices, a volatility, a z-score or a bounded oscillator;
    no feature is a price, volume or dollar *level*;
  * every feature at date t uses only data up to and including the close of t (checked by
    ``scripts/v2_lookahead_test.py``);
  * stock-level features are later normalised by their cross-sectional rank on each date, which
    makes them comparable across stocks and across time; market-state features are shared by all
    stocks on a date and carry the time-series (regime) information.

Output of ``compute_panel``:  dict  feature name -> DataFrame (dates x tickers), plus targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import FIRST_VALID_DATE, SECTOR_OF

HORIZON = 5
ANN = np.sqrt(252.0)

# feature name -> group (order defines the column order of the panel tensor)
STOCK_GROUPS: dict[str, list[str]] = {
    "returns": ["ret_1", "ret_5", "ret_21", "ret_63", "ret_126", "mom_12_1", "ret_5_vn", "ret_21_vn"],
    "volatility": ["vol_5", "vol_21", "vol_63", "vol_ratio", "park_21", "gk_21", "dvol_21"],
    "range": ["hl_range", "clv", "gap", "intraday"],
    "trend": ["ma_gap_5", "ma_gap_21", "ma_gap_63", "ma_gap_200", "macd_n", "macd_hist_n", "rsi_14", "boll_z"],
    "position": ["dist_high_252", "dist_low_252", "dist_high_21", "dist_low_21", "range_pos_21"],
    "liquidity": ["vlm_z_63", "rel_volume_21", "dollar_vol", "amihud_21", "vlm_ret_corr_21"],
    "moments": ["skew_63", "kurt_63", "max_ret_21", "min_ret_21"],
    "market_rel": ["beta_63", "idio_vol_63", "corr_mkt_63", "ret_21_xmkt"],
    "sector_rel": ["ret_5_xsec", "ret_21_xsec", "ret_63_xsec"],
}
MARKET_GROUPS: dict[str, list[str]] = {
    "market": ["mkt_ret_1", "mkt_ret_5", "mkt_ret_21", "mkt_ret_63", "mkt_vol_21", "mkt_ma_gap_200",
               "vix_log", "vix_chg_5", "vrp", "cs_disp_21", "breadth_200"],
    "calendar": ["dow_sin", "dow_cos", "month_sin", "month_cos"],
}
STOCK_FEATURES = [f for g in STOCK_GROUPS.values() for f in g]
MARKET_FEATURES = [f for g in MARKET_GROUPS.values() for f in g]
ALL_FEATURES = STOCK_FEATURES + MARKET_FEATURES
GROUPS = {**STOCK_GROUPS, **MARKET_GROUPS}


def _wide(raw: dict[str, pd.DataFrame], col: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({t: d.set_index("Date")[col] for t, d in raw.items()}).reindex(calendar)


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def compute_panel(stocks: dict[str, pd.DataFrame], spy: pd.DataFrame, vix: pd.DataFrame,
                  end: str | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Original 100-stock interface: (features, y_abs, y_rel) for h = 5, market features replicated."""
    out = compute_panel_ext(stocks, spy, vix, end=end)
    return out["F"], out["targets"][HORIZON][0], out["targets"][HORIZON][1]


def compute_panel_ext(stocks: dict[str, pd.DataFrame], spy: pd.DataFrame, vix: pd.DataFrame,
                      end: str | None = None, sector_of: dict[str, str] | None = None,
                      eligible_from: dict[str, str] | None = None, horizons=(HORIZON,),
                      replicate_market: bool = True, as_float32: bool = False) -> dict:
    """Compute all features and targets.

    ``end`` truncates every input at that date (used by the look-ahead test).
    ``eligible_from``: ticker -> first date of index membership.  Stock features and targets of a
    stock are set to NaN before that date, *after* its own time-series features have been computed
    from its full price history (prices before inclusion are public), so cross-sectional quantities
    (ranks, sector-relative returns, breadth, dispersion, the cross-sectional mean of the target) only
    involve stocks that are in the universe on that date.
    Returns {"F": stock (+ replicated market) features, "M": market frame, "targets": {h: (y_abs, y_rel)},
             "eligible": DataFrame of bools}.
    """
    sector_of = sector_of or SECTOR_OF
    if end is not None:
        stocks = {t: d[d.Date <= end] for t, d in stocks.items()}
        spy, vix = spy[spy.Date <= end], vix[vix.Date <= end]
    cal = pd.DatetimeIndex(spy.Date)
    tick = list(stocks)
    O, H, L, C, V = (_wide(stocks, c, cal) for c in ("Open", "High", "Low", "Close", "Volume"))
    for t, first in FIRST_VALID_DATE.items():
        if t in C:
            for frame in (O, H, L, C, V):
                frame.loc[frame.index < pd.Timestamp(first), t] = np.nan
    lr = np.log(C).diff()
    s = spy.set_index("Date").reindex(cal)
    m_lr = np.log(s.Close).diff()
    vix_c = vix.set_index("Date").Close.reindex(cal).ffill(limit=3)
    F: dict[str, pd.DataFrame] = {}

    def rolling_std(x, n):
        return x.rolling(n, min_periods=n).std()

    # -- returns / momentum
    logc = np.log(C)
    F["ret_1"] = lr
    for n in (5, 21, 63, 126):
        F[f"ret_{n}"] = logc - logc.shift(n)
    F["mom_12_1"] = logc.shift(21) - logc.shift(252)
    vol63d = rolling_std(lr, 63)
    F["ret_5_vn"] = F["ret_5"] / (vol63d * np.sqrt(5))
    F["ret_21_vn"] = F["ret_21"] / (vol63d * np.sqrt(21))
    # -- volatility
    for n in (5, 21, 63):
        F[f"vol_{n}"] = rolling_std(lr, n) * ANN
    F["vol_ratio"] = np.log(F["vol_5"] / F["vol_63"])
    hl = np.log(H / L)
    F["park_21"] = np.sqrt((hl ** 2).rolling(21, min_periods=21).mean() / (4 * np.log(2))) * ANN
    co = np.log(C / O)
    gk = (0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).rolling(21, min_periods=21).mean()
    F["gk_21"] = np.sqrt(gk.clip(lower=0)) * ANN
    F["dvol_21"] = np.sqrt((lr.clip(upper=0) ** 2).rolling(21, min_periods=21).mean()) * ANN
    # -- intraday range
    F["hl_range"] = hl
    rng = (H - L)
    F["clv"] = ((C - L) / rng.where(rng > 0)).fillna(0.5).where(C.notna())
    F["gap"] = np.log(O / C.shift(1))
    F["intraday"] = co
    # -- trend / mean reversion
    for n in (5, 21, 63, 200):
        F[f"ma_gap_{n}"] = np.log(C / C.rolling(n, min_periods=n).mean())
    ema12 = C.ewm(span=12, adjust=False, min_periods=26).mean()
    ema26 = C.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    F["macd_n"] = macd / C
    F["macd_hist_n"] = (macd - sig) / C
    F["rsi_14"] = _rsi(C) / 100 - 0.5
    ma20, sd20 = C.rolling(20, min_periods=20).mean(), C.rolling(20, min_periods=20).std()
    F["boll_z"] = (C - ma20) / sd20
    # -- position in range
    F["dist_high_252"] = np.log(C / H.rolling(252, min_periods=252).max())
    F["dist_low_252"] = np.log(C / L.rolling(252, min_periods=252).min())
    h21, l21 = H.rolling(21, min_periods=21).max(), L.rolling(21, min_periods=21).min()
    F["dist_high_21"] = np.log(C / h21)
    F["dist_low_21"] = np.log(C / l21)
    F["range_pos_21"] = (C - l21) / (h21 - l21).where(h21 > l21)
    # -- volume / liquidity
    lv = np.log(V.where(V > 0))
    F["vlm_z_63"] = (lv - lv.rolling(63, min_periods=63).mean()) / lv.rolling(63, min_periods=63).std()
    F["rel_volume_21"] = lv - np.log(V.rolling(21, min_periods=21).mean())
    dv = C * V
    F["dollar_vol"] = np.log(dv.rolling(21, min_periods=21).mean())      # used only through its rank
    F["amihud_21"] = np.log((lr.abs() / dv.where(dv > 0)).rolling(21, min_periods=21).mean() + 1e-15)
    dlv = lv.diff()
    F["vlm_ret_corr_21"] = lr.rolling(21, min_periods=21).corr(dlv)
    # -- higher moments / lottery
    F["skew_63"] = lr.rolling(63, min_periods=63).skew()
    F["kurt_63"] = lr.rolling(63, min_periods=63).kurt()
    F["max_ret_21"] = lr.rolling(21, min_periods=21).max()
    F["min_ret_21"] = lr.rolling(21, min_periods=21).min()
    # -- market-relative
    mk = pd.DataFrame({t: m_lr for t in tick}, index=cal)
    mean_x = lr.rolling(63, min_periods=63).mean()
    mean_m = mk.rolling(63, min_periods=63).mean()
    cov = (lr * mk).rolling(63, min_periods=63).mean() - mean_x * mean_m
    var_m = (mk ** 2).rolling(63, min_periods=63).mean() - mean_m ** 2
    beta = cov / var_m
    F["beta_63"] = beta
    resid = lr - beta * mk          # residual with the trailing beta (no look-ahead)
    F["idio_vol_63"] = rolling_std(resid, 63) * ANN
    F["corr_mkt_63"] = lr.rolling(63, min_periods=63).corr(mk)
    m_ret21 = np.log(s.Close / s.Close.shift(21))
    F["ret_21_xmkt"] = F["ret_21"].sub(m_ret21, axis=0)
    del mk, mean_x, mean_m, cov, var_m, beta, resid
    # -- point-in-time universe: mask stock features before index inclusion
    if eligible_from is not None:
        added = pd.Series({t: pd.Timestamp(eligible_from.get(t, "1900-01-01")) for t in tick})
        eligible = pd.DataFrame(cal.to_numpy()[:, None] >= added.to_numpy()[None, :], index=cal, columns=tick)
        eligible &= C.notna()
    else:
        eligible = C.notna()
    if eligible_from is not None or as_float32:
        for k in list(F):
            F[k] = F[k].where(eligible)
            if as_float32:
                F[k] = F[k].astype(np.float32)
    # -- sector-relative (leave-one-out equal-weighted sector mean)
    for n in (5, 21, 63):
        r = F["ret_1"] if n == 1 else F[f"ret_{n}"]
        rel = pd.DataFrame(index=cal, columns=tick, dtype=float)
        for sec in set(sector_of[t] for t in tick):
            cols = [t for t in tick if sector_of[t] == sec]
            sub = r[cols]
            cnt = sub.notna().sum(axis=1)
            tot = sub.sum(axis=1, min_count=1)
            for c in cols:
                loo = (tot - sub[c]) / (cnt - 1)
                rel[c] = sub[c] - loo.where(cnt > 1)
        F[f"ret_{n}_xsec"] = rel
    # -- market state (one value per date, broadcast to all stocks later)
    M = pd.DataFrame(index=cal)
    M["mkt_ret_1"] = m_lr
    for n in (5, 21, 63):
        M[f"mkt_ret_{n}"] = np.log(s.Close / s.Close.shift(n))
    M["mkt_vol_21"] = m_lr.rolling(21, min_periods=21).std() * ANN
    M["mkt_ma_gap_200"] = np.log(s.Close / s.Close.rolling(200, min_periods=200).mean())
    M["vix_log"] = np.log(vix_c)
    M["vix_chg_5"] = np.log(vix_c / vix_c.shift(5))
    M["vrp"] = vix_c / 100 - M["mkt_vol_21"]
    M["cs_disp_21"] = F["ret_21"].std(axis=1)
    above = (F["ma_gap_200"] > 0).astype(float).where(F["ma_gap_200"].notna())
    M["breadth_200"] = above.mean(axis=1)
    dow = cal.dayofweek.to_numpy()
    mon = cal.month.to_numpy()
    M["dow_sin"], M["dow_cos"] = np.sin(2 * np.pi * dow / 5), np.cos(2 * np.pi * dow / 5)
    M["month_sin"], M["month_cos"] = np.sin(2 * np.pi * (mon - 1) / 12), np.cos(2 * np.pi * (mon - 1) / 12)
    if replicate_market:
        for f in MARKET_FEATURES:
            F[f] = pd.DataFrame({t: M[f] for t in tick}, index=cal).where(C.notna())
    # -- targets (cross-sectional mean over the stocks in the universe on each date)
    targets = {}
    for h in horizons:
        y_abs = (C.shift(-h) / C - 1.0).where(eligible)
        targets[h] = (y_abs, y_abs.sub(y_abs.mean(axis=1), axis=0))
    for k in F:
        F[k] = F[k].replace([np.inf, -np.inf], np.nan)
    M = M.replace([np.inf, -np.inf], np.nan)
    expected = set(ALL_FEATURES) if replicate_market else set(STOCK_FEATURES)
    assert set(F) == expected, set(F) ^ expected
    return {"F": F, "M": M, "targets": targets, "eligible": eligible}


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank each date's cross-section and map it to (-1, 1): (rank - 0.5)/n * 2 - 1."""
    n = df.notna().sum(axis=1)
    r = df.rank(axis=1, method="average")
    return ((r - 0.5).div(n, axis=0) * 2 - 1)
