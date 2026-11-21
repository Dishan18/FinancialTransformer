"""Meta-model iteration on the development period (meta OOS years 2017-2022; 2016 is the first
meta-training year).  Base models and meta-models are compared on exactly the same (t, i) samples.

Usage:  python scripts/v2_meta.py [--models A,B,...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.stats import hac_mean_test  # noqa: E402
from aft2.evaluate import daily_ic, dm_vs, evaluate  # noqa: E402
from aft2.meta import H, build_meta_data  # noqa: E402
from aft2.panel import Panel  # noqa: E402
from aft2.protocol import DEV_LAST_YEAR, load_panel, wf_dir  # noqa: E402

OUT = ROOT / "results_v2" / "meta"
OUT.mkdir(parents=True, exist_ok=True)
FACTORS = {"mom": ("mom_12_1", +1), "rev5": ("ret_5", -1), "rev21": ("ret_21", -1), "lowvol": ("vol_63", -1),
           "max": ("max_ret_21", -1)}


# ----------------------------------------------------------------------------- meta-learners
def ew(train, test, md, cols=None):
    cols = cols or md.rank_cols
    return test[cols].mean(axis=1).to_numpy()


def skill_weighted(train, test, md):
    R = test[md.rank_cols].to_numpy()
    S = np.nan_to_num(test[md.skill_cols].to_numpy(), nan=0.0).clip(min=0)
    w = S / np.where(S.sum(1, keepdims=True) > 0, S.sum(1, keepdims=True), 1)
    return (w * R).sum(1)


def stack_ridge(train, test, md):
    from sklearn.linear_model import Ridge
    X, y = train[md.rank_cols].to_numpy(), train.y.to_numpy()
    m = Ridge(alpha=float(len(X)) * 0.01, fit_intercept=False).fit(X, y)
    return m.predict(test[md.rank_cols].to_numpy())


def stack_xgb(train, test, md, seed=42):
    from xgboost import XGBRegressor
    cols = md.rank_cols + md.skill_cols + md.state_cols
    lo, hi = np.quantile(train.y, [0.005, 0.995])
    m = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                     min_child_weight=200, random_state=seed, tree_method="hist", device="cuda")
    m.fit(np.nan_to_num(train[cols].to_numpy()), np.clip(train.y.to_numpy(), lo, hi), verbose=False)
    return m.predict(np.nan_to_num(test[cols].to_numpy()))


def gated_stack(train, test, md, seed=42, epochs=30):
    """Regime-and-skill-gated stacking: w_t = softmax(g([state_t, skill_t])); y_hat = a * sum_m w_tm rank_tm.
    The gate is a small MLP; trained with MSE on past OOS data only."""
    import torch
    import torch.nn as nn
    from aft.train import DEVICE, set_seed
    set_seed(seed)
    ctx_cols = md.state_cols + md.skill_cols
    mu = train[ctx_cols].mean()
    sd = train[ctx_cols].std().replace(0, 1)

    def tens(df):
        C = np.nan_to_num(((df[ctx_cols] - mu) / sd).to_numpy(np.float32)).clip(-5, 5)
        R = df[md.rank_cols].to_numpy(np.float32)
        return torch.as_tensor(C, device=DEVICE), torch.as_tensor(R, device=DEVICE)

    M = len(md.rank_cols)
    gate = nn.Sequential(nn.Linear(len(ctx_cols), 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, M)).to(DEVICE)
    scale = nn.Parameter(torch.tensor(0.01, device=DEVICE))
    opt = torch.optim.AdamW(list(gate.parameters()) + [scale], lr=1e-3, weight_decay=1e-3)
    C, R = tens(train)
    lo, hi = np.quantile(train.y, [0.005, 0.995])
    y = torch.as_tensor(np.clip(train.y.to_numpy(np.float32), lo, hi), device=DEVICE)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        gate.train()
        perm = torch.randperm(len(y), generator=gen).to(DEVICE)
        for b in range(0, len(y), 2048):
            j = perm[b:b + 2048]
            w = torch.softmax(gate(C[j]), -1)
            loss = torch.mean((scale * (w * R[j]).sum(-1) - y[j]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
    gate.eval()
    Ct, Rt = tens(test)
    with torch.no_grad():
        w = torch.softmax(gate(Ct), -1)
        return (scale * (w * Rt).sum(-1)).cpu().numpy().astype(np.float64)


def long_skill_ew(train, test, md):
    """Equal weight over the inputs whose trailing 252-day realised IC is positive (else all inputs)."""
    R = test[md.rank_cols].to_numpy()
    S = np.nan_to_num(test[["skill252_" + c[5:] for c in md.rank_cols]].to_numpy(), nan=0.0)
    w = (S > 0).astype(float)
    w[w.sum(1) == 0] = 1.0
    return (w * R).sum(1) / w.sum(1)


def regime_tilted_ew(train, test, md):
    """y_hat = a * EW + sum_{m,k} g_mk * state_k(t) * rank_m(t, i), with the tilts g ridge-shrunk to 0, so
    the model reduces to the equal-weight combination unless past OOS data support a regime tilt.
    a is fitted by OLS; the ridge penalty is chosen on the last 20 % of the (past) training dates."""
    from sklearn.linear_model import Ridge
    mu = train[md.state_cols].mean()
    sd = train[md.state_cols].std().replace(0, 1)

    def design(df):
        Z = ((df[md.state_cols] - mu) / sd).fillna(0).clip(-4, 4).to_numpy()
        R = df[md.rank_cols].to_numpy()
        return R.mean(1), np.einsum("nm,nk->nmk", R, Z).reshape(len(df), -1)

    ew_tr, X_tr = design(train)
    y = np.clip(train.y.to_numpy(), *np.quantile(train.y, [0.005, 0.995]))
    a = float(np.dot(ew_tr, y) / np.dot(ew_tr, ew_tr))
    resid = y - a * ew_tr
    cut = np.quantile(train.t, 0.8)
    fit_m, val_m = train.t.to_numpy() <= cut, train.t.to_numpy() > cut
    best = None
    for alpha in (1e2, 1e3, 1e4, 1e5, 1e6):
        m = Ridge(alpha=alpha, fit_intercept=False).fit(X_tr[fit_m], resid[fit_m])
        p = a * ew_tr[val_m] + m.predict(X_tr[val_m])
        v = pd.DataFrame({"t": train.t.to_numpy()[val_m], "p": p, "y": train.y.to_numpy()[val_m]})
        ic = v.groupby("t").apply(lambda g: g.p.rank().corr(g.y.rank()), include_groups=False).mean()
        if best is None or ic > best[0]:
            best = (ic, alpha)
    m = Ridge(alpha=best[1], fit_intercept=False).fit(X_tr, resid)
    ew_te, X_te = design(test)
    return a * ew_te + m.predict(X_te)


LEARNERS = {"Meta: EW all": ew, "Meta: regime-tilted EW": regime_tilted_ew,
            "Meta: skill-weighted": skill_weighted, "Meta: long-skill EW": long_skill_ew,
            "Meta: stacked ridge": stack_ridge, "Meta: stacked XGB": stack_xgb, "Meta: gated stacking": gated_stack}


def timed_exposure(pred: pd.DataFrame, frame: pd.DataFrame, md, h: int, years: range) -> pd.Series:
    """Regime-gated exposure: walk-forward ridge regression of the signal's realised daily IC on the
    market state; exposure is 1 when the predicted IC is positive, else 0.  Training uses only dates
    whose IC is realised before the refit (t <= first test date - h)."""
    from sklearn.linear_model import Ridge
    ic = daily_ic(pred)
    state = frame.groupby("t")[md.state_cols].first()
    yr = frame.groupby("t")["year"].first()
    df = state.join(ic.rename("ic"), how="inner").join(yr)
    out = {}
    for year in years:
        test = df[df.year == year]
        if test.empty:
            continue
        t0 = test.index.min()
        train = df[(df.index <= t0 - h)].dropna()
        if len(train) < 100:
            continue
        mu, sd = train[md.state_cols].mean(), train[md.state_cols].std().replace(0, 1)
        m = Ridge(alpha=10.0).fit(((train[md.state_cols] - mu) / sd).to_numpy(), train.ic.to_numpy())
        p = m.predict(((test[md.state_cols] - mu) / sd).fillna(0).to_numpy())
        out.update(dict(zip(test.index, (p > 0).astype(float))))
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Ridge,XGBoost,MLP,DLinear,LSTM,GRU,Transformer,AFT,AFT-R")
    ap.add_argument("--panel", default="top100")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    h = a.horizon
    P = load_panel(a.panel, h)[0]
    md = build_meta_data(a.models.split(","), P, factor_feats=FACTORS, wf=wf_dir(a.panel, h), h=h)
    f = md.frame
    f = f[f.year <= DEV_LAST_YEAR].reset_index(drop=True)          # development period only
    print("base models:", md.models, "| meta inputs:", len(md.rank_cols), "ranks,", len(md.skill_cols), "skills,",
          len(md.state_cols), "state")
    preds = {}
    for year in range(2017, DEV_LAST_YEAR + 1):
        t0 = f.loc[f.year == year, "t"].min()
        train = f[(f.year < year) & (f.t <= t0 - h)]
        test = f[f.year == year]
        for name, fn in LEARNERS.items():
            p = fn(train, test, md)
            preds.setdefault(name, []).append(test[["t", "i", "y", "y_abs"]].assign(y_hat=p))
    evald = f[f.year >= 2017]
    for col in md.rank_cols:
        preds["Base: " + col[5:]] = [evald[["t", "i", "y", "y_abs"]].assign(y_hat=evald[col].to_numpy())]
    preds = {k: pd.concat(v, ignore_index=True) for k, v in preds.items()}
    ref = "Meta: EW all"
    ic = {k: daily_ic(v) for k, v in preds.items()}
    rows = []
    for k, d in preds.items():
        r = {"model": k}
        r.update(evaluate(d, costs=(0.0, 5.0, 10.0), h=h))
        if k != ref:
            j = ic[k].index.intersection(ic[ref].index)
            t = hac_mean_test((ic[k].loc[j] - ic[ref].loc[j]).to_numpy(), lag=h - 1)
            r.update({"dIC_vs_EW": t["mean"], "dIC_p": t["p"]})
        rows.append(r)
    # regime-gated exposure timing of selected signals (evaluated on the same years)
    for k in ["Meta: EW all", "Base: rev5", "Base: rev21", "Base: Ridge", "Base: MLP"]:
        if k not in preds:
            continue
        expo = timed_exposure(preds[k], f, md, h, range(2018, DEV_LAST_YEAR + 1))
        d = preds[k][preds[k].t.isin(expo.index)]
        r = {"model": "Timed: " + k, "exposure_mean": float(expo.mean())}
        r.update(evaluate(d, costs=(0.0, 5.0, 10.0), exposure=expo, h=h))
        u = {"model": "Untimed same years: " + k}
        u.update(evaluate(d, costs=(0.0, 5.0, 10.0), h=h))
        rows += [r, u]
    res = pd.DataFrame(rows).sort_values("IC", ascending=False)
    res.to_csv(OUT / f"dev_meta_{a.panel}_h{h}_{a.tag}.csv", index=False)
    cols = ["model", "exposure_mean", "IC", "IC_t", "ICIR", "dIC_vs_EW", "dIC_p", "LS_sharpe@0", "LS_sharpe@5", "LS_sharpe@10",
            "LS_ann_return@5", "LS_turnover@5", "R2_OS"]
    pd.set_option("display.width", 250)
    print(res[[c for c in cols if c in res]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
