"""Compute every reported number from saved predictions.

Outputs
  results/summary/*.csv        machine-readable tables
  tables/*.tex                 LaTeX tables included by research.tex
  numbers.tex                  \\newcommand macros for every number quoted in the text
  figures/*.pdf                figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.data import load_master_dataset, make_splits  # noqa: E402
from aft.experiment import Store  # noqa: E402
from aft.metrics import COSTS, benchmark_row, ensemble, per_run_table, run_metrics  # noqa: E402
from aft.stats import diebold_mariano, hac_mean_test, mean_ci, paired_comparison  # noqa: E402

RES = ROOT / "results"
SUM = RES / "summary"
TAB = ROOT / "tables"
FIG = ROOT / "figures"
for d in (SUM, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

MAIN = ["Zero", "Train mean", "Momentum (5d)", "Ridge", "Random Forest", "XGBoost", "LSTM", "GRU",
        "DLinear", "Transformer", "AFT", "AFT (v1 tuned cfg)", "AFT-HPO (composite)", "AFT-HPO (MSE)"]
ABL = ["AFT", "AFT -regime/gate", "AFT -fin. context", "AFT -temporal bias", "AFT -both biases",
       "AFT +composite loss", "AFT top-40", "AFT-R (renorm. gate)"]
LABEL = {"Zero": "Zero forecast", "Train mean": "Train mean (= buy \\& hold)", "Momentum (5d)": "5-day momentum",
         "Random Forest": "Random forest", "AFT (v1 tuned cfg)": "AFT, v1 tuned config",
         "AFT-HPO (composite)": "AFT-HPO (composite)", "AFT-HPO (MSE)": "AFT-HPO (MSE)",
         "AFT -regime/gate": "$-$ regime encoder \\& gate", "AFT -fin. context": "$-$ financial-context bias",
         "AFT -temporal bias": "$-$ temporal bias", "AFT -both biases": "$-$ both attention biases",
         "AFT +composite loss": "$+$ composite training loss", "AFT top-40": "Top-40 features",
         "AFT-R (renorm. gate)": "AFT-R (renormalised gate)", "AFT": "AFT (full)"}
KEY = {"Zero": "Zero", "Train mean": "Mean", "Momentum (5d)": "Mom", "Ridge": "Ridge", "Random Forest": "Rf",
       "XGBoost": "Xgb", "LSTM": "Lstm", "GRU": "Gru", "DLinear": "Dlin", "Transformer": "Trf", "AFT": "Aft",
       "AFT (v1 tuned cfg)": "Aftvone", "AFT-HPO (composite)": "Afthpoc", "AFT-HPO (MSE)": "Afthpom",
       "AFT -regime/gate": "Ablreg", "AFT -fin. context": "Ablfin", "AFT -temporal bias": "Abltmp",
       "AFT -both biases": "Ablboth", "AFT +composite loss": "Ablloss", "AFT top-40": "Abltop",
       "AFT-R (renorm. gate)": "Aftr"}
NUM: dict[str, str] = {}


def num(name: str, value: str) -> None:
    assert name.isalpha(), name
    NUM[name] = value


def f(x, d=2, pct=False, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    v = 100 * x if pct else x
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def pm(mean, half, d=2, pct=False, n=5):
    if n <= 1 or half == 0 or np.isnan(half):
        return f(mean, d, pct)
    return f"{f(mean, d, pct)}\\,$\\pm$\\,{f(half, d, pct)}"


def pval(p):
    if p is None or np.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def load_store(path: Path):
    preds, metas = Store(path).load()
    preds["date"] = pd.to_datetime(preds["date"])
    return preds, metas


def train_means(preds):
    tm = preds[(preds.model == "Train mean")].groupby("asset").y_hat.first().to_dict()
    return tm


def summarize(runs: pd.DataFrame, metas: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    metrics = ["MAE", "RMSE", "R2_OS", "DA", "Turnover", "Long share"] + \
              [f"{m}@{int(c)}" for c in COSTS for m in ("Sharpe", "Return", "MDD")]
    rows = []
    for m in models:
        g = runs[runs.model == m]
        if g.empty:
            continue
        row = {"model": m, "n_seeds": g.seed.nunique()}
        for k in metrics:
            mean, sd, half = mean_ci(g[k].to_numpy())
            row[k], row[k + "_sd"], row[k + "_ci"] = mean, sd, half
        mt = metas[metas.model == m]
        row["params"] = int(mt["params"].iloc[0]) if "params" in mt and len(mt) else 0
        row["train_time_s"] = float(mt["train_time_s"].mean()) if "train_time_s" in mt and mt["train_time_s"].notna().any() else np.nan
        row["epochs"] = float(mt["epochs_run"].mean()) if "epochs_run" in mt and mt["epochs_run"].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================ AAPL main study
def aapl():
    preds, metas = load_store(RES / "aapl")
    tm = train_means(preds)
    runs = per_run_table(preds, tm)
    runs.drop(columns=["Sharpe_offsets"]).to_csv(SUM / "aapl_runs_test.csv", index=False)
    runs_val = per_run_table(preds, tm, split="val")
    runs_val.drop(columns=["Sharpe_offsets"]).to_csv(SUM / "aapl_runs_val.csv", index=False)
    summ = summarize(runs, metas, MAIN + [m for m in ABL if m not in MAIN])
    summ.to_csv(SUM / "aapl_summary_test.csv", index=False)
    summ_val = summarize(runs_val, metas, MAIN + [m for m in ABL if m not in MAIN])
    summ_val.to_csv(SUM / "aapl_summary_val.csv", index=False)
    test = preds[(preds.split == "test") & (preds.model == "Zero")].sort_values("row")
    y = test.y.to_numpy()
    bh = benchmark_row(y, tm["AAPL"])

    # ---- data facts
    df = load_master_dataset()
    sd = make_splits(df)
    d = df.Date
    num("nRows", f"{len(df):,}".replace(",", "{,}"))
    num("nTrain", f"{len(sd.train_idx):,}".replace(",", "{,}"))
    num("nVal", f"{len(sd.val_idx)}")
    num("nTest", f"{len(sd.test_idx)}")
    num("dateFirst", d.iloc[0].strftime("%Y-%m-%d"))
    num("dateLast", d.iloc[-1].strftime("%Y-%m-%d"))
    num("dateTrainStart", d.iloc[sd.train_idx[0]].strftime("%Y-%m-%d"))
    num("dateTrainEnd", d.iloc[sd.train_idx[-1]].strftime("%Y-%m-%d"))
    num("dateValStart", d.iloc[sd.val_idx[0]].strftime("%Y-%m-%d"))
    num("dateValEnd", d.iloc[sd.val_idx[-1]].strftime("%Y-%m-%d"))
    num("dateTestStart", d.iloc[sd.test_idx[0]].strftime("%Y-%m-%d"))
    num("dateTestEnd", d.iloc[sd.test_idx[-1]].strftime("%Y-%m-%d"))
    num("upRate", f(float(np.mean(y > 0)), 1, pct=True))
    num("bhSharpe", f(bh["Sharpe@5"]))
    num("bhReturn", f(bh["Return@5"], 1, pct=True))
    num("bhMdd", f(bh["MDD@5"], 1, pct=True))
    num("nTradesPerPhase", f"{len(y) / 5:.0f}")

    # ---- ensemble-level tests
    ens = ensemble(preds)
    E = {m: g.sort_values("row") for m, g in ens.groupby("model")}
    zero_e = E["Zero"].y.to_numpy() - 0.0
    tests = []
    for m in MAIN + ABL:
        if m not in E or m in [t["model"] for t in tests]:
            continue
        g = E[m]
        e = g.y.to_numpy() - g.y_hat.to_numpy()
        row = {"model": m, "n_seeds": int(g.n_seeds.iloc[0])}
        em = run_metrics(g.y.to_numpy(), g.y_hat.to_numpy(), tm["AAPL"])
        row.update({f"ens_{k}": v for k, v in em.items() if k != "Sharpe_offsets"})
        if m != "Zero":
            dm = diebold_mariano(e, zero_e, power=2)
            row["dm_zero_stat"], row["dm_zero_p"] = dm["stat"], dm["p"]
            dm1 = diebold_mariano(e, zero_e, power=1)
            row["dm1_zero_stat"], row["dm1_zero_p"] = dm1["stat"], dm1["p"]
        if m != "AFT" and "AFT" in E:
            ea = E["AFT"].y.to_numpy() - E["AFT"].y_hat.to_numpy()
            dm = diebold_mariano(e, ea, power=2)
            row["dm_aft_stat"], row["dm_aft_p"] = dm["stat"], dm["p"]
        hits = (np.sign(g.y_hat.to_numpy()) == np.sign(g.y.to_numpy())).astype(float)
        hits[np.abs(g.y_hat.to_numpy()) < 1e-6] = 0.0
        row["da_hac_p"] = hac_mean_test(hits, lag=4, mu0=0.5)["p"]
        up = (g.y.to_numpy() > 0).astype(float)
        row["da_vs_long_p"] = hac_mean_test(hits - up, lag=4)["p"]
        tests.append(row)
    tests = pd.DataFrame(tests)
    tests.to_csv(SUM / "aapl_ensemble_tests.csv", index=False)

    # ---- paired seed comparisons vs AFT (same seeds)
    pairs = []
    ref = runs[runs.model == "AFT"].set_index("seed")
    for m in MAIN + ABL:
        g = runs[runs.model == m].set_index("seed")
        if m == "AFT" or g.empty or len(g) < 5 or not set(g.index) == set(ref.index):
            continue
        for k in ("MAE", "DA", "Sharpe@5", "R2_OS"):
            pc = paired_comparison(g.loc[ref.index, k].to_numpy(), ref[k].to_numpy())
            pairs.append({"model": m, "metric": k, **pc})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(SUM / "aapl_paired_vs_aft.csv", index=False)

    # ---- table: main benchmark
    S = summ.set_index("model")
    T = tests.set_index("model")
    lines = []
    groups = [("Naive", ["Zero", "Train mean", "Momentum (5d)"]),
              ("Tabular (features at $t$)", ["Ridge", "Random Forest", "XGBoost"]),
              ("Sequence models ($L=60$)", ["LSTM", "GRU", "DLinear", "Transformer"]),
              ("Adaptive Financial Transformer", ["AFT", "AFT (v1 tuned cfg)", "AFT-HPO (composite)", "AFT-HPO (MSE)"])]
    for title, ms in groups:
        ms = [m for m in ms if m in S.index]
        if not ms:
            continue
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for m in ms:
            r = S.loc[m]
            n = int(r.n_seeds)
            zero = m == "Zero"
            cells = [LABEL.get(m, m), f"{int(r.params):,}" if r.params else "--",
                     pm(r.MAE, r.MAE_ci, 2, True, n), pm(r.RMSE, r.RMSE_ci, 2, True, n),
                     pm(r.R2_OS, r.R2_OS_ci, 1, True, n),
                     "--" if zero else pm(r.DA, r.DA_ci, 1, True, n),
                     "--" if zero else pm(r["Sharpe@5"], r["Sharpe@5_ci"], 2, False, n),
                     "--" if zero else pm(r["Return@5"], r["Return@5_ci"], 1, True, n),
                     "--" if zero else pm(r["MDD@5"], r["MDD@5_ci"], 1, True, n),
                     pval(T.loc[m, "dm_zero_p"]) if (m in T.index and not zero) else "--"]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    (TAB / "aapl_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- markdown version for the README
    md = ["| Model | Params | MAE (%) | RMSE (%) | R²_OS (%) | DA (%) | Sharpe (5 bp) | Net return (%) | Max DD (%) | DM p vs zero |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def mdpm(mean, half, d, pct, n):
        s = (100 * mean if pct else mean)
        h = (100 * half if pct else half)
        return f"{s:.{d}f}" if n <= 1 or half == 0 else f"{s:.{d}f} ± {h:.{d}f}"
    for _, ms in groups:
        for m in ms:
            if m not in S.index:
                continue
            r = S.loc[m]
            n = int(r.n_seeds)
            zero = m == "Zero"
            md.append("| " + " | ".join([
                LABEL.get(m, m).replace("\\&", "&"), f"{int(r.params):,}" if r.params else "–",
                mdpm(r.MAE, r.MAE_ci, 2, True, n), mdpm(r.RMSE, r.RMSE_ci, 2, True, n), mdpm(r.R2_OS, r.R2_OS_ci, 1, True, n),
                "–" if zero else mdpm(r.DA, r.DA_ci, 1, True, n),
                "–" if zero else mdpm(r["Sharpe@5"], r["Sharpe@5_ci"], 2, False, n),
                "–" if zero else mdpm(r["Return@5"], r["Return@5_ci"], 1, True, n),
                "–" if zero else mdpm(r["MDD@5"], r["MDD@5_ci"], 1, True, n),
                "–" if zero or m not in T.index else (f"{T.loc[m, 'dm_zero_p']:.3f}" if T.loc[m, "dm_zero_p"] >= 0.001 else "<0.001")]) + " |")
    (SUM / "aapl_main.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- table: ablation
    P = pairs.set_index(["model", "metric"]) if len(pairs) else None
    lines = []
    for m in ABL:
        if m not in S.index:
            continue
        r = S.loc[m]
        n = int(r.n_seeds)
        def pp(metric):
            if m == "AFT" or P is None or (m, metric) not in P.index:
                return "--"
            x = P.loc[(m, metric)]
            return f"{f(x.mean_diff, 2 if metric == 'Sharpe@5' else 1 if metric == 'DA' else 3, metric != 'Sharpe@5', sign=True)} ({pval(x.t_p)})"
        cells = [LABEL.get(m, m), f"{int(r.params):,}",
                 pm(r.MAE, r.MAE_ci, 2, True, n), pm(r.DA, r.DA_ci, 1, True, n),
                 pm(r["Sharpe@5"], r["Sharpe@5_ci"], 2, False, n), pp("MAE"), pp("DA"), pp("Sharpe@5")]
        lines.append(" & ".join(cells) + " \\\\")
    (TAB / "aapl_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- table: cost sensitivity (seed means)
    lines = []
    for m in ["Train mean", "Momentum (5d)", "Ridge", "XGBoost", "LSTM", "Transformer", "AFT", "AFT-HPO (composite)"]:
        if m not in S.index:
            continue
        r = S.loc[m]
        cells = [LABEL.get(m, m)] + [f(r[f"Sharpe@{int(c)}"]) for c in COSTS] + [f(r["Turnover"])]
        lines.append(" & ".join(cells) + " \\\\")
    (TAB / "cost_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- macros for the text
    for m in MAIN + ABL:
        if m not in S.index:
            continue
        r, k = S.loc[m], KEY[m]
        num(f"m{k}Mae", f(r.MAE, 2, True)); num(f"m{k}MaeCi", f(r.MAE_ci, 2, True))
        num(f"m{k}Rmse", f(r.RMSE, 2, True))
        num(f"m{k}Rtwo", f(r.R2_OS, 1, True))
        num(f"m{k}Da", f(r.DA, 1, True)); num(f"m{k}DaCi", f(r.DA_ci, 1, True))
        num(f"m{k}Sr", f(r["Sharpe@5"])); num(f"m{k}SrCi", f(r["Sharpe@5_ci"]))
        num(f"m{k}SrZero", f(r["Sharpe@0"])); num(f"m{k}SrTwenty", f(r["Sharpe@20"]))
        num(f"m{k}Ret", f(r["Return@5"], 1, True)); num(f"m{k}Mdd", f(r["MDD@5"], 1, True))
        num(f"m{k}Turn", f(r["Turnover"])); num(f"m{k}Long", f(r["Long share"], 1, True))
        num(f"m{k}Params", f"{int(r.params):,}".replace(",", "{,}"))
        if not np.isnan(r.train_time_s):
            num(f"m{k}Time", f"{r.train_time_s:.1f}")
        if m in T.index:
            t = T.loc[m]
            if "dm_zero_p" in t and not np.isnan(t.get("dm_zero_p", np.nan)):
                num(f"m{k}DmZeroP", pval(t.dm_zero_p)); num(f"m{k}DmZeroStat", f(t.dm_zero_stat))
            if "dm_aft_p" in t and not np.isnan(t.get("dm_aft_p", np.nan)):
                num(f"m{k}DmAftP", pval(t.dm_aft_p)); num(f"m{k}DmAftStat", f(t.dm_aft_stat))
            num(f"m{k}DaP", pval(t.da_hac_p)); num(f"m{k}DaLongP", pval(t.da_vs_long_p))
            num(f"m{k}EnsMae", f(t.ens_MAE, 2, True)); num(f"m{k}EnsDa", f(t.ens_DA, 1, True))
            num(f"m{k}EnsSr", f(t["ens_Sharpe@5"]))
    for _, x in pairs.iterrows():
        k = KEY[x.model] + {"MAE": "Mae", "DA": "Da", "Sharpe@5": "Sr", "R2_OS": "Rtwo"}[x.metric]
        num(f"p{k}", pval(x.t_p)); num(f"w{k}", pval(x.wilcoxon_p)); num(f"dz{k}", f(x.d_z))
    return summ, tests, pairs, runs


# ============================================================================ HPO
def hpo():
    out = {}
    for obj in ("composite", "mse"):
        p = RES / "hpo" / f"best_{obj}.json"
        t = RES / "hpo" / f"trials_{obj}.csv"
        if not p.exists():
            continue
        best = json.loads(p.read_text())
        trials = pd.read_csv(t)
        out[obj] = (best, trials)
        k = "C" if obj == "composite" else "M"
        num(f"hpo{k}Trial", str(best["trial"]))
        num(f"hpo{k}Complete", str(int((trials.state == "COMPLETE").sum())))
        num(f"hpo{k}Pruned", str(int((trials.state == "PRUNED").sum())))
        ua = best["user_attrs"]
        num(f"hpo{k}ValSr", f(ua["mean_sharpe"]))
        num(f"hpo{k}ValDa", f(ua["mean_da"], 1, True))
        num(f"hpo{k}ValMae", f(ua["mean_mae"], 2, True))
        num(f"hpo{k}Value", f(best["value"], 3 if obj == "composite" else 6))
        bp = best["params"]
        num(f"hpo{k}Lr", f"{bp['lr']:.2e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}")
        num(f"hpo{k}L", str(bp["seq_len"]))
        num(f"hpo{k}D", str(bp["d_model"]))
        num(f"hpo{k}Top", "Top-40" if bp["use_top40"] else "all 95")
        # distribution of validation Sharpe among completed trials
        comp = trials[trials.state == "COMPLETE"]
        if "user_attrs_mean_sharpe" in comp:
            num(f"hpo{k}ValSrMedian", f(comp["user_attrs_mean_sharpe"].median()))
        tsum = RES / "hpo" / f"transfer_{obj}_summary.json"
        if tsum.exists():
            ts = json.loads(tsum.read_text())
            num(f"tr{k}N", str(ts["n_trials"]))
            for col, key in (("test_sharpe", "TestSr"), ("test_mae_ratio", "TestMae"), ("test_da", "TestDa"),
                             ("val_sharpe", "ValSr"), ("test_mse", "TestMse"), ("val_mse", "ValMse")):
                num(f"tr{k}{key}Rho", f(ts[col]["rho"]))
                num(f"tr{k}{key}P", pval(ts[col]["p"]))
    return out


def hpo_table(hp):
    v1 = json.loads((ROOT / "experiments" / "best_hyperparams_financial.json").read_text())
    metas = Store(RES / "aapl").load()[1]
    def params_of(name):
        g = metas[metas.model == name]
        return f"{int(g.params.iloc[0]):,}" if len(g) else "--"
    rows = [("Baseline AFT (default)", {"lr": 1e-3, "wd": 1e-4, "dropout": 0.15, "seq_len": 60, "d_model": 128,
                                        "ff_multiplier": 2, "num_heads": 8, "num_layers": 2, "use_top40": False}, "AFT"),
            ("v1 tuned (NB 12.1)", v1, "AFT (v1 tuned cfg)")]
    for obj, label in (("composite", "Re-tuned, composite"), ("mse", "Re-tuned, MSE")):
        if obj in hp:
            rows.append((label, hp[obj][0]["params"], f"AFT-HPO ({'composite' if obj == 'composite' else 'MSE'})"))
    lines = []
    for label, p, name in rows:
        lr = f"{p['lr']:.1e}".replace("e-0", "e-").replace("e-", "$\\times10^{-") + "}$"
        wd = f"{p['wd']:.1e}".replace("e-0", "e-").replace("e-", "$\\times10^{-") + "}$"
        cells = [label, lr, wd, f"{p['dropout']:.2f}", str(p["seq_len"]), str(p["d_model"]),
                 str(p["ff_multiplier"] * p["d_model"]), str(p["num_heads"]), str(p["num_layers"]),
                 "40" if p["use_top40"] else "95", params_of(name)]
        lines.append(" & ".join(cells) + " \\\\")
    (TAB / "hpo_configs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================ multi-asset
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "JPM", "XOM", "JNJ", "WMT", "SPY"]
MA_MODELS = ["Train mean", "Momentum (5d)", "Ridge", "XGBoost", "LSTM", "DLinear", "Transformer", "AFT", "AFT-HPO (composite)"]


def multiasset():
    allruns, alltests = [], []
    for t in TICKERS:
        path = RES / "multiasset" / t
        if not (path / "preds").exists():
            continue
        preds, metas = load_store(path)
        tm = train_means(preds)
        runs = per_run_table(preds, tm)
        y = preds[(preds.split == "test") & (preds.model == "Zero")].sort_values("row").y.to_numpy()
        zero_mae = float(np.mean(np.abs(y)))
        runs["MAE_ratio"] = runs["MAE"] / zero_mae
        runs["up_rate"] = float(np.mean(y > 0))
        allruns.append(runs)
        ens = ensemble(preds)
        for m, g in ens.groupby("model"):
            g = g.sort_values("row")
            e = g.y.to_numpy() - g.y_hat.to_numpy()
            row = {"asset": t, "model": m}
            if m != "Zero":
                dm = diebold_mariano(e, g.y.to_numpy(), power=2)
                row.update({"dm_zero_stat": dm["stat"], "dm_zero_p": dm["p"]})
            alltests.append(row)
    if not allruns:
        return None
    runs = pd.concat(allruns, ignore_index=True)
    runs.drop(columns=["Sharpe_offsets"]).to_csv(SUM / "multiasset_runs_test.csv", index=False)
    agg = (runs.groupby(["asset", "model"]).agg(MAE_ratio=("MAE_ratio", "mean"), DA=("DA", "mean"),
                                                 SR=("Sharpe@5", "mean"), SR_sd=("Sharpe@5", "std"),
                                                 R2=("R2_OS", "mean"), up=("up_rate", "first"),
                                                 n=("seed", "nunique")).reset_index())
    tests = pd.DataFrame(alltests)
    agg = agg.merge(tests, on=["asset", "model"], how="left")
    agg.to_csv(SUM / "multiasset_summary.csv", index=False)
    assets = [t for t in TICKERS if t in set(agg.asset)]
    A = agg.set_index(["asset", "model"])
    # table: Sharpe and DA by asset
    show = ["Train mean", "Ridge", "XGBoost", "LSTM", "DLinear", "Transformer", "AFT", "AFT-HPO (composite)"]
    lines = []
    for a in assets:
        up = A.loc[(a, "Zero"), "up"] if (a, "Zero") in A.index else np.nan
        cells = [a, f(up, 1, True)]
        cells += [f(A.loc[(a, m), "SR"]) if (a, m) in A.index else "--" for m in show]
        cells += [f(A.loc[(a, "AFT"), "MAE_ratio"], 3) if (a, "AFT") in A.index else "--",
                  f(A.loc[(a, "AFT"), "DA"], 1, True) if (a, "AFT") in A.index else "--",
                  pval(A.loc[(a, "AFT"), "dm_zero_p"]) if (a, "AFT") in A.index else "--"]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    mean_cells = ["Mean", f(np.mean([A.loc[(a, "Zero"), "up"] for a in assets]), 1, True)]
    for m in show:
        vals = [A.loc[(a, m), "SR"] for a in assets if (a, m) in A.index]
        mean_cells.append(f(np.mean(vals)) if vals else "--")
    mean_cells += [f(np.mean([A.loc[(a, "AFT"), "MAE_ratio"] for a in assets]), 3),
                   f(np.mean([A.loc[(a, "AFT"), "DA"] for a in assets]), 1, True), ""]
    lines.append(" & ".join(mean_cells) + " \\\\")
    wins = ["Beats B\\&H", ""]
    for m in show:
        if m == "Train mean":
            wins.append("--")
            continue
        w = sum(A.loc[(a, m), "SR"] > A.loc[(a, "Train mean"), "SR"] for a in assets if (a, m) in A.index)
        wins.append(f"{w}/{len(assets)}")
    w_mae = sum(A.loc[(a, "AFT"), "MAE_ratio"] < 1 for a in assets)
    w_da = sum(A.loc[(a, "AFT"), "DA"] > A.loc[(a, "Zero"), "up"] for a in assets)
    wins += [f"{w_mae}/{len(assets)}", f"{w_da}/{len(assets)}", ""]
    lines.append(" & ".join(wins) + " \\\\")
    (TAB / "multiasset.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # macros
    num("maNAssets", str(len(assets)))
    for m, k in (("AFT", "Aft"), ("AFT-HPO (composite)", "Afthpoc"), ("Transformer", "Trf"), ("LSTM", "Lstm"),
                 ("Ridge", "Ridge"), ("XGBoost", "Xgb"), ("DLinear", "Dlin"), ("Train mean", "Mean"),
                 ("Momentum (5d)", "Mom")):
        vals = agg[agg.model == m]
        if vals.empty:
            continue
        num(f"ma{k}SrMean", f(vals.SR.mean()))
        num(f"ma{k}MaeRatioMean", f(vals.MAE_ratio.mean(), 3))
        num(f"ma{k}DaMean", f(vals.DA.mean(), 1, True))
        num(f"ma{k}MaeWins", str(int((vals.MAE_ratio < 1).sum())))
        bh = agg[agg.model == "Train mean"].set_index("asset").SR
        num(f"ma{k}SrWins", str(int(sum(vals.set_index("asset").SR > bh.reindex(vals.asset).to_numpy()))))
        sig = vals[(vals.dm_zero_p < 0.05) & (vals.dm_zero_stat < 0)]
        num(f"ma{k}DmBetter", str(len(sig)))
        sigw = vals[(vals.dm_zero_p < 0.05) & (vals.dm_zero_stat > 0)]
        num(f"ma{k}DmWorse", str(len(sigw)))
    up = agg[agg.model == "Zero"].set_index("asset").up
    num("maUpMean", f(up.mean(), 1, True))
    num("maUpMin", f(up.min(), 1, True)); num("maUpMax", f(up.max(), 1, True))
    num("maAftDaWins", str(w_da))
    return agg


# ============================================================================ main
if __name__ == "__main__":
    out = {}
    if (RES / "aapl" / "preds").exists():
        out["aapl"] = aapl()
    hp = hpo()
    if (RES / "aapl" / "preds").exists():
        hpo_table(hp)
    out["ma"] = multiasset()
    diag = RES / "diagnostics" / "numbers.json"
    if diag.exists():
        for k, v in json.loads(diag.read_text()).items():
            num(k, v)
    with open(ROOT / "numbers.tex", "w", encoding="utf-8") as fh:
        fh.write("% Generated by scripts/analyze.py -- do not edit by hand.\n")
        for k in sorted(NUM):
            fh.write(f"\\newcommand{{\\{k}}}{{{NUM[k]}}}\n")
    print(f"wrote {len(NUM)} macros")
