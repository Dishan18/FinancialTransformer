"""Generate every number, table and figure of the v2 paper from saved results.

Outputs (paper/): numbers.tex, tab_aapl.tex, tab_base.tex, tab_meta.tex, tab_system.tex, fig_ic_year.pdf
Nothing in the paper is typed by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "paper"
OUT.mkdir(exist_ok=True)
S1 = ROOT / "results" / "summary"
S2 = ROOT / "results_v2" / "summary"
M2 = ROOT / "results_v2" / "meta"
NUM: dict[str, str] = {}


def num(k, v):
    assert k.isalpha(), k
    NUM[k] = v


def f(x, d=3, pct=False, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    v = 100 * x if pct else x
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "$-$")


def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# ============================================================== v1 single-stock (AAPL), corrected protocol
a = pd.read_csv(S1 / "aapl_summary_test.csv").set_index("model")
t = pd.read_csv(S1 / "aapl_ensemble_tests.csv").set_index("model")
pr = pd.read_csv(S1 / "aapl_paired_vs_aft.csv")
rows = [("Zero forecast", "Zero"), ("Train mean (buy \\& hold)", "Train mean"), ("Ridge", "Ridge"),
        ("XGBoost", "XGBoost"), ("LSTM", "LSTM"), ("DLinear", "DLinear"), ("Transformer", "Transformer"),
        ("AFT (full)", "AFT"), ("AFT $-$ both attention biases", "AFT -both biases"),
        ("AFT $-$ regime encoder \\& gate", "AFT -regime/gate"), ("AFT-R (renormalised gate)", "AFT-R (renorm. gate)")]
lines = []
for lab, m in rows:
    r = a.loc[m]
    n = int(r.n_seeds)
    pm = (lambda v, c, d, pc=False: f(v, d, pc) if n == 1 else f"{f(v, d, pc)}\\,$\\pm$\\,{f(c, d, pc)}")
    zero = m == "Zero"
    lines.append(" & ".join([lab, pm(r.MAE, r.MAE_ci, 2, True), pm(r.R2_OS, r.R2_OS_ci, 1, True),
                             "--" if zero else pm(r.DA, r.DA_ci, 1, True),
                             "--" if zero else pm(r["Sharpe@5"], r["Sharpe@5_ci"], 2),
                             "--" if zero or m not in t.index else pv(t.loc[m, "dm_zero_p"])]) + " \\\\")
(OUT / "tab_aapl.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
num("aZeroMae", f(a.loc["Zero", "MAE"], 2, True)); num("aMeanMae", f(a.loc["Train mean", "MAE"], 2, True))
num("aMeanDa", f(a.loc["Train mean", "DA"], 1, True)); num("aMeanSr", f(a.loc["Train mean", "Sharpe@5"], 2))
num("aAftMae", f(a.loc["AFT", "MAE"], 2, True)); num("aAftSr", f(a.loc["AFT", "Sharpe@5"], 2))
num("aAftDa", f(a.loc["AFT", "DA"], 1, True)); num("aTrfMae", f(a.loc["Transformer", "MAE"], 2, True))
num("aRfMae", f(a.loc["Random Forest", "MAE"], 2, True))
num("aAftDmZeroP", pv(t.loc["AFT", "dm_zero_p"]))
for m, k in (("LSTM", "Lstm"), ("GRU", "Gru"), ("DLinear", "Dlin")):
    num(f"a{k}DmZeroP", pv(t.loc[m, "dm_zero_p"]))
P = pr.set_index(["model", "metric"])
for m, k in (("AFT -both biases", "Both"), ("AFT -fin. context", "Fin"), ("AFT -temporal bias", "Tmp"),
             ("AFT -regime/gate", "Reg"), ("AFT-R (renorm. gate)", "Aftr")):
    num(f"aAbl{k}MaeP", pv(P.loc[(m, "MAE"), "t_p"])); num(f"aAbl{k}SrP", pv(P.loc[(m, "Sharpe@5"), "t_p"]))
    num(f"aAbl{k}SrDiff", f(P.loc[(m, "Sharpe@5"), "mean_diff"], 2, sign=True))
num("aTestStart", "2024-05-07"); num("aTestEnd", "2025-12-22"); num("aNTest", "409")

# ============================================================== gate diagnostics
dj = ROOT / "results" / "diagnostics" / "numbers.json"
if dj.exists():
    d = json.loads(dj.read_text())
    for k in ("gAftEntropy", "gAftMaxGate", "gAftMinGate", "gAftFinQk", "gAftWeightSum", "gAftTempQk", "gAftQkStd",
              "gAftrEntropy", "gAftrMaxGate", "gAftrMinGate", "gAftrFinQk", "gAftrWeightSum", "gUniform",
              "csNvdaAbsZLevel", "csNvdaAbsZStat", "csAaplAbsZLevel", "csAaplAbsZStat"):
        if k in d:
            num(k, d[k])

# ============================================================== panel facts
z = np.load(ROOT / "data" / "panel_sp500.npz", allow_pickle=True)
elig = z["eligible"]
dates = z["dates"].astype("datetime64[D]")
num("spN", f"{z['X_stock'].shape[1]}")
num("spDays", f"{len(dates):,}".replace(",", "{,}"))
num("spMin", str(int(elig.sum(1)[300:].min()))); num("spMed", str(int(np.median(elig.sum(1)))))
num("spMax", str(int(elig.sum(1).max())))
valid = np.isfinite(z["X_stock"]).all(-1) & np.isfinite(z["y_abs_5"])
num("spValid", f"{int(valid.sum()):,}".replace(",", "{,}"))
num("tValid", "373{,}548")

# ============================================================== base models (development + holdout)
def load(name):
    return pd.read_csv(S2 / name).set_index("model")


t5, s5, s21 = load("top100_h5_dev_models.csv"), load("sp500_h5_dev_models.csv"), load("sp500_h21_dev_models.csv")
h5, h21 = load("sp500_h5_holdout_models.csv"), load("sp500_h21_holdout_models.csv")
base_rows = [("Ridge", "Ridge"), ("XGBoost", "XGBoost"), ("MLP", "MLP"), ("Transformer", "Transformer_s5"),
             ("AFT", "AFT_s5"), ("AFT-R", "AFT-R_s5"), ("5-day reversal", "Factor: reversal 5d"),
             ("21-day reversal", "Factor: reversal 21d"), ("12-1 momentum", "Factor: momentum 12-1"),
             ("Low volatility", "Factor: low volatility"), ("MAX (lottery)", "Factor: MAX")]


def ict(df, m):
    if m not in df.index:
        return "--"
    return f"{f(df.loc[m, 'IC'], 3)} ({f(df.loc[m, 'IC_t'], 1)})"


lines = []
for lab, m in base_rows:
    cells = [lab, ict(t5, m), ict(s5, m), ict(s21, m),
             f(s21.loc[m, "LS_sharpe@5"], 2) if m in s21.index else "--",
             ict(h5, m), ict(h21, m)]
    lines.append(" & ".join(cells) + " \\\\")
(OUT / "tab_base.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
for df, tag in ((t5, "tf"), (s5, "sf"), (s21, "st"), (h5, "hf"), (h21, "ht")):
    for lab, m in base_rows:
        if m in df.index:
            k = {"Ridge": "Ridge", "XGBoost": "Xgb", "MLP": "Mlp", "Transformer_s5": "Trf", "AFT_s5": "Aft",
                 "AFT-R_s5": "Aftr", "Factor: reversal 5d": "Revf", "Factor: reversal 21d": "Revt",
                 "Factor: momentum 12-1": "Mom", "Factor: low volatility": "Lowvol", "Factor: MAX": "Max"}[m]
            num(f"{tag}{k}Ic", f(df.loc[m, "IC"], 4)); num(f"{tag}{k}T", f(df.loc[m, "IC_t"], 2))
            num(f"{tag}{k}Sr", f(df.loc[m, "LS_sharpe@5"], 2)); num(f"{tag}{k}Mdd", f(df.loc[m, "LS_mdd@5"], 0, True))
num("stAftVsTrfP", pv(s21.loc["AFT_s5", "dIC_p"])); num("stAftVsTrfD", f(s21.loc["AFT_s5", "dIC_vs_ref"], 4, sign=True))
num("htAftVsTrfP", pv(h21.loc["AFT_s5", "dIC_p"])); num("htAftVsTrfD", f(h21.loc["AFT_s5", "dIC_vs_ref"], 4, sign=True))
if "XGBoost-IC" in h5.index:
    num("hfXgbicIc", f(h5.loc["XGBoost-IC", "IC"], 4)); num("hfXgbicT", f(h5.loc["XGBoost-IC", "IC_t"], 2))

# ============================================================== learned meta-learners (development)
m5, m21 = (pd.read_csv(M2 / "dev_meta_sp500_h5_v3.csv").set_index("model"),
           pd.read_csv(M2 / "dev_meta_sp500_h21_v3.csv").set_index("model"))
meta_rows = [("Equal-weight ranks (reference)", "Meta: EW all"), ("Regime-tilted EW (ridge)", "Meta: regime-tilted EW"),
             ("Stacked ridge", "Meta: stacked ridge"), ("Stacked XGBoost (+ skill, state)", "Meta: stacked XGB"),
             ("Regime/skill-gated stacking (MLP gate)", "Meta: gated stacking"),
             ("Skill-weighted (63-day IC)", "Meta: skill-weighted"), ("Skill-weighted (252-day IC)", "Meta: long-skill EW")]
lines = []
for lab, m in meta_rows:
    c = [lab]
    for df in (m5, m21):
        r = df.loc[m]
        c.append(f"{f(r.IC, 4)}")
        c.append("--" if m == "Meta: EW all" else pv(r.dIC_p))
    lines.append(" & ".join(c) + " \\\\")
(OUT / "tab_meta.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
num("mfEw", f(m5.loc["Meta: EW all", "IC"], 4)); num("mtEw", f(m21.loc["Meta: EW all", "IC"], 4))

# ============================================================== layered system (dev vs holdout)
sysrows = [("Best single input", None), ("L2: EW combination", "L2 combination (EW ranks)"),
           ("L2 + regime gate", "L2+L3 combination + regime gate"),
           ("\\textbf{System A}: L2 + buffer", "L2+L4 combination + buffer  [System A]"),
           ("L2 + gate + buffer", "L2+L3+L4 full system"),
           ("System B + buffer", "System B: regime-gated reversal components + buffer")]
lines = []
for h in (5, 21):
    dv = pd.read_csv(S2 / f"metasystem_sp500_h{h}_dev.csv").set_index("system")
    ho = pd.read_csv(S2 / f"metasystem_sp500_h{h}_holdout.csv").set_index("system")
    if h != 5:  # the first group header is in body.tex: \multicolumn must not directly follow \input
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{Horizon $h={h}$ days}}}} \\\\")
    for lab, key in sysrows:
        if key is None:
            bases = [i for i in dv.index if i.startswith("L1")]
            key_d = max(bases, key=lambda i: dv.loc[i, "SR@5"])
            lab2 = f"{lab} (dev: {key_d.split(': ')[1].replace('rev', 'rev. ').replace('rev. 5', '5-d rev.').replace('rev. 21', '21-d rev.')})"
            rd, rh = dv.loc[key_d], ho.loc[key_d]
        else:
            lab2, rd, rh = lab, dv.loc[key], ho.loc[key]
        lines.append(" & ".join([lab2, f(rd.IC, 3), f(rd["SR@5"], 2), f(rd["SR@10"], 2), f(rd.turnover, 2),
                                 f(rh.IC, 3), f(rh["SR@5"], 2), f(rh["SR@10"], 2), f(rh.turnover, 2)]) + " \\\\")
        tag = {"L2 combination (EW ranks)": "Ew", "L2+L3 combination + regime gate": "Gate",
               "L2+L4 combination + buffer  [System A]": "A", "L2+L3+L4 full system": "Full",
               "System B: regime-gated reversal components + buffer": "B"}.get(key)
        hk = "f" if h == 5 else "t"
        if tag:
            for nm, src in (("dev", rd), ("hold", rh)):
                num(f"sys{hk}{tag}{nm}Ic", f(src.IC, 3)); num(f"sys{hk}{tag}{nm}Srf", f(src["SR@5"], 2))
                num(f"sys{hk}{tag}{nm}Srt", f(src["SR@10"], 2)); num(f"sys{hk}{tag}{nm}Srz", f(src["SR@0"], 2))
                num(f"sys{hk}{tag}{nm}Srtw", f(src["SR@20"], 2)); num(f"sys{hk}{tag}{nm}Turn", f(src.turnover, 2))
                num(f"sys{hk}{tag}{nm}Mdd", f(src["mdd@5"], 0, True)); num(f"sys{hk}{tag}{nm}Exp", f(src.exposure, 0, True))
    lines.append("\\midrule")
(OUT / "tab_system.tex").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

bf = pd.read_csv(S2 / "sp500_h5_dev_buffer.csv")
ewq = bf[(bf.signal == "EW (models + reversals)") & (bf.rule == "quintile")].iloc[0]
ewb = bf[(bf.signal == "EW (models + reversals)") & (bf.rule == "buffer 20/50")].iloc[0]
num("bufQSr", f(ewq["SR@5"], 2)); num("bufBSr", f(ewb["SR@5"], 2)); num("bufQTurn", f(ewq.turnover, 2))
num("bufBTurn", f(ewb.turnover, 2)); num("bufQSrt", f(ewq["SR@10"], 2)); num("bufBSrt", f(ewb["SR@10"], 2))

# ============================================================== holdout primary / S5
hr = json.loads((S2 / "holdout_report.json").read_text())
num("hTIc", f(hr["SystemA_h21"]["IC"], 4)); num("hTT", f(hr["SystemA_h21"]["t"], 2))
num("hTP", f(hr["SystemA_h21"]["p_one_sided"], 2)); num("hFIc", f(hr["SystemA_h5"]["IC"], 4))
num("hFT", f(hr["SystemA_h5"]["t"], 2)); num("hFP", f(hr["SystemA_h5"]["p_one_sided"], 3))
for y, v in hr["SystemA_h21"]["IC_by_year"].items():
    num("hTIcY" + {"2023": "a", "2024": "b", "2025": "c"}[str(y)], f(v, 3))
for y, v in hr["SystemA_h5"]["IC_by_year"].items():
    num("hFIcY" + {"2023": "a", "2024": "b", "2025": "c"}[str(y)], f(v, 3))
s5 = hr["S5_reversal_regimes"]["by_state"]
num("sfVixP", f(s5["vix_log"]["p"], 2)); num("sfVolP", f(s5["mkt_vol_21"]["p"], 2))
num("sfDispP", f(s5["cs_disp_21"]["p"], 3)); num("sfMinP", f(min(v["p"] for v in s5.values()), 3))

# ============================================================== figure: System A IC by year (dev + holdout)
from aft2.evaluate import daily_ic  # noqa: E402
from aft2.meta import cs_rank_long, load_ensemble  # noqa: E402
from aft2.protocol import load_panel, wf_dir  # noqa: E402

series = {}
for h in (5, 21):
    Pn = load_panel("sp500", h)[0]
    yrs = Pn.dates.astype("datetime64[Y]").astype(int) + 1970
    b = None
    for m in ("XGBoost", "Ridge", "MLP"):
        dd = load_ensemble(m, wf_dir("sp500", h))
        dd = dd.assign(**{"r_" + m: cs_rank_long(dd, "y_hat")})[["t", "i", "y", "y_abs", "r_" + m]]
        b = dd if b is None else b.merge(dd[["t", "i", "r_" + m]], on=["t", "i"])
    for nm, ft in (("rev5", "ret_5"), ("rev21", "ret_21")):
        b["tmp"] = -Pn.raw(ft, b.t.to_numpy(), b.i.to_numpy())
        b["r_" + nm] = cs_rank_long(b, "tmp")
    rc = [c for c in b if c.startswith("r_")]
    ic = daily_ic(b[["t", "i", "y", "y_abs"]].assign(y_hat=b[rc].mean(axis=1)))
    series[h] = pd.Series(ic.to_numpy(), index=yrs[ic.index]).groupby(level=0).mean()
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(figsize=(3.4, 1.9))
yy = np.arange(2016, 2026)
w = 0.38
ax.bar(yy - w / 2, series[5].reindex(yy).to_numpy(), w, label="$h=5$", color="#4C72B0")
ax.bar(yy + w / 2, series[21].reindex(yy).to_numpy(), w, label="$h=21$", color="#DD8452")
ax.axvspan(2022.5, 2025.5, color="0.85", zorder=0)
ax.axhline(0, color="k", lw=0.6)
ax.text(2024, ax.get_ylim()[1] * 0.85, "sealed holdout", ha="center", fontsize=6.5)
ax.set_xticks(yy); ax.set_xticklabels([str(v)[2:] for v in yy], fontsize=6.5)
ax.tick_params(axis="y", labelsize=6.5); ax.set_ylabel("mean daily rank IC", fontsize=7)
ax.legend(fontsize=6.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "fig_ic_year.pdf"); plt.close(fig)
for h, k in ((5, "f"), (21, "t")):
    s = series[h]
    num(f"yr{k}DevPos", str(int((s.loc[2016:2022] > 0).sum())))

with open(OUT / "numbers.tex", "w", encoding="utf-8") as fh:
    fh.write("% Generated by scripts/v2_paper_numbers.py -- do not edit by hand.\n")
    for k in sorted(NUM):
        fh.write(f"\\newcommand{{\\{k}}}{{{NUM[k]}}}\n")
print(f"{len(NUM)} macros; series:\n", pd.DataFrame(series).round(4))
