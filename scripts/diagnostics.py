"""Mechanistic and distribution-shift diagnostics, computed from saved checkpoints.

1. Gate behaviour of the AFT and AFT-R on the AAPL test windows (per seed): mean gate per group,
   normalised entropy H(w)/log G (1 = uniform), mean max-gate, average temporal std of the gates,
   and the magnitude of the financial-context bias relative to the query-key logits.
2. Covariate shift of every asset in the multi-asset study (train vs test), related to the AFT's
   out-of-sample performance.
Writes results/diagnostics/*.csv, results/diagnostics/numbers.json and figures/fig_gates.pdf.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aft.data import (ALL_FEATURES, FEATURE_GROUPS, LEVEL_FEATURES, build_features, download_ohlcv,  # noqa: E402
                      group_indices, load_master_dataset, make_splits)
from aft.experiment import AFT_BASE, SEEDS, Store, build_neural  # noqa: E402
from aft.train import DEVICE  # noqa: E402

OUT = ROOT / "results" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
NUM: dict[str, str] = {}


def fmt(x, d=3):
    return f"{x:.{d}f}".replace("-", "$-$")


def gate_stats(name: str, variant: dict, sd, store_root: Path):
    L = AFT_BASE["seq_len"]
    X = torch.as_tensor(sd.windows(sd.test_idx, L), device=DEVICE)
    G = len(FEATURE_GROUPS)
    rows, gates_all = [], {}
    for seed in SEEDS:
        wpath = store_root / "weights" / f"{Store(store_root)._stem(name, seed)}.pt"
        if not wpath.exists():
            continue
        model = build_neural("AFT", 95, list(ALL_FEATURES), AFT_BASE, variant).to(DEVICE)
        model.load_state_dict(torch.load(wpath, map_location=DEVICE))
        model.eval()
        model.set_record(True)
        rec = {"qk_std": [], "fin_std": [], "fin_absmean": [], "weight_sum": [], "temp_std": []}
        gates = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                _, aux = model(X[i:i + 128], return_aux=True)
                gates.append(aux["gates"].cpu().numpy())
                for layer in model.layers:
                    for k, v in layer.attention.attention.last.items():
                        rec[k].append(v)
        model.set_record(False)
        g = np.concatenate(gates)
        gates_all[seed] = g
        ent = -(g * np.log(g + 1e-12)).sum(1) / np.log(G)
        row = {"model": name, "seed": seed, "norm_entropy": float(ent.mean()),
               "max_gate": float(g.max(1).mean()), "min_gate": float(g.min(1).mean()),
               "gate_time_std": float(g.std(0).mean()), "gate_range": float(g.max() - g.min())}
        row.update({k: float(np.mean(v)) for k, v in rec.items() if v})
        row["fin_over_qk"] = row.get("fin_std", np.nan) / row["qk_std"]
        for j, grp in enumerate(FEATURE_GROUPS):
            row[f"gate_{grp}"] = float(g[:, j].mean())
        rows.append(row)
    return pd.DataFrame(rows), gates_all


def aapl_gates():
    df = load_master_dataset()
    sd = make_splits(df)
    root = ROOT / "results" / "aapl"
    a, ga = gate_stats("AFT", {}, sd, root)
    r, gr = gate_stats("AFT-R (renorm. gate)", {"gate_renorm": True}, sd, root)
    tab = pd.concat([a, r], ignore_index=True)
    tab.to_csv(OUT / "gate_stats_aapl.csv", index=False)
    for name, key in (("AFT", "Aft"), ("AFT-R (renorm. gate)", "Aftr")):
        t = tab[tab.model == name]
        if t.empty:
            continue
        NUM[f"g{key}Entropy"] = fmt(t.norm_entropy.mean(), 4)
        NUM[f"g{key}EntropyMin"] = fmt(t.norm_entropy.min(), 4)
        NUM[f"g{key}MaxGate"] = fmt(t.max_gate.mean(), 3)
        NUM[f"g{key}MinGate"] = fmt(t.min_gate.mean(), 3)
        NUM[f"g{key}TimeStd"] = fmt(t.gate_time_std.mean(), 4)
        NUM[f"g{key}FinQk"] = fmt(t.fin_over_qk.mean(), 3)
        NUM[f"g{key}FinAbs"] = fmt(t.fin_absmean.mean(), 3)
        NUM[f"g{key}WeightSum"] = fmt(t.weight_sum.mean(), 3)
        NUM[f"g{key}TempQk"] = fmt((t.temp_std / t.qk_std).mean(), 3)
        NUM[f"g{key}QkStd"] = fmt(t.qk_std.mean(), 2)
    NUM["gUniform"] = fmt(1 / len(FEATURE_GROUPS), 4)
    # figure
    if ga:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = list(FEATURE_GROUPS)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=False)
        for ax, (title, gd) in zip(axes, (("AFT (original gating)", ga), ("AFT-R (renormalised gate)", gr))):
            if not gd:
                ax.axis("off")
                continue
            g = gd[SEEDS[0]] if SEEDS[0] in gd else next(iter(gd.values()))
            ax.boxplot([g[:, j] for j in range(len(groups))], showfliers=False, widths=0.6)
            ax.axhline(1 / len(groups), ls="--", lw=0.8, color="k")
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels(groups, rotation=60, fontsize=6.5)
            ax.set_title(title, fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_ylabel("gate weight $w_g$", fontsize=7)
        fig.tight_layout()
        (ROOT / "figures").mkdir(exist_ok=True)
        fig.savefig(ROOT / "figures" / "fig_gates.pdf")
        plt.close(fig)


TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "JPM", "XOM", "JNJ", "WMT", "SPY"]


def covariate_shift():
    rows = []
    level = [c for c in ALL_FEATURES if c in LEVEL_FEATURES]
    stat = [c for c in ALL_FEATURES if c not in LEVEL_FEATURES]
    for t in TICKERS:
        df = build_features(download_ohlcv(t))
        sd = make_splits(df)
        Z = sd.Z
        tr, te = np.arange(0, sd.train_end), sd.test_idx
        li = [ALL_FEATURES.index(c) for c in level]
        si = [ALL_FEATURES.index(c) for c in stat]
        r = df["Return"].to_numpy()
        rows.append({
            "asset": t,
            "vol_ratio": float(r[te].std() / r[tr].std()),
            "price_ratio": float(df.Close.iloc[te].mean() / df.Close.iloc[tr].mean()),
            "absz_level_test": float(np.abs(Z[np.ix_(te, li)]).mean()),
            "absz_stat_test": float(np.abs(Z[np.ix_(te, si)]).mean()),
            "frac_absz_gt5": float((np.abs(Z[te]) > 5).mean()),
            "test_bh_return": float(df.Close.iloc[te[-1]] / df.Close.iloc[te[0]] - 1),
        })
    cs = pd.DataFrame(rows)
    ma = ROOT / "results" / "summary" / "multiasset_summary.csv"
    if ma.exists():
        s = pd.read_csv(ma)
        aft = s[s.model == "AFT"].set_index("asset")
        cs["aft_sr"] = cs.asset.map(aft.SR)
        cs["aft_mae_ratio"] = cs.asset.map(aft.MAE_ratio)
        cs["aft_da_minus_up"] = cs.asset.map(aft.DA - aft.up)
        from scipy.stats import spearmanr
        for metric in ("aft_mae_ratio", "aft_sr", "aft_da_minus_up"):
            if cs[metric].notna().sum() >= 5:
                rho, p = spearmanr(cs.absz_level_test, cs[metric], nan_policy="omit")
                NUM[f"rho{''.join(w.capitalize() for w in metric.replace('aft_', '').split('_'))}"] = fmt(rho, 2)
                NUM[f"rhoP{''.join(w.capitalize() for w in metric.replace('aft_', '').split('_'))}"] = fmt(p, 3)
    cs.to_csv(OUT / "covariate_shift.csv", index=False)
    for _, x in cs.iterrows():
        k = x.asset.capitalize()
        NUM[f"cs{k}AbsZLevel"] = fmt(x.absz_level_test, 2)
        NUM[f"cs{k}AbsZStat"] = fmt(x.absz_stat_test, 2)
        NUM[f"cs{k}VolRatio"] = fmt(x.vol_ratio, 2)
        NUM[f"cs{k}PriceRatio"] = fmt(x.price_ratio, 1)
    NUM["csMedianAbsZLevel"] = fmt(cs.absz_level_test.median(), 2)
    return cs


if __name__ == "__main__":
    aapl_gates()
    covariate_shift()
    (OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
    print(json.dumps(NUM, indent=1))
