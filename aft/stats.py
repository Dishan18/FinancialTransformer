"""Statistical utilities: confidence intervals over seeds, paired tests, Diebold-Mariano."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(values, level: float = 0.95) -> tuple[float, float, float]:
    """Mean, sample standard deviation and the half-width of a t confidence interval."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n < 2:
        return float(v.mean()), 0.0, 0.0
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n))
    return float(v.mean()), sd, half


def paired_comparison(a, b) -> dict:
    """Compare paired samples a (candidate) and b (reference): a - b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    mean, _, half = mean_ci(d)
    if sd == 0:
        t_p = 1.0 if mean == 0 else 0.0
        dz = 0.0 if mean == 0 else np.inf * np.sign(mean)
    else:
        t_p = float(stats.ttest_rel(a, b).pvalue)
        dz = float(mean / sd)
    try:
        w_p = float(stats.wilcoxon(a, b, method="exact").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w_p = float("nan")
    return {"mean_diff": float(mean), "ci_low": float(mean - half), "ci_high": float(mean + half),
            "t_p": t_p, "wilcoxon_p": w_p, "d_z": dz, "n": n}


def _autocov(x: np.ndarray, k: int) -> float:
    xc = x - x.mean()
    return float(np.dot(xc[k:], xc[: len(xc) - k]) / len(xc))


def diebold_mariano(e_model: np.ndarray, e_ref: np.ndarray, h: int = 5, power: int = 2) -> dict:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold (1997) small-sample correction.

    d_t = |e_model,t|^p - |e_ref,t|^p.  Negative statistic => the model has lower loss.
    The long-run variance uses autocovariances up to lag h-1 (rectangular kernel); if that is
    not positive, a Bartlett (Newey-West) kernel is used instead.  Two-sided p-value from t_{T-1}.
    """
    d = np.abs(np.asarray(e_model, float)) ** power - np.abs(np.asarray(e_ref, float)) ** power
    T = len(d)
    gam = [_autocov(d, k) for k in range(h)]
    lrv = gam[0] + 2 * sum(gam[1:])
    if lrv <= 0:
        lrv = gam[0] + 2 * sum((1 - k / h) * gam[k] for k in range(1, h))
    dm = d.mean() / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), T - 1))
    return {"stat": stat, "p": p, "mean_loss_diff": float(d.mean()), "T": T}


def dm_from_differential(d: np.ndarray, h: int = 5) -> dict:
    """Diebold-Mariano test (HLN-corrected) applied to a given loss-differential series d_t."""
    d = np.asarray(d, float)
    T = len(d)
    gam = [_autocov(d, k) for k in range(h)]
    lrv = gam[0] + 2 * sum(gam[1:])
    if lrv <= 0:
        lrv = gam[0] + 2 * sum((1 - k / h) * gam[k] for k in range(1, h))
    dm = d.mean() / np.sqrt(lrv / T)
    stat = float(dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T))
    return {"stat": stat, "p": float(2 * stats.t.sf(abs(stat), T - 1)), "mean_loss_diff": float(d.mean()), "T": T}


def hac_mean_test(x: np.ndarray, lag: int = 4, mu0: float = 0.0) -> dict:
    """t-test of E[x] = mu0 with a Newey-West (Bartlett) long-run variance."""
    x = np.asarray(x, float)
    T = len(x)
    gam = [_autocov(x, k) for k in range(lag + 1)]
    lrv = gam[0] + 2 * sum((1 - k / (lag + 1)) * gam[k] for k in range(1, lag + 1))
    stat = (x.mean() - mu0) / np.sqrt(lrv / T)
    return {"stat": float(stat), "p": float(2 * stats.t.sf(abs(stat), T - 1)), "mean": float(x.mean())}
