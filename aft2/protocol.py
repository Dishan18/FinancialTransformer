"""Research protocol constants (fixed on 2026-11-21, before any v2 model was evaluated).

* Walk-forward out-of-sample years: 2016-2025, yearly refits, expanding training window,
  252-day validation window, 5-day purges (``aft2.panel.walk_forward``).
* Development period: out-of-sample years 2016-2022.  All design iterations (features, models,
  meta-models) are compared on these years only.
* Sealed holdout: out-of-sample years 2023-2025.  Holdout metrics are computed once, for designs
  frozen at the end of the iteration loop, and reported whatever they show.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEV_FIRST_YEAR = 2016
DEV_LAST_YEAR = 2022
HOLDOUT_FIRST_YEAR = 2023
HOLDOUT_LAST_YEAR = 2025

# Disclosure: before the seal was defined, one timing smoke test (1 training epoch of an MLP, a
# Transformer and the AFT on the 2024 refit) printed validation rank ICs computed on 2023 dates
# (0.038-0.049).  No design decision used these numbers.  No holdout *test* prediction was evaluated.


# ----------------------------------------------------------------------------- experiment locations
PANELS = {
    # name: (panel file, rank cache)
    "top100": ("panel_v2.npz", "panel_v2_ranks.npy"),            # 100 hand-picked large caps (h = 5 only)
    "top100pit": ("panel_top100.npz", "panel_top100_ranks.npy"),  # same universe, new format, h = 5 and 21
    "sp500": ("panel_sp500.npz", "panel_sp500_ranks.npy"),      # point-in-time S&P 500 members, h = 5 and 21
}


def wf_dir(panel: str, horizon: int) -> Path:
    if panel == "top100" and horizon == 5:
        return ROOT / "results_v2" / "wf"                          # first experiments (kept for provenance)
    return ROOT / "results_v2" / f"wf_{panel}_h{horizon}"


def load_panel(panel: str, horizon: int):
    from .panel import Panel
    if panel == "top100" and horizon != 5:
        panel = "top100pit"
    f, r = PANELS[panel]
    P = Panel.load(ROOT / "data" / f, horizon=horizon)
    R = P.ranked_stock_features(ROOT / "data" / r)
    return P, R
