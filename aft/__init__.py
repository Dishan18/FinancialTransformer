"""Reproducible research code for the Adaptive Financial Transformer (AFT) study.

Modules
-------
data      Feature engineering (identical to Notebook 01), chronological purged splits,
          sliding-window construction with identical evaluation dates for every model.
models    AFT (identical to Notebooks 12.1/13) plus ablation switches, and baselines.
train     Deterministic training loop shared by every neural model.
backtest  Cost-aware, non-overlapping 5-day long/short backtest.
stats     Seed-level confidence intervals, paired tests, Diebold-Mariano tests.
"""
