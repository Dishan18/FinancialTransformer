"""AFT v2: panel (multi-stock) research pipeline.

universe  ticker list with sectors
data      download, cache and quality-check daily OHLCV
features  stationary, scale-free feature engineering (+ cross-sectional normalisation)
panel     panel tensors, purged walk-forward splits, GPU window gathering
"""
