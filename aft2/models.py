"""Panel models.  Sequence models are re-used from ``aft.models`` (the AFT is architecture-identical
to v1; only its feature groups change).  Tabular models act on the current feature vector x_t."""
from __future__ import annotations

import torch
import torch.nn as nn

from aft.models import AdaptiveFinancialTransformer, DLinear, RNNRegressor, VanillaTransformer, count_parameters  # noqa: F401


class MLP(nn.Module):
    """Feed-forward network on x_t (the 'NN3' architecture of Gu, Kelly and Xiu: 32-16-8, ReLU)."""

    def __init__(self, input_dim, hidden=(32, 16, 8), dropout=0.1):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):            # x: (B, K)
        return self.net(x).squeeze(-1)


SEQUENCE_KINDS = {"LSTM", "GRU", "DLinear", "Transformer", "AFT", "AFT-R"}


def build(kind: str, K: int, groups: dict, L: int, d_model: int = 64, heads: int = 4, layers: int = 2,
          dropout: float = 0.1):
    if kind == "MLP":
        return MLP(K)
    if kind in ("LSTM", "GRU"):
        return RNNRegressor(K, cell=kind.lower(), hidden=64, layers=2, dropout=dropout)
    if kind == "DLinear":
        return DLinear(K, L, kernel=min(25, L - (1 - L % 2)))
    if kind == "Transformer":
        return VanillaTransformer(K, d_model=d_model, num_heads=heads, ff_dim=2 * d_model, num_layers=layers,
                                  dropout=dropout)
    if kind in ("AFT", "AFT-R"):
        return AdaptiveFinancialTransformer(K, groups, L, d_model=d_model, num_heads=heads, ff_dim=2 * d_model,
                                            num_layers=layers, dropout=dropout, gate_renorm=(kind == "AFT-R"))
    raise ValueError(kind)
