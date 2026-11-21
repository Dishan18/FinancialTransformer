"""Model definitions.

``AdaptiveFinancialTransformer`` reproduces the architecture of Notebooks 12.1 / 13 line by
line (same modules, same parameter shapes, same initialisation order) and adds switches used
for the ablation study.  All sub-modules are always instantiated, so switching a component off
changes only the forward pass and never the random initialisation of the remaining weights.

Ablation switches
-----------------
use_regime            False -> uniform gates 1/G (the regime encoder and gate network are bypassed;
                      the regime vector has no other consumer, so this also removes the encoder).
use_financial_context False -> no financial-context bias B_fin.
use_temporal_bias     False -> no relative temporal bias B_temp.
gate_renorm           True  -> "AFT-R": the product of the static group weights (softmax) and the
                      dynamic gates (softmax) is renormalised to sum to one, removing the O(1/G)
                      attenuation of the original formulation (see paper, Sec. "Diagnosis").
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def _group_perm(feature_groups) -> torch.Tensor:
    return torch.tensor([i for idx in feature_groups.values() for i in idx], dtype=torch.long)


class MarketRegimeEncoder(nn.Module):
    """Per-group MLP embedding, attention pooling over time, and an MLP fusion layer.

    ``forward`` evaluates all group MLPs with block-diagonal / batched matrix products; this is
    mathematically identical to the per-group loop of the original notebooks (``forward_loop``)
    and uses exactly the same parameters.
    """

    def __init__(self, feature_groups, hidden_dim=64, regime_dim=32):
        super().__init__()
        self.feature_groups = feature_groups
        self.group_embeddings = nn.ModuleDict({
            g: nn.Sequential(nn.Linear(len(idx), hidden_dim), nn.GELU(), nn.Linear(hidden_dim, regime_dim))
            for g, idx in feature_groups.items()
        })
        self.temporal_score = nn.Linear(regime_dim, 1)
        self.fusion = nn.Sequential(
            nn.Linear(len(feature_groups) * regime_dim, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, regime_dim))
        self.register_buffer("perm", _group_perm(feature_groups), persistent=False)

    def forward_loop(self, x):
        vecs = []
        for g, idx in self.feature_groups.items():
            e = self.group_embeddings[g](x[:, :, idx])
            a = torch.softmax(self.temporal_score(e), dim=1)
            vecs.append((a * e).sum(dim=1))
        return self.fusion(torch.cat(vecs, dim=1))

    def forward(self, x):
        B, L, _ = x.shape
        mods = list(self.group_embeddings.values())
        G = len(mods)
        w1 = torch.block_diag(*[m[0].weight.t() for m in mods])          # (D, G*H)
        b1 = torch.cat([m[0].bias for m in mods])
        h = F.gelu(x[:, :, self.perm] @ w1 + b1).view(B, L, G, -1)      # (B, L, G, H)
        w2 = torch.stack([m[2].weight for m in mods])                     # (G, R, H)
        b2 = torch.stack([m[2].bias for m in mods])                       # (G, R)
        e = torch.einsum("blgh,grh->blgr", h, w2) + b2                    # (B, L, G, R)
        a = torch.softmax(self.temporal_score(e), dim=1)
        pooled = (a * e).sum(dim=1)                                       # (B, G, R)
        return self.fusion(pooled.reshape(B, -1))


class AdaptiveGateNetwork(nn.Module):
    def __init__(self, regime_dim, num_groups):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(regime_dim, 64), nn.GELU(), nn.Dropout(0.15),
                                     nn.Linear(64, num_groups))

    def forward(self, regime):
        return torch.softmax(self.network(regime), dim=-1)


class AdaptiveFinancialContext(nn.Module):
    def __init__(self, d_head, feature_groups):
        super().__init__()
        self.feature_groups = feature_groups
        self.projections = nn.ModuleDict({g: nn.Linear(len(idx), d_head) for g, idx in feature_groups.items()})
        self.register_buffer("perm", _group_perm(feature_groups), persistent=False)

    def similarities_loop(self, x):
        mats = []
        for g, idx in self.feature_groups.items():
            e = F.normalize(self.projections[g](x[:, :, idx]), p=2, dim=-1)
            mats.append(torch.matmul(e, e.transpose(-2, -1)))
        return torch.stack(mats, dim=1)

    def similarities(self, x):
        """Cosine-similarity matrices, shape (B, G, L+1, L+1) (vectorised; equals the loop)."""
        B, L1, _ = x.shape
        mods = list(self.projections.values())
        w = torch.block_diag(*[m.weight.t() for m in mods])                # (D, G*d)
        b = torch.cat([m.bias for m in mods])
        e = (x[:, :, self.perm] @ w + b).view(B, L1, len(mods), -1)
        e = F.normalize(e, p=2, dim=-1).permute(0, 2, 1, 3)                # (B, G, L+1, d)
        return torch.matmul(e, e.transpose(-2, -1))


class RelativeTemporalBias(nn.Module):
    def __init__(self, max_length, num_heads):
        super().__init__()
        self.max_length = max_length
        self.bias = nn.Parameter(torch.zeros(num_heads, 2 * max_length - 1))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, length):
        pos = torch.arange(length, device=self.bias.device)
        rel = pos[:, None] - pos[None, :] + self.max_length - 1
        return self.bias[:, rel]


class ModularFinancialAttention(nn.Module):
    def __init__(self, d_head, num_heads, feature_groups, seq_len):
        super().__init__()
        self.scale = math.sqrt(d_head)
        self.num_groups = len(feature_groups)
        self.context = AdaptiveFinancialContext(d_head, feature_groups)
        self.group_weights = nn.Parameter(torch.ones(self.num_groups))
        self.financial_weight = nn.Parameter(torch.tensor(0.5))
        self.temporal_bias = RelativeTemporalBias(seq_len + 1, num_heads)
        self.temporal_weight = nn.Parameter(torch.tensor(0.5))
        self.cfg = {}
        self.last = {}

    def forward(self, Q, K, V, raw, gates):
        cfg = self.cfg
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        record = cfg.get("record", False)
        if record:
            self.last = {"qk_std": scores.detach().std().item()}
        if cfg.get("use_financial_context", True):
            sims = self.context.similarities(raw)                           # (B, G, L+1, L+1)
            static = torch.softmax(self.group_weights, dim=0)                # (G,)
            w = gates * static.unsqueeze(0)                                  # (B, G)
            if cfg.get("gate_renorm", False):
                w = w / w.sum(dim=-1, keepdim=True)
            fin = torch.einsum("bg,bgij->bij", w, sims).unsqueeze(1)         # (B, 1, L+1, L+1)
            fin = torch.sigmoid(self.financial_weight) * fin
            scores = scores + fin
            if record:
                self.last.update({"fin_std": fin.detach().std().item(),
                                  "fin_absmean": fin.detach().abs().mean().item(),
                                  "weight_sum": w.detach().sum(-1).mean().item()})
        if cfg.get("use_temporal_bias", True):
            tb = torch.sigmoid(self.temporal_weight) * self.temporal_bias(Q.shape[-2]).unsqueeze(0)
            scores = scores + tb
            if record:
                self.last["temp_std"] = tb.detach().std().item()
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn


class ModularMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, feature_groups, seq_len):
        super().__init__()
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attention = ModularFinancialAttention(self.head_dim, num_heads, feature_groups, seq_len)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, raw, gates):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out, attn = self.attention(q, k, v, raw, gates)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, L, self.d_model)), attn


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.15):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(ff_dim, d_model))

    def forward(self, x):
        return self.network(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, feature_groups, seq_len, dropout=0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = ModularMultiHeadAttention(d_model, num_heads, feature_groups, seq_len)
        self.ffn = FeedForward(d_model, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, raw, gates):
        a, w = self.attention(self.norm1(x), raw, gates)
        x = x + self.dropout(a)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, w


class AdaptiveFinancialTransformer(nn.Module):
    def __init__(self, input_dim, feature_groups, seq_len, d_model=128, num_heads=8, ff_dim=256,
                 num_layers=2, dropout=0.15, use_regime=True, use_financial_context=True,
                 use_temporal_bias=True, gate_renorm=False):
        super().__init__()
        self.feature_groups = feature_groups
        self.num_groups = len(feature_groups)
        self.embedding = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, ff_dim, feature_groups, seq_len, dropout)
                                     for _ in range(num_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))
        self.market_encoder = MarketRegimeEncoder(feature_groups)
        self.gate_network = AdaptiveGateNetwork(32, self.num_groups)
        self.use_regime = use_regime
        self.cfg = {"use_financial_context": use_financial_context, "use_temporal_bias": use_temporal_bias,
                    "gate_renorm": gate_renorm, "record": False}
        for layer in self.layers:
            layer.attention.attention.cfg = self.cfg

    def set_record(self, flag: bool):
        self.cfg["record"] = flag

    def forward(self, x, return_aux: bool = False):
        B = x.shape[0]
        if self.use_regime:
            regime = self.market_encoder(x)
            gates = self.gate_network(regime)
        else:
            regime = None
            gates = torch.full((B, self.num_groups), 1.0 / self.num_groups, device=x.device)
        raw = torch.cat([torch.zeros(B, 1, x.size(-1), device=x.device), x], dim=1)
        h = self.embedding(x)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        h = self.dropout(self.position(h))
        maps = []
        for layer in self.layers:
            h, w = layer(h, raw, gates)
            maps.append(w)
        pred = self.head(h[:, 0]).squeeze(-1)
        if return_aux:
            return pred, {"gates": gates, "regime": regime, "attn": maps}
        return pred

    def active_parameter_count(self) -> int:
        """Parameters that influence the output under the current switches."""
        skip = set()
        if not self.use_regime:
            skip |= {id(p) for p in self.market_encoder.parameters()}
            skip |= {id(p) for p in self.gate_network.parameters()}
        for layer in self.layers:
            att = layer.attention.attention
            if not self.cfg["use_financial_context"]:
                skip |= {id(p) for p in att.context.parameters()}
                skip |= {id(att.group_weights), id(att.financial_weight)}
                if self.use_regime:  # gates would have no consumer
                    skip |= {id(p) for p in self.market_encoder.parameters()}
                    skip |= {id(p) for p in self.gate_network.parameters()}
            if not self.cfg["use_temporal_bias"]:
                skip |= {id(p) for p in att.temporal_bias.parameters()}
                skip.add(id(att.temporal_weight))
        return sum(p.numel() for p in self.parameters() if id(p) not in skip)


# ----------------------------------------------------------------------------- baselines

class RNNRegressor(nn.Module):
    """LSTM / GRU baseline of Notebook 03 (2 layers, 128 hidden units, dropout 0.2)."""

    def __init__(self, input_dim, cell="lstm", hidden=128, layers=2, dropout=0.2):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(input_size=input_dim, hidden_size=hidden, num_layers=layers, batch_first=True,
                       dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        h = h[0] if isinstance(h, tuple) else h
        return self.fc(h[-1]).squeeze(-1)


class VanillaTransformer(nn.Module):
    """Standard pre-norm Transformer encoder with the same embedding, CLS token, positional
    encoding, depth, width and regression head as the AFT, but no financial/temporal biases,
    no regime encoder and no gate network."""

    def __init__(self, input_dim, d_model=128, num_heads=8, ff_dim=256, num_layers=2, dropout=0.15):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        B = x.shape[0]
        h = torch.cat([self.cls_token.expand(B, -1, -1), self.embedding(x)], dim=1)
        h = self.encoder(self.dropout(self.position(h)))
        return self.head(h[:, 0]).squeeze(-1)


class DLinear(nn.Module):
    """DLinear (Zeng et al., AAAI 2023) adapted to a scalar target: each input channel is split
    into a moving-average trend and a remainder, each mapped over time by a shared linear layer
    (L -> 1); a final linear layer combines the D channel outputs into the forecast."""

    def __init__(self, input_dim, seq_len, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(seq_len, 1)
        self.seasonal = nn.Linear(seq_len, 1)
        self.readout = nn.Linear(input_dim, 1)

    def forward(self, x):                       # x: (B, L, D)
        pad = (self.kernel - 1) // 2
        xt = x.transpose(1, 2)                  # (B, D, L)
        front = xt[:, :, :1].repeat(1, 1, pad)
        back = xt[:, :, -1:].repeat(1, 1, self.kernel - 1 - pad)
        trend = F.avg_pool1d(torch.cat([front, xt, back], dim=-1), self.kernel, stride=1)
        seasonal = xt - trend
        out = self.trend(trend).squeeze(-1) + self.seasonal(seasonal).squeeze(-1)   # (B, D)
        return self.readout(out).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    if hasattr(model, "active_parameter_count"):
        return model.active_parameter_count()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
