"""Trans-SVNet temporal head (causal/online adaptation) for surgical phase recognition.

Clean-room causal adaptation of Trans-SVNet (Gao et al., MICCAI 2021). The original
fuses each frame's SPATIAL embedding with TeCNO TEMPORAL embeddings through a light
transformer. We keep that two-part design on the shared feature stream and make every
operation causal so the head is strictly online:
  Stage 1 : causal dilated TCN (TeCNO-style) -> temporal embeddings + phase logits.
  Stage 2 : a transformer whose QUERY is the per-frame spatial embedding (1x1 proj of
            the input feature) and whose KEY/VALUE is the temporal-embedding stream;
            causal self-attention (refines memory) + causal cross-attention (spatial
            query attends past temporal memory) -> refined logits.

Distinct from the other heads: TeCNO (conv only), LoViT (Conformer), ASFormer (dilated
conv + attention). Here the characteristic element is spatial-query x temporal-memory
cross-attention aggregation. This head is online by construction; `causal` must be True.

Interface matches MultiStageTCN / ASFormer:
    input : (B, in_dim, T)
    output: list of (B, num_classes, T), one per stage
    forward(x, return_boundary=True) -> (outputs, boundary_logits (B,1,T))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mstcn import SingleStageTCN


class CausalMHA(nn.Module):
    """Multi-head attention with a causal mask. Query and key/value may differ but
    share the time length T, so `is_causal=True` gives the right lower-triangular mask
    (query t attends only key positions <= t)."""

    def __init__(self, d, heads, dropout=0.2):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):                              # (B,T,d) -> (B,h,T,dk)
        B, T, _ = x.shape
        return x.reshape(B, T, self.h, self.dk).transpose(1, 2)

    def forward(self, q_in, kv_in):                  # (B,T,d), (B,T,d)
        B, T, d = q_in.shape
        o = F.scaled_dot_product_attention(
            self._split(self.q(q_in)), self._split(self.k(kv_in)),
            self._split(self.v(kv_in)), is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, d)
        return self.drop(self.proj(o))


class FusionBlock(nn.Module):
    """Causal self-attention over temporal memory, then causal spatial->temporal
    cross-attention, then a feed-forward — all pre-norm with residuals."""

    def __init__(self, d, heads, dropout=0.2):
        super().__init__()
        self.n_self = nn.LayerNorm(d)
        self.self_attn = CausalMHA(d, heads, dropout)
        self.n_q = nn.LayerNorm(d)
        self.n_kv = nn.LayerNorm(d)
        self.cross = CausalMHA(d, heads, dropout)
        self.n_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, spatial, temporal):            # (B,T,d), (B,T,d)
        mem = temporal + self.self_attn(self.n_self(temporal), self.n_self(temporal))
        x = spatial + self.cross(self.n_q(spatial), self.n_kv(mem))
        x = x + self.ff(self.n_ff(x))
        return x


class TransSVNet(nn.Module):
    def __init__(self, in_dim=2048, num_classes=7, num_stages=2, num_layers=9,
                 d=128, heads=8, causal=True, boundary=False, blocks=2):
        super().__init__()
        assert causal, "TransSVNet is an online head; use causal=True"
        # num_stages kept for interface parity; the architecture is TCN(stage1) + the
        # fusion transformer(stage2) -> always a 2-output multi-stage contract.
        self.tcn = SingleStageTCN(num_layers, d, in_dim, num_classes, causal)
        self.spatial_proj = nn.Conv1d(in_dim, d, 1)
        self.blocks = nn.ModuleList([FusionBlock(d, heads) for _ in range(blocks)])
        self.conv_out = nn.Conv1d(d, num_classes, 1)
        self.boundary = boundary
        if boundary:
            self.boundary_head = nn.Conv1d(d, 1, 1)

    def forward(self, x, return_boundary=False):     # x: (B, in_dim, T)
        logits1, temporal = self.tcn(x, return_feat=True)      # (B,K,T), (B,d,T)
        spatial = self.spatial_proj(x)                         # (B,d,T)
        s = spatial.transpose(1, 2)                            # (B,T,d)
        t = temporal.transpose(1, 2)                           # (B,T,d)
        for blk in self.blocks:
            s = blk(s, t)
        feat = s.transpose(1, 2)                               # (B,d,T)
        outputs = [logits1, self.conv_out(feat)]
        if return_boundary:
            return outputs, self.boundary_head(feat)
        return outputs


if __name__ == "__main__":
    m = TransSVNet(causal=True)
    x = torch.randn(1, 2048, 400)
    ys = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f"{len(ys)} stages, out {tuple(ys[-1].shape)}, params {n/1e6:.2f}M")
    outs, b = TransSVNet(causal=True, boundary=True)(torch.randn(1, 2048, 200),
                                                     return_boundary=True)
    assert b.shape == (1, 1, 200)
    print("TransSVNet boundary head OK", tuple(b.shape))
