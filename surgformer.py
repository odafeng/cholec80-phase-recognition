"""Surgformer temporal head (causal/online adaptation) for surgical phase recognition.

Clean-room causal adaptation of Surgformer (Yang et al., MICCAI 2024). The original
contribution is Hierarchical Temporal Attention (HTA): aggregating temporal context at
multiple scales. We keep that idea on the shared feature stream as a pure-attention
multi-stage transformer where the attention heads are split into groups, each restricted
to a different *causal* temporal window (local -> global), giving a hierarchical
temporal receptive field. Every operation is causal (banded / lower-triangular masks,
1x1 convs, per-frame LayerNorm) so the head is strictly online.

Distinct from the other heads: pure multi-scale attention, no dilated convolution
(unlike ASFormer / TeCNO) and no Conformer conv module (unlike LoViT). This head is
online by construction; `causal` must be True.

Interface matches MultiStageTCN / ASFormer:
    input : (B, in_dim, T)
    output: list of (B, num_classes, T), one per stage
    forward(x, return_boundary=True) -> (outputs, boundary_logits (B,1,T))
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _band_mask(T, w, device, dtype):
    """Additive causal-band mask (T,T): position i may attend j iff 0 <= i-j < w."""
    i = torch.arange(T, device=device)
    diff = i[:, None] - i[None, :]
    allowed = (diff >= 0) & (diff < w)
    mask = torch.zeros(T, T, device=device, dtype=dtype)
    return mask.masked_fill(~allowed, float("-inf"))


class HierTemporalAttn(nn.Module):
    """Causal multi-head attention with head groups at different temporal windows.
    `windows`: tuple of ints (band width) or None (full causal). Heads are split evenly
    across the groups, so each frame mixes local and global causal context (HTA)."""

    def __init__(self, d, heads, windows, dropout=0.2):
        super().__init__()
        assert d % heads == 0 and heads % len(windows) == 0
        self.h, self.dk = heads, d // heads
        self.windows = windows
        self.g = heads // len(windows)               # heads per window group
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                             # (B,T,d)
        B, T, d = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]             # (B,h,T,dk)
        outs = []
        for gi, w in enumerate(self.windows):
            sl = slice(gi * self.g, (gi + 1) * self.g)
            if w is None:
                og = F.scaled_dot_product_attention(q[:, sl], k[:, sl], v[:, sl],
                                                    is_causal=True)
            else:
                mask = _band_mask(T, w, x.device, q.dtype)
                og = F.scaled_dot_product_attention(q[:, sl], k[:, sl], v[:, sl],
                                                    attn_mask=mask)
            outs.append(og)
        o = torch.cat(outs, dim=1).transpose(1, 2).reshape(B, T, d)
        return self.drop(self.proj(o))


class SurgLayer(nn.Module):
    def __init__(self, d, heads, windows, dropout=0.2):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = HierTemporalAttn(d, heads, windows, dropout)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, x):                            # (B,T,d)
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x


class SurgStage(nn.Module):
    def __init__(self, in_dim, d, num_classes, num_layers, heads, windows):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, d, 1)
        self.layers = nn.ModuleList([SurgLayer(d, heads, windows) for _ in range(num_layers)])
        self.conv_out = nn.Conv1d(d, num_classes, 1)

    def forward(self, x, return_feat=False):         # x: (B,in_dim,T)
        h = self.conv_in(x).transpose(1, 2)          # (B,T,d)
        for layer in self.layers:
            h = layer(h)
        feat = h.transpose(1, 2)                      # (B,d,T)
        logits = self.conv_out(feat)
        if return_feat:
            return logits, feat
        return logits


class Surgformer(nn.Module):
    """Encoder + (num_stages-1) decoder refinements (same multi-stage contract as
    ASFormer: later stages refine the previous stage's softmax)."""

    def __init__(self, in_dim=2048, num_classes=7, num_stages=2, num_layers=6,
                 d=192, heads=6, causal=True, boundary=False, windows=(32, 256, None)):
        super().__init__()
        assert causal, "Surgformer is an online head; use causal=True"
        self.stage1 = SurgStage(in_dim, d, num_classes, num_layers, heads, windows)
        self.stages = nn.ModuleList([
            copy.deepcopy(SurgStage(num_classes, d, num_classes, num_layers, heads, windows))
            for _ in range(num_stages - 1)])
        self.boundary = boundary
        if boundary:
            self.boundary_head = nn.Conv1d(d, 1, 1)

    def forward(self, x, return_boundary=False):
        out, feat = self.stage1(x, return_feat=True)
        outputs = [out]
        for stage in self.stages:
            out, feat = stage(F.softmax(out, dim=1), return_feat=True)
            outputs.append(out)
        if return_boundary:
            return outputs, self.boundary_head(feat)
        return outputs


if __name__ == "__main__":
    m = Surgformer(causal=True)
    x = torch.randn(1, 2048, 400)
    ys = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f"{len(ys)} stages, out {tuple(ys[-1].shape)}, params {n/1e6:.2f}M")
    outs, b = Surgformer(causal=True, boundary=True)(torch.randn(1, 2048, 200),
                                                     return_boundary=True)
    assert b.shape == (1, 1, 200)
    print("Surgformer boundary head OK", tuple(b.shape))
