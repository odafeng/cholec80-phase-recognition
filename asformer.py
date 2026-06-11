"""ASFormer-style temporal head (causal/online variant) for surgical phase recognition.

A clean-room implementation of the ASFormer idea (Yi et al., BMVC 2021): combine
dilated temporal convolutions (local, multi-scale) with self-attention (global)
in an encoder + multi-stage decoder refinement. We use the CAUSAL variant
(left-only conv padding + causal SDPA) so each frame sees only the past -> online.

This is a THIRD architecture, distinct from:
  - TeCNO / MS-TCN : dilated conv only (no attention)
  - LoViT          : Conformer (attention + local conv, no exponential dilation)
so it strengthens the "BUA is architecture-agnostic" claim.

Interface matches MultiStageTCN / LoViT exactly:
    input : (B, in_dim, T)
    output: list of (B, num_classes, T), one per stage
    forward(x, return_boundary=True) -> (outputs, boundary_logits (B,1,T))
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttLayer(nn.Module):
    """Dilated causal conv (local, multi-scale) + causal self-attention (global)."""

    def __init__(self, d, heads, dilation, causal=True, dropout=0.2):
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.pad = 2 * dilation if causal else dilation
        self.conv = nn.Conv1d(d, d, 3, padding=0 if causal else dilation, dilation=dilation)
        # attention
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                              # x: (B, d, T)
        # --- local: dilated causal conv ---
        if self.causal:
            h = F.pad(x, (self.pad, 0))
            h = self.conv(h)
        else:
            h = self.conv(x)
        h = F.relu(h)
        x = x + self.drop(self.ff(h))                  # residual

        # --- global: causal self-attention ---
        B, d, T = x.shape
        z = self.norm(x.transpose(1, 2))               # (B, T, d)
        qkv = self.qkv(z).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               # (B, heads, T, dk)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        o = o.transpose(1, 2).reshape(B, T, d)
        o = self.proj(o).transpose(1, 2)               # (B, d, T)
        return x + self.drop(o)


class SingleStageASFormer(nn.Module):
    def __init__(self, num_layers, d, heads, in_dim, num_classes, causal):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, d, 1)
        self.layers = nn.ModuleList([
            AttLayer(d, heads, dilation=2 ** i, causal=causal)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(d, num_classes, 1)

    def forward(self, x, return_feat=False):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        logits = self.conv_out(out)
        if return_feat:
            return logits, out
        return logits


class ASFormer(nn.Module):
    """Encoder + (num_stages-1) decoder refinements. Same multi-stage contract as
    MultiStageTCN: later stages refine the previous stage's softmax."""

    def __init__(self, in_dim=2048, num_classes=7, num_stages=3, num_layers=9,
                 d=64, heads=1, causal=True, boundary=False):
        super().__init__()
        self.stage1 = SingleStageASFormer(num_layers, d, heads, in_dim, num_classes, causal)
        self.stages = nn.ModuleList([
            copy.deepcopy(SingleStageASFormer(num_layers, d, heads,
                                              num_classes, num_classes, causal))
            for _ in range(num_stages - 1)
        ])
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
    for causal in (True, False):
        m = ASFormer(causal=causal)
        x = torch.randn(1, 2048, 800)
        ys = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f"causal={causal}: {len(ys)} stages, out {tuple(ys[-1].shape)}, params {n/1e6:.2f}M")
    # boundary head
    mb = ASFormer(causal=True, boundary=True)
    outs, b = mb(torch.randn(1, 2048, 300), return_boundary=True)
    assert b.shape == (1, 1, 300)
    print("ASFormer boundary head OK", tuple(b.shape))
