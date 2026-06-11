"""LoViT-style temporal head for surgical phase recognition.

A clean-room implementation that captures LoViT's core idea (Liu et al., MedIA
2023): model long-range temporal context with self-attention AND short-range
dynamics with local convolution ("long-short"), causally for online inference,
then refine over multiple stages.

This is NOT a line-by-line reproduction of the official LoViT; it keeps the same
two-stage setup as our MS-TCN/TeCNO (operates on frozen ResNet50 features) so the
ONLY thing that changes vs TeCNO is the temporal model -> a fair comparison.

Interface matches MultiStageTCN exactly:
    input : (B, in_dim, T)          in_dim = 2048
    output: list of (B, num_classes, T), one per refinement stage.

`causal=True`  -> online (each frame sees only the past)  -> compare vs TeCNO.
`causal=False` -> offline (sees future too)               -> compare vs MS-TCN.

Long sequences (T up to ~4000 at 1 fps) are handled by torch's memory-efficient
scaled_dot_product_attention, so we never materialise a TxT matrix.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d, mult=4, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(
            nn.Linear(d, mult * d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mult * d, d), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class CausalConvModule(nn.Module):
    """Local / short-term temporal modelling (Conformer-style conv module)."""

    def __init__(self, d, kernel=7, causal=True, dropout=0.1):
        super().__init__()
        self.causal = causal
        self.pad = kernel - 1 if causal else kernel // 2
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)          # pointwise -> GLU gate
        self.dw = nn.Conv1d(d, d, kernel, groups=d,
                            padding=0 if causal else self.pad)
        # GroupNorm (not BatchNorm): we train with batch_size=1 (one video at a
        # time), where BatchNorm's running stats are unstable and wreck the
        # train/test consistency. GroupNorm is batch-independent.
        self.bn = nn.GroupNorm(8, d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                          # x: (B, T, d)
        res = x
        h = self.norm(x).transpose(1, 2)           # (B, d, T)
        h = F.glu(self.pw1(h), dim=1)
        if self.causal:
            h = F.pad(h, (self.pad, 0))            # left-only pad -> causal
        h = self.dw(h)
        h = F.silu(self.bn(h))
        h = self.pw2(h)
        h = self.drop(h).transpose(1, 2)           # (B, T, d)
        return res + h


class CausalMHSA(nn.Module):
    """Long-range temporal modelling via causal self-attention (SDPA)."""

    def __init__(self, d, heads, causal=True):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk, self.causal = heads, d // heads, causal
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):                          # (B, T, d)
        B, T, d = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]           # (B, heads, T, dk)
        # memory-efficient attention; is_causal masks the future when online.
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        o = o.transpose(1, 2).reshape(B, T, d)
        return x + self.proj(o)


class ConformerBlock(nn.Module):
    """long (attention) + short (conv) temporal block."""

    def __init__(self, d, heads, causal):
        super().__init__()
        self.ffn1 = FFN(d)
        self.attn = CausalMHSA(d, heads, causal)
        self.conv = CausalConvModule(d, causal=causal)
        self.ffn2 = FFN(d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.attn(x)        # long-range
        x = self.conv(x)        # short-range
        x = self.ffn2(x)
        return self.norm(x)


class _Stage(nn.Module):
    def __init__(self, in_dim, d, heads, layers, num_classes, causal):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([ConformerBlock(d, heads, causal) for _ in range(layers)])
        self.cls = nn.Linear(d, num_classes)

    def forward(self, x, return_feat=False):       # x: (B, T, in_dim)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        logits = self.cls(h)                       # (B, T, num_classes)
        if return_feat:
            return logits, h                       # h = penultimate (B, T, d)
        return logits


class LoViT(nn.Module):
    def __init__(self, in_dim=2048, num_classes=7, d=256, heads=8,
                 layers=5, num_stages=2, causal=True, boundary=False):
        super().__init__()
        self.stage1 = _Stage(in_dim, d, heads, layers, num_classes, causal)
        # refinement stages consume the previous stage's class probabilities
        self.refine = nn.ModuleList([
            _Stage(num_classes, d, heads, max(2, layers // 2), num_classes, causal)
            for _ in range(num_stages - 1)
        ])
        # Optional auxiliary boundary head (BUA), off by default.
        self.boundary = boundary
        if boundary:
            self.boundary_head = nn.Linear(d, 1)

    def forward(self, x, return_boundary=False):   # x: (B, in_dim, T)
        x = x.transpose(1, 2)                      # (B, T, in_dim)
        out, feat = self.stage1(x, return_feat=True)   # out (B,T,C), feat (B,T,d)
        outputs = [out.transpose(1, 2)]            # (B, C, T)
        for stage in self.refine:
            out, feat = stage(F.softmax(out, dim=-1), return_feat=True)
            outputs.append(out.transpose(1, 2))
        if return_boundary:                        # training-time auxiliary output
            return outputs, self.boundary_head(feat).transpose(1, 2)   # (B, 1, T)
        return outputs


if __name__ == "__main__":
    for causal in (True, False):
        m = LoViT(causal=causal)
        x = torch.randn(1, 2048, 1500)
        ys = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f"causal={causal}: {len(ys)} stages, out {tuple(ys[-1].shape)}, params {n/1e6:.1f}M")
