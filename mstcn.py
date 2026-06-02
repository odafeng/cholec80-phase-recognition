"""MS-TCN / TeCNO temporal model.

Input : per-frame features  x of shape (B, C_in, T)   C_in = 2048 (ResNet50)
Output: list of per-stage logits, each (B, num_classes, T)
        (multi-stage = the loss is applied to EVERY stage's output)

The ONLY difference between MS-TCN and TeCNO is `causal`:
  causal=False -> symmetric padding, each timestep sees past AND future (offline).
  causal=True  -> left-only padding, each timestep sees ONLY the past (online).
                  This is exactly what TeCNO does.

Ref: Farha & Gall, "MS-TCN" (CVPR 2019); Czempiel et al., "TeCNO" (MICCAI 2020).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果causal改成True其實就是TeCNO
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_ch, out_ch, causal=False, dropout=0.5):
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        if causal:
            # pad only on the left by 2*dilation (kernel=3 -> 2 taps before center)
            self.pad = 2 * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, 3, padding=0, dilation=dilation)
        else:
            # symmetric padding keeps length and centers the kernel (sees future)
            self.conv = nn.Conv1d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.conv1x1 = nn.Conv1d(out_ch, out_ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.causal:
            out = F.pad(x, (self.pad, 0))   # (left, right)
            out = self.conv(out)
        else:
            out = self.conv(x)
        out = F.relu(out)
        out = self.conv1x1(out)
        out = self.drop(out)
        return x + out                      # residual connection


class SingleStageTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, in_dim, num_classes, causal):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, causal)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)


class MultiStageTCN(nn.Module):
    """Stack of single-stage TCNs. Stage 1 sees features; later stages refine
    the previous stage's softmax prediction."""

    def __init__(self, num_stages=2, num_layers=9, num_f_maps=64,
                 in_dim=2048, num_classes=7, causal=False):
        super().__init__()
        self.stage1 = SingleStageTCN(num_layers, num_f_maps, in_dim, num_classes, causal)
        self.stages = nn.ModuleList([
            copy.deepcopy(SingleStageTCN(num_layers, num_f_maps, num_classes, num_classes, causal))
            for _ in range(num_stages - 1)
        ])

    def forward(self, x):
        out = self.stage1(x)
        outputs = [out]
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)
        return outputs   # list of (B, num_classes, T), one per stage


if __name__ == "__main__":
    # quick shape + causality sanity check
    for causal in (False, True):
        m = MultiStageTCN(causal=causal)
        x = torch.randn(1, 2048, 100)
        ys = m(x)
        print(f"causal={causal}: {len(ys)} stages, out shape {tuple(ys[-1].shape)}")
