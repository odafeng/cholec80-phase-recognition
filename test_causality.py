"""Regression test: every ONLINE head must be strictly causal.

Prefix-invariance: for a causal model, the first-k per-frame outputs must be
identical whether the model is fed only the first k frames or the whole sequence.
Any dependence on future frames (a non-causal conv pad, bidirectional attention,
or a normalization over the time axis) breaks this. This guards the paper's core
online/causal claim. Run:  python test_causality.py
"""
import torch
from mstcn import MultiStageTCN
from lovit import LoViT
from asformer import ASFormer


def causal_ok(model, name, T=120, k=60, tol=1e-4):
    torch.manual_seed(0)
    x = torch.randn(1, 2048, T)
    model.eval()
    with torch.no_grad():
        full = model(x)[-1][:, :, :k]
        pref = model(x[:, :, :k])[-1]
    d = (full - pref).abs().max().item()
    print(f"  {name:18s} max|Δ| first {k} frames = {d:.2e}  "
          f"{'CAUSAL ok' if d < tol else 'LEAKS FUTURE'}")
    return d < tol


def main():
    ok = []
    ok.append(causal_ok(MultiStageTCN(causal=True), "TeCNO"))
    ok.append(causal_ok(LoViT(causal=True), "LoViT"))
    ok.append(causal_ok(ASFormer(causal=True), "ASFormer"))
    ok.append(causal_ok(MultiStageTCN(causal=True, boundary=True), "TeCNO+boundary"))
    ok.append(causal_ok(LoViT(causal=True, boundary=True), "LoViT+boundary"))
    print("ALL CAUSAL" if all(ok) else "*** CAUSALITY VIOLATION ***")
    return all(ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
