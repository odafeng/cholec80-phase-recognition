"""Stage 2 — train the temporal model on per-video feature sequences.

ONE script for BOTH models:
  --causal           (omit) -> MS-TCN  (non-causal, offline)
  --causal           (set)  -> TeCNO   (causal, online)

Loss (classic MS-TCN), applied to EVERY stage's output:
  CrossEntropy  +  lambda * truncated-MSE smoothing loss (T-MSE)
The smoothing term penalises abrupt frame-to-frame probability changes, which
removes over-segmentation (phases are long contiguous segments).

batch size = 1: each video is a full variable-length sequence (1, 2048, T).

Run (MS-TCN):  python train_tcn.py --features features --out checkpoints/mstcn.pt
Run (TeCNO) :  python train_tcn.py --features features --out checkpoints/tecno.pt --causal
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mstcn import MultiStageTCN
from lovit import LoViT
from asformer import ASFormer
from phases import NUM_PHASES
from splits import TRAIN_IDS, VAL_IDS


def load_features(features_dir: Path, ids):
    data = []
    for i in ids:
        p = features_dir / f"video{i:02d}.pt"
        if not p.exists():
            print(f"[warn] missing {p}")
            continue
        d = torch.load(p)
        # to (1, C, T) and (T,)
        feats = d["feats"].t().unsqueeze(0).contiguous()  # (1, 2048, T)
        labels = d["labels"].long()                       # (T,)
        data.append((feats, labels))
    return data


def temporal_augment(feats, labels, min_frac=0.5, noise=0.05):
    """Combat overfitting on few (32) sequences: each step sees a RANDOM
    contiguous sub-window of the video plus small feature jitter, so the model
    effectively trains on endless variations instead of memorising 32 videos.
    feats: (1, C, T), labels: (T,)."""
    T = labels.shape[0]
    L = int(T * (min_frac + (1 - min_frac) * torch.rand(1).item()))
    L = max(min(T, 64), L)
    start = torch.randint(0, T - L + 1, (1,)).item() if T > L else 0
    feats = feats[:, :, start:start + L]
    labels = labels[start:start + L]
    if noise > 0:
        feats = feats + noise * torch.randn_like(feats)
    return feats, labels


def mstcn_loss(outputs, labels, lam=0.15, tau=4):
    """outputs: list of (1, C, T); labels: (T,)."""
    total = 0.0
    y = labels.unsqueeze(0)                      # (1, T)
    for out in outputs:
        ce = F.cross_entropy(out.transpose(2, 1).reshape(-1, out.shape[1]),
                             y.reshape(-1))
        # truncated MSE on log-probs between consecutive timesteps
        logp = F.log_softmax(out, dim=1)
        mse = torch.clamp((logp[:, :, 1:] - logp[:, :, :-1]) ** 2, max=tau ** 2)
        total = total + ce + lam * mse.mean()
    return total


def set_seed(seed):
    """Seed everything for reproducible multi-seed runs / deep ensembling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def boundary_targets(labels, tol=10):
    """labels (T,) -> (T,) float in {0,1}: 1 within +/-tol of a GT transition."""
    T = labels.shape[0]
    tgt = torch.zeros(T, device=labels.device)
    if T > 1:
        trans = (labels[1:] != labels[:-1]).nonzero().flatten() + 1
        for b in trans.tolist():
            tgt[max(0, b - tol): min(T, b + tol)] = 1.0
    return tgt


def bua_loss(outputs, b_logits, labels, tol=10, w_wce=0.5, w_bce=1.0):
    """BUA boundary terms, ADDITIVE to mstcn_loss (RESEARCH_PLAN.md):
      - boundary-weighted CE: extra CE focused on frames near GT transitions,
      - boundary-head BCE: predict the boundary region, class-balanced.
    """
    tgt = boundary_targets(labels, tol)                       # (T,)
    last = outputs[-1].squeeze(0).t()                         # (T, C)
    ce_frames = F.cross_entropy(last, labels, reduction="none")  # (T,)
    wce = (ce_frames * tgt).sum() / tgt.sum().clamp(min=1.0)
    pos = tgt.sum()
    neg = tgt.numel() - pos
    pos_weight = (neg / pos.clamp(min=1.0)).clamp(max=20.0)
    bce = F.binary_cross_entropy_with_logits(
        b_logits.squeeze(0).squeeze(0), tgt, pos_weight=pos_weight)
    return w_wce * wce + w_bce * bce


@torch.no_grad()
def val_accuracy(model, data, device):
    model.eval()
    correct = total = 0
    for feats, labels in data:
        out = model(feats.to(device))[-1]        # last stage
        pred = out.argmax(1).squeeze(0).cpu()
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--causal", action="store_true", help="causal -> online (TeCNO/LoViT-online)")
    ap.add_argument("--model", choices=["mstcn", "lovit", "asformer"], default="mstcn")
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--layers", type=int, default=9, help="mstcn: 9; lovit: ~5")
    ap.add_argument("--fmaps", type=int, default=64, help="mstcn feature maps")
    ap.add_argument("--d", type=int, default=256, help="lovit model dim")
    ap.add_argument("--heads", type=int, default=8, help="lovit attention heads")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--augment", action="store_true", help="temporal crop + feature jitter")
    ap.add_argument("--boundary", action="store_true",
                    help="BUA: auxiliary boundary head + boundary-weighted loss")
    ap.add_argument("--seed", type=int, default=0, help="seed everything (multi-seed runs)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "causal/online" if args.causal else "non-causal/offline"
    name = f"{args.model.upper()} ({tag})"
    print(f"=== training {name} ===")

    train_data = load_features(args.features, TRAIN_IDS)
    val_data = load_features(args.features, VAL_IDS)
    in_dim = train_data[0][0].shape[1]
    print(f"videos: train={len(train_data)} val={len(val_data)} | feat dim={in_dim}")

    if args.model == "lovit":
        model = LoViT(in_dim=in_dim, num_classes=NUM_PHASES, d=args.d,
                      heads=args.heads, layers=args.layers,
                      num_stages=args.stages, causal=args.causal,
                      boundary=args.boundary).to(device)
    elif args.model == "asformer":
        model = ASFormer(in_dim=in_dim, num_classes=NUM_PHASES, num_stages=args.stages,
                         num_layers=args.layers, d=args.d, heads=args.heads,
                         causal=args.causal, boundary=args.boundary).to(device)
    else:
        model = MultiStageTCN(args.stages, args.layers, args.fmaps,
                              in_dim, NUM_PHASES, causal=args.causal,
                              boundary=args.boundary).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"params: {n_param/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    order = list(range(len(train_data)))
    for ep in range(1, args.epochs + 1):
        model.train()
        # real seed-dependent shuffle (set_seed(args.seed) seeds `random`), so the
        # 5 seeds differ in data ORDER too -> genuine ensemble diversity, not just
        # init/dropout noise.
        random.shuffle(order)
        ep_loss = 0.0
        for j in order:
            feats, labels = train_data[j]
            if args.augment:
                feats, labels = temporal_augment(feats, labels)
            feats, labels = feats.to(device), labels.to(device)
            opt.zero_grad()
            if args.boundary:
                outputs, b_logits = model(feats, return_boundary=True)
                loss = mstcn_loss(outputs, labels) + bua_loss(outputs, b_logits, labels)
            else:
                outputs = model(feats)
                loss = mstcn_loss(outputs, labels)
            loss.backward()
            # gradient clipping: prevents the loss-spike divergence that the
            # transformer (LoViT) otherwise hits mid-training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        acc = val_accuracy(model, val_data, device)
        print(f"epoch {ep:3d}: loss={ep_loss/len(train_data):.3f}  val_acc={acc:.4f}")
        if acc > best:
            best = acc
            # store ONLY plain scalars in cfg (no pathlib.Path) so the checkpoint
            # loads cleanly under torch>=2.6 weights_only=True.
            clean_cfg = {"arch": args.model, "stages": args.stages,
                         "layers": args.layers, "fmaps": args.fmaps,
                         "d": args.d, "heads": args.heads, "causal": args.causal,
                         "in_dim": in_dim, "boundary": args.boundary,
                         "seed": args.seed}
            torch.save({"model": model.state_dict(), "causal": args.causal,
                        "val_acc": acc, "epoch": ep,
                        "cfg": clean_cfg}, args.out)
    print(f"done. best val_acc={best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
