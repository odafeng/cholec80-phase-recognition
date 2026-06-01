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
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mstcn import MultiStageTCN
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
    ap.add_argument("--causal", action="store_true", help="set -> TeCNO; unset -> MS-TCN")
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--fmaps", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "TeCNO (causal)" if args.causal else "MS-TCN (non-causal)"
    print(f"=== training {name} ===")

    train_data = load_features(args.features, TRAIN_IDS)
    val_data = load_features(args.features, VAL_IDS)
    in_dim = train_data[0][0].shape[1]
    print(f"videos: train={len(train_data)} val={len(val_data)} | feat dim={in_dim}")

    model = MultiStageTCN(args.stages, args.layers, args.fmaps,
                          in_dim, NUM_PHASES, causal=args.causal).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    order = list(range(len(train_data)))
    for ep in range(1, args.epochs + 1):
        model.train()
        # simple shuffle without RNG seed dependence
        order = order[1:] + order[:1]
        ep_loss = 0.0
        for j in order:
            feats, labels = train_data[j]
            feats, labels = feats.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(feats)
            loss = mstcn_loss(outputs, labels)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        acc = val_accuracy(model, val_data, device)
        print(f"epoch {ep:3d}: loss={ep_loss/len(train_data):.3f}  val_acc={acc:.4f}")
        if acc > best:
            best = acc
            # store ONLY plain scalars in cfg (no pathlib.Path) so the checkpoint
            # loads cleanly under torch>=2.6 weights_only=True.
            clean_cfg = {"stages": args.stages, "layers": args.layers,
                         "fmaps": args.fmaps, "causal": args.causal}
            torch.save({"model": model.state_dict(), "causal": args.causal,
                        "val_acc": acc, "epoch": ep,
                        "cfg": clean_cfg}, args.out)
    print(f"done. best val_acc={best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
