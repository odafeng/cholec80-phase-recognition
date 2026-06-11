"""End-to-end joint training of the CNN backbone + a causal temporal head.

The two-stage pipeline freezes the backbone after per-frame training. This script
instead trains backbone + TeCNO head JOINTLY on contiguous frame windows, so the
backbone learns features tuned for the temporal task. It is the most GPU-heavy
experiment and tests whether the frozen-feature design leaves accuracy on the table.

To plug into the existing reliability pipeline, we save the trained backbone in the
SAME checkpoint format as train_cnn.py (a ResNet50 with a 7-way fc). Then
extract_features.py produces `features_e2e/`, on which run_reliability.sh runs the
full BUA matrix exactly as for ResNet50 / EndoViT.

Memory: a full Cholec80 video (up to ~6000 frames at 1 fps) cannot be back-propped
through the CNN at once, so each step samples a window of W frames (default 128 ~=
2 min of context) and back-props through backbone+head for that window.

Run:
  python train_e2e.py --frames data/frames --anno data/phase_annotations \
      --init checkpoints/resnet50.pt --window 128 --epochs 8 --out checkpoints/e2e.pt
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

from dataset import build_index, Cholec80FrameDataset  # noqa: F401 (transforms reuse)
from train_cnn import make_transforms, build_model
from mstcn import MultiStageTCN
from train_tcn import mstcn_loss, set_seed
from phases import NUM_PHASES
from splits import TRAIN_IDS, VAL_IDS
from PIL import Image


class WindowDataset(Dataset):
    """One item = a random contiguous window (W frames + labels) from a video."""

    def __init__(self, per_video, transform, window, samples_per_video=4):
        self.videos = [pairs for pairs in per_video.values() if len(pairs) > 8]
        self.transform = transform
        self.window = window
        self.spv = samples_per_video

    def __len__(self):
        return len(self.videos) * self.spv

    def __getitem__(self, idx):
        pairs = self.videos[idx % len(self.videos)]
        T = len(pairs)
        W = min(self.window, T)
        start = random.randint(0, T - W)
        imgs, labs = [], []
        for k in range(start, start + W):
            p, l = pairs[k]
            imgs.append(self.transform(Image.open(p).convert("RGB")))
            labs.append(l)
        return torch.stack(imgs), torch.tensor(labs, dtype=torch.long)


class E2EModel(nn.Module):
    """ResNet50 trunk (2048-d) + causal MS-TCN head. Keeps a 7-way fc on the trunk
    (unused here) so the trunk saves/loads in train_cnn.py format."""

    def __init__(self, init_ckpt=None):
        super().__init__()
        self.backbone = build_model()                  # resnet50, fc->7
        if init_ckpt and Path(init_ckpt).exists():
            sd = torch.load(init_ckpt, map_location="cpu")["model"]
            self.backbone.load_state_dict(sd)
            print(f"warm-started backbone from {init_ckpt}")
        # feature trunk = everything up to (not incl.) fc
        self.trunk = nn.Sequential(*list(self.backbone.children())[:-1])  # -> (N,2048,1,1)
        self.head = MultiStageTCN(num_stages=2, num_layers=9, num_f_maps=64,
                                  in_dim=2048, num_classes=NUM_PHASES, causal=True)

    def forward(self, imgs):                            # imgs: (W, 3, 224, 224)
        f = self.trunk(imgs).flatten(1)                # (W, 2048)
        f = f.t().unsqueeze(0)                         # (1, 2048, W)
        return self.head(f)                            # list of (1, C, W)


@torch.no_grad()
def val_full(model, per_video, device, chunk=512):
    model.eval()
    correct = total = 0
    tf = make_transforms()[1]
    for pairs in per_video.values():
        feats = []
        for i in range(0, len(pairs), chunk):
            imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in pairs[i:i+chunk]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats.append(model.trunk(imgs.to(device)).flatten(1))
        f = torch.cat(feats).t().unsqueeze(0)          # (1,2048,T)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.head(f)[-1].argmax(1).squeeze(0).cpu()
        gt = torch.tensor([l for _, l in pairs])
        n = min(len(gt), pred.shape[0])
        correct += (pred[:n] == gt[:n]).sum().item()
        total += n
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--anno", required=True, type=Path)
    ap.add_argument("--init", type=Path, default=Path("checkpoints/resnet50.pt"))
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--samples_per_video", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/e2e.pt"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_t, _ = make_transforms()
    _, train_pv = build_index(args.frames, args.anno, TRAIN_IDS)
    _, val_pv = build_index(args.frames, args.anno, VAL_IDS)

    ds = WindowDataset(train_pv, train_t, args.window, args.samples_per_video)
    ld = DataLoader(ds, batch_size=1, shuffle=True, num_workers=args.workers,
                    pin_memory=True, persistent_workers=True)
    print(f"train windows/epoch={len(ds)} (W={args.window}) | val videos={len(val_pv)}")

    model = E2EModel(args.init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for imgs, labs in ld:
            imgs = imgs.squeeze(0).to(device, non_blocking=True)   # (W,3,224,224)
            labs = labs.squeeze(0).to(device, non_blocking=True)   # (W,)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(imgs)
                loss = mstcn_loss(outputs, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run = 0.9 * run + 0.1 * loss.item()
        acc = val_full(model, val_pv, device)
        print(f"epoch {ep:2d}: loss={run:.3f}  val_acc={acc:.4f}")
        if acc > best:
            best = acc
            # save the TRUNK weights back into a resnet50(fc->7) state_dict so
            # extract_features.py can consume it like any train_cnn checkpoint.
            torch.save({"model": model.backbone.state_dict(), "val_acc": acc,
                        "epoch": ep, "e2e": True}, args.out)
            print(f"  -> saved backbone to {args.out} (val_acc={acc:.4f})")
    print(f"done. best val_acc={best:.4f}")


if __name__ == "__main__":
    main()
