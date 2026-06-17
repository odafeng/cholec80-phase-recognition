"""Cataract-101 stage-1: fine-tune ResNet50 (10 phases) on cataract frames, then
extract 2048-d features for all videos (keyed by VideoID) -> features_cataract/.
Mirrors train_cnn.py + extract_features.py but for the cataract dataset/splits.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from collections import Counter
from pathlib import Path

from train_cnn import make_transforms, IMAGENET_MEAN, IMAGENET_STD  # noqa
from dataset import Cholec80FrameDataset
from cataract_dataset import video_ids, labels_1fps, get_splits, NUM_PHASES_CAT

FRAMES = Path("data/cataract101/frames")
OUTF = Path("features_cataract")
CKPT = Path("checkpoints/cataract_resnet50.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_index(vids):
    idx = {}
    for v in vids:
        d = FRAMES / str(v)
        frames = sorted(d.glob("*.jpg"))
        labs = labels_1fps(v)
        n = min(len(frames), len(labs))
        idx[v] = [(frames[k], labs[k]) for k in range(n)]
    return idx


def model_10():
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, NUM_PHASES_CAT)
    return m


def main():
    tr_t, ev_t = make_transforms()
    tr_ids, va_ids, te_ids = get_splits()
    tr_idx = build_index(tr_ids); va_idx = build_index(va_ids)
    tr_flat = [p for v in tr_ids for p in tr_idx[v]]
    va_flat = [p for v in va_ids for p in va_idx[v]]
    print(f"cataract train frames {len(tr_flat)} | val {len(va_flat)}")

    train_ld = DataLoader(Cholec80FrameDataset(tr_flat, tr_t), 64, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    val_ld = DataLoader(Cholec80FrameDataset(va_flat, ev_t), 64, shuffle=False,
                        num_workers=8, pin_memory=True, persistent_workers=True)

    if not CKPT.exists():
        model = model_10().to(device)
        cnt = Counter(l for _, l in tr_flat); tot = sum(cnt.values())
        w = torch.tensor([tot / (NUM_PHASES_CAT * cnt.get(c, 1)) for c in range(NUM_PHASES_CAT)])
        crit = nn.CrossEntropyLoss(weight=(w / w.mean()).to(device))
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        best = 0.0
        for ep in range(1, 6):
            model.train()
            for x, y in train_ld:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad()
                with torch.autocast("cuda"):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            model.eval(); corr = tot_ = 0
            with torch.no_grad():
                for x, y in val_ld:
                    with torch.autocast("cuda"):
                        p = model(x.to(device)).argmax(1)
                    corr += (p == y.to(device)).sum().item(); tot_ += y.numel()
            acc = corr / max(tot_, 1)
            print(f"epoch {ep}: val_acc {acc:.4f}", flush=True)
            if acc > best:
                best = acc; CKPT.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "val_acc": acc}, CKPT)
        print(f"CNN done best val_acc {best:.4f}")

    # extract features for ALL videos
    OUTF.mkdir(exist_ok=True)
    model = model_10().to(device)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    feat_model = nn.Sequential(*list(model.children())[:-1]).eval()
    all_idx = build_index(sorted(video_ids()))
    for v, pairs in all_idx.items():
        out = OUTF / f"video{v}.pt"
        if out.exists():
            continue
        feats = []
        ds = Cholec80FrameDataset(pairs, ev_t)
        ld = DataLoader(ds, 128, shuffle=False, num_workers=8, pin_memory=True)
        with torch.no_grad():
            for x, _ in ld:
                with torch.autocast("cuda"):
                    f = feat_model(x.to(device)).flatten(1)
                feats.append(f.float().cpu())
        feats = torch.cat(feats)
        labels = torch.tensor([l for _, l in pairs[:len(feats)]], dtype=torch.long)
        torch.save({"feats": feats, "labels": labels}, out)
        print(f"video {v}: {tuple(feats.shape)} -> {out}", flush=True)
    print("CAT101_FEATURES_DONE")


if __name__ == "__main__":
    main()
