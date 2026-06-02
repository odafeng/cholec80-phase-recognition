"""Stage 1 (EndoViT) — fine-tune EndoViT on Cholec80 phase classification.

The fair comparison to our fine-tuned ResNet50: take EndoViT (surgical MAE
pretraining) and fine-tune it end-to-end on phase labels, exactly as we did
ResNet50. Only THEN do we know whether surgical pretraining helps.

Saves checkpoint with the fine-tuned backbone, to be used by
extract_features_endovit.py --ckpt.

Run:
  python train_cnn_endovit.py --frames data/frames --anno data/phase_annotations \
        --epochs 3 --bs 48 --out checkpoints/endovit_ft.pt
"""
import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.models.vision_transformer import VisionTransformer
from huggingface_hub import snapshot_download
from tqdm import tqdm

from dataset import build_index, Cholec80FrameDataset
from phases import NUM_PHASES
from splits import TRAIN_IDS, VAL_IDS
from train_cnn import class_weights
from extract_features_endovit import ENDOVIT_MEAN, ENDOVIT_STD


def load_endovit_backbone():
    path = snapshot_download(repo_id="egeozsoy/EndoViT")
    w = torch.load(Path(path) / "pytorch_model.bin", map_location="cpu",
                   weights_only=False)["model"]
    m = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                          mlp_ratio=4, qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    m.load_state_dict(w, strict=False)
    return m


class EndoViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = load_endovit_backbone()
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        tokens = self.backbone.forward_features(x)   # (B, 197, 768)
        return self.head(tokens[:, 1:].mean(1))      # mean patch tokens


def make_transforms():
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(ENDOVIT_MEAN, ENDOVIT_STD),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(ENDOVIT_MEAN, ENDOVIT_STD),
    ])
    return train_t, eval_t


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda"):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--anno", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/endovit_ft.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_t, eval_t = make_transforms()

    print("building index...")
    train_idx, _ = build_index(args.frames, args.anno, TRAIN_IDS)
    val_idx, _ = build_index(args.frames, args.anno, VAL_IDS)
    print(f"train frames: {len(train_idx)} | val frames: {len(val_idx)}")

    train_ld = DataLoader(Cholec80FrameDataset(train_idx, train_t), args.bs,
                          shuffle=True, num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_ld = DataLoader(Cholec80FrameDataset(val_idx, eval_t), args.bs,
                        shuffle=False, num_workers=args.workers, pin_memory=True,
                        persistent_workers=True)

    model = EndoViTClassifier(NUM_PHASES).to(device)
    w = class_weights(train_idx, NUM_PHASES).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    # lower lr for the pretrained backbone, higher for the fresh head
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr * 10},
    ], weight_decay=0.05)
    # cosine decay over all steps — helps ViT-B fine-tuning converge.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_ld))
    scaler = torch.amp.GradScaler("cuda")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_ld, desc=f"epoch {ep}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running = 0.9 * running + 0.1 * loss.item()
            pbar.set_postfix(loss=f"{running:.3f}")
        acc = evaluate(model, val_ld, device)
        print(f"epoch {ep}: val_acc = {acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"backbone": model.backbone.state_dict(),
                        "val_acc": acc, "epoch": ep}, args.out)
            print(f"  -> saved best to {args.out} (val_acc={acc:.4f})")
    print(f"done. best val_acc = {best:.4f}")


if __name__ == "__main__":
    main()
