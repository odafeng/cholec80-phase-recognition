"""Stage 1.5 (alternative) — extract per-frame features with EndoViT instead of
our ResNet50.

EndoViT (egeozsoy/EndoViT, apache-2.0) is a ViT-B/16 pretrained with MAE on
surgical images (Endo700k incl. Cholec80). We run it on the SAME 1fps frames and
dump 768-d per-frame features, so the temporal heads (MS-TCN/TeCNO/LoViT) can be
retrained on stronger, domain-specific features without changing anything else.

Output: features_endovit/videoXX.pt = { feats:(T,768), labels:(T,) }
Feature = mean of patch tokens (excl. CLS) from forward_features — robust for MAE.

Run:
  python extract_features_endovit.py --frames data/frames \
        --anno data/phase_annotations --out features_endovit
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
from splits import ALL_IDS

# EndoViT's surgical-domain normalization (from the model card)
ENDOVIT_MEAN = [0.3464, 0.2280, 0.2228]
ENDOVIT_STD = [0.2520, 0.2128, 0.2093]


def load_endovit(device, ckpt=None):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                              mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6)).eval()
    if ckpt is not None:
        # fine-tuned backbone (from train_cnn_endovit.py)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["backbone"]
        print(f"loading fine-tuned EndoViT backbone from {ckpt}")
    else:
        path = snapshot_download(repo_id="egeozsoy/EndoViT")
        state = torch.load(Path(path) / "pytorch_model.bin",
                           map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=False)
    return model.to(device)


def endovit_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(ENDOVIT_MEAN, ENDOVIT_STD),
    ])


@torch.no_grad()
def extract_video(model, pairs, transform, device, bs=64, workers=8):
    ds = Cholec80FrameDataset(pairs, transform)
    ld = DataLoader(ds, bs, shuffle=False, num_workers=workers, pin_memory=True)
    feats, labels = [], []
    for x, y in ld:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda"):
            tokens = model.forward_features(x)      # (B, 197, 768)
        f = tokens[:, 1:].mean(1)                    # mean patch tokens -> (B, 768)
        feats.append(f.float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--anno", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("features_endovit"))
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt", type=Path, default=None, help="fine-tuned EndoViT backbone")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_endovit(device, args.ckpt)
    tf = endovit_transform()
    args.out.mkdir(parents=True, exist_ok=True)

    for i in tqdm(ALL_IDS, desc="videos"):
        out_f = args.out / f"video{i:02d}.pt"
        if out_f.exists():
            continue
        _, per_video = build_index(args.frames, args.anno, [i])
        if i not in per_video:
            continue
        feats, labels = extract_video(model, per_video[i], tf, device, args.bs, args.workers)
        torch.save({"feats": feats, "labels": labels}, out_f)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
