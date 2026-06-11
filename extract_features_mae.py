"""Extract per-frame features (768-d) from a self-trained surgical-MAE encoder,
for ALL 80 videos at 1 fps -> features_surgmae/. Same output format as
extract_features.py so run_reliability.sh can use it as another backbone.

Run:
  python extract_features_mae.py --frames data/frames --anno data/phase_annotations \
      --ckpt checkpoints/surgmae/surgmae_encoder_latest.pt --out features_surgmae
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from mae_pretrain import MAE
from dataset import build_index
from splits import ALL_IDS


class _Imgs(torch.utils.data.Dataset):
    def __init__(self, pairs, tf):
        self.pairs = pairs; self.tf = tf
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        return self.tf(Image.open(self.pairs[i][0]).convert("RGB"))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--anno", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("features_surgmae"))
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MAE().to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu")
    enc = ck.get("encoder", ck.get("model"))
    missing, unexpected = model.load_state_dict(enc, strict=False)
    print(f"loaded encoder from {args.ckpt} (epoch {ck.get('epoch','?')}); "
          f"missing={len(missing)} unexpected={len(unexpected)}")

    tf = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    args.out.mkdir(parents=True, exist_ok=True)
    _, per_video = build_index(args.frames, args.anno, ALL_IDS)
    for vid, pairs in per_video.items():
        out_p = args.out / f"video{vid:02d}.pt"
        if out_p.exists():
            continue
        ld = DataLoader(_Imgs(pairs, tf), args.bs, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        feats = []
        for x in ld:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model.forward_features(x.to(device))
            feats.append(f.float().cpu())
        feats = torch.cat(feats)
        labels = torch.tensor([l for _, l in pairs[:len(feats)]], dtype=torch.long)
        torch.save({"feats": feats, "labels": labels}, out_p)
        print(f"video{vid:02d}: {tuple(feats.shape)} -> {out_p}", flush=True)
    print("MAE feature extraction done.")


if __name__ == "__main__":
    main()
