"""Self-supervised MAE pretraining of a ViT-B/16 surgical backbone, FROM SCRATCH,
on Cholec80 TRAIN+VAL frames only (videos 1-40; test 41-80 NEVER touched).

This produces a backbone we fully own and that is provably uncontaminated by the
test set, unlike the public EndoViT (MAE on surgical images that may overlap
Cholec80). The encoder is later used in the BUA pipeline as another backbone.

Implementation follows He et al. "Masked Autoencoders Are Scalable Vision
Learners" (CVPR 2022): encode only the visible 25% of patches, reconstruct the
masked 75% with a lightweight decoder, loss = normalized-pixel MSE on masked
patches. ViT blocks come from timm.

Run (long, multi-day; in tmux):
  python mae_pretrain.py --frames data/frames_ssl --epochs 800 --bs 512 \
      --out checkpoints/surgmae --save_every 25
"""
import argparse
import glob
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from timm.models.vision_transformer import Block, PatchEmbed


# --------------------------------------------------------------------------- #
# 2D sin-cos positional embedding (fixed, MAE-style)
# --------------------------------------------------------------------------- #
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True):
    gh = gw = grid_size
    gy, gx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    grid = np.stack([gx, gy], axis=0).reshape(2, -1)

    def from_grid(d, pos):
        omega = np.arange(d // 2) / (d / 2.0)
        omega = 1.0 / (10000 ** omega)
        out = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb = np.concatenate([from_grid(embed_dim // 2, grid[0]),
                          from_grid(embed_dim // 2, grid[1])], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)


# --------------------------------------------------------------------------- #
# dataset: all frames under data/frames_ssl/<video>/<frame>.jpg
# --------------------------------------------------------------------------- #
class FrameFolder(Dataset):
    def __init__(self, root, transform, cache=None):
        if cache and Path(cache).exists():
            self.files = Path(cache).read_text().splitlines()
        else:
            self.files = sorted(glob.glob(str(Path(root) / "*" / "*.jpg")))
            if cache:
                Path(cache).write_text("\n".join(self.files))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(Image.open(self.files[i]).convert("RGB"))


# --------------------------------------------------------------------------- #
# MAE model
# --------------------------------------------------------------------------- #
class MAE(nn.Module):
    def __init__(self, img_size=224, patch=16, in_chans=3,
                 embed_dim=768, depth=12, heads=12,
                 dec_dim=512, dec_depth=8, dec_heads=16, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed(img_size, patch, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.grid = int(self.num_patches ** 0.5)
        self.patch_dim = patch * patch * in_chans
        self.patch = patch

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed",
                             get_2d_sincos_pos_embed(embed_dim, self.grid, True))
        self.blocks = nn.ModuleList([Block(embed_dim, heads, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # decoder
        self.dec_embed = nn.Linear(embed_dim, dec_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("dec_pos_embed",
                             get_2d_sincos_pos_embed(dec_dim, self.grid, True))
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads, qkv_bias=True) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, self.patch_dim, bias=True)

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], h * w, p * p * 3)

    def random_masking(self, x):
        N, L, D = x.shape
        keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones(N, L, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_keep, mask, ids_restore

    def forward_encoder(self, imgs):
        x = self.patch_embed(imgs) + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x)
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.dec_embed(x)
        N = x.shape[0]
        n_mask = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.repeat(N, n_mask, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1) + self.dec_pos_embed
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        return self.dec_pred(x)[:, 1:, :]      # drop cls -> (N, L, patch_dim)

    @torch.no_grad()
    def forward_features(self, imgs):
        """Encode ALL patches (no masking) -> global-avg-pooled token (N, embed_dim).
        Used for downstream feature extraction."""
        x = self.patch_embed(imgs) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:, :].mean(dim=1)             # avg-pool patch tokens

    def forward(self, imgs):
        target = self.patchify(imgs)
        # per-patch normalization (MAE default) stabilises the target
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = ((pred - target) ** 2).mean(dim=-1)         # per-patch MSE
        return (loss * mask).sum() / mask.sum()            # masked patches only


def save_encoder(model, path, epoch):
    """Save encoder weights in a self-contained dict for downstream feature use."""
    enc = {k: v for k, v in model.state_dict().items()
           if not k.startswith(("dec_", "mask_token"))}
    torch.save({"encoder": enc, "epoch": epoch,
                "cfg": {"embed_dim": 768, "patch": 16, "depth": 12, "heads": 12}}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-4, help="base lr per 256 imgs")
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/surgmae"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = FrameFolder(args.frames, tf, cache=str(args.out / "filelist.txt"))
    ld = DataLoader(ds, args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    print(f"frames={len(ds)}  steps/epoch={len(ld)}  bs={args.bs}")

    model = MAE(mask_ratio=args.mask_ratio).to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MAE params: {nparams:.1f}M")
    eff_lr = args.lr * args.bs / 256
    opt = torch.optim.AdamW(model.parameters(), lr=eff_lr, betas=(0.9, 0.95),
                            weight_decay=args.wd)

    def lr_at(ep):
        if ep < args.warmup:
            return eff_lr * ep / max(1, args.warmup)
        t = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return eff_lr * 0.5 * (1 + math.cos(math.pi * t))

    # resume from latest checkpoint if present
    start = 0
    last = sorted(args.out.glob("surgmae_ep*.pt"))
    if last:
        ck = torch.load(last[-1], map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start = ck["epoch"]
        print(f"resumed from {last[-1]} (epoch {start})")

    for ep in range(start, args.epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep)
        model.train()
        t0 = time.time(); run = 0.0; n = 0
        for imgs in ld:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(imgs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item(); n += 1
        print(f"epoch {ep+1}/{args.epochs}  loss={run/max(n,1):.4f}  "
              f"lr={lr_at(ep):.2e}  {time.time()-t0:.0f}s", flush=True)
        if (ep + 1) % args.save_every == 0 or ep + 1 == args.epochs:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": ep + 1}, args.out / f"surgmae_ep{ep+1:04d}.pt")
            save_encoder(model, args.out / "surgmae_encoder_latest.pt", ep + 1)
            print(f"  saved checkpoint @ epoch {ep+1}", flush=True)
    print("MAE pretraining done.")


if __name__ == "__main__":
    main()
