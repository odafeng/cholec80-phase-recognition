"""Label parsing + PyTorch datasets for Cholec80 phase recognition.

Two things happen here:
  1) parse `videoXX-phase.txt` (25 fps, one row per original frame) into ids.
  2) align those labels to our 1 fps extracted frames: the k-th frame is original
     frame 25*k, so its label is the 25*k-th row of the annotation.

Datasets:
  Cholec80FrameDataset  -> (image_tensor, label)  for Stage-1 CNN training.
  build_index(...)      -> list of (jpg_path, label) for a set of video ids.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from phases import phase_to_id


# ---------- label parsing ----------------------------------------------------

def read_phase_file(phase_txt: Path):
    """Return list of phase ids, one per ORIGINAL (25 fps) frame."""
    ids = []
    with open(phase_txt) as f:
        header = f.readline()  # "Frame\tPhase"
        for line in f:
            line = line.strip()
            if not line:
                continue
            # rows look like:  "123\tCalotTriangleDissection"
            parts = line.split()
            ids.append(phase_to_id(parts[1]))
    return ids


def labels_1fps(phase_txt: Path, step: int = 25):
    """Sample the 25 fps labels every `step` frames -> 1 fps label list."""
    full = read_phase_file(phase_txt)
    return full[::step]


# ---------- index building ---------------------------------------------------

def video_frame_paths(frames_dir: Path):
    """Sorted jpg paths for one video folder."""
    return sorted(frames_dir.glob("*.jpg"))


def build_index(frames_root: Path, anno_root: Path, video_ids, step: int = 25):
    """Build a flat [(jpg_path, label_id), ...] list across the given video ids.

    Aligns by min(#frames, #labels) so a tail off-by-one never crashes us.
    `anno_root` holds files named videoXX-phase.txt.
    """
    index = []
    per_video = {}
    for i in video_ids:
        vid = f"video{i:02d}"
        fdir = frames_root / vid
        if not fdir.is_dir():
            print(f"[warn] frames missing for {vid}, skip")
            continue
        frames = video_frame_paths(fdir)
        # annotation filename in Cholec80 is lowercase: videoXX-phase.txt
        anno = anno_root / f"{vid}-phase.txt"
        labs = labels_1fps(anno, step)
        n = min(len(frames), len(labs))
        pairs = [(frames[k], labs[k]) for k in range(n)]
        per_video[i] = pairs
        index.extend(pairs)
        if len(frames) != len(labs):
            print(f"[info] {vid}: frames={len(frames)} labels={len(labs)} -> use {n}")
    return index, per_video


# ---------- dataset ----------------------------------------------------------

class Cholec80FrameDataset(Dataset):
    """Per-frame dataset for Stage-1 CNN. Returns (img_tensor, label)."""

    def __init__(self, index, transform):
        self.index = index
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, label = self.index[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
