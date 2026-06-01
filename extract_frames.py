"""Stage 0 — extract frames from Cholec80 videos at 1 fps.

WHY this approach:
  Videos are 25 fps and phase labels are given per original frame (also 25 fps).
  We pick EXACTLY original frames 0, 25, 50, ... using ffmpeg's select filter:
      select='not(mod(n,25))'
  so the k-th extracted jpg corresponds to original frame index 25*k, which is
  exactly annotation row 25*k. This makes frame<->label alignment trivial and
  robust (no half-frame timestamp drift).

  Each frame is resized to 250x250 (TeCNO convention; we random/center-crop to
  224 later during training).

Output layout:
  <out_dir>/video01/00000000.jpg, 00000001.jpg, ...
  <out_dir>/video02/...

Usage:
  python extract_frames.py --videos data/videos --out data/frames
  python extract_frames.py --videos data/videos --out data/frames --only 1 2 3
"""
import argparse
import os
import re
import subprocess
from pathlib import Path


def find_videos(videos_dir: Path):
    """Return {video_id: path} for files named like videoNN.* or VIDEONN.*."""
    out = {}
    for p in sorted(videos_dir.iterdir()):
        if not p.is_file():
            continue
        m = re.search(r"(?i)video[_-]?(\d+)", p.stem)
        if m and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
            out[int(m.group(1))] = p
    return out


def extract_one(video_path: Path, out_dir: Path, step: int = 25, size: int = 250):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resume guard: if frames already exist, skip.
    existing = list(out_dir.glob("*.jpg"))
    if existing:
        print(f"  [skip] {out_dir.name}: {len(existing)} frames already present")
        return len(existing)

    vf = f"select='not(mod(n\\,{step}))',scale={size}:{size}"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", vf,
        "-vsync", "0",          # keep every selected frame, no dup/drop
        "-q:v", "2",            # high-quality jpg
        str(out_dir / "%08d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    n = len(list(out_dir.glob("*.jpg")))
    print(f"  [ok]   {out_dir.name}: {n} frames")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--step", type=int, default=25, help="orig fps (pick every Nth frame)")
    ap.add_argument("--size", type=int, default=250)
    ap.add_argument("--only", type=int, nargs="*", help="optional subset of video ids")
    args = ap.parse_args()

    vids = find_videos(args.videos)
    if not vids:
        raise SystemExit(f"No videos found in {args.videos}")
    ids = sorted(args.only) if args.only else sorted(vids)
    print(f"Extracting {len(ids)} videos -> {args.out}")

    total = 0
    for i in ids:
        if i not in vids:
            print(f"  [warn] video id {i} not found, skip")
            continue
        total += extract_one(vids[i], args.out / f"video{i:02d}", args.step, args.size)
    print(f"Done. total frames: {total}")


if __name__ == "__main__":
    main()
