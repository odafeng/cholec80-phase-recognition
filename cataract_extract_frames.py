"""Extract 1 fps frames from Cataract-101 videos, keyed by VideoID (case_<VID>.mp4
-> frames/<VID>/00000000.jpg ...). Mirrors extract_frames.py for Cholec80 but
uses the case filename = VideoID. Resumable (skips videos already done)."""
import subprocess
from pathlib import Path

from cataract_dataset import video_ids, video_meta, ROOT

OUT = Path("data/cataract101/frames")
STEP = 25            # 25 fps -> 1 fps
SIZE = 250


def extract_one(mp4, dst):
    dst.mkdir(parents=True, exist_ok=True)
    # pick every STEP-th frame, scale shorter side to SIZE
    vf = f"select='not(mod(n\\,{STEP}))',scale={SIZE}:{SIZE}:force_original_aspect_ratio=increase,crop={SIZE}:{SIZE}"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
           "-vf", vf, "-vsync", "0", "-q:v", "3", str(dst / "%08d.jpg")]
    subprocess.run(cmd, check=False)
    return len(list(dst.glob("*.jpg")))


def main():
    vids = video_ids()
    meta = video_meta()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"extracting {len(vids)} cataract videos -> {OUT}")
    for i, vid in enumerate(sorted(vids)):
        dst = OUT / str(vid)
        mp4 = ROOT / "videos" / f"case_{vid}.mp4"
        if dst.is_dir() and len(list(dst.glob("*.jpg"))) > 0:
            continue
        if not mp4.exists():
            print(f"[warn] missing {mp4}"); continue
        n = extract_one(mp4, dst)
        exp = meta[vid][0] // STEP
        print(f"[{i+1}/{len(vids)}] video {vid}: {n} frames (expect ~{exp})", flush=True)
    print("CAT101_FRAMES_DONE")


if __name__ == "__main__":
    main()
