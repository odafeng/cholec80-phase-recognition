"""Cataract-101 label adapter (cross-procedure dataset for QCD generalization).

Cataract-101 (Schoeffmann et al., MMSys'18): 101 cataract surgery videos, 25 fps,
10 surgical phases. annotations.csv is in TRANSITION-POINT form
(VideoID;FrameNo;Phase = "at FrameNo the phase switches to Phase"); we expand it to
per-frame labels and sample at 1 fps (every 25th original frame) to match the
Cholec80 pipeline. Phases 1..10 are mapped to ids 0..9.

Layout expected after unzip:
  data/cataract101/cataract-101/{annotations.csv, videos.csv, phases.csv, videos/case_*.mp4}
"""
from pathlib import Path
import csv

NUM_PHASES_CAT = 10
ROOT = Path("data/cataract101/cataract-101")
PHASES_CAT = ["Incision", "ViscousAgentInjection", "Rhexis", "Hydrodissection",
              "Phacoemulsification", "IrrigationAspiration", "CapsulePolishing",
              "LensImplantSetup", "ViscousAgentRemoval", "TonifyingAntibiotics"]


def _read_csv(path):
    with open(path) as f:
        return list(csv.reader(f, delimiter=";"))


def video_ids(root=ROOT):
    rows = _read_csv(root / "videos.csv")[1:]
    return [int(r[0]) for r in rows if r and r[0].strip()]


def video_meta(root=ROOT):
    """{vid: (n_frames, fps)} from videos.csv."""
    out = {}
    for r in _read_csv(root / "videos.csv")[1:]:
        if r and r[0].strip():
            out[int(r[0])] = (int(r[1]), int(r[2]))
    return out


def labels_1fps(video_id, step=25, root=ROOT):
    """Per-1fps-frame phase ids (0..9) for one video, expanding transition points.

    Frames before the first annotated transition are assigned the first phase that
    occurs (pre-incision setup is rare/short); this matches how phase sequences are
    treated elsewhere. Returns a list aligned to extracted 1 fps frames.
    """
    trans = []  # (frame_no, phase_id)
    for r in _read_csv(root / "annotations.csv")[1:]:
        if r and int(r[0]) == video_id:
            trans.append((int(r[1]), int(r[2]) - 1))   # phase 1..10 -> 0..9
    trans.sort()
    n_frames, fps = video_meta(root)[video_id]
    # build per-original-frame labels
    full = [trans[0][1]] * n_frames if trans else [0] * n_frames
    for i, (fn, ph) in enumerate(trans):
        end = trans[i + 1][0] if i + 1 < len(trans) else n_frames
        for k in range(max(0, fn), min(n_frames, end)):
            full[k] = ph
    return full[::step]


# reproducible split over the 101 videos (sorted by id): 60/20/20
def _split():
    vids = sorted(video_ids())
    n = len(vids)
    n_tr, n_va = int(n * 0.6), int(n * 0.2)
    return vids[:n_tr], vids[n_tr:n_tr + n_va], vids[n_tr + n_va:]


def get_splits():
    return _split()


if __name__ == "__main__":
    import numpy as np
    vids = video_ids()
    tr, va, te = get_splits()
    print(f"Cataract-101: {len(vids)} videos | split train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    print(f"phases: {NUM_PHASES_CAT}")
    v = vids[0]
    lab = labels_1fps(v)
    print(f"video {v}: {len(lab)} 1fps frames, phase id range {min(lab)}..{max(lab)}, "
          f"#transitions {1 + sum(np.diff(lab) != 0)}")
