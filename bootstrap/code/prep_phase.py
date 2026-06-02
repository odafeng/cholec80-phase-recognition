"""Subsample Cholec80 25fps phase annotations to 1fps and align EXACTLY to the
number of preprocessed frames (VideoClipDataset asserts label count == frame count)."""
from pathlib import Path

PHASE_SRC = Path("/home/KHUser/data/phase_annotations")
FRAMES = Path("cholec80_preprocessed")
OUT = Path("phase_pp"); OUT.mkdir(exist_ok=True)
STEP = 25

for i in range(41, 81):
    tag = f"video{i:02d}"
    fdir = FRAMES / tag
    if not fdir.is_dir():
        continue
    nframes = len(list(fdir.glob("*.jpg")))
    lines = [ln for ln in open(PHASE_SRC / f"{tag}-phase.txt").read().splitlines()[1:] if ln.strip()]
    phases = [ln.split()[1] for ln in lines]
    sub = phases[::STEP]
    if len(sub) >= nframes:
        sub = sub[:nframes]
    else:
        sub = sub + [sub[-1]] * (nframes - len(sub))
    with open(OUT / f"{tag}-phase.txt", "w") as f:
        f.write("Frame\tPhase\n")
        for k, p in enumerate(sub):
            f.write(f"{k}\t{p}\n")
    print(f"{tag}: frames={nframes} labels={len(sub)}")
print("done -> phase_pp/")
