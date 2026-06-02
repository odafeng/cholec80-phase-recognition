"""Generate 1fps phase annotations for ALL videos with preprocessed frames,
aligned exactly to frame counts. Output -> cholec80_preprocessed/phase_ann_pp/."""
from pathlib import Path

PHASE_SRC = Path("/home/KHUser/data/phase_annotations")
FRAMES = Path("cholec80_preprocessed")
OUT = Path("cholec80_preprocessed/phase_ann_pp"); OUT.mkdir(parents=True, exist_ok=True)
STEP = 25

n_done = 0
for i in range(1, 81):
    tag = f"video{i:02d}"
    fdir = FRAMES / tag
    if not fdir.is_dir():
        continue
    nframes = len(list(fdir.glob("*.jpg")))
    if nframes == 0:
        continue
    lines = [ln for ln in open(PHASE_SRC / f"{tag}-phase.txt").read().splitlines()[1:] if ln.strip()]
    phases = [ln.split()[1] for ln in lines]
    sub = phases[::STEP]
    sub = sub[:nframes] if len(sub) >= nframes else sub + [sub[-1]] * (nframes - len(sub))
    with open(OUT / f"{tag}-phase.txt", "w") as f:
        f.write("Frame\tPhase\n")
        for k, p in enumerate(sub):
            f.write(f"{k}\t{p}\n")
    n_done += 1
print(f"done -> {OUT} ({n_done} videos)")
