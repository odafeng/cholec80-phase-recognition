"""Pareto frontier: latency vs over-segmentation, plain confidence smoothing vs
boundary-gated smoothing. If boundary-gating's frontier DOMINATES (lower latency at
every over-seg level), the boundary head is a genuine contribution -- it reaches
operating points plain smoothing cannot. Selection should be on val; here we print
both splits. CPU. Run:
  python frontier.py --ckpt checkpoints/rel_e2e/tecno_bua_s0.pt --features features_e2e
"""
import argparse
from pathlib import Path
import numpy as np

from compare_gate import collect, metrics_of

GAMMAS = [0.2, 0.3, 0.5, 0.7, 1.0]
BETAS = [0.3, 0.5, 1.0]


def frontier(seqs):
    conf = []
    for g in GAMMAS:
        a, o, l, f, b = metrics_of(seqs, "conf", gamma=g)
        conf.append((o, l, g))
    bg = []
    for g in GAMMAS:
        for be in BETAS:
            a, o, l, f, b = metrics_of(seqs, "bgate", beta=be, gamma=g)
            bg.append((o, l, g, be))
    return conf, bg


def pareto_min(points):
    """keep points not dominated on (over-seg, latency) — both lower better."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    out, best = [], float("inf")
    for p in pts:
        if p[1] < best:
            out.append(p); best = p[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    from splits import VAL_IDS, TEST_IDS
    ids = VAL_IDS if args.split == "val" else TEST_IDS
    seqs = collect(args.ckpt, args.features, ids)
    conf, bg = frontier(seqs)
    print(f"# {args.ckpt.name} [{args.split}]  (over-seg, latency s)")
    print("conf-smooth frontier:")
    for o, l, g in pareto_min([(o, l, g) for o, l, g in conf]):
        print(f"    over-seg {o:5.2f}  lat {l:5.1f}  (gamma={g})")
    print("boundary-gated frontier:")
    for o, l, gb in pareto_min([(o, l, (g, be)) for o, l, g, be in bg]):
        print(f"    over-seg {o:5.2f}  lat {l:5.1f}  (gamma={gb[0]}, beta={gb[1]})")
    # dominance check: for matched over-seg buckets, does bgate give lower latency?
    print("dominance @ over-seg<=3.0:")
    cb = min([l for o, l, *_ in conf if o <= 3.0], default=float("nan"))
    bb = min([l for o, l, *_ in bg if o <= 3.0], default=float("nan"))
    print(f"    best latency conf={cb:.1f}s  bgate={bb:.1f}s  "
          f"-> bgate {'WINS' if bb < cb - 0.2 else 'ties/loses'} by {cb-bb:+.1f}s")


if __name__ == "__main__":
    main()
