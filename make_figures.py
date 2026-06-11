"""Generate the paper's figures + a master results table from the per-video npz
files written by evaluate.py (results/<variant>/<seed>.npz).

Figures (saved to docs/figures/):
  fig1_boundary_ece     headline: interior-ECE vs boundary-ECE per head
  fig2_oversegmentation online heads over-segment; BUA fixes it
  fig3_segmental_f1     F1@{10,25,50} per head/variant
  fig4_latency          transition latency + miss-rate

Also writes docs/figures/summary_table.md (mean +/- std over seeds).

Usage:
  python make_figures.py --results results/baseline           # Phase 2 (single seed)
  python make_figures.py --results results/rel --variants tecno_baseline tecno_bua ...
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/figures")


def load_files(files):
    """list of npz paths -> {metric: (n_files, n_videos)} stacked over seeds."""
    files = [f for f in files if Path(f).exists()]
    if not files:
        return None
    per = [dict(np.load(f)) for f in files]
    keys = set.intersection(*[set(p) for p in per])
    return {k: np.stack([p[k] for p in per]) for k in keys}


def load_variant(d):
    """dir with <seed>.npz -> {metric: (n_seeds, n_videos)}."""
    return load_files(sorted(glob.glob(str(Path(d) / "*.npz"))))


def agg(v, key):
    """mean over (seeds, videos) and std over per-video-seed-mean."""
    a = v[key]
    per_video = np.nanmean(a, axis=0)          # average seeds -> per video
    return float(np.nanmean(per_video)), float(np.nanstd(per_video))


def discover(results_dir, variants):
    """Two layouts supported:
      - Phase 3: results/<variant>/<seed>.npz  -> each subdir is a variant.
      - Phase 2: results/<dir>/<model>.npz      -> each loose file is a variant.
    """
    rd = Path(results_dir)
    res = {}
    if variants:
        for name in variants:
            p = rd / name
            v = load_variant(p) if p.is_dir() else load_files(
                sorted(glob.glob(str(rd / f"{name}*.npz"))))
            if v: res[name] = v
        return res
    subdirs = [s for s in sorted(rd.iterdir()) if s.is_dir()]
    if subdirs:
        for sub in subdirs:
            v = load_variant(sub)
            if v: res[sub.name] = v
    else:
        for f in sorted(glob.glob(str(rd / "*.npz"))):   # loose files = variants
            v = load_files([f])
            if v: res[Path(f).stem] = v
    return res


def fig_boundary_ece(res):
    names = list(res)
    inter = [agg(res[n], "interior_ece")[0] * 100 for n in names]
    bound = [agg(res[n], "boundary_ece")[0] * 100 for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(names)), 4))
    ax.bar(x - w/2, inter, w, label="interior-ECE", color="#4c72b0")
    ax.bar(x + w/2, bound, w, label="boundary-ECE", color="#c44e52")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("ECE (%)")
    ax.set_title("Calibration is far worse at phase boundaries\n(standard ECE dilutes this)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig1_boundary_ece.png", dpi=150); plt.close(fig)


def fig_simple_bar(res, key, ylabel, title, fname, scale=1.0):
    names = list(res)
    vals = [agg(res[n], key)[0] * scale for n in names]
    errs = [agg(res[n], key)[1] * scale for n in names]
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(names)), 4))
    ax.bar(np.arange(len(names)), vals, yerr=errs, color="#55a868", capsize=3)
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)


def fig_segmental_f1(res):
    names = list(res)
    keys = ["f1@10", "f1@25", "f1@50"]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(names)), 4))
    for j, k in enumerate(keys):
        vals = [agg(res[n], k)[0] for n in names]
        ax.bar(x + (j - 1) * w, vals, w, label=k)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("segmental F1 (%)"); ax.set_title("Segmental F1@{10,25,50}")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig3_segmental_f1.png", dpi=150); plt.close(fig)


TABLE_METRICS = [
    ("accuracy", 100, "acc%"), ("relaxed_accuracy", 100, "relaxed%"),
    ("jaccard", 100, "Jacc"), ("f1@10", 1, "F1@10"), ("f1@50", 1, "F1@50"),
    ("edit", 1, "edit"), ("over_segmentation_ratio", 1, "over-seg"),
    ("median_latency", 1, "lat(s)"), ("miss_rate", 100, "miss%"),
    ("ece", 100, "ECE"), ("interior_ece", 100, "int-ECE"),
    ("boundary_ece", 100, "bound-ECE"),
]


def write_table(res):
    cols = [t[2] for t in TABLE_METRICS]
    lines = ["| variant | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for n, v in res.items():
        cells = []
        for key, sc, _ in TABLE_METRICS:
            if key in v:
                m, s = agg(v, key)
                cells.append(f"{m*sc:.1f}")
            else:
                cells.append("-")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    (OUT / "summary_table.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = discover(args.results, args.variants)
    if not res:
        raise SystemExit(f"no npz variants found under {args.results}")
    print("variants:", list(res))
    fig_boundary_ece(res)
    fig_segmental_f1(res)
    fig_simple_bar(res, "over_segmentation_ratio", "pred/GT segment ratio",
                   "Over-segmentation (online heads fragment)", "fig2_oversegmentation.png")
    fig_simple_bar(res, "median_latency", "median latency (s)",
                   "Transition detection latency", "fig4_latency.png")
    print(write_table(res))
    print(f"\nfigures + summary_table.md -> {OUT}/")


if __name__ == "__main__":
    main()
