"""Corrected statistics for the QCD reliability study (fixes the red-team findings).

The earlier `significance.py` pre-averaged seeds before a paired Wilcoxon over the 40
test videos. That throws away the seed dimension and gives no interval estimate. This
module instead treats the data as **per-(video, seed)** and reports:

  * cluster-bootstrap 95% CIs (resample the 40 *videos* with replacement — the cluster
    unit — keeping all seeds; this respects that seeds on the same video are correlated),
  * paired deltas (decoder/variant vs baseline) with a cluster-bootstrap CI on the delta
    AND a paired Wilcoxon p-value (per-video, seed-averaged), Holm-corrected across the
    metric family,
  * TOST non-inferiority for accuracy against a pre-registered margin.

Data: results/rel_<backbone>/<head>_<variant>/s*.json, each {videoXX: {metrics...}}.
Run:  env -u LD_LIBRARY_PATH python stats_corrected.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESULTS = Path("results")
BACKBONES = ["rel_rn50", "rel_e2e", "rel_endovit", "rel_endovitft", "rel_surgmae"]
HEADS = ["tecno", "lovit_causal", "asformer"]
# lower-is-better metrics (improvement = decrease)
LOWER = {"over_seg_ratio", "median_latency", "false_start_count", "boundary_ece",
         "interior_ece", "ece", "miss_rate"}
# (flat name -> path in the per-video json)
METRIC_PATHS = {
    "accuracy": ("accuracy",),
    "relaxed_accuracy": ("relaxed_accuracy",),
    "f1@10": ("f1@10",),
    "edit": ("edit",),
    "over_seg_ratio": ("over_segmentation", "ratio"),
    "median_latency": ("latency", "median_latency"),
    "false_start_count": ("latency", "false_start_count"),
}
# calibration metrics live only in the baseline .npz (per-video), not the rel json
RNG = np.random.default_rng(0)


def _dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return np.nan
        d = d[k]
    return float(d) if d is not None else np.nan


def load_variant(backbone: str, head: str, variant: str) -> dict[str, np.ndarray]:
    """Return {metric: (n_seeds, n_videos)} or {} if missing."""
    d = RESULTS / backbone / f"{head}_{variant}"
    files = sorted(glob.glob(str(d / "s*.json")))
    if not files:
        return {}
    out: dict[str, list] = {m: [] for m in METRIC_PATHS}
    vids = None
    for f in files:
        js = json.load(open(f))
        if vids is None:
            vids = sorted(js)
        for m, path in METRIC_PATHS.items():
            out[m].append([_dig(js[v], path) for v in vids])
    return {m: np.asarray(rows, float) for m, rows in out.items()}


def cluster_bootstrap_mean(mat: np.ndarray, b: int = 2000) -> tuple[float, float, float]:
    """mat: (seeds, videos). Mean + 95% CI, resampling videos (clusters)."""
    seeds, nv = mat.shape
    point = float(np.nanmean(mat))
    boot = np.empty(b)
    for i in range(b):
        idx = RNG.integers(0, nv, nv)
        boot[i] = np.nanmean(mat[:, idx])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired(a: np.ndarray, bsl: np.ndarray, metric: str, b: int = 2000) -> dict:
    """a, bsl: (seeds, videos). Improvement of `a` over baseline `bsl`."""
    av, bv = np.nanmean(a, 0), np.nanmean(bsl, 0)  # seed-mean per video
    keep = ~(np.isnan(av) | np.isnan(bv))
    av, bv = av[keep], bv[keep]
    nv = len(av)
    sign = -1.0 if metric in LOWER else 1.0
    delta = sign * (av - bv)  # positive == improvement
    # cluster bootstrap CI on the mean delta
    boot = np.array([np.mean(delta[RNG.integers(0, nv, nv)]) for _ in range(b)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    if np.allclose(av, bv):
        p = 1.0
    else:
        try:
            p = float(wilcoxon(av, bv, zero_method="zsplit").pvalue)
        except ValueError:
            p = 1.0
    return {"metric": metric, "mean_delta": float(np.mean(delta)),
            "ci": [float(lo), float(hi)], "p": p, "wins": int((delta > 0).sum()), "n": nv}


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def tost_noninferiority(a: np.ndarray, bsl: np.ndarray, margin: float, b: int = 2000) -> dict:
    """Non-inferiority of `a` vs baseline on a higher-is-better metric (accuracy):
    a is non-inferior if the lower bound of the (a-baseline) mean CI exceeds -margin.
    Reports the one-sided bootstrap p that mean(a-baseline) > -margin."""
    av, bv = np.nanmean(a, 0), np.nanmean(bsl, 0)
    keep = ~(np.isnan(av) | np.isnan(bv))
    d = (av - bv)[keep]
    nv = len(d)
    boot = np.array([np.mean(d[RNG.integers(0, nv, nv)]) for _ in range(b)])
    lo = float(np.percentile(boot, 2.5))
    p_ni = float(np.mean(boot <= -margin))  # prob the true delta is below the margin
    return {"mean_delta": float(np.mean(d)), "ci_lo": lo, "margin": -margin,
            "non_inferior": lo > -margin, "p_inferior": p_ni}


def pool_grid(variant: str) -> dict[str, np.ndarray]:
    """Stack a variant across all backbone x head cells -> {metric: (cells*seeds, videos)}."""
    pooled: dict[str, list] = {m: [] for m in METRIC_PATHS}
    for bb in BACKBONES:
        for h in HEADS:
            d = load_variant(bb, h, variant)
            for m in METRIC_PATHS:
                if m in d:
                    pooled[m].append(d[m])
    return {m: np.vstack(rows) for m, rows in pooled.items() if rows}


def boundary_ece_contrast(b: int = 2000) -> dict:
    """Pool per-video boundary- vs interior-ECE from results/baseline/*.npz and test
    that boundary-region ECE exceeds the interior (the headline novel-metric claim)."""
    bnd, intr = [], []
    for f in sorted(glob.glob("results/baseline/*.npz")):
        d = dict(np.load(f))
        if "boundary_ece" in d and "interior_ece" in d:
            bnd.append(np.asarray(d["boundary_ece"], float))
            intr.append(np.asarray(d["interior_ece"], float))
    if not bnd:
        return {}
    bnd, intr = np.concatenate(bnd), np.concatenate(intr)  # (configs*videos,)
    keep = ~(np.isnan(bnd) | np.isnan(intr))
    bnd, intr = bnd[keep], intr[keep]
    n = len(bnd)
    diff = bnd - intr
    boot_ratio = np.array([np.mean(bnd[idx]) / np.mean(intr[idx])
                           for idx in (RNG.integers(0, n, n) for _ in range(b))])
    p = float(wilcoxon(bnd, intr).pvalue) if not np.allclose(bnd, intr) else 1.0
    rlo, rhi = np.percentile(boot_ratio, [2.5, 97.5])
    return {"boundary_mean": float(np.mean(bnd)), "interior_mean": float(np.mean(intr)),
            "ratio": float(np.mean(bnd) / np.mean(intr)), "ratio_ci": [float(rlo), float(rhi)],
            "mean_diff": float(np.mean(diff)), "p": p, "n_units": n}


def main():
    out = {"baseline_ci": {}, "bua_vs_baseline": {}, "tost": {}, "boundary_ece": {}}

    print("=" * 78)
    print("BASELINE (argmax) metrics pooled over 5 backbones x 3 heads x 5 seeds")
    print("  cluster-bootstrap mean [95% CI] over 40 test videos")
    print("=" * 78)
    base = pool_grid("baseline")
    for m in METRIC_PATHS:
        if m not in base:
            continue
        pt, lo, hi = cluster_bootstrap_mean(base[m])
        out["baseline_ci"][m] = {"mean": pt, "ci": [lo, hi]}
        print(f"  {m:20s} {pt:8.3f}  [{lo:7.3f}, {hi:7.3f}]")

    print("\n" + "=" * 78)
    print("BUA heuristic vs baseline (paired, seed-mean per video) — Holm across metrics")
    print("  delta>0 = improvement; CI is cluster-bootstrap on the delta")
    print("=" * 78)
    bua = pool_grid("bua")
    rows, metrics = [], []
    for m in METRIC_PATHS:
        if m in base and m in bua:
            rows.append(paired(bua[m], base[m], m))
            metrics.append(m)
    for r, padj in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = padj
        sig = "*" if padj < 0.05 else " "
        out["bua_vs_baseline"][r["metric"]] = r
        print(f"  {r['metric']:20s} Δ={r['mean_delta']:+8.3f}  "
              f"[{r['ci'][0]:+7.3f},{r['ci'][1]:+7.3f}]  p={r['p']:.4f}  "
              f"p_holm={padj:.4f} {sig}")

    print("\n" + "=" * 78)
    print("TOST non-inferiority of accuracy (margin = 1.0% absolute)")
    print("=" * 78)
    for variant, label in [("bua", "BUA"), ("calib", "calibrated")]:
        v = pool_grid(variant)
        if "accuracy" in v and "accuracy" in base:
            t = tost_noninferiority(v["accuracy"], base["accuracy"], margin=0.01)
            out["tost"][variant] = t
            verdict = "NON-INFERIOR" if t["non_inferior"] else "cannot conclude NI"
            print(f"  {label:12s} Δacc={t['mean_delta']:+.4f}  CI_lo={t['ci_lo']:+.4f}  "
                  f"(margin {t['margin']:+.3f}) -> {verdict}")

    print("\n" + "=" * 78)
    print("BOUNDARY-region ECE vs INTERIOR ECE (per-video, baseline npz, cluster-bootstrap)")
    print("=" * 78)
    bec = boundary_ece_contrast()
    out["boundary_ece"] = bec
    if bec:
        print(f"  boundary-ECE {bec['boundary_mean']:.3f}  vs  interior-ECE "
              f"{bec['interior_mean']:.3f}   ratio {bec['ratio']:.2f}x "
              f"[{bec['ratio_ci'][0]:.2f}, {bec['ratio_ci'][1]:.2f}]   "
              f"Wilcoxon p={bec['p']:.2e}  (n={bec['n_units']} video-configs)")

    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/stats_corrected.json", "w"), indent=2)
    print("\n-> results/stats_corrected.json")


def _selftest():
    rng = np.random.default_rng(1)
    a = rng.normal(0.9, 0.02, (5, 40))
    bsl = a - rng.normal(0.01, 0.005, (5, 40))  # a is slightly better
    r = paired(a, bsl, "accuracy")
    assert r["mean_delta"] > 0 and r["ci"][0] < r["ci"][1]
    assert abs(holm([0.01, 0.5, 0.5])[0] - 0.03) < 1e-9  # smallest p * m
    t = tost_noninferiority(a, bsl, 0.05)
    assert t["non_inferior"]
    print("selftest OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
