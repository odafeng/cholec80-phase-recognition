"""Per-video cluster-bootstrap CIs for the QCD frontier operating points (#16 item 2).

`qcd_p2.json` reported the frontier as point estimates only (means over the 40 test
videos x 5 seeds). This adds interval estimates so the frontier-dominance claim rests
on CIs, not point estimates. For each (backbone, head, seed) we pick the operating
point at over-seg <= OSEG_TARGET on each decoder's frontier (exactly as `qcd_p2`), then
record the PER-VIDEO metric at that point. Stacking over cells x seeds gives
(units, videos) matrices; we cluster-bootstrap over the 40 videos (the cluster unit):

  * each decoder's operating-point metric: mean [95% CI]
  * QCD-vs-heuristic paired delta on detection latency at matched over-seg
    (the frontier-dominance claim): cluster-bootstrap CI + paired Wilcoxon, Holm-corrected.

Reuses qcd_p2's posteriors/frontier code and stats_corrected's bootstrap helpers.
CPU-only. Run:  env -u LD_LIBRARY_PATH python frontier_ci.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qcd_p2 import (FEAT as _FEAT_BASE, HEADS, SEEDS, H_GRID, WINDOWS, GAMMAS,
                    OSEG_TARGET, graph_from_train, posteriors)

# qcd_p2's FEAT predates the surgical-MAE backbone; add it so the frontier covers the
# full 5-backbone grid (stats_corrected already includes rel_surgmae).
FEAT = {**_FEAT_BASE, "surgmae": "features_surgmae"}
from qcd import cusum_decode, fully_connected_graph
from smooth import online_uncertainty_smooth, causal_mode_filter
from metrics import accuracy, over_segmentation, transition_latency, segmental_f1
from stats_corrected import cluster_bootstrap_mean, paired, holm

METRICS = ("oseg", "lat", "acc", "f1")


def ev_pv(decode, seqs):
    """Per-video metric arrays for one decoder setting. seqs: list of (probs, gt)."""
    o, l, a, f = [], [], [], []
    for p, gt in seqs:
        pred = decode(p)
        n = min(len(gt), len(pred))
        pred, g = pred[:n], gt[:n]
        o.append(over_segmentation(pred, g)["ratio"])
        l.append(transition_latency(pred, g)["median_latency"])
        a.append(accuracy(pred, g) * 100)
        f.append(segmental_f1(pred, g, 0.10)[0])
    return {"oseg": np.asarray(o, float), "lat": np.asarray(l, float),
            "acc": np.asarray(a, float), "f1": np.asarray(f, float)}


def select_op(frontier, target=OSEG_TARGET):
    """frontier: list of per-video dicts. Pick min aggregate-latency with aggregate
    over-seg <= target (else the lowest-over-seg point) — same rule as qcd_p2.op_at_oseg,
    but applied to the aggregate while returning the chosen point's per-video arrays."""
    aggs = [{k: float(np.nanmean(pv[k])) for k in METRICS} for pv in frontier]
    elig = [i for i, ag in enumerate(aggs) if ag["oseg"] <= target]
    i = min(elig, key=lambda i: aggs[i]["lat"]) if elig else \
        min(range(len(aggs)), key=lambda i: aggs[i]["oseg"])
    return frontier[i]


def cusum_frontier_pv(seqs, graph):
    return [ev_pv(lambda p, h=h: cusum_decode(p, graph, h), seqs) for h in H_GRID]


def heur_frontier_pv(seqs):
    pts = [ev_pv(lambda p, w=w: causal_mode_filter(p.argmax(1), w), seqs) for w in WINDOWS]
    pts += [ev_pv(lambda p, g=g: online_uncertainty_smooth(p, gamma=g).argmax(1), seqs)
            for g in GAMMAS]
    return pts


def collect():
    """Per decoder, stack the chosen operating point's per-video metric over all
    (backbone, head, seed) cells -> {decoder: {metric: (units, videos)}}."""
    G = graph_from_train()
    Gf = fully_connected_graph()
    decoders = {"qcd_est": [], "qcd_full": [], "heuristic": []}  # each: list of per-video dicts
    n_vid = None
    cells = 0
    for bb in FEAT:
        for head in HEADS:
            for s in SEEDS:
                ck = Path(f"checkpoints/rel_{bb}/{head}_base_s{s}.pt")
                if not ck.exists():
                    continue
                seqs = posteriors(ck, FEAT[bb])
                if n_vid is None:
                    n_vid = len(seqs)
                if len(seqs) != n_vid:
                    print(f"  skip {bb}/{head} s{s}: {len(seqs)} videos != {n_vid}")
                    continue
                chosen = {
                    "qcd_est": select_op(cusum_frontier_pv(seqs, G)),
                    "qcd_full": select_op(cusum_frontier_pv(seqs, Gf)),
                    "heuristic": select_op(heur_frontier_pv(seqs)),
                }
                for d, pv in chosen.items():
                    decoders[d].append(pv)
                cells += 1
                print(f"  {bb}/{head} s{s}: "
                      f"qcd_est lat={np.nanmean(chosen['qcd_est']['lat']):.1f}/oseg={np.nanmean(chosen['qcd_est']['oseg']):.2f}  "
                      f"heur lat={np.nanmean(chosen['heuristic']['lat']):.1f}/oseg={np.nanmean(chosen['heuristic']['oseg']):.2f}")
    # -> {decoder: {metric: (units, videos)}}
    out = {}
    for d, rows in decoders.items():
        if not rows:
            continue
        out[d] = {m: np.vstack([pv[m] for pv in rows]) for m in METRICS}
    return out, cells, (n_vid or 0)


def main():
    print(f"=== QCD frontier per-video bootstrap CIs (oseg target <= {OSEG_TARGET}) ===")
    mats, cells, n_vid = collect()
    if not mats:
        print("no cells found — checkpoints missing?")
        return
    res = {"oseg_target": OSEG_TARGET, "n_cells": cells, "n_videos": n_vid,
           "operating_point": {}, "dominance": {}}

    print(f"\n--- operating-point metric  mean [95% CI]  (pooled over {cells} cells, "
          f"cluster-bootstrap over {n_vid} videos) ---")
    print(f"{'decoder':12s}" + "".join(f"{m:>22s}" for m in METRICS))
    for d, mm in mats.items():
        row = {}
        cells_txt = ""
        for m in METRICS:
            pt, lo, hi = cluster_bootstrap_mean(mm[m])
            row[m] = {"mean": pt, "ci": [lo, hi]}
            cells_txt += f"{pt:7.2f}[{lo:5.1f},{hi:5.1f}]"
        res["operating_point"][d] = row
        print(f"{d:12s}{cells_txt}")

    # frontier dominance: QCD vs heuristic on detection latency at matched over-seg
    print(f"\n--- frontier dominance: latency delta vs heuristic at matched over-seg "
          f"(<= {OSEG_TARGET}) ---")
    print("  delta>0 = QCD LOWER latency (better); CI = cluster-bootstrap on the paired delta")
    rows, labels = [], []
    for d in ("qcd_est", "qcd_full"):
        if d in mats and "heuristic" in mats:
            for m in ("lat", "acc", "f1", "oseg"):
                metric_name = {"lat": "median_latency", "oseg": "over_seg_ratio"}.get(m, m)
                r = paired(mats[d][m], mats["heuristic"][m], metric_name)
                r["_decoder"], r["_metric"] = d, m
                rows.append(r)
                labels.append((d, m))
    for r, padj in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = padj
        sig = "*" if padj < 0.05 else " "
        res["dominance"].setdefault(r["_decoder"], {})[r["_metric"]] = r
        print(f"  {r['_decoder']:9s} {r['_metric']:5s}  Δ={r['mean_delta']:+7.2f}  "
              f"[{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}]  p={r['p']:.4f}  "
              f"p_holm={padj:.4f} {sig}  (wins {r['wins']}/{r['n']})")

    Path("results").mkdir(exist_ok=True)
    Path("results/frontier_ci.json").write_text(json.dumps(res, indent=2, default=float))
    print("\n-> results/frontier_ci.json")
    print("FRONTIER_CI_EXIT=0")


def _selftest():
    """Per-video eval + operating-point selection on a tiny synthetic stream."""
    rng = np.random.default_rng(0)
    K = 3
    T = 120

    def seq(changes):
        z = np.zeros(T, int)
        for t, ph in changes:
            z[t:] = ph
        lg = rng.normal(0, 1.0, (T, K))
        lg[np.arange(T), z] += 1.5
        e = np.exp(lg - lg.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), z

    seqs = [seq([(0, 0), (40, 1), (80, 2)]) for _ in range(4)]
    G = {0: [1], 1: [2], 2: []}
    front = cusum_frontier_pv(seqs, G)
    assert all(len(pv["lat"]) == 4 for pv in front), "per-video arrays wrong length"
    op = select_op(front)
    assert set(op) == set(METRICS) and len(op["oseg"]) == 4
    # cluster bootstrap shape sanity via stats helper
    pt, lo, hi = cluster_bootstrap_mean(np.vstack([op["acc"], op["acc"]]))
    assert lo <= pt <= hi
    print("frontier_ci selftest OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
