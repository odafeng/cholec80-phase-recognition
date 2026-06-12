"""Paired significance testing: Baseline vs +BUA across the reliability suite.

Generalises bootstrap/code/significance.py from a single accuracy number to the
whole suite. Loads two result directories, each containing one npz per seed; each
npz holds per-test-video arrays keyed by metric name (written by evaluate.py).
For every metric we:
  * average each video across seeds (cross-seed stability -> one value/video),
  * run a paired Wilcoxon signed-rank test over the 40 test videos,
  * report mean delta, win count, and p-value, oriented by whether the metric is
    higher-is-better (accuracy/F1/edit/Jaccard) or lower-is-better
    (latency/ECE/over-seg/miss-rate).

Run:
  python significance.py --baseline results/tecno_baseline --bua results/tecno_bua
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# Metrics where a LOWER value is better (improvement = negative delta).
LOWER_IS_BETTER = {
    "ece", "mce", "boundary_ece", "interior_ece", "boundary_mce",
    "median_latency", "mean_latency", "miss_rate", "false_start_count",
    "over_segmentation_ratio",
}


def load_dir(d):
    """dir with <name>_s*.npz -> {metric: (n_seeds, n_videos)} stacked."""
    files = sorted(glob.glob(str(Path(d) / "*.npz")))
    if not files:
        raise FileNotFoundError(f"no .npz in {d}")
    per_seed = [dict(np.load(f)) for f in files]
    keys = set(per_seed[0])
    for ps in per_seed[1:]:
        keys &= set(ps)
    return {k: np.stack([ps[k] for ps in per_seed]) for k in sorted(keys)}, files


def compare(baseline, bua, metrics=None):
    """baseline/bua: {metric: (seeds, videos)} (or (videos,) for single seed).
    Returns list of per-metric result dicts."""
    keys = metrics or sorted(set(baseline) & set(bua))
    rows = []
    for k in keys:
        b = np.asarray(baseline[k]); u = np.asarray(bua[k])
        if b.ndim == 2:
            b = b.mean(0)
        if u.ndim == 2:
            u = u.mean(0)
        n = min(len(b), len(u))
        b, u = b[:n], u[:n]
        lower = k in LOWER_IS_BETTER
        delta = u - b                                  # bua - baseline
        improve = -delta if lower else delta           # positive == improvement
        wins = int((improve > 0).sum())
        # Wilcoxon needs some non-zero differences
        if np.allclose(b, u):
            stat, p = float("nan"), 1.0
        else:
            try:
                stat, p = wilcoxon(u, b, zero_method="zsplit")
            except ValueError:
                stat, p = float("nan"), 1.0
        rows.append({
            "metric": k, "baseline": float(b.mean()), "bua": float(u.mean()),
            "delta": float(delta.mean()), "improved": bool(improve.mean() > 0),
            "wins": wins, "n": n, "p": float(p), "lower_is_better": lower,
            "significant": bool(p < 0.05),
        })
    # Holm-Bonferroni family-wise correction across the metric family (~15 tests):
    # report adjusted p so significance survives multiple comparisons.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m = len(rows)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * rows[i]["p"])
        adj = max(adj, prev)                       # enforce monotonicity
        prev = adj
        rows[i]["p_holm"] = float(adj)
        rows[i]["significant_holm"] = bool(adj < 0.05)
    return rows


def print_table(rows):
    print(f"\n{'metric':<22}{'baseline':>10}{'+BUA':>10}{'delta':>9}"
          f"{'wins':>7}{'p':>9}{'p_holm':>9}  verdict")
    print("-" * 92)
    for r in rows:
        arrow = "v" if r["lower_is_better"] else "^"
        verdict = ""
        if r.get("significant_holm"):
            verdict = "SIG +" if r["improved"] else "SIG -(worse)"
        elif r["significant"]:
            verdict = "(raw only)" + (" +" if r["improved"] else " -worse")
        print(f"{r['metric']:<22}{r['baseline']:>10.3f}{r['bua']:>10.3f}"
              f"{r['delta']:>+9.3f}{r['wins']:>5}/{r['n']:<2}{r['p']:>9.4f}"
              f"{r.get('p_holm', float('nan')):>9.4f}  {arrow} {verdict}")
    print("-" * 92)
    print("  ^ higher-is-better  v lower-is-better  SIG = Holm-adjusted p<0.05  "
          "(raw only = nominal p<0.05 but not after correction)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--bua", required=True, type=Path)
    args = ap.parse_args()
    base, bf = load_dir(args.baseline)
    bua, uf = load_dir(args.bua)
    print(f"baseline: {len(bf)} seed files in {args.baseline}")
    print(f"+BUA    : {len(uf)} seed files in {args.bua}")
    rows = compare(base, bua)
    print_table(rows)


# --------------------------------------------------------------------------- #
# self-test (synthetic, in-memory)
# --------------------------------------------------------------------------- #
def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(1)
    V = 40
    # baseline accuracy ~0.89; BUA slightly higher and consistently so
    base_acc = rng.normal(0.89, 0.05, V)
    bua_acc = base_acc + rng.normal(0.008, 0.004, V)        # small but consistent gain
    # baseline ECE high; BUA much lower (clear improvement on a lower-is-better metric)
    base_ece = rng.normal(0.12, 0.03, V).clip(0.01)
    bua_ece = (base_ece - rng.normal(0.05, 0.01, V)).clip(0.001)
    # boundary_ece: BUA clearly better
    base_bece = rng.normal(0.22, 0.04, V).clip(0.01)
    bua_bece = (base_bece - rng.normal(0.08, 0.02, V)).clip(0.001)

    rows = compare(
        {"accuracy": base_acc, "ece": base_ece, "boundary_ece": base_bece},
        {"accuracy": bua_acc, "ece": bua_ece, "boundary_ece": bua_bece},
    )
    by = {r["metric"]: r for r in rows}
    check("accuracy improvement significant", by["accuracy"]["significant"] and by["accuracy"]["improved"])
    check("ece improvement significant (lower better)",
          by["ece"]["significant"] and by["ece"]["improved"] and by["ece"]["delta"] < 0)
    check("boundary_ece improvement significant",
          by["boundary_ece"]["significant"] and by["boundary_ece"]["improved"])

    # a NULL difference must NOT be significant
    same = rng.normal(0.9, 0.05, V)
    rows2 = compare({"accuracy": same}, {"accuracy": same.copy()})
    check("identical arrays -> not significant", not rows2[0]["significant"])

    # a metric where BUA is WORSE is flagged as worse, not improved
    rows3 = compare({"ece": base_ece}, {"ece": base_ece + 0.05})
    check("worse lower-is-better flagged not-improved", not rows3[0]["improved"])

    print_table(rows)
    print("\n" + ("ALL SIGNIFICANCE SELF-TESTS PASSED" if ok else "*** SELF-TEST FAILURES ***"))
    return ok


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        sys.exit(0 if _selftest() else 1)
