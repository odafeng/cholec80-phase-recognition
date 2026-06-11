"""Reliability evaluation suite for ONLINE surgical phase recognition (Cholec80).

This is the core deliverable of the project (see RESEARCH_PLAN.md). It is pure
numpy so it is unit-testable on CPU with no GPU/data: run `python metrics.py` to
execute the self-test.

Conventions (match evaluate.py / dataset.py):
  * sequences are 1-D integer arrays of phase ids in [0, NUM_PHASES) at **1 fps**,
    so 1 frame == 1 second. `tol`/latency are therefore in seconds == frames.
  * `probs` is a (T, C) array of per-frame class probabilities (already softmaxed
    and, where relevant, temperature-scaled).

Metric groups
-------------
  accuracy / relaxed accuracy          relaxed_correct(), accuracy()
  per-phase relaxed P/R/Jaccard        per_phase_relaxed()
  segmental F1@{10,25,50} & edit       segmental_f1(), segmental_edit()
  over-segmentation                    over_segmentation()
  transition latency / miss / false    transition_latency()
  calibration (ECE/MCE, interior vs    calibration()
    boundary-region ECE)

The "relaxed boundary" follows the standard Cholec80 convention used by TeCNO /
Jin et al. and discussed by Funke et al. ("Metrics Matter ..."): within +/-tol
frames of a ground-truth phase transition, predicting *either* of the two phases
adjacent to that transition is not penalised.
"""
from __future__ import annotations

import numpy as np

# Local import kept optional so metrics.py can be unit-tested standalone.
try:
    from phases import NUM_PHASES
except Exception:  # pragma: no cover - fallback for standalone testing
    NUM_PHASES = 7


# --------------------------------------------------------------------------- #
# segments
# --------------------------------------------------------------------------- #
def get_segments(seq):
    """Run-length encode a label sequence.

    Returns a list of (label, start, end) with end EXCLUSIVE, so
    seq[start:end] is a constant run of `label`.
    """
    seq = np.asarray(seq)
    if seq.size == 0:
        return []
    change = np.where(np.diff(seq) != 0)[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(seq)]])
    return [(int(seq[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def transition_frames(gt):
    """Frame indices that are the FIRST frame of a new phase (i.e. gt[t]!=gt[t-1])."""
    gt = np.asarray(gt)
    if gt.size == 0:
        return np.array([], dtype=int)
    return np.where(np.diff(gt) != 0)[0] + 1


def boundary_mask(gt, tol=10):
    """Boolean mask: True for frames within +/-tol of any GT transition.

    A transition occurs *between* frame b-1 and b (b = transition_frames). A frame
    t is in the boundary region if |t - b| <= tol for some transition b... we use
    the half-open convention that the window covers [b-tol, b+tol-1] so that the
    window width is exactly 2*tol frames centred on the boundary.
    """
    gt = np.asarray(gt)
    mask = np.zeros(len(gt), dtype=bool)
    for b in transition_frames(gt):
        lo = max(0, b - tol)
        hi = min(len(gt), b + tol)
        mask[lo:hi] = True
    return mask


# --------------------------------------------------------------------------- #
# accuracy (strict + relaxed)
# --------------------------------------------------------------------------- #
def relaxed_correct(pred, gt, tol=10):
    """Per-frame correctness under the relaxed-boundary rule.

    A frame is correct if pred==gt, OR it lies within +/-tol of a GT transition and
    pred equals one of the two phases adjacent to that transition.
    With tol=0 this reduces to strict correctness.
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    correct = (pred == gt).copy()
    if tol <= 0:
        return correct
    for b in transition_frames(gt):           # transition between b-1 and b
        before, after = gt[b - 1], gt[b]
        lo = max(0, b - tol)
        hi = min(len(gt), b + tol)
        for t in range(lo, hi):
            if pred[t] == before or pred[t] == after:
                correct[t] = True
    return correct


def accuracy(pred, gt, relaxed=False, tol=10):
    if relaxed:
        return float(relaxed_correct(pred, gt, tol).mean())
    return float((np.asarray(pred) == np.asarray(gt)).mean())


def _relabel_relaxed(pred, gt, tol=10):
    """Return a copy of pred where, inside relaxed windows, a prediction matching a
    neighbouring GT phase is snapped to gt. Used for relaxed per-phase P/R/Jaccard,
    mirroring the Cholec80 MATLAB toolkit's relabel-then-score approach."""
    pred = np.asarray(pred).copy()
    gt = np.asarray(gt)
    if tol <= 0:
        return pred
    for b in transition_frames(gt):
        before, after = gt[b - 1], gt[b]
        lo = max(0, b - tol)
        hi = min(len(gt), b + tol)
        for t in range(lo, hi):
            if pred[t] == before or pred[t] == after:
                pred[t] = gt[t]
    return pred


def per_phase_relaxed(pred, gt, num_classes=NUM_PHASES, relaxed=True, tol=10):
    """Per-phase precision / recall / Jaccard(IoU).

    With relaxed=True, predictions inside transition windows that match a
    neighbouring phase are corrected first (standard Cholec80 relaxed scoring).
    Returns dict of arrays length num_classes plus their macro means.
    """
    gt = np.asarray(gt)
    p = _relabel_relaxed(pred, gt, tol) if relaxed else np.asarray(pred)
    prec = np.zeros(num_classes)
    rec = np.zeros(num_classes)
    jac = np.zeros(num_classes)
    for c in range(num_classes):
        tp = int(np.sum((p == c) & (gt == c)))
        fp = int(np.sum((p == c) & (gt != c)))
        fn = int(np.sum((p != c) & (gt == c)))
        prec[c] = tp / (tp + fp) if (tp + fp) else 0.0
        rec[c] = tp / (tp + fn) if (tp + fn) else 0.0
        jac[c] = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        "precision": prec, "recall": rec, "jaccard": jac,
        "mean_precision": float(prec.mean()),
        "mean_recall": float(rec.mean()),
        "mean_jaccard": float(jac.mean()),
    }


# --------------------------------------------------------------------------- #
# segmental metrics (action-segmentation standard, Lea et al. 2017)
# --------------------------------------------------------------------------- #
def _levenshtein(seq_a, seq_b):
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1,
                           dp[i, j - 1] + 1,
                           dp[i - 1, j - 1] + cost)
    return int(dp[n, m])


def segmental_edit(pred, gt):
    """Normalised segmental edit score in [0,100] (100 = perfect ordering).

    Edit distance over the SEQUENCE OF SEGMENT LABELS (order, not duration).
    """
    p_lab = [s[0] for s in get_segments(pred)]
    g_lab = [s[0] for s in get_segments(gt)]
    if len(p_lab) == 0 and len(g_lab) == 0:
        return 100.0
    dist = _levenshtein(p_lab, g_lab)
    norm = max(len(p_lab), len(g_lab))
    return float((1.0 - dist / norm) * 100.0) if norm else 100.0


def segmental_f1(pred, gt, overlap=0.5):
    """Segmental F1@overlap (Lea et al.).

    Each predicted segment is a TP if it has IoU >= overlap with an unmatched GT
    segment of the SAME label; else FP. Unmatched GT segments are FN.
    Returns (f1, precision, recall, tp, fp, fn).
    """
    p_seg = get_segments(pred)
    g_seg = get_segments(gt)
    matched = np.zeros(len(g_seg), dtype=bool)
    tp = 0
    fp = 0
    for (pl, ps, pe) in p_seg:
        best_iou, best_j = 0.0, -1
        for j, (gl, gs, ge) in enumerate(g_seg):
            if gl != pl or matched[j]:
                continue
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= overlap and best_j >= 0:
            tp += 1
            matched[best_j] = True
        else:
            fp += 1
    fn = int(np.sum(~matched))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return float(f1 * 100), float(prec * 100), float(rec * 100), tp, fp, fn


def over_segmentation(pred, gt):
    """Predicted vs GT segment counts and their ratio (>1 => over-segmented)."""
    n_pred = len(get_segments(pred))
    n_gt = len(get_segments(gt))
    return {
        "pred_segments": n_pred,
        "gt_segments": n_gt,
        "ratio": float(n_pred / n_gt) if n_gt else float("inf"),
    }


# --------------------------------------------------------------------------- #
# transition detection latency (online by construction)
# --------------------------------------------------------------------------- #
def transition_latency(pred, gt, stable_k=3):
    """Per GT transition into phase b at frame t, latency = first t'>=t where pred
    becomes b and stays b for >= stable_k frames (clipped to the GT b-segment),
    minus t. If never satisfied within the GT b-segment -> missed. A false start is
    a switch to b that begins strictly before t (within the preceding segment).

    Returns dict with per-transition latencies (s), median/mean latency over
    DETECTED transitions, miss_rate, false_start_count, n_transitions.
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    g_seg = get_segments(gt)
    latencies = []
    missed = 0
    false_starts = 0
    n_trans = 0
    for idx in range(1, len(g_seg)):
        b, t, e = g_seg[idx]          # phase b starts at frame t, ends at e (excl)
        prev_start = g_seg[idx - 1][1]
        n_trans += 1
        # detection: first frame in [t, e) where pred==b stably for stable_k frames
        detected_at = None
        for tp in range(t, e):
            k = min(stable_k, e - tp)
            if np.all(pred[tp:tp + k] == b):
                detected_at = tp
                break
        if detected_at is None:
            missed += 1
        else:
            latencies.append(detected_at - t)
        # false start: pred shows b somewhere in the previous segment [prev_start, t)
        if np.any(pred[prev_start:t] == b):
            false_starts += 1
    lat = np.asarray(latencies, dtype=float)
    return {
        "latencies": latencies,
        "median_latency": float(np.median(lat)) if lat.size else float("nan"),
        "mean_latency": float(lat.mean()) if lat.size else float("nan"),
        "miss_rate": float(missed / n_trans) if n_trans else 0.0,
        "false_start_count": int(false_starts),
        "n_transitions": int(n_trans),
    }


# --------------------------------------------------------------------------- #
# calibration (ECE / MCE, reliability diagram, interior vs boundary-region ECE)
# --------------------------------------------------------------------------- #
def _ece_from_conf(conf, correct, bins=15):
    """Expected & Maximum Calibration Error from confidences + correctness."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return 0.0, 0.0, []
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    diagram = []
    n = conf.size
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(sel.sum())
        if cnt == 0:
            diagram.append((0.5 * (lo + hi), float("nan"), float("nan"), 0))
            continue
        avg_conf = float(conf[sel].mean())
        avg_acc = float(correct[sel].mean())
        gap = abs(avg_acc - avg_conf)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        diagram.append((0.5 * (lo + hi), avg_conf, avg_acc, cnt))
    return float(ece), float(mce), diagram


def calibration(probs, gt, bins=15, boundary_tol=10):
    """Full calibration report.

    Returns dict with ece, mce, reliability diagram, plus interior_ece (frames
    OUTSIDE +/-boundary_tol of any GT transition) and boundary_ece (frames INSIDE).
    The interior-vs-boundary contrast is the headline metric (RESEARCH_PLAN.md).
    """
    probs = np.asarray(probs, dtype=float)
    gt = np.asarray(gt)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == gt).astype(float)

    ece, mce, diagram = _ece_from_conf(conf, correct, bins)

    bmask = boundary_mask(gt, boundary_tol)
    imask = ~bmask
    b_ece, b_mce, _ = _ece_from_conf(conf[bmask], correct[bmask], bins)
    i_ece, i_mce, _ = _ece_from_conf(conf[imask], correct[imask], bins)

    return {
        "ece": ece, "mce": mce, "diagram": diagram,
        "boundary_ece": b_ece, "boundary_mce": b_mce,
        "interior_ece": i_ece, "interior_mce": i_mce,
        "n_boundary": int(bmask.sum()), "n_interior": int(imask.sum()),
        "mean_confidence": float(conf.mean()),
        "accuracy": float(correct.mean()),
    }


# --------------------------------------------------------------------------- #
# convenience: full per-video report
# --------------------------------------------------------------------------- #
def evaluate_video(pred, gt, probs=None, tol=10, stable_k=3,
                   num_classes=NUM_PHASES, bins=15):
    """Compute the whole suite for one video. probs optional (skips calibration)."""
    rep = {
        "accuracy": accuracy(pred, gt, relaxed=False),
        "relaxed_accuracy": accuracy(pred, gt, relaxed=True, tol=tol),
        "f1@10": segmental_f1(pred, gt, 0.10)[0],
        "f1@25": segmental_f1(pred, gt, 0.25)[0],
        "f1@50": segmental_f1(pred, gt, 0.50)[0],
        "edit": segmental_edit(pred, gt),
        "over_segmentation": over_segmentation(pred, gt),
        "latency": transition_latency(pred, gt, stable_k),
    }
    rep.update({"perphase_" + k: v
                for k, v in per_phase_relaxed(pred, gt, num_classes, True, tol).items()})
    if probs is not None:
        rep["calibration"] = calibration(probs, gt, bins, tol)
    return rep


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # ---- strict accuracy consistency with (pred==gt).mean() ----
    rng = np.random.default_rng(0)
    gt = np.array([0]*10 + [1]*10 + [2]*10)
    pred = gt.copy()
    pred[5] = 3          # one interior error (far from boundary)
    check("strict accuracy == (pred==gt).mean()",
          abs(accuracy(pred, pred*0+gt, relaxed=False) - (pred == gt).mean()) < 1e-12)
    check("relaxed(tol=0) == strict",
          accuracy(pred, gt, relaxed=True, tol=0) == accuracy(pred, gt, relaxed=False))

    # ---- relaxed boundary forgives a near-boundary neighbour prediction ----
    gt2 = np.array([0]*10 + [1]*10)
    pred2 = gt2.copy()
    pred2[9] = 1         # at frame 9 (boundary at 10), predict the upcoming phase 1
    pred2[10] = 0        # at frame 10, predict the previous phase 0
    check("relaxed forgives boundary neighbours (acc==1.0)",
          accuracy(pred2, gt2, relaxed=True, tol=10) == 1.0)
    check("strict penalises those 2 frames",
          abs(accuracy(pred2, gt2, relaxed=False) - 18/20) < 1e-12)
    # an interior far error is NOT forgiven
    pred3 = gt2.copy(); pred3[3] = 5
    check("relaxed does NOT forgive interior far error",
          accuracy(pred3, gt2, relaxed=True, tol=2) < 1.0)

    # ---- segments ----
    segs = get_segments(np.array([0,0,1,1,1,2]))
    check("get_segments", segs == [(0,0,2),(1,2,5),(2,5,6)])

    # ---- segmental edit: perfect order -> 100, reorder penalised ----
    check("edit perfect == 100", segmental_edit(gt2, gt2) == 100.0)
    over = np.array([0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1])  # flickery first half
    check("edit penalises over-segmentation", segmental_edit(over, gt2) < 100.0)

    # ---- segmental F1: identical -> 100; over-seg lowers precision ----
    f1_id = segmental_f1(gt2, gt2, 0.5)[0]
    check("F1@50 identical == 100", f1_id == 100.0)
    f1_over = segmental_f1(over, gt2, 0.5)[0]
    check("F1@50 over-seg < 100", f1_over < 100.0)

    # ---- over_segmentation count ----
    osg = over_segmentation(over, gt2)
    check("over_segmentation ratio>1", osg["ratio"] > 1.0 and osg["gt_segments"] == 2)

    # ---- transition latency: delayed detection ----
    gt4 = np.array([0]*10 + [1]*10)
    pred4 = np.array([0]*13 + [1]*7)   # phase 1 truly starts at 10, detected at 13
    lat = transition_latency(pred4, gt4, stable_k=3)
    check("latency == 3s", lat["median_latency"] == 3.0 and lat["n_transitions"] == 1)
    check("latency miss_rate == 0", lat["miss_rate"] == 0.0)
    # missed transition
    pred5 = np.array([0]*20)
    lat5 = transition_latency(pred5, gt4, stable_k=3)
    check("missed transition -> miss_rate 1.0", lat5["miss_rate"] == 1.0)
    # false start
    pred6 = np.array([0]*7 + [1]*3 + [0]*0 + [1]*10)  # shows phase1 at 7-9 then back
    pred6 = np.array([0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0])
    gt6 = np.array([0]*15 + [1]*5)
    lat6 = transition_latency(pred6, gt6, stable_k=2)
    check("false start detected", lat6["false_start_count"] == 1)

    # ---- calibration: over-confident wrong predictions -> high ECE; boundary worse ----
    T = 40
    gtc = np.array([0]*20 + [1]*20)
    # interior: confident & correct; boundary frames: confident & WRONG
    probs = np.full((T, 3), 0.02)
    pred_lab = gtc.copy()
    bm = boundary_mask(gtc, tol=5)
    pred_lab[bm] = 2                      # wrong, class 2, at boundary
    for t in range(T):
        probs[t] = 0.02
        probs[t, pred_lab[t]] = 0.96      # very confident on the (sometimes wrong) pred
    probs = probs / probs.sum(1, keepdims=True)
    cal = calibration(probs, gtc, bins=10, boundary_tol=5)
    check("boundary_ece > interior_ece (dilution)",
          cal["boundary_ece"] > cal["interior_ece"])
    check("interior near-perfectly calibrated", cal["interior_ece"] < 0.05)

    print("\n" + ("ALL METRICS SELF-TESTS PASSED" if ok else "*** SELF-TEST FAILURES ***"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
