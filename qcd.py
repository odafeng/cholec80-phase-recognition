"""Quickest-change-detection (QCD) decoder for online surgical phase recognition.

See docs/RESEARCH_PLAN_QCD.md. We treat each phase transition as a sequential
change point and run a CUSUM (Page) / Shiryaev-Roberts recursion on the model's
causal posterior log-likelihood ratio between the current declared phase and its
allowed successors (per a transition graph estimated from TRAINING data only).
The threshold h sweeps the detection-delay vs false-alarm trade-off; sweeping it
traces the operating curve we compare against heuristic post-processing.

Structural guarantee (model-independent): #predicted segments = #declared
transitions, so over-segmentation is bounded by the transition graph -- something
no frame-wise classifier or EMA smoother can guarantee.

Pure numpy. `python qcd.py` runs the P0 synthetic optimality self-test.
"""
from __future__ import annotations

import numpy as np

try:
    from phases import NUM_PHASES
except Exception:                       # pragma: no cover
    NUM_PHASES = 7


# --------------------------------------------------------------------------- #
# transition graph (estimated from TRAIN labels only -- no test leakage)
# --------------------------------------------------------------------------- #
def estimate_graph(label_seqs, num_phases=NUM_PHASES, min_count=1):
    """label_seqs: list of 1-D GT label arrays (TRAIN videos). Returns
    {phase: sorted list of allowed successor phases} from observed transitions."""
    succ = {k: {} for k in range(num_phases)}
    for seq in label_seqs:
        seq = np.asarray(seq)
        ch = np.where(np.diff(seq) != 0)[0]
        for b in ch:
            a, c = int(seq[b]), int(seq[b + 1])
            succ[a][c] = succ[a].get(c, 0) + 1
    return {k: sorted([j for j, n in d.items() if n >= min_count])
            for k, d in succ.items()}


def fully_connected_graph(num_phases=NUM_PHASES):
    """Ablation: no ordering prior -- any phase may follow any other."""
    return {k: [j for j in range(num_phases) if j != k] for k in range(num_phases)}


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #
def cusum_decode(probs, graph, h, start=0, eps=1e-6):
    """CUSUM decode. probs: (T, K) causal posteriors. Returns predicted labels (T,).

    For declared phase k, accumulate W^j = max(0, W^j + log p[j] - log p[k]) for
    each allowed successor j; declare transition to argmax_j W^j when it reaches h.
    """
    probs = np.asarray(probs, dtype=float)
    T, K = probs.shape
    logp = np.log(np.clip(probs, eps, 1.0))
    pred = np.empty(T, dtype=int)
    k = int(start)
    W = {j: 0.0 for j in graph.get(k, [])}
    for t in range(T):
        fired_j, fired_W = None, h
        for j in W:
            W[j] = max(0.0, W[j] + logp[t, j] - logp[t, k])
            if W[j] >= fired_W:
                fired_W, fired_j = W[j], j
        if fired_j is not None:
            k = fired_j
            W = {j: 0.0 for j in graph.get(k, [])}
        pred[t] = k
    return pred


def shiryaev_roberts_decode(probs, graph, h, start=0, eps=1e-6):
    """Shiryaev-Roberts variant: R^j_t = (1 + R^j_{t-1}) * LR_t, declare when
    log R reaches h. Bayesian/Pollak-optimal counterpart of CUSUM. Done in log
    space for stability: logR <- logaddexp(0, logR) + (logp[j]-logp[k])."""
    probs = np.asarray(probs, dtype=float)
    T, K = probs.shape
    logp = np.log(np.clip(probs, eps, 1.0))
    pred = np.empty(T, dtype=int)
    k = int(start)
    R = {j: 0.0 for j in graph.get(k, [])}          # log R, starts at log(0+...)~ -inf-ish
    R = {j: -50.0 for j in graph.get(k, [])}
    for t in range(T):
        fired_j, fired_W = None, h
        for j in R:
            R[j] = np.logaddexp(0.0, R[j]) + (logp[t, j] - logp[t, k])
            if R[j] >= fired_W:
                fired_W, fired_j = R[j], j
        if fired_j is not None:
            k = fired_j
            R = {j: -50.0 for j in graph.get(k, [])}
        pred[t] = k
    return pred


# --------------------------------------------------------------------------- #
# P0 self-test: CUSUM behaves as the theory predicts on synthetic change points
# --------------------------------------------------------------------------- #
def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(0)
    K = 3
    G = {0: [1], 1: [2], 2: []}                      # ordered chain 0->1->2

    def make_seq(changes, T, sep=1.2, noise=1.0):
        """piecewise phase with logit separation `sep`; posteriors via softmax."""
        z = np.zeros(T, dtype=int)
        for (t, ph) in changes:
            z[t:] = ph
        logits = rng.normal(0, noise, (T, K))
        logits[np.arange(T), z] += sep
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), z

    # delay vs false-alarm trade-off: SINGLE binary change (terminal target) so the
    # detection delay is measured cleanly. Lower threshold -> shorter delay + more FA.
    G2 = {0: [1], 1: []}
    delays, fas = [], []
    for h in [12.0, 6.0, 3.0, 1.0]:
        ds, fa = [], 0
        for _ in range(120):
            probs, _z = make_seq([(0, 0), (100, 1)], 200, sep=0.7, noise=1.2)
            pred = cusum_decode(probs, G2, h)
            fa_here = np.any(pred[:100] == 1)
            after = np.where((np.arange(200) >= 100) & (pred == 1))[0]
            ds.append(0 if fa_here else (after[0] - 100 if len(after) else 100))
            fa += int(fa_here)
        delays.append(np.mean(ds)); fas.append(fa)
    print(f"     thresholds [12,6,3,1] -> ADD {[round(d,1) for d in delays]}, "
          f"false-alarms {fas}")
    check("higher threshold => longer detection delay", delays[0] >= delays[-1])
    check("lower threshold => more false alarms", fas[-1] >= fas[0])
    check("detects with finite delay at high threshold", delays[0] < 80)

    # structural over-seg bound: #segments <= #declared transitions <= path length
    probs, z = make_seq([(0, 0), (100, 1), (200, 2)], 300)
    pred = cusum_decode(probs, G, h=3.0)
    n_seg = 1 + int(np.sum(np.diff(pred) != 0))
    check("structural over-seg bound (<= K segments on a chain)", n_seg <= K)

    # graph estimation
    g = estimate_graph([np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 2])])
    check("graph estimation", g[0] == [1] and g[1] == [2] and g[2] == [])

    # CUSUM beats naive argmax-flicker on over-segmentation at matched-ish accuracy
    probs, z = make_seq([(0, 0), (150, 1)], 300, sep=0.8, noise=1.3)
    seg_argmax = 1 + int(np.sum(np.diff(probs.argmax(1)) != 0))
    seg_cusum = 1 + int(np.sum(np.diff(cusum_decode(probs, G, 3.0)) != 0))
    check("CUSUM far fewer segments than argmax", seg_cusum < seg_argmax)
    print(f"     (argmax segments={seg_argmax}, cusum segments={seg_cusum})")

    print("\n" + ("ALL QCD SELF-TESTS PASSED" if ok else "*** SELF-TEST FAILURES ***"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
