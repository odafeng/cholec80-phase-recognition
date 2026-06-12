"""Uncertainty-aware causal smoothing for ONLINE surgical phase recognition.

Part of the BUA module (see RESEARCH_PLAN.md). Strictly online: the estimate at
frame t depends only on frames <= t, so it is deployable in real time.

Idea
----
A plain causal EMA over probabilities smooths flicker but lags at real
transitions. We instead make the EMA gain *confidence-adaptive*: a new frame can
only pull the running estimate as hard as it is confident. Low-confidence frames
(typically the flickery, ambiguous ones that cause over-segmentation and spurious
early switches) are largely ignored; a genuinely new phase, once the model is
confident about it, is adopted quickly.

    s_t = (1 - g_t) * s_{t-1} + g_t * p_t
    g_t = gamma * clip((conf_t - conf_floor) / (1 - conf_floor), 0, 1)

where conf_t = max_c p_t[c]. With conf_t <= conf_floor the gain is 0 (frame
ignored); at conf_t = 1 the gain is gamma. gamma in (0,1] caps responsiveness.
The hard label is argmax(s_t).
"""
from __future__ import annotations

import numpy as np


def online_uncertainty_smooth(probs, gamma=0.5, conf_floor=0.4):
    """Causally smooth a (T, C) probability sequence.

    Args:
        probs: (T, C) per-frame probabilities (softmaxed / calibrated).
        gamma: max EMA gain in (0, 1]. Higher = more responsive, less smoothing.
        conf_floor: confidence below which a frame cannot move the estimate.
    Returns:
        smoothed: (T, C) smoothed probabilities (rows sum to 1).
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probs must be (T, C)")
    T, C = probs.shape
    if T == 0:
        return probs.copy()
    denom = max(1e-8, 1.0 - conf_floor)
    out = np.empty_like(probs)
    s = probs[0].copy()
    out[0] = s
    for t in range(1, T):
        conf = probs[t].max()
        g = gamma * np.clip((conf - conf_floor) / denom, 0.0, 1.0)
        s = (1.0 - g) * s + g * probs[t]
        out[t] = s
    # renormalise defensively (convex combinations of prob vectors stay normalised,
    # but guard against tiny fp drift)
    out = np.clip(out, 0.0, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def smooth_labels(probs, gamma=0.5, conf_floor=0.4):
    """Convenience: return hard argmax labels of the smoothed sequence."""
    return online_uncertainty_smooth(probs, gamma, conf_floor).argmax(axis=1)


def causal_mode_filter(labels, window=15):
    """Standard online post-processor: causal sliding-window MODE (majority vote
    over the last `window` frames). This is the dumb, established baseline a
    reviewer demands we beat -- if our confidence smoothing can't dominate this on
    the over-seg/latency frontier, then the 'smoothing' is not a contribution.
    Strictly causal (only past frames). Returns hard labels (T,)."""
    labels = np.asarray(labels)
    T = labels.shape[0]
    out = np.empty(T, dtype=labels.dtype)
    for t in range(T):
        lo = max(0, t - window + 1)
        vals, cnt = np.unique(labels[lo:t + 1], return_counts=True)
        out[t] = vals[cnt.argmax()]
    return out


def boundary_gated_smooth(probs, bprob, gamma=0.5, conf_floor=0.4, beta=1.0):
    """Boundary-GATED causal smoothing -- gives the boundary head a real inference
    job (the BUA component used at test time, not just an aux training loss).

    Key idea: a frame should be allowed to move the running estimate if it is
    EITHER confident OR at a predicted phase boundary. So we OPEN the gate with the
    max of the confidence gate and the predicted boundary probability:
        g_t = gamma * max( conf_gate_t , beta * b_t )
    - INTERIOR flicker = momentary low confidence with LOW boundary prob -> both
      terms small -> heavy smoothing -> over-segmentation suppressed.
    - REAL transition = low confidence but HIGH boundary prob -> gate opens via b_t
      -> the model switches phase PROMPTLY despite low confidence -> low latency.
    This separates "uncertain because flickering" from "uncertain because the phase
    is actually changing" -- something plain confidence smoothing cannot do, so it
    can cut over-segmentation AND latency together. Strictly causal (b_t from the
    causal boundary head; s_t depends only on frames <= t).

    Args:
        probs: (T, C) calibrated per-frame probabilities.
        bprob: (T,) predicted boundary probability (sigmoid of the boundary head).
        gamma, conf_floor: as in online_uncertainty_smooth.
        beta: how strongly a predicted boundary opens the gate (1.0 = fully).
    """
    probs = np.asarray(probs, dtype=float)
    bprob = np.asarray(bprob, dtype=float).reshape(-1)
    T, C = probs.shape
    if T == 0:
        return probs.copy()
    denom = max(1e-8, 1.0 - conf_floor)
    out = np.empty_like(probs)
    s = probs[0].copy()
    out[0] = s
    for t in range(1, T):
        conf = probs[t].max()
        conf_gate = np.clip((conf - conf_floor) / denom, 0.0, 1.0)
        gate = max(conf_gate, beta * np.clip(bprob[t], 0.0, 1.0))
        g = gamma * min(gate, 1.0)
        s = (1.0 - g) * s + g * probs[t]
        out[t] = s
    out = np.clip(out, 0.0, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _onehot(labels, C, conf=0.95):
    labels = np.asarray(labels)
    p = np.full((len(labels), C), (1 - conf) / (C - 1))
    p[np.arange(len(labels)), labels] = conf
    return p


def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    C = 3
    # constant confident sequence must be unchanged in argmax
    const = _onehot([1] * 30, C, conf=0.95)
    out = smooth_labels(const, gamma=0.5, conf_floor=0.4)
    check("constant phase unchanged", np.all(out == 1))

    # single-frame flicker removed
    lab = np.array([1] * 15 + [2] + [1] * 14)      # one-frame spurious '2'
    probs = _onehot(lab, C, conf=0.92)
    # make the flicker frame LOW confidence (ambiguous) -> should be ignored
    probs[15] = np.array([0.30, 0.36, 0.34])
    out = smooth_labels(probs, gamma=0.5, conf_floor=0.5)
    check("low-confidence 1-frame flicker removed", np.all(out == 1))

    # a confident, sustained new phase IS adopted (just possibly with small lag)
    lab2 = np.array([0] * 20 + [1] * 20)
    probs2 = _onehot(lab2, C, conf=0.95)
    out2 = smooth_labels(probs2, gamma=0.6, conf_floor=0.4)
    # must reach phase 1 and end in phase 1
    check("sustained confident transition adopted", out2[-1] == 1 and np.any(out2 == 1))
    # lag is small (<= 5 frames)
    first1 = np.argmax(out2 == 1)
    check("transition lag small (<=5)", 0 < first1 <= 25)

    # output rows are valid distributions
    sm = online_uncertainty_smooth(probs2, 0.6, 0.4)
    check("rows sum to 1", np.allclose(sm.sum(1), 1.0))

    # gain=1, floor=0, PURE one-hot (conf=1) reduces to identity (full overwrite)
    pure = _onehot(lab2, C, conf=1.0)
    sm2 = online_uncertainty_smooth(pure, gamma=1.0, conf_floor=0.0)
    check("gamma=1,floor=0,conf=1 -> identity", np.allclose(sm2, pure))

    print("\n" + ("ALL SMOOTH SELF-TESTS PASSED" if ok else "*** SELF-TEST FAILURES ***"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
