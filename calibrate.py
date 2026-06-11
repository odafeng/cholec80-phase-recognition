"""Calibration for online surgical phase recognition (BUA component).

Three uncertainty sources, compared head-to-head in the paper (RESEARCH_PLAN.md):
  1. single model              (raw softmax)
  2. temperature scaling       (Guo et al. 2017) -- a single global scalar T,
                               fit on the VAL set (videos 33-40). Global, not
                               per-phase, because val is only 8 videos.
  3. 5-seed deep ensemble      (average softmax of the per-seed models)

CLI fits T on val logits and writes a sidecar `<ckpt>.temp.json` consumed at
eval time (logits are divided by T before softmax).

Pure-torch temperature fitting (`fit_temperature`) is CPU-unit-testable; the
model/feature-dependent collectors are exercised in Phase 2/3 on the GB10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from phases import NUM_PHASES
from splits import VAL_IDS


# --------------------------------------------------------------------------- #
# temperature fitting (Guo et al. 2017)
# --------------------------------------------------------------------------- #
def fit_temperature(logits, labels, max_iter=100, init=1.5):
    """Fit a single scalar temperature T>0 minimising NLL of softmax(logits / T).

    logits: (N, C) tensor/array. labels: (N,) int. Returns float T.
    """
    logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_T = torch.nn.Parameter(torch.log(torch.tensor(float(init))))
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T).detach())


def fit_per_phase_temperature(logits, labels, num_classes=NUM_PHASES, **kw):
    """ABLATION ONLY. One temperature per (true) class. Unstable on rare short
    phases given the 8-video val set -- reported as an ablation, not the default.
    Falls back to the global T for classes with too few samples.
    """
    labels = np.asarray(labels)
    global_T = fit_temperature(logits, labels, **kw)
    Ts = np.full(num_classes, global_T, dtype=float)
    logits = np.asarray(logits)
    for c in range(num_classes):
        sel = labels == c
        if sel.sum() >= 50:                      # need enough frames to be stable
            Ts[c] = fit_temperature(logits[sel], labels[sel], **kw)
    return Ts


def apply_temperature(logits, T):
    """Divide logits by scalar or per-class-vector T (returns same type as input)."""
    arr = np.asarray(logits, dtype=float)
    if np.isscalar(T) or np.ndim(T) == 0:
        return arr / float(T)
    return arr / np.asarray(T)[None, :]          # per-phase vector over columns


# --------------------------------------------------------------------------- #
# collectors (model + features -> logits)   [exercised on GB10 in Phase 2/3]
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_logits(ckpt_path, feat_dir, ids=VAL_IDS, device="cpu"):
    """Stack last-stage logits + labels over the given video ids."""
    from evaluate import build_from_ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_from_ckpt(ckpt, device)
    L, Y = [], []
    for i in ids:
        p = Path(feat_dir) / f"video{i:02d}.pt"
        if not p.exists():
            continue
        d = torch.load(p)
        feats = d["feats"].t().unsqueeze(0).to(device)       # (1, C, T)
        logits = model(feats)[-1].squeeze(0).t().cpu().numpy()  # (T, C)
        L.append(logits)
        Y.append(d["labels"].numpy())
    return np.concatenate(L), np.concatenate(Y)


@torch.no_grad()
def ensemble_probs(ckpt_paths, feat_dirs, ids, device="cpu"):
    """Average softmax across seed checkpoints. ckpt_paths and feat_dirs are lists
    (feat_dirs may be length 1 -> shared). Returns {id: (probs (T,C), labels)}."""
    from evaluate import build_from_ckpt
    if len(feat_dirs) == 1:
        feat_dirs = feat_dirs * len(ckpt_paths)
    acc = {}
    for ckpt_path, feat_dir in zip(ckpt_paths, feat_dirs):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_from_ckpt(ckpt, device)
        for i in ids:
            p = Path(feat_dir) / f"video{i:02d}.pt"
            if not p.exists():
                continue
            d = torch.load(p)
            feats = d["feats"].t().unsqueeze(0).to(device)
            probs = F.softmax(model(feats)[-1], dim=1).squeeze(0).t().cpu().numpy()
            if i not in acc:
                acc[i] = [np.zeros_like(probs), 0, d["labels"].numpy()]
            n = min(acc[i][0].shape[0], probs.shape[0])
            acc[i][0] = acc[i][0][:n] + probs[:n]
            acc[i][1] += 1
    return {i: (v[0] / v[1], v[2][:v[0].shape[0]]) for i, v in acc.items()}


# --------------------------------------------------------------------------- #
# CLI: fit + save sidecar
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--per_phase", action="store_true", help="also fit per-phase T (ablation)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logits, labels = collect_logits(args.ckpt, args.features, VAL_IDS, device)
    T = fit_temperature(logits, labels)
    out = {"temperature": T, "n_val_frames": int(len(labels))}
    if args.per_phase:
        out["per_phase_temperature"] = fit_per_phase_temperature(
            logits, labels, NUM_PHASES).tolist()
    sidecar = args.ckpt.with_suffix(args.ckpt.suffix + ".temp.json")
    sidecar.write_text(json.dumps(out, indent=2))
    print(f"fit T={T:.3f} on {len(labels)} val frames -> {sidecar}")


# --------------------------------------------------------------------------- #
# self-test (CPU, synthetic) -- plan verification: over-confident -> T>1, ECE down
# --------------------------------------------------------------------------- #
def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    from metrics import _ece_from_conf
    rng = np.random.default_rng(0)
    N, C = 6000, NUM_PHASES
    # A genuinely calibrated model at logits z: sample labels FROM softmax(z).
    z = rng.normal(0, 1.3, (N, C))
    p_true = torch.softmax(torch.as_tensor(z), 1).numpy()
    cum = np.cumsum(p_true, 1)
    u = rng.random((N, 1))
    labels = (u < cum).argmax(1)                   # label ~ p_true  => z is calibrated
    s = 3.0
    sharp = z * s                                  # over-confident by known factor s

    def ece_of(lg):
        p = torch.softmax(torch.as_tensor(lg), 1).numpy()
        conf = p.max(1); correct = (p.argmax(1) == labels)
        return _ece_from_conf(conf, correct, bins=15)[0]

    ece_before = ece_of(sharp)
    T = fit_temperature(sharp, labels)
    ece_after = ece_of(apply_temperature(sharp, T))
    print(f"    fitted T={T:.3f} (true sharpen s={s})  ECE {ece_before:.3f} -> {ece_after:.3f}")
    check("fitted T recovers sharpen factor (~s)", abs(T - s) < 0.6)
    check("fitted T > 1 (cools over-confidence)", T > 1.0)
    check("ECE reduced by temperature scaling", ece_after < ece_before)

    # apply_temperature scalar vs vector shapes
    v = apply_temperature(sharp, np.ones(C) * 2.0)
    check("per-phase apply shape ok", v.shape == sharp.shape)

    print("\n" + ("ALL CALIBRATE SELF-TESTS PASSED" if ok else "*** SELF-TEST FAILURES ***"))
    return ok


if __name__ == "__main__":
    import sys
    if len(__import__("sys").argv) > 1:
        main()
    else:
        sys.exit(0 if _selftest() else 1)
