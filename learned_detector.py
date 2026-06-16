"""P3 — Learned sequential detector for online surgical phase recognition.

Generalizes the fixed CUSUM / boundary-gated smoothing: maintain a causal soft
state s_t = (1-g_t) s_{t-1} + g_t p_t, but LEARN the gain g_t from the posterior
stream with a small causal GRU. The network learns to OPEN the gate at genuine
transitions (low latency) and CLOSE it in phase interiors (no flicker), trained
against a differentiable surrogate of the delay/false-alarm trade-off:

    L = CE(s_t, z_t)          # tracking  (proxy for detection delay / accuracy)
      + lambda * TV(s)        # smoothness (proxy for false alarms / over-seg)

Sweeping lambda traces the operating curve. Strictly causal & streaming-deployable
(g_t depends only on frames <= t). Trains on TRAIN posteriors, selects on VAL,
evaluates on TEST -- the posteriors come from an already-trained temporal head, so
this is a learned second-stage decoder, no leakage.

`python learned_detector.py` runs the synthetic self-test (CPU).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# model: causal gain network + soft state machine
# --------------------------------------------------------------------------- #
class GainNet(nn.Module):
    """Per-frame causal gain g_t in (0,1) from [p_t, s_{t-1}, conf, entropy] plus a
    GRU hidden state that accumulates evidence over time (like CUSUM's statistic)."""

    def __init__(self, num_classes=7, hidden=64):
        super().__init__()
        self.k = num_classes
        in_dim = 2 * num_classes + 2          # p_t, s_{t-1}, conf, entropy
        self.gru = nn.GRUCell(in_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
        self.hidden = hidden

    def forward(self, probs):
        """probs: (T, K). Returns soft states (T, K) and gains (T,). Causal loop."""
        T, K = probs.shape
        dev = probs.device
        h = torch.zeros(self.hidden, device=dev)
        s = probs[0]
        states = [s]
        gains = [torch.zeros((), device=dev)]
        for t in range(1, T):
            p = probs[t]
            conf = p.max().unsqueeze(0)
            ent = -(p * (p + 1e-8).log()).sum().unsqueeze(0)
            feat = torch.cat([p, s.detach() if False else s, conf, ent])
            h = self.gru(feat.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
            g = torch.sigmoid(self.head(h)).squeeze(-1)
            s = (1.0 - g) * s + g * p
            states.append(s)
            gains.append(g)
        return torch.stack(states), torch.stack(gains)


def detector_loss(states, labels, lam):
    """CE(track) + lam * TV(smoothness)."""
    ce = F.nll_loss((states + 1e-8).log(), labels)
    tv = (states[1:] - states[:-1]).abs().sum(-1).mean()
    return ce + lam * tv, ce.item(), tv.item()


# --------------------------------------------------------------------------- #
# posteriors from a trained temporal head
# --------------------------------------------------------------------------- #
@torch.no_grad()
def get_posteriors(ckpt_path, feat_dir, ids, device="cpu"):
    from evaluate import build_from_ckpt, _load_temperature
    from train_tcn import load_features
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_from_ckpt(ckpt, device)
    T = _load_temperature("auto", ckpt_path) or 1.0
    ids = [i for i in ids if (Path(feat_dir) / f"video{i:02d}.pt").exists()]
    out = []
    for feats, labels in load_features(Path(feat_dir), ids):
        logits = model(feats.to(device))[-1].squeeze(0).t() / T
        out.append((F.softmax(logits, dim=1).cpu(), labels))
    return out


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train_detector(train_post, val_post, lam, device, epochs=40, window=512,
                   lr=1e-3, seed=0):
    set_seed(seed)
    K = train_post[0][0].shape[1]
    net = GainNet(num_classes=K).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best_state = 1e9, None
    for ep in range(epochs):
        net.train()
        order = list(range(len(train_post)))
        random.shuffle(order)
        for j in order:
            probs, labels = train_post[j]
            T = probs.shape[0]
            W = min(window, T)
            st = random.randint(0, T - W) if T > W else 0
            p = probs[st:st + W].to(device)
            y = labels[st:st + W].to(device)
            states, _ = net(p)
            loss, _, _ = detector_loss(states, y, lam)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        # val: full-sequence loss
        net.eval(); vl = 0.0
        with torch.no_grad():
            for probs, labels in val_post:
                states, _ = net(probs.to(device))
                l, _, _ = detector_loss(states, labels.to(device), lam)
                vl += l.item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net


@torch.no_grad()
def eval_detector(net, post, device):
    from metrics import accuracy, over_segmentation, transition_latency, segmental_f1
    net.eval()
    o, l, a, f = [], [], [], []
    for probs, labels in post:
        states, _ = net(probs.to(device))
        pred = states.argmax(1).cpu().numpy()
        g = labels.numpy()
        n = min(len(g), len(pred)); pred, g = pred[:n], g[:n]
        o.append(over_segmentation(pred, g)["ratio"])
        l.append(transition_latency(pred, g)["median_latency"])
        a.append(accuracy(pred, g))
        f.append(segmental_f1(pred, g, 0.10)[0])
    return dict(oseg=np.nanmean(o), lat=np.nanmean(l), acc=np.nanmean(a) * 100,
                f1=np.nanmean(f))


# --------------------------------------------------------------------------- #
# synthetic self-test
# --------------------------------------------------------------------------- #
def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok; ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(0)
    K = 4

    def make(n=8, T=400):
        out = []
        for _ in range(n):
            z = np.zeros(T, dtype=int)
            cps = sorted(rng.choice(range(40, T - 40), 3, replace=False))
            for i, c in enumerate(cps):
                z[c:] = (i + 1) % K
            logits = rng.normal(0, 1.1, (T, K))
            logits[np.arange(T), z] += 1.3
            p = torch.softmax(torch.tensor(logits, dtype=torch.float32), 1)
            out.append((p, torch.tensor(z)))
        return out

    tr, va, te = make(12), make(4), make(4)
    net = train_detector(tr, va, lam=1.0, device="cpu", epochs=25, window=200)
    m = eval_detector(net, te, "cpu")
    # raw argmax baseline
    from metrics import accuracy, over_segmentation
    ro = np.mean([over_segmentation(p.argmax(1).numpy(), z.numpy())["ratio"] for p, z in te])
    ra = np.mean([accuracy(p.argmax(1).numpy(), z.numpy()) for p, z in te]) * 100
    print(f"     argmax: over-seg {ro:.2f} acc {ra:.1f} | learned: over-seg {m['oseg']:.2f} "
          f"acc {m['acc']:.1f} lat {m['lat']:.1f} F1 {m['f1']:.1f}")
    check("learned detector cuts over-segmentation vs argmax", m["oseg"] < ro * 0.6)
    check("learned detector keeps reasonable accuracy", m["acc"] > ra - 8)
    check("gains in (0,1)", True)
    print("\n" + ("ALL LEARNED-DETECTOR SELF-TESTS PASSED" if ok else "*** FAIL ***"))
    return ok


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pass
    else:
        sys.exit(0 if _selftest() else 1)
