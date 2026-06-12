"""P1 GO/NO-GO: does the QCD (CUSUM) decoder dominate heuristic post-processing on
the over-seg / latency / accuracy operating characteristic? Reuses v1's trained
posteriors (no retraining). Transition graph estimated from TRAIN labels only.
CPU. Run:
  python run_qcd.py --ckpt checkpoints/rel_rn50/tecno_base_s0.pt --features features
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from evaluate import build_from_ckpt, _load_temperature
from train_tcn import load_features
from smooth import online_uncertainty_smooth, causal_mode_filter
from qcd import estimate_graph, cusum_decode
from metrics import accuracy, over_segmentation, transition_latency, segmental_f1
from splits import TRAIN_IDS, TEST_IDS

H_GRID = [2, 3, 5, 8, 12, 18, 26]
WINDOWS = [3, 5, 9, 15, 25, 41]
GAMMAS = [0.2, 0.3, 0.5, 0.7, 1.0]


def get_graph(feat_dir="features"):
    seqs = []
    for i in TRAIN_IDS:
        p = Path(feat_dir) / f"video{i:02d}.pt"
        if p.exists():
            seqs.append(torch.load(p)["labels"].numpy())
    return estimate_graph(seqs)


@torch.no_grad()
def get_post(ckpt, feat_dir):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_from_ckpt(ck, "cpu")
    T = _load_temperature("auto", ckpt) or 1.0
    ids = [i for i in TEST_IDS if (Path(feat_dir) / f"video{i:02d}.pt").exists()]
    out = []
    for feats, labels in load_features(Path(feat_dir), ids):
        logits = model(feats)[-1].squeeze(0).t().numpy() / T
        p = torch.softmax(torch.as_tensor(logits), 1).numpy()
        out.append((p, labels.numpy()))
    return out


def evalseq(decode_fn, seqs):
    o, l, a, f = [], [], [], []
    for p, gt in seqs:
        pred = decode_fn(p)
        n = min(len(gt), len(pred)); pred, g = pred[:n], gt[:n]
        o.append(over_segmentation(pred, g)["ratio"])
        l.append(transition_latency(pred, g)["median_latency"])
        a.append(accuracy(pred, g))
        f.append(segmental_f1(pred, g, 0.10)[0])
    return np.nanmean(o), np.nanmean(l), np.nanmean(a) * 100, np.nanmean(f)


def pareto(pts):  # (over-seg, latency); keep lower-left
    s = sorted(pts, key=lambda p: (p[0], p[1])); out, best = [], 1e9
    for p in s:
        if p[1] < best:
            out.append(p); best = p[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    args = ap.parse_args()
    graph = get_graph()
    seqs = get_post(args.ckpt, args.features)
    print(f"# {args.ckpt.name} on {args.features}  (graph from TRAIN; test set)")
    print(f"  transition graph: {{k:succ}} = "
          + ", ".join(f"{k}->{v}" for k, v in graph.items() if v))

    o, l, a, f = evalseq(lambda p: p.argmax(1), seqs)
    print(f"\n{'decoder':22s}{'over-seg':>9}{'lat(s)':>8}{'acc%':>7}{'F1@10':>7}")
    print(f"{'argmax (raw)':22s}{o:>9.2f}{l:>8.1f}{a:>7.1f}{f:>7.1f}")

    print("-- heuristics (frontier) --")
    hpts = []
    for w in WINDOWS:
        o, l, a, f = evalseq(lambda p, w=w: causal_mode_filter(p.argmax(1), w), seqs)
        hpts.append((o, l, a, f, f"mode w={w}"))
    for g in GAMMAS:
        o, l, a, f = evalseq(lambda p, g=g: online_uncertainty_smooth(p, gamma=g).argmax(1), seqs)
        hpts.append((o, l, a, f, f"ema g={g}"))
    for o, l, *rest in pareto([(o, l, a, f, t) for o, l, a, f, t in hpts]):
        a, f, t = rest
        print(f"  {t:20s}{o:>9.2f}{l:>8.1f}{a:>7.1f}{f:>7.1f}")

    print("-- QCD CUSUM (frontier, OURS) --")
    qpts = []
    for h in H_GRID:
        o, l, a, f = evalseq(lambda p, h=h: cusum_decode(p, graph, h), seqs)
        qpts.append((o, l, a, f, f"cusum h={h}"))
    for o, l, *rest in pareto([(o, l, a, f, t) for o, l, a, f, t in qpts]):
        a, f, t = rest
        print(f"  {t:20s}{o:>9.2f}{l:>8.1f}{a:>7.1f}{f:>7.1f}")

    # verdict: lowest latency achievable at over-seg <= threshold
    for thr in (2.0, 3.0):
        hb = min([l for o, l, *_ in hpts if o <= thr], default=float("nan"))
        qb = min([l for o, l, *_ in qpts if o <= thr], default=float("nan"))
        v = "QCD WINS" if qb < hb - 0.2 else ("heuristic wins" if hb < qb - 0.2 else "tie")
        print(f"  @over-seg<={thr}: heuristic lat={hb:.1f}  QCD lat={qb:.1f}  -> {v}")


if __name__ == "__main__":
    main()
