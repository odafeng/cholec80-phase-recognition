"""P3 experiment: does the LEARNED sequential detector dominate fixed CUSUM and
heuristic post-processing on the over-seg / latency / ACCURACY operating
characteristic? Train detectors at several lambda on TRAIN posteriors (select on
VAL), evaluate on TEST, and compare to argmax / mode-filter / EMA / CUSUM at
matched over-segmentation. The key question: can the learned detector get clean
segmentation AND keep accuracy (which fixed-graph CUSUM sacrifices)?

Run:  python run_p3.py            # default representative cells, GPU
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from learned_detector import get_posteriors, train_detector, eval_detector
from qcd import estimate_graph, cusum_decode
from smooth import online_uncertainty_smooth, causal_mode_filter
from metrics import accuracy, over_segmentation, transition_latency, segmental_f1
from splits import TRAIN_IDS, VAL_IDS, TEST_IDS

LAMBDAS = [0.2, 0.5, 1.0, 2.0, 4.0]
CELLS = [("rn50", "features", "tecno"),
         ("e2e", "features_e2e", "lovit_causal"),
         ("endovitft", "features_endovit_ft", "asformer"),
         ("endovit", "features_endovit", "tecno")]


def ev(pred_fn, post):
    o, l, a, f = [], [], [], []
    for probs, labels in post:
        pred = pred_fn(probs.numpy())
        g = labels.numpy(); n = min(len(g), len(pred)); pred, g = pred[:n], g[:n]
        o.append(over_segmentation(pred, g)["ratio"]); l.append(transition_latency(pred, g)["median_latency"])
        a.append(accuracy(pred, g)); f.append(segmental_f1(pred, g, 0.10)[0])
    return dict(oseg=np.nanmean(o), lat=np.nanmean(l), acc=np.nanmean(a) * 100, f1=np.nanmean(f))


def at_oseg(points, thr=2.0):
    el = [p for p in points if p["oseg"] <= thr]
    return min(el, key=lambda p: p["lat"]) if el else min(points, key=lambda p: p["oseg"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("results/p3.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph = estimate_graph([torch.load(f"features/video{i:02d}.pt")["labels"].numpy()
                            for i in TRAIN_IDS])
    res = {}
    for bb, fd, head in CELLS:
        ck = f"checkpoints/rel_{bb}/{head}_base_s0.pt"
        if not Path(ck).exists():
            print(f"skip {bb}/{head} (no ckpt)"); continue
        tr = get_posteriors(ck, fd, TRAIN_IDS, device)
        va = get_posteriors(ck, fd, VAL_IDS, device)
        te = get_posteriors(ck, fd, TEST_IDS, device)
        print(f"\n===== {bb}/{head} =====")
        print(f"{'decoder':22s}{'over-seg':>9}{'lat':>7}{'acc%':>7}{'F1':>7}")
        argm = ev(lambda p: p.argmax(1), te)
        print(f"{'argmax':22s}{argm['oseg']:>9.2f}{argm['lat']:>7.1f}{argm['acc']:>7.1f}{argm['f1']:>7.1f}")

        heur = [ev(lambda p, w=w: causal_mode_filter(p.argmax(1), w), te) for w in (3,5,9,15,25)]
        heur += [ev(lambda p, g=g: online_uncertainty_smooth(p, gamma=g).argmax(1), te) for g in (0.2,0.3,0.5,0.7)]
        hh = at_oseg(heur)
        print(f"{'heuristic @oseg<=2':22s}{hh['oseg']:>9.2f}{hh['lat']:>7.1f}{hh['acc']:>7.1f}{hh['f1']:>7.1f}")

        cus = [ev(lambda p, h=h: cusum_decode(p, graph, h), te) for h in (3,5,8,12,18,26)]
        cc = at_oseg(cus)
        print(f"{'CUSUM @oseg<=2':22s}{cc['oseg']:>9.2f}{cc['lat']:>7.1f}{cc['acc']:>7.1f}{cc['f1']:>7.1f}")

        learned = []
        for lam in LAMBDAS:
            net = train_detector(tr, va, lam, device, epochs=args.epochs)
            m = eval_detector(net, te, device)
            learned.append({**m, "lam": lam})
            print(f"{'learned lam='+str(lam):22s}{m['oseg']:>9.2f}{m['lat']:>7.1f}{m['acc']:>7.1f}{m['f1']:>7.1f}")
        lh = at_oseg(learned)
        # verdict at matched over-seg<=2: learned vs best of {heuristic, cusum}
        best_other_lat = min(hh["lat"], cc["lat"])
        best_other_acc = max(hh["acc"], cc["acc"])
        print(f"  >> @oseg<=2  learned(acc {lh['acc']:.1f}, lat {lh['lat']:.1f})  "
              f"vs best-other(acc {best_other_acc:.1f}, lat {best_other_lat:.1f})")
        res[f"{bb}/{head}"] = {"argmax": argm, "heuristic": hh, "cusum": cc, "learned": learned}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {args.out}")
    print("P3_EXIT=0")


if __name__ == "__main__":
    main()
