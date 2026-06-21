"""QCD P2 — systematic study over the full grid (CPU, reuses v1 posteriors):
  (1) domination + accuracy cost: QCD(estimated graph) vs best heuristic at matched
      over-seg, per (backbone, head), averaged over seeds.
  (2) graph ablation: estimated vs fully-connected graph (the accuracy<->segmentation
      knob).
  (3) calibration coupling (claim 3): single-no-temp vs single-temp vs 5-seed
      ensemble posteriors -> posterior ECE and QCD performance.

Writes results/qcd_p2.json and prints tables.
"""
import json
from pathlib import Path
import numpy as np
import torch

from evaluate import build_from_ckpt, _load_temperature
from train_tcn import load_features
from smooth import online_uncertainty_smooth, causal_mode_filter
from qcd import estimate_graph, cusum_decode, fully_connected_graph
from metrics import accuracy, over_segmentation, transition_latency, segmental_f1, calibration
from splits import TRAIN_IDS, TEST_IDS

FEAT = {"rn50": "features", "endovit": "features_endovit",
        "endovitft": "features_endovit_ft", "e2e": "features_e2e"}
HEADS = ["tecno", "lovit_causal", "asformer", "transsvnet", "surgformer"]
SEEDS = [0, 1, 2, 3, 4]
H_GRID = [2, 3, 5, 8, 12, 18, 26]
WINDOWS = [3, 5, 9, 15, 25, 41]
GAMMAS = [0.2, 0.3, 0.5, 0.7, 1.0]
OSEG_TARGET = 2.0


def graph_from_train():
    seqs = [torch.load(Path("features") / f"video{i:02d}.pt")["labels"].numpy()
            for i in TRAIN_IDS if (Path("features") / f"video{i:02d}.pt").exists()]
    return estimate_graph(seqs)


@torch.no_grad()
def posteriors(ckpt, feat_dir, apply_temp=True):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_from_ckpt(ck, "cpu")
    T = (_load_temperature("auto", ckpt) or 1.0) if apply_temp else 1.0
    ids = [i for i in TEST_IDS if (Path(feat_dir) / f"video{i:02d}.pt").exists()]
    out = []
    for feats, labels in load_features(Path(feat_dir), ids):
        logits = model(feats)[-1].squeeze(0).t().numpy() / T
        out.append((torch.softmax(torch.as_tensor(logits), 1).numpy(), labels.numpy()))
    return out


def ev(decode, seqs):
    o, l, a, f = [], [], [], []
    for p, gt in seqs:
        pred = decode(p)
        n = min(len(gt), len(pred)); pred, g = pred[:n], gt[:n]
        o.append(over_segmentation(pred, g)["ratio"]); l.append(transition_latency(pred, g)["median_latency"])
        a.append(accuracy(pred, g)); f.append(segmental_f1(pred, g, 0.10)[0])
    return dict(oseg=np.nanmean(o), lat=np.nanmean(l), acc=np.nanmean(a) * 100, f1=np.nanmean(f))


def op_at_oseg(points, target):
    """from list of metric-dicts, pick min-latency with oseg<=target."""
    elig = [p for p in points if p["oseg"] <= target]
    return min(elig, key=lambda p: p["lat"]) if elig else min(points, key=lambda p: p["oseg"])


def cusum_frontier(seqs, graph):
    return [ev(lambda p, h=h: cusum_decode(p, graph, h), seqs) for h in H_GRID]


def heur_frontier(seqs):
    pts = [ev(lambda p, w=w: causal_mode_filter(p.argmax(1), w), seqs) for w in WINDOWS]
    pts += [ev(lambda p, g=g: online_uncertainty_smooth(p, gamma=g).argmax(1), seqs) for g in GAMMAS]
    return pts


def main():
    G = graph_from_train()
    Gf = fully_connected_graph()
    res = {"graph": {str(k): v for k, v in G.items()}, "grid": {}, "calib": {}}

    print("=== (1)+(2) domination + graph ablation (per backbone/head, mean over seeds) ===")
    print(f"{'cell':22s}{'QCD-G oseg/lat/acc/F1':>26}{'QCDfull acc/oseg':>18}{'heur lat@oseg<=2':>18}")
    for bb in FEAT:
        for head in HEADS:
            qg, qf, hu = [], [], []
            for s in SEEDS:
                ck = Path(f"checkpoints/rel_{bb}/{head}_base_s{s}.pt")
                if not ck.exists():
                    continue
                seqs = posteriors(ck, FEAT[bb])
                qg.append(op_at_oseg(cusum_frontier(seqs, G), OSEG_TARGET))
                qf.append(op_at_oseg(cusum_frontier(seqs, Gf), OSEG_TARGET))
                hu.append(op_at_oseg(heur_frontier(seqs), OSEG_TARGET))
            if not qg:
                continue
            def m(rows, k): return float(np.mean([r[k] for r in rows]))
            cell = f"{bb}/{head}"
            res["grid"][cell] = {
                "qcd_est": {k: m(qg, k) for k in ("oseg", "lat", "acc", "f1")},
                "qcd_full": {k: m(qf, k) for k in ("oseg", "lat", "acc", "f1")},
                "heuristic": {k: m(hu, k) for k in ("oseg", "lat", "acc", "f1")}}
            print(f"{cell:22s}"
                  f"{m(qg,'oseg'):6.2f}/{m(qg,'lat'):4.1f}/{m(qg,'acc'):4.1f}/{m(qg,'f1'):4.1f}   "
                  f"{m(qf,'acc'):5.1f}/{m(qf,'oseg'):4.2f}      "
                  f"{m(hu,'lat'):5.1f}  (F1 {m(hu,'f1'):4.1f})")

    print("\n=== (3) calibration coupling: posterior ECE & QCD (rn50/e2e x tecno/lovit) ===")
    print(f"{'cell':28s}{'variant':12s}{'ECE%':>7}{'QCD acc':>9}{'oseg':>7}{'F1':>7}")
    for bb in ["rn50", "e2e"]:
        for head in ["tecno", "lovit_causal"]:
            cks = [Path(f"checkpoints/rel_{bb}/{head}_base_s{s}.pt") for s in SEEDS]
            cks = [c for c in cks if c.exists()]
            if not cks:
                continue
            variants = {}
            variants["no-temp"] = posteriors(cks[0], FEAT[bb], apply_temp=False)
            variants["temp"] = posteriors(cks[0], FEAT[bb], apply_temp=True)
            # 5-seed ensemble (avg raw softmax)
            per = [posteriors(c, FEAT[bb], apply_temp=False) for c in cks]
            ens = []
            for vi in range(len(per[0])):
                n = min(per[k][vi][0].shape[0] for k in range(len(per)))
                avg = sum(per[k][vi][0][:n] for k in range(len(per))) / len(per)
                ens.append((avg, per[0][vi][1][:n]))
            variants["ensemble"] = ens
            for vname, seqs in variants.items():
                ece = float(np.nanmean([calibration(p, gt)["ece"] for p, gt in seqs])) * 100
                q = op_at_oseg(cusum_frontier(seqs, G), OSEG_TARGET)
                res["calib"].setdefault(f"{bb}/{head}", {})[vname] = {"ece": ece, **{k: q[k] for k in ("acc", "oseg", "f1")}}
                print(f"{bb+'/'+head:28s}{vname:12s}{ece:7.2f}{q['acc']:9.1f}{q['oseg']:7.2f}{q['f1']:7.1f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/qcd_p2.json").write_text(json.dumps(res, indent=2, default=float))
    print("\nwrote results/qcd_p2.json")
    print("QCD_P2_EXIT=0")


if __name__ == "__main__":
    main()
