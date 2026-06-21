"""Cross-dataset QCD frontier with per-video bootstrap CIs on Cataract-101 (#14, P4).

Paper-grade version of cataract_qcd.py: trains 5 seeds of a causal TeCNO on standardized
Cataract-101 features (60/20/21 split, 10 phases), then for each seed selects the operating
point at over-seg <= 2.0 on each decoder's frontier and records PER-VIDEO metrics. Stacking
over the 5 seeds gives (seeds, videos) matrices; cluster-bootstrap over the 21 test videos:
  * each decoder's operating-point metric mean [95% CI]
  * QCD-vs-heuristic paired deltas (latency / over-seg / acc / seg-F1), Holm-corrected
  * the raw-argmax failure modes (over-seg, boundary-ECE) with CIs — same modes as Cholec80?

Reuses qcd / frontier_ci / stats_corrected helpers. Run:
  env -u LD_LIBRARY_PATH python cataract_frontier_ci.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mstcn import MultiStageTCN
from train_tcn import mstcn_loss, set_seed
from cataract_dataset import get_splits, NUM_PHASES_CAT
from qcd import estimate_graph, cusum_decode, fully_connected_graph
from smooth import online_uncertainty_smooth, causal_mode_filter
from metrics import accuracy, over_segmentation, transition_latency, segmental_f1, calibration
from qcd_p2 import H_GRID, WINDOWS, GAMMAS, OSEG_TARGET
from frontier_ci import ev_pv, select_op, METRICS
from stats_corrected import cluster_bootstrap_mean, paired, holm

device = "cuda" if torch.cuda.is_available() else "cpu"
FEAT = "features_cataract"
SEEDS = [0, 1, 2, 3, 4]


def load(vids, mu=None, sd=None):
    data = []
    for v in vids:
        d = torch.load(f"{FEAT}/video{v}.pt")
        f = d["feats"].numpy()
        if mu is not None:
            f = (f - mu) / sd
        data.append((torch.tensor(f, dtype=torch.float32).t().unsqueeze(0), d["labels"]))
    return data


def train_one(tr, va, seed):
    set_seed(seed)
    model = MultiStageTCN(2, 9, 64, in_dim=2048, num_classes=NUM_PHASES_CAT, causal=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best, best_sd = 0.0, None
    order = list(range(len(tr)))
    for _ in range(40):
        model.train(); np.random.shuffle(order)
        for j in order:
            f, y = tr[j]
            f, y = f.to(device), y.to(device)
            opt.zero_grad()
            loss = mstcn_loss(model(f), y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); c = t = 0
        with torch.no_grad():
            for f, y in va:
                p = model(f.to(device))[-1].argmax(1).squeeze(0).cpu()
                c += (p == y).sum().item(); t += y.numel()
        if c / t > best:
            best = c / t; best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd)
    return model, best


@torch.no_grad()
def posteriors(model, te):
    model.eval()
    out = []
    for f, y in te:
        logits = model(f.to(device))[-1].squeeze(0).t().cpu().numpy()
        out.append((torch.softmax(torch.as_tensor(logits), 1).numpy(), y.numpy()))
    return out


def argmax_failure_pv(seqs):
    """Per-video raw-argmax over-seg + boundary-ECE (the failure modes)."""
    o, b = [], []
    for p, gt in seqs:
        pred = p.argmax(1)
        n = min(len(gt), len(pred))
        o.append(over_segmentation(pred[:n], gt[:n])["ratio"])
        b.append(calibration(p[:n], gt[:n])["boundary_ece"] * 100)
    return np.asarray(o, float), np.asarray(b, float)


def main():
    tr_ids, va_ids, te_ids = get_splits()
    trraw = np.concatenate([torch.load(f"{FEAT}/video{v}.pt")["feats"].numpy() for v in tr_ids])
    mu, sd = trraw.mean(0), trraw.std(0) + 1e-6
    tr, va, te = load(tr_ids, mu, sd), load(va_ids, mu, sd), load(te_ids, mu, sd)
    print(f"Cataract-101: train/val/test {len(tr)}/{len(va)}/{len(te)}, {NUM_PHASES_CAT} phases, "
          f"{len(SEEDS)} seeds")

    G = estimate_graph([y.numpy() for _, y in tr], num_phases=NUM_PHASES_CAT)
    Gf = fully_connected_graph(NUM_PHASES_CAT)
    dec = {"qcd_est": [], "qcd_full": [], "heuristic": []}
    oseg_raw, bece_raw = [], []

    for s in SEEDS:
        model, vacc = train_one(tr, va, s)
        seqs = posteriors(model, te)
        o, b = argmax_failure_pv(seqs)
        oseg_raw.append(o); bece_raw.append(b)
        qe = select_op([ev_pv(lambda p, h=h: cusum_decode(p, G, h), seqs) for h in H_GRID])
        qf = select_op([ev_pv(lambda p, h=h: cusum_decode(p, Gf, h), seqs) for h in H_GRID])
        hu_front = [ev_pv(lambda p, w=w: causal_mode_filter(p.argmax(1), w), seqs) for w in WINDOWS]
        hu_front += [ev_pv(lambda p, g=g: online_uncertainty_smooth(p, gamma=g).argmax(1), seqs)
                     for g in GAMMAS]
        hu = select_op(hu_front)
        dec["qcd_est"].append(qe); dec["qcd_full"].append(qf); dec["heuristic"].append(hu)
        print(f"  seed {s}: val_acc {vacc:.3f}  qcd_est lat={np.nanmean(qe['lat']):.1f}/"
              f"oseg={np.nanmean(qe['oseg']):.2f}  heur lat={np.nanmean(hu['lat']):.1f}/"
              f"oseg={np.nanmean(hu['oseg']):.2f}")

    mats = {d: {m: np.vstack([pv[m] for pv in rows]) for m in METRICS} for d, rows in dec.items()}
    res = {"oseg_target": OSEG_TARGET, "n_seeds": len(SEEDS), "n_videos": len(te),
           "operating_point": {}, "dominance": {}, "failure_modes": {}}

    # raw-argmax failure modes (do they appear on cataract too?)
    om, ol, oh = cluster_bootstrap_mean(np.vstack(oseg_raw))
    bm, bl, bh = cluster_bootstrap_mean(np.vstack(bece_raw))
    res["failure_modes"] = {"argmax_over_seg": {"mean": om, "ci": [ol, oh]},
                            "argmax_boundary_ece_pct": {"mean": bm, "ci": [bl, bh]}}
    print(f"\n--- raw-argmax failure modes (cluster-bootstrap over {len(te)} videos) ---")
    print(f"  over-seg ratio   {om:.2f} [{ol:.2f}, {oh:.2f}]   (vs ~1 ideal)")
    print(f"  boundary-ECE %   {bm:.2f} [{bl:.2f}, {bh:.2f}]   (hidden behind frame accuracy)")

    print(f"\n--- operating point  mean [95% CI] ---\n{'decoder':12s}"
          + "".join(f"{m:>20s}" for m in METRICS))
    for d, mm in mats.items():
        row = {}; txt = ""
        for m in METRICS:
            pt, lo, hi = cluster_bootstrap_mean(mm[m])
            row[m] = {"mean": pt, "ci": [lo, hi]}; txt += f"{pt:6.2f}[{lo:5.1f},{hi:5.1f}]"
        res["operating_point"][d] = row
        print(f"{d:12s}{txt}")

    print("\n--- dominance vs heuristic (Holm; +Δ = QCD better) ---")
    rows = []
    for d in ("qcd_est", "qcd_full"):
        for m in ("lat", "acc", "f1", "oseg"):
            name = {"lat": "median_latency", "oseg": "over_seg_ratio"}.get(m, m)
            r = paired(mats[d][m], mats["heuristic"][m], name)
            r["_decoder"], r["_metric"] = d, m
            rows.append(r)
    for r, padj in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = padj
        sig = "*" if padj < 0.05 else " "
        res["dominance"].setdefault(r["_decoder"], {})[r["_metric"]] = r
        print(f"  {r['_decoder']:9s} {r['_metric']:5s}  Δ={r['mean_delta']:+7.2f}  "
              f"[{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}]  p_holm={padj:.4f} {sig}")

    Path("results").mkdir(exist_ok=True)
    Path("results/cataract_frontier_ci.json").write_text(json.dumps(res, indent=2, default=float))
    print("\n-> results/cataract_frontier_ci.json")
    print("CAT_FRONTIER_CI_EXIT=0")


if __name__ == "__main__":
    main()
