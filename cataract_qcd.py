"""Cross-procedure result: train a causal TeCNO (10 phases) on standardized
Cataract-101 features, then test whether (a) the SAME failure modes appear
(high over-segmentation, boundary mis-calibration invisible to accuracy) and
(b) the QCD decoder generalizes (dominates heuristics on the over-seg/latency
frontier) on a procedure utterly different from Cholec80.
"""
import numpy as np
import torch
import torch.nn as nn

from mstcn import MultiStageTCN
from train_tcn import mstcn_loss, set_seed
from cataract_dataset import get_splits, NUM_PHASES_CAT
from qcd import estimate_graph, cusum_decode
from smooth import online_uncertainty_smooth, causal_mode_filter
from metrics import accuracy, over_segmentation, transition_latency, segmental_f1, calibration

device = "cuda" if torch.cuda.is_available() else "cpu"
FEAT = "features_cataract"


def load(vids, mu=None, sd=None):
    data = []
    for v in vids:
        d = torch.load(f"{FEAT}/video{v}.pt")
        f = d["feats"].numpy()
        if mu is not None:
            f = (f - mu) / sd
        data.append((torch.tensor(f, dtype=torch.float32).t().unsqueeze(0), d["labels"]))
    return data


def main():
    set_seed(0)
    tr_ids, va_ids, te_ids = get_splits()
    # standardize with TRAIN stats
    trraw = np.concatenate([torch.load(f"{FEAT}/video{v}.pt")["feats"].numpy() for v in tr_ids])
    mu, sd = trraw.mean(0), trraw.std(0) + 1e-6
    tr, va, te = load(tr_ids, mu, sd), load(va_ids, mu, sd), load(te_ids, mu, sd)
    print(f"Cataract-101: train/val/test {len(tr)}/{len(va)}/{len(te)} videos, {NUM_PHASES_CAT} phases")

    model = MultiStageTCN(2, 9, 64, in_dim=2048, num_classes=NUM_PHASES_CAT, causal=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best, best_sd = 0.0, None
    order = list(range(len(tr)))
    for ep in range(1, 41):
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
    print(f"TeCNO trained, best val_acc {best:.3f}")

    # test posteriors
    graph = estimate_graph([y.numpy() for _, y in tr], num_phases=NUM_PHASES_CAT)
    post = []
    model.eval()
    with torch.no_grad():
        for f, y in te:
            logits = model(f.to(device))[-1].squeeze(0).t().cpu().numpy()
            post.append((torch.softmax(torch.as_tensor(logits), 1).numpy(), y.numpy()))

    def ev(fn):
        o, l, a, f1, b = [], [], [], [], []
        for p, g in post:
            pred = fn(p); n = min(len(g), len(pred)); pred, gg = pred[:n], g[:n]
            o.append(over_segmentation(pred, gg)["ratio"]); l.append(transition_latency(pred, gg)["median_latency"])
            a.append(accuracy(pred, gg)); f1.append(segmental_f1(pred, gg, 0.10)[0]); b.append(calibration(p[:n], gg)["boundary_ece"])
        return dict(oseg=np.nanmean(o), lat=np.nanmean(l), acc=np.nanmean(a)*100, f1=np.nanmean(f1), bece=np.nanmean(b)*100)

    print(f"\n{'decoder':22s}{'over-seg':>9}{'lat':>7}{'acc%':>7}{'F1':>7}{'bECE%':>7}")
    am = ev(lambda p: p.argmax(1))
    print(f"{'argmax (raw)':22s}{am['oseg']:>9.2f}{am['lat']:>7.1f}{am['acc']:>7.1f}{am['f1']:>7.1f}{am['bece']:>7.1f}")
    cal = calibration  # noqa
    # best heuristic & CUSUM at over-seg<=2
    hpts = [ev(lambda p, w=w: causal_mode_filter(p.argmax(1), w)) for w in (3,5,9,15,25)]
    hpts += [ev(lambda p, g=g: online_uncertainty_smooth(p, gamma=g).argmax(1)) for g in (0.2,0.3,0.5,0.7)]
    qpts = [ev(lambda p, h=h: cusum_decode(p, graph, h)) for h in (3,5,8,12,18,26)]
    def at2(pts):
        el = [p for p in pts if p["oseg"] <= 2.0]; return min(el or pts, key=lambda p: p["lat"] if el else p["oseg"])
    hh, cc = at2(hpts), at2(qpts)
    print(f"{'heuristic @oseg<=2':22s}{hh['oseg']:>9.2f}{hh['lat']:>7.1f}{hh['acc']:>7.1f}{hh['f1']:>7.1f}{hh['bece']:>7.1f}")
    print(f"{'CUSUM @oseg<=2':22s}{cc['oseg']:>9.2f}{cc['lat']:>7.1f}{cc['acc']:>7.1f}{cc['f1']:>7.1f}{cc['bece']:>7.1f}")
    v = "QCD WINS latency" if cc['lat'] < hh['lat'] - 0.2 else "tie"
    print(f"\n  發現: argmax over-seg {am['oseg']:.1f}x, boundary-ECE {am['bece']:.1f}% (acc {am['acc']:.1f} 藏住)")
    print(f"        @over-seg<=2: heuristic lat {hh['lat']:.1f} vs CUSUM lat {cc['lat']:.1f} -> {v}")
    print("CAT101_QCD_DONE")


if __name__ == "__main__":
    main()
