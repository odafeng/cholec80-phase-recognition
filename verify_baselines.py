"""Verify the new online heads (Trans-SVNet, Surgformer) reach a sane accuracy on the
Cholec80 TEST set (videos 41-80, never trained on), as additional QCD posterior sources.
Reports frame + relaxed accuracy, mean over the 5 seeds, per backbone and overall.
Temperature scaling is monotonic so it does not change argmax accuracy. CPU/GPU.
Run:  env -u LD_LIBRARY_PATH python verify_baselines.py
"""
import json
from pathlib import Path

import numpy as np

from qcd_p2 import posteriors
from metrics import accuracy

BACKBONES = {"rn50": "features", "endovit": "features_endovit",
             "endovitft": "features_endovit_ft", "e2e": "features_e2e",
             "surgmae": "features_surgmae"}
HEADS = ["transsvnet", "surgformer"]
SEEDS = [0, 1, 2, 3, 4]


def main():
    res = {}
    print(f"{'head':12s}{'backbone':12s}{'frame-acc':>12s}{'relaxed-acc':>14s}  (mean over 5 seeds)")
    for head in HEADS:
        res[head] = {}
        all_f, all_r = [], []
        for bb, feat in BACKBONES.items():
            fr, rl = [], []
            for s in SEEDS:
                ck = Path(f"checkpoints/rel_{bb}/{head}_base_s{s}.pt")
                if not ck.exists():
                    continue
                seqs = posteriors(ck, feat)
                fr.append(float(np.mean([accuracy(p.argmax(1), gt) for p, gt in seqs])))
                rl.append(float(np.mean([accuracy(p.argmax(1), gt, relaxed=True) for p, gt in seqs])))
            if not fr:
                continue
            mf, mr = 100 * np.mean(fr), 100 * np.mean(rl)
            res[head][bb] = {"frame_acc": mf, "relaxed_acc": mr, "n_seeds": len(fr)}
            all_f += fr; all_r += rl
            print(f"{head:12s}{bb:12s}{mf:11.2f}%{mr:13.2f}%")
        if all_f:
            of, orl = 100 * np.mean(all_f), 100 * np.mean(all_r)
            res[head]["_overall"] = {"frame_acc": of, "relaxed_acc": orl}
            print(f"{head:12s}{'ALL':12s}{of:11.2f}%{orl:13.2f}%  <-- overall\n")

    Path("results").mkdir(exist_ok=True)
    Path("results/baselines_test_acc.json").write_text(json.dumps(res, indent=2))
    print("-> results/baselines_test_acc.json")
    print("VERIFY_EXIT=0")


if __name__ == "__main__":
    main()
