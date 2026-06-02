"""Cross-seed significance test: does the SurgeNet (surgical) backbone beat the
ImageNet control, holding the SurgicalMamba temporal head + protocol fixed?

Loads per-video accuracy npz files (one per seed per backbone), averages each
video's accuracy across seeds, and runs a paired Wilcoxon signed-rank test over
the 40 test videos. Also reports seed-level mean +/- std."""
import glob
import numpy as np
from scipy.stats import wilcoxon

RESULTS = "results_seeds"


def load(group):
    """group in {'control','surgenet'} -> (n_seeds, 40) acc matrix."""
    files = sorted(glob.glob(f"{RESULTS}/{group}_s*.npz"))
    accs = np.stack([np.load(f)["accs"] for f in files])  # (seeds, 40)
    return accs, files


ctrl, cf = load("control")
surg, sf = load("surgenet")
print(f"control  seeds: {len(cf)}  -> {[f.split('/')[-1] for f in cf]}")
print(f"surgenet seeds: {len(sf)}  -> {[f.split('/')[-1] for f in sf]}")

# seed-level means (training-stability view)
print(f"\ncontrol  per-seed acc: {[f'{a*100:.2f}' for a in ctrl.mean(1)]}  "
      f"mean={ctrl.mean()*100:.2f} +/- {ctrl.mean(1).std()*100:.2f}")
print(f"surgenet per-seed acc: {[f'{a*100:.2f}' for a in surg.mean(1)]}  "
      f"mean={surg.mean()*100:.2f} +/- {surg.mean(1).std()*100:.2f}")

# paired test on seed-averaged per-video accuracy
ctrl_v = ctrl.mean(0)   # (40,) per-video, averaged over seeds
surg_v = surg.mean(0)
diff = surg_v - ctrl_v
wins = int((diff > 0).sum())
stat, p = wilcoxon(surg_v, ctrl_v)

print(f"\n{'='*60}")
print("PAIRED Wilcoxon signed-rank (surgenet vs control), 40 videos")
print(f"  mean acc:  surgenet {surg_v.mean()*100:.2f}%  vs  control {ctrl_v.mean()*100:.2f}%"
      f"  (delta {diff.mean()*100:+.2f})")
print(f"  surgenet wins on {wins}/40 videos")
print(f"  Wilcoxon stat={stat:.1f}, p={p:.4f}  -> "
      f"{'SIGNIFICANT (p<0.05)' if p < 0.05 else 'NOT significant'}")
print(f"  reference: reproduced official weights = 94.49%")
print("=" * 60)
