# Phase 2 — Baseline Reliability Table (Cholec80 test, videos 41–80)

First application of the new `metrics.py` reliability suite to the baseline
temporal heads on identical ResNet50 (ImageNet) features. **No BUA yet** — this is
the "what's broken" baseline that motivates the method.

## Parity check (no regression vs prior repo numbers)

| Head | mode | frame acc | video-avg acc | prior repo | status |
|------|------|-----------|---------------|------------|--------|
| **TeCNO** (causal MS-TCN) | online | 87.80% | **88.96%** | 88.95% | ✅ reproduced |
| **MS-TCN** | offline | **90.84%** | 91.59% | 90.81% | ✅ reproduced |
| **LoViT-causal** | online | 87.35% | 88.33% | 86.19% | ✅ reproduced |

CNN backbone: ResNet50, best val acc **82.1%** (prior 81.75%). ✅

## Reliability suite (the new numbers)

| Metric | TeCNO (online) | MS-TCN (offline) | reading |
|--------|---------------:|-----------------:|---------|
| relaxed acc (±10s) | 90.55% | 93.23% | standard inflated metric |
| segmental F1@10 | **32.7** | 76.5 | online head is fragmented |
| segmental F1@25 / @50 | 31.6 / 26.9 | 75.2 / 70.2 | |
| segmental edit | **20.8** | 65.2 | online ordering is noisy |
| **over-segmentation** | **6.84×** | ~1.x | online predicts 6.8× too many segments |
| transition latency (median) | 1.9 s | 1.1 s | |
| transition miss-rate | 0.8% | 2.2% | |
| false-starts / video | 4.0 | 3.2 | |
| ECE | 8.49 | 6.41 | global calibration looks OK-ish |
| **interior-ECE** | **7.18** | 5.27 | calibrated AWAY from boundaries |
| **boundary-ECE** | **31.75** | 28.03 | **4–5× worse at transitions** |

(LoViT-causal, online, mirrors TeCNO: F1@10 28.4, edit 17.7, over-seg **7.80×**,
interior-ECE 9.35 vs **boundary-ECE 34.40** = 3.7×.)

## Two headline findings (both support the thesis)

1. **Accuracy hides failure.** TeCNO has a respectable 88.96% video-avg accuracy
   but **F1@10 = 32.7 and over-segmentation = 6.84×** — the online prediction
   stream is heavily fragmented (flicker), which frame accuracy (dominated by long
   stable phases) completely masks. This is exactly the clinically-relevant failure
   the paper argues standard evaluation ignores.

2. **Over-confident exactly at boundaries.** For both heads, global ECE (~6–8%)
   looks acceptable, but **boundary-region ECE is 28–32% — 4–5× the interior-ECE.**
   Standard ECE *dilutes* this by averaging over the many easy interior frames.
   `boundary-region ECE` (our metric) exposes it: the model is most over-confident
   precisely where it is most likely to be wrong (phase transitions).

These two motivate BUA (Phase 3): boundary-aware loss + uncertainty-aware causal
smoothing should cut over-segmentation and boundary-ECE **without** hurting the
accuracy that is already saturated.

_Per-video arrays in `results/baseline/*.npz`; per-video detail in `*.json`._
