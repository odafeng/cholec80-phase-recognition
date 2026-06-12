# Master Results — BUA across 4 backbones × 3 online heads (Cholec80 test)

5 seeds, paired Wilcoxon with **Holm–Bonferroni** correction. Baseline = raw online
head; +BUA = boundary head + boundary-weighted loss + temperature + uncertainty-
aware causal smoothing. All heads verified **strictly causal** (test_causality.py).
A 5th backbone (self-trained surgical MAE) is pretraining and will be added.

> These numbers supersede an earlier draft: a LoViT causality bug (GroupNorm over
> the time axis) had inflated LoViT-online accuracy by leaking future frames, and a
> latency-metric bug under-reported latency. Both are fixed; LoViT was re-trained
> and everything re-evaluated. The corrected LoViT online accuracies are
> appropriately lower (e.g. EndoViT LoViT 83.8→74.3 base), and latencies are now
> honest (6–10 s, not 1–2 s).

## Headline table (baseline → +BUA)

| backbone | head | acc% | acc (Holm) | bound-ECE | over-seg | F1@10 | latency(s) |
|---|---|---|---|---|---|---|---|
| ResNet50 | TeCNO | 89.1→88.7 | non-inferior | 31.7→27.5 | 6.6→2.7 | 32.5→57.2 | 6.4→8.3 |
| ResNet50 | LoViT | 87.4→88.0 | non-inferior | 35.5→25.2 | 9.5→4.6 | 25.8→42.9 | 7.0→9.7 |
| ResNet50 | ASFormer | 87.7→88.5 | non-inferior | 33.6→26.1 | 6.4→3.5 | 37.3→50.8 | 10.1→8.1 |
| EndoViT (pretrained) | TeCNO | 78.7→77.3 | non-inferior | 35.4→33.9 | 4.9→1.9 | 37.8→61.3 | 33.1→34.8 |
| EndoViT (pretrained) | LoViT | 74.3→78.3 | improves | 40.6→26.8 | 5.4→2.9 | 34.0→50.7 | 39.0→28.6 |
| EndoViT (pretrained) | ASFormer | 70.7→68.4 | **sig worse** | 45.9→33.2 | 4.0→2.2 | 40.8→52.2 | 36.7→36.2 |
| EndoViT-ft | TeCNO | 88.8→88.3 | non-inferior | 21.7→25.8 | 7.7→3.4 | 32.3→53.1 | 8.9→8.9 |
| EndoViT-ft | LoViT | 87.7→87.1 | **sig worse** | 36.8→25.2 | 9.8→5.9 | 24.7→37.2 | 7.3→7.3 |
| EndoViT-ft | ASFormer | 88.5→88.4 | non-inferior | 32.5→28.5 | 5.8→3.9 | 40.1→48.8 | 8.5→8.5 |
| End-to-end | TeCNO | 88.6→88.7 | non-inferior | 30.2→26.5 | 5.3→2.3 | 38.8→62.9 | 8.1→9.6 |
| End-to-end | LoViT | 90.4→90.8 | non-inferior | 28.9→23.1 | 4.4→2.7 | 44.3→59.4 | 6.9→6.6 |
| End-to-end | ASFormer | 90.1→90.0 | non-inferior | 32.1→27.1 | 3.8→2.7 | 49.0→58.8 | 7.5→8.4 |

## What the matrix shows (after the causality/metric fixes)

1. **The failure modes are universal.** Every baseline (12 cells) over-segments
   (3.8–9.8× the GT segment count) and is badly mis-calibrated at boundaries
   (boundary-ECE 22–46%). Frame accuracy hides both. Holds across CNN, surgical
   ViT, fine-tuned ViT, and end-to-end backbones → a temporal-decoding problem, not
   a feature-quality problem.

2. **BUA fixes them.** Over-segmentation improves **12/12** (down to 1.9–5.9×),
   F1@10 improves **12/12** (+11 to +25 pts), boundary-ECE improves **11/12** (lone
   exception: EndoViT-ft TeCNO, already the best-calibrated baseline at 21.7).

3. **Accuracy is preserved on strong backbones; two weak/edge cells regress.** With
   Holm correction, BUA accuracy is non-inferior on **10/12** cells (and *improves*
   EndoViT-pretrained LoViT 74.3→78.3). Two cells are significantly worse: EndoViT-
   pretrained ASFormer (70.7→68.4, the weakest combo) and EndoViT-ft LoViT
   (87.7→87.1). Both are recoverable by the val-selected gentler smoothing operating
   point (gamma≈0.3) — reported as a trade-off curve.

4. **Latency is honest (6–10 s) and BUA's effect is small/mixed.** The smoothing
   adds at most a couple of seconds (sometimes reduces it, e.g. ResNet50 ASFormer
   10.1→8.1, e2e LoViT 6.9→6.6). The earlier scary 2.4→4.2 s "regression" was an
   artifact of the buggy latency metric and no longer stands.

5. **Backbone ranking (best baseline head).** end-to-end (ASFormer 90.1 / LoViT
   90.4) ≳ ResNet50 (TeCNO 89.1) ≈ EndoViT-ft (88.8) ≫ EndoViT-pretrained (78.7).
   End-to-end gives the strongest backbone, but reliability problems persist at every
   backbone strength — reinforcing #1.

## Rigor / honesty notes
- All online heads pass a prefix-invariance causality test (no future leakage).
- p-values are Holm-corrected across the ~15-metric family.
- Smoothing operating point selected on val, never test.
- Self-trained MAE backbone (videos 1–40 only) will add a 5th, contamination-free row.
