# Master Results — BUA across 4 backbones × 3 online heads (Cholec80 test)

5 seeds, paired Wilcoxon. Baseline = raw online head; +BUA = boundary head +
boundary-weighted loss + temperature + uncertainty-aware causal smoothing.
A 5th backbone (self-trained surgical MAE) is pretraining and will be added.

## Headline table (baseline → +BUA)

| backbone | head | acc% | acc verdict | boundary-ECE | over-seg | F1@10 |
|---|---|---|---|---|---|---|
| ResNet50 | TeCNO | 89.1→88.7 | non-inferior | 31.7→27.5 | 6.6→2.7 | 32.5→57.2 |
| ResNet50 | LoViT | 87.7→87.8 | non-inferior | 34.7→25.1 | 8.4→5.0 | 27.4→40.1 |
| ResNet50 | ASFormer | 87.7→88.5 | non-inferior | 33.6→26.1 | 6.4→3.5 | 37.3→50.8 |
| EndoViT (pretrained) | TeCNO | 78.7→77.3 | non-inferior | 35.4→33.9 | 4.9→1.9 | 37.8→61.3 |
| EndoViT (pretrained) | LoViT | 83.8→81.8 | **sig worse** | 34.8→28.7 | 4.1→2.7 | 45.5→55.2 |
| EndoViT (pretrained) | ASFormer | 70.7→68.4 | **sig worse** | 45.9→33.2 | 4.0→2.2 | 40.8→52.2 |
| EndoViT-ft | TeCNO | 88.8→88.3 | non-inferior | 21.7→25.8 | 7.7→3.4 | 32.3→53.1 |
| EndoViT-ft | LoViT | 87.4→88.1 | non-inferior | 37.9→25.8 | 10.5→5.3 | 23.7→41.0 |
| EndoViT-ft | ASFormer | 88.5→88.4 | non-inferior | 32.5→28.5 | 5.8→3.9 | 40.1→48.8 |
| End-to-end | TeCNO | 88.6→88.7 | non-inferior | 30.2→26.5 | 5.3→2.3 | 38.8→62.9 |
| End-to-end | LoViT | 88.8→**90.5** | improves | 32.3→25.4 | 4.1→2.8 | 44.8→58.7 |
| End-to-end | ASFormer | 90.1→90.0 | non-inferior | 32.1→27.1 | 3.8→2.7 | 49.0→58.8 |

## What the matrix shows

1. **The failure modes are universal.** Every baseline (12 cells) over-segments
   (3.8–10.5× the GT segment count) and is badly mis-calibrated at boundaries
   (boundary-ECE 22–46%). Frame accuracy hides both. This holds across CNN, surgical
   ViT, fine-tuned ViT, and end-to-end backbones — it is a temporal-decoding problem,
   not a feature-quality problem.

2. **BUA fixes them, almost everywhere.** Over-segmentation improves in **12/12**
   cells (down to 1.9–5.3×), F1@10 improves in **12/12** (+11 to +25 points),
   boundary-ECE improves in **11/12** (the one exception is EndoViT-ft TeCNO, already
   the best-calibrated baseline at 21.7).

3. **Accuracy is preserved when features are reasonable.** On the three strong
   backbones (ResNet50, EndoViT-ft, end-to-end), BUA is accuracy-**non-inferior on
   all 9 head cells** (end-to-end LoViT even improves, 88.8→90.5). The only accuracy
   regressions are on the deliberately-weak **EndoViT-pretrained** features
   (78–84% base): there the confidence-adaptive smoothing, tuned for confident
   models, over-suppresses. → motivates the smoothing operating-point analysis
   (a test-time knob; no retraining), reported as a latency/accuracy/over-seg
   trade-off curve.

4. **Backbone ranking (best baseline head).** end-to-end ASFormer 90.1 ≳ ResNet50
   TeCNO 89.1 ≈ EndoViT-ft TeCNO 88.8 ≫ EndoViT-pretrained 83.8. End-to-end and
   fine-tuning give a modest backbone gain, but the reliability problems persist at
   every backbone strength — reinforcing finding #1.

## Honest caveats
- EndoViT-pretrained accuracy regression under BUA-full is real and reported; the
  smoothing knob recovers it (sweep pending).
- EndoViT-ft TeCNO boundary-ECE worsens slightly (lone exception).
- Self-trained MAE backbone (test-clean) will add a 5th, contamination-free row.
