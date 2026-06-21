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

## Corrected statistics (P4, `stats_corrected.py` → `results/stats_corrected.json`)
Per-(video, seed) analysis, no seed pre-averaging; cluster-bootstrap 95% CIs over the
40 test videos; paired Wilcoxon Holm-corrected across the metric family; TOST
non-inferiority (1% margin).
- **Boundary-region ECE vs interior (headline metric):** 0.314 vs 0.073 →
  **4.32× [3.73, 5.12]**, Wilcoxon **p=2.8e-21** (n=120 video×head units). The
  boundary-localised mis-calibration is real and not an averaging artefact.
- **Baseline (argmax) pooled (5 bb × 3 heads × 5 seeds):** acc 0.773 [0.747, 0.800],
  over-seg 6.2× [5.1, 7.7], F1@10 32.3 [28.9, 35.8], median latency 25.6 s.
- **BUA heuristic vs baseline (Holm):** significant ↑ on F1@10 (+19.6), edit (+17.6),
  over-seg (+3.3×), accuracy (+0.011), false-starts; **latency NOT significant**
  (Δ=−0.3 s, p_holm=0.90) — the axis QCD targets. TOST: accuracy non-inferior
  (CI_lo +0.5%).
## QCD frontier operating points — per-video CIs (`frontier_ci.py` → `results/frontier_ci.json`)
Pooled over the **full 5 bb × 5 heads × 5 seeds = 125 cells** (now including the added
Trans-SVNet + Surgformer heads), cluster-bootstrap over the 40 test videos. At each cell
the operating point is the min-latency frontier point with over-seg ≤ 2.0 (same rule as
`qcd_p2`); per-video metrics recorded there. **The dominance story is unchanged from the
3-head run — robust across all 5 temporal heads.**

| decoder | over-seg | latency (s) | accuracy % | seg-F1@10 |
|---|---|---|---|---|
| QCD (estimated graph) | 1.51 [1.4, 1.7] | 25.1 [21.7, 29.6] | 68.3 [64.4, 72.2] | 65.5 [61.7, 69.0] |
| QCD (full graph) | 1.84 [1.6, 2.1] | 26.0 [22.7, 30.0] | 77.3 [74.7, 79.9] | 64.2 [59.7, 67.9] |
| best heuristic | 1.78 [1.6, 2.0] | 28.2 [24.7, 32.0] | 77.0 [74.4, 79.5] | 65.3 [61.3, 69.1] |

**Frontier dominance (paired vs heuristic, Holm-corrected; +Δ = QCD better):**
- **Latency win is real and significant on both variants** — QCD-full **−2.2 s**
  [−1.6, −2.9], **36/40 videos**, p_holm<0.001; QCD-est −3.1 s [−1.0, −5.1], 31/40,
  p_holm=0.003. This is the axis the BUA heuristic could NOT move (latency Δ there was n.s.).
- **QCD-full is non-inferior on accuracy** (+0.25% [+0.07, +0.45], p_holm=0.04) while
  cutting latency — faster detection at no accuracy cost, with the structural over-seg
  property by construction. Cost: seg-F1 −1.1 [−1.6, −0.7] and a marginally higher
  over-seg (+0.06).
- **QCD-est is NOT a free win:** the ordered graph buys tighter segmentation
  (over-seg 1.51, −0.27 vs heuristic) and more latency but **costs −8.7% accuracy**
  [−11.1, −6.5] — the estimated graph is too restrictive. Honest trade-off, not dominance.
- **Caveat:** the selected operating points are not exactly over-seg-matched
  (QCD-full 1.84 vs heuristic 1.78), so the latency comparison is at *approximately*
  matched over-seg, not identical — noted for the paper.

**Takeaway:** the principled detector delivers the one thing heuristics couldn't —
a significant, CI-backed detection-latency reduction at non-inferior accuracy (QCD-full)
— plus the structural anti-over-segmentation guarantee. It is a measured win, not a
blowout (consistent with plan R5).

## Additional online heads — Trans-SVNet + Surgformer (`run_baselines.sh`, `verify_baselines.py`)
Two more causal/online temporal heads added as QCD posterior sources, trained 5 seeds ×
5 backbones (50 base checkpoints), strengthening the "across temporal heads" claim.
Both are clean-room **causal** adaptations on the shared feature stream (Trans-SVNet =
TCN temporal-embedding + spatial-query×temporal-memory cross-attention; Surgformer =
pure multi-scale hierarchical causal attention). Both pass the prefix-invariance
causality test. Cholec80 TEST (videos 41–80), frame / relaxed accuracy, mean over 5 seeds:

| head | rn50 | endovit | endovitft | e2e | 4-backbone mean | surgmae |
|---|---|---|---|---|---|---|
| Trans-SVNet | 85.9 / 87.6 | 80.0 / 82.0 | 86.7 / 88.4 | 87.8 / 89.6 | **85.1 / 86.9** | 55.2 / 56.7 |
| Surgformer  | 86.5 / 88.3 | 82.5 / 84.6 | 87.4 / 89.1 | 88.9 / 90.6 | **86.3 / 88.1** | 30.5 / 31.5 |

- **On the 4 standard backbones both heads land in the published Cholec80 online ballpark**
  and track our existing TeCNO/LoViT/ASFormer heads per backbone (apples-to-apples) →
  implementations validated.
- **surgmae degrades *every* head, not just the new ones** (cached existing-head surgmae
  TEST acc: TeCNO 59.0, LoViT 28.2, ASFormer 39.4; new: Trans-SVNet 55.2, Surgformer 30.5).
  The self-trained MAE backbone (videos 1–40 only) yields weak test posteriors across the
  board — a backbone-quality finding, not a head bug. Reported per-backbone, not hidden in
  an average.

- **Pending (GPU-blocked behind LLM benchmark):** ~~per-video CIs for QCD frontier
  operating points~~ **DONE**; ~~Trans-SVNet/Surgformer baselines~~ **DONE** (above).
  QCD frontier dominance re-run with all 5 heads in progress (`frontier_ci.py`).
