# Beyond Frame Accuracy: Reliability-Oriented Evaluation and a Boundary-and-Uncertainty-Aware Module for Online Surgical Phase Recognition

*Working draft — Cholec80. Target: IJCARS / Computers in Biology and Medicine / IEEE JBHI.*
Numbers in **[[brackets]]** are auto-filled from `results/` (Phase 2/3); see
`docs/PHASE2_RELIABILITY.md` and `docs/figures/summary_table.md`.

---

## Abstract

Surgical phase recognition on Cholec80 is approaching saturation on frame accuracy,
yet accuracy is a poor proxy for clinical deployability: it is dominated by long,
stable phases and is blind to (i) how cleanly and promptly phase *transitions* are
detected and (ii) whether a model's confidence can be *trusted*. We make three
contributions in the clinically realistic **online/causal** setting. First, we show
that standard Expected Calibration Error (ECE) is *diluted* by easy interior frames
and introduce **boundary-region ECE**, revealing that strong online heads are 4–5×
worse calibrated at phase transitions than in phase interiors — they are most
over-confident exactly where they err. Second, we package a reproducible
**reliability evaluation suite** (relaxed accuracy, segmental F1/edit,
over-segmentation, transition latency/miss-rate, and interior-vs-boundary
calibration) and apply it identically across temporal heads on identical features.
Third, we propose **BUA**, a lightweight, architecture-agnostic module
(boundary-aware loss, temperature/deep-ensemble calibration, and uncertainty-aware
causal smoothing) that improves boundary calibration, over-segmentation and latency
**without sacrificing accuracy**, demonstrated across two online heads and verified
with paired significance tests over five seeds.

## 1. Introduction

Automated recognition of the surgical phase from laparoscopic video underpins
intra-operative decision support, automated documentation, and OR workflow
optimisation. On the de-facto benchmark Cholec80, frame-level accuracy of
offline models now exceeds 95% under relaxed-boundary scoring, and even our
reproductions reach **[[MS-TCN 90.8% / TeCNO online 89.0%]]**. Continued
score-chasing on a saturated metric yields little clinical insight.

We argue, in line with recent critiques of evaluation practice, that the field
should foreground *reliability*. Two failure modes matter clinically and are
invisible to frame accuracy:

1. **Transition behaviour.** A model that is correct on 90% of frames can still
   thrash between phases around every transition, producing many spurious segments.
   Frame accuracy, dominated by long phases, hides this. We measure it with
   segmental F1/edit, an explicit over-segmentation ratio, and
   transition-detection latency / miss-rate.

2. **Trustworthy confidence.** For a model to defer or trigger an alert, its
   confidence must be calibrated *where decisions are hard* — at transitions. We
   show standard ECE averages this away and introduce **boundary-region ECE**.

We focus on the **online/causal** setting (each frame sees only the past), the only
setting deployable intra-operatively. We then introduce BUA, an architecture-
agnostic add-on, and show it improves reliability without hurting the (saturated)
accuracy.

## 2. Related Work

*Surgical phase recognition.* EndoNet/PhaseNet established CNN per-frame models on
Cholec80; SV-RCNet added recurrence; TeCNO brought multi-stage temporal convolution
with a strictly causal variant for online use; Transformer/Conformer heads (e.g.
LoViT, Trans-SVNet) model longer context. Most report frame accuracy and per-phase
precision/recall/Jaccard, often under a ±10 s relaxed boundary.

*Evaluation critique.* "Metrics Matter in Surgical Phase Recognition" (Funke et al.)
showed evaluation is inconsistent and that relaxed-boundary scoring inflates and
obscures differences. We extend this from accuracy-style metrics to **calibration**
and **online transition dynamics**, and provide an open, parameter-robust suite.

*Calibration / uncertainty.* Temperature scaling (Guo et al.) is the standard
post-hoc fix; deep ensembles give better-calibrated uncertainty at a cost we get for
free from multi-seed training. Calibration has been little studied for surgical
phase recognition, and never (to our knowledge) localised to phase boundaries.

## 3. Method

### 3.1 Two-stage backbone (reused)
Frames at 1 fps → frozen per-frame features (ResNet50 ImageNet, 2048-d; optionally
EndoViT surgical-MAE, 768-d) → an online temporal head. We use two heads to show
architecture-agnosticism: **TeCNO** (causal multi-stage TCN) and **LoViT-causal**
(causal Conformer). Both are trained with cross-entropy plus the standard truncated-
MSE temporal-smoothness term, batch size 1 (one full video per step).

### 3.2 The reliability suite (`metrics.py`)
All metrics operate on per-video integer label streams at 1 fps and per-frame
probabilities; all are causal-safe.

- **Relaxed accuracy / per-phase P-R-Jaccard.** Within ±tol s of a ground-truth
  transition, predicting either adjacent phase is not penalised (standard Cholec80
  convention); with tol=0 this is strict accuracy.
- **Segmental F1@{10,25,50} and edit** (action-segmentation standard): a predicted
  segment is a true positive if its temporal IoU with an unmatched same-label
  ground-truth segment ≥ the overlap threshold; edit is the normalised Levenshtein
  distance over the *sequence of segment labels*.
- **Over-segmentation ratio** = predicted segments / ground-truth segments.
- **Transition latency / miss-rate / false-starts.** For each ground-truth
  transition into phase *b* at time *t*, latency is the first time ≥ *t* at which the
  prediction is *b* and remains *b* for ≥ *k* frames; never reached ⇒ a miss; a
  switch to *b* before *t* ⇒ a false start. Online by construction.
- **Calibration.** ECE, MCE and reliability curves, plus the headline
  **interior-ECE** (frames outside ±tol of any transition) vs **boundary-region ECE**
  (frames within ±tol). The contrast quantifies the dilution that hides boundary
  over-confidence.

We report a sensitivity analysis over tol ∈ {5,10,15} s and k ∈ {1,3,5} to show the
conclusions are not artefacts of metric hyper-parameters.

### 3.3 BUA: Boundary-and-Uncertainty-Aware module
Four independently switchable components that touch only generic logit/probability
tensors, so the same module wraps any head:

1. **Auxiliary boundary head + boundary-weighted loss.** A 1×1 conv/linear on the
   last stage's penultimate features predicts "near a transition"; training adds a
   class-balanced boundary BCE and up-weights cross-entropy on frames within ±tol of
   a transition, focusing capacity where errors concentrate.
2. **Temperature scaling.** A single global temperature fit by NLL on the 8-video
   validation set (per-phase temperature is reported only as an ablation, being
   unstable on rare short phases at this val size).
3. **Deep-ensemble uncertainty.** Averaging the softmax of the 5 per-seed models —
   nearly free, since the seeds are trained anyway for significance. We compare
   single-model vs temperature vs ensemble calibration head-to-head.
4. **Uncertainty-aware causal smoothing.** A strictly causal EMA over calibrated
   probabilities whose gain scales with predictive confidence: low-confidence frames
   cannot flip the running estimate, suppressing flicker/over-segmentation and
   spurious early switches while preserving genuine, confident transitions.

## 4. Experiments

*Protocol.* Cholec80, 32/8/40 train/val/test (videos 1–32 / 33–40 / 41–80); the test
set is touched only for final reporting. Identical frozen features across heads.
Five seeds per configuration; paired Wilcoxon signed-rank over the 40 test videos.

### 4.1 Baselines reproduce prior accuracy (no regression)
**[[TeCNO video-avg 88.96% (prior 88.95); MS-TCN frame 90.84% (prior 90.81);
LoViT-causal 88.33% (prior 86.19)]]** — see `docs/PHASE2_RELIABILITY.md`.

### 4.2 Accuracy hides failure (the diagnosis)
Despite ~89% online accuracy, both online heads massively over-segment
(**[[TeCNO 6.8×, LoViT 7.8×]]** predicted/GT segments; F1@10 **[[≈30]]**), whereas
offline MS-TCN does not (**[[1.8×]]**). *Figure 2, 3.*

### 4.3 Over-confident at boundaries (the headline)
For every head, **boundary-ECE ≈ [[28–34%]]** is 4–5× the **interior-ECE ≈
[[5–9%]]**, while global ECE (**[[6–11%]]**) looks benign. Standard ECE hides the
problem; boundary-region ECE exposes it. *Figure 1.*

### 4.4 BUA improves reliability without hurting accuracy
**[[Phase 3 main table + paired Wilcoxon: Δboundary-ECE, Δover-seg, Δlatency with
p-values; accuracy non-inferior]].** Component ablation (−boundary, −smooth, −calib)
and the single/temperature/ensemble calibration comparison: **[[…]]**.

### 4.5 Robustness
Conclusions hold across tol ∈ {5,10,15} s and k ∈ {1,3,5} (appendix): **[[…]]**.

## 5. Discussion & Limitations
Single dataset (Cholec80); cross-procedure calibration transfer (AutoLaparo) is
future work and, if access lands, a fine-tune-then-measure design (not naive
zero-shot, since label spaces differ). EndoViT pretraining may overlap Cholec80
imagery (no phase labels) — stated as a caveat. The single/temperature/ensemble
ordering is reported as an empirical finding, not a gate.

## 6. Conclusion
High frame accuracy is not reliability. A boundary-localised calibration view plus a
lightweight, architecture-agnostic BUA module make online surgical phase recognition
measurably more trustworthy without trading away the accuracy the field already has.
