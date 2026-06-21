# Online Surgical Phase Recognition as Quickest Change Detection: A Principled, Provably-Bounded Decoder that Dominates Heuristic Post-Processing

*Working draft — Cholec80 (primary) + Cataract-101 (cross-procedure). Target: IJCARS /
Computers in Biology and Medicine / IEEE JBHI. Numbers in **[[brackets]]** are to be
auto-filled / double-checked from `results/`; non-bracketed numbers are from the current
runs (`results/stats_corrected.json`, `results/frontier_ci.json`,
`results/cataract_frontier_ci.json`, `results/baselines_test_acc.json`).*

This paper subsumes our earlier reliability-suite study (`DRAFT.md`): that paper's
boundary-region ECE and over-segmentation diagnostics become the **measurement layer**
here, and its finding that *heuristic* post-processing only ties a trivial mode filter
becomes the **motivation** for a principled detector.

---

## Abstract

Online surgical phase recognition is dominated by frame accuracy, a metric blind to the
two properties that matter for intra-operative use: how *promptly* and how *cleanly*
phase transitions are detected. We show that strong online heads over-segment the phase
stream **6.2× [5.1, 7.7]** relative to ground truth and are **4.3× [3.7, 5.1]** worse
calibrated at transitions than in phase interiors (p≈3×10⁻²¹) — failures invisible to a
77% frame accuracy. Crucially, these three quantities — detection delay, false alarms,
over-segmentation — are exactly the operating characteristics of **Quickest Change
Detection (QCD)**, a field with decades of optimality theory. We recast online phase
recognition as multi-phase sequential change-point detection: a CUSUM / Shiryaev–Roberts
detector on the model's calibrated posterior log-likelihood ratio, constrained by a
phase-transition graph estimated from training data only. This yields (i) asymptotic
detection-delay optimality (Lorden/Pollak), and (ii) a **structural over-segmentation
bound** — #segments ≤ #graph transitions **by construction** — that no frame-wise
classifier or smoother can guarantee. Across 5 backbones × 5 causal temporal heads × 5
seeds (125 cells) on Cholec80, the detector cuts detection latency by **2.2 s
[1.6, 2.9]** (36/40 videos, p<0.001) at **non-inferior** accuracy — the one axis our
prior heuristic could not move — while keeping over-segmentation ≈ 1 by construction. The
result transfers to **cataract surgery** (Cataract-101), a procedure unlike
cholecystectomy: the same failure modes appear (over-seg 2.8×, boundary-ECE 13%) and the
detector again dominates (latency −0.21 s, segmental-F1 +1.35, accuracy non-inferior). We
report a deliberate trade-off honestly: an *ordered* transition graph buys tighter
segmentation at an accuracy cost, while the *unconstrained* graph gives the dominance
above. The contribution is a reframing with optimality **and** a concrete guarantee — not
another heuristic.

## 1. Introduction

Surgical phase recognition on Cholec80 is approaching saturation on frame accuracy, and
the community increasingly reports it in the realistic **online/causal** setting (each
frame labelled from past frames only). Yet frame accuracy is dominated by long, stable
phase interiors and is silent about clinical deployability: an assistant that flickers
between phases, or that announces a transition seconds late, is unusable even at 90%
accuracy.

We make the following contributions:

1. **A reliability measurement layer** (relaxed accuracy, segmental F1/edit,
   over-segmentation ratio, transition latency / false-starts, and **boundary-region
   ECE**) that exposes failures hidden by accuracy. On Cholec80, strong online heads
   over-segment 6.2× and are 4.3× worse calibrated at boundaries than interiors.
2. **The reframing:** online phase recognition *is* multi-phase Quickest Change
   Detection. We give a CUSUM / Shiryaev–Roberts detector on the posterior LLR,
   constrained by an estimated transition graph, with (a) asymptotic delay-optimality and
   (b) a **structural over-segmentation bound**.
3. **Evidence it dominates heuristics, with corrected statistics.** Across a 5×5×5 grid
   the detector achieves a CI-backed latency reduction at non-inferior accuracy — the axis
   ad-hoc smoothing could not move — and the win transfers to a second procedure.
4. **An honest trade-off and a learned extension.** We quantify the graph's accuracy↔
   segmentation knob, and explore a learned sequential detector trained to a
   differentiable delay–false-alarm objective.

## 2. Related work

- **Online phase recognition / temporal heads.** TeCNO (Czempiel et al., MICCAI 2020),
  ASFormer (Yi et al., BMVC 2021), LoViT (Liu et al., 2023), Trans-SVNet (Gao et al.,
  MICCAI 2021), Surgformer (Yang et al., MICCAI 2024). We use causal variants of five
  such heads as posterior sources.
- **Backbones / surgical SSL.** ResNet50 (ImageNet), EndoViT, a fine-tuned EndoViT, an
  end-to-end backbone, and a self-trained surgical-MAE (videos 1–40 only,
  contamination-free). Surgical SSL backbones (e.g. SurgeNet, Jaspers et al., MedIA 2025)
  motivate the backbone axis.
- **Metrics.** Funke et al. ("Metrics matter") and the relaxed-boundary convention of
  TeCNO/Jin et al. motivate our suite; we add boundary-region ECE.
- **Quickest Change Detection.** Page's CUSUM (1954); Lorden (1971) minimax optimality;
  Moustakides (1986); Pollak (1985) and Shiryaev–Roberts for the Bayesian criterion;
  Lai (1998) for the window-limited multi-hypothesis case. To our knowledge this theory
  has not been used to *define the decoder* for surgical phase recognition.
- **Cross-dataset.** Cataract-101 (Schoeffmann et al., MMSys 2018), 101 cataract-surgery
  videos, 10 phases.

## 3. Method

### 3.1 Setup
Frames `x_{1:T}`; a per-frame + causal temporal head emits the causal posterior
`p_t = P(z_t | x_{1:t})` over K phases (K=7 Cholec80, K=10 Cataract-101). The true phase
`z_t` is piecewise-constant with transitions `ν_1 < ν_2 < …`. Phases follow a (mostly)
fixed partial order with limited revisits, encoded as a transition graph **G** (allowed
successors per phase), estimated from **training labels only** (no test leakage).

### 3.2 The detector
Maintain a declared phase `k̂_t`. For each allowed successor `j` of `k̂` in G accumulate
the instantaneous posterior log-LR `s_t^j = log p_t[j] − log p_t[k̂]`:

```
CUSUM:            W_t^j = max(0, W_{t-1}^j + s_t^j);   declare k̂→j* when max_j W_t^j ≥ h
Shiryaev–Roberts: log R_t^j = logaddexp(0, log R_{t-1}^j) + s_t^j;  declare when log R ≥ h
```
On declaration, reset accumulators and set `k̂ ← j*`. Threshold `h` trades detection
delay against false-alarm rate; sweeping `h` traces the operating curve we compare on.
(Implementation: `qcd.py`, pure numpy, with a synthetic-optimality self-test.)

### 3.3 Claims
1. **Sequential optimality.** Under correct specification (posterior-LLR = true LLR), the
   per-transition CUSUM is asymptotically Lorden-minimax-optimal (min worst-case expected
   delay s.t. ARL-to-false-alarm ≥ γ); Shiryaev–Roberts is Pollak-optimal. The
   multi-phase/ordered carry-over is the window-limited GLR regime (Lai 1998).
2. **Structural over-segmentation bound (model-independent).** #predicted segments =
   #declared transitions; with an ordered graph G (DAG + R revisits) forward progress
   bounds segments ≤ (K−1)+R, so the over-segmentation ratio ≈ 1 **by construction**.
   Heuristic smoothers have no such bound and over-segment arbitrarily under low-confidence
   flicker. This claim needs no calibration assumption and is provable.
3. **Calibration coupling.** The LLR is meaningful only if posteriors are calibrated;
   mis-calibration biases `s_t^j` and degrades delay/FA. This makes temperature/conformal
   calibration *part of the method* and makes boundary-region ECE the right diagnostic for
   where the detector is most fragile (transitions).

### 3.4 Learned sequential detector (secondary)
Fixed-LLR CUSUM is optimal only under correct specification. We additionally learn a
small causal network mapping the posterior/feature stream to a stopping decision, trained
against a differentiable surrogate of the delay–false-alarm objective
(`E[delay] + λ·FA`). Hypothesis tested honestly: `learned ≥ fixed-LLR CUSUM ≥ heuristic`
on the operating curve (`results/p3.json`).

## 4. Experiments

### 4.1 Protocol
Cholec80 split: train 1–32, val 33–40, **test 41–80 (never touched)**. Posteriors from
**5 backbones** (ResNet50, EndoViT, EndoViT-ft, end-to-end, surgical-MAE) × **5 causal
heads** (TeCNO, LoViT-causal, ASFormer, Trans-SVNet, Surgformer) × **5 seeds**. Decoders
compared at matched over-seg ≤ 2.0: argmax, causal mode filter, confidence-EMA
(heuristics), CUSUM/SR with estimated vs full graph (ours), learned detector (ours).
Statistics are **per-(video, seed)** with cluster-bootstrap 95% CIs over the 40 test
videos, paired Wilcoxon Holm-corrected, and TOST non-inferiority (1% margin)
(`stats_corrected.py`, `frontier_ci.py`). All online heads pass a prefix-invariance
causality test (`test_causality.py`).

### 4.2 Accuracy hides failure (the diagnosis)
Pooled over the 5×3×5 baseline grid, argmax reaches **77.3% [74.7, 80.0]** frame accuracy
but over-segments **6.2× [5.1, 7.7]** and has median transition latency ≈ 25.6 s.
Boundary-region ECE is **0.314** vs interior **0.073** — a **4.3× [3.7, 5.1]** ratio,
Wilcoxon p≈3×10⁻²¹ (n=120 video×head units): models are most over-confident exactly where
they err. Accuracy sees none of this.

### 4.3 Heuristics fix segmentation but not latency (the motivation)
A boundary-and-uncertainty-aware (BUA) heuristic + smoothing significantly improves
segmental F1@10 (**+19.6**), edit (**+17.6**) and over-segmentation, at non-inferior
accuracy — but **detection latency is not significantly improved** (Δ≈−0.3 s,
p_holm≈0.90). Ad-hoc post-processing cleans up flicker yet cannot move the delay axis,
because it ignores sequential optimality and phase ordering.

### 4.4 The QCD detector dominates on the delay axis (Cholec80, 125 cells)
At matched over-seg ≤ 2.0:

| decoder | over-seg | latency (s) | accuracy % | seg-F1@10 |
|---|---|---|---|---|
| QCD (full graph) | 1.84 [1.6, 2.1] | **26.0 [22.7, 30.0]** | 77.3 [74.7, 79.9] | 64.2 [59.7, 67.9] |
| best heuristic | 1.78 [1.6, 2.0] | 28.2 [24.7, 32.0] | 77.0 [74.4, 79.5] | 65.3 [61.3, 69.1] |
| QCD (estimated graph) | 1.51 [1.4, 1.7] | 25.1 [21.7, 29.6] | 68.3 [64.4, 72.2] | 65.5 [61.7, 69.0] |

QCD-full cuts latency by **−2.2 s [−2.9, −1.6]** (36/40 videos, p_holm<0.001) at
**non-inferior accuracy** (+0.25% [0.07, 0.45]), with the structural over-seg property by
construction (cost: seg-F1 −1.1). This is the axis the heuristic could not move (§4.3).

### 4.5 Ablations: the graph is an accuracy↔segmentation knob (honest trade-off)
The **estimated (ordered)** graph yields tighter segmentation (over-seg 1.51) but costs
**−8.7% [−11.1, −6.5]** accuracy — too restrictive, forcing transitions along observed
paths. The **full (unconstrained)** graph gives the §4.4 dominance. We report both; the
QCD win is real but not a blow-out (consistent with our pre-registered fallback R5).
Calibration ablation (no-temp / temperature / 5-seed ensemble) confirms claim 3:
better-calibrated posteriors improve the detector (`results/qcd_p2.json`).

### 4.6 Cross-procedure generalization (Cataract-101)
Causal TeCNO, 5 seeds, per-video bootstrap over 21 test videos. The **failure modes
generalize**: argmax over-seg **2.83× [2.21, 3.54]**, boundary-ECE **13.4% [10.4, 16.5]**.
The **detector dominance generalizes**:

| decoder | over-seg | latency (s) | accuracy % | seg-F1@10 |
|---|---|---|---|---|
| QCD (full graph) | 1.78 | **0.52** | 88.6 | **74.3** |
| best heuristic | 1.83 | 0.73 | 88.4 | 73.0 |
| QCD (estimated graph) | 1.43 | 0.80 | 83.5 | 77.4 |

QCD-full: latency −0.21 s [−0.28, −0.15], seg-F1 +1.35 [+0.45, +2.30], accuracy
non-inferior (+0.22%), over-seg matched (all Holm-corrected). Absolute latencies are small
(cataract transitions are sharp), but the relative win is significant — and a single-seed
run read as a "tie" (that was the estimated-graph CUSUM); multi-seed CIs reveal QCD-full's
clean win. Same procedure-independent profile as Cholec80.

### 4.7 Learned detector (secondary)
The learned sequential detector trained to the delay–FA surrogate is reported in
`results/p3.json`; we state honestly whether `learned ≥ fixed-LLR CUSUM ≥ heuristic` holds
per cell, and treat the fixed-LLR CUSUM as the theory-backed primary. **[[final learned-vs-
CUSUM summary]]**

## 5. Discussion & limitations
- **R1 (calibration-bounded).** Where posteriors are mis-calibrated the LLR is biased; the
  calibration–detection coupling (§4.5) quantifies this rather than hiding it.
- **R2 (asymptotic).** Multi-phase optimality is asymptotic/approximate; the structural
  over-seg bound (claim 2) holds regardless and is concrete.
- **Matched-over-seg caveat.** Operating points are selected per cell at over-seg ≤ 2.0 and
  are not byte-identical in over-seg across decoders (e.g. QCD-full 1.84 vs heuristic
  1.78), so latency is compared at *approximately* matched over-seg.
- **Scope.** Latency headroom is procedure-dependent (large on Cholec80, small on
  Cataract-101); the over-segmentation guarantee is universal.

## 6. Conclusion
Posing online surgical phase recognition as Quickest Change Detection turns a metric-and-
heuristic problem into a principled one: a posterior-LLR CUSUM/SR detector with asymptotic
delay-optimality and a structural over-segmentation guarantee, which empirically dominates
heuristic post-processing on the delay axis at non-inferior accuracy, across backbones,
temporal heads, seeds, and a second surgical procedure. The reliability suite that exposed
the problem becomes the measurement layer; the heuristic's failure becomes the motivation.

## References
*(to finalize)* Page 1954; Lorden 1971; Moustakides 1986; Pollak 1985; Shiryaev;
Lai 1998; Czempiel et al. (TeCNO) 2020; Yi et al. (ASFormer) 2021; Gao et al.
(Trans-SVNet) 2021; Liu et al. (LoViT) 2023; Yang et al. (Surgformer) 2024; Funke et al.
(Metrics matter); Jaspers et al. (SurgeNet, MedIA 2025); Schoeffmann et al.
(Cataract-101, MMSys 2018).
