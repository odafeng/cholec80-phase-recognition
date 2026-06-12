# Research Plan v2 — Online Surgical Phase Recognition as Quickest Change Detection

> **STATUS: PROPOSED — for user review. Supersedes the v1 "BUA module" plan
> (`RESEARCH_PLAN.md`), which is retained as archive. v1's reliability suite,
> boundary-region ECE, multi-backbone matrix, and its *negative* result (heuristic
> post-processing only ties a dumb causal mode filter) are SUBSUMED here: they
> become the measurement layer and the motivation.**

## 1. One-line thesis
Online surgical phase recognition should be posed as **multi-phase sequential
change-point detection**. A detector on the model's *calibrated* phase posteriors,
constrained by the phase-transition graph, attains a **provably near-optimal
detection-delay / false-alarm trade-off** (Lorden/Pollak/Shiryaev) and a
**structural over-segmentation bound** that no frame-wise classifier or post-hoc
smoother can guarantee — and empirically **dominates heuristic post-processing** on
the delay–false-alarm–over-segmentation operating characteristic, across backbones,
temporal heads, and datasets.

## 2. Why this is a theory, not a salvage (and how it grew out of v1)
v1 measured, per online head, transition **latency**, **false-starts**, and
**over-segmentation** (3.8–15× the GT segment count, invisible to frame accuracy).
It then showed that heuristic fixes (confidence EMA, the "BUA" module) only **tie a
15-line causal mode filter** — because they are *ad hoc*: they ignore (a) sequential
**optimality** and (b) the phase-**ordering** structure.

But those three quantities — latency, false-alarm, over-segmentation — *are* the
operating characteristics of **Quickest Change Detection (QCD)**, a field with
decades of optimality theory (Page's CUSUM; Lorden 1971; Moustakides 1986; Pollak;
Shiryaev; Lai 1998 for the window-limited multi-hypothesis case). The reframing
turns v1's negative result into the motivation for a *principled* detector that is
provably optimal **for exactly the frontier we were comparing on**.

## 3. Theory

### 3.1 Setup
Frames `x_{1:T}`; a per-frame + causal temporal model emits the causal posterior
`p_t = P(z_t | x_{1:t})` over K=7 phases. True phase `z_t` is piecewise constant
with transitions `ν_1 < ν_2 < …`. Cholec80 phases follow a (mostly) fixed partial
order with limited revisits — encode this as a transition graph **G** (allowed
successors of each phase).

### 3.2 The detector
Maintain a declared phase `k̂_t`. For each allowed successor `j` of `k̂` in G,
accumulate the instantaneous posterior log-likelihood ratio
`s_t^j = log p_t[j] − log p_t[k̂]` with a CUSUM (or Shiryaev–Roberts) recursion:
```
W_t^j = max(0, W_{t-1}^j + s_t^j)          # CUSUM
declare transition k̂ → j*  when  max_j W_t^j ≥ h ;  then reset, k̂ ← j*
```
Threshold `h` trades **detection delay** against **false-alarm rate (ARL-to-FA)**.
Sweeping `h` traces the operating curve.

### 3.3 Claims to establish
1. **Sequential optimality.** Under correct specification (posterior-LLR = true
   LLR), the per-transition CUSUM is asymptotically minimax-optimal in **Lorden's**
   sense (min worst-case expected delay s.t. ARL-to-false-alarm ≥ γ); Shiryaev–
   Roberts is optimal in the **Pollak/Bayesian** sense. State precisely the
   multi-phase / ordered conditions (window-limited GLR, Lai 1998) under which the
   asymptotic optimality carries over.
2. **Structural over-segmentation bound (the concrete, model-independent guarantee).**
   #predicted segments = #declared transitions. With the ordered graph G (DAG plus
   `R` allowed revisits), forward progress bounds segments ≤ (K−1)+R ⇒ over-seg
   ratio ≈ 1 **by construction**. Heuristic smoothers have **no** such bound — under
   low-confidence flicker they over-segment arbitrarily (v1: up to 15×). This claim
   needs no calibration assumption and is provable.
3. **Calibration coupling.** The LLR is meaningful only if posteriors are calibrated;
   miscalibration biases `s_t^j` and provably degrades delay/FA. This makes
   calibration (temperature / conformal) *part of the method*, and makes v1's
   **boundary-region ECE** the right diagnostic for *where* the detector is most
   fragile (transitions). v1 is thus the measurement + prerequisite layer.

### 3.4 Learned sequential detector (the extra-novel, compute-heavy part)
Fixed-LLR CUSUM is optimal only under correct specification. We additionally **learn**
a causal sequential detector — a small network mapping the posterior/feature stream
to a stopping decision — trained against a **differentiable surrogate of the
delay–false-alarm objective** (soft stopping time; Lagrangian `E[delay] + λ·FA`; or
a policy-gradient stopping rule). This bridges classical optimality with learned
representations and is where DGX compute is spent. Hypothesis to test honestly:
`learned ≥ fixed-LLR CUSUM ≥ heuristic smoothing` on the operating curve.

## 4. Experiments
- **Posteriors** from 5 backbones (ResNet50, EndoViT, EndoViT-ft, end-to-end,
  **surgical-MAE (ours)**) × 3 causal heads (TeCNO, LoViT-causal, ASFormer) × 5 seeds.
  *(All already trained in v1 except MAE — reuse them.)*
- **Decoders compared on the operating characteristic** (delay vs FA vs over-seg vs
  accuracy): (1) argmax, (2) causal mode filter [standard], (3) confidence EMA
  [v1 heuristic], (4) **CUSUM / Shiryaev–Roberts [ours, fixed-LLR]**, (5) **learned
  sequential detector [ours]**.
- **Metrics**: detection delay (ADD), false-alarm rate / ARL-to-FA, over-seg,
  segmental F1/edit, frame + relaxed accuracy, ECE + boundary-ECE (v1 suite = the
  measurement layer).
- **Statistics (fixing v1's red-team findings)**: per-(video,seed) without
  pre-averaging; cluster-robust / bootstrap CIs; TOST + pre-registered margin for any
  non-inferiority claim; Holm/FDR across the family; dominance tested on the
  **frontier** (area-under-operating-curve; delay at matched FA & over-seg).
- **Datasets**: Cholec80 (primary); **AutoLaparo** (access applied) — show the
  framing + dominance transfer to a *different procedure* (train+test within
  AutoLaparo; same failure modes + same QCD win = "this is general").
- **Ablations**: with/without ordering graph G; CUSUM vs Shiryaev–Roberts;
  calibrated vs uncalibrated posteriors (quantify claim 3); per-backbone/head.

## 5. How the DGX is genuinely used
- Train/calibrate posteriors across the 5×3×5 grid (incl. the multi-day MAE).
- **Train the learned sequential detector**: architecture / λ / RL-vs-surrogate
  search — the real compute sink.
- Full-grid frontier evaluation + bootstrap across 2 datasets.
- **Honest hardware note:** GB10 is compute-bound (~155 img/s ViT-B); we scale by
  **breadth** (grid, seeds, detector search, 2 datasets) and the MAE, NOT by
  foundation-model-scale pretraining, which this chip cannot do competitively. FLOPs
  don't make the theory; the reframing does.

## 6. Risks & fallbacks (honest — no pre-promised success)
- **R1 — posteriors too miscalibrated/noisy ⇒ CUSUM LLR biased ⇒ may not beat
  heuristics.** Fallback: the **calibration–detection coupling** becomes the result
  ("near-optimal detection is bottlenecked by posterior calibration; here is the
  quantified gap and what closes it"). Still theory-grounded.
- **R2 — multi-phase optimality is only asymptotic/approximate.** State it honestly;
  the **structural over-seg bound (claim 2) holds regardless** and is concrete.
- **R3 — phase revisits break monotonicity.** The transition graph G handles it;
  ablate G.
- **R4 — learned detector overfits small data.** Keep the fixed-LLR CUSUM as the
  theory-backed *primary*; learned detector is the "can we do better" secondary.
- **R5 — it only marginally beats the mode filter.** Report the frontier honestly;
  even a modest but *principled, guaranteed* gain + the over-seg structural bound +
  the reframing is a real contribution (unlike v1's pure heuristics). Bar: dominate
  the frontier with theory, OR cleanly explain why not.

## 7. Decisive cheap go/no-go (do this FIRST)
**Phase 1 reuses v1's already-trained posteriors** (every backbone×head×seed
checkpoint exists). Implementing the CUSUM/SR decoder + operating-curve evaluation is
CPU-cheap. So **within hours we learn whether a principled detector dominates the
heuristics on the delay–FA–over-seg frontier.** That is the go/no-go for the entire
theory — before spending DGX-weeks on the learned detector. If QCD dominates → full
program; if it ties → we fall back to R1/R5 framing, still honest.

## 8. Phased execution (stop & report each phase)
- **P0 — Theory + sims.** Write the formal detector + state optimality conditions +
  prove the over-seg bound + verify CUSUM optimality on *synthetic* change-point data
  (CPU). Deliverable: theory note + passing sims.
- **P1 — GO/NO-GO (cheap, reuses v1 checkpoints).** CUSUM/SR decoder + operating-curve
  eval vs {argmax, mode filter, EMA} on existing posteriors, a few backbone/head
  cells. Deliverable: does principled detection dominate? Decision point.
- **P2 — Calibration coupling + ordering ablation** across the full grid.
- **P3 — Learned sequential detector** (DGX-heavy).
- **P4 — Cross-dataset (AutoLaparo) + MAE backbone + full grid + corrected stats.**
- **P5 — Paper.**

## 9. What is genuinely novel vs v1
- A **theoretical reframing** (QCD) with optimality **and** a structural
  over-segmentation guarantee — not a metric+heuristic.
- A **learned sequential detector** trained to the delay–FA objective.
- v1's reliability suite + boundary-ECE become the rigorous **measurement layer**;
  v1's negative result becomes the **motivation**. Nothing from v1 is wasted.
