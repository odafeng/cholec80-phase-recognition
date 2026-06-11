# Research outline (working draft)

**Working title:** *Does surgical-domain backbone pretraining still help once the
temporal model is state-of-the-art? A controlled study on Cholec80.*

## 1. Research question
Prior work shows surgical-domain self-supervised backbone pretraining improves
surgical phase recognition — **but those gains were measured with a weak temporal
head (MS-TCN/TeCNO)**, which leaves a lot of headroom for the backbone to fill.
We ask the unexplored question:

> When the temporal model is already state-of-the-art (the causal SSD/Mamba head
> that reaches ~94.6% on Cholec80), does a stronger surgical-pretrained backbone
> **still add accuracy, or has the strong temporal model already absorbed that
> benefit (saturation)?**

## 2. Gap vs. prior work (and explicit credit)
- **SurgicalMamba** (Oh & Sun, 2026): SOTA temporal model on Cholec80 (94.6% acc),
  but uses an **ImageNet** ConvNeXt-Tiny backbone. *We reuse their architecture,
  code, and released weights — our reproduction: 94.49%.*
- **SurgeNet** (Jaspers et al., MedIA 2025): surgical SSL backbones (4.7M frames)
  that beat ImageNet — but evaluated phase recognition with the **MS-TCN/TeCNO**
  head, and **not on Cholec80** (they use AutoLaparo / RAMIE). *We reuse their
  released backbone weights.*
- **Unfilled intersection:** surgical-pretrained backbone **×** SOTA temporal head
  **×** Cholec80. Nobody has tested whether the backbone benefit *stacks* with a
  SOTA temporal model.

**Our delta (what is new):** not a new method, but a **controlled single-variable
study** isolating the backbone's contribution under a SOTA temporal model, and a
characterisation of stacking vs. saturation across backbones and datasets.

## 3. Experimental matrix (single variable = backbone pretraining)
Temporal head fixed = SurgicalMamba (SOTA). Vary only the backbone init:

| Backbone | Pretraining data | Arch |
|---|---|---|
| ConvNeXt-Tiny (baseline) | ImageNet-1k | ConvNeXt v1 |
| ConvNeXt-V2 (SurgeNet) | 4.7M surgical frames (DINO) | ConvNeXt v2 |
| CaFormer (SurgeNetXL) | 4.7M surgical frames (DINO) | CaFormer |
| *(stretch)* video-SSL backbone | surgical video | e.g. SurgVISTA |

Datasets (cross-dataset validation): **Cholec80**, **AutoLaparo**, **M2CAI16**.
Each cell: end-to-end training, **3 seeds**, official 10s relaxed-boundary,
per-video metrics, **paired significance test** vs. the ImageNet baseline.

## 4. Two predicted outcomes — both publishable
- **Stacking:** surgical backbone still improves the SOTA temporal model (and may
  set a new Cholec80 number). → "backbone pretraining and strong temporal modeling
  are complementary."
- **Saturation:** no significant gain once the temporal head is SOTA. → a useful
  **negative result**: "strong temporal models absorb the backbone-pretraining
  benefit on Cholec80; gains reported with weak heads do not transfer to SOTA
  heads." Tells the field *where* the gains actually come from.

## 5. Honest scope / limitations
- **Narrow contribution**: a controlled ablation, not a new method → target tier is
  workshop / short paper / arXiv technical report, **not** a top-venue method paper.
- Single GPU (T4): limited seeds/epochs; we report exactly what we ran (no silent caps).
- EndoViT/SurgeNet were SSL-pretrained on surgical images that may include Cholec80
  frames (no phase labels used) — we will state this contamination caveat explicitly.
- Reproduction (94.49%) validates the pipeline but is not itself a contribution.

## 6. Why it's worth doing regardless of outcome
The disciplined protocol (fixed test set, multi-seed, significance, official metric,
cross-dataset) turns "try a swap" into evidence about *whether feature gains persist
under SOTA temporal modeling* — a question practitioners actually face when choosing
where to spend effort (better backbone vs. better temporal model).
