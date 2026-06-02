# Results & Analysis — MS-TCN / TeCNO / LoViT on Cholec80

Two-stage pipeline: a ResNet50 per-frame feature extractor (trained once,
val accuracy **81.75%**) feeds four interchangeable temporal heads. Test set =
videos 41–80 (32 train / 8 val / 40 test split).

## Final comparison (test, videos 41–80)

| Temporal model | Causal? | Frame acc | Video-avg acc | Mean P | Mean R | Mean Jacc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MS-TCN** | no (offline) | **90.81%** | 90.80 ±6.4 | 87.3 | 86.0 | **76.3** |
| **TeCNO** | yes (online) | 88.95% | 88.96 ±5.9 | 84.2 | 82.2 | 71.4 |
| **LoViT** (ours) | no (offline) | 83.22% | 83.44 ±8.0 | 78.5 | 74.5 | 61.8 |
| **LoViT** (ours) | yes (online) | 86.19% | 86.91 ±7.5 | 80.4 | 78.1 | 65.0 |

Headline observations:
1. The temporal model matters a lot: it lifts the per-frame CNN (81.75%) by up
   to **+9%** (MS-TCN 90.81%).
2. **The TCN heads beat our LoViT-style transformer** on this dataset.
3. Causal LoViT (online) actually beats non-causal LoViT here — the opposite of
   the MS-TCN/TeCNO ordering — a symptom of the transformer overfitting (the
   bidirectional model exploits training-set future context that doesn't transfer).

## Why the transformer loses: overfitting on a tiny dataset

Cholec80 has only **32 training videos** = 32 sequences seen per epoch. We
observed a persistent **val ≈ 0.92 but test ≈ 84–89%** gap for every LoViT
variant, while MS-TCN/TeCNO show no such gap.

We tried five ways to close it — **none worked**:

| Attempt | Change | LoViT-causal test |
|---|---|:---:|
| 1 | 11M params, lr 1e-3 | diverged (loss spike) |
| 2 | + grad-clip, lr 3e-4 | 89.3% (best, but not reproducible) |
| 3 | BatchNorm → GroupNorm (batch=1 safe) | 87.4% |
| 4 | shrink to 2M params + more dropout | 86.3% |
| 5 | + temporal augmentation (crop + jitter) | 84.5% |
| final | clean GroupNorm config | 86.2% |

Capacity, normalization, dropout and augmentation all failed to make the
transformer competitive. The bottleneck is **data**, not the temporal head.

## The lesson

> On small surgical datasets, a model with the *right inductive bias* beats a
> more flexible but data-hungry one. MS-TCN's dilated temporal convolutions
> bake in "phases are locally smooth and long-range" — exactly the structure of
> surgical workflow — so it generalises from 32 videos. A Transformer's
> attention is more expressive but has weak temporal priors, so with 32
> sequences it overfits no matter how it is regularised.

This matches the literature: published LoViT (~92%) reaches its numbers not from
a cleverer temporal head alone but from a **stronger / self-supervised feature
extractor and large-scale pretraining**. The real lever for beating these
baselines is **better Stage-1 features**, not a fancier Stage-2 model.

## Reproduce

```bash
./run_full.sh     # Stage 0–2: frames → ResNet50 → features → MS-TCN + TeCNO → eval
./run_lovit.sh    # trains LoViT (causal + offline) on the same features and compares all four
```
