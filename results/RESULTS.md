# Results

Generated from `benchmark_UCSDped2.json` and `ablations_UCSDped2.json`. Do not edit by hand; run `scripts/render_results.py`.

## Setup

| setting | value |
| --- | --- |
| dataset | UCSDped2 |
| backbone | `openai/clip-vit-base-patch32` |
| device | cpu |
| memory-bank k | 5 |
| prompt set | ensemble |
| smoothing window | 11 frames |
| fusion weight | 0.5 |
| python | 3.11.9 |

Test split: **2010 frames** across **12 clips**, of which **1648 are anomalous** (82.0%). 4 clips are entirely anomalous and are excluded from the macro average.

## Frame-level detection

| method | window | AUC (micro) | AUC (macro) | AP | EER |
| --- | --- | --- | --- | --- | --- |
| memory-bank (k=5) | 1 | 0.8545 | 0.8418 | 0.9676 | 0.2296 |
| memory-bank (k=5) | 11 | 0.8820 | 0.8601 | 0.9741 | 0.1929 |
| open-vocab (ensemble, zero-shot) | 1 | 0.8231 | 0.8775 | 0.9608 | 0.2681 |
| open-vocab (ensemble, zero-shot) | 11 | 0.8361 | 0.9082 | 0.9641 | 0.2571 |
| fusion (w=0.5) | 1 | 0.8527 | 0.8533 | 0.9675 | 0.2267 |
| fusion (w=0.5) | 11 | 0.8746 | 0.8705 | 0.9730 | 0.2093 |

Best configuration: **memory-bank (k=5)**, smoothing window 11 — micro AUC **0.8820**.

## Throughput

| measurement | value |
| --- | --- |
| measured encode throughput | 8.3 frames/s |
| feature extraction wall time | 0.12 s (features were served from cache this run, so the extraction time below is a cache read, not encoding work) |
| frames encoded | 4560 |
| memory-bank scoring | 0.154 s |
| open-vocabulary scoring | 0.006 s |

## Per-clip breakdown (best configuration)

| clip | frames | anomalous | AUC |
| --- | --- | --- | --- |
| Test001 | 180 | 120 | 0.9328 |
| Test002 | 180 | 86 | 0.9445 |
| Test003 | 150 | 146 | 0.9606 |
| Test004 | 180 | 150 | 0.9702 |
| Test005 | 150 | 129 | 0.9601 |
| Test006 | 180 | 159 | 0.9706 |
| Test007 | 180 | 135 | 0.8059 |
| Test008 | 180 | 180 | n/a |
| Test009 | 120 | 120 | n/a |
| Test010 | 150 | 150 | n/a |
| Test011 | 180 | 180 | n/a |
| Test012 | 180 | 93 | 0.3362 |

Clips reporting `n/a` contain only anomalous frames, so a within-clip ROC curve is undefined.

### Failure case

**Test012 scores 0.3362 — below chance.** The detector is not merely uninformative there, it is anti-correlated with the ground truth. On this clip the distance score averages 0.0182 on normal frames against 0.0171 on anomalous ones — inverted.

This is the structural weakness of one-class distance scoring: it detects **novelty relative to the training set**, which is not the same thing as the anomaly of interest. Whatever makes those opening frames unusual — crowd configuration, lighting — the memory bank cannot distinguish 'unlike training data' from 'anomalous'. The semantic scorer has no such failure on this clip, because a cyclist either matches the text or does not, which is precisely why the two methods' errors are not interchangeable.

## What the open-vocabulary head reported

On ground-truth anomalous frames, the anomaly prompt with the highest similarity was:

| prompt label | frames | share |
| --- | --- | --- |
| cyclist | 1494 | 90.7% |
| skateboard | 106 | 6.4% |
| vehicle | 48 | 2.9% |

This is the explanation channel: the label is a by-product of scoring, not a separate classifier. It is only meaningful where the detector is right in the first place.

## Ablations

### Prompt set

| prompt set | AUC (raw) | AUC (smoothed) | what it tests |
| --- | --- | --- | --- |
| generic | 0.7166 | 0.7367 | No anomaly class is named; measures unaided semantic sensitivity. |
| minimal | 0.8261 | 0.8426 | Single prompt per side, naming the anomaly family only. |
| specific | 0.8026 | 0.8165 | Names the Ped2 anomaly categories; yields per-frame explanations. |
| ensemble | 0.8231 | 0.8361 | Paraphrase ensemble over both sides; the default configuration. |

The gap between the best vocabulary-specific set (`minimal`, 0.8426) and the `generic` control (0.7367) is **+0.1059 AUC**. That difference is the value of knowing the anomaly vocabulary in advance, and it is not available in a genuinely open-world setting.

### Prompt aggregation

| aggregation | AUC |
| --- | --- |
| mean | 0.8361 |
| max | 0.8295 |

### Memory-bank neighbourhood size

| k | AUC |
| --- | --- |
| 1 | 0.8827 |
| 3 | 0.8826 |
| 5 | 0.8820 |
| 10 | 0.8818 |
| 20 | 0.8811 |

### Temporal smoothing

| window | AUC memory bank | AUC open-vocab |
| --- | --- | --- |
| 1 | 0.8545 | 0.8231 |
| 5 | 0.8695 | 0.8298 |
| 11 | 0.8820 | 0.8361 |
| 21 | 0.8874 | 0.8432 |
| 31 | 0.8899 | 0.8503 |
| 51 | 0.8892 | 0.8595 |

Smoothing moves the memory bank from 0.8545 at window 1 to 0.8899 at window 31 (+0.0354).

### Distance / semantic fusion

| semantic weight | AUC |
| --- | --- |
| 0.0 | 0.8820 |
| 0.2 | 0.8814 |
| 0.4 | 0.8787 |
| 0.5 | 0.8746 |
| 0.6 | 0.8686 |
| 0.8 | 0.8526 |
| 1.0 | 0.8361 |

Pure distance scores 0.8820; pure semantic scores 0.8361. Fusion does **not** help on this dataset: the best mix (w=0.0, 0.8820) does not improve on pure distance scoring (0.8820). The two signals are not complementary here.

## The micro / macro disagreement

| method | AUC (micro) | AUC (macro) | macro − micro |
| --- | --- | --- | --- |
| memory bank | 0.8820 | 0.8601 | -0.0219 |
| open-vocab | 0.8361 | 0.9082 | +0.0721 |

The two pooling rules disagree about which method is better. Pooling every frame, the memory bank wins (0.8820 vs 0.8361); averaging per-clip AUCs, the zero-shot semantic scorer wins by a wide margin (0.9082 vs 0.8601).

Macro AUC only cares about ranking *within* a clip; micro AUC also requires scores to be comparable *across* clips. So the gap is a calibration diagnosis: the semantic scorer separates the classes decisively inside a scene, but its absolute scale drifts from scene to scene, and pooling destroys that ordering. Per-clip mean scores bear this out — on Test001 the semantic score rises from 0.013 on normal frames to 0.236 on anomalous ones, an 18× ratio, while on Test012 the same scorer spans only 0.011 to 0.014.

### Control

A *global* rank transform is monotone and so cannot change AUC: 0.8361 → 0.8361. Any movement below therefore comes from discarding **per-clip** scale, not from the rank transform itself.

### Calibration-aware fusion

Rank-normalising the semantic stream *within each clip* before fusing removes the drift while the distance stream keeps its global calibration.

**Smoothing window 11**

| semantic weight | AUC (micro) | AUC (macro) |
| --- | --- | --- |
| 0.0 | 0.8820 | 0.8601 |
| 0.2 | 0.8840 | 0.8910 |
| 0.4 | 0.8706 | 0.9012 |
| 0.5 | 0.8640 | 0.9025 |
| 0.6 | 0.8569 | 0.9024 |
| 0.8 | 0.8393 | 0.9015 |
| 1.0 | 0.8153 | 0.9002 |

Best micro at this window is 0.8840 (w=0.2), against 0.8820 for the memory bank alone — +0.0020. Macro moves +0.0309.

**Smoothing window 31**

| semantic weight | AUC (micro) | AUC (macro) |
| --- | --- | --- |
| 0.0 | 0.8899 | 0.8685 |
| 0.2 | 0.9048 | 0.9168 |
| 0.4 | 0.8988 | 0.9262 |
| 0.5 | 0.8924 | 0.9278 |
| 0.6 | 0.8847 | 0.9294 |
| 0.8 | 0.8659 | 0.9317 |
| 1.0 | 0.8406 | 0.9328 |

Best micro at this window is 0.9048 (w=0.2), against 0.8899 for the memory bank alone — +0.0149. Macro moves +0.0483.

The size of the effect depends on the smoothing window, which is why both are shown rather than only the flattering one. At window 11 the micro gain is marginal; at window 31 it is real and both metrics improve together, which is what distinguishes a genuine improvement from metric-shopping.

**This is a diagnostic, not a deployable detector.** Per-clip normalisation needs clip boundaries at inference time and is transductive within a clip. It identifies *where* the semantic signal is weak — calibration, not discrimination — and that is the useful finding. A deployable version would need an online estimate of the per-scene score distribution.

## Caveat on configuration choices

UCSD Ped2 provides no validation split. The smoothing window, neighbourhood size and fusion weight reported above were therefore swept directly on the test split. Those sweeps are a **sensitivity analysis**, not model selection, and the best cell of a sweep is an optimistic estimate of what the same configuration would achieve on unseen data. The pre-registered defaults in `run_benchmark.py` (k=5, window=11) were fixed before any of these numbers were seen.
