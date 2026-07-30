"""Frame-level evaluation for video anomaly detection.

The standard UCSD protocol reports *frame-level AUC*: concatenate every
test frame across clips and compute one ROC-AUC. Papers are not always
explicit about whether they concatenate ("micro") or average per-clip
AUCs ("macro"), and the two can differ by several points, so both are
reported here alongside average precision, which is more informative
under class imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import auc, average_precision_score, roc_auc_score, roc_curve


@dataclass
class Metrics:
    """Frame-level detection metrics for one configuration."""

    auc_micro: float
    auc_macro: float
    average_precision: float
    eer: float
    n_frames: int
    n_anomalous: int
    n_clips_scored: int
    n_clips_skipped: int

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"AUC(micro)={self.auc_micro:.4f}  AUC(macro)={self.auc_macro:.4f}  "
            f"AP={self.average_precision:.4f}  EER={self.eer:.4f}"
        )


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rate at which false positives and false negatives coincide."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def evaluate(
    scores_by_clip: dict[str, np.ndarray], labels_by_clip: dict[str, np.ndarray]
) -> Metrics:
    """Compute frame-level metrics from per-clip scores and labels.

    Clips whose labels are single-class carry no ROC signal on their own;
    they still contribute to the micro metrics but are excluded from the
    macro average, and the count of skipped clips is reported so the
    reader knows how many were dropped.
    """
    missing = set(labels_by_clip) - set(scores_by_clip)
    if missing:
        raise KeyError(f"No scores for clips: {sorted(missing)}")

    all_scores, all_labels, per_clip_auc = [], [], []
    skipped = 0
    for name, labels in labels_by_clip.items():
        scores = scores_by_clip[name]
        if len(scores) != len(labels):
            raise ValueError(
                f"{name}: {len(scores)} scores for {len(labels)} labels"
            )
        all_scores.append(scores)
        all_labels.append(labels)
        if len(np.unique(labels)) < 2:
            skipped += 1
        else:
            per_clip_auc.append(roc_auc_score(labels, scores))

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    if len(np.unique(labels)) < 2:
        raise ValueError("Test split contains a single class; AUC undefined.")

    return Metrics(
        auc_micro=float(roc_auc_score(labels, scores)),
        auc_macro=float(np.mean(per_clip_auc)) if per_clip_auc else float("nan"),
        average_precision=float(average_precision_score(labels, scores)),
        eer=equal_error_rate(labels, scores),
        n_frames=int(len(labels)),
        n_anomalous=int(labels.sum()),
        n_clips_scored=len(labels_by_clip),
        n_clips_skipped=skipped,
    )


def per_clip_table(
    scores_by_clip: dict[str, np.ndarray], labels_by_clip: dict[str, np.ndarray]
) -> list[dict]:
    """Per-clip AUC breakdown, for failure analysis."""
    rows = []
    for name in sorted(labels_by_clip):
        labels = labels_by_clip[name]
        scores = scores_by_clip[name]
        single_class = len(np.unique(labels)) < 2
        rows.append(
            {
                "clip": name,
                "frames": int(len(labels)),
                "anomalous": int(labels.sum()),
                "auc": None if single_class else float(roc_auc_score(labels, scores)),
            }
        )
    return rows
