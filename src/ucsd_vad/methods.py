"""Training-free anomaly scorers over frozen CLIP embeddings.

Two complementary views of "anomalous":

``MemoryBankScorer``   distance-based. A frame is anomalous when it does
                       not resemble any *normal training frame*. Uses the
                       one-class protocol (normal clips only, no labels).

``OpenVocabScorer``    semantic. A frame is anomalous when it matches a
                       *textual description* of an anomaly better than a
                       description of normality. Uses no training data at
                       all, and names the anomaly it detected.

Neither fits any parameters by gradient descent, so both run on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import ClipEncoder
from .prompts import PromptSet


def _check_unit_norm(feats: np.ndarray, tol: float = 1e-3) -> None:
    norms = np.linalg.norm(feats, axis=1)
    if np.abs(norms - 1.0).max() > tol:
        raise ValueError("Embeddings must be L2-normalised before scoring.")


class MemoryBankScorer:
    """k-NN distance to a bank of normal training embeddings.

    Score is the mean cosine distance to the ``k`` nearest normal frames:
    high when a frame looks unlike anything seen during training.
    """

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.bank: np.ndarray | None = None

    def fit(self, normal_features: np.ndarray) -> "MemoryBankScorer":
        _check_unit_norm(normal_features)
        if len(normal_features) < self.k:
            raise ValueError(f"Bank has {len(normal_features)} frames but k={self.k}")
        self.bank = normal_features.astype(np.float32)
        return self

    def score(self, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
        if self.bank is None:
            raise RuntimeError("Call fit() before score()")
        _check_unit_norm(features)
        out = np.empty(len(features), dtype=np.float32)
        for start in range(0, len(features), batch_size):
            chunk = features[start : start + batch_size]
            # Unit-norm vectors: cosine distance = 1 - dot product.
            sims = chunk @ self.bank.T
            # Partial sort is enough; we only need the k largest similarities.
            top_k = np.partition(sims, -self.k, axis=1)[:, -self.k :]
            out[start : start + len(chunk)] = 1.0 - top_k.mean(axis=1)
        return out


@dataclass
class OpenVocabResult:
    """Per-frame anomaly probability plus the prompt that explains it."""

    scores: np.ndarray          # (N,) in [0, 1]
    explanations: list[str]     # (N,) best-matching anomaly label per frame
    anomaly_sims: np.ndarray    # (N, n_anomaly_prompts) raw cosine similarities


class OpenVocabScorer:
    """Zero-shot anomaly scoring by comparing frames to text descriptions.

    The score is a softmax over the best normal and best anomaly
    similarity, scaled by CLIP's logit scale. Uses no training frames.
    """

    def __init__(
        self,
        encoder: ClipEncoder,
        prompt_set: PromptSet,
        logit_scale: float = 100.0,
        aggregate: str = "mean",
    ):
        if aggregate not in {"mean", "max"}:
            raise ValueError("aggregate must be 'mean' or 'max'")
        self.prompt_set = prompt_set
        self.logit_scale = logit_scale
        self.aggregate = aggregate
        self.normal_emb = encoder.encode_texts(prompt_set.normal)
        self.anomaly_emb = encoder.encode_texts(prompt_set.anomaly)
        self.anomaly_labels = prompt_set.labels()

    def _pool(self, sims: np.ndarray) -> np.ndarray:
        """Reduce per-prompt similarities to one value per frame."""
        return sims.max(axis=1) if self.aggregate == "max" else sims.mean(axis=1)

    def score(self, features: np.ndarray) -> OpenVocabResult:
        _check_unit_norm(features)
        sims_normal = features @ self.normal_emb.T     # (N, Tn)
        sims_anomaly = features @ self.anomaly_emb.T   # (N, Ta)

        normal_logit = self._pool(sims_normal) * self.logit_scale
        anomaly_logit = self._pool(sims_anomaly) * self.logit_scale

        # Two-way softmax, computed stably.
        stacked = np.stack([normal_logit, anomaly_logit], axis=1)
        stacked -= stacked.max(axis=1, keepdims=True)
        probs = np.exp(stacked)
        probs /= probs.sum(axis=1, keepdims=True)

        best = sims_anomaly.argmax(axis=1)
        explanations = [self.anomaly_labels[i] for i in best]
        return OpenVocabResult(
            scores=probs[:, 1].astype(np.float32),
            explanations=explanations,
            anomaly_sims=sims_anomaly.astype(np.float32),
        )


def zscore(x: np.ndarray) -> np.ndarray:
    """Standardise scores so heterogeneous scorers can be summed."""
    std = x.std()
    return (x - x.mean()) / std if std > 1e-8 else np.zeros_like(x)


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Map scores to (0, 1] by rank, discarding their scale entirely.

    Applied to a whole split this is a monotone transform and therefore
    cannot change AUC — a useful property to assert in tests. Applied
    *within each clip* it removes cross-clip scale differences, which is
    the point of :func:`~ucsd_vad.pipeline.fuse_calibrated`.
    """
    from scipy.stats import rankdata

    return (rankdata(x) / len(x)).astype(np.float32)


def fuse(distance_scores: np.ndarray, semantic_scores: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """Convex combination of the two scorers after standardisation.

    ``weight`` is the semantic share: 0.0 is pure memory bank, 1.0 pure
    open-vocabulary. Standardisation is fitted on the evaluated split,
    which is transductive but score-scale only -- it uses no labels.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    return (1.0 - weight) * zscore(distance_scores) + weight * zscore(semantic_scores)
