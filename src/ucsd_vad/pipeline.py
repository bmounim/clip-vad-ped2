"""Shared experiment plumbing used by the benchmark and ablation scripts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Clip, find_dataset_root, load_split
from .features import ClipEncoder, cache_path, extract_split
from .methods import MemoryBankScorer, OpenVocabScorer, rank_normalize, zscore
from .prompts import get_prompt_set
from .temporal import smooth_per_clip


@dataclass
class Experiment:
    """Loaded data, cached features and labels for one (subset, backbone)."""

    subset: str
    backbone: str
    encoder: ClipEncoder
    train_features: dict[str, np.ndarray]
    test_features: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    train_clips: list[Clip]
    test_clips: list[Clip]
    extraction_seconds: float
    frames_encoded: int
    features_from_cache: bool

    @property
    def bank(self) -> np.ndarray:
        """All normal training embeddings stacked into one memory bank."""
        return np.concatenate([self.train_features[c.name] for c in self.train_clips])

    @property
    def clip_order(self) -> list[str]:
        return [c.name for c in self.test_clips]


def setup(
    data_root: Path,
    cache_dir: Path,
    subset: str = "UCSDped2",
    backbone: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    batch_size: int = 32,
) -> Experiment:
    """Load clips and produce (or reuse) cached CLIP features."""
    root = find_dataset_root(Path(data_root))
    train_clips = load_split(root, subset, "Train")
    test_clips = load_split(root, subset, "Test")

    encoder = ClipEncoder(backbone=backbone, device=device)

    start = time.perf_counter()
    train_features, fresh_train = extract_split(
        train_clips, encoder, cache_path(cache_dir, backbone, subset, "Train"), batch_size
    )
    test_features, fresh_test = extract_split(
        test_clips, encoder, cache_path(cache_dir, backbone, subset, "Test"), batch_size
    )
    elapsed = time.perf_counter() - start
    from_cache = not (fresh_train or fresh_test)

    labels = {c.name: c.labels for c in test_clips if c.labels is not None}
    frames = sum(len(c) for c in train_clips) + sum(len(c) for c in test_clips)

    return Experiment(
        subset=subset,
        backbone=backbone,
        encoder=encoder,
        train_features=train_features,
        test_features=test_features,
        labels=labels,
        train_clips=train_clips,
        test_clips=test_clips,
        extraction_seconds=elapsed,
        frames_encoded=frames,
        features_from_cache=from_cache,
    )


def fuse_globally(
    distance: dict[str, np.ndarray],
    semantic: dict[str, np.ndarray],
    order: list[str],
    weight: float,
) -> dict[str, np.ndarray]:
    """Standardise both score streams across the whole split, then combine.

    Standardising globally rather than per clip matters: per-clip
    normalisation implicitly assumes every clip contains anomalies and
    inflates the reported AUC.
    """
    d = np.concatenate([distance[n] for n in order])
    s = np.concatenate([semantic[n] for n in order])
    fused = (1.0 - weight) * zscore(d) + weight * zscore(s)

    out, cursor = {}, 0
    for name in order:
        n = len(distance[name])
        out[name] = fused[cursor : cursor + n]
        cursor += n
    return out


def fuse_calibrated(
    distance: dict[str, np.ndarray],
    semantic: dict[str, np.ndarray],
    order: list[str],
    weight: float,
) -> dict[str, np.ndarray]:
    """Fusion that corrects the semantic stream's cross-clip miscalibration.

    The open-vocabulary scorer separates classes decisively *within* a clip
    but its absolute scale drifts between scenes, so pooling raw semantic
    scores across clips destroys the ordering. Rank-normalising it inside
    each clip removes that scale while leaving the well-calibrated
    distance stream global.

    This buys accuracy at a cost that must be stated: it needs clip
    boundaries at inference time and is transductive within a clip. It is
    therefore reported as a diagnostic of *where* the semantic signal's
    weakness lies, not as a drop-in deployable detector.
    """
    d = np.concatenate([distance[n] for n in order])
    s = np.concatenate([rank_normalize(semantic[n]) for n in order])
    fused = (1.0 - weight) * zscore(d) + weight * zscore(s)

    out, cursor = {}, 0
    for name in order:
        n = len(distance[name])
        out[name] = fused[cursor : cursor + n]
        cursor += n
    return out


def score_memory_bank(exp: Experiment, k: int = 5) -> tuple[dict[str, np.ndarray], float]:
    """Score every test clip by k-NN distance to the normal memory bank."""
    scorer = MemoryBankScorer(k=k).fit(exp.bank)
    start = time.perf_counter()
    scores = {name: scorer.score(feats) for name, feats in exp.test_features.items()}
    return scores, time.perf_counter() - start


def score_open_vocab(
    exp: Experiment, prompt_set: str = "ensemble", aggregate: str = "mean"
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], float]:
    """Zero-shot score every test clip, returning per-frame explanations too."""
    scorer = OpenVocabScorer(
        exp.encoder, get_prompt_set(prompt_set), aggregate=aggregate
    )
    start = time.perf_counter()
    results = {name: scorer.score(feats) for name, feats in exp.test_features.items()}
    elapsed = time.perf_counter() - start
    scores = {name: r.scores for name, r in results.items()}
    explanations = {name: r.explanations for name, r in results.items()}
    return scores, explanations, elapsed


def postprocess(
    scores: dict[str, np.ndarray], window: int, mode: str = "mean"
) -> dict[str, np.ndarray]:
    """Temporal smoothing, applied independently within each clip."""
    return smooth_per_clip(scores, window=window, mode=mode)
