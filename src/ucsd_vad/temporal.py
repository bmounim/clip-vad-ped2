"""Temporal post-processing of per-frame anomaly scores.

Frame-independent scorers ignore the fact that anomalies persist across
consecutive frames while scoring noise does not. Smoothing within each
clip suppresses isolated spikes and is standard practice in the video
anomaly detection literature.

Smoothing is always applied *per clip*: a window must never straddle a
clip boundary, since consecutive clips are unrelated scenes.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d


def smooth_scores(scores: np.ndarray, window: int = 11, mode: str = "mean") -> np.ndarray:
    """Smooth one clip's score sequence.

    Args:
        scores: (N,) per-frame scores for a single clip.
        window: window length in frames; ``1`` disables smoothing.
        mode: ``"mean"`` or ``"median"``.

    Returns:
        Smoothed scores, same shape. Edges use nearest-value padding so the
        output length matches the input exactly.
    """
    if window <= 1:
        return scores.astype(np.float32)
    if window > len(scores):
        window = len(scores) if len(scores) % 2 == 1 else len(scores) - 1
    if window <= 1:
        return scores.astype(np.float32)

    if mode == "mean":
        out = uniform_filter1d(scores.astype(np.float32), size=window, mode="nearest")
    elif mode == "median":
        out = median_filter(scores.astype(np.float32), size=window, mode="nearest")
    else:
        raise ValueError("mode must be 'mean' or 'median'")
    return out.astype(np.float32)


def smooth_per_clip(
    scores_by_clip: dict[str, np.ndarray], window: int = 11, mode: str = "mean"
) -> dict[str, np.ndarray]:
    """Apply :func:`smooth_scores` independently to every clip."""
    return {
        name: smooth_scores(scores, window=window, mode=mode)
        for name, scores in scores_by_clip.items()
    }


def minmax_per_clip(scores_by_clip: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rescale each clip's scores to [0, 1] independently.

    Widely used in the VAD literature, but it quietly assumes every clip
    contains both normal and anomalous frames and inflates the reported
    AUC. Reported here as a clearly-labelled variant, never as the
    headline number.
    """
    out = {}
    for name, scores in scores_by_clip.items():
        lo, hi = float(scores.min()), float(scores.max())
        span = hi - lo
        out[name] = (
            (scores - lo) / span if span > 1e-8 else np.zeros_like(scores)
        ).astype(np.float32)
    return out
