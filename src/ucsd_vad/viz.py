"""Figures for qualitative and quantitative analysis.

Matplotlib only, no seaborn, and no explicit colours beyond defaults so
the figures stay legible when printed in greyscale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: scripts must run without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402


def _shade_anomalies(ax, labels: np.ndarray) -> None:
    """Shade the ground-truth anomalous intervals of a clip."""
    padded = np.concatenate([[0], labels, [0]])
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(
            s, e - 1, alpha=0.18, color="tab:red",
            label="ground-truth anomaly" if i == 0 else None,
        )


def plot_score_timeline(
    scores_by_method: dict[str, np.ndarray],
    labels: np.ndarray,
    clip_name: str,
    out_path: Path,
) -> Path:
    """Per-frame anomaly score against ground truth for a single clip."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    _shade_anomalies(ax, labels)
    for name, scores in scores_by_method.items():
        # Rescale to [0,1] for display only, so methods on different
        # scales can share one axis. Metrics are never computed from this.
        s = np.asarray(scores, dtype=np.float64)
        span = s.max() - s.min()
        ax.plot((s - s.min()) / span if span > 1e-8 else s * 0, label=name, linewidth=1.4)
    ax.set_xlabel("frame")
    ax.set_ylabel("anomaly score\n(display-normalised)")
    ax.set_title(f"{clip_name}: per-frame anomaly score")
    ax.set_xlim(0, len(labels) - 1)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc(
    scores_by_method: dict[str, np.ndarray],
    labels: np.ndarray,
    out_path: Path,
    title: str = "Frame-level ROC (UCSD Ped2)",
) -> Path:
    """Pooled frame-level ROC curves across all test clips."""
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for name, scores in scores_by_method.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=name, linewidth=1.6)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey", label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ablation_curve(
    x: list, y: list, xlabel: str, ylabel: str, title: str, out_path: Path
) -> Path:
    """Single-variable ablation curve with the maximum annotated."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(x, y, marker="o", linewidth=1.6)
    best = int(np.argmax(y))
    ax.annotate(
        f"best {y[best]:.3f} @ {x[best]}",
        xy=(x[best], y[best]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_qualitative_grid(
    frame_paths: list[Path],
    scores: list[float],
    explanations: list[str],
    out_path: Path,
    title: str = "",
) -> Path:
    """Frames annotated with their score and the prompt that explains them."""
    from PIL import Image

    n = len(frame_paths)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, path, score, why in zip(axes, frame_paths, scores, explanations):
        with Image.open(path) as im:
            ax.imshow(np.asarray(im.convert("L")), cmap="gray")
        ax.set_title(f"score {score:.2f} · “{why}”", fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
