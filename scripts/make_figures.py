"""Generate every figure referenced by the README.

Requires `run_benchmark.py` and `run_ablations.py` to have run first, so
that the feature cache and the ablation JSON exist.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ucsd_vad.pipeline import (  # noqa: E402
    fuse_globally,
    postprocess,
    score_memory_bank,
    score_open_vocab,
    setup,
)
from ucsd_vad.viz import (  # noqa: E402
    plot_ablation_curve,
    plot_qualitative_grid,
    plot_roc,
    plot_score_timeline,
)


def rel(path: Path) -> str:
    """Repo-relative path for logging, absolute if outside the repository."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=REPO / "data" / "raw")
    p.add_argument("--cache-dir", type=Path, default=REPO / "data" / "features")
    p.add_argument("--results-dir", type=Path, default=REPO / "results")
    p.add_argument("--figures-dir", type=Path, default=REPO / "results" / "figures")
    p.add_argument("--subset", default="UCSDped2")
    p.add_argument("--window", type=int, default=11)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--fusion-weight", type=float, default=0.5)
    p.add_argument("--timeline-clip", default=None, help="defaults to a mixed clip")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    exp = setup(data_root=args.data_root, cache_dir=args.cache_dir, subset=args.subset)
    labels, order = exp.labels, exp.clip_order

    mb, _ = score_memory_bank(exp, k=args.k)
    ov, explanations, _ = score_open_vocab(exp, prompt_set="ensemble")
    fu = fuse_globally(mb, ov, order, weight=args.fusion_weight)

    streams = {
        "memory bank": postprocess(mb, args.window),
        "open-vocab": postprocess(ov, args.window),
        "fusion": postprocess(fu, args.window),
    }

    # Timeline: prefer a clip that contains both normal and anomalous frames.
    clip = args.timeline_clip
    if clip is None:
        mixed = [n for n, y in labels.items() if 0 < y.sum() < len(y)]
        clip = mixed[0] if mixed else order[0]
    written = [
        plot_score_timeline(
            {k: v[clip] for k, v in streams.items()}, labels[clip], clip,
            args.figures_dir / f"timeline_{clip}.png",
        )
    ]

    # Timeline for the worst scored clip, if the benchmark identified one.
    bench_file = args.results_dir / f"benchmark_{args.subset}.json"
    if bench_file.exists():
        rows = json.loads(bench_file.read_text(encoding="utf-8"))["per_clip_best"]
        scored = [r for r in rows if r["auc"] is not None]
        if scored:
            worst = min(scored, key=lambda r: r["auc"])["clip"]
            if worst != clip:
                written.append(
                    plot_score_timeline(
                        {k: v[worst] for k, v in streams.items()},
                        labels[worst], f"{worst} (worst clip)",
                        args.figures_dir / f"timeline_{worst}_failure.png",
                    )
                )

    # Pooled ROC across all test frames.
    y = np.concatenate([labels[n] for n in order])
    written.append(
        plot_roc(
            {k: np.concatenate([v[n] for n in order]) for k, v in streams.items()},
            y,
            args.figures_dir / "roc.png",
        )
    )

    # Ablation curves, if the ablation study has been run.
    ablation_file = args.results_dir / f"ablations_{args.subset}.json"
    if ablation_file.exists():
        ab = json.loads(ablation_file.read_text(encoding="utf-8"))["ablations"]
        written.append(
            plot_ablation_curve(
                [r["window"] for r in ab["temporal_window"]],
                [r["auc_memory_bank"] for r in ab["temporal_window"]],
                "smoothing window (frames)", "frame-level AUC",
                "Temporal smoothing (memory bank)",
                args.figures_dir / "ablation_window.png",
            )
        )
        written.append(
            plot_ablation_curve(
                [r["weight"] for r in ab["fusion_weight"]],
                [r["auc"] for r in ab["fusion_weight"]],
                "semantic weight (0 = distance, 1 = semantic)", "frame-level AUC",
                "Distance / semantic fusion",
                args.figures_dir / "ablation_fusion.png",
            )
        )
    else:
        print(f"skipping ablation figures: {ablation_file.name} not found")

    # Qualitative panel: highest-scoring anomalous frames and what fired.
    ranked: list[tuple[float, Path, str]] = []
    for clip_obj in exp.test_clips:
        n = clip_obj.name
        for i, path in enumerate(clip_obj.frames):
            if labels[n][i]:
                ranked.append((float(streams["fusion"][n][i]), path, explanations[n][i]))
    ranked.sort(key=lambda t: -t[0])
    top = ranked[:8]
    written.append(
        plot_qualitative_grid(
            [t[1] for t in top], [t[0] for t in top], [t[2] for t in top],
            args.figures_dir / "qualitative_top.png",
            title="Highest-scoring anomalous frames, with the prompt that explains them",
        )
    )

    for path in written:
        print(f"wrote {rel(path)}")


if __name__ == "__main__":
    main()
