"""Main benchmark: three training-free detectors on UCSD Ped2.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --subset UCSDped1 --window 21
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ucsd_vad import describe, evaluate, per_clip_table, resolve_device  # noqa: E402
from ucsd_vad.features import measure_encode_fps  # noqa: E402
from ucsd_vad.pipeline import (  # noqa: E402
    fuse_globally,
    postprocess,
    score_memory_bank,
    score_open_vocab,
    setup,
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
    p.add_argument("--subset", default="UCSDped2", choices=["UCSDped1", "UCSDped2"])
    p.add_argument("--backbone", default="openai/clip-vit-base-patch32")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--k", type=int, default=5, help="memory-bank neighbours")
    p.add_argument("--prompt-set", default="ensemble")
    p.add_argument("--window", type=int, default=11, help="temporal smoothing window")
    p.add_argument("--fusion-weight", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"device      : {resolve_device(args.device)}")
    print(f"backbone    : {args.backbone}")
    print(f"subset      : {args.subset}\n")

    exp = setup(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        subset=args.subset,
        backbone=args.backbone,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"train : {describe(exp.train_clips)}")
    print(f"test  : {describe(exp.test_clips)}")
    print(f"memory bank : {exp.bank.shape[0]} normal frames, dim {exp.bank.shape[1]}")
    source = "loaded from cache" if exp.features_from_cache else "encoded fresh"
    print(f"features    : {exp.extraction_seconds:.1f}s for {exp.frames_encoded} frames ({source})")

    # Throughput is measured on real frames every run, so the number stays
    # honest whether or not the feature cache was warm.
    fps = measure_encode_fps(exp.encoder, exp.test_clips[0].frames, args.batch_size)
    print(f"throughput  : {fps:.1f} frames/s end-to-end on {resolve_device(args.device)}\n")

    order = exp.clip_order
    raw_mb, t_mb = score_memory_bank(exp, k=args.k)
    raw_ov, explanations, t_ov = score_open_vocab(exp, prompt_set=args.prompt_set)
    raw_fu = fuse_globally(raw_mb, raw_ov, order, weight=args.fusion_weight)

    streams = {
        f"memory-bank (k={args.k})": raw_mb,
        f"open-vocab ({args.prompt_set}, zero-shot)": raw_ov,
        f"fusion (w={args.fusion_weight})": raw_fu,
    }

    rows = []
    for name, raw in streams.items():
        for window in (1, args.window):
            scored = postprocess(raw, window=window)
            metrics = evaluate(scored, exp.labels)
            rows.append({"method": name, "window": window, **metrics.to_dict()})
            print(f"{name:<38} window={window:<3} {metrics}")

    # Per-clip breakdown of the best configuration, for failure analysis.
    best = max(rows, key=lambda r: r["auc_micro"])
    best_stream = streams[best["method"]]
    breakdown = per_clip_table(
        postprocess(best_stream, window=best["window"]), exp.labels
    )

    # What the open-vocabulary head actually "said" on anomalous frames.
    fired: dict[str, int] = {}
    for name, labels in exp.labels.items():
        for label, is_anom in zip(explanations[name], labels):
            if is_anom:
                fired[label] = fired.get(label, 0) + 1

    # Mean score on normal vs anomalous frames per clip. Makes a failure like
    # an inverted clip visible as data rather than as a claim in prose.
    best_scored = postprocess(best_stream, window=best["window"])
    mean_scores = {}
    for name, labels in exp.labels.items():
        s = best_scored[name]
        mean_scores[name] = {
            "normal": round(float(s[labels == 0].mean()), 6) if (labels == 0).any() else None,
            "anomalous": round(float(s[labels == 1].mean()), 6) if (labels == 1).any() else None,
        }

    payload = {
        "config": {
            "subset": args.subset,
            "backbone": args.backbone,
            "device": resolve_device(args.device),
            "k": args.k,
            "prompt_set": args.prompt_set,
            "smoothing_window": args.window,
            "fusion_weight": args.fusion_weight,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "timing": {
            "feature_extraction_seconds": round(exp.extraction_seconds, 2),
            "features_from_cache": exp.features_from_cache,
            "frames_encoded": exp.frames_encoded,
            "encode_fps_measured": round(fps, 1),
            "memory_bank_scoring_seconds": round(t_mb, 3),
            "open_vocab_scoring_seconds": round(t_ov, 3),
        },
        "results": rows,
        "best": best,
        "per_clip_best": breakdown,
        "per_clip_mean_scores": mean_scores,
        "explanation_histogram_on_anomalous_frames": dict(
            sorted(fired.items(), key=lambda kv: -kv[1])
        ),
    }

    out = args.results_dir / f"benchmark_{args.subset}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nbest: {best['method']} (window={best['window']}) "
          f"AUC={best['auc_micro']:.4f}")
    print(f"wrote {rel(out)}")


if __name__ == "__main__":
    main()
