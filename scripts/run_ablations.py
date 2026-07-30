"""Ablation study over the design choices that actually matter.

Every ablation reuses the cached CLIP features, so the whole sweep costs
seconds once `run_benchmark.py` has populated the cache.

Usage:
    python scripts/run_ablations.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ucsd_vad import PROMPT_SETS, evaluate, get_prompt_set  # noqa: E402
from ucsd_vad.methods import rank_normalize  # noqa: E402
from ucsd_vad.pipeline import (  # noqa: E402
    fuse_calibrated,
    fuse_globally,
    postprocess,
    score_memory_bank,
    score_open_vocab,
    setup,
)

WINDOWS = (1, 5, 11, 21, 31, 51)
K_VALUES = (1, 3, 5, 10, 20)
FUSION_WEIGHTS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=REPO / "data" / "raw")
    p.add_argument("--cache-dir", type=Path, default=REPO / "data" / "features")
    p.add_argument("--results-dir", type=Path, default=REPO / "results")
    p.add_argument("--subset", default="UCSDped2")
    p.add_argument("--backbone", default="openai/clip-vit-base-patch32")
    p.add_argument("--device", default="auto")
    p.add_argument("--base-window", type=int, default=11)
    p.add_argument("--long-window", type=int, default=31,
                   help="second window for the calibration sweep")
    p.add_argument("--base-k", type=int, default=5)
    return p.parse_args()


def rel(path: Path) -> str:
    """Repo-relative path for logging, absolute if outside the repository."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def auc_of(scores, labels, window: int) -> float:
    return evaluate(postprocess(scores, window=window), labels).auc_micro


def both_auc(scores, labels, window: int) -> tuple[float, float]:
    m = evaluate(postprocess(scores, window=window), labels)
    return m.auc_micro, m.auc_macro


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    exp = setup(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        subset=args.subset,
        backbone=args.backbone,
        device=args.device,
    )
    labels, order = exp.labels, exp.clip_order
    ablations: dict[str, list[dict]] = {}

    # 1. Prompt set. The generic set names no anomaly class, so the gap to
    #    the specific sets is the value of knowing your vocabulary.
    print("\n[1] prompt set (zero-shot open-vocabulary)")
    rows = []
    for name in ("generic", "minimal", "specific", "ensemble"):
        scores, _, _ = score_open_vocab(exp, prompt_set=name)
        row = {
            "prompt_set": name,
            "description": get_prompt_set(name).description,
            "auc_raw": auc_of(scores, labels, 1),
            "auc_smoothed": auc_of(scores, labels, args.base_window),
        }
        rows.append(row)
        print(f"  {name:<10} raw={row['auc_raw']:.4f}  smoothed={row['auc_smoothed']:.4f}")
    ablations["prompt_set"] = rows

    # 2. Prompt aggregation across the ensemble.
    print("\n[2] prompt aggregation")
    rows = []
    for agg in ("mean", "max"):
        scores, _, _ = score_open_vocab(exp, prompt_set="ensemble", aggregate=agg)
        row = {"aggregate": agg, "auc": auc_of(scores, labels, args.base_window)}
        rows.append(row)
        print(f"  {agg:<10} AUC={row['auc']:.4f}")
    ablations["aggregation"] = rows

    # 3. Memory-bank neighbourhood size.
    print("\n[3] memory-bank k")
    rows = []
    for k in K_VALUES:
        scores, _ = score_memory_bank(exp, k=k)
        row = {"k": k, "auc": auc_of(scores, labels, args.base_window)}
        rows.append(row)
        print(f"  k={k:<3} AUC={row['auc']:.4f}")
    ablations["memory_bank_k"] = rows

    # 4. Temporal smoothing window, for both scorers.
    print("\n[4] temporal smoothing window")
    mb_scores, _ = score_memory_bank(exp, k=args.base_k)
    ov_scores, _, _ = score_open_vocab(exp, prompt_set="ensemble")
    rows = []
    for w in WINDOWS:
        row = {
            "window": w,
            "auc_memory_bank": auc_of(mb_scores, labels, w),
            "auc_open_vocab": auc_of(ov_scores, labels, w),
        }
        rows.append(row)
        print(f"  w={w:<3} bank={row['auc_memory_bank']:.4f}  "
              f"open-vocab={row['auc_open_vocab']:.4f}")
    ablations["temporal_window"] = rows

    # 5. Fusion weight: 0 is pure distance, 1 is pure semantics.
    print("\n[5] fusion weight (0=distance only, 1=semantic only)")
    rows = []
    for w in FUSION_WEIGHTS:
        fused = fuse_globally(mb_scores, ov_scores, order, weight=w)
        row = {"weight": w, "auc": auc_of(fused, labels, args.base_window)}
        rows.append(row)
        print(f"  w={w:<4} AUC={row['auc']:.4f}")
    ablations["fusion_weight"] = rows

    # 6. Micro vs macro for both scorers. The two disagree about which
    #    method is better, which is the most informative result in the study.
    print("\n[6] micro vs macro AUC (same scores, two pooling rules)")
    rows = []
    for name, s in (("memory bank", mb_scores), ("open-vocab", ov_scores)):
        micro, macro = both_auc(s, labels, args.base_window)
        rows.append({"method": name, "auc_micro": micro, "auc_macro": macro,
                     "gap": macro - micro})
        print(f"  {name:<14} micro={micro:.4f}  macro={macro:.4f}  "
              f"gap={macro - micro:+.4f}")
    ablations["micro_vs_macro"] = rows

    # 7. Calibration-aware fusion. Tests the explanation for that gap: the
    #    semantic stream is well-ordered within clips but badly scaled
    #    across them.
    #    Swept over two smoothing windows, because the size of the effect
    #    turns out to depend on the window and quoting a single one would
    #    overstate it.
    print("\n[7] calibration-aware fusion (per-clip rank on semantic stream)")
    rows = []
    for window in (args.base_window, args.long_window):
        print(f"  -- smoothing window {window}")
        for w in FUSION_WEIGHTS:
            fused = fuse_calibrated(mb_scores, ov_scores, order, weight=w)
            micro, macro = both_auc(fused, labels, window)
            rows.append({"window": window, "weight": w,
                         "auc_micro": micro, "auc_macro": macro})
            print(f"     w={w:<4} micro={micro:.4f}  macro={macro:.4f}")
    ablations["calibrated_fusion"] = rows

    # Reference points at the same windows, so the comparison is like-for-like.
    baseline_rows = []
    for window in (args.base_window, args.long_window):
        for name, s in (("memory bank", mb_scores), ("open-vocab", ov_scores)):
            micro, macro = both_auc(s, labels, window)
            baseline_rows.append({"window": window, "method": name,
                                  "auc_micro": micro, "auc_macro": macro})
    ablations["baselines_by_window"] = baseline_rows

    # Control: a *global* rank transform is monotone, so it must leave AUC
    # unchanged. Only the per-clip version can move the number. If this
    # control ever drifts, the explanation above is wrong.
    smoothed = postprocess(ov_scores, window=args.base_window)
    ranked_global = rank_normalize(np.concatenate([smoothed[n] for n in order]))
    split_back, cursor = {}, 0
    for n in order:
        split_back[n] = ranked_global[cursor : cursor + len(smoothed[n])]
        cursor += len(smoothed[n])

    plain = evaluate(smoothed, labels).auc_micro
    ranked = evaluate(split_back, labels).auc_micro
    print(f"\n  control: global rank transform is AUC-neutral "
          f"({plain:.4f} -> {ranked:.4f})")
    ablations["monotonicity_control"] = {
        "auc_raw": plain,
        "auc_global_rank": ranked,
        "note": "A global rank transform is monotone and must not change AUC; "
                "only per-clip ranking can, because it discards cross-clip scale.",
    }

    best_fusion = max(ablations["fusion_weight"], key=lambda r: r["auc"])
    best_calibrated = max(ablations["calibrated_fusion"], key=lambda r: r["auc_micro"])
    best_calibrated_macro = max(ablations["calibrated_fusion"], key=lambda r: r["auc_macro"])
    summary = {
        "config": {
            "subset": args.subset,
            "backbone": args.backbone,
            "base_window": args.base_window,
            "base_k": args.base_k,
        },
        "ablations": ablations,
        "best_fusion_weight": best_fusion,
        "best_calibrated_fusion": best_calibrated,
        "best_calibrated_fusion_by_macro": best_calibrated_macro,
    }
    out = args.results_dir / f"ablations_{args.subset}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nbest fusion weight: {best_fusion['weight']} "
          f"(AUC={best_fusion['auc']:.4f})")
    print(f"best calibrated fusion by micro: w={best_calibrated['weight']} "
          f"@window {best_calibrated['window']} "
          f"(micro={best_calibrated['auc_micro']:.4f}, "
          f"macro={best_calibrated['auc_macro']:.4f})")
    print(f"best calibrated fusion by macro: w={best_calibrated_macro['weight']} "
          f"@window {best_calibrated_macro['window']} "
          f"(micro={best_calibrated_macro['auc_micro']:.4f}, "
          f"macro={best_calibrated_macro['auc_macro']:.4f})")
    print(f"wrote {rel(out)}")


if __name__ == "__main__":
    main()
