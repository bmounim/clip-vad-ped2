"""Unit tests for the scoring, smoothing and evaluation logic.

All tests use synthetic arrays, so the suite runs in under a second and
needs neither the dataset nor the CLIP weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from ucsd_vad.evaluate import equal_error_rate, evaluate, per_clip_table
from ucsd_vad.methods import MemoryBankScorer, fuse, rank_normalize, zscore
from ucsd_vad.prompts import PROMPT_SETS, PromptSet, get_prompt_set
from ucsd_vad.temporal import minmax_per_clip, smooth_scores


def unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


# --------------------------------------------------------------------- methods


def test_memory_bank_scores_seen_points_lower_than_unseen():
    rng = np.random.default_rng(0)
    bank = unit(rng.normal(size=(64, 8)).astype(np.float32))
    scorer = MemoryBankScorer(k=3).fit(bank)

    seen = scorer.score(bank[:16])
    far = unit(-bank[:16])  # antipodal points are maximally dissimilar
    assert seen.mean() < scorer.score(far).mean()


def test_memory_bank_identical_frame_has_near_zero_distance():
    bank = unit(np.eye(4, dtype=np.float32))
    scorer = MemoryBankScorer(k=1).fit(bank)
    assert scorer.score(bank)[0] == pytest.approx(0.0, abs=1e-5)


def test_memory_bank_rejects_unnormalised_input():
    scorer = MemoryBankScorer(k=1).fit(unit(np.eye(3, dtype=np.float32)))
    with pytest.raises(ValueError, match="L2-normalised"):
        scorer.score(np.full((2, 3), 5.0, dtype=np.float32))


def test_memory_bank_rejects_k_larger_than_bank():
    with pytest.raises(ValueError):
        MemoryBankScorer(k=10).fit(unit(np.eye(3, dtype=np.float32)))


def test_zscore_of_constant_input_is_zero_not_nan():
    out = zscore(np.full(10, 3.0))
    assert np.all(out == 0.0) and not np.isnan(out).any()


def test_fuse_endpoints_recover_each_scorer():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([3.0, 2.0, 1.0, 0.0])
    assert np.allclose(fuse(a, b, weight=0.0), zscore(a))
    assert np.allclose(fuse(a, b, weight=1.0), zscore(b))


def test_global_rank_normalisation_cannot_change_auc():
    """The calibration analysis rests on this: ranking a whole split is a
    monotone transform, so only *per-clip* ranking can move the metric."""
    rng = np.random.default_rng(7)
    labels = {"c": rng.integers(0, 2, size=400)}
    scores = {"c": rng.normal(size=400)}
    before = evaluate(scores, labels).auc_micro
    after = evaluate({"c": rank_normalize(scores["c"])}, labels).auc_micro
    assert before == pytest.approx(after, abs=1e-9)


def test_rank_normalize_is_bounded_and_order_preserving():
    x = np.array([5.0, -1.0, 3.0, 100.0])
    r = rank_normalize(x)
    assert r.min() > 0.0 and r.max() <= 1.0
    assert list(np.argsort(x)) == list(np.argsort(r))


def test_fuse_rejects_weight_outside_unit_interval():
    a = b = np.zeros(3)
    with pytest.raises(ValueError):
        fuse(a, b, weight=1.5)


# -------------------------------------------------------------------- temporal


def test_smoothing_preserves_length_for_every_window():
    x = np.random.default_rng(1).normal(size=57).astype(np.float32)
    for w in (1, 3, 11, 51, 101):
        assert len(smooth_scores(x, window=w)) == len(x)


def test_window_of_one_is_the_identity():
    x = np.random.default_rng(2).normal(size=20).astype(np.float32)
    assert np.allclose(smooth_scores(x, window=1), x)


def test_smoothing_suppresses_an_isolated_spike():
    x = np.zeros(41, dtype=np.float32)
    x[20] = 10.0
    assert smooth_scores(x, window=11, mode="mean")[20] < x[20]
    # A median filter should remove the spike outright.
    assert smooth_scores(x, window=11, mode="median")[20] == pytest.approx(0.0)


def test_minmax_per_clip_handles_a_constant_clip():
    out = minmax_per_clip({"a": np.full(5, 2.0, dtype=np.float32)})
    assert np.all(out["a"] == 0.0)


# -------------------------------------------------------------------- evaluate


def test_perfect_ranking_gives_auc_one():
    labels = {"c1": np.array([0, 0, 1, 1])}
    scores = {"c1": np.array([0.1, 0.2, 0.8, 0.9])}
    m = evaluate(scores, labels)
    assert m.auc_micro == pytest.approx(1.0)
    assert m.eer == pytest.approx(0.0, abs=1e-9)


def test_inverted_ranking_gives_auc_zero():
    labels = {"c1": np.array([0, 0, 1, 1])}
    scores = {"c1": np.array([0.9, 0.8, 0.2, 0.1])}
    assert evaluate(scores, labels).auc_micro == pytest.approx(0.0)


def test_single_class_clips_are_excluded_from_macro_but_counted():
    labels = {
        "mixed": np.array([0, 0, 1, 1]),
        "all_anom": np.array([1, 1, 1, 1]),  # no ROC signal alone
    }
    scores = {
        "mixed": np.array([0.1, 0.2, 0.8, 0.9]),
        "all_anom": np.array([0.5, 0.6, 0.7, 0.8]),
    }
    m = evaluate(scores, labels)
    assert m.n_clips_skipped == 1
    assert m.auc_macro == pytest.approx(1.0)  # only 'mixed' contributes
    assert m.n_frames == 8 and m.n_anomalous == 6


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="scores for"):
        evaluate({"c": np.zeros(3)}, {"c": np.array([0, 1])})


def test_missing_scores_are_rejected():
    with pytest.raises(KeyError):
        evaluate({}, {"c": np.array([0, 1])})


def test_per_clip_table_reports_none_for_single_class_clips():
    rows = per_clip_table(
        {"a": np.array([0.1, 0.9]), "b": np.array([0.3, 0.4])},
        {"a": np.array([0, 1]), "b": np.array([1, 1])},
    )
    by_clip = {r["clip"]: r for r in rows}
    assert by_clip["a"]["auc"] == pytest.approx(1.0)
    assert by_clip["b"]["auc"] is None


def test_equal_error_rate_is_half_for_random_scores():
    rng = np.random.default_rng(3)
    labels = np.repeat([0, 1], 500)
    eer = equal_error_rate(labels, rng.normal(size=1000))
    assert 0.35 < eer < 0.65


# --------------------------------------------------------------------- prompts


def test_every_registered_prompt_set_is_well_formed():
    for name, ps in PROMPT_SETS.items():
        assert ps.normal and ps.anomaly, name
        assert len(ps.labels()) == len(ps.anomaly), name


def test_generic_prompt_set_names_no_ped2_anomaly_class():
    """The generic set exists to measure the value of the vocabulary prior;
    if it leaks a class name the ablation is meaningless."""
    text = " ".join(get_prompt_set("generic").anomaly).lower()
    for leaked in ("bicycle", "cyclist", "skateboard", "cart", "truck", "vehicle"):
        assert leaked not in text


def test_prompt_set_rejects_mismatched_labels():
    with pytest.raises(ValueError):
        PromptSet(name="bad", normal=["a"], anomaly=["b", "c"], anomaly_labels=["only-one"])


def test_unknown_prompt_set_raises():
    with pytest.raises(KeyError):
        get_prompt_set("does-not-exist")
