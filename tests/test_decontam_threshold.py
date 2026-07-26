"""Tests for the embedding threshold calibration. No unit test loads a model."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from andbench.decontam import DEFAULT_SIMILARITY_THRESHOLD
from andbench.decontam_threshold import (
    DEFAULT_BETA,
    DEFAULT_PAIRS_PATH,
    Calibration,
    LabelledPair,
    PairSet,
    calibrate,
    load_pairs,
    pair_similarities,
    score_threshold,
    write_calibration,
)

ROOT = Path(__file__).resolve().parents[1]


class _ScriptedEmbedder:
    """Returns unit vectors whose pairwise cosine is a scripted angle.

    Each text is mapped to an angle, so a pair's cosine is exactly cos(difference).
    That lets a test state "these two sit at 0.95" without any model.
    """

    def __init__(self, angles: dict[str, float]) -> None:
        self.angles = angles
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[math.cos(self.angles[t]), math.sin(self.angles[t])] for t in texts]


def _pairs_at(*similarities: tuple[float, bool]) -> tuple[PairSet, _ScriptedEmbedder]:
    """Build a pair set whose cosines are exactly the given similarities."""
    angles: dict[str, float] = {}
    pairs: list[LabelledPair] = []
    for index, (similarity, collision) in enumerate(similarities):
        a, b = f"a{index}", f"b{index}"
        angles[a] = 0.0
        angles[b] = math.acos(max(-1.0, min(1.0, similarity)))
        pairs.append(LabelledPair(a=a, b=b, collision=collision))
    return PairSet(version=1, pairs=pairs), _ScriptedEmbedder(angles)


# --- the committed pair set ------------------------------------------------


def test_the_committed_pairs_load_with_both_classes() -> None:
    pairs = load_pairs(ROOT / DEFAULT_PAIRS_PATH)
    assert pairs.positives >= 5
    assert pairs.negatives >= 5


def test_the_committed_pairs_include_same_topic_hard_negatives() -> None:
    """They set the ceiling; separating a paraphrase from a recipe proves nothing."""
    pairs = load_pairs(ROOT / DEFAULT_PAIRS_PATH)
    hard = [p for p in pairs.pairs if not p.collision and "hard negative" in p.note.lower()]
    assert hard, "at least one pair must be labelled as the critical hard negative"


def test_the_shipped_default_threshold_is_the_calibrated_one() -> None:
    """It must not drift back to a round guessed number."""
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.71


# --- scoring one cut-off ---------------------------------------------------


def test_a_cut_off_below_everything_flags_everything() -> None:
    score = score_threshold([(0.9, True), (0.4, False)], 0.1)
    assert (score.true_positives, score.false_positives) == (1, 1)
    assert score.recall == 1.0
    assert score.precision == pytest.approx(0.5)


def test_a_cut_off_above_everything_flags_nothing() -> None:
    score = score_threshold([(0.9, True), (0.4, False)], 0.99)
    assert (score.true_positives, score.false_negatives) == (0, 1)
    assert score.recall == 0.0


def test_a_cut_off_inside_the_gap_is_perfect() -> None:
    score = score_threshold([(0.9, True), (0.4, False)], 0.7)
    assert (score.true_positives, score.true_negatives) == (1, 1)
    assert score.precision == 1.0
    assert score.recall == 1.0


def test_the_boundary_is_inclusive() -> None:
    """`>= threshold` is a collision, matching andbench.decontam."""
    assert score_threshold([(0.7, True)], 0.7).true_positives == 1


def test_recall_is_weighted_above_precision() -> None:
    """A missed collision ships contamination; a false alarm costs ten minutes."""
    high_recall = score_threshold([(0.9, True), (0.8, True), (0.75, False)], 0.7)
    high_precision = score_threshold([(0.9, True), (0.8, True), (0.75, False)], 0.85)
    assert high_recall.f_beta(DEFAULT_BETA) > high_precision.f_beta(DEFAULT_BETA)


def test_f_beta_is_zero_when_nothing_is_found() -> None:
    assert score_threshold([(0.2, True)], 0.9).f_beta() == 0.0


def test_precision_is_one_when_nothing_was_flagged() -> None:
    """Vacuously, and it must not divide by zero."""
    assert score_threshold([(0.2, False)], 0.9).precision == 1.0


# --- the sweep -------------------------------------------------------------


def test_a_clean_gap_recommends_its_midpoint() -> None:
    """Every value in the gap scores the same, so margin is the only tie-breaker."""
    pairs, embedder = _pairs_at((0.90, True), (0.86, True), (0.60, False), (0.20, False))
    calibration = calibrate(pairs, embedder)
    recommended = calibration.recommended
    assert recommended is not None
    assert recommended.threshold == pytest.approx(0.73, abs=0.011)  # midpoint of 0.60-0.86


def test_a_clean_gap_is_reported_as_such() -> None:
    pairs, embedder = _pairs_at((0.90, True), (0.30, False))
    summary = calibrate(pairs, embedder).summary()
    assert "Clean separation" in summary
    assert calibrate(pairs, embedder).perfect_separation


def test_overlapping_classes_are_flagged_loudly() -> None:
    """No cut-off is clean, and the recommendation must not pretend otherwise."""
    pairs, embedder = _pairs_at((0.90, True), (0.80, True), (0.85, False), (0.30, False))
    calibration = calibrate(pairs, embedder)
    summary = calibration.summary()
    assert "OVERLAP" in summary
    assert "add more hard negatives" in summary
    assert not calibration.perfect_separation


def test_an_overlap_falls_back_to_the_best_f_beta() -> None:
    pairs, embedder = _pairs_at((0.90, True), (0.80, True), (0.85, False), (0.30, False))
    recommended = calibrate(pairs, embedder).recommended
    assert recommended is not None
    # Recall-weighted, so it prefers catching both positives over avoiding the alarm.
    assert recommended.recall == 1.0


def test_the_margin_reports_the_two_bounds() -> None:
    pairs, embedder = _pairs_at((0.90, True), (0.60, False))
    margin = calibrate(pairs, embedder).margin
    assert margin is not None
    highest_negative, lowest_positive = margin
    assert highest_negative == pytest.approx(0.60, abs=1e-6)
    assert lowest_positive == pytest.approx(0.90, abs=1e-6)


def test_the_margin_is_undefined_with_only_one_class() -> None:
    pairs, embedder = _pairs_at((0.90, True))
    assert calibrate(pairs, embedder).margin is None


def test_similarities_are_computed_in_a_single_batched_call() -> None:
    """Embedding is the expensive part; one call per sweep, not one per pair."""
    pairs, embedder = _pairs_at((0.9, True), (0.4, False), (0.3, False))
    pair_similarities(pairs.pairs, embedder)
    assert embedder.calls == 1


def test_an_empty_pair_set_embeds_nothing_and_recommends_nothing() -> None:
    _pairs, embedder = _pairs_at()
    assert pair_similarities([], embedder) == []
    assert embedder.calls == 0
    empty = Calibration(model="m", scores=[], beta=DEFAULT_BETA, similarities=[])
    assert empty.recommended is None
    assert "No pairs to calibrate on." in empty.summary()


def test_labels_are_preserved_alongside_the_similarities() -> None:
    pairs, embedder = _pairs_at((0.9, True), (0.4, False))
    assert [label for _s, label in pair_similarities(pairs.pairs, embedder)] == [True, False]


# --- the artifact ----------------------------------------------------------


def test_the_sweep_is_written_as_auditable_json(tmp_path: Path) -> None:
    pairs, embedder = _pairs_at((0.90, True), (0.30, False))
    calibration = calibrate(pairs, embedder, model_name="test/model")
    payload = json.loads(
        write_calibration(calibration, tmp_path / "sweep.json").read_text(encoding="utf-8")
    )
    assert payload["model"] == "test/model"
    assert payload["perfect_separation"] is True
    assert payload["recommended_threshold"] is not None
    assert len(payload["scores"]) > 10
