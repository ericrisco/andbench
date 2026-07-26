"""Tests for the And-Obert judge calibration gate (B3.04, constitution P14)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from andbench.harness.calibration import (
    DEFAULT_MIN_AGREEMENT,
    MIN_KAPPA,
    SUBSTANTIAL_KAPPA,
    CalibrationCase,
    build_sheet,
    calibrate,
    cohen_kappa,
    load_answers,
    load_sheet,
    unlabelled,
    write_record,
    write_sheet,
)
from andbench.harness.judge import JudgeVerdict, ModelAnswer
from andbench.schema import Item


def _obert(item_id: str, area: str = "historia", reference: str = "La referència.") -> Item:
    return Item.model_validate(
        {
            "id": item_id,
            "track": "and-obert",
            "area": area,
            "question": f"Pregunta oberta {item_id}?",
            "answer_text": reference,
            "difficulty": 2,
            "source_doc_id": "doc-1",
            "author": "alice",
            "verifier": "bob",
            "public": True,
            "tags": [],
        }
    )


def _mcq(item_id: str) -> Item:
    return Item.model_validate(
        {
            "id": item_id,
            "track": "and-coneix",
            "area": "geografia",
            "question": "Quina?",
            "choices": ["a", "b", "c", "d"],
            "answer": 0,
            "difficulty": 1,
            "source_doc_id": "doc-1",
            "author": "alice",
            "verifier": "bob",
            "public": True,
            "tags": [],
        }
    )


def _answer(item_id: str, text: str = "Resposta del model.") -> ModelAnswer:
    return ModelAnswer(item_id=item_id, text=text)


def _verdict(correct: bool) -> JudgeVerdict:
    return JudgeVerdict(correct=correct, score=1.0 if correct else 0.0)


def _labelled(item_id: str, human_correct: bool, area: str = "historia") -> CalibrationCase:
    return CalibrationCase(
        item_id=item_id,
        area=area,
        question="q?",
        reference_answer="ref",
        model_answer="ans",
        human_correct=human_correct,
    )


# --- sampling -------------------------------------------------------------


def test_sheet_only_covers_and_obert_items_that_have_an_answer() -> None:
    items = [_obert("o-1"), _obert("o-2"), _mcq("m-1")]
    cases = build_sheet(items, [_answer("o-1")], size=10)
    assert [c.item_id for c in cases] == ["o-1"]


def test_sheet_is_deterministic_and_order_independent() -> None:
    items = [_obert(f"o-{i:02d}") for i in range(20)]
    answers = [_answer(i.id) for i in items]
    a = build_sheet(items, answers, size=8, seed=7)
    b = build_sheet(list(reversed(items)), list(reversed(answers)), size=8, seed=7)
    assert [c.item_id for c in a] == [c.item_id for c in b]


def test_a_different_seed_draws_a_different_sample() -> None:
    items = [_obert(f"o-{i:02d}") for i in range(20)]
    answers = [_answer(i.id) for i in items]
    a = build_sheet(items, answers, size=5, seed=1)
    b = build_sheet(items, answers, size=5, seed=2)
    assert [c.item_id for c in a] != [c.item_id for c in b]


def test_sample_is_stratified_across_areas() -> None:
    items = [_obert(f"h-{i}", area="historia") for i in range(10)]
    items += [_obert(f"g-{i}", area="geografia") for i in range(10)]
    answers = [_answer(i.id) for i in items]
    cases = build_sheet(items, answers, size=6)
    per_area = {"historia": 0, "geografia": 0}
    for case in cases:
        per_area[case.area] += 1
    assert per_area == {"historia": 3, "geografia": 3}


def test_stratification_survives_an_unbalanced_pool() -> None:
    """A thin area must not block the sample from reaching its size."""
    items = [_obert(f"h-{i}", area="historia") for i in range(10)]
    items += [_obert("g-0", area="geografia")]
    answers = [_answer(i.id) for i in items]
    cases = build_sheet(items, answers, size=6)
    assert len(cases) == 6
    assert sum(1 for c in cases if c.area == "geografia") == 1


def test_asking_for_more_than_exists_yields_everything() -> None:
    items = [_obert("o-1"), _obert("o-2")]
    cases = build_sheet(items, [_answer("o-1"), _answer("o-2")], size=50)
    assert len(cases) == 2


def test_sheet_carries_the_context_a_labeller_needs() -> None:
    case = build_sheet(
        [_obert("o-1", reference="La referència real.")], [_answer("o-1", "Diu X.")]
    )[0]
    assert case.question.startswith("Pregunta oberta")
    assert case.reference_answer == "La referència real."
    assert case.model_answer == "Diu X."


def test_sheet_never_exposes_the_judge_verdict() -> None:
    """Blindness is the point: a labeller who sees the verdict anchors to it."""
    case = build_sheet([_obert("o-1")], [_answer("o-1")])[0]
    payload = json.loads(case.model_dump_json())
    assert "correct" not in payload
    assert "score" not in payload
    assert payload["human_correct"] is None


def test_empty_pool_is_rejected() -> None:
    with pytest.raises(ValueError, match="no And-Obert item"):
        build_sheet([_mcq("m-1")], [_answer("m-1")])


@pytest.mark.parametrize("size", [0, -3])
def test_non_positive_size_is_rejected(size: int) -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        build_sheet([_obert("o-1")], [_answer("o-1")], size=size)


def test_sheet_roundtrips_through_jsonl(tmp_path: Path) -> None:
    cases = build_sheet([_obert("o-1")], [_answer("o-1")])
    assert load_sheet(write_sheet(cases, tmp_path / "sheet.jsonl")) == cases


def test_unlabelled_lists_the_rows_still_to_judge() -> None:
    cases = [
        _labelled("o-1", True),
        CalibrationCase(**{**_labelled("o-2", True).model_dump(), "human_correct": None}),
    ]
    assert unlabelled(cases) == ["o-2"]


# --- Cohen's kappa --------------------------------------------------------


def test_kappa_is_one_on_perfect_agreement_with_mixed_labels() -> None:
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)


def test_kappa_is_zero_at_chance() -> None:
    judge = [True, True, False, False]
    human = [True, False, True, False]
    assert cohen_kappa(judge, human) == pytest.approx(0.0)


def test_kappa_is_undefined_when_a_rater_never_varies() -> None:
    """A judge that always says 'correct' earns no credit for agreeing."""
    assert cohen_kappa([True, True, True], [True, True, True]) is None


def test_kappa_is_negative_below_chance() -> None:
    kappa = cohen_kappa([True, False], [False, True])
    assert kappa is not None and kappa < 0


def test_kappa_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        cohen_kappa([True], [True, False])
    with pytest.raises(ValueError, match="zero labels"):
        cohen_kappa([], [])


# --- the gate -------------------------------------------------------------


def _sheet_and_verdicts(
    pairs: list[tuple[bool, bool]],
) -> tuple[list[CalibrationCase], dict[str, JudgeVerdict]]:
    """Build a labelled sheet from (judge_correct, human_correct) pairs."""
    cases = [_labelled(f"o-{i}", human) for i, (_judge, human) in enumerate(pairs)]
    verdicts = {f"o-{i}": _verdict(judge) for i, (judge, _human) in enumerate(pairs)}
    return cases, verdicts


def test_gate_passes_at_or_above_the_p14_bar() -> None:
    # 17/20 = 85% exactly: the bar is inclusive.
    pairs = [(True, True)] * 10 + [(False, False)] * 7 + [(True, False)] * 3
    cases, verdicts = _sheet_and_verdicts(pairs)
    record = calibrate(cases, verdicts, rubric_version="v1.0")
    assert record.agreement == pytest.approx(0.85)
    assert record.ok


def test_gate_fails_below_the_bar_and_names_the_disagreements() -> None:
    pairs = [(True, True)] * 6 + [(True, False)] * 4
    cases, verdicts = _sheet_and_verdicts(pairs)
    record = calibrate(cases, verdicts, rubric_version="v1.0")
    assert record.agreement == pytest.approx(0.6)
    assert not record.ok
    assert record.disagreement_ids == ["o-6", "o-7", "o-8", "o-9"]
    assert "revise the rubric" in record.summary()


def test_confusion_matrix_counts_every_cell() -> None:
    pairs = [(True, True), (True, False), (False, True), (False, False)]
    cases, verdicts = _sheet_and_verdicts(pairs)
    record = calibrate(cases, verdicts, rubric_version="v1.0")
    assert record.confusion.to_dict() == {
        "judge_yes_human_yes": 1,
        "judge_yes_human_no": 1,
        "judge_no_human_yes": 1,
        "judge_no_human_no": 1,
    }
    assert record.confusion.total == 4


def test_a_lenient_judge_is_flagged() -> None:
    pairs = [(True, True)] * 8 + [(True, False)] * 2
    cases, verdicts = _sheet_and_verdicts(pairs)
    record = calibrate(cases, verdicts, rubric_version="v1.0")
    assert any("lenient" in w for w in record.warnings)


def test_a_judge_that_always_says_yes_is_blocked_despite_90_percent_agreement() -> None:
    """The trap the original P14 could not see, now closed (amended v1.1.0)."""
    pairs = [(True, True)] * 9 + [(True, False)]
    cases, verdicts = _sheet_and_verdicts(pairs)
    record = calibrate(cases, verdicts, rubric_version="v1.0")

    assert record.agreement == pytest.approx(0.9)
    assert record.agreement_ok  # the raw-agreement half still passes...
    assert record.kappa == pytest.approx(0.0)  # ...while carrying no information
    assert not record.kappa_ok
    assert not record.ok
    assert "FAIL on kappa" in record.summary()
    assert "near chance" in record.summary()


def test_an_undefined_kappa_blocks_and_blames_the_sample_not_the_judge() -> None:
    """All-one-label proves nothing about catching a wrong answer."""
    pairs = [(True, True)] * 10
    cases, verdicts = _sheet_and_verdicts(pairs)
    record = calibrate(cases, verdicts, rubric_version="v1.0")

    assert record.kappa is None
    assert not record.ok
    summary = record.summary()
    assert "undefined" in summary
    assert "Enlarge the sample" in summary
    assert "the judge is not the problem here" in summary


def test_the_two_kappa_tiers() -> None:
    from andbench.harness.calibration import MIN_KAPPA, SUBSTANTIAL_KAPPA

    assert MIN_KAPPA == 0.41  # Landis-Koch "moderate" floor
    assert SUBSTANTIAL_KAPPA == 0.61
    assert MIN_KAPPA < SUBSTANTIAL_KAPPA


def test_a_kappa_in_the_advisory_band_ships_with_a_caveat() -> None:
    """Clears the floor, below 'substantial': thin evidence, not a blocker.

    Proportions chosen so agreement is 90 % (over the 85 % bar) while the skewed
    marginals hold kappa at 0.44 — the exact situation the two tiers exist for.
    """
    pairs = [(True, True)] * 17 + [(True, False), (False, True), (False, False)]
    cases, verdicts = _sheet_and_verdicts(pairs)
    record = calibrate(cases, verdicts, rubric_version="v1.0")

    assert record.agreement == pytest.approx(0.90)
    assert record.kappa is not None
    assert MIN_KAPPA <= record.kappa < SUBSTANTIAL_KAPPA
    assert record.ok, record.summary()
    assert any("evidence is thin" in w for w in record.warnings)


def test_a_low_agreement_still_fails_on_agreement_first() -> None:
    """Both halves are reported separately so the fix is unambiguous."""
    pairs = [(True, True)] * 5 + [(True, False)] * 5
    cases, verdicts = _sheet_and_verdicts(pairs)
    record = calibrate(cases, verdicts, rubric_version="v1.0")
    assert not record.agreement_ok
    assert "FAIL on agreement" in record.summary()


def test_the_floor_is_configurable_for_a_deliberate_exception() -> None:
    pairs = [(True, True)] * 9 + [(True, False)]
    cases, verdicts = _sheet_and_verdicts(pairs)
    record = calibrate(cases, verdicts, rubric_version="v1.0", min_kappa=0.0)
    assert record.kappa_ok
    assert record.ok


def test_the_record_publishes_both_halves() -> None:
    pairs = [(True, True)] * 9 + [(True, False)]
    cases, verdicts = _sheet_and_verdicts(pairs)
    payload = calibrate(cases, verdicts, rubric_version="v1.0").to_dict()
    assert payload["agreement_ok"] is True
    assert payload["kappa_ok"] is False
    assert payload["ok"] is False
    assert payload["min_kappa"] == 0.41


def test_a_genuinely_skilled_judge_earns_high_kappa_and_no_warnings() -> None:
    pairs = [(True, True)] * 9 + [(False, False)] * 9 + [(True, False)] + [(False, True)]
    cases, verdicts = _sheet_and_verdicts(pairs)
    record = calibrate(cases, verdicts, rubric_version="v1.0")
    assert record.ok
    assert record.agreement_ok and record.kappa_ok
    assert record.kappa is not None and record.kappa > SUBSTANTIAL_KAPPA
    assert record.warnings == []


def test_an_unlabelled_sheet_is_refused() -> None:
    cases = [
        _labelled("o-1", True),
        CalibrationCase(
            item_id="o-2", area="a", question="q", reference_answer="r", model_answer="m"
        ),
    ]
    with pytest.raises(ValueError, match="unlabelled"):
        calibrate(cases, {"o-1": _verdict(True), "o-2": _verdict(True)}, rubric_version="v1.0")


def test_a_case_without_a_verdict_is_refused() -> None:
    cases = [_labelled("o-1", True)]
    with pytest.raises(ValueError, match="no judge verdict"):
        calibrate(cases, {}, rubric_version="v1.0")


def test_an_empty_sheet_is_refused() -> None:
    with pytest.raises(ValueError, match="empty sheet"):
        calibrate([], {}, rubric_version="v1.0")


def test_record_is_tied_to_the_rubric_version_and_seed() -> None:
    cases, verdicts = _sheet_and_verdicts([(True, True)] * 4 + [(False, False)] * 4)
    record = calibrate(cases, verdicts, rubric_version="v2.3", seed=99)
    assert record.rubric_version == "v2.3"
    assert record.seed == 99
    assert "v2.3" in record.summary()


def test_record_is_written_as_sorted_json(tmp_path: Path) -> None:
    cases, verdicts = _sheet_and_verdicts([(True, True)] * 4 + [(False, False)] * 4)
    record = calibrate(cases, verdicts, rubric_version="v1.0")
    payload = json.loads(write_record(record, tmp_path / "rec.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["rubric_version"] == "v1.0"
    assert payload["confusion"]["judge_yes_human_yes"] == 4


def test_the_default_bar_is_the_constitutional_one() -> None:
    assert DEFAULT_MIN_AGREEMENT == 0.85


def test_load_answers_reads_recorded_answers(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(
        json.dumps({"item_id": "o-1", "text": "t", "citations": [], "used_rag": False}) + "\n",
        encoding="utf-8",
    )
    assert load_answers(path) == [ModelAnswer(item_id="o-1", text="t")]
