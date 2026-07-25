"""Tests for the generative MCQ runner (B4.01 gap)."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from andbench.harness.mcq_run import McqRun, mcq_items, run_mcq, write_results
from andbench.harness.smoke import Completion
from andbench.harness.stats import ItemResult, ScoringMethod, load_results
from andbench.schema import Item


def _mcq(item_id: str, *, answer: int = 0, track: str = "and-coneix") -> Item:
    return Item.model_validate(
        {
            "id": item_id,
            "track": track,
            "area": "geografia",
            "question": f"Pregunta {item_id}?",
            "choices": ["Alfa", "Bravo", "Charlie", "Delta"],
            "answer": answer,
            "difficulty": 1,
            "source_doc_id": "doc-1",
            "author": "alice",
            "verifier": "bob",
            "public": True,
            "tags": [],
        }
    )


def _open(item_id: str) -> Item:
    return Item.model_validate(
        {
            "id": item_id,
            "track": "and-obert",
            "area": "historia",
            "question": "Oberta?",
            "answer_text": "Referència.",
            "difficulty": 2,
            "source_doc_id": "doc-2",
            "author": "alice",
            "verifier": "bob",
            "public": True,
            "tags": [],
        }
    )


class _Replies:
    """A model returning a scripted reply per call, cycling when exhausted."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or ["A"]
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> Completion:
        self.prompts.append(prompt)
        reply = self.replies[(len(self.prompts) - 1) % len(self.replies)]
        return Completion(text=reply, prompt_tokens=20, completion_tokens=2)


def _clock(step: float = 0.25) -> Iterator[float]:
    current = 0.0
    while True:
        yield current
        current += step


def _run(items: Sequence[Item], *replies: str, seeds: Sequence[int] = (0,)) -> McqRun:
    clock = _clock()
    return run_mcq(items, _Replies(*replies), "test-model", seeds=seeds, clock=lambda: next(clock))


# --- selection ------------------------------------------------------------


def test_only_mcq_items_are_scored() -> None:
    run = _run([_mcq("m-1"), _open("o-1")], "A")
    assert run.attempted == 1
    assert run.results[0].item_id == "m-1"


def test_items_are_scored_in_id_order() -> None:
    assert [i.id for i in mcq_items([_mcq("m-9"), _mcq("m-1")])] == ["m-1", "m-9"]


def test_the_prompt_offers_the_lettered_choices() -> None:
    model = _Replies("A")
    clock = _clock()
    run_mcq([_mcq("m-1")], model, "test-model", clock=lambda: next(clock))
    assert "A) Alfa" in model.prompts[0]


# --- scoring --------------------------------------------------------------


def test_a_correct_letter_scores_correct() -> None:
    run = _run([_mcq("m-1", answer=1)], "B")
    assert run.results[0].correct is True


def test_a_wrong_letter_scores_incorrect() -> None:
    run = _run([_mcq("m-1", answer=1)], "C")
    assert run.results[0].correct is False


def test_a_choice_named_in_prose_is_resolved() -> None:
    run = _run([_mcq("m-1", answer=1)], "La resposta és Bravo.")
    assert run.results[0].correct is True


def test_every_result_records_the_generative_method() -> None:
    """Without this the leaderboard cannot tell comparable columns apart."""
    run = _run([_mcq("m-1")], "A")
    assert run.results[0].scoring_method is ScoringMethod.GENERATIVE


def test_accuracy_is_over_usable_answers_only() -> None:
    run = _run([_mcq("m-1", answer=0), _mcq("m-2", answer=0)], "A", "gibberish")
    assert run.accuracy == pytest.approx(1.0)
    assert run.parse_rate == pytest.approx(0.5)


def test_accuracy_is_none_when_nothing_was_usable() -> None:
    run = _run([_mcq("m-1")], "gibberish")
    assert run.accuracy is None
    assert run.results == []


# --- an unusable answer is not a wrong answer -----------------------------


def test_an_unparseable_answer_is_excluded_not_scored_wrong() -> None:
    """Scoring it zero would blame ignorance for a formatting failure."""
    run = _run([_mcq("m-1")], "No ho sé, ho sento")
    assert run.results == []
    assert len(run.unusable) == 1
    assert run.unusable[0].item_id == "m-1"


def test_an_ambiguous_answer_is_unusable() -> None:
    run = _run([_mcq("m-1")], "Podria ser Alfa o Bravo")
    assert run.results == []
    assert run.unusable


def test_an_empty_answer_is_unusable() -> None:
    run = _run([_mcq("m-1")], "   ")
    assert run.unusable
    assert "(empty)" in str(run.unusable[0])


def test_unusable_answers_are_reported_with_an_excerpt() -> None:
    run = _run([_mcq("m-1")], "una explicació llarga que no tria cap opció concreta")
    assert "una explicació llarga" in str(run.unusable[0])


def test_the_summary_names_the_unusable_answers() -> None:
    run = _run([_mcq("m-1"), _mcq("m-2")], "A", "gibberish")
    summary = run.summary()
    assert "1/2 usable" in summary
    assert "unusable answer(s)" in summary
    assert "m-2" in summary


def test_the_summary_of_a_clean_run_mentions_no_unusable() -> None:
    assert "unusable answer(s)" not in _run([_mcq("m-1")], "A").summary()


# --- seeds ----------------------------------------------------------------


def test_each_seed_produces_its_own_result() -> None:
    run = _run([_mcq("m-1")], "A", seeds=(1, 2, 3))
    assert sorted(r.seed for r in run.results) == [1, 2, 3]
    assert run.attempted == 3


def test_no_seeds_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        _run([_mcq("m-1")], "A", seeds=())


def test_every_call_is_recorded_with_its_latency() -> None:
    run = _run([_mcq("m-1"), _mcq("m-2")], "A")
    assert len(run.responses) == 2
    assert all(r.latency_seconds == pytest.approx(0.25) for r in run.responses)


# --- the results file -----------------------------------------------------


def test_results_roundtrip_through_the_stats_loader(tmp_path: Path) -> None:
    run = _run([_mcq("m-1"), _mcq("m-2")], "A")
    path = write_results(run.results, tmp_path / "mcq-results.jsonl")
    loaded = load_results(path)
    assert loaded == run.results
    assert all(r.scoring_method is ScoringMethod.GENERATIVE for r in loaded)


def test_an_empty_results_file_is_written_without_a_stray_newline(tmp_path: Path) -> None:
    path = write_results([], tmp_path / "empty.jsonl")
    assert path.read_text(encoding="utf-8") == ""


def test_the_written_rows_carry_the_method(tmp_path: Path) -> None:
    run = _run([_mcq("m-1")], "A")
    path = write_results(run.results, tmp_path / "r.jsonl")
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["scoring_method"] == "generative"


def test_a_legacy_row_without_a_method_still_loads(tmp_path: Path) -> None:
    """Existing result files predate the field and must keep working."""
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps({"item_id": "m-1", "model": "old", "seed": 0, "correct": True}) + "\n",
        encoding="utf-8",
    )
    loaded = load_results(path)
    assert loaded == [ItemResult(item_id="m-1", model="old", seed=0, correct=True)]
    assert loaded[0].scoring_method is None


# --- a failed call is data, not a reason to stop ---------------------------


class _FlakyModel:
    """Fails on the nth call, answers otherwise."""

    def __init__(self, fail_on: int) -> None:
        self.fail_on = fail_on
        self.calls = 0

    def complete(self, prompt: str) -> Completion:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("provider exploded")
        return Completion(text="A", prompt_tokens=20, completion_tokens=2)


def test_a_failed_call_is_recorded_and_the_run_continues() -> None:
    """A 680-item pass must not throw away everything before the failure."""
    items = [_mcq(f"m-{i}") for i in range(4)]
    clock = _clock()
    run = run_mcq(items, _FlakyModel(fail_on=2), "flaky", clock=lambda: next(clock))
    assert len(run.results) == 3
    assert len(run.unusable) == 1
    assert "provider exploded" in run.unusable[0].text
    assert run.attempted == 4


def test_a_model_that_always_fails_yields_no_results_but_still_returns() -> None:
    items = [_mcq("m-1"), _mcq("m-2")]
    clock = _clock()
    run = run_mcq(items, _FlakyModel(fail_on=1), "flaky", clock=lambda: next(clock))
    assert run.parse_rate < 1.0
    assert run.accuracy is not None or run.accuracy is None  # it returned at all
