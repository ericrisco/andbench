"""Tests for the leaderboard builder (B4.01)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from andbench.harness.stats import ItemResult, ScoringMethod
from andbench.leaderboard import (
    SUSPICIOUS_GAP,
    AndObertRow,
    build_leaderboard,
    load_andobert_rows,
    write_leaderboard,
)
from andbench.schema import Item


def _mcq(
    item_id: str, *, track: str = "and-coneix", area: str = "geografia", public: bool = True
) -> Item:
    return Item.model_validate(
        {
            "id": item_id,
            "track": track,
            "area": area,
            "question": f"Pregunta {item_id}?",
            "choices": ["a", "b", "c", "d"],
            "answer": 0,
            "difficulty": 1,
            "source_doc_id": "doc-1",
            "author": "alice",
            "verifier": "bob",
            "public": public,
            "tags": [],
        }
    )


def _obert_item(item_id: str, reference: str = "La referència.") -> Item:
    return Item.model_validate(
        {
            "id": item_id,
            "track": "and-obert",
            "area": "historia",
            "question": "Pregunta oberta?",
            "answer_text": reference,
            "difficulty": 2,
            "source_doc_id": "doc-2",
            "author": "alice",
            "verifier": "bob",
            "public": True,
            "tags": [],
        }
    )


def _result(item_id: str, model: str, correct: bool, seed: int = 1) -> ItemResult:
    return ItemResult(item_id=item_id, model=model, seed=seed, correct=correct)


def _row(item_id: str, model: str, **kwargs: object) -> AndObertRow:
    payload: dict[str, object] = {
        "item_id": item_id,
        "model": model,
        "correct": True,
        "score": 1.0,
    }
    payload.update(kwargs)
    return AndObertRow.model_validate(payload)


# --- accuracy per track and per area --------------------------------------


def test_accuracy_is_reported_per_track_and_per_area() -> None:
    items = [
        _mcq("c-1", track="and-coneix", area="geografia"),
        _mcq("c-2", track="and-coneix", area="historia"),
        _mcq("l-1", track="and-llengua", area="lexic"),
    ]
    results = [
        _result("c-1", "m1", True),
        _result("c-2", "m1", False),
        _result("l-1", "m1", True),
    ]
    board = build_leaderboard(items, results)
    row = board.rows[0]

    assert row.by_track["and-coneix"].accuracy == pytest.approx(0.5)
    assert row.by_track["and-llengua"].accuracy == pytest.approx(1.0)
    assert row.by_area["and-coneix/geografia"].accuracy == pytest.approx(1.0)
    assert row.by_area["and-coneix/historia"].accuracy == pytest.approx(0.0)
    assert row.mcq_overall is not None
    assert row.mcq_overall.accuracy == pytest.approx(2 / 3)


def test_accuracy_micro_averages_across_seeds() -> None:
    items = [_mcq("c-1")]
    results = [_result("c-1", "m1", True, seed=1), _result("c-1", "m1", False, seed=2)]
    board = build_leaderboard(items, results)
    row = board.rows[0]
    assert row.mcq_overall is not None
    assert row.mcq_overall.accuracy == pytest.approx(0.5)
    assert row.mcq_overall.n_results == 2
    assert row.mcq_overall.n_items == 1
    assert row.seeds == (1, 2)


def test_rows_are_ranked_by_overall_accuracy() -> None:
    items = [_mcq("c-1"), _mcq("c-2")]
    results = [
        _result("c-1", "weak", False),
        _result("c-2", "weak", False),
        _result("c-1", "strong", True),
        _result("c-2", "strong", True),
    ]
    board = build_leaderboard(items, results)
    assert [r.model for r in board.rows] == ["strong", "weak"]


def test_open_items_do_not_leak_into_mcq_accuracy() -> None:
    items = [_mcq("c-1"), _obert_item("o-1")]
    results = [_result("c-1", "m1", True), _result("o-1", "m1", False)]
    board = build_leaderboard(items, results)
    row = board.rows[0]
    assert row.mcq_overall is not None
    assert row.mcq_overall.n_items == 1
    assert row.mcq_overall.accuracy == pytest.approx(1.0)


# --- the contamination column (the point of the private split) -------------


def test_public_and_private_accuracy_are_reported_separately() -> None:
    items = [_mcq("pub-1", public=True), _mcq("priv-1", public=False)]
    results = [_result("pub-1", "m1", True), _result("priv-1", "m1", False)]
    board = build_leaderboard(items, results)
    row = board.rows[0]
    assert row.public is not None and row.public.accuracy == pytest.approx(1.0)
    assert row.private is not None and row.private.accuracy == pytest.approx(0.0)
    assert row.contamination_gap == pytest.approx(1.0)


def test_a_wide_public_private_gap_is_flagged() -> None:
    items = [_mcq(f"pub-{i}", public=True) for i in range(10)]
    items += [_mcq(f"priv-{i}", public=False) for i in range(10)]
    results = [_result(f"pub-{i}", "leaky", True) for i in range(10)]
    results += [_result(f"priv-{i}", "leaky", i < 5) for i in range(10)]
    board = build_leaderboard(items, results)
    assert board.rows[0].contamination_gap == pytest.approx(0.5)
    assert any("contamination" in w for w in board.warnings)
    assert "⚠️" in board.to_markdown()


def test_a_narrow_gap_is_not_flagged() -> None:
    items = [_mcq(f"pub-{i}", public=True) for i in range(10)]
    items += [_mcq(f"priv-{i}", public=False) for i in range(10)]
    results = [_result(f"pub-{i}", "clean", i < 8) for i in range(10)]
    results += [_result(f"priv-{i}", "clean", i < 8) for i in range(10)]
    board = build_leaderboard(items, results)
    assert board.rows[0].contamination_gap == pytest.approx(0.0)
    assert not any("contamination" in w for w in board.warnings)


def test_missing_private_results_are_called_out_not_silently_ignored() -> None:
    """Publishing without the private split means the check simply did not happen."""
    items = [_mcq("pub-1", public=True)]
    board = build_leaderboard(items, [_result("pub-1", "m1", True)])
    assert board.rows[0].contamination_gap is None
    assert any("contamination cannot be checked" in w for w in board.warnings)


def test_the_default_gap_threshold_is_documented() -> None:
    assert SUSPICIOUS_GAP == 0.10


# --- And-Obert ------------------------------------------------------------


def test_andobert_metrics_are_computed_per_model() -> None:
    items = [_obert_item("o-1"), _obert_item("o-2")]
    rows = [
        _row("o-1", "m1", correct=True, has_citation=True, cited_correctly=True),
        _row("o-2", "m1", correct=False, score=0.0),
    ]
    board = build_leaderboard(items, [], rows)
    metrics = board.rows[0].andobert
    assert metrics is not None
    assert metrics.factual_accuracy == pytest.approx(0.5)
    assert metrics.citation_precision == pytest.approx(1.0)


def test_honesty_is_credited_on_abstention_items() -> None:
    items = [_obert_item("o-1", reference="no ho sé")]
    board = build_leaderboard(items, [], [_row("o-1", "m1", abstained=True)])
    metrics = board.rows[0].andobert
    assert metrics is not None
    assert metrics.honesty_accuracy == pytest.approx(1.0)


def test_a_model_with_only_andobert_results_still_gets_a_row() -> None:
    board = build_leaderboard([_obert_item("o-1")], [], [_row("o-1", "obert-only")])
    assert [r.model for r in board.rows] == ["obert-only"]
    assert board.rows[0].mcq_overall is None


def test_rag_variants_are_separate_rows() -> None:
    items = [_mcq("c-1")]
    results = [_result("c-1", "maia", False), _result("c-1", "maia+rag", True)]
    board = build_leaderboard(items, results)
    assert [r.model for r in board.rows] == ["maia+rag", "maia"]


def test_andobert_rows_roundtrip_through_jsonl(tmp_path: Path) -> None:
    rows = [_row("o-1", "m1"), _row("o-2", "m1", correct=False, score=0.1)]
    path = tmp_path / "verdicts.jsonl"
    path.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
    assert load_andobert_rows(path) == rows


# --- guards ---------------------------------------------------------------


def test_models_scored_on_different_item_sets_are_not_publishable() -> None:
    items = [_mcq("c-1"), _mcq("c-2")]
    results = [
        _result("c-1", "a", True),
        _result("c-2", "a", True),
        _result("c-1", "b", True),
    ]
    board = build_leaderboard(items, results)
    assert not board.ok
    assert any("not comparable" in p for p in board.problems)
    assert "not fit to publish" in board.to_markdown()


def test_a_result_for_an_unknown_item_is_a_problem() -> None:
    board = build_leaderboard([_mcq("c-1")], [_result("ghost", "m1", True)])
    assert not board.ok
    assert any("ghost" in p for p in board.problems)


def test_an_andobert_verdict_for_an_unknown_item_is_a_problem() -> None:
    board = build_leaderboard([_obert_item("o-1")], [], [_row("ghost", "m1")])
    assert not board.ok
    assert any("ghost" in p for p in board.problems)


def test_no_results_at_all_is_a_problem() -> None:
    board = build_leaderboard([_mcq("c-1")], [])
    assert not board.ok
    assert any("no results" in p for p in board.problems)
    assert board.rows == []


def test_a_single_seed_is_warned_about() -> None:
    board = build_leaderboard([_mcq("c-1")], [_result("c-1", "m1", True)])
    assert any("single seed" in w for w in board.warnings)


def test_several_seeds_earn_no_seed_warning() -> None:
    results = [_result("c-1", "m1", True, seed=s) for s in (1, 2, 3)]
    board = build_leaderboard([_mcq("c-1")], results)
    assert not any("single seed" in w for w in board.warnings)


# --- rendering ------------------------------------------------------------


def test_markdown_has_a_row_per_model_and_a_column_per_track() -> None:
    items = [_mcq("c-1", track="and-coneix"), _mcq("l-1", track="and-llengua")]
    results = [
        _result("c-1", "m1", True),
        _result("l-1", "m1", True),
        _result("c-1", "m2", False),
        _result("l-1", "m2", False),
    ]
    md = build_leaderboard(items, results).to_markdown()
    assert "| Model | And-Coneix | And-Llengua | And-Cotidià | MCQ overall" in md
    assert "| m1 |" in md
    assert "| m2 |" in md


def test_markdown_includes_a_per_area_table_for_each_track_present() -> None:
    items = [
        _mcq("c-1", track="and-coneix", area="geografia"),
        _mcq("c-2", track="and-coneix", area="historia"),
    ]
    results = [_result("c-1", "m1", True), _result("c-2", "m1", False)]
    md = build_leaderboard(items, results).to_markdown()
    assert "#### And-Coneix by area" in md
    assert "geografia" in md and "historia" in md
    assert "#### And-Llengua by area" not in md


def test_missing_cells_render_as_a_dash_not_a_zero() -> None:
    """An unmeasured cell must never look like a score of 0 %."""
    md = build_leaderboard([_mcq("c-1")], [_result("c-1", "m1", True)]).to_markdown()
    assert "—" in md


def test_summary_names_each_model_and_its_gap() -> None:
    items = [_mcq("pub-1", public=True), _mcq("priv-1", public=False)]
    results = [_result("pub-1", "m1", True), _result("priv-1", "m1", True)]
    summary = build_leaderboard(items, results).summary()
    assert "m1: MCQ 100.0%" in summary
    assert "public-private +0.0%" in summary


def test_artifacts_are_written(tmp_path: Path) -> None:
    board = build_leaderboard([_mcq("c-1")], [_result("c-1", "m1", True)])
    paths = write_leaderboard(board, tmp_path / "b.json", tmp_path / "b.md")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["rows"][0]["model"] == "m1"
    assert payload["ok"] is True
    assert paths["markdown"].read_text(encoding="utf-8").startswith("| Model |")


def test_summary_lists_problems_when_the_table_is_unpublishable() -> None:
    board = build_leaderboard([_mcq("c-1")], [])
    assert "NOT publishable" in board.summary()


# --- scoring methods must not be mixed (B4.01 gap) ------------------------


def _res(item_id: str, model: str, correct: bool, method: object = None) -> ItemResult:
    payload: dict[str, object] = {
        "item_id": item_id,
        "model": model,
        "seed": 1,
        "correct": correct,
    }
    if method is not None:
        payload["scoring_method"] = method
    return ItemResult.model_validate(payload)


def test_a_single_scoring_method_is_recorded_on_the_row() -> None:
    board = build_leaderboard([_mcq("c-1")], [_res("c-1", "m1", True, "generative")])
    assert board.ok
    assert board.rows[0].scoring_method is ScoringMethod.GENERATIVE
    assert board.to_dict()["rows"][0]["scoring_method"] == "generative"  # type: ignore[index]


def test_mixing_loglikelihood_and_generative_is_refused() -> None:
    """They measure different things; one column holding both is not a ranking."""
    items = [_mcq("c-1")]
    results = [
        _res("c-1", "gemma", True, "loglikelihood"),
        _res("c-1", "gpt", True, "generative"),
    ]
    board = build_leaderboard(items, results)
    assert not board.ok
    assert any("not comparable" in p for p in board.problems)
    assert "not fit to publish" in board.to_markdown()


def test_mixing_a_recorded_method_with_an_unrecorded_one_is_refused() -> None:
    """Unknown is not 'probably the same' — that is how a mixture slips through."""
    items = [_mcq("c-1")]
    results = [_res("c-1", "a", True, "generative"), _res("c-1", "b", True)]
    board = build_leaderboard(items, results)
    assert not board.ok
    assert any("unrecorded" in p for p in board.problems)


def test_all_unrecorded_is_allowed_so_legacy_files_still_publish() -> None:
    board = build_leaderboard([_mcq("c-1")], [_res("c-1", "m1", True)])
    assert board.ok
    assert board.rows[0].scoring_method is None


# --- the judge must not share a lab with a graded model -------------------


def test_a_judge_sharing_a_lab_with_an_evaluated_model_is_flagged() -> None:
    items = [_mcq("c-1")]
    results = [_res("c-1", "openai/gpt-5.4", True), _res("c-1", "google/gemma-4-31b-it", True)]
    board = build_leaderboard(items, results, judge_model="openai/gpt-5.6-luna")
    assert any("self-preference" in w and "openai/gpt-5.4" in w for w in board.warnings)
    assert not any("gemma" in w for w in board.warnings if "self-preference" in w)


def test_no_conflict_when_the_judge_lab_is_absent() -> None:
    items = [_mcq("c-1")]
    results = [_res("c-1", "google/gemma-4-31b-it", True)]
    board = build_leaderboard(items, results, judge_model="openai/gpt-5.6-luna")
    assert not any("self-preference" in w for w in board.warnings)


def test_the_conflict_reaches_the_published_caveats() -> None:
    items = [_mcq("c-1")]
    results = [_res("c-1", "openai/gpt-5.4", True)]
    board = build_leaderboard(items, results, judge_model="openai/gpt-5.6-luna")
    assert "self-preference" in board.to_markdown()


def test_conflicts_are_only_checked_when_a_judge_is_named() -> None:
    board = build_leaderboard([_mcq("c-1")], [_res("c-1", "openai/gpt-5.4", True)])
    assert not any("self-preference" in w for w in board.warnings)


def test_lab_of_treats_a_bare_name_as_its_own_lab() -> None:
    from andbench.leaderboard import lab_of, same_lab_conflicts

    assert lab_of("local-gemma") == "local-gemma"
    assert same_lab_conflicts("openai/x", ["local-gemma"]) == []
