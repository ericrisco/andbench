"""Tests for the And-Obert migration (B2.01)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from andbench.ingest import (
    MIGRATED_TAG,
    FieldMap,
    IngestCandidate,
    item_id_for,
    load_queue,
    load_raw,
    pending,
    promote,
    to_candidates,
    write_items,
    write_queue,
)
from andbench.schema import Track


def _records(n: int = 2) -> list[dict[str, object]]:
    return [
        {
            "question": f"Pregunta migrada {i}?",
            "answer": f"Resposta de referència {i}.",
            "doc": f"pool_bench/doc-{i}.md",
        }
        for i in range(n)
    ]


def _convert(
    records: list[dict[str, object]], **kwargs: object
) -> tuple[list[IngestCandidate], list[str]]:
    defaults: dict[str, object] = {
        "origin": "andorraqa",
        "field_map": FieldMap(source_doc_id="doc"),
        "default_area": "historia",
        "default_author": "po",
    }
    defaults.update(kwargs)
    return to_candidates(records, **defaults)  # type: ignore[arg-type]


def _reviewed(candidate: IngestCandidate, verifier: str = "verifier-1") -> IngestCandidate:
    return candidate.model_copy(update={"accepted": True, "verifier": verifier})


# --- reading the external file --------------------------------------------


def test_jsonl_is_read(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _records(3)) + "\n", encoding="utf-8")
    assert len(load_raw(path)) == 3


def test_a_json_array_is_read_too(tmp_path: Path) -> None:
    """Exports arrive in both shapes; a migration should not care which."""
    path = tmp_path / "qa.json"
    path.write_text(json.dumps(_records(2)), encoding="utf-8")
    assert len(load_raw(path)) == 2


def test_an_empty_file_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("  \n", encoding="utf-8")
    assert load_raw(path) == []


def test_malformed_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": 1}\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_raw(path)


def test_a_non_object_jsonl_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('"just a string"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="each line must be a JSON object"):
        load_raw(path)


def test_a_json_array_of_non_objects_is_rejected_with_its_index(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('[{"question": "q"}, 42]', encoding="utf-8")
    with pytest.raises(ValueError, match="record 1 must be a JSON object"):
        load_raw(path)


def test_a_json_export_that_is_not_an_array_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '["a"]'.replace('["a"]', '[{"question":"q","answer":"a","doc":"d"}]'), encoding="utf-8"
    )
    assert len(load_raw(path)) == 1


# --- the field map --------------------------------------------------------


def test_field_map_parses_a_cli_spec() -> None:
    mapping = FieldMap.parse("question=pregunta,answer=resposta,area=tema")
    assert mapping.question == "pregunta"
    assert mapping.answer == "resposta"
    assert mapping.area == "tema"


def test_field_map_tolerates_trailing_commas() -> None:
    assert FieldMap.parse("question=q,").question == "q"


def test_field_map_rejects_an_unknown_field() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        FieldMap.parse("nonsense=x")


def test_field_map_rejects_a_missing_key() -> None:
    with pytest.raises(ValueError, match="field=key"):
        FieldMap.parse("question=")


def test_field_map_defaults_match_the_common_shape() -> None:
    assert FieldMap().question == "question"
    assert FieldMap().answer == "answer"


# --- conversion -----------------------------------------------------------


def test_records_become_candidates_awaiting_review() -> None:
    candidates, errors = _convert(_records(2))
    assert errors == []
    assert len(candidates) == 2
    assert all(c.verifier == "" for c in candidates)
    assert all(c.accepted is False for c in candidates)


def test_ids_are_derived_from_the_question_so_reimports_are_stable() -> None:
    records = _records(3)
    first, _ = _convert(records)
    reordered, _ = _convert(list(reversed(records)))
    assert [c.item_id for c in first] == [c.item_id for c in reordered]


def test_the_id_helper_is_pure() -> None:
    assert item_id_for("andorraqa", "Què?") == item_id_for("andorraqa", " què? ")
    assert item_id_for("a", "Q") != item_id_for("b", "Q")


def test_a_repeated_question_is_reported_as_a_duplicate() -> None:
    records = _records(1) * 2
    candidates, errors = _convert(records)
    assert len(candidates) == 1
    assert any("duplicate question" in e for e in errors)


def test_a_record_without_a_question_is_reported_not_dropped() -> None:
    candidates, errors = _convert([{"answer": "a", "doc": "d"}])
    assert candidates == []
    assert any("no question" in e for e in errors)


def test_a_record_without_a_reference_answer_is_reported() -> None:
    candidates, errors = _convert([{"question": "q", "doc": "d"}])
    assert candidates == []
    assert any("cannot be judged" in e for e in errors)


def test_a_record_without_a_source_is_refused() -> None:
    """P8: every item cites a verification source, so this cannot be defaulted away."""
    candidates, errors = _convert([{"question": "q", "answer": "a"}])
    assert candidates == []
    assert any("P8" in e for e in errors)


def test_a_default_source_can_be_supplied_for_a_whole_migration() -> None:
    candidates, errors = _convert(
        [{"question": "q", "answer": "a"}],
        field_map=FieldMap(),
        default_source_doc_id="maia/test-split.jsonl",
    )
    assert errors == []
    assert candidates[0].source_doc_id == "maia/test-split.jsonl"


def test_difficulty_is_mapped_when_present() -> None:
    candidates, errors = _convert(
        [{"question": "q", "answer": "a", "doc": "d", "lvl": 3}],
        field_map=FieldMap(source_doc_id="doc", difficulty="lvl"),
    )
    assert errors == []
    assert candidates[0].difficulty == 3


def test_a_non_numeric_difficulty_is_reported() -> None:
    candidates, errors = _convert(
        [{"question": "q", "answer": "a", "doc": "d", "lvl": "hard"}],
        field_map=FieldMap(source_doc_id="doc", difficulty="lvl"),
    )
    assert candidates == []
    assert any("not an integer" in e for e in errors)


def test_an_out_of_range_difficulty_is_reported() -> None:
    candidates, errors = _convert(
        [{"question": "q", "answer": "a", "doc": "d", "lvl": 9}],
        field_map=FieldMap(source_doc_id="doc", difficulty="lvl"),
    )
    assert candidates == []
    assert any("validation error" in e for e in errors)


def test_unmapped_extra_keys_are_simply_ignored() -> None:
    candidates, errors = _convert([{**_records(1)[0], "irrelevant": "noise"}])
    assert errors == []
    assert len(candidates) == 1


def test_queue_roundtrips(tmp_path: Path) -> None:
    candidates, _ = _convert(_records(2))
    assert load_queue(write_queue(candidates, tmp_path / "q.jsonl")) == candidates


# --- the P8 gate: verification is never fabricated ------------------------


def test_nothing_is_promoted_before_a_human_accepts() -> None:
    candidates, _ = _convert(_records(2))
    items, blocked = promote(candidates)
    assert items == []
    assert all("not accepted yet" in b for b in blocked)


def test_an_accepted_candidate_without_a_verifier_is_not_promoted() -> None:
    candidates, _ = _convert(_records(1))
    accepted = candidates[0].model_copy(update={"accepted": True})
    items, blocked = promote([accepted])
    assert items == []
    assert any("no verifier assigned" in b for b in blocked)


def test_a_verifier_who_is_the_author_is_refused() -> None:
    """The whole point of P8: self-verification is not verification."""
    candidates, _ = _convert(_records(1))
    items, blocked = promote([_reviewed(candidates[0], verifier="PO")])  # default author is "po"
    assert items == []
    assert any("same person as the author" in b for b in blocked)


def test_a_fully_reviewed_candidate_becomes_an_and_obert_item() -> None:
    candidates, _ = _convert(_records(1))
    items, blocked = promote([_reviewed(candidates[0])])
    assert blocked == []
    assert len(items) == 1
    item = items[0]
    assert item.track is Track.AND_OBERT
    assert item.answer_text == "Resposta de referència 0."
    assert item.verifier == "verifier-1"
    assert item.choices is None


def test_promoted_items_are_tagged_with_their_origin() -> None:
    candidates, _ = _convert(_records(1))
    items, _ = promote([_reviewed(candidates[0])])
    assert MIGRATED_TAG in items[0].tags
    assert "andorraqa" in items[0].tags


def test_a_partially_reviewed_queue_promotes_only_the_finished_rows() -> None:
    candidates, _ = _convert(_records(3))
    mixed = [_reviewed(candidates[0]), candidates[1], _reviewed(candidates[2])]
    items, blocked = promote(mixed)
    assert len(items) == 2
    assert len(blocked) == 1


def test_promotion_can_target_the_private_split() -> None:
    candidates, _ = _convert(_records(1))
    items, _ = promote([_reviewed(candidates[0])], public=False)
    assert items[0].public is False


def test_source_url_is_carried_when_present() -> None:
    candidates, errors = _convert(
        [{"question": "q", "answer": "a", "doc": "d", "url": "https://example.org/andorra/x"}],
        field_map=FieldMap(source_doc_id="doc", source_url="url"),
    )
    assert errors == []
    items, _ = promote([_reviewed(candidates[0])])
    assert items[0].source_url is not None
    assert "example.org" in str(items[0].source_url)


def test_promoted_items_are_sorted_by_id() -> None:
    candidates, _ = _convert(_records(5))
    items, _ = promote([_reviewed(c) for c in candidates])
    assert [i.id for i in items] == sorted(i.id for i in items)


def test_pending_lists_every_unfinished_reason() -> None:
    candidates, _ = _convert(_records(3))
    mixed = [
        candidates[0],
        candidates[1].model_copy(update={"accepted": True}),
        _reviewed(candidates[2]),
    ]
    assert len(pending(mixed)) == 2


def test_promoted_items_write_as_a_valid_andbench_file(tmp_path: Path) -> None:
    from andbench.validation import validate_jsonl

    candidates, _ = _convert(_records(3))
    items, _ = promote([_reviewed(c) for c in candidates])
    path = write_items(items, tmp_path / "items.jsonl")
    report = validate_jsonl(path)
    assert report.ok, report.summary()
    assert len(report.items) == 3


def test_a_json_object_export_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]".replace("[]", "[\n]"), encoding="utf-8")
    assert load_raw(path) == []


def test_blank_lines_between_records_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "gaps.jsonl"
    body = "\n\n".join(json.dumps(r) for r in _records(2))
    path.write_text(body + "\n", encoding="utf-8")
    assert len(load_raw(path)) == 2


def test_a_candidate_that_cannot_become_an_item_is_held_back_not_dropped() -> None:
    """A queue edited into an unschematic state must surface, not vanish."""
    candidates, _ = _convert(_records(1))
    broken = _reviewed(candidates[0]).model_copy(update={"source_url": "not-a-url"})
    items, blocked = promote([broken])
    assert items == []
    assert any("schema error" in b for b in blocked)


def test_an_unmapped_source_url_is_left_empty() -> None:
    candidates, errors = _convert(_records(1))
    assert errors == []
    items, _ = promote([_reviewed(candidates[0])])
    assert items[0].source_url is None
