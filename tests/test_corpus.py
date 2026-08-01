"""Tests for corpus ingestion, provenance and chunking."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from andbench.card import Permission
from andbench.corpus import (
    DEFAULT_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    POOL_BENCH,
    POOL_TRAIN,
    CorpusDocument,
    chunk_corpus,
    chunk_document,
    iter_pool,
    licence_warnings,
    load_documents,
    load_passages,
    load_pools,
    passage_id_for,
    split_text,
    summarise,
    write_documents,
    write_manifest,
    write_passages,
)


def _doc(
    doc_id: str = "bopa-001",
    *,
    text: str | None = None,
    source: str = "bopa",
    topic: str = "institucions",
    permission: str = "open-licence",
) -> CorpusDocument:
    return CorpusDocument(
        doc_id=doc_id,
        source=source,
        topic=topic,
        title="Un document",
        url="https://example.org/doc",
        licence="CC-BY-4.0",
        permission=Permission(permission),
        retrieved=date(2026, 8, 1),
        text=text or ("Un paràgraf prou llarg per a superar el mínim. " * 5),
    )


# --- provenance ------------------------------------------------------------


def test_a_document_carries_everything_needed_to_publish_from_it() -> None:
    document = _doc()
    assert document.licence == "CC-BY-4.0"
    assert document.url
    assert document.retrieved == date(2026, 8, 1)
    assert document.permission is Permission.OPEN_LICENCE


def test_permission_defaults_to_pending() -> None:
    """Unknown provenance must not read as cleared."""
    minimal = CorpusDocument(doc_id="d", source="s", topic="t", text="prou text llarg aquí.")
    assert minimal.permission is Permission.PENDING
    assert not minimal.permission.publishable


def test_the_manifest_entry_is_what_the_partition_consumes() -> None:
    entry = _doc().manifest_entry()
    assert (entry.doc_id, entry.source, entry.topic) == ("bopa-001", "bopa", "institucions")


def test_documents_roundtrip_through_jsonl(tmp_path: Path) -> None:
    documents = [_doc("a"), _doc("b")]
    assert load_documents(write_documents(documents, tmp_path / "corpus.jsonl")) == documents


def test_a_malformed_document_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"doc_id": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_documents(path)


def test_the_manifest_file_feeds_the_partition(tmp_path: Path) -> None:
    from andbench.partition import load_manifest

    path = write_manifest([_doc("a"), _doc("b")], tmp_path / "manifest.jsonl")
    assert [d.doc_id for d in load_manifest(path)] == ["a", "b"]


def test_licence_warnings_name_the_sources_that_would_block_a_release() -> None:
    documents = [_doc("a", permission="pending"), _doc("b", permission="pending")]
    warnings = licence_warnings(documents)
    assert len(warnings) == 1
    assert "2 document(s)" in warnings[0]
    assert "blocked at card time" in warnings[0]


def test_permitted_sources_raise_no_warning() -> None:
    assert licence_warnings([_doc(permission="open-licence")]) == []


# --- chunking --------------------------------------------------------------


def test_paragraphs_are_kept_whole_when_they_fit() -> None:
    text = "\n\n".join(["Paràgraf " + "x" * 200] * 3)
    chunks = split_text(text, chunk_chars=1000)
    assert len(chunks) == 1
    assert chunks[0].count("\n\n") == 2


def test_the_budget_starts_a_new_passage() -> None:
    text = "\n\n".join(["Paràgraf " + "x" * 400] * 4)
    chunks = split_text(text, chunk_chars=900)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_a_paragraph_longer_than_the_budget_is_split_on_its_own() -> None:
    chunks = split_text("y" * 3000, chunk_chars=1000, overlap_chars=100)
    assert len(chunks) >= 3


def test_overlap_carries_context_across_a_forced_split() -> None:
    chunks = split_text("z" * 2000, chunk_chars=1000, overlap_chars=200)
    assert chunks[1].startswith("z")
    assert len(chunks) >= 2


def test_headings_and_stray_lines_are_dropped() -> None:
    text = "Títol\n\n" + "Cos del document prou llarg per a passar el mínim. " * 5
    chunks = split_text(text)
    assert all(len(c) >= MIN_CHUNK_CHARS for c in chunks)


def test_a_document_shorter_than_the_floor_is_kept_not_lost() -> None:
    """A short source should be visibly present, not silently vanish."""
    chunks = split_text("Text curt.")
    assert chunks == ["Text curt."]


def test_empty_text_yields_nothing() -> None:
    assert split_text("   \n\n   ") == []


def test_whitespace_is_normalised() -> None:
    chunks = split_text("Hi   ha    espais.\n\n" + "x" * 200)
    assert "Hi ha espais." in chunks[0]


@pytest.mark.parametrize(
    ("chunk_chars", "overlap"), [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)]
)
def test_nonsensical_chunking_parameters_are_rejected(chunk_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("text", chunk_chars=chunk_chars, overlap_chars=overlap)


def test_the_default_budget_is_reviewable_by_a_human() -> None:
    """A passage a verifier cannot check at a glance defeats the purpose."""
    assert 500 <= DEFAULT_CHUNK_CHARS <= 2000


# --- passage identity ------------------------------------------------------


def test_a_passage_id_names_its_document_and_position() -> None:
    passage_id = passage_id_for("bopa-001", 3, "text")
    assert passage_id.startswith("bopa-001#0003-")


def test_the_id_changes_when_the_text_changes() -> None:
    """Otherwise an edit silently repoints existing items at different text."""
    assert passage_id_for("d", 0, "original") != passage_id_for("d", 0, "editat")


def test_the_id_is_stable_for_the_same_text() -> None:
    assert passage_id_for("d", 0, " text ") == passage_id_for("d", 0, "text")


# --- pools -----------------------------------------------------------------


def test_a_passage_carries_its_pool() -> None:
    passages = chunk_document(_doc(), POOL_BENCH)
    assert passages
    assert all(p.pool == POOL_BENCH for p in passages)


def test_an_unknown_pool_is_rejected() -> None:
    with pytest.raises(ValueError, match="pool must be"):
        chunk_document(_doc(), "somewhere-else")


def test_a_document_outside_the_partition_is_skipped_not_guessed() -> None:
    """Guessing a pool is how training text ends up under an item."""
    passages, problems = chunk_corpus([_doc("known"), _doc("orphan")], {"known": POOL_BENCH})
    assert {p.doc_id for p in passages} == {"known"}
    assert len(problems) == 1
    assert "P9" in problems[0]
    assert "orphan" in problems[0]


def test_chunking_is_ordered_so_two_runs_agree() -> None:
    documents = [_doc("b"), _doc("a")]
    pools = {"a": POOL_BENCH, "b": POOL_TRAIN}
    first, _ = chunk_corpus(documents, pools)
    second, _ = chunk_corpus(list(reversed(documents)), pools)
    assert [p.passage_id for p in first] == [p.passage_id for p in second]


def test_iter_pool_selects_one_side() -> None:
    passages, _ = chunk_corpus([_doc("a"), _doc("b")], {"a": POOL_BENCH, "b": POOL_TRAIN})
    assert {p.doc_id for p in iter_pool(passages, POOL_BENCH)} == {"a"}


def test_pools_are_read_from_the_partition_output(tmp_path: Path) -> None:
    (tmp_path / "pool_train.txt").write_text("t1\nt2\n", encoding="utf-8")
    (tmp_path / "pool_bench.txt").write_text("b1\n", encoding="utf-8")
    assert load_pools(tmp_path) == {"t1": POOL_TRAIN, "t2": POOL_TRAIN, "b1": POOL_BENCH}


def test_a_missing_partition_says_which_command_makes_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="andbench partition"):
        load_pools(tmp_path)


# --- artifacts -------------------------------------------------------------


def test_passages_roundtrip(tmp_path: Path) -> None:
    passages = chunk_document(_doc(), POOL_BENCH)
    assert load_passages(write_passages(passages, tmp_path / "p.jsonl")) == passages


def test_the_summary_separates_the_two_pools() -> None:
    documents = [_doc("a"), _doc("b")]
    passages, _ = chunk_corpus(documents, {"a": POOL_BENCH, "b": POOL_TRAIN})
    line = summarise(documents, passages)
    assert "2 document(s)" in line
    assert "bench" in line and "train" in line
