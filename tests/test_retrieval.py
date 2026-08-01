"""Tests for local retrieval. The pool rule is the one that matters."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from andbench.corpus import POOL_BENCH, POOL_TRAIN, Passage
from andbench.retrieval import (
    DEFAULT_TOP_K,
    INDEX_VERSION,
    CorpusIndex,
    PoolViolationError,
    build_index,
    index_path,
    load_index,
    normalise,
    retrieve,
    retrieve_for_authoring,
    write_index,
)


class _AxisEmbedder:
    """Maps a keyword to a basis vector, so similarity is exactly controllable."""

    KEYWORDS = ("parlament", "muntanya", "cuina")

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = [1.0 if word in lowered else 0.0 for word in self.KEYWORDS]
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


def _passage(passage_id: str, text: str, *, pool: str = POOL_BENCH, ordinal: int = 0) -> Passage:
    return Passage(
        passage_id=passage_id,
        doc_id=passage_id.split("#")[0],
        source="bopa",
        topic="institucions",
        pool=pool,
        ordinal=ordinal,
        text=text,
    )


def _bench_passages() -> list[Passage]:
    return [
        _passage("d1#0000", "El parlament aprova el pressupost."),
        _passage("d2#0000", "La muntanya supera els dos mil metres."),
        _passage("d3#0000", "La cuina tradicional fa servir trinxat."),
    ]


def _index(passages: list[Passage] | None = None, pool: str = POOL_BENCH) -> CorpusIndex:
    return build_index(passages or _bench_passages(), _AxisEmbedder(), pool=pool)


# --- the pool rule ---------------------------------------------------------


def test_an_index_cannot_hold_a_passage_from_another_pool() -> None:
    """P9 made structural: the wrong passage is unreachable, not just unwanted."""
    with pytest.raises(PoolViolationError, match="structural rather than a filter"):
        CorpusIndex(
            pool=POOL_BENCH,
            model="m",
            passages=[_passage("d#0000", "text", pool=POOL_TRAIN)],
            vectors=[[1.0]],
        )


def test_building_a_bench_index_drops_training_passages() -> None:
    mixed = [
        _passage("b#0000", "El parlament aprova."),
        _passage("t#0000", "El parlament entrena.", pool=POOL_TRAIN),
    ]
    index = build_index(mixed, _AxisEmbedder(), pool=POOL_BENCH)
    assert len(index) == 1
    assert index.passages[0].pool == POOL_BENCH


def test_authoring_refuses_a_training_index() -> None:
    """The remaining hole is loading the wrong file; this closes it loudly."""
    train = _index([_passage("t#0000", "El parlament entrena.", pool=POOL_TRAIN)], POOL_TRAIN)
    with pytest.raises(PoolViolationError, match="P9"):
        retrieve_for_authoring(train, "parlament", _AxisEmbedder())


def test_the_refusal_explains_the_consequence_not_just_the_rule() -> None:
    train = _index([_passage("t#0000", "parlament", pool=POOL_TRAIN)], POOL_TRAIN)
    with pytest.raises(PoolViolationError, match="contaminate the benchmark"):
        retrieve_for_authoring(train, "parlament", _AxisEmbedder())


def test_authoring_from_a_bench_index_works() -> None:
    hits = retrieve_for_authoring(_index(), "parlament", _AxisEmbedder(), top_k=1)
    assert hits[0].passage.doc_id == "d1"


def test_an_unknown_pool_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="pool must be"):
        build_index(_bench_passages(), _AxisEmbedder(), pool="elsewhere")


def test_the_two_pools_live_in_separate_files() -> None:
    assert index_path("/tmp/x", POOL_BENCH) != index_path("/tmp/x", POOL_TRAIN)
    assert "bench" in index_path("/tmp/x", POOL_BENCH).name


# --- retrieval -------------------------------------------------------------


def test_the_closest_passage_comes_first() -> None:
    hits = retrieve(_index(), "muntanya", _AxisEmbedder(), top_k=3)
    assert hits[0].passage.doc_id == "d2"
    assert hits[0].score == pytest.approx(1.0)


def test_top_k_limits_the_result() -> None:
    assert len(retrieve(_index(), "parlament", _AxisEmbedder(), top_k=2)) == 2


def test_asking_for_more_than_exists_returns_everything() -> None:
    assert len(retrieve(_index(), "parlament", _AxisEmbedder(), top_k=99)) == 3


def test_ties_are_broken_deterministically() -> None:
    """Two runs over the same corpus must agree on the order."""
    passages = [_passage(f"d{i}#0000", "cap paraula coneguda") for i in range(5)]
    index = build_index(passages, _AxisEmbedder(), pool=POOL_BENCH)
    first = [h.passage.passage_id for h in retrieve(index, "res", _AxisEmbedder(), top_k=5)]
    second = [h.passage.passage_id for h in retrieve(index, "res", _AxisEmbedder(), top_k=5)]
    assert first == second == sorted(first)


def test_an_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        retrieve(_index(), "   ", _AxisEmbedder())


@pytest.mark.parametrize("top_k", [0, -1])
def test_a_nonsensical_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        retrieve(_index(), "parlament", _AxisEmbedder(), top_k=top_k)


def test_the_corpus_is_embedded_once_at_build_time() -> None:
    embedder = _AxisEmbedder()
    build_index(_bench_passages(), embedder, pool=POOL_BENCH)
    assert embedder.calls == 1


def test_an_empty_index_searches_without_crashing() -> None:
    index = build_index([], _AxisEmbedder(), pool=POOL_BENCH)
    assert len(index) == 0
    assert index.dimensions == 0
    assert retrieve(index, "parlament", _AxisEmbedder()) == []


def test_the_default_k_is_small_enough_to_read() -> None:
    assert 3 <= DEFAULT_TOP_K <= 10


# --- consistency -----------------------------------------------------------


def test_a_mismatched_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        CorpusIndex(pool=POOL_BENCH, model="m", passages=_bench_passages(), vectors=[[1.0]])


def test_the_summary_reports_what_is_in_the_index() -> None:
    summary = _index().summary()
    assert "bench index" in summary
    assert "3 passage(s)" in summary
    assert "institucions" in summary


# --- on disk ---------------------------------------------------------------


def test_an_index_roundtrips(tmp_path: Path) -> None:
    index = _index()
    loaded = load_index(write_index(index, tmp_path / "i.jsonl"))
    assert loaded.pool == index.pool
    assert [p.passage_id for p in loaded.passages] == [p.passage_id for p in index.passages]
    assert loaded.dimensions == index.dimensions


def test_the_saved_index_keeps_its_pool(tmp_path: Path) -> None:
    """Loading must not be a way to launder a training index into authoring."""
    train = _index([_passage("t#0000", "parlament", pool=POOL_TRAIN)], POOL_TRAIN)
    loaded = load_index(write_index(train, tmp_path / "t.jsonl"))
    with pytest.raises(PoolViolationError):
        retrieve_for_authoring(loaded, "parlament", _AxisEmbedder())


def test_the_header_records_the_model_and_size(tmp_path: Path) -> None:
    path = write_index(_index(), tmp_path / "i.jsonl")
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header["index_version"] == INDEX_VERSION
    assert header["pool"] == POOL_BENCH
    assert header["passages"] == 3


def test_the_file_is_plain_diffable_jsonl(tmp_path: Path) -> None:
    lines = write_index(_index(), tmp_path / "i.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # header + 3 passages
    assert all(json.loads(line) for line in lines)


def test_an_old_index_version_says_how_to_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "i.jsonl"
    path.write_text(json.dumps({"index_version": 0, "pool": "bench"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus-index"):
        load_index(path)


def test_an_empty_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "i.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        load_index(path)


def test_a_malformed_entry_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / "i.jsonl"
    path.write_text(
        json.dumps({"index_version": INDEX_VERSION, "pool": "bench", "model": "m"})
        + "\n"
        + json.dumps({"vector": [1.0]})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"i\.jsonl:2"):
        load_index(path)


def test_normalise_scales_to_unit_length() -> None:
    assert normalise([3.0, 4.0]) == pytest.approx([0.6, 0.8])


def test_normalise_leaves_a_zero_vector_alone() -> None:
    assert normalise([0.0, 0.0]) == [0.0, 0.0]
