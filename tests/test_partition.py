"""Invariant tests for the corpus partition (anti-contamination §1).

The properties under test are the ones the firewall depends on: determinism,
order-independence, exhaustive+disjoint assignment, and per-stratum stratification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from andbench.partition import (
    CorpusDoc,
    load_manifest,
    partition_corpus,
    write_partition,
)


def _corpus(n_per_stratum: int = 50) -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []
    for source in ("bopa", "wiki", "escola"):
        for topic in ("historia", "geografia"):
            for i in range(n_per_stratum):
                docs.append(
                    CorpusDoc(doc_id=f"{source}-{topic}-{i:03d}", source=source, topic=topic)
                )
    return docs


def test_exhaustive_and_disjoint() -> None:
    docs = _corpus()
    part = partition_corpus(docs)
    all_ids = {d.doc_id for d in docs}
    train, bench = set(part.train_ids), set(part.bench_ids)
    assert train | bench == all_ids
    assert train & bench == set()
    assert part.total == len(all_ids)


def test_deterministic_same_seed() -> None:
    docs = _corpus()
    a = partition_corpus(docs, seed=123)
    b = partition_corpus(docs, seed=123)
    assert a.train_ids == b.train_ids
    assert a.bench_ids == b.bench_ids


def test_order_independent() -> None:
    docs = _corpus()
    reversed_docs = list(reversed(docs))
    a = partition_corpus(docs, seed=7)
    b = partition_corpus(reversed_docs, seed=7)
    assert a.bench_ids == b.bench_ids


def test_different_seed_changes_assignment() -> None:
    docs = _corpus()
    a = partition_corpus(docs, seed=1)
    b = partition_corpus(docs, seed=2)
    assert a.bench_ids != b.bench_ids


def test_stratified_each_stratum_hits_fraction() -> None:
    docs = _corpus(n_per_stratum=50)  # 6 strata x 50 = 300 docs
    part = partition_corpus(docs, bench_fraction=0.10)
    # Each stratum of 50 contributes round(50*0.10)=5 to bench.
    for (_source, _topic), (n_train, n_bench) in part.strata.items():
        assert n_bench == 5
        assert n_train == 45
    assert len(part.bench_ids) == 30
    assert part.actual_bench_fraction == pytest.approx(0.10)


def test_bench_ids_come_only_from_corpus() -> None:
    docs = _corpus()
    part = partition_corpus(docs)
    corpus_ids = {d.doc_id for d in docs}
    assert set(part.bench_ids) <= corpus_ids


def test_duplicate_doc_id_rejected() -> None:
    docs = [
        CorpusDoc(doc_id="dup", source="a", topic="t"),
        CorpusDoc(doc_id="dup", source="a", topic="t"),
    ]
    with pytest.raises(ValueError, match="duplicate doc_id"):
        partition_corpus(docs)


def test_empty_corpus_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        partition_corpus([])


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_invalid_fraction_rejected(frac: float) -> None:
    with pytest.raises(ValueError, match="bench_fraction"):
        partition_corpus(_corpus(), bench_fraction=frac)


def test_load_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    lines = [
        json.dumps({"doc_id": "a", "source": "bopa", "topic": "dret", "extra": "ignored"}),
        "",  # blank line tolerated
        json.dumps({"doc_id": "b", "source": "wiki", "topic": "geografia"}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    docs = load_manifest(path)
    assert [d.doc_id for d in docs] == ["a", "b"]


def test_load_manifest_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_manifest(path)


def test_write_partition_outputs(tmp_path: Path) -> None:
    part = partition_corpus(_corpus(), seed=42)
    paths = write_partition(part, tmp_path / "out")

    train_lines = paths["train"].read_text(encoding="utf-8").splitlines()
    bench_lines = paths["bench"].read_text(encoding="utf-8").splitlines()
    assert train_lines == list(part.train_ids)  # sorted, one per line
    assert bench_lines == list(part.bench_ids)

    meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert meta["seed"] == 42
    assert meta["n_bench"] == len(part.bench_ids)
    assert meta["total"] == part.total
