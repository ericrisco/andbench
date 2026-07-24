"""Invariant tests for decontamination (anti-contamination §2, constitution P10)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from andbench.decontam import (
    Collision,
    DecontaminationReport,
    cosine,
    decontaminate,
    embedding_collisions,
    item_text,
    ngram_collisions,
    ngrams,
    tokenize,
)
from andbench.schema import Item

# A 20-token Catalan-ish sentence we can slice a >=13-gram out of.
TRAIN_SENTENCE = (
    "el principat andorra es un microestat situat als pirineus entre "
    "espanya i franca amb una llarga historia de coprincipat feudal molt antiga"
)


def _open_item(text: str, **overrides: Any) -> Item:
    base: dict[str, Any] = {
        "id": "and-obert-0001",
        "track": "and-obert",
        "area": "historia",
        "question": text,
        "answer_text": "resposta de referencia",
        "difficulty": 2,
        "source_doc_id": "pool_bench/x.md",
        "author": "alice",
        "verifier": "bob",
        "public": True,
        "tags": [],
    }
    base.update(overrides)
    return Item.model_validate(base)


class BagOfWordsEmbedder:
    """Deterministic hashing bag-of-words embedder for tests (no real model)."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in tokenize(text):
                vec[hash(token) % self.dim] += 1.0
            vectors.append(vec)
        return vectors


# --- primitives ----------------------------------------------------------


def test_ngrams_empty_when_too_short() -> None:
    assert ngrams(["a", "b"], 13) == set()


def test_ngrams_sliding_window() -> None:
    assert ngrams(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}


def test_cosine_identical_is_one() -> None:
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero() -> None:
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_item_text_includes_choices_and_answer() -> None:
    item = Item.model_validate(
        {
            "id": "and-coneix-0001",
            "track": "and-coneix",
            "area": "geografia",
            "question": "quin riu?",
            "choices": ["valira", "segre", "ebre", "garona"],
            "answer": 0,
            "difficulty": 1,
            "source_doc_id": "x",
            "author": "a",
            "verifier": "b",
            "public": True,
            "tags": [],
        }
    )
    text = item_text(item)
    assert "valira" in text and "quin riu?" in text


# --- n-gram check --------------------------------------------------------


def test_verbatim_reuse_is_flagged() -> None:
    # The item repeats a >=13-word span verbatim from the training sentence.
    item = _open_item(TRAIN_SENTENCE)
    collisions = ngram_collisions([item], [TRAIN_SENTENCE], n=13)
    assert len(collisions) == 1
    assert collisions[0].kind == "ngram"


def test_unrelated_item_is_clean() -> None:
    item = _open_item("una pregunta completament diferent sobre gastronomia local")
    assert ngram_collisions([item], [TRAIN_SENTENCE], n=13) == []


def test_short_item_cannot_ngram_collide() -> None:
    # Fewer than 13 tokens → no 13-gram exists → never an n-gram collision.
    item = _open_item("pregunta curta", answer_text="si")
    assert ngram_collisions([item], [TRAIN_SENTENCE], n=13) == []


def test_ngram_below_minimum_rejected() -> None:
    item = _open_item(TRAIN_SENTENCE)
    with pytest.raises(ValueError, match="must be >= 13"):
        ngram_collisions([item], [TRAIN_SENTENCE], n=12)


# --- embedding check -----------------------------------------------------


def test_embedding_paraphrase_flagged() -> None:
    embedder = BagOfWordsEmbedder()
    # A reordering the n-gram check would MISS: the training passage holds the
    # same content words as the whole item (question + answer_text), shuffled.
    item = _open_item("andorra microestat pirineus", answer_text="coprincipat feudal historia")
    train = ["historia feudal coprincipat pirineus microestat andorra"]
    # Sanity: no shared 13-gram, so only the embedding check can catch this.
    assert ngram_collisions([item], train, n=13) == []
    collisions = embedding_collisions([item], train, embedder, threshold=0.9)
    assert len(collisions) == 1
    assert collisions[0].kind == "embedding"


def test_embedding_dissimilar_clean() -> None:
    embedder = BagOfWordsEmbedder()
    train = ["andorra microestat pirineus coprincipat feudal historia"]
    item = _open_item("recepta de truites de riu amb all i julivert fresc")
    assert embedding_collisions([item], train, embedder, threshold=0.9) == []


def test_embedding_empty_train_is_clean() -> None:
    embedder = BagOfWordsEmbedder()
    assert embedding_collisions([_open_item("qualsevol cosa")], [], embedder) == []


# --- combined ------------------------------------------------------------


def test_decontaminate_without_embedder_runs_ngram_only() -> None:
    item = _open_item(TRAIN_SENTENCE)
    report = decontaminate([item], [TRAIN_SENTENCE])
    assert report.embedding_checked is False
    assert not report.clean
    assert item.id in report.contaminated_ids


def test_decontaminate_with_embedder_runs_both() -> None:
    embedder = BagOfWordsEmbedder()
    item = _open_item("una pregunta neta i original sobre formatges")
    report = decontaminate([item], ["text d'entrenament sense cap relacio"], embedder=embedder)
    assert report.embedding_checked is True
    assert report.clean


def test_report_summary_clean_and_dirty() -> None:
    clean = DecontaminationReport(checked=1)
    assert "CLEAN" in clean.summary()
    dirty = DecontaminationReport(
        checked=1, collisions=[Collision("x", "ngram", "shares a 13-gram")]
    )
    assert "contaminated" in dirty.summary()
