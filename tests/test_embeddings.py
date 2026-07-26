"""Tests for the local embedder. Unit tests never download or load a model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from andbench.decontam import Embedder, decontaminate
from andbench.providers.embeddings import (
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    EmbedderUnavailableError,
    SentenceTransformerEmbedder,
    build_embedder,
)
from andbench.schema import Item


class _FakeSentenceTransformer:
    """Stands in for the real model: records how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append({"texts": list(texts), **kwargs})
        return [[1.0, 0.0] for _ in texts]


def _loaded(model: _FakeSentenceTransformer) -> SentenceTransformerEmbedder:
    embedder = SentenceTransformerEmbedder()
    embedder._model = model
    return embedder


# --- the defaults ----------------------------------------------------------


def test_the_default_model_is_paraphrase_tuned_and_multilingual() -> None:
    """The task is literally paraphrase detection, and the items are Catalan."""
    assert "paraphrase" in DEFAULT_EMBEDDING_MODEL
    assert "multilingual" in DEFAULT_EMBEDDING_MODEL


def test_the_default_device_is_cpu() -> None:
    """The project has no GPU, and this has to run wherever CI does."""
    assert DEFAULT_DEVICE == "cpu"


def test_build_embedder_defers_loading() -> None:
    """Constructing one must not pull in torch for a caller that only n-grams."""
    embedder = build_embedder()
    assert embedder._model is None
    assert embedder.model_name == DEFAULT_EMBEDDING_MODEL


def test_a_custom_model_and_device_are_honoured() -> None:
    embedder = build_embedder("other/model", device="mps")
    assert (embedder.model_name, embedder.device) == ("other/model", "mps")


def test_describe_names_the_model_and_device_for_the_record() -> None:
    assert "cpu" in build_embedder().describe()
    assert DEFAULT_EMBEDDING_MODEL in build_embedder().describe()


# --- embedding -------------------------------------------------------------


def test_it_satisfies_the_decontamination_seam() -> None:
    assert isinstance(build_embedder(), Embedder)


def test_embeddings_are_normalised_so_cosine_is_a_dot_product() -> None:
    model = _FakeSentenceTransformer()
    _loaded(model).embed(["hola"])
    assert model.calls[0]["normalize_embeddings"] is True


def test_the_progress_bar_is_off_so_ci_logs_stay_readable() -> None:
    model = _FakeSentenceTransformer()
    _loaded(model).embed(["hola"])
    assert model.calls[0]["show_progress_bar"] is False


def test_texts_are_passed_through_in_order() -> None:
    model = _FakeSentenceTransformer()
    _loaded(model).embed(["un", "dos", "tres"])
    assert model.calls[0]["texts"] == ["un", "dos", "tres"]


def test_the_batch_size_is_forwarded() -> None:
    model = _FakeSentenceTransformer()
    embedder = _loaded(model)
    embedder.batch_size = 8
    embedder.embed(["hola"])
    assert model.calls[0]["batch_size"] == 8


def test_no_texts_means_no_call_at_all() -> None:
    model = _FakeSentenceTransformer()
    assert _loaded(model).embed([]) == []
    assert model.calls == []


def test_vectors_come_back_as_plain_floats() -> None:
    """The report is serialised to JSON, so numpy scalars would not survive."""
    vectors = _loaded(_FakeSentenceTransformer()).embed(["hola"])
    assert vectors == [[1.0, 0.0]]
    assert all(isinstance(value, float) for value in vectors[0])


def test_the_model_is_loaded_once_and_reused() -> None:
    model = _FakeSentenceTransformer()
    embedder = _loaded(model)
    first, second = embedder.load(), embedder.load()
    assert first is second
    assert embedder._model is not None


# --- the optional dependency ----------------------------------------------


def test_a_missing_dependency_says_exactly_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _fail(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sentence_transformers":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail)
    with pytest.raises(EmbedderUnavailableError, match="uv sync --group decontam"):
        build_embedder().load()


def test_the_missing_dependency_message_says_the_ngram_half_still_runs() -> None:
    """A gate that silently skips is not a gate; the user must know what they lost."""
    import builtins

    real_import = builtins.__import__

    def _fail(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sentence_transformers":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _fail
    try:
        with pytest.raises(EmbedderUnavailableError, match="n-gram check runs without it"):
            build_embedder().load()
    finally:
        builtins.__import__ = real_import


# --- end to end through the decontamination check -------------------------


def _item(item_id: str, question: str) -> Item:
    return Item.model_validate(
        {
            "id": item_id,
            "track": "and-coneix",
            "area": "geografia",
            "question": question,
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


class _TwoClusterEmbedder:
    """Anything containing 'parlament' embeds one way, everything else another."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "parlament" in text.lower() else [0.0, 1.0] for text in texts]


def test_a_paraphrase_collision_is_caught_by_the_embedding_half() -> None:
    """The n-gram half cannot see this: no shared 13-token run."""
    items = [_item("i-1", "Qui aprova el pressupost al parlament?")]
    train = ["El parlament és qui dona llum verda als comptes de l'any."]
    report = decontaminate(items, train, embedder=_TwoClusterEmbedder())
    assert not report.clean
    assert report.embedding_checked is True
    assert any(c.kind == "embedding" for c in report.collisions)


def test_an_unrelated_passage_stays_clean() -> None:
    items = [_item("i-1", "Qui aprova el pressupost al parlament?")]
    train = ["La recepta es prepara amb farina i ous."]
    report = decontaminate(items, train, embedder=_TwoClusterEmbedder())
    assert report.clean


def test_the_embedding_half_is_off_without_an_embedder() -> None:
    items = [_item("i-1", "Qui aprova el pressupost al parlament?")]
    report = decontaminate(items, ["El parlament dona llum verda als comptes."])
    assert report.embedding_checked is False
    assert report.clean
