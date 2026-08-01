"""The Andorran source corpus: documents, provenance, and passages.

The institutional requests (B0.02) went unanswered, so items have to be written
from sources the project can gather itself. This module is the front of that path:
it takes documents, records where each one came from and under what licence, and
splits them into the passages an author (or a draft model) actually works from.

Two things it insists on, because both are cheap now and impossible to reconstruct
later:

**Provenance travels with the text.** Every passage knows its document, and every
document knows its source, licence, URL and retrieval date. An item cites a passage
id; from that alone, anyone can walk back to what was read and check the licence
still permits publication. A corpus assembled without this is a corpus you cannot
publish from.

**Pool membership is part of the passage.** AndBench items may be written **only**
from ``pool_bench`` (constitution P9). Carrying the pool on the passage rather than
in a separate list is what lets :mod:`andbench.retrieval` make the wrong retrieval
structurally impossible rather than merely discouraged.

Chunking is deliberately dull: paragraph-first, with a character budget and a small
overlap. A benchmark item needs a self-contained passage a human can check at a
glance, not a maximally-packed context window.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from andbench.card import Permission
from andbench.partition import CorpusDoc

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

#: Target passage size in characters. Big enough for a fact plus its context, small
#: enough that a verifier can check an item against it without hunting.
DEFAULT_CHUNK_CHARS = 1200

#: Carried between adjacent passages so a fact split across a paragraph boundary is
#: not lost to both sides.
DEFAULT_OVERLAP_CHARS = 150

#: A passage shorter than this is a heading or a stray line, not something an item
#: can be written from.
MIN_CHUNK_CHARS = 120

#: The two pools of the anti-contamination partition (protocol §1).
POOL_TRAIN = "train"
POOL_BENCH = "bench"

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_WHITESPACE_RE = re.compile(r"[ \t]+")


class CorpusDocument(BaseModel):
    """One source document, with everything needed to publish from it later."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: NonEmptyStr
    #: Must match an ``id_prefix`` in ``configs/sources.yaml`` so the P23 permission
    #: gate can find it. Recording the family here is what links a passage to a
    #: licence.
    source: NonEmptyStr
    topic: NonEmptyStr
    title: str = ""
    url: str = ""
    licence: str = ""
    permission: Permission = Permission.PENDING
    #: When the text was captured. A page that changes later is still auditable.
    retrieved: date | None = None
    text: NonEmptyStr

    def manifest_entry(self) -> CorpusDoc:
        """The reduced record the partition consumes."""
        return CorpusDoc(doc_id=self.doc_id, source=self.source, topic=self.topic)


class Passage(BaseModel):
    """A chunk of a document, carrying its provenance and its pool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_id: NonEmptyStr
    doc_id: NonEmptyStr
    source: NonEmptyStr
    topic: NonEmptyStr
    #: ``bench`` or ``train``. Items may only ever be written from ``bench`` (P9).
    pool: NonEmptyStr
    ordinal: int = Field(ge=0)
    text: NonEmptyStr

    def citation(self) -> str:
        """What an item records as its ``source_doc_id``."""
        return self.passage_id


def passage_id_for(doc_id: str, ordinal: int, text: str) -> str:
    """A stable id: the document, the position, and a digest of the text.

    Including the digest means re-chunking after an edit produces a *different* id
    rather than silently pointing an existing item at changed text.
    """
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}#{ordinal:04d}-{digest}"


def _normalise(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_text(
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:
    """Split prose into passages on paragraph boundaries where it can.

    Paragraphs are accumulated until the budget is reached, so a passage is a whole
    number of paragraphs whenever possible — an item written from half a sentence is
    an item a verifier cannot check.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than chunk_chars")

    paragraphs = [_normalise(p) for p in _PARAGRAPH_RE.split(text) if _normalise(p)]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for paragraph in paragraphs:
        # A paragraph longer than the budget is split on its own, by characters:
        # there is no smaller structural boundary to respect.
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, length = [], 0
            chunks.extend(_split_long(paragraph, chunk_chars, overlap_chars))
            continue
        if length + len(paragraph) > chunk_chars and current:
            chunks.append("\n\n".join(current))
            current, length = [], 0
        current.append(paragraph)
        length += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if len(c) >= min_chars] or (
        # Everything fell below the floor: keep the longest rather than returning
        # nothing, so a short document is visibly present rather than vanishing.
        [max(chunks, key=len)] if chunks else []
    )


def _split_long(paragraph: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    step = chunk_chars - overlap_chars
    return [paragraph[i : i + chunk_chars] for i in range(0, len(paragraph), step)]


def chunk_document(
    document: CorpusDocument,
    pool: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Passage]:
    """Split one document into passages tagged with their pool."""
    if pool not in (POOL_TRAIN, POOL_BENCH):
        raise ValueError(f"pool must be {POOL_BENCH!r} or {POOL_TRAIN!r}, got {pool!r}")
    return [
        Passage(
            passage_id=passage_id_for(document.doc_id, ordinal, text),
            doc_id=document.doc_id,
            source=document.source,
            topic=document.topic,
            pool=pool,
            ordinal=ordinal,
            text=text,
        )
        for ordinal, text in enumerate(
            split_text(document.text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        )
    ]


def chunk_corpus(
    documents: Sequence[CorpusDocument],
    pools: dict[str, str],
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> tuple[list[Passage], list[str]]:
    """Chunk every document whose pool is known, reporting the ones that are not.

    A document missing from the partition is **skipped, not defaulted**. Guessing a
    pool is how training text ends up under an item, which is the one thing the
    partition exists to prevent.
    """
    passages: list[Passage] = []
    problems: list[str] = []
    for document in documents:
        pool = pools.get(document.doc_id)
        if pool is None:
            problems.append(
                f"{document.doc_id}: not in the partition, so its pool is unknown — "
                "skipped rather than guessed (constitution P9)"
            )
            continue
        passages.extend(
            chunk_document(document, pool, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        )
    passages.sort(key=lambda p: (p.doc_id, p.ordinal))
    return passages, problems


# --- I/O -------------------------------------------------------------------


def load_documents(path: str | Path) -> list[CorpusDocument]:
    """Load a JSONL corpus. Each line is one document with its provenance."""
    source = Path(path)
    documents: list[CorpusDocument] = []
    for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            documents.append(CorpusDocument.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{source}:{lineno}: {exc}") from exc
    return documents


def write_documents(documents: Sequence[CorpusDocument], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(d.model_dump_json() for d in documents) + ("\n" if documents else ""),
        encoding="utf-8",
    )
    return target


def write_manifest(documents: Sequence[CorpusDocument], path: str | Path) -> Path:
    """Write the reduced manifest the partition commands consume."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(d.manifest_entry().model_dump_json() for d in documents)
        + ("\n" if documents else ""),
        encoding="utf-8",
    )
    return target


def write_passages(passages: Sequence[Passage], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(p.model_dump_json() for p in passages) + ("\n" if passages else ""),
        encoding="utf-8",
    )
    return target


def load_passages(path: str | Path) -> list[Passage]:
    with Path(path).open(encoding="utf-8") as handle:
        return [Passage.model_validate_json(line) for line in handle if line.strip()]


def iter_pool(passages: Sequence[Passage], pool: str) -> Iterator[Passage]:
    """Only the passages of one pool."""
    return (p for p in passages if p.pool == pool)


def licence_warnings(documents: Sequence[CorpusDocument]) -> list[str]:
    """Sources whose permission would block a release built from them.

    A warning, not a block: indexing something you may not yet publish from is a
    reasonable thing to do while a request is pending. The refusal happens at card
    time (P23), where it belongs — this just means nobody discovers it after
    writing two hundred items.
    """
    unpermitted: dict[str, int] = {}
    for document in documents:
        if not document.permission.publishable:
            unpermitted[f"{document.source} ({document.permission.value})"] = (
                unpermitted.get(f"{document.source} ({document.permission.value})", 0) + 1
            )
    return [
        f"{count} document(s) from {label}: items written from these will be blocked at "
        "card time until the permission is recorded"
        for label, count in sorted(unpermitted.items())
    ]


def summarise(documents: Sequence[CorpusDocument], passages: Sequence[Passage]) -> str:
    """A line a human can read after an ingest."""
    bench = sum(1 for p in passages if p.pool == POOL_BENCH)
    train = len(passages) - bench
    sources = len({d.source for d in documents})
    chars = sum(len(p.text) for p in passages)
    return (
        f"{len(documents)} document(s) from {sources} source(s) → {len(passages)} passage(s) "
        f"({bench} bench / {train} train), {chars:,} characters"
    )


def load_pools(partition_dir: str | Path) -> dict[str, str]:
    """Read ``pool_train.txt`` / ``pool_bench.txt`` into a doc_id → pool map."""
    base = Path(partition_dir)
    pools: dict[str, str] = {}
    for name, pool in (("pool_train.txt", POOL_TRAIN), ("pool_bench.txt", POOL_BENCH)):
        path = base / name
        if not path.is_file():
            raise ValueError(f"{path} is missing; run `andbench partition` first")
        for doc_id in path.read_text(encoding="utf-8").splitlines():
            if doc_id.strip():
                pools[doc_id.strip()] = pool
    return pools


def json_ready(passage: Passage) -> dict[str, object]:
    """A passage as a plain dict, for embedding in a prompt or a report."""
    payload: dict[str, object] = json.loads(passage.model_dump_json())
    return payload
