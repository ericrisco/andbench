"""Local retrieval over the Andorran corpus (the RAG half of the workaround).

Items are written from passages, so something has to find the right passages. This
does it locally: embed once, keep the vectors on disk, score by cosine at query
time. No service, no key, no GPU — the same reasoning as the decontamination
embedder, and it reuses that embedder rather than introducing a second one.

Brute force on purpose. A corpus of a few thousand passages scores in milliseconds
against a normalised matrix, and an approximate-nearest-neighbour dependency would
buy nothing here except a build step and a reason for two machines to disagree.

**An index belongs to exactly one pool.** Constitution P9 says AndBench items are
written *only* from ``pool_bench``. That could have been a ``pool=`` filter on the
search call — one that works until somebody forgets it, and whose failure is
invisible because contaminated items look exactly like clean ones. Instead the pool
is a property of the index: a bench index physically contains no training passage,
so retrieving one is not something the API can be asked to do.
:func:`retrieve_for_authoring` refuses a train index outright, which catches the
remaining case of loading the wrong file.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from andbench.corpus import POOL_BENCH, POOL_TRAIN, Passage
from andbench.decontam import Embedder, cosine
from andbench.providers.embeddings import DEFAULT_EMBEDDING_MODEL

#: Passages returned by default. Enough context for one item without burying the
#: author in near-duplicates.
DEFAULT_TOP_K = 5

#: Format version of the on-disk index, so a change is detectable.
INDEX_VERSION = 1


class PoolViolationError(RuntimeError):
    """An attempt to author items from training material (constitution P9)."""


@dataclass(frozen=True)
class Hit:
    """One retrieved passage and its score."""

    passage: Passage
    score: float

    def line(self) -> str:
        excerpt = self.passage.text.replace("\n", " ")[:90]
        return f"  {self.score:.3f}  {self.passage.passage_id}  {excerpt}…"


@dataclass
class CorpusIndex:
    """Passages of a single pool with their embeddings."""

    pool: str
    model: str
    passages: list[Passage]
    vectors: list[list[float]]

    def __post_init__(self) -> None:
        if len(self.passages) != len(self.vectors):
            raise ValueError(
                f"index is inconsistent: {len(self.passages)} passage(s) but "
                f"{len(self.vectors)} vector(s)"
            )
        wrong = {p.pool for p in self.passages} - {self.pool}
        if wrong:
            raise PoolViolationError(
                f"a {self.pool!r} index cannot hold {sorted(wrong)} passages — the pool is "
                "what makes P9 structural rather than a filter someone must remember"
            )

    def __len__(self) -> int:
        return len(self.passages)

    @property
    def dimensions(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0

    def search(self, query_vector: Sequence[float], *, top_k: int = DEFAULT_TOP_K) -> list[Hit]:
        """The ``top_k`` closest passages by cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scored = [
            Hit(passage=passage, score=cosine(query_vector, vector))
            for passage, vector in zip(self.passages, self.vectors, strict=True)
        ]
        # Ties broken by passage id so two runs over the same corpus agree.
        scored.sort(key=lambda hit: (-hit.score, hit.passage.passage_id))
        return scored[:top_k]

    def topics(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for passage in self.passages:
            counts[passage.topic] = counts.get(passage.topic, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        return (
            f"{self.pool} index: {len(self)} passage(s), {self.dimensions} dims, "
            f"model {self.model}, topics {self.topics()}"
        )


def build_index(
    passages: Sequence[Passage],
    embedder: Embedder,
    *,
    pool: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> CorpusIndex:
    """Embed one pool's passages into a searchable index.

    Only passages of ``pool`` are taken; anything else is dropped rather than
    embedded, so a mixed input cannot produce a mixed index.
    """
    if pool not in (POOL_BENCH, POOL_TRAIN):
        raise ValueError(f"pool must be {POOL_BENCH!r} or {POOL_TRAIN!r}, got {pool!r}")
    selected = [p for p in passages if p.pool == pool]
    vectors = embedder.embed([p.text for p in selected]) if selected else []
    return CorpusIndex(pool=pool, model=model, passages=selected, vectors=vectors)


def retrieve(
    index: CorpusIndex, query: str, embedder: Embedder, *, top_k: int = DEFAULT_TOP_K
) -> list[Hit]:
    """Embed ``query`` and return its nearest passages."""
    if not query.strip():
        raise ValueError("query must not be empty")
    return index.search(embedder.embed([query])[0], top_k=top_k)


def retrieve_for_authoring(
    index: CorpusIndex, query: str, embedder: Embedder, *, top_k: int = DEFAULT_TOP_K
) -> list[Hit]:
    """Retrieve passages an item may be written from.

    Refuses a training index. The index being pool-scoped already makes the wrong
    passage unreachable; this catches the one case that remains — loading the wrong
    file — and says why rather than quietly returning usable-looking text.
    """
    if index.pool != POOL_BENCH:
        raise PoolViolationError(
            f"items may only be written from {POOL_BENCH!r}, and this is a {index.pool!r} "
            "index. AndBench items written from training material would contaminate the "
            "benchmark against the model it exists to evaluate (constitution P9)."
        )
    return retrieve(index, query, embedder, top_k=top_k)


# --- on-disk format --------------------------------------------------------


def write_index(index: CorpusIndex, path: str | Path) -> Path:
    """Write the index as JSONL: a header line, then one line per passage.

    Plain text on purpose — it diffs, it survives a Python upgrade, and a corpus of
    this size does not justify a binary format.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "index_version": INDEX_VERSION,
        "pool": index.pool,
        "model": index.model,
        "dimensions": index.dimensions,
        "passages": len(index),
    }
    lines = [json.dumps(header, ensure_ascii=False, sort_keys=True)]
    lines.extend(
        json.dumps(
            {
                "passage": json.loads(passage.model_dump_json()),
                # Rounded: 6 decimals is far below the resolution any ranking
                # depends on, and it keeps the file diffable and reproducible.
                "vector": [round(value, 6) for value in vector],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for passage, vector in zip(index.passages, index.vectors, strict=True)
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def load_index(path: str | Path) -> CorpusIndex:
    """Load an index written by :func:`write_index`."""
    source = Path(path)
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{source} is empty")

    header = json.loads(lines[0])
    if header.get("index_version") != INDEX_VERSION:
        raise ValueError(
            f"{source}: index_version {header.get('index_version')} is not "
            f"{INDEX_VERSION}; rebuild it with `andbench corpus-index`"
        )

    passages: list[Passage] = []
    vectors: list[list[float]] = []
    for lineno, line in enumerate(lines[1:], start=2):
        payload = json.loads(line)
        try:
            passages.append(Passage.model_validate(payload["passage"]))
            vectors.append([float(v) for v in payload["vector"]])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{source}:{lineno}: malformed index entry: {exc}") from exc

    return CorpusIndex(
        pool=str(header["pool"]),
        model=str(header.get("model", "unknown")),
        passages=passages,
        vectors=vectors,
    )


def index_path(base: str | Path, pool: str) -> Path:
    """Where a pool's index lives. Separate files, so the two cannot be confused."""
    return Path(base) / f"index-{pool}.jsonl"


def normalise(vector: Sequence[float]) -> list[float]:
    """Scale to unit length. A no-op for an already-normalised embedder."""
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else list(vector)
