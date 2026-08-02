"""The embedding half of the decontamination check (closes the P10 gap).

Constitution P10 requires **two** independent checks between every item and Maia's
training set: n-gram overlap for near-verbatim reuse, and embedding similarity for
paraphrase collisions. Until now only the first ran — the second was an injectable
seam with no implementation, so a rewritten-but-equivalent item passed a protocol
the project describes as two-layered.

**Local, not an API.** The check gates a release and runs in CI of both ``andbench``
and ``maia`` (P10). A hosted embedder would need a credential, which means it
cannot run on a fork's pull request — and a gate that silently skips is not a gate.
So the model is downloaded and run on CPU, in an optional ``decontam`` dependency
group: the lean default install still gates on near-verbatim reuse, and the heavier
half is opt-in.

**Why this model.** ``paraphrase-multilingual-MiniLM-L12-v2`` is trained for
paraphrase identification across 50+ languages including Catalan, and at 118M
parameters it is CPU-practical for a release-sized item set. The task here is
literally paraphrase detection, so an on-task model beats a larger general-purpose
retriever. It is configurable because that reasoning may not survive contact with
the real corpus.

**The threshold is calibrated, not guessed.** See
:mod:`andbench.decontam_threshold` — a cosine cut-off asserted without evidence is
the kind of number this project keeps removing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from sentence_transformers import SentenceTransformer

#: Trained for paraphrase identification, multilingual (Catalan included), 118M
#: parameters, 384 dimensions. Small enough to run on a CI runner's CPU.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#: CPU by default: the project has no GPU (PRD §6) and this must run anywhere CI does.
DEFAULT_DEVICE = "cpu"

DEFAULT_BATCH_SIZE = 32


class EmbedderUnavailableError(RuntimeError):
    """The optional dependency is not installed."""


@dataclass
class SentenceTransformerEmbedder:
    """A local :class:`~andbench.decontam.Embedder`.

    The model is loaded on first use, so constructing one is free — importing this
    module must not pull in torch for callers that only run the n-gram check.
    """

    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: str = DEFAULT_DEVICE
    batch_size: int = DEFAULT_BATCH_SIZE
    _model: SentenceTransformer | None = field(default=None, repr=False)

    def load(self) -> SentenceTransformer:
        """Load the model, or explain exactly how to make it available."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderUnavailableError(
                "the embedding decontamination check needs sentence-transformers: "
                "run `uv sync --group decontam`. The n-gram check runs without it."
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts``, order preserved, L2-normalised.

        Normalising here means the cosine in :mod:`andbench.decontam` is a plain dot
        product and the numbers are comparable between runs and models.
        """
        if not texts:
            return []
        vectors = self.load().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def describe(self) -> str:
        """What to record in a decontamination report, so a run is auditable."""
        return f"{self.model_name} on {self.device}"


def build_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL, *, device: str = DEFAULT_DEVICE
) -> SentenceTransformerEmbedder:
    """Construct the default embedder. Loading is deferred to first use."""
    return SentenceTransformerEmbedder(model_name=model_name, device=device)
