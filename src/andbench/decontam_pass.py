"""Full-dataset decontamination pass (B2.08).

B1.03 provides the per-item checks; this is the release-level pass that runs them
over the **whole** item set (all tracks, public and private) against the training
corpus and produces the artifacts the rewrite workflow needs:

* a machine-readable JSON report (per-collision detail + a ``rewrite_ids`` list), and
* a plain rewrite-list of the item ids that collided.

It gates the release: any collision means the pass fails until those items are
rewritten. The training corpus and (optional) embedder are supplied per release.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from andbench.decontam import (
    DEFAULT_SIMILARITY_THRESHOLD,
    MIN_NGRAM,
    DecontaminationReport,
    Embedder,
    decontaminate,
)
from andbench.schema import Item
from andbench.validation import validate_jsonl


@dataclass(frozen=True)
class PassArtifacts:
    report_path: Path
    rewrite_path: Path
    report: DecontaminationReport


def load_training_texts(path: str | Path) -> list[str]:
    """Load a training-text file: one passage per non-blank line."""
    path = Path(path)
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_pass(
    items: Sequence[Item],
    train_texts: Sequence[str],
    *,
    embedder: Embedder | None = None,
    n: int = MIN_NGRAM,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> DecontaminationReport:
    """Run the full decontamination pass over every item."""
    return decontaminate(items, train_texts, embedder=embedder, n=n, threshold=threshold)


def write_artifacts(report: DecontaminationReport, out_dir: str | Path) -> PassArtifacts:
    """Write ``decontam-report.json`` and ``rewrite-list.txt``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "decontam-report.json"
    rewrite_path = out / "rewrite-list.txt"

    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rewrite_path.write_text(
        "\n".join(report.rewrite_ids) + ("\n" if report.rewrite_ids else ""),
        encoding="utf-8",
    )
    return PassArtifacts(report_path=report_path, rewrite_path=rewrite_path, report=report)


def run_pass_from_files(
    items_path: str | Path,
    train_path: str | Path,
    out_dir: str | Path,
    *,
    n: int = MIN_NGRAM,
) -> PassArtifacts:
    """Load items + training texts, run the pass, and write the artifacts.

    Raises ``ValueError`` if the item file itself does not validate — a
    contamination pass over invalid items would be meaningless.
    """
    validation = validate_jsonl(items_path)
    if not validation.ok:
        raise ValueError(f"item file failed schema validation:\n{validation.summary()}")
    train_texts = load_training_texts(train_path)
    report = run_pass(validation.items, train_texts, n=n)
    return write_artifacts(report, out_dir)
