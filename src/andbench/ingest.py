"""Migrate an external QA set into And-Obert items (B2.01).

PLAN B2.01 migrates the Maia test split and the project owner's ~100 manual
questions into the AndBench schema. Those files live outside this repo and their
field names are not ours, so the input shape is **declared per migration** with a
field map rather than guessed.

The important thing this module refuses to do is **fabricate verification**.
Constitution P8: every item is human-verified, by someone other than its author,
and cites a verification source. An importer that filled in ``verifier`` to make
the schema happy would convert an unreviewed question into a "100 % human-verified"
item with one line of code, and the benchmark's central claim would quietly become
false. So the import lands in a **review queue**: each candidate carries the author
it came with, an empty ``verifier``, and ``accepted: false``. A human assigns a
different verifier and accepts; only then does :func:`promote` mint an
:class:`~andbench.schema.Item`.

Ids are derived from a hash of the question, so re-importing the same source yields
the same ids no matter how the file has been reordered, and a repeated question is
caught as the duplicate it is.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from andbench.schema import Item, Track

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

#: Length of the id suffix taken from the question hash.
ID_HASH_LENGTH = 8

#: Tag recording that an item was migrated rather than authored here.
MIGRATED_TAG = "migrated"


@dataclass(frozen=True)
class FieldMap:
    """Which incoming keys hold which AndBench fields.

    Only ``question`` and ``answer`` are required; the rest fall back to the
    per-migration defaults, and ``source_doc_id`` has no default at all because
    an item that cites no source cannot be verified (P8).
    """

    question: str = "question"
    answer: str = "answer"
    area: str | None = None
    difficulty: str | None = None
    source_doc_id: str | None = None
    source_url: str | None = None
    author: str | None = None

    @classmethod
    def parse(cls, spec: str) -> FieldMap:
        """Parse ``field=key,field=key`` as passed on the command line."""
        pairs: dict[str, str] = {}
        for chunk in spec.split(","):
            if not chunk.strip():
                continue
            field, _, key = chunk.partition("=")
            field, key = field.strip(), key.strip()
            if not key:
                raise ValueError(f"field map entry {chunk!r} must look like field=key")
            if field not in cls.__dataclass_fields__:
                allowed = ", ".join(sorted(cls.__dataclass_fields__))
                raise ValueError(f"unknown field {field!r} in the map (allowed: {allowed})")
            pairs[field] = key
        return cls(**pairs)


class IngestCandidate(BaseModel):
    """One migrated question, awaiting human verification."""

    model_config = ConfigDict(extra="forbid")

    item_id: NonEmptyStr
    origin: NonEmptyStr
    area: NonEmptyStr
    question: NonEmptyStr
    answer_text: NonEmptyStr
    difficulty: int = Field(ge=1, le=3)
    source_doc_id: NonEmptyStr
    source_url: str = ""
    author: NonEmptyStr
    #: A human fills this in, and it must differ from ``author`` (P8).
    verifier: str = ""
    #: A human sets this once they have checked the item against its source.
    accepted: bool = False
    note: str = ""


def item_id_for(origin: str, question: str) -> str:
    """A stable id derived from the question, so re-imports do not renumber."""
    digest = hashlib.sha256(question.strip().casefold().encode("utf-8")).hexdigest()
    return f"and-obert-{origin}-{digest[:ID_HASH_LENGTH]}"


def load_raw(path: str | Path) -> list[dict[str, object]]:
    """Load a JSONL or JSON-array export of an external QA set."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("a JSON export must be an array of records")
        records: list[dict[str, object]] = []
        for index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise ValueError(f"{path}: record {index} must be a JSON object")
            records.append(dict(record))
        return records
    lines: list[dict[str, object]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
        lines.append(record)
    return lines


def _text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    return "" if value is None else str(value).strip()


def to_candidates(
    records: Sequence[Mapping[str, object]],
    *,
    origin: str,
    field_map: FieldMap | None = None,
    default_area: str,
    default_author: str,
    default_difficulty: int = 2,
    default_source_doc_id: str | None = None,
) -> tuple[list[IngestCandidate], list[str]]:
    """Convert raw records into review-queue candidates, collecting per-record errors.

    Nothing is dropped silently: a record that cannot be converted produces a
    message naming its index, so a migration can be fixed rather than half-done.
    """
    mapping = field_map or FieldMap()
    candidates: list[IngestCandidate] = []
    errors: list[str] = []
    seen: dict[str, int] = {}

    for index, record in enumerate(records):
        question = _text(record, mapping.question)
        answer = _text(record, mapping.answer)
        if not question:
            errors.append(f"record {index}: no question under key {mapping.question!r}")
            continue
        if not answer:
            errors.append(
                f"record {index}: no reference answer under key {mapping.answer!r} — an "
                "And-Obert item without one cannot be judged"
            )
            continue

        source = (
            _text(record, mapping.source_doc_id)
            if mapping.source_doc_id
            else (default_source_doc_id or "")
        )
        if not source:
            errors.append(
                f"record {index}: no source document — every item must cite a verification "
                "source (constitution P8); map source_doc_id or pass a default"
            )
            continue

        difficulty = default_difficulty
        if mapping.difficulty:
            raw_difficulty = _text(record, mapping.difficulty)
            if raw_difficulty:
                try:
                    difficulty = int(raw_difficulty)
                except ValueError:
                    errors.append(
                        f"record {index}: difficulty {raw_difficulty!r} is not an integer"
                    )
                    continue

        item_id = item_id_for(origin, question)
        if item_id in seen:
            errors.append(
                f"record {index}: duplicate question, already imported as record {seen[item_id]}"
            )
            continue
        seen[item_id] = index

        try:
            candidates.append(
                IngestCandidate(
                    item_id=item_id,
                    origin=origin,
                    area=_text(record, mapping.area) if mapping.area else default_area,
                    question=question,
                    answer_text=answer,
                    difficulty=difficulty,
                    source_doc_id=source,
                    source_url=_text(record, mapping.source_url) if mapping.source_url else "",
                    author=_text(record, mapping.author) if mapping.author else default_author,
                )
            )
        except ValidationError as exc:
            errors.append(f"record {index}: {exc.error_count()} validation error(s)")

    candidates.sort(key=lambda c: c.item_id)
    return candidates, errors


def write_queue(candidates: Sequence[IngestCandidate], path: str | Path) -> Path:
    """Write the review queue a human works through."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(c.model_dump_json() for c in candidates) + ("\n" if candidates else ""),
        encoding="utf-8",
    )
    return target


def load_queue(path: str | Path) -> list[IngestCandidate]:
    """Load a (possibly human-edited) review queue."""
    with Path(path).open(encoding="utf-8") as handle:
        return [IngestCandidate.model_validate_json(line) for line in handle if line.strip()]


def pending(candidates: Sequence[IngestCandidate]) -> list[str]:
    """Candidates a human has not finished: unaccepted, or missing a valid verifier."""
    blocked: list[str] = []
    for candidate in candidates:
        if not candidate.accepted:
            blocked.append(f"{candidate.item_id}: not accepted yet")
        elif not candidate.verifier.strip():
            blocked.append(f"{candidate.item_id}: accepted but no verifier assigned")
        elif candidate.verifier.strip().casefold() == candidate.author.casefold():
            blocked.append(
                f"{candidate.item_id}: verifier is the same person as the author (constitution P8)"
            )
    return blocked


def promote(
    candidates: Sequence[IngestCandidate], *, public: bool = True
) -> tuple[list[Item], list[str]]:
    """Turn fully-reviewed candidates into And-Obert items.

    Returns the items and the reasons the rest were left behind. A candidate is
    never promoted on the importer's authority: a human must have accepted it and
    named a verifier who is not its author.
    """
    blocked = pending(candidates)
    ready = [
        c
        for c in candidates
        if c.accepted
        and c.verifier.strip()
        and c.verifier.strip().casefold() != c.author.casefold()
    ]

    items: list[Item] = []
    for candidate in ready:
        payload: dict[str, object] = {
            "id": candidate.item_id,
            "track": Track.AND_OBERT.value,
            "area": candidate.area,
            "question": candidate.question,
            "answer_text": candidate.answer_text,
            "difficulty": candidate.difficulty,
            "source_doc_id": candidate.source_doc_id,
            "author": candidate.author,
            "verifier": candidate.verifier.strip(),
            "public": public,
            "tags": [MIGRATED_TAG, candidate.origin],
        }
        if candidate.source_url:
            payload["source_url"] = candidate.source_url
        try:
            items.append(Item.model_validate(payload))
        except ValidationError as exc:
            blocked.append(f"{candidate.item_id}: {exc.error_count()} schema error(s)")

    items.sort(key=lambda i: i.id)
    return items, blocked


def write_items(items: Sequence[Item], path: str | Path) -> Path:
    """Write promoted items as an AndBench JSONL file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(i.model_dump_json() for i in items) + ("\n" if items else ""),
        encoding="utf-8",
    )
    return target
