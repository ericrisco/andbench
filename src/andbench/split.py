"""Public / private split of the item set (anti-contamination §3, B2.09).

~85% of items are published; ~15% (a stratified sample) stays **private**,
custodied out of the repo by the PO. The private split is a permanent
over-fitting detector: a model scoring much higher on the public set than the
private one has been contaminated by the public benchmark.

The split is **stratified** by ``(track, area)`` and **deterministic** — driven
by a seeded SHA-256 rank of each item id, so it is reproducible and independent
of item order. The public export carries the canary GUID (B1.04); the private
export is written under ``data/private/`` (git-ignored) and must be moved to PO
custody.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from andbench.canary import CanaryRecord, write_public_dataset
from andbench.schema import Item

#: Default fraction of items published (the rest are private).
DEFAULT_PUBLIC_FRACTION = 0.85

#: Default split seed (matches ``configs/tracks.yaml`` split.seed).
DEFAULT_SPLIT_SEED = 20260724


@dataclass
class SplitResult:
    public: list[Item] = field(default_factory=list)
    private: list[Item] = field(default_factory=list)
    #: (track, area) -> (n_public, n_private)
    strata: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.public) + len(self.private)

    @property
    def actual_public_fraction(self) -> float:
        return len(self.public) / self.total if self.total else 0.0


def _rank_key(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def split_items(
    items: Sequence[Item],
    *,
    public_fraction: float = DEFAULT_PUBLIC_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> SplitResult:
    """Stratified, deterministic public/private split by ``(track, area)``."""
    if not 0.0 < public_fraction < 1.0:
        raise ValueError(f"public_fraction must be in (0, 1), got {public_fraction}")
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("items must have unique ids before splitting")

    strata: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in items:
        strata[(item.track.value, item.area)].append(item)

    result = SplitResult()
    for key in sorted(strata):
        group = strata[key]
        ranked = sorted(group, key=lambda i: _rank_key(seed, i.id))
        k = round(len(ranked) * public_fraction)
        public_group = ranked[:k]
        private_group = ranked[k:]
        result.public.extend(public_group)
        result.private.extend(private_group)
        result.strata[key] = (len(public_group), len(private_group))

    result.public.sort(key=lambda i: i.id)
    result.private.sort(key=lambda i: i.id)
    return result


def write_split(
    result: SplitResult,
    public_path: str | Path,
    private_path: str | Path,
    *,
    canary: CanaryRecord | None = None,
) -> dict[str, Path]:
    """Write the public export (canary-embedded) and the private export."""
    public_path = Path(public_path)
    private_path = Path(private_path)

    public_lines = [i.model_dump_json() for i in result.public]
    write_public_dataset(public_lines, public_path, canary)

    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        "\n".join(i.model_dump_json() for i in result.private) + ("\n" if result.private else ""),
        encoding="utf-8",
    )
    return {"public": public_path, "private": private_path}
