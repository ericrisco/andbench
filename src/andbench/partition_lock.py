"""Freeze and verify the corpus partition (anti-contamination §1, B1.02).

Once the ``(source, topic)`` split is computed (B1.01) it must be **frozen**: a
small lockfile records the seed, the fraction, and a SHA-256 over each pool's
sorted ``doc_id`` list. That lockfile is committed in **both** ``andbench`` and
``pirene-lm``. CI in each repo recomputes the partition from the live corpus
manifest and checks it still hashes to the committed lock — so neither side can
silently move a document between pools, and Pirene's synthetic generation
(which consumes ``pool_train`` only) is auditable against the same hash.

The real corpus manifest is external, so the committed lock is generated when
the manifest arrives (a documented per-release step). The freeze/verify logic is
fully exercised here against fixtures.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from andbench.partition import Partition

#: Version of the lockfile format, so a future change is detectable.
LOCK_VERSION = 1


def pool_hash(ids: Iterable[str]) -> str:
    """Stable SHA-256 of a pool: sorted, newline-joined ``doc_id`` list."""
    joined = "\n".join(sorted(ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class PartitionLock(BaseModel):
    """The committed, auditable fingerprint of a frozen partition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lock_version: int = Field(default=LOCK_VERSION)
    seed: int
    bench_fraction: float
    n_train: int = Field(ge=0)
    n_bench: int = Field(ge=0)
    pool_train_sha256: str
    pool_bench_sha256: str

    @property
    def total(self) -> int:
        return self.n_train + self.n_bench

    @classmethod
    def from_partition(cls, partition: Partition) -> Self:
        return cls(
            seed=partition.seed,
            bench_fraction=partition.bench_fraction,
            n_train=len(partition.train_ids),
            n_bench=len(partition.bench_ids),
            pool_train_sha256=pool_hash(partition.train_ids),
            pool_bench_sha256=pool_hash(partition.bench_ids),
        )


def write_lock(lock: PartitionLock, path: str | Path) -> Path:
    """Write the lockfile as canonical, sorted JSON."""
    path = Path(path)
    path.write_text(
        json.dumps(lock.model_dump(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_lock(path: str | Path) -> PartitionLock:
    """Load and validate a committed lockfile."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return PartitionLock.model_validate(json.load(handle))


def verify_against_lock(partition: Partition, lock: PartitionLock) -> list[str]:
    """Return a list of mismatch messages (empty == the partition matches the lock)."""
    current = PartitionLock.from_partition(partition)
    problems: list[str] = []

    if current.seed != lock.seed:
        problems.append(f"seed changed: lock={lock.seed} current={current.seed}")
    if current.bench_fraction != lock.bench_fraction:
        problems.append(
            f"bench_fraction changed: lock={lock.bench_fraction} current={current.bench_fraction}"
        )
    if current.pool_train_sha256 != lock.pool_train_sha256:
        problems.append(
            "pool_train hash mismatch "
            f"(lock={lock.pool_train_sha256[:12]}… current={current.pool_train_sha256[:12]}…); "
            f"train count lock={lock.n_train} current={current.n_train}"
        )
    if current.pool_bench_sha256 != lock.pool_bench_sha256:
        problems.append(
            "pool_bench hash mismatch "
            f"(lock={lock.pool_bench_sha256[:12]}… current={current.pool_bench_sha256[:12]}…); "
            f"bench count lock={lock.n_bench} current={current.n_bench}"
        )
    return problems
