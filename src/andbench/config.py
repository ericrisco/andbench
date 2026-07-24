"""Typed loader for ``configs/tracks.yaml`` (B0.03).

The YAML declares each track's sub-areas and the per-track / per-area item
budgets. Item *forms* deliberately live in :data:`andbench.schema.TRACK_FORMS`,
not here; :class:`TracksConfig` asserts that its track set matches the schema
exactly so the config and the schema can never drift.

It also provides the release-time checks that need the taxonomy: whether every
item's ``area`` is a declared slug, and whether a set of items meets the quotas.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from andbench.schema import TRACK_FORMS, Item, ItemForm, Track

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

#: Area slug pattern: lowercase words joined by single hyphens.
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class AreaSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: NonEmptyStr
    min: int = Field(ge=0, description="Minimum verified items required in this area.")


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def _min_le_max(self) -> Self:
        if self.min > self.max:
            raise ValueError(f"item_budget min ({self.min}) exceeds max ({self.max})")
        return self


class TrackSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: NonEmptyStr
    description: NonEmptyStr
    item_budget: Budget
    areas: dict[str, AreaSpec]

    @model_validator(mode="after")
    def _check_areas(self) -> Self:
        if not self.areas:
            raise ValueError("a track must declare at least one area")
        for slug in self.areas:
            if not SLUG_PATTERN.match(slug):
                raise ValueError(f"area slug {slug!r} must match {SLUG_PATTERN.pattern}")
        return self


class TrapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_fraction: float = Field(gt=0.0, lt=1.0)
    tolerance: float = Field(ge=0.0, lt=1.0)


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    public_fraction: float = Field(gt=0.0, lt=1.0)
    private_fraction: float = Field(gt=0.0, lt=1.0)
    seed: int

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> Self:
        total = self.public_fraction + self.private_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"public + private fractions must sum to 1.0, got {total}")
        return self


class TracksConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=1)
    traps: TrapConfig
    split: SplitConfig
    tracks: dict[str, TrackSpec]

    @model_validator(mode="after")
    def _tracks_match_schema(self) -> Self:
        declared = set(self.tracks)
        expected = {t.value for t in Track}
        if declared != expected:
            missing = expected - declared
            extra = declared - expected
            raise ValueError(
                "tracks in config must match the schema exactly "
                f"(missing: {sorted(missing)}, unexpected: {sorted(extra)})"
            )
        return self

    # --- accessors -------------------------------------------------------

    def track(self, track: Track | str) -> TrackSpec:
        key = track.value if isinstance(track, Track) else track
        return self.tracks[key]

    def allowed_areas(self, track: Track | str) -> frozenset[str]:
        return frozenset(self.track(track).areas)

    @property
    def mcq_tracks(self) -> list[Track]:
        return [t for t in Track if ItemForm.MCQ in TRACK_FORMS[t]]


def load_config(path: str | Path) -> TracksConfig:
    """Load and fully validate the track configuration."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TracksConfig.model_validate(raw)


# --- release-time dataset checks -----------------------------------------


@dataclass(frozen=True)
class AreaViolation:
    item_id: str
    track: str
    area: str

    def __str__(self) -> str:
        return f"item {self.item_id!r}: area {self.area!r} is not declared for track {self.track!r}"


def unknown_areas(items: list[Item], config: TracksConfig) -> list[AreaViolation]:
    """Return every item whose ``area`` is not a declared slug for its track."""
    violations: list[AreaViolation] = []
    for item in items:
        if item.area not in config.allowed_areas(item.track):
            violations.append(AreaViolation(item.id, item.track.value, item.area))
    return violations


@dataclass
class QuotaReport:
    """Whether a set of items meets the track/area budgets and trap fraction."""

    shortfalls: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.shortfalls

    def summary(self) -> str:
        if self.ok:
            return "Quotas met."
        return "Quota shortfalls:\n" + "\n".join(f"  - {s}" for s in self.shortfalls)


def quota_report(items: list[Item], config: TracksConfig) -> QuotaReport:
    """Compare item counts against per-track budgets, per-area minima, and trap fraction."""
    report = QuotaReport()
    by_track: dict[str, list[Item]] = {t.value: [] for t in Track}
    for item in items:
        by_track[item.track.value].append(item)

    for track in Track:
        spec = config.track(track)
        track_items = by_track[track.value]
        if len(track_items) < spec.item_budget.min:
            report.shortfalls.append(
                f"{track.value}: {len(track_items)} items < budget min {spec.item_budget.min}"
            )
        area_counts = Counter(i.area for i in track_items)
        for slug, area in spec.areas.items():
            have = area_counts.get(slug, 0)
            if have < area.min:
                report.shortfalls.append(f"{track.value}/{slug}: {have} items < min {area.min}")

    _check_trap_fraction(by_track, config, report)
    return report


def _check_trap_fraction(
    by_track: dict[str, list[Item]], config: TracksConfig, report: QuotaReport
) -> None:
    target = config.traps.target_fraction
    tol = config.traps.tolerance
    mcq_items = [i for t in config.mcq_tracks for i in by_track[t.value]]
    if not mcq_items:
        return
    frac = sum(1 for i in mcq_items if i.is_trap) / len(mcq_items)
    if abs(frac - target) > tol:
        report.shortfalls.append(
            f"trap fraction {frac:.2%} of MCQ items is outside {target:.0%} ± {tol:.0%}"
        )
