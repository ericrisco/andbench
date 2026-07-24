"""Tests for the track configuration loader and release-time checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from andbench.config import (
    QuotaReport,
    TracksConfig,
    load_config,
    quota_report,
    unknown_areas,
)
from andbench.schema import Item, Track

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "tracks.yaml"


def _mcq(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "and-coneix-0001",
        "track": "and-coneix",
        "area": "historia",
        "question": "En quin any es va signar el Pareatge?",
        "choices": ["1278", "1288", "1519", "1993"],
        "answer": 0,
        "difficulty": 2,
        "source_doc_id": "pool_bench/hist/pareatge.md",
        "author": "alice",
        "verifier": "bob",
        "public": True,
        "tags": [],
    }
    base.update(overrides)
    return base


def _minimal_config() -> dict[str, Any]:
    """A structurally valid config covering all four schema tracks."""
    return {
        "version": 1,
        "traps": {"target_fraction": 0.10, "tolerance": 0.03},
        "split": {"public_fraction": 0.85, "private_fraction": 0.15, "seed": 1},
        "tracks": {
            "and-coneix": {
                "label": "And-Coneix",
                "description": "x",
                "item_budget": {"min": 1, "max": 2},
                "areas": {"historia": {"label": "història", "min": 1}},
            },
            "and-llengua": {
                "label": "And-Llengua",
                "description": "x",
                "item_budget": {"min": 1, "max": 2},
                "areas": {"lexic": {"label": "lèxic", "min": 1}},
            },
            "and-cotidia": {
                "label": "And-Cotidià",
                "description": "x",
                "item_budget": {"min": 1, "max": 2},
                "areas": {"festes": {"label": "festes", "min": 1}},
            },
            "and-obert": {
                "label": "And-Obert",
                "description": "x",
                "item_budget": {"min": 1, "max": 2},
                "areas": {"historia": {"label": "història", "min": 1}},
            },
        },
    }


# --- the shipped config --------------------------------------------------


def test_shipped_config_loads_and_covers_all_tracks() -> None:
    config = load_config(CONFIG_PATH)
    assert {t.value for t in Track} == set(config.tracks)
    # And-Coneix carries the five canonical areas (PRD §3).
    assert config.allowed_areas(Track.AND_CONEIX) == frozenset(
        {
            "historia",
            "institucions-i-dret",
            "geografia",
            "cultura-i-tradicions",
            "societat-i-economia",
        }
    )


def test_shipped_config_budget_sum_meets_release_floor() -> None:
    config = load_config(CONFIG_PATH)
    total_min = sum(config.track(t).item_budget.min for t in Track)
    assert total_min >= 800  # DoD B-O1


def test_mcq_tracks_derived_from_schema() -> None:
    config = load_config(CONFIG_PATH)
    assert set(config.mcq_tracks) == {
        Track.AND_CONEIX,
        Track.AND_LLENGUA,
        Track.AND_COTIDIA,
    }


# --- internal consistency ------------------------------------------------


def test_config_missing_a_track_rejected() -> None:
    raw = _minimal_config()
    del raw["tracks"]["and-obert"]
    with pytest.raises(ValidationError, match="match the schema exactly"):
        TracksConfig.model_validate(raw)


def test_config_unexpected_track_rejected() -> None:
    raw = _minimal_config()
    raw["tracks"]["and-extra"] = raw["tracks"]["and-coneix"]
    with pytest.raises(ValidationError, match="match the schema exactly"):
        TracksConfig.model_validate(raw)


def test_budget_min_gt_max_rejected() -> None:
    raw = _minimal_config()
    raw["tracks"]["and-coneix"]["item_budget"] = {"min": 5, "max": 2}
    with pytest.raises(ValidationError, match="exceeds max"):
        TracksConfig.model_validate(raw)


def test_split_fractions_must_sum_to_one() -> None:
    raw = _minimal_config()
    raw["split"] = {"public_fraction": 0.8, "private_fraction": 0.1, "seed": 1}
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        TracksConfig.model_validate(raw)


def test_bad_area_slug_rejected() -> None:
    raw = _minimal_config()
    raw["tracks"]["and-coneix"]["areas"] = {"Bad Slug": {"label": "x", "min": 1}}
    with pytest.raises(ValidationError, match="slug"):
        TracksConfig.model_validate(raw)


# --- dataset checks ------------------------------------------------------


def test_unknown_area_detected() -> None:
    config = load_config(CONFIG_PATH)
    good = Item.model_validate(_mcq(area="historia"))
    bad = Item.model_validate(_mcq(id="x", area="astrofisica"))
    violations = unknown_areas([good, bad], config)
    assert len(violations) == 1
    assert violations[0].item_id == "x"
    assert "astrofisica" in str(violations[0])


def test_quota_shortfalls_reported_for_small_set() -> None:
    config = load_config(CONFIG_PATH)
    report = quota_report([Item.model_validate(_mcq())], config)
    assert not report.ok
    # Under budget on every track, so the shipped floors all show up.
    assert any("budget min" in s for s in report.shortfalls)


def test_quota_report_ok_summary() -> None:
    assert QuotaReport().ok
    assert QuotaReport().summary() == "Quotas met."


def test_trap_fraction_flagged_when_absent() -> None:
    config = load_config(CONFIG_PATH)
    # 10 MCQ items, zero traps → fraction 0% is outside 10% ± 3%.
    items = [Item.model_validate(_mcq(id=f"and-coneix-{i:04d}")) for i in range(10)]
    report = quota_report(items, config)
    assert any("trap fraction" in s for s in report.shortfalls)
