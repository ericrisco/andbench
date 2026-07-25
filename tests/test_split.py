"""Tests for the public/private split (anti-contamination §3, B2.09)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from andbench.canary import dataset_has_canary
from andbench.schema import Item
from andbench.split import split_items, write_split


def _items(n_per_area: int = 40) -> list[Item]:
    items: list[Item] = []
    for area in ("historia", "geografia"):
        for i in range(n_per_area):
            items.append(
                Item.model_validate(
                    {
                        "id": f"and-coneix-{area}-{i:03d}",
                        "track": "and-coneix",
                        "area": area,
                        "question": f"pregunta {area} {i}?",
                        "choices": ["a", "b", "c", "d"],
                        "answer": i % 4,
                        "difficulty": 1,
                        "source_doc_id": f"pool_bench/{area}/{i}.md",
                        "author": "alice",
                        "verifier": "bob",
                        "public": True,
                        "tags": [],
                    }
                )
            )
    return items


def test_exhaustive_and_disjoint() -> None:
    items = _items()
    result = split_items(items)
    ids = {i.id for i in items}
    pub = {i.id for i in result.public}
    priv = {i.id for i in result.private}
    assert pub | priv == ids
    assert pub & priv == set()
    assert result.total == len(ids)


def test_public_flag_is_stamped_to_match_membership() -> None:
    """An export must never contradict itself: `public` follows the computed split."""
    result = split_items(_items())  # every input item declares public=True
    assert result.private, "the fixture must produce a non-empty private split"
    assert all(item.public is True for item in result.public)
    assert all(item.public is False for item in result.private)


def test_written_exports_carry_the_stamped_flag(tmp_path: Path) -> None:
    result = split_items(_items())
    paths = write_split(result, tmp_path / "pub.jsonl", tmp_path / "priv.jsonl")
    public_lines = paths["public"].read_text(encoding="utf-8").splitlines()[1:]  # skip canary
    assert all(json.loads(line)["public"] is True for line in public_lines)
    private_lines = [ln for ln in paths["private"].read_text().splitlines() if ln.strip()]
    assert all(json.loads(line)["public"] is False for line in private_lines)


def test_fraction_is_about_85_percent() -> None:
    result = split_items(_items(n_per_area=100))  # 200 items
    assert result.actual_public_fraction == pytest.approx(0.85, abs=0.01)


def test_stratified_each_area_split_proportionally() -> None:
    result = split_items(_items(n_per_area=40))
    for (_track, _area), (n_pub, n_priv) in result.strata.items():
        assert n_pub == 34  # round(40 * 0.85)
        assert n_priv == 6


def test_deterministic_and_order_independent() -> None:
    items = _items()
    a = split_items(items, seed=42)
    b = split_items(list(reversed(items)), seed=42)
    assert [i.id for i in a.public] == [i.id for i in b.public]
    assert [i.id for i in a.private] == [i.id for i in b.private]


def test_different_seed_changes_split() -> None:
    items = _items()
    a = split_items(items, seed=1)
    b = split_items(items, seed=2)
    assert [i.id for i in a.private] != [i.id for i in b.private]


def test_duplicate_ids_rejected() -> None:
    items = _items(n_per_area=1)
    with pytest.raises(ValueError, match="unique ids"):
        split_items([*items, items[0]])


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.2, 2.0])
def test_invalid_fraction_rejected(frac: float) -> None:
    with pytest.raises(ValueError, match="public_fraction"):
        split_items(_items(n_per_area=1), public_fraction=frac)


def test_write_split_public_has_canary_private_does_not(tmp_path: Path) -> None:
    result = split_items(_items())
    paths = write_split(
        result,
        tmp_path / "public" / "andbench.jsonl",
        tmp_path / "private" / "andbench-private.jsonl",
    )
    assert dataset_has_canary(paths["public"]) is True
    assert dataset_has_canary(paths["private"]) is False

    # Public export: first line is the canary, the rest are valid items.
    lines = paths["public"].read_text(encoding="utf-8").splitlines()
    assert "andbench_canary" in lines[0]
    reparsed = [Item.model_validate(json.loads(line)) for line in lines[1:]]
    assert len(reparsed) == len(result.public)


def test_private_count_matches(tmp_path: Path) -> None:
    result = split_items(_items())
    paths = write_split(result, tmp_path / "pub.jsonl", tmp_path / "priv.jsonl")
    priv_lines = [line for line in paths["private"].read_text().splitlines() if line.strip()]
    assert len(priv_lines) == len(result.private)
