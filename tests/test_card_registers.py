"""Tests for the errata register, per-release statistics and consented credits."""

from __future__ import annotations

from pathlib import Path

import pytest

from andbench.card import (
    DEFAULT_CONTRIBUTORS_PATH,
    DEFAULT_ERRATA_PATH,
    ContributorsConfig,
    ErrataRegister,
    ErratumKind,
    SourcesConfig,
    collect_credits,
    errata_problems,
    load_contributors,
    load_errata,
    render_card,
)
from andbench.config import load_config
from andbench.harness.stats import SanityReport
from andbench.schema import Item

ROOT = Path(__file__).resolve().parents[1]
TRACKS = load_config(ROOT / "configs" / "tracks.yaml")


def _sources() -> SourcesConfig:
    return SourcesConfig.model_validate(
        {
            "version": 1,
            "sources": [
                {
                    "id": "s",
                    "id_prefix": "doc-",
                    "label": "A source",
                    "licence": "CC-BY-4.0",
                    "permission": "own-work",
                }
            ],
        }
    )


def _item(item_id: str, *, author: str = "alice", verifier: str = "bob") -> Item:
    return Item.model_validate(
        {
            "id": item_id,
            "track": "and-coneix",
            "area": "geografia",
            "question": f"Pregunta {item_id}?",
            "choices": ["a", "b", "c", "d"],
            "answer": 0,
            "difficulty": 1,
            "source_doc_id": "doc-1",
            "author": author,
            "verifier": verifier,
            "public": True,
            "tags": [],
        }
    )


def _register(*entries: dict[str, str]) -> ErrataRegister:
    return ErrataRegister.model_validate({"version": 1, "errata": list(entries)})


def _erratum(item_id: str, kind: str = "corrected", version: str = "v1.1.0") -> dict[str, str]:
    return {
        "version": version,
        "item_id": item_id,
        "kind": kind,
        "change": "Distractor C replaced.",
        "reason": "Two options were defensible.",
    }


# --- the committed registers ----------------------------------------------


def test_the_committed_errata_register_loads_and_starts_empty() -> None:
    register = load_errata(ROOT / DEFAULT_ERRATA_PATH)
    assert register.version >= 1
    assert register.errata == []


def test_the_committed_contributor_register_loads() -> None:
    contributors = load_contributors(ROOT / DEFAULT_CONTRIBUTORS_PATH)
    assert {c.id for c in contributors.contributors} == {"fixture-author", "fixture-verifier"}


def test_the_fixture_contributors_are_not_credited() -> None:
    """They are placeholder strings, not people."""
    contributors = load_contributors(ROOT / DEFAULT_CONTRIBUTORS_PATH)
    assert all(not c.credit for c in contributors.contributors)


# --- errata contradictions -------------------------------------------------


def test_a_clean_register_has_no_problems() -> None:
    assert errata_problems(_register(_erratum("i-1")), [_item("i-1")]) == []


def test_correcting_an_item_that_is_not_shipping_is_a_contradiction() -> None:
    problems = errata_problems(_register(_erratum("ghost")), [_item("i-1")])
    assert len(problems) == 1
    assert "was corrected, but that item is not in the dataset" in problems[0]


def test_removing_an_item_that_is_still_shipping_is_a_contradiction() -> None:
    problems = errata_problems(_register(_erratum("i-1", kind="removed")), [_item("i-1")])
    assert len(problems) == 1
    assert "still" in problems[0]


def test_a_removed_item_stays_listed_without_complaint() -> None:
    """The policy: removed items stay in the register forever."""
    assert errata_problems(_register(_erratum("gone", kind="removed")), [_item("i-1")]) == []


def test_an_added_entry_is_informational_only() -> None:
    assert errata_problems(_register(_erratum("i-1", kind="added")), [_item("i-1")]) == []
    assert errata_problems(_register(_erratum("ghost", kind="added")), [_item("i-1")]) == []


def test_the_card_refuses_to_build_on_a_contradicted_register() -> None:
    with pytest.raises(ValueError, match="not in the dataset"):
        render_card(
            [_item("i-1")],
            TRACKS,
            _sources(),
            version="v1.0",
            errata=_register(_erratum("ghost")),
        )


def test_errata_render_with_their_kind() -> None:
    card = render_card(
        [_item("i-1")], TRACKS, _sources(), version="v1.1", errata=_register(_erratum("i-1"))
    )
    assert "**corrected**" in card
    assert "and-coneix" in card or "i-1" in card
    assert "Two options were defensible." in card


def test_an_empty_register_says_so() -> None:
    card = render_card([_item("i-1")], TRACKS, _sources(), version="v1.0", errata=_register())
    assert "_No errata recorded as of v1.0._" in card


@pytest.mark.parametrize("kind", ["corrected", "removed", "added"])
def test_every_kind_is_accepted(kind: str) -> None:
    assert _register(_erratum("i-1", kind=kind)).errata[0].kind is ErratumKind(kind)


def test_an_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        _register(_erratum("i-1", kind="tweaked"))


def test_an_erratum_without_a_reason_is_rejected() -> None:
    """An unexplained change is not a record."""
    with pytest.raises(ValueError):
        ErrataRegister.model_validate(
            {
                "version": 1,
                "errata": [
                    {"version": "v1.1", "item_id": "i-1", "kind": "corrected", "change": "x"}
                ],
            }
        )


# --- credits, and the consent default -------------------------------------


def _contributors(*rows: dict[str, object]) -> ContributorsConfig:
    return ContributorsConfig.model_validate({"version": 1, "contributors": list(rows)})


def test_a_consenting_contributor_is_named() -> None:
    credits = collect_credits(
        [_item("i-1", author="alice", verifier="bob")],
        _contributors(
            {"id": "alice", "display_name": "Alice A.", "role": "author", "credit": True},
            {"id": "bob", "display_name": "Bob B.", "role": "verifier", "credit": True},
        ),
    )
    assert [c.display_name for c in credits.named] == ["Alice A.", "Bob B."]
    assert credits.withheld == 0


def test_a_contributor_who_declined_is_counted_not_named() -> None:
    credits = collect_credits(
        [_item("i-1", author="alice", verifier="bob")],
        _contributors(
            {"id": "alice", "display_name": "Alice A.", "credit": True},
            {"id": "bob", "display_name": "Bob B.", "credit": False},
        ),
    )
    assert [c.display_name for c in credits.named] == ["Alice A."]
    assert credits.withheld == 1


def test_an_undeclared_contributor_is_withheld_by_default() -> None:
    """A name is personal data; the safe default is not to publish it."""
    credits = collect_credits([_item("i-1", author="alice", verifier="bob")], _contributors())
    assert credits.named == ()
    assert credits.withheld == 2


def test_the_card_publishes_the_withheld_count() -> None:
    card = render_card(
        [_item("i-1")],
        TRACKS,
        _sources(),
        version="v1.0",
        contributors=_contributors(),
    )
    assert "2 further contributor(s) are not named" in card
    assert "visible rather than silent" in card


def test_the_card_names_those_who_consented() -> None:
    card = render_card(
        [_item("i-1", author="alice", verifier="bob")],
        TRACKS,
        _sources(),
        version="v1.0",
        contributors=_contributors(
            {"id": "alice", "display_name": "Alice A.", "role": "author", "credit": True},
            {"id": "bob", "display_name": "Bob B.", "role": "verifier", "credit": True},
        ),
    )
    assert "Alice A." in card
    assert "Bob B." in card
    assert "not named here" not in card


def test_the_credits_section_is_omitted_without_a_register() -> None:
    assert "## Credits" not in render_card([_item("i-1")], TRACKS, _sources(), version="v1.0")


# --- per-release statistics ------------------------------------------------


def _sanity(**kwargs: object) -> SanityReport:
    report = SanityReport()
    report.accuracy_by_difficulty = {1: 0.9, 2: 0.7, 3: 0.4}
    report.accuracy_by_area = {"and-coneix/geografia": 0.75}
    report.seed_variance = {"m1": 0.0025}
    report.always_failed_ids = ["i-9"]
    report.always_passed_ids = ["i-1"]
    for key, value in kwargs.items():
        setattr(report, key, value)
    return report


def _card_with_stats(report: SanityReport) -> str:
    return render_card([_item("i-1")], TRACKS, _sources(), version="v1.0", sanity=report)


def test_statistics_render_accuracy_by_difficulty_and_area() -> None:
    card = _card_with_stats(_sanity())
    assert "## Per-release statistics" in card
    assert "1 — easy" in card
    assert "90.0%" in card
    assert "and-coneix/geografia" in card


def test_a_monotonic_difficulty_curve_is_reported_as_intended() -> None:
    card = _card_with_stats(_sanity())
    assert "behaving as intended" in card
    assert "⚠️" not in card.split("## Per-release statistics")[1].split("##")[0]


def test_a_non_monotonic_difficulty_curve_is_flagged() -> None:
    """If 'hard' items are easier than 'medium' ones, the labels are not tracking difficulty."""
    card = _card_with_stats(_sanity(accuracy_by_difficulty={1: 0.5, 2: 0.4, 3: 0.8}))
    assert "does **not** fall monotonically" in card


def test_review_candidate_ids_are_withheld() -> None:
    """A published list of items every model fails is a shopping list for training on them."""
    card = _card_with_stats(_sanity())
    assert "2 review candidate(s)" in card
    assert "i-9" not in card
    assert "shopping list" in card


def test_seed_variance_is_published() -> None:
    assert "0.0025" in _card_with_stats(_sanity())


def test_a_single_seed_run_says_so_rather_than_showing_zero() -> None:
    card = _card_with_stats(_sanity(seed_variance={}))
    assert "no variance estimate" in card


def test_the_statistics_section_is_omitted_without_a_report() -> None:
    assert "## Per-release statistics" not in render_card(
        [_item("i-1")], TRACKS, _sources(), version="v1.0"
    )


def test_the_credits_section_handles_an_item_set_with_no_people() -> None:
    """Defensive: a card built from zero items still renders a coherent section."""
    from andbench.card import Credits, _credits_section

    assert "_No contributors recorded._" in _credits_section(Credits(named=(), withheld=0))
