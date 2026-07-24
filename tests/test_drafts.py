"""Tests for the semi-automatic draft pipeline (B2.02)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from andbench.drafts import (
    DRAFT_GUIDE,
    Draft,
    DraftDecision,
    DraftProposal,
    SourceDoc,
    accepted,
    build_prompt,
    draft_to_item,
    load_queue,
    parse_proposals,
    propose_drafts,
    write_queue,
)
from andbench.schema import Track


def _proposal(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "question": "Quin riu travessa Andorra la Vella?",
        "choices": ["Valira", "Segre", "Ebre", "Garona"],
        "answer": 0,
        "area": "geografia",
        "difficulty": 1,
        "rationale": "La font diu que la Valira travessa la capital.",
    }
    base.update(overrides)
    return base


class FakeModel:
    """Deterministic model: returns a preset payload regardless of the prompt."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.last_prompt: str | None = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.payload


# --- proposal validation -------------------------------------------------


def test_valid_proposal() -> None:
    p = DraftProposal.model_validate(_proposal())
    assert p.answer == 0


def test_proposal_wrong_choice_count_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly 4 choices"):
        DraftProposal.model_validate(_proposal(choices=["a", "b", "c"]))


def test_proposal_duplicate_choices_rejected() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        DraftProposal.model_validate(_proposal(choices=["a", "a", "b", "c"]))


def test_proposal_answer_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        DraftProposal.model_validate(_proposal(answer=5))


# --- prompt --------------------------------------------------------------


def test_prompt_embeds_guide_and_source() -> None:
    doc = SourceDoc(doc_id="pool_bench/geo/valira.md", text="La Valira travessa la capital.")
    prompt = build_prompt(doc, 3)
    assert DRAFT_GUIDE in prompt
    assert "La Valira travessa" in prompt
    assert "3 multiple-choice" in prompt


def test_prompt_rejects_non_positive_n() -> None:
    doc = SourceDoc(doc_id="x", text="y")
    with pytest.raises(ValueError, match="n must be positive"):
        build_prompt(doc, 0)


# --- parsing -------------------------------------------------------------


def test_parse_valid_array() -> None:
    raw = json.dumps([_proposal(), _proposal(answer=1)])
    proposals, errors = parse_proposals(raw)
    assert len(proposals) == 2
    assert errors == []


def test_parse_collects_invalid_without_dropping_silently() -> None:
    raw = json.dumps([_proposal(), _proposal(choices=["only", "three", "here"])])
    proposals, errors = parse_proposals(raw)
    assert len(proposals) == 1
    assert len(errors) == 1


def test_parse_non_json() -> None:
    proposals, errors = parse_proposals("not json at all")
    assert proposals == []
    assert "not valid JSON" in errors[0]


def test_parse_non_array() -> None:
    proposals, errors = parse_proposals(json.dumps({"question": "x"}))
    assert proposals == []
    assert "must be a JSON array" in errors[0]


# --- generation seam -----------------------------------------------------


def test_propose_drafts_with_fake_model() -> None:
    doc = SourceDoc(doc_id="pool_bench/geo/valira.md", text="La Valira.", track=Track.AND_CONEIX)
    model = FakeModel(json.dumps([_proposal(), _proposal(answer=2)]))
    drafts, errors = propose_drafts(doc, model, 2)
    assert errors == []
    assert len(drafts) == 2
    assert all(d.decision is DraftDecision.PENDING for d in drafts)
    assert all(d.source_doc_id == doc.doc_id for d in drafts)
    assert model.last_prompt is not None and DRAFT_GUIDE in model.last_prompt


# --- review queue --------------------------------------------------------


def test_queue_roundtrip_and_human_edit(tmp_path: Path) -> None:
    doc = SourceDoc(doc_id="d1", text="t")
    model = FakeModel(json.dumps([_proposal(), _proposal(answer=1), _proposal(answer=2)]))
    drafts, _ = propose_drafts(doc, model, 3)
    path = write_queue(drafts, tmp_path / "queue.jsonl")

    # Simulate a human editing decisions in the sheet.
    loaded = load_queue(path)
    loaded[0].decision = DraftDecision.ACCEPT
    loaded[1].decision = DraftDecision.REJECT
    loaded[2].decision = DraftDecision.EDIT
    write_queue(loaded, path)

    reviewed = load_queue(path)
    keep = accepted(reviewed)
    assert len(keep) == 2  # accept + edit, not reject


# --- conversion to items -------------------------------------------------


def test_accepted_draft_converts_to_item() -> None:
    draft = Draft(
        source_doc_id="pool_bench/geo/valira.md",
        track=Track.AND_CONEIX,
        proposal=DraftProposal.model_validate(_proposal()),
        decision=DraftDecision.ACCEPT,
    )
    item = draft_to_item(draft, item_id="and-coneix-0001", author="alice", verifier="bob")
    assert item.id == "and-coneix-0001"
    assert item.verifier == "bob"


def test_convert_enforces_author_ne_verifier() -> None:
    draft = Draft(
        source_doc_id="x",
        track=Track.AND_CONEIX,
        proposal=DraftProposal.model_validate(_proposal()),
        decision=DraftDecision.ACCEPT,
    )
    with pytest.raises(ValidationError, match="different person"):
        draft_to_item(draft, item_id="and-coneix-0002", author="sam", verifier="sam")


def test_pending_draft_cannot_convert() -> None:
    draft = Draft(
        source_doc_id="x",
        track=Track.AND_CONEIX,
        proposal=DraftProposal.model_validate(_proposal()),
    )
    with pytest.raises(ValueError, match="only accepted/edited"):
        draft_to_item(draft, item_id="x", author="a", verifier="b")
