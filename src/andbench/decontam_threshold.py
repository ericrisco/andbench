"""Calibrate the embedding decontamination threshold (P10).

The n-gram check needs no threshold: two texts either share a 13-token run or they
do not. The embedding check needs a cosine cut-off, and the committed default of 0.9
was asserted rather than measured — which is exactly the sort of number this project
removes elsewhere. A cut-off that is too low flags every same-topic item as a
collision and the release never ships; too high and the paraphrase check does
nothing while appearing to run, which is worse.

So: label pairs, sweep the cut-off, and report what each one costs. The pairs that
matter are the **hard negatives** — same topic, same vocabulary, different content —
because they set the ceiling. Anyone can separate a paraphrase from a recipe.

The output is a recommendation, not a decision: it prints the trade-off at each
candidate and names the value that maximises F1, with **recall weighted above
precision** on request. A missed collision ships a contaminated item; a false
positive costs somebody ten minutes of reading. Those are not symmetric errors, and
the tool says so rather than quietly optimising the wrong thing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from andbench.decontam import Embedder, cosine

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

#: The committed labelled set.
DEFAULT_PAIRS_PATH = "configs/decontam_pairs.yaml"

#: Cut-offs to try. Coarse enough to read, fine enough to choose from.
CANDIDATE_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.01 * i, 2) for i in range(50))

#: Missing a collision ships contamination; a false positive costs a human ten
#: minutes. Weighting recall above precision is the honest default for this gate.
DEFAULT_BETA = 2.0


class LabelledPair(BaseModel):
    """Two texts and whether they should count as a collision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    a: NonEmptyStr
    b: NonEmptyStr
    collision: bool
    note: str = ""


class PairSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=1)
    pairs: list[LabelledPair]

    @property
    def positives(self) -> int:
        return sum(1 for p in self.pairs if p.collision)

    @property
    def negatives(self) -> int:
        return len(self.pairs) - self.positives


def load_pairs(path: str | Path = DEFAULT_PAIRS_PATH) -> PairSet:
    """Load the labelled calibration pairs."""
    with Path(path).open(encoding="utf-8") as handle:
        return PairSet.model_validate(yaml.safe_load(handle))


@dataclass(frozen=True)
class ThresholdScore:
    """What one cut-off would do to the labelled set."""

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 1.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 1.0

    def f_beta(self, beta: float = DEFAULT_BETA) -> float:
        """F-measure with recall weighted ``beta`` times precision."""
        precision, recall = self.precision, self.recall
        if precision == 0.0 and recall == 0.0:
            return 0.0
        weight = beta * beta
        return (1 + weight) * precision * recall / (weight * precision + recall)

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f_beta": round(self.f_beta(), 6),
        }

    def line(self) -> str:
        return (
            f"  {self.threshold:.2f}  precision {self.precision:6.1%}  recall {self.recall:6.1%}  "
            f"missed {self.false_negatives}  false alarms {self.false_positives}"
        )


@dataclass
class Calibration:
    """The sweep, and what it recommends."""

    model: str
    scores: list[ThresholdScore]
    beta: float
    similarities: list[tuple[float, bool]]

    @property
    def recommended(self) -> ThresholdScore | None:
        """The cut-off to ship.

        When the classes separate cleanly, this is the **midpoint of the gap**, not
        the lowest cut-off that scores perfectly. Every threshold inside a clean gap
        scores identically on *these* pairs, so the score cannot choose between
        them — but a cut-off sitting 0.01 above the highest negative will start
        firing on the first slightly-closer negative the real corpus produces.
        Maximum margin on both sides is the choice that survives new data.

        When the classes overlap there is no gap to sit in the middle of, so it falls
        back to the best F-beta, ties broken toward the lower (more catching) value.
        """
        if not self.scores:
            return None

        gap = self.margin
        if gap is not None and gap[0] < gap[1]:
            midpoint = (gap[0] + gap[1]) / 2.0
            return min(self.scores, key=lambda s: (abs(s.threshold - midpoint), s.threshold))
        return max(self.scores, key=lambda s: (s.f_beta(self.beta), -s.threshold))

    @property
    def perfect_separation(self) -> bool:
        """Whether some cut-off separates the two classes completely."""
        return any(s.false_positives == 0 and s.false_negatives == 0 for s in self.scores)

    @property
    def margin(self) -> tuple[float, float] | None:
        """(highest negative, lowest positive) — the gap a threshold has to sit in.

        Inverted bounds mean the classes overlap and no cut-off is clean; the width
        of the gap is how much confidence the recommendation deserves.
        """
        positives = [s for s, label in self.similarities if label]
        negatives = [s for s, label in self.similarities if not label]
        if not positives or not negatives:
            return None
        return (max(negatives), min(positives))

    def to_dict(self) -> dict[str, object]:
        recommended = self.recommended
        return {
            "model": self.model,
            "beta": self.beta,
            "recommended_threshold": None if recommended is None else recommended.threshold,
            "perfect_separation": self.perfect_separation,
            "margin": list(self.margin) if self.margin else None,
            "scores": [s.to_dict() for s in self.scores],
        }

    def summary(self) -> str:
        lines = [f"Threshold calibration with {self.model} (recall weighted {self.beta}x):"]
        recommended = self.recommended
        interesting = [
            s
            for s in self.scores
            if recommended is not None and abs(s.threshold - recommended.threshold) <= 0.05
        ]
        lines.extend(s.line() for s in interesting)

        if recommended is None:
            lines.append("No pairs to calibrate on.")
            return "\n".join(lines)

        lines.append(f"Recommended threshold: {recommended.threshold:.2f}")
        gap = self.margin
        if gap is not None:
            highest_negative, lowest_positive = gap
            if highest_negative < lowest_positive:
                lines.append(
                    f"Clean separation: negatives top out at {highest_negative:.3f}, positives "
                    f"start at {lowest_positive:.3f}. The recommendation is the midpoint of that "
                    "gap — every value inside it scores the same here, so the score cannot "
                    "choose, and maximum margin is what survives new data."
                )
            else:
                lines.append(
                    f"⚠️ The classes OVERLAP: a negative reaches {highest_negative:.3f} while a "
                    f"positive sits at {lowest_positive:.3f}. No cut-off is clean, so this "
                    "threshold trades one error for the other — add more hard negatives before "
                    "trusting it."
                )
        return "\n".join(lines)


def score_threshold(similarities: Sequence[tuple[float, bool]], threshold: float) -> ThresholdScore:
    """What this cut-off would have decided about each labelled pair."""
    tp = sum(1 for s, label in similarities if label and s >= threshold)
    fp = sum(1 for s, label in similarities if not label and s >= threshold)
    fn = sum(1 for s, label in similarities if label and s < threshold)
    tn = sum(1 for s, label in similarities if not label and s < threshold)
    return ThresholdScore(
        threshold=threshold,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


def pair_similarities(
    pairs: Sequence[LabelledPair], embedder: Embedder
) -> list[tuple[float, bool]]:
    """Cosine similarity of each pair, with its label. One batched embed call."""
    if not pairs:
        return []
    texts = [p.a for p in pairs] + [p.b for p in pairs]
    vectors = embedder.embed(texts)
    half = len(pairs)
    return [(cosine(vectors[i], vectors[half + i]), pairs[i].collision) for i in range(half)]


def calibrate(
    pairs: PairSet,
    embedder: Embedder,
    *,
    model_name: str = "unknown",
    beta: float = DEFAULT_BETA,
    thresholds: Sequence[float] = CANDIDATE_THRESHOLDS,
) -> Calibration:
    """Sweep the candidate cut-offs over the labelled pairs."""
    similarities = pair_similarities(pairs.pairs, embedder)
    return Calibration(
        model=model_name,
        scores=[score_threshold(similarities, t) for t in thresholds],
        beta=beta,
        similarities=similarities,
    )


def write_calibration(calibration: Calibration, path: str | Path) -> Path:
    """Write the sweep as canonical JSON, so a chosen threshold is auditable."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target
