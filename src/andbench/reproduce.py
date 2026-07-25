"""The one-command reproduction pipeline (B3.06, constitution P16).

P16 requires that "the full evaluation runs from **one documented command** on a
clean machine". This module is that command's body: it takes a **reproduction
bundle** (a directory of committed inputs) and replays every model-free stage of
the pipeline into a fresh output directory, in order, failing fast at the first
gate that does not hold:

1. ``validate`` — items against the schema and the declared areas;
2. ``partition-verify`` — the corpus manifest still hashes to the committed lock;
3. ``decontam-pass`` — no n-gram collision with the training corpus;
4. ``split`` — the stratified public/private export;
5. ``canary`` — the public export carries the canary GUID;
6. ``export-lm-eval`` / ``gen-lm-eval`` — the harness data files and task configs;
7. ``sanity`` — the statistical report over a results table;
8. ``andobert-metrics`` — factual / citation / honesty metrics from judge verdicts;
9. ``leaderboard`` — the published table by track and area, including the
   public-vs-private contamination column;
10. ``dataset-card`` — the Hugging Face card, which also gates on source
    permissions (P23);
11. ``publish-build`` — assembles the Hub dataset and Space folders and checks
    that no private item is in them (it never uploads);
12. ``checksums`` — SHA-256 of every artifact, optionally compared against the
    bundle's committed baseline.

The last stage is what makes the claim testable: a third party does not merely get *a*
result, they get **byte-identical** artifacts or a loud diff. Seeds come from
``configs/tracks.yaml`` and the committed lock, never from the clock.

What is deliberately **out** of this pipeline: running the models. Scoring MCQ
tracks needs LM Evaluation Harness plus model weights, and And-Obert needs a judge
LLM (an open gap), so those are documented commands in the README rather than
steps here. This pipeline consumes their *recorded outputs* — the results table
and the judge verdicts — which is exactly what a third party reproducing a
leaderboard row is handed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from andbench.canary import CANARY_GUID, dataset_has_canary
from andbench.card import (
    DEFAULT_SOURCES_PATH,
    load_sources,
    permission_problems,
    render_card,
    write_card,
)
from andbench.config import load_config, unknown_areas
from andbench.decontam_pass import run_pass_from_files
from andbench.harness.judge import load_verdicts_by_id, metrics_from_files
from andbench.harness.lm_eval import generate_configs, write_area_files, write_configs
from andbench.harness.stats import analyze, load_results, write_report
from andbench.leaderboard import (
    Leaderboard,
    build_leaderboard,
    load_andobert_rows,
    write_leaderboard,
)
from andbench.partition import load_manifest, partition_corpus, write_partition
from andbench.partition_lock import PartitionLock, load_lock, verify_against_lock
from andbench.publish import build_dataset_repo, build_space_repo, publish_problems
from andbench.schema import Item
from andbench.split import split_items, write_split
from andbench.validation import ValidationReport, validate_jsonl

#: The bundle's input contract. A release bundle uses these same names, so the
#: command that reproduces the sample also reproduces a real release.
BUNDLE_INPUTS: dict[str, str] = {
    "items": "items.jsonl",
    "corpus_manifest": "corpus-manifest.jsonl",
    "partition_lock": "partition.lock",
    "train_texts": "maia-train.txt",
    "mcq_results": "mcq-results.jsonl",
    "andobert_verdicts": "andobert-verdicts.jsonl",
}

#: The committed reproduction baseline. Needed only by ``--verify``, and absent by
#: definition the first time a bundle is built (it is produced *by* a run).
BASELINE_KEY = "expected_checksums"

#: Per-model And-Obert verdicts. **Optional**: a bundle without them still
#: reproduces, and the leaderboard simply renders no And-Obert column rather than
#: inventing one.
LEADERBOARD_VERDICTS_KEY = "leaderboard_verdicts"

#: Every file a bundle directory may hold.
BUNDLE_FILES: dict[str, str] = {
    **BUNDLE_INPUTS,
    LEADERBOARD_VERDICTS_KEY: "leaderboard-verdicts.jsonl",
    BASELINE_KEY: "expected-checksums.txt",
}

#: Name of the checksum manifest written into every run directory.
CHECKSUM_FILENAME = "checksums.txt"

#: The bundle shipped with the repo, so the command runs with no arguments.
DEFAULT_BUNDLE_DIR = "data/sample"

#: Dataset version stamped into the generated card. Bumped by an explicit release
#: decision, never derived from a clock — the card must be byte-reproducible.
DATASET_VERSION = "v0.1.0"

#: Where the harness data files live *relative to the run directory*. The
#: generated task configs point here, so a run directory is self-contained.
LM_EVAL_DATA_DIR = "lm_eval/data"


@dataclass(frozen=True)
class Bundle:
    """The committed inputs a reproduction consumes."""

    root: Path
    items: Path
    corpus_manifest: Path
    partition_lock: Path
    train_texts: Path
    mcq_results: Path
    andobert_verdicts: Path
    leaderboard_verdicts: Path
    expected_checksums: Path

    @classmethod
    def from_dir(cls, root: str | Path) -> Bundle:
        base = Path(root)
        return cls(root=base, **{key: base / name for key, name in BUNDLE_FILES.items()})

    def inputs(self) -> list[Path]:
        return [getattr(self, key) for key in BUNDLE_INPUTS]

    def missing(self, *, require_baseline: bool = False) -> list[Path]:
        """Absent files — checked up front so no stage half-runs.

        The baseline is only required when the run is asked to verify against it;
        a bundle is legitimately baseline-less until its first run produces one.
        """
        required = [*self.inputs()]
        if require_baseline:
            required.append(self.expected_checksums)
        return [path for path in required if not path.is_file()]


@dataclass(frozen=True)
class StageResult:
    """The outcome of one pipeline stage."""

    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        return f"[{'ok ' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class ReproductionReport:
    """Every stage that ran, in order."""

    out_dir: Path
    stages: list[StageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.stages) and all(stage.ok for stage in self.stages)

    def stage(self, name: str) -> StageResult | None:
        return next((s for s in self.stages if s.name == name), None)

    def first_failure(self) -> StageResult | None:
        return next((s for s in self.stages if not s.ok), None)

    def summary(self) -> str:
        lines = [stage.line() for stage in self.stages]
        verdict = (
            f"Reproduction OK — {len(self.stages)} stage(s), artifacts in {self.out_dir}"
            if self.ok
            else "Reproduction FAILED"
        )
        return "\n".join([*lines, verdict])


# --- checksums ------------------------------------------------------------


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 16) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def artifact_checksums(out_dir: str | Path) -> dict[str, str]:
    """Map every artifact's POSIX-relative path to its SHA-256, sorted by path.

    The manifest itself is skipped — it cannot contain its own hash.
    """
    root = Path(out_dir)
    return {
        rel: sha256_file(path)
        for path in _walk_files(root)
        if (rel := path.relative_to(root).as_posix()) != CHECKSUM_FILENAME
    }


def write_checksums(checksums: Mapping[str, str], path: str | Path) -> Path:
    """Write a ``sha256␠␠relpath`` manifest, sorted by path (sha256sum format)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{checksums[rel]}  {rel}\n" for rel in sorted(checksums))
    target.write_text(body, encoding="utf-8")
    return target


def load_checksums(path: str | Path) -> dict[str, str]:
    """Read a checksum manifest. Raises ``ValueError`` on a malformed line."""
    source = Path(path)
    checksums: dict[str, str] = {}
    for lineno, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        digest, _, rel = line.partition("  ")
        if not rel or len(digest) != 64:
            raise ValueError(f"{source}:{lineno}: malformed checksum line: {raw!r}")
        checksums[rel] = digest
    return checksums


def compare_checksums(expected: Mapping[str, str], actual: Mapping[str, str]) -> list[str]:
    """Diff two checksum maps into human-readable problems (empty == identical)."""
    problems: list[str] = []
    for rel in sorted(set(expected) | set(actual)):
        want, got = expected.get(rel), actual.get(rel)
        if want is None:
            problems.append(f"unexpected artifact: {rel} ({got[:12] if got else '?'}…)")
        elif got is None:
            problems.append(f"missing artifact: {rel}")
        elif want != got:
            problems.append(f"changed: {rel} (expected {want[:12]}…, got {got[:12]}…)")
    return problems


# --- stages ---------------------------------------------------------------


def _validate_stage(bundle: Bundle, tracks_config: Path) -> tuple[StageResult, ValidationReport]:
    report = validate_jsonl(bundle.items)
    if not report.ok:
        return StageResult("validate", False, report.summary()), report

    violations = unknown_areas(report.items, load_config(tracks_config))
    if violations:
        detail = "; ".join(str(v) for v in violations)
        return StageResult("validate", False, detail), report

    return (
        StageResult("validate", True, f"{len(report.items)} item(s) valid, areas declared"),
        report,
    )


def _partition_verify_stage(
    bundle: Bundle, out_dir: Path
) -> tuple[StageResult, PartitionLock | None]:
    """Verify the partition and hand back the lock the card must publish."""
    docs = load_manifest(bundle.corpus_manifest)
    lock = load_lock(bundle.partition_lock)
    partition = partition_corpus(docs, bench_fraction=lock.bench_fraction, seed=lock.seed)
    problems = verify_against_lock(partition, lock)
    if problems:
        return StageResult("partition-verify", False, "; ".join(problems)), None

    write_partition(partition, out_dir / "pools")
    return (
        StageResult(
            "partition-verify",
            True,
            f"{lock.n_train} train / {lock.n_bench} bench match the lock (seed {lock.seed})",
        ),
        lock,
    )


def _decontam_stage(bundle: Bundle, out_dir: Path) -> StageResult:
    artifacts = run_pass_from_files(bundle.items, bundle.train_texts, out_dir / "decontam")
    report = artifacts.report
    if not report.clean:
        return StageResult("decontam-pass", False, report.summary())
    return StageResult("decontam-pass", True, f"{report.checked} item(s) clean at n={report.n}")


def _split_stage(
    items_report: ValidationReport, out_dir: Path, tracks_config: Path
) -> tuple[StageResult, list[Item]]:
    """Write the exports and return the **released** items, ``public`` flags stamped.

    Downstream stages take these rather than the validated input, so the
    leaderboard's public/private columns describe the dataset that was actually
    released instead of whatever the pre-split file happened to declare.
    """
    config = load_config(tracks_config)
    result = split_items(
        items_report.items,
        public_fraction=config.split.public_fraction,
        seed=config.split.seed,
    )
    write_split(
        result,
        out_dir / "dataset" / "andbench-public.jsonl",
        out_dir / "dataset" / "andbench-private.jsonl",
    )
    stage = StageResult(
        "split",
        True,
        f"{len(result.public)} public ({result.actual_public_fraction:.1%}) / "
        f"{len(result.private)} private, seed {config.split.seed}",
    )
    return stage, [*result.public, *result.private]


def _canary_stage(out_dir: Path) -> StageResult:
    public = out_dir / "dataset" / "andbench-public.jsonl"
    if not dataset_has_canary(public, CANARY_GUID):
        return StageResult("canary", False, f"public export is missing {CANARY_GUID}")
    return StageResult("canary", True, f"public export carries {CANARY_GUID}")


def _export_lm_eval_stage(items_report: ValidationReport, out_dir: Path) -> StageResult:
    written = write_area_files(items_report.items, out_dir / LM_EVAL_DATA_DIR)
    if not written:
        return StageResult("export-lm-eval", False, "no MCQ items to export")
    return StageResult("export-lm-eval", True, f"{len(written)} per-area MCQ file(s)")


def _gen_lm_eval_stage(out_dir: Path, tracks_config: Path) -> StageResult:
    configs = generate_configs(load_config(tracks_config), data_dir=LM_EVAL_DATA_DIR)
    written = write_configs(configs, out_dir / "lm_eval" / "configs")
    return StageResult("gen-lm-eval", True, f"{len(written)} task config(s)")


def _sanity_stage(bundle: Bundle, items_report: ValidationReport, out_dir: Path) -> StageResult:
    results = load_results(bundle.mcq_results)
    if not results:
        return StageResult("sanity", False, f"no results in {bundle.mcq_results}")
    report = analyze(items_report.items, results)
    write_report(report, out_dir / "analysis" / "sanity-report.json")
    return StageResult("sanity", True, report.summary())


def _andobert_stage(bundle: Bundle, items_report: ValidationReport, out_dir: Path) -> StageResult:
    verdicts = load_verdicts_by_id(bundle.andobert_verdicts)
    try:
        metrics = metrics_from_files(items_report.items, verdicts)
    except ValueError as exc:
        return StageResult("andobert-metrics", False, str(exc))

    path = out_dir / "analysis" / "andobert-metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return StageResult("andobert-metrics", True, metrics.summary())


def _leaderboard_stage(
    bundle: Bundle, released: Sequence[Item], out_dir: Path
) -> tuple[StageResult, Leaderboard | None]:
    obert_rows = (
        load_andobert_rows(bundle.leaderboard_verdicts)
        if bundle.leaderboard_verdicts.is_file()
        else []
    )
    board = build_leaderboard(released, load_results(bundle.mcq_results), obert_rows)
    write_leaderboard(
        board,
        out_dir / "leaderboard" / "leaderboard.json",
        out_dir / "leaderboard" / "leaderboard.md",
    )
    if not board.ok:
        return StageResult("leaderboard", False, "; ".join(board.problems)), None
    detail = f"{len(board.rows)} model(s)"
    if board.warnings:
        detail += f", {len(board.warnings)} caveat(s)"
    return StageResult("leaderboard", True, detail), board


def _card_stage(
    released: Sequence[Item],
    out_dir: Path,
    tracks_config: Path,
    sources_config: Path,
    *,
    lock: PartitionLock,
    decontam_clean: bool,
) -> StageResult:
    sources = load_sources(sources_config)
    problems = permission_problems(released, sources)
    if problems:
        return StageResult("dataset-card", False, "; ".join(problems))

    board_path = out_dir / "leaderboard" / "leaderboard.md"
    markdown = render_card(
        released,
        load_config(tracks_config),
        sources,
        version=DATASET_VERSION,
        lock=lock,
        decontam_clean=decontam_clean,
        leaderboard_markdown=(
            board_path.read_text(encoding="utf-8") if board_path.is_file() else None
        ),
    )
    path = write_card(markdown, out_dir / "dataset-card" / "README.md")
    return StageResult(
        "dataset-card",
        True,
        f"{len(released)} item(s), {len(sources.sources)} declared source(s) → {path.name}",
    )


def _publish_stage(released: Sequence[Item], board: Leaderboard, out_dir: Path) -> StageResult:
    """Assemble the Hub folders and check them. Never uploads."""
    card = (out_dir / "dataset-card" / "README.md").read_text(encoding="utf-8")
    dataset_dir = out_dir / "publish" / "dataset"
    space_dir = out_dir / "publish" / "space"

    build_dataset_repo(released, card, dataset_dir)
    build_space_repo(board, space_dir, version=DATASET_VERSION)

    problems = publish_problems(dataset_dir, space_dir)
    if problems:
        return StageResult("publish-build", False, "; ".join(problems))
    n_public = sum(1 for item in released if item.public)
    return StageResult(
        "publish-build",
        True,
        f"{n_public} public item(s) staged for the Hub, {len(released) - n_public} withheld",
    )


def _checksum_stage(bundle: Bundle, out_dir: Path, verify: bool) -> StageResult:
    actual = artifact_checksums(out_dir)
    write_checksums(actual, out_dir / CHECKSUM_FILENAME)
    if not verify:
        return StageResult("checksums", True, f"{len(actual)} artifact(s) hashed (not verified)")

    expected = load_checksums(bundle.expected_checksums)
    problems = compare_checksums(expected, actual)
    if problems:
        return StageResult(
            "checksums",
            False,
            f"{len(problems)} artifact(s) differ from the committed baseline:\n"
            + "\n".join(f"  - {p}" for p in problems),
        )
    return StageResult("checksums", True, f"{len(actual)} artifact(s) match the committed baseline")


# --- the pipeline ---------------------------------------------------------


def run_reproduction(
    bundle: Bundle,
    out_dir: str | Path,
    *,
    tracks_config: str | Path,
    sources_config: str | Path = DEFAULT_SOURCES_PATH,
    verify: bool = False,
) -> ReproductionReport:
    """Replay every model-free stage of the pipeline; fail fast at the first gate.

    ``out_dir`` is wiped first: a reproduction must not be able to pass on a stale
    artifact left by an earlier run.
    """
    out = Path(out_dir)
    tracks = Path(tracks_config)
    sources = Path(sources_config)
    report = ReproductionReport(out_dir=out)

    missing = bundle.missing(require_baseline=verify)
    if missing:
        report.stages.append(
            StageResult(
                "bundle",
                False,
                f"{len(missing)} missing input(s): " + ", ".join(str(p) for p in missing),
            )
        )
        return report
    report.stages.append(
        StageResult("bundle", True, f"{len(bundle.inputs())} input(s) from {bundle.root}")
    )

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    validate, items_report = _validate_stage(bundle, tracks)
    report.stages.append(validate)
    if not validate.ok:
        return report

    partition, lock = _partition_verify_stage(bundle, out)
    report.stages.append(partition)
    if not partition.ok or lock is None:
        return report

    decontam = _decontam_stage(bundle, out)
    report.stages.append(decontam)
    if not decontam.ok:
        return report

    # The split is sequenced explicitly because what it releases — items with their
    # `public` flag stamped — is what every later stage must score.
    split, released = _split_stage(items_report, out, tracks)
    report.stages.append(split)
    if not split.ok:
        return report

    before_board: tuple[Callable[[], StageResult], ...] = (
        lambda: _canary_stage(out),
        lambda: _export_lm_eval_stage(items_report, out),
        lambda: _gen_lm_eval_stage(out, tracks),
        lambda: _sanity_stage(bundle, items_report, out),
        lambda: _andobert_stage(bundle, items_report, out),
    )
    for stage in before_board:
        result = stage()
        report.stages.append(result)
        if not result.ok:
            return report

    # The leaderboard is sequenced explicitly because the Space renders it.
    leaderboard, board = _leaderboard_stage(bundle, released, out)
    report.stages.append(leaderboard)
    if not leaderboard.ok or board is None:
        return report

    after_board: tuple[Callable[[], StageResult], ...] = (
        # Reaching here means the decontamination gate passed, so the card may say
        # so: the pipeline stops at the first failure.
        lambda: _card_stage(released, out, tracks, sources, lock=lock, decontam_clean=True),
        lambda: _publish_stage(released, board, out),
        lambda: _checksum_stage(bundle, out, verify),
    )
    for stage in after_board:
        result = stage()
        report.stages.append(result)
        if not result.ok:
            return report

    return report
