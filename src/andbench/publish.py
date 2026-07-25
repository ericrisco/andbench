"""Assemble and publish the Hugging Face repositories (B4.03, B4.04).

Two repositories come out of a release:

* the **dataset** repo — the generated card as ``README.md``, the public items as
  per-track JSONL the Hub's viewer can read, and the canary record;
* the **Space** repo — a static leaderboard (PLAN B4.04: "a static table suffices
  in v1"), which needs no runtime and therefore cannot rot.

Both are *assembled and checked locally first*, then uploaded. That order is the
whole design. An upload is the one irreversible step in this project: the private
split is a permanent over-fitting detector (anti-contamination §3) and pushing a
single private item destroys it forever — no un-publishing recovers a dataset
someone has already cloned. So :func:`publish_problems` re-reads what was actually
written and refuses to upload if a private item is present, rather than trusting
that the filter upstream did its job.

Uploading needs a token, so it goes through an injectable :class:`Uploader` seam and
**dry-run is the default**: with no uploader, the plan prints the exact ``hf upload``
commands and touches nothing.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from andbench.canary import CANARY_GUID, CanaryRecord
from andbench.leaderboard import MCQ_TRACKS, TRACK_LABELS, Leaderboard, ModelRow
from andbench.schema import Item

#: Default Hub repositories.
DEFAULT_DATASET_REPO = "ericrisco/andbench"
DEFAULT_SPACE_REPO = "ericrisco/andbench-leaderboard"

#: Where the per-track item files live inside the dataset repo. The card's
#: ``configs`` front matter points here.
DATA_DIR = "data"

#: The canary sits in its own file so the per-track files stay homogeneous for the
#: dataset viewer. The GUID is *also* in the card, which is what a scraper of the
#: whole repo picks up.
CANARY_FILE = "canary.jsonl"

SPACE_ENTRY_FILE = "index.html"


@runtime_checkable
class Uploader(Protocol):
    """Upload seam. A real implementation wraps ``huggingface_hub``."""

    def upload_folder(self, *, repo_id: str, repo_type: str, folder: Path) -> str: ...


# --- the dataset repository ----------------------------------------------


def build_dataset_repo(
    items: Sequence[Item], card_markdown: str, out_dir: str | Path
) -> list[Path]:
    """Write the dataset repo: card, public items per track/area, canary.

    Private items are dropped here, and :func:`publish_problems` independently
    verifies that they really are gone.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = [_write(root / "README.md", card_markdown)]

    buckets: dict[tuple[str, str], list[Item]] = {}
    for item in items:
        if not item.public:
            continue
        buckets.setdefault((item.track.value, item.area), []).append(item)

    for (track, area), group in sorted(buckets.items()):
        path = root / DATA_DIR / track / f"{area}.jsonl"
        body = "\n".join(i.model_dump_json() for i in sorted(group, key=lambda i: i.id)) + "\n"
        written.append(_write(path, body))

    written.append(_write(root / CANARY_FILE, CanaryRecord().to_jsonl() + "\n"))
    return written


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# --- the Space -------------------------------------------------------------


def space_frontmatter(*, title: str = "AndBench Leaderboard") -> dict[str, object]:
    """Static-Space configuration (``sdk: static`` serves ``index.html``)."""
    return {
        "title": title,
        "emoji": "🏔️",
        "colorFrom": "blue",
        "colorTo": "indigo",
        "sdk": "static",
        "pinned": False,
        "license": "apache-2.0",
    }


def _row_cells(row: ModelRow, suspicious_gap: float) -> list[str]:
    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value:.1%}"

    gap = row.contamination_gap
    gap_text = "—" if gap is None else f"{gap:+.1%}"
    if gap is not None and gap > suspicious_gap:
        gap_text += " ⚠️"
    return [
        row.model,
        *(
            pct(row.by_track[t.value].accuracy if t.value in row.by_track else None)
            for t in MCQ_TRACKS
        ),
        pct(row.mcq_overall.accuracy if row.mcq_overall else None),
        pct(row.andobert.factual_accuracy if row.andobert else None),
        pct(row.public.accuracy if row.public else None),
        pct(row.private.accuracy if row.private else None),
        gap_text,
    ]


def render_space_html(
    board: Leaderboard, *, version: str, dataset_repo: str = DEFAULT_DATASET_REPO
) -> str:
    """A self-contained static page. No scripts, no CDN, nothing to break."""
    headers = [
        "Model",
        *(TRACK_LABELS[t] for t in MCQ_TRACKS),
        "MCQ overall",
        "And-Obert factual",
        "Public",
        "Private",
        "Gap",
    ]
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(c)}</td>" for c in _row_cells(row, board.suspicious_gap))
        + "</tr>"
        for row in board.rows
    )
    caveats = (
        "<h2>Caveats</h2><ul>"
        + "".join(f"<li>{html.escape(w)}</li>" for w in board.warnings)
        + "</ul>"
        if board.warnings
        else ""
    )
    repo = html.escape(dataset_repo)
    dataset_link = f'<a href="https://huggingface.co/datasets/{repo}">{repo}</a>'
    return f"""<!doctype html>
<html lang="ca">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AndBench Leaderboard {html.escape(version)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0 auto; max-width: 60rem;
         padding: 2rem 1rem; line-height: 1.5; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .sub {{ color: #6b7280; margin-top: 0; }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 0.5rem 0.6rem; border-bottom: 1px solid #d1d5db; text-align: right;
            white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ border-bottom-width: 2px; }}
  tbody tr:first-child {{ font-weight: 600; }}
  .note {{ background: rgba(120,120,120,0.12); padding: 0.75rem 1rem; border-radius: 0.5rem; }}
  footer {{ margin-top: 2.5rem; color: #6b7280; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>AndBench Leaderboard</h1>
<p class="sub">{html.escape(version)} — knowledge of Andorra and Andorran Catalan.</p>

<div class="scroll">
<table>
<thead><tr>{head}</tr></thead>
<tbody>{body}</tbody>
</table>
</div>

<p class="note"><strong>Read the Gap column first.</strong> It is public accuracy minus private
accuracy. AndBench holds back a stratified private split precisely so contamination is visible: a
model scoring markedly higher on the published items than on the held-out ones has been trained on
the benchmark, and its other numbers cannot be trusted. A gap above
{board.suspicious_gap:.0%} is marked ⚠️.</p>

{caveats}

<footer>
Dataset: {dataset_link} ·
Harness: <a href="https://github.com/ericrisco/andbench">github.com/ericrisco/andbench</a> ·
Every figure is reproducible from recorded results with one command.
</footer>
</body>
</html>
"""


def build_space_repo(
    board: Leaderboard,
    out_dir: str | Path,
    *,
    version: str,
    dataset_repo: str = DEFAULT_DATASET_REPO,
) -> list[Path]:
    """Write the static Space: configured ``README.md`` plus ``index.html``."""
    root = Path(out_dir)
    front = yaml.safe_dump(space_frontmatter(), sort_keys=False, allow_unicode=True)
    readme = (
        f"---\n{front}---\n\n"
        f"# AndBench Leaderboard\n\n"
        f"A static table — no runtime, nothing to break. Version {version}.\n\n"
        f"Built from recorded evaluation results by `andbench leaderboard`, so every figure "
        f"can be re-derived from the same files.\n"
    )
    return [
        _write(root / "README.md", readme),
        _write(
            root / SPACE_ENTRY_FILE,
            render_space_html(board, version=version, dataset_repo=dataset_repo),
        ),
    ]


# --- the gate --------------------------------------------------------------


def publish_problems(dataset_dir: str | Path, space_dir: str | Path | None = None) -> list[str]:
    """Everything that must be true before an upload. Re-reads what was written.

    Deliberately independent of the code that produced the folder: the point is to
    catch a private item that slipped through, not to agree with the filter.
    """
    problems: list[str] = []
    root = Path(dataset_dir)

    readme = root / "README.md"
    if not readme.is_file():
        problems.append("dataset repo has no README.md (the dataset card)")
    elif CANARY_GUID not in readme.read_text(encoding="utf-8"):
        problems.append("the dataset card does not carry the canary GUID")

    canary = root / CANARY_FILE
    if not canary.is_file():
        problems.append(f"dataset repo has no {CANARY_FILE}")
    elif CANARY_GUID not in canary.read_text(encoding="utf-8"):
        problems.append(f"{CANARY_FILE} does not carry {CANARY_GUID}")

    data_root = root / DATA_DIR
    item_files = sorted(data_root.rglob("*.jsonl")) if data_root.is_dir() else []
    if not item_files:
        problems.append(f"dataset repo has no item files under {DATA_DIR}/")

    leaked: list[str] = []
    for path in item_files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"{path.relative_to(root)}:{lineno}: invalid JSON")
                continue
            if payload.get("public") is False:
                leaked.append(f"{payload.get('id', '?')} ({path.relative_to(root)}:{lineno})")
    if leaked:
        problems.append(
            f"{len(leaked)} PRIVATE item(s) are in the upload folder and would be published "
            f"irreversibly, destroying the over-fitting detector: " + ", ".join(leaked[:5])
        )

    if space_dir is not None:
        space = Path(space_dir)
        if not (space / SPACE_ENTRY_FILE).is_file():
            problems.append(f"Space repo has no {SPACE_ENTRY_FILE}")
        space_readme = space / "README.md"
        if not space_readme.is_file():
            problems.append("Space repo has no README.md (its configuration lives there)")
        else:
            front = space_readme.read_text(encoding="utf-8").split("---")
            if len(front) < 3 or yaml.safe_load(front[1]).get("sdk") != "static":
                problems.append("Space README.md does not declare `sdk: static`")

    return problems


@dataclass
class PublishPlan:
    """What would be uploaded, whether it may be, and how."""

    dataset_dir: Path
    dataset_repo: str
    space_dir: Path | None = None
    space_repo: str | None = None
    problems: list[str] = field(default_factory=list)
    uploaded: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def commands(self) -> list[str]:
        """The exact commands a human would run instead."""
        cmds = [f"hf upload {self.dataset_repo} {self.dataset_dir} . --repo-type dataset"]
        if self.space_dir is not None and self.space_repo is not None:
            cmds.append(f"hf upload {self.space_repo} {self.space_dir} . --repo-type space")
        return cmds

    def summary(self) -> str:
        lines: list[str] = []
        if not self.ok:
            lines.append(f"Publication BLOCKED — {len(self.problems)} problem(s):")
            lines.extend(f"  - {p}" for p in self.problems)
            return "\n".join(lines)
        if self.uploaded:
            lines.append(f"Uploaded {len(self.uploaded)} repo(s):")
            lines.extend(f"  - {u}" for u in self.uploaded)
        else:
            lines.append("Dry run — nothing uploaded. Checks passed; run:")
            lines.extend(f"  {c}" for c in self.commands())
        return "\n".join(lines)


def publish(
    dataset_dir: str | Path,
    *,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    space_dir: str | Path | None = None,
    space_repo: str | None = DEFAULT_SPACE_REPO,
    uploader: Uploader | None = None,
) -> PublishPlan:
    """Check the assembled folders and, with an uploader, push them.

    Without an ``uploader`` this is a dry run: it verifies and prints, and cannot
    publish anything by accident.
    """
    plan = PublishPlan(
        dataset_dir=Path(dataset_dir),
        dataset_repo=dataset_repo,
        space_dir=None if space_dir is None else Path(space_dir),
        space_repo=space_repo if space_dir is not None else None,
    )
    plan.problems = publish_problems(plan.dataset_dir, plan.space_dir)
    if not plan.ok or uploader is None:
        return plan

    plan.uploaded.append(
        uploader.upload_folder(repo_id=dataset_repo, repo_type="dataset", folder=plan.dataset_dir)
    )
    if plan.space_dir is not None and plan.space_repo is not None:
        plan.uploaded.append(
            uploader.upload_folder(
                repo_id=plan.space_repo, repo_type="space", folder=plan.space_dir
            )
        )
    return plan
