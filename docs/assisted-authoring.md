# Assisted authoring: corpus → RAG → three-model filter → you

The institutional requests ([institutional-requests.md](institutional-requests.md)) went
unanswered, so AndBench writes its items from a corpus the project gathers itself. This page is the
operating manual for that path.

**What it changes:** what a human reads.
**What it does not change:** that a human reads it. Every shipped item is still accepted by one
person and verified by a different one (constitution P8). Nothing here writes to the dataset.

---

## The shape

```
corpus (with provenance)
   │  andbench corpus-index
   ▼
bench index ──── andbench draft-corpus ────►  A  writes proposals from a passage
   │                                          │
   │                                          ▼
   └──────────── andbench screen-drafts ───►  B  answers with NO source
                                              │   correct → discarded (does not discriminate)
                                              ▼
                                              C  reads the passage, blind to the key,
                                                 and names every defensible option
                                              │   ≠ exactly {the key} → discarded
                                              ▼
                                        review queue (pending)
                                              │
                                              ▼
                                        human accepts · second human verifies
```

Three different models, from three different labs by default — `deepseek` writes, `anthropic`
reads closed-book, `openai` adjudicates. A repeated model is refused outright: a model checking its
own output agrees with itself, which is not a check.

## What each stage is actually testing

| | Question | A "no" means |
|---|---|---|
| **B** | Can a model answer this *without the source*? | The item measures nothing about knowledge of Andorra. Discard. |
| **C** | Does the passage defend exactly one option, and is it the keyed one? | The item is ambiguous, ungrounded, or miskeyed. Discard. |

**B is not a contamination check.** A model answering correctly closed-book is consistent with
leakage, but equally consistent with the fact being common knowledge or with a lucky guess — one in
four, on four options. Contamination is P10's job (n-gram plus embedding similarity against
`pool_train`), and that gate runs unchanged. B measures *discrimination*.

Read the reported closed-book rate against chance (25 %). At or below it, the batch is genuinely
not answerable from the source. Far above it, the passages are probably restating widely-known
facts and the queries need to go somewhere less obvious.

## Running it

Build the index once (see [the corpus README](../data/sample-corpus/README.md)):

```bash
andbench partition corpus-manifest.jsonl --out pools/
andbench corpus-index corpus/documents.jsonl --pools pools/ --out index/
```

Stage A — draft from retrieved bench passages:

```bash
andbench draft-corpus \
  --index index/index-bench.jsonl \
  --query "el Consell General i la Casa de la Vall" \
  --query "festes i tradicions de les set parròquies" \
  --n 2 --max-passages 20 \
  --out drafts.jsonl
```

Stages B and C — screen them:

```bash
andbench screen-drafts \
  --queue drafts.jsonl \
  --index index/index-bench.jsonl \
  --out screened.jsonl \
  --report screening.md \
  --kept review-queue.jsonl
```

`screened.jsonl` keeps the **rejects too**, each with the reason and the model that gave it. A file
of survivors alone would be unauditable — you could not tell a working filter from one that
approved everything.

Then review `review-queue.jsonl` by hand, exactly as before. Set each `decision` to `accept`,
`edit`, or `reject`; only accepted drafts convert to items, and `draft_to_item` still demands an
`author` and a distinct `verifier` from you.

## Things that will bite you

- **The exit code is not cosmetic.** `screen-drafts` exits 1 when the *filter* misbehaved —
  unreadable closed-book answers, or drafts left unscreened. A batch that exits 1 has not been
  filtered, however clean the survivors look.
- **Options are rotated** before B and C see them, by a hash of the question. Generators put the key
  early and readers favour the first option; together those two harmless habits manufacture
  agreement. The record always cites the *original* index, so the rotation is invisible downstream.
- **An unreadable answer from B keeps the item.** Discarding on a formatting failure would blame the
  item for the reader's output habits. The parse rate is reported instead — watch it.
- **`--max-passages` / `--max-drafts` report what they dropped.** A silent cap reads as full coverage.
- **Licences first.** `corpus-index` warns which sources the P23 permission gate would block. Fix
  that before writing two hundred items from them, not after.

## And-Llengua does not use this path

And-Llengua measures whether a model uses the **Andorran** variety of Catalan. No model produces it
reliably, so a generated item would be general Catalan wearing an Andorran label — the benchmark
would then reward exactly the failure it exists to detect. Those items are written by hand, by
people who speak it. See [verifier-onboarding.md](verifier-onboarding.md).

## Cost

Two paid calls per draft, plus one per passage for stage A. On the chosen models a few hundred
drafts costs low single-digit dollars — far less than the human hours it saves, which was the point.
Both commands print the provider's own reported cost after the run.
