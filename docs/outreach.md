# Outreach kit (B4.05)

Joint announcement with **Pirene**. Two projects announced together is one story with twice the
substance — a benchmark that shows the gap, and a model that closes some of it — for roughly half the
outreach effort. Announcing them separately halves the interest of each.

**Do not announce before** the dataset card is published, the leaderboard has real numbers, and every
institution that contributed material has been thanked *in advance of* the public post. An
institution reading about its own contribution in a press item it did not know about is a
relationship spent for nothing.

## Order of operations

1. **Institutions first**, individually, before anything public. Send the dataset card and the
   leaderboard, show them their citation, and give them a few days to object or to ask for different
   wording.
2. **The technical/linguistic community** (below), same day as the repository goes public.
3. **General press**, only if the community round goes well and someone asks. A benchmark is not a
   press story on its own; the *finding* is.

## Who to contact

| Who | Why them | The angle |
|---|---|---|
| **Softcatalà** | The reference community for Catalan language technology; broad reach among exactly the people who care | The Andorran variety is systematically under-served by tools trained on general Catalan, and now there is a number for it |
| **AINA** (Catalan language AI project) | Institutional counterpart building Catalan models and evaluation | AndBench is a variety-specific evaluation that complements theirs; the tracks separate factual knowledge from linguistic competence |
| **ARI** (Andorra Recerca + Innovació) | Andorran research body; a natural local home for the work | Andorran digital sovereignty: measurable, reproducible, published from Andorra |
| **Contributing institutions** | They gave the material and are owed the citation | Their exam material now serves a public technical purpose, credited |
| **Universitat d'Andorra** | Local academic anchor, potential future verifiers and collaborators | Reproducible methodology, open licence, room for student work |

## The announcement

Keep it short and lead with the finding, not the artefact. Nobody clicks "we built a benchmark";
people click "these models get Andorran questions wrong in this specific way".

> **[Català — post principal]**
>
> Els models de llenguatge que fem servir cada dia saben poc d'Andorra, i quan escriuen en català
> sovint no fan servir la varietat andorrana. Fins ara ningú no ho havia mesurat.
>
> Presentem **AndBench**, la primera prova pública per a avaluar models de llenguatge sobre
> **coneixement d'Andorra** i **català d'Andorra**, i **Pirene**, un model ajustat per a millorar-ho.
>
> AndBench té **[N] preguntes**, totes verificades per persones, repartides en quatre àmbits:
> coneixement factual, llengua, cultura quotidiana i generació oberta. Els resultats es poden
> reproduir amb una sola ordre.
>
> Què hem trobat: **[el resultat concret, 1–2 frases: p. ex. els models generalistes encerten X %
> de les preguntes de coneixement d'Andorra però només Y % de les de llengua andorrana]**.
>
> Tot és obert: el conjunt de dades, el codi i la taula de resultats.
> Dades: [enllaç] · Codi: [enllaç] · Taula: [enllaç]
>
> Amb la col·laboració de **[institucions]**.

Include, in every version:

- **The number of items and that they are 100 % human-verified.** It is the differentiator against
  auto-generated benchmarks.
- **The reproduction command.** A benchmark nobody can re-run is a claim, not a result.
- **The contamination note.** AndBench holds back a private split and publishes the public-vs-private
  gap, so contamination is visible rather than assumed away. This is the part technical readers will
  respect.
- **The institutional credits**, named.

Leave out: model bashing, any suggestion that a low score means a vendor is careless, and superlatives
the data does not support.

## Handling the awkward questions

**"Isn't 800 items too few?"** Yes, deliberately — the CulturalBench trade-off. Coverage is traded
for verification: every item is human-verified against a cited source. Per-area figures rest on small
samples and are reported with their `n`.

**"Pirene tops your own leaderboard — isn't that convenient?"** It is the obvious objection and it has
a real answer: AndBench items come only from a held-out pool that Pirene's training never touched, the
pool hashes are frozen and committed in *both* repositories, and the leaderboard publishes each
model's public-vs-private gap — including Pirene's. If Pirene were contaminated, that gap would show
it. Say this before anyone asks.

**"Who decides what correct Andorran Catalan is?"** Not us. Each item cites a source, and the
verifiers are Andorran. Where usage is genuinely unsettled we say so in the limitations rather than
picking a side.

**"Will you keep it updated?"** Say what is actually true: there is a versioned errata policy,
corrections land in the next version rather than in place, and old scores stay interpretable. Do not
promise a maintenance cadence nobody has committed to.

## After the announcement

- Log who was contacted and what came back. The second release is much easier when this exists.
- Watch for the benchmark appearing in training data — that is what the canary GUID is for. If a
  model reproduces it, say so publicly and calmly.
- Feed everything learned into [the retrospective](retrospective.md).
