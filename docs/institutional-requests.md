# Institutional requests for exam materials (B0.02)

> **Send these first.** Institutional answers take weeks, and And-Llengua's official part (B2.05) is
> conditional on them arriving. Everything else in the plan can proceed in parallel; this cannot be
> compressed later.

Reusing official examinations as pre-validated questions is the **Latxa** method (Basque) with
**INCLUDE**'s licence hygiene: the questions were already written and validated by the institution
that owns the domain, which is a far stronger provenance than anything a project can generate. The
condition is that permission is **documented in writing before publication** — project constitution
P23, enforced in code by [`configs/sources.yaml`](../configs/sources.yaml) (see
[Recording the answer](#recording-the-answer)).

## Who to write to, and for what

| # | Institution | Asking for | Feeds |
|---|---|---|---|
| a | **Servei de Política Lingüística** | Past models of the Catalan certificate exams (any level), including answer keys where they exist | And-Llengua official part (B2.05) |
| b | **Servei de Funció Pública** | Past *oposició* questionnaires, particularly the general-knowledge and institutional sections | And-Coneix (B2.03), And-Llengua register |
| c | **Ministeri d'Educació / Escola Andorrana** | "Coneixement del medi" materials and assessment items for Andorran geography, history and institutions | And-Coneix (B2.03), And-Cotidià (B2.06) |

Send them as three separate requests, not one. They are different services with different archives,
and a single request routed to the wrong desk stalls all three.

## What every request must contain

1. **What the project is**, in one paragraph, without jargon.
2. **Exactly what is being asked for** — be narrow. "Past exam models from any year" is answerable;
   "your materials" is not.
3. **What will be published and what will not.** Items derived from their materials are *adapted*,
   not reproduced verbatim, and roughly 15 % of every item set stays unpublished.
4. **A guaranteed, named citation** in the dataset card and in every announcement.
5. **The precedent.** Latxa did this with Basque institutions; INCLUDE did it across many countries.
   This is an established practice, not an unusual request.
6. **A concrete reply channel and a person.** Anonymous requests get filed.

Do **not** promise: exclusivity, editorial control over the benchmark, or that a model will score
well. Do not imply the institution endorses the results.

## The common template

> Reusable body. Adapt the bracketed parts per institution; keep the commitments identical, because
> they are the ones recorded in the dataset card.

**Assumpte:** Sol·licitud d'ús de materials d'avaluació per a un projecte públic d'avaluació de models de llenguatge

Benvolguts / Benvolgudes,

Em dirigeixo a vosaltres en nom del projecte **AndBench**, una prova d'avaluació pública i oberta
per a mesurar el coneixement que els models de llenguatge (la tecnologia darrere d'eines com
ChatGPT) tenen sobre **Andorra** i sobre el **català d'Andorra**.

El problema que volem documentar és senzill: aquests models s'entrenen majoritàriament amb text
d'altres territoris, i quan se'ls pregunta sobre Andorra o s'espera que facin servir la varietat
andorrana del català, sovint responen amb dades d'altres llocs o amb formes que aquí no són les
pròpies. Avui no existeix cap prova pública que ho mesuri, i el que no es mesura no es corregeix.

Per a construir la prova necessitem preguntes **fiables i ja validades**. Per això us sol·licitem
autorització per a fer servir **[materials sol·licitats: p. ex. models d'exàmens de convocatòries
anteriors]** com a base per a elaborar-ne preguntes d'avaluació.

Concretament, ens comprometem a:

- **Adaptar, no reproduir.** Les preguntes es reescriuen per al format de la prova; no publiquem els
  vostres materials originals.
- **Citar-vos de manera explícita i permanent** com a font, tant a la documentació del conjunt de
  dades com en tota comunicació pública del projecte.
- **No publicar res sense la vostra autorització escrita.** El projecte té un control tècnic que
  impedeix publicar cap pregunta la procedència de la qual no tingui permís documentat.
- **Reservar una part del material sense publicar** (aproximadament un 15 %), com és pràctica
  habitual en aquest tipus de proves, per a poder detectar-ne usos indeguts.
- **Fer-vos arribar el resultat** abans de qualsevol publicació, i atendre qualsevol correcció.

Aquesta pràctica té precedents directes: el projecte **Latxa**, per al basc, va construir la seva
prova d'avaluació a partir d'exàmens oficials amb la col·laboració de les institucions basques, i el
projecte internacional **INCLUDE** ha fet el mateix amb materials d'avaluació de desenes de països.

AndBench és un projecte **sense ànim de lucre i de codi obert**: tant la prova com les eines es
publiquen amb llicència lliure, i qualsevol persona o institució en pot verificar els resultats.

Quedo a la vostra disposició per a explicar-ho amb més detall, en persona si ho preferiu, i per a
signar qualsevol document que necessiteu.

Ben cordialment,

**[Nom]**
[Càrrec / afiliació]
[Correu electrònic] · [Telèfon]
[Enllaç al repositori del projecte]

## Per-institution adjustments

**(a) Política Lingüística** — lead with the linguistic argument, not the technical one: the risk
is that a widely-used tool normalises non-Andorran forms. Ask specifically for exam models *with*
answer keys, since a key is what makes an item pre-validated. Mention And-Llengua by name (local
lexicon, toponymy, institutional register).

**(b) Funció Pública** — frame it as public-administration knowledge: institutions, competences,
procedures. Past *oposició* questionnaires are usually already public in some form, so the ask is
often about *confirming reuse* rather than obtaining the material. Say so — it is a smaller ask than
it looks.

**(c) Ministeri d'Educació / Escola Andorrana** — the material is aimed at school level, which maps
well onto difficulty 1–2 items. Note that adapted items will be published openly and can be reused
in teaching, which is a genuine reciprocal benefit rather than a courtesy.

## Tracking

Keep this table updated; it is the audit trail behind the permission states in
`configs/sources.yaml`.

| Institution | Sent | Contact | Answer | Permission | Reference |
|---|---|---|---|---|---|
| Política Lingüística | — | — | — | pending | — |
| Funció Pública | — | — | — | pending | — |
| Educació / Escola Andorrana | — | — | — | pending | — |

Chase after **three weeks** with a short, friendly message that repeats the ask in one sentence.
After six weeks with no answer, treat that source as unavailable for v1.0 and record it as such —
the plan already has And-Llengua's own part (B2.04) as the non-conditional path.

## Recording the answer

When an answer arrives, edit [`configs/sources.yaml`](../configs/sources.yaml):

```yaml
  - id: official-exams-politica-linguistica
    id_prefix: exam-pl-
    label: Servei de Política Lingüística — models d'examen de certificació
    licence: Conditional on the written permission obtained
    permission: granted            # or: refused
    permission_ref: "email 2026-05-12, ref. SPL-2026-041"
    note: Levels B2 and C1 exam models, answer keys included.
```

`permission_ref` must point at something a third party could ask to see. Until it is `granted` (or
`open-licence`), `andbench card` **refuses to build a dataset card** for any item citing that
source, and the release stops:

```
[FAIL] dataset-card: source 'official-exams' has permission 'pending', so its
       37 item(s) may not be published (constitution P23)
```

That failure is the feature. It means an unlicensed item cannot reach a release by being forgotten.
If permission is **refused**, the affected items are rewritten from another source, not quietly
published.
