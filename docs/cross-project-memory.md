# Cross-project memory — governance record (design-only)

**Status: design-only in V3, by decision.** The V3 expansion spec (W3, "Cross-project
memory (W3)") records the decision, made by Nihanth: V3 writes the governance analysis,
not the code. This document is that deliverable — the analysis a V4 implementation
inherits, grounded in the codebase as of `d2ab240`.

The choice between the two viable architectures is genuinely open, and taking it without
live-run evidence would be the one avoidable irreversibility in the V3 plan. Nothing in
this document changes behavior; it names the fork and the constraints any path through
it must respect.

---

## 1. Why design-only

### 1.1 The store has no scope concept anywhere

The learning store is deliberately, structurally single-tenant. Verified against the
source:

- **One `Engine` is one store root is one flat id space.** `Engine.__init__` takes a
  single `root` (`src/evogine/engine/store.py`); every record is one JSON file under
  `records/`, and the engine owns that directory as its own git repository. There is no
  partition, namespace, or project dimension — only a flat pool.
- **`promoted_records()` returns the whole pool.** Every record with
  `status == "promoted"`, unfiltered (`store.py`). Retrieval draws from exactly that:
  `retrieve()` scores `store.promoted_records()` with no other source of candidates
  (`src/evogine/engine/retrieval.py`).
- **Ids are unique only within a store.** A record id is its filename under `records/`
  (`_record_path`), pattern-locked to `^[A-Za-z0-9][A-Za-z0-9._-]*$` (`_ID_PATTERN`),
  and `propose` refuses an id that already exists **in that store**. Nothing anywhere
  asserts uniqueness across stores; an id like `lr-2026-09-17-0001` is store-local by
  construction and carries no qualification.
- **The retrieval seam is single-object.** `retrieve(store, task_context, k, floor)`
  takes exactly one engine. There is no multi-store dispatch, no remote lookup, no
  aggregation layer.

The word "scope" does appear in the codebase — `BudgetScope` in
`src/evogine/bench/budgets.py` (RUN / CASE_FAMILY / PROPOSAL) — but it is a different
concept: queue-time replay-budget accounting for the benchmark, not memory
partitioning. No storage-layer scoping exists.

### 1.2 Why unscoped multi-repo storage would void the promotion-only contract

The promotion-only retrieval guarantee is the engine's central knowledge-safety
property: **warm runs see exactly the knowledge the memory gate has proven.** A record
reaches the retrieval pool only through `promote`, which requires candidate status,
counterfactual replay evidence naming every queued run, a stamped material verdict
change, and — since the human-approval boundary (V2-6) — a human-approved promotion
proposal.

Every one of those proofs is **made against runs in this store, for this pool's
operating context**. The counterfactual replay compares cold and warm verdicts on the
queued failure cases; the memory gate's materiality judgment and the human approval both
attach to that evidence trail.

Now put two projects' records in one pool with no scope boundary. A warm run in project
A can retrieve a record the gate proved in project B's replay runs — knowledge that was
genuinely promoted, genuinely human-approved, and never evaluated for A. The
promotion-only contract reads "warm runs see what the gate proved"; unscoped pooling
breaks the *for this context* half of that sentence. This is not a bug a patch can fix:
the failure is the shape of the pool. That is why cross-project memory is the one V3
candidate whose **unscoped** implementation could structurally void the
promotion-only contract, and why the architecture decision deserves live-run evidence
first rather than a premature pick.

---

## 2. The two viable architectures

Both options are honest, viable, and not equivalent. They differ in where the boundary
lives: inside the store, or above it.

### Option A — a scope dimension inside one store

A scope field on records, scope-enforced filtering in `promoted_records()` and
retrieval, scope fixed at `Engine` construction. One store, many scopes; the audit
trail stays single and simple.

**Costs, honestly:**

- **Touches the record contract and every pool scan.** `LearningRecord` gains a field;
  `promoted_records()`, `candidates()`, `find_duplicates()`, `decay()` — every path
  that iterates live records must filter, and every one that forgets is a leak.
- **Needs a migration story for existing stores.** There is precedent for compatible
  evolution — the V2 lifecycle fields (`merged_sources`, `retired_at`, …) all default
  so v1 JSON round-trips unchanged — but a defaulted scope field is not a migration:
  an existing store's records must resolve to a *real* scope, and the store's git
  history (one commit per lifecycle transition) means that resolution is itself an
  audited event.
- **A mis-scoped filter is a silent contract violation.** A pool scan that drops the
  scope predicate returns cross-scope rows that look perfectly normal — no error, no
  exception, warm retrieval simply serves knowledge it should not. The failure mode is
  invisible until audited, which is exactly the failure shape §1.2 describes.

### Option B — federation across per-project stores

One root per project — the store contract unchanged, each store's invariants intact —
and a router above `Engine` dispatching by scope. Cross-store import is an explicit,
human-approved event carrying the source's provenance verbatim; the `MergedSource`
pattern is the existing template (`src/evogine/engine/records.py`): when `merge()`
absorbs a duplicate, the survivor keeps the absorbed record's full provenance —
never rewritten, never summarized.

**Costs, honestly:**

- **Per-store decay clocks raise real questions.** Confidence decay runs per store on
  `DecayConfig` half-life curves anchored per record (`decay()`, `decayed_confidence()`).
  Once a record is imported across stores, which clock governs the shared survivor —
  the source's, the destination's, or a new one — and what does its audit trail show?
- **Duplicate detection stays per-pool.** `find_duplicates()` scans one store's live
  records pairwise. Federated stores never see each other's duplicates; the router
  inherits per-pool blindness it cannot fix without becoming a second store.
- **Remote-store sync does not exist.** Every store is its own git repository pinned to
  its own root; there is no fetch, push, or merge protocol between stores, and building
  one is new protocol surface, not new storage.
- **The retrieval seam becomes multi-object.** `retrieve()` takes one store today. A
  federation turns the seam into a multi-object dispatch with its own consistency
  questions — and cross-project promotion and id-qualification become protocol
  problems at that seam rather than fields in one schema.

### The trade, in one view

|                    | A — scope inside one store                          | B — federation over per-project stores              |
|--------------------|-----------------------------------------------------|-----------------------------------------------------|
| Blast radius       | One store: every scope shares lifecycle, audit, and any store-level failure | Isolated per project; one project's failure stays its own |
| Audit              | Single, simple append-only trail                    | Per-store trails; no global ordering (invariant 3)  |
| Governance surface | One policy, enforced once                           | Per-project policy; the import event is the human gate |
| Contract churn     | Record contract + every pool scan + migration       | None inside the store; new protocol above it        |
| Characteristic failure | Mis-scoped filter — silent, invisible until audited | Import/sync/decay ambiguity — visible but protocol-heavy |

Neither column dominates. That is the point: this is a real fork, and V3 does not take
it.

---

## 3. The four invariants

Any future implementation — Option A, Option B, or a third shape — must preserve all
four. Each one is the survival of an existing contract under multi-scope operation;
each can be checked structurally.

### 3.1 Scoping is enforced at the storage/retrieval layer — never by caller discipline

Scope boundaries must live where the pool is assembled: in `promoted_records()`,
`retrieve()`, and their equivalents. Prompt conventions, swarm-role discipline, or CLI
etiquette are not boundaries. The engine's existing guarantees are all structural — the
store refuses to retrieve anything but promoted records, refuses drafts that arrive
pre-stamped, refuses promotion evidence that omits queued runs — and scoping must join
that class: enforced where records move, not where intentions live.

### 3.2 Promotion-only, per scope

Warm runs see exactly what the gate proved **for this scope**. The human-approval
boundary (V2-6) and the counterfactual-replay evidence chain attach to a scope's
operating context; retrieval in scope S must draw only from records whose promotion
evidence was earned and approved within S. Any implementation where a scope-S warm run
can act on a record proven elsewhere — without an explicit, approved cross-scope event
— violates this invariant regardless of how the violation is spelled in code.

### 3.3 Per-store append-only audit, with no invented global ordering

Each store's `audit.jsonl` is append-only, never rewritten; `seq` is derived from the
live line count of that file, and one lifecycle transition is one commit. Under
federation (or any multi-store layout), these trails stay per-store: **no
implementation may stitch per-store sequences into a "global" audit order that no
single clock produced.** Cross-store events (imports, borrows) may reference the
source trail, but each trail's internal order is the only order that exists.

### 3.4 Qualified ids before anything references records across a scope boundary

Record ids are unique only within one store (§1.1). The moment any record, audit
event, provenance chain, or import manifest references a record that lives outside its
own scope, the reference must be **qualified** — store/scope identity plus id — before
it is written. Unqualified cross-scope references are ambiguous by construction and
are the seeds of silent provenance corruption; the `MergedSource` pattern shows the
standard (full provenance travels verbatim with the survivor), and qualification
extends it across boundaries.

---

## 4. The deciding question — deferred to V4

> **Does cross-project reuse mean sharing one pool, or occasionally borrowing with
> explicit provenance?**

- **Sharing one pool → Option A.** If projects are variants of the same operating
  context and knowledge genuinely accumulates in common, a scope dimension inside one
  store is the honest shape: one pool, one audit, one promotion pipeline.
- **Occasional borrowing with explicit provenance → Option B.** If projects are
  genuinely separate operating contexts whose knowledge should move rarely and
  deliberately, federation is the honest shape: isolated stores, contracts unchanged,
  and a human-approved import carrying provenance verbatim.

**The evidence the V3 live run and console operation will produce** — exactly what the
V4 spec needs to decide:

- **How much promoted knowledge accumulates.** A trickle supports federation's
  isolation; a dense, fast-growing pool makes the shared-pool economics of Option A
  real rather than hypothetical.
- **How often proposals generalize beyond their origin.** If promoted records
  repeatedly prove material outside the context that proposed them, borrowing is
  frequent enough to need a real protocol (A, or B with a first-class import flow); if
  generalization is rare, B's explicit-borrow model costs almost nothing.
- **What audit traffic looks like.** Single-trail audit (A) and per-store trails (B)
  scale differently; live operation shows which shape the promotion/decay/merge
  traffic actually produces.

A V4 spec decides **with that evidence in hand**. V3's deliverable is this analysis
plus the four invariants in §3, which any future implementation must satisfy.

---

## Appendix: code pointers

Every claim above, as it lives in the source at `d2ab240`:

| Claim | Where it lives |
|---|---|
| One `Engine` = one root; store root is its own git repo | `Engine.__init__`, module docstring — `src/evogine/engine/store.py` |
| One file per record; ids are store-local filenames | `_record_path`, `_ID_PATTERN` — `store.py` |
| Whole-pool promotion retrieval | `Engine.promoted_records`, `Engine.retrieve` — `store.py`; `retrieve()` — `src/evogine/engine/retrieval.py` |
| Promotion requires gate evidence + human approval | `Engine.promote` docstring and checks — `store.py`; approvals CLI (V2-6) |
| Provenance travels verbatim on merge | `MergedSource` — `src/evogine/engine/records.py`; `Engine.merge` lineage transfer — `store.py` |
| Per-store append-only audit, seq = live line count | `AuditEvent`, `Engine._append_audit` — `store.py` |
| Per-store decay clock | `Engine.decay` — `store.py`; `DecayConfig`, `decayed_confidence` — `records.py` |
| Duplicate detection is per-pool | `Engine.find_duplicates` — `store.py` |
| Single-object retrieval seam | `retrieve(store, …)` — `retrieval.py` |
| The only "scope" today is budget accounting | `BudgetScope` — `src/evogine/bench/budgets.py` |
