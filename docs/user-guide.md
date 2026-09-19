# User guide

How to operate a store day to day: lifecycle, approvals, retrieval,
promotion, benchmarks, the proposal funnel, and cross-project memory
transfer. Every CLI command shown was executed against the real binary
from this repository's `main`; behavior that comes from the Python API
rather than the CLI is labeled as such.

Contents:

1. [The store and its lifecycle](#the-store-and-its-lifecycle)
2. [The approvals CLI](#the-approvals-cli)
3. [Retrieval behavior and the honest empty state](#retrieval-behavior-and-the-honest-empty-state)
4. [Memory gates and promotion](#memory-gates-and-promotion)
5. [Bench protocols](#bench-protocols)
6. [The proposal funnel](#the-proposal-funnel)
7. [Cross-project export and import](#cross-project-export-and-import)

## The store and its lifecycle

One store root is one git repository holding one flat pool of learning
records. There is no partition or namespace inside a store; separation
between projects is federation's job (per-project stores, see
[Cross-project export and import](#cross-project-export-and-import) and
[Architecture](architecture.md)).

A record is a versioned JSON contract (`LearningRecord`) — id, type,
title, the four body fields (`signal`, `failure_mode`, `detection`,
`fix_strategy`), confidence, provenance, and a counterfactual-evidence
block. The status field moves through the lifecycle, and every transition
is one append-only audit event plus one git commit:

| status | reached by | meaning |
|---|---|---|
| `ephemeral` | `propose` (swarm sink or a seed script) | a draft — unproven, never retrievable |
| `candidate` | `queue_replay` with run ids | queued for counterfactual replay |
| `promoted` | `approvals approve` on a cleared proposal | live in the retrieval pool |
| `dropped` | demotion (gate refusal or `approvals reject`) | refused — kept, auditable, never deleted |
| `retired` | `retire` (store API, with a reason) | withdrawn from the pool — reversible via `reinstate`; `dropped` and `merged` are terminal and cannot be retired |

Promotion-only retrieval is the contract that ties this together: warm
runs see exactly the knowledge the memory gate has proven and a person
has approved — nothing else.

## The approvals CLI

The memory gate renders promotion proposals into the store's approval
log; `approvals` is the human boundary. The gate never promotes on its
own, and no promotion can exist without an approved proposal.

```console
$ evogine approvals list --root .evogine/store
approval proposals: 1 total, 1 pending
  appr-retry-jitter-on-429 [pending]
    learning: retry-jitter-on-429
    trail: cleared: 8 binary flips (evidence: runs/gate/retry-jitter-on-429.json)
```

Approve — `--by` records who decided; `--reason` is an optional note:

```bash
evogine approvals approve appr-retry-jitter-on-429 \
  --by "Nihanth nihanthnaidu008@gmail.com" \
  --reason "replay evidence checks out" \
  --root .evogine/store
```

Reject — the reason is mandatory; the record is demoted and stays
auditable:

```bash
evogine approvals reject appr-retry-jitter-on-429 \
  --by "Nihanth nihanthnaidu008@gmail.com" \
  --reason "fix strategy is not production-safe yet" \
  --root .evogine/store
```

The same two decisions exist on the operator console
(`POST /approvals/{proposal_id}/approve|reject`), which records the
operator identity the console was launched with. There is no batch
decide: each route decides exactly one proposal.

## Retrieval behavior and the honest empty state

Retrieval scores the store's promoted pool against a task context with
two signals combined by `max`:

- cosine similarity of embedder vectors over the record's **body fields**,
- lexical overlap between the context and the same fields.

Three properties matter in practice:

- **The relevance floor is 0.35.** Below it, retrieval returns `[]`. An
  empty result is a first-class outcome — "nothing relevant here" — never
  a fallback to the least-unrelated record.
- **Titles are never scored.** The v1 defect was title-only matching;
  scoring reads only the four body fields. A title match alone can never
  qualify a record.
- **Only promoted records can be served.** Drafts, candidates, and
  dropped records are invisible to retrieval by construction.

Retrieval is a library seam (`Engine.retrieve(task_context, k=5)`), not a
CLI command on main today; warm arms of `run`, `bench`, and `shadow` all
go through it. The spend-visible consequence: if the pool holds nothing
relevant to a case, the offline runner reports `no known fix for this
failure` rather than pretending knowledge helped — which is why a cold
pass rate of 0.000 means exactly what it says.

## Memory gates and promotion

`evogine run --root <root>` is the promotion pass. For every candidate it
replays the recorded failure cases cold (baseline — knowledge absent) and
warm (knowledge applied as if promoted), then decides through three
knowledge stages plus one infrastructure stage:

| stage | question | refusal reasons (machine-checked vocabulary) |
|---|---|---|
| `replay` | did the warm verdict actually flip? | `no_flip` |
| `materiality` | does the flip qualify? (binary verdicts, threshold) | `not_material` |
| `never_worse` | did any evaluated case regress? | `cleared`, `regression` |
| `approval` | the human boundary — proposal rendered | `proposed`, `refused_no_evidence` |
| `gate_error` | replay infrastructure failed — not a knowledge verdict | `gate_error` |

The vocabulary is machine-checked: a reason string outside this set
refuses to be recorded. Knowledge-stage rejections demote the record to
`dropped` (kept, never deleted); `gate_error` and evidence refusals leave
the record a candidate — infrastructure says nothing about the knowledge,
so it stays eligible for a retry. A pass exits non-zero (exit code 2)
when any `gate_error` decision was made.

Each pass mints a `pass_id`, stamps it on every decision it produced,
and writes `passes/<pass_id>.json` — what ran, what it decided, and
per-decision review latency. The manifest is written only after every
candidate is decided: an absent manifest means the pass did not finish,
never that it decided nothing.

Evidence lands in the store too: paired replay verdicts under
`runs/gate/<learning_id>.json`, and per-replay run directories under
`runs/replay/` — resumable, inspectable, committed with the lifecycle.

## Bench protocols

```console
$ evogine bench --root .evogine/store --repeats 4
bench (offline runner): 10 cases, 80 replays at 4 repeats
  run dir: .evogine/store/runs/bench/20260919T222156-d22437d0
  cold pass rate: 0.000
  warm pass rate: 0.000
  mean delta:     +0.000
  tz-offset-drift: cold 0.000 -> warm 0.000 (+0.000)
  retry-jitter-budget: cold 0.000 -> warm 0.000 (+0.000)
  ...
```

`bench` measures warm versus cold pass rates over the default 10-case
corpus. It is informational and always exits 0 when the run completes.
The offline runner composes submissions from served knowledge only, so
the benchmark is an honest function of store contents and the corpus —
nothing else.

The live runner (`--runner live`) drives the swarm model client for real,
and is gated before anything else — a pre-registered protocol and a
configured provider key are required, and pre-run validation fails
closed. These refusals spend nothing, in order:

- no pre-registered protocol (`--protocol`) or provider key — refused
  before the first model call;
- a projected spend over the dollar ceiling (`--max-cost-usd`, default
  25.00) — the projection uses the mandatory `--expected-input-tokens`,
  `--expected-output-tokens`, and per-mtok prices, so the ceiling cannot
  project zero;
- a repeat count below the floor;
- a contaminated holdout — the retrieval-only recall probe runs first,
  before any client exists.

Run directories are immutable and resumable: `--run-id` resumes a
specific run directory instead of starting a fresh one.

`evogine shadow` profiles a candidate retrieval configuration without
touching lifecycle state. The store is handed over read-only, the
hypothetical records are overlaid as if promoted, and results land only
under `runs/shadow/<candidate_id>/`. A contaminated or over-ceiling run
is refused before anything is evaluated, exiting non-zero with $0 spent.
The candidate declaration is YAML (`candidate_id`, `k`, `floor`,
`budget` with the full spend `expectation`, optional `records` and
`heldout_family`); a real run:

```console
$ evogine shadow --root .evogine/store --candidate shadow-candidate.yaml --repeats 4
shadow widen-retrieval: 80 replays at 4 repeats (projected $0.01200 of $1.00)
  cold pass rate: 0.000
  warm pass rate: 0.000
  mean delta:     +0.000
  heldout recall probe: clean (0 case(s) probed)
  run: runs/shadow/widen-retrieval (complete — immutable)
```

## The proposal funnel

`decisions` reads the store's own telemetry files — decisions, pass
manifests, approvals, records; nothing new is written to report it.

```console
$ evogine decisions report --root .evogine/store
promotion decisions: 1 total
  replay: 0
  materiality: 0
  never_worse: 0
  approval: 1
  gate_error: 0
promoted: 0
```

```console
$ evogine decisions funnel --root .evogine/store
proposal funnel: .evogine/store
  candidates:          1  decided proposals (latest verdict each)
  replay-cleared:      1  replay flipped a verdict
  materiality-cleared: 1  flip met the materiality threshold
  approved:            1  human approved (or applied) the proposal
  promoted:            1  live in the promoted pool
  gate errors:         0  infrastructure — undecided, retryable
  proposals pending:   0  awaiting a human decision
  proposals rejected:  0  refused at the approval boundary

decision reasons by stage (machine-checked vocabulary, zero-filled):
  replay:
    no_flip: 0
  materiality:
    not_material: 0
  never_worse:
    cleared: 0
    regression: 0
  approval:
    refused_no_evidence: 0
    proposed: 1
  gate_error:
    gate_error: 0

passes: 1
  pass-20260919T221945-ab07df48:
    candidates: 1, decisions: 1
    replay-cleared: 1, materiality-cleared: 1
    approved: 1, promoted: 1, gate errors: 0

live run token evidence: none found (no per-call token records under <root>/runs)
```

Read the funnel as the pipeline's honesty check: where counts shrink
between stages is where knowledge was refused, and the zero-filled
reason breakdown says exactly which refusal fired. `--format json` gives
the same report for tooling. The operator console's `/funnel` panel
renders the same computation over the same files.

## Cross-project export and import

Every project keeps its own store; knowledge moves between stores only
as an export bundle that a person reviews before it lands. This is the
borrowing-with-provenance model: the destination store's invariants are
never relaxed for incoming knowledge.

Export the promoted pool of a store (a pure read — the source is never
touched):

```console
$ evogine memory export --store alpha --root .evogine/store --out alpha-bundle.json
exported 1 promoted record(s) from alpha — provenance verbatim, lineage qualified
  bundle: alpha-bundle.json
```

The bundle is one JSON document: an envelope (`schema_version: 1`,
`kind: "evogine-memory-export"`) and a manifest with one entry per
record — the **qualified id** (`alpha:retry-jitter-on-429`) plus the
verbatim record payload. Provenance and merged lineage travel whole,
never summarized.

Import is a two-step, review-first flow. Without `--yes` it is a dry
run — what would land, nothing written:

```console
$ evogine memory import --store beta --root .evogine/store-beta --bundle alpha-bundle.json
dry run — 1 record(s) would land in beta as new candidates (nothing written):
  alpha:retry-jitter-on-429 -> retry-jitter-on-429 [failure-pattern] "Unhandled 429 storms escalate across rollouts" confidence 0.60
  on apply: provenance verbatim, confidence clock re-anchored to this store, source evidence left in its own trail
re-run with --yes to apply the reviewed import
```

With `--yes` the reviewed records enter the destination as **new
candidates** — never as promoted knowledge:

```console
$ evogine memory import --store beta --root .evogine/store-beta --bundle alpha-bundle.json --yes
imported retry-jitter-on-429 as candidate in beta
import complete: 1 record(s) landed, provenance verbatim, recorded in the audit trail
```

The destination's audit trail records the landing, referencing the
source by qualified id only — no source sequence numbers, no stitched
clocks:

```json
{"seq":1,"ts":"2026-09-19T22:20:12.290216Z","event":"import","record_id":"retry-jitter-on-429","from_status":null,"to_status":"candidate","reason":"imported from alpha:retry-jitter-on-429 — provenance verbatim; confidence clock re-anchored to this store's clock at 2026-09-19T22:20:12.289930+00:00","run_ids":[],"pass_id":null}
```

Three rules are enforced, not conventional:

- **Imports land as candidates.** Source-scope promotion evidence never
  masquerades as destination evidence; the record must earn promotion in
  beta's own replays, through beta's own approval boundary.
- **The confidence clock re-anchors** to the destination store at landing.
- **An import is all-or-nothing.** Bundle grammar, schema version,
  provenance completeness, and destination conflicts are validated
  before the first write; a refused bundle persists nothing. An export
  of an empty promoted pool is not an error — it exports a manifest with
  zero records.
