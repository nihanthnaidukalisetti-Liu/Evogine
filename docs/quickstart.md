# Quickstart: from clone to first governed run

Every command on this page was executed against the real CLI in a fresh
directory, exactly as printed. Where the tool refuses something, the
refusal is shown too — refusals are the product working as designed, and
each one names its fix.

## Prerequisites

- Python 3.12+ and `git`. The learning store is a git repository; nothing
  runs without git.
- Optional extras, installed only when you need them:
  - `pip install -e '.[docker]'` — the code gate's sandboxed execution
    (Docker must be reachable; the gate fails closed without it)
  - `pip install -e '.[console]'` — the localhost operator console
  - `pip install -e '.[server]'` — the sync daemon

## Install

```bash
git clone https://github.com/nihanthnaidukalisetti-Liu/Evogine
cd Evogine
pip install -e .   # core, offline
```

## First-run setup

`evogine init` creates the learning store, writes the starter
`evogine.yaml`, and reports the provider key's state (the key itself is
read from the environment at run time; it is never printed or stored):

```console
$ evogine init
evogine init
  store:  created at .evogine/store (0 record(s))
  config: wrote ./evogine.yaml (default profile: default)
  provider key: ANTHROPIC_API_KEY is not set — live runs refuse until it is configured
    fix: export ANTHROPIC_API_KEY=<your provider key>
         the key is read from the environment at run time; it is never written to the config or any file
next: evogine doctor   # verify the environment end-to-end
```

Two properties to know before you use it:

- An existing config file is never overwritten — init refuses before
  creating anything:

  ```console
  $ evogine init
  error: config ./evogine.yaml already exists — refusing to overwrite it (remove it by hand or pass --config to write elsewhere); nothing was created
  ```

- `--root` picks the store root and `--config` picks the file to write;
  this is how a second project's store is created beside the first:

  ```bash
  evogine init --root .evogine/store-beta --config beta.yaml
  ```

## Check the environment

`evogine doctor` renders every readiness check with an explicit state —
`ok`, `warn`, `fail` — and the fix beside each non-ok row. It exits 0
only when nothing failed (warnings do not fail the run):

```console
$ evogine doctor
evogine doctor — environment readiness
  config   ok    ./evogine.yaml (profile: default)
  git      ok    git version 2.47.3
  docker   fail  no usable Docker daemon: Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))
           fix: start the Docker daemon — the code gate refuses (fails closed) without it
  provider warn  ANTHROPIC_API_KEY is not set — offline work proceeds, but live runs refuse before spending until a key is configured
           fix: export ANTHROPIC_API_KEY=<your provider key> — read from the environment at run time, never stored in the config
  disk     ok    32.3 GiB free on .evogine/store
  sync     ok    no sync endpoints configured (nothing to probe)
doctor: not ready — 1 failure(s), 1 warning(s), 4 ok
```

On a machine with Docker running and the key exported, the same command
prints `doctor: ready (…)` and exits 0.

## Store candidates

The CLI has no `propose` command on main today: candidates arrive from
the swarm's learning sink or from a reviewed cross-project import
(`evogine memory import`). To make the next step a real one, seed one
candidate through the store's own write path — the same path the swarm
writes through:

```python
# seed_candidate.py
from datetime import UTC, datetime

from evogine.bench.gate_replay import replay_run_ids
from evogine.engine.records import (
    LearningRecord, RecordBody, RecordCounterfactual, RecordProvenance,
)
from evogine.engine.store import Engine

record = LearningRecord(
    id="retry-jitter-on-429",
    type="failure-pattern",
    title="Unhandled 429 storms escalate across rollouts",
    body=RecordBody(
        signal="rising 429 counts across consecutive rollouts",
        failure_mode="429 amplifies without jittered exponential backoff",
        detection="counterfactual replay flips the verifier verdict on the degraded case",
        fix_strategy="apply jittered exponential backoff with a hard retry cap per rollout",
    ),
    status="ephemeral",
    confidence=0.6,
    provenance=RecordProvenance(
        proposed_by="reflector",
        origin="rollout-2026-09-19",
        source="repair-swarm",
        note="proposed by the reflector role after the 429 storm rollout",
    ),
    counterfactual=RecordCounterfactual(),
    created=datetime.now(UTC),
    promoted_at=None,
)
engine = Engine(".evogine/store")
stored = engine.propose(record)
engine.queue_replay(stored.id, replay_run_ids(stored.id))
```

```console
$ python3 seed_candidate.py
$   # silent on success — the script prints nothing and exits 0; the store, not
  # the script's output, is the record. The audit trail now opens with the two
  # lifecycle events it wrote:
$ head -n 2 .evogine/store/audit.jsonl
{"seq":1,"ts":"2026-09-20T01:42:34.737914Z","event":"propose","record_id":"retry-jitter-on-429","from_status":null,"to_status":"ephemeral","reason":null,"run_ids":[],"pass_id":null}
{"seq":2,"ts":"2026-09-20T01:42:34.746045Z","event":"queue_replay","record_id":"retry-jitter-on-429","from_status":"ephemeral","to_status":"candidate","reason":null,"run_ids":["replay-retry-jitter-on-429-baseline","replay-retry-jitter-on-429-warm"],"pass_id":null}
```

## Run an offline review

`evogine run` drives a full promotion pass over the store: it replays
every candidate cold and warm on the default corpus, decides the
candidate through the memory gate, records one telemetry decision per
proposal, and renders an approval proposal when the knowledge stages
clear. It never promotes on its own:

```console
$ evogine run --root .evogine/store
promotion pass: 1 candidate(s) decided
  retry-jitter-on-429 [approval] proposed: 8 binary flips
    evidence: runs/gate/retry-jitter-on-429.json
next: 1 proposal(s) await human approval (evogine approvals list --root .evogine/store)
```

A candidate that does not change any verdict is refused at the replay
stage and demoted — the refusal names the stage and reason:

```console
$ evogine run --root .evogine/store
promotion pass: 1 candidate(s) decided
  retry-jitter-on-429 [replay] no_flip: 0/80 replays changed verdict
    evidence: runs/gate/retry-jitter-on-429.json
```

On a fresh store with no candidates, a pass is honest about deciding
nothing: `promotion pass: 0 candidate(s) decided`.

## Decide the proposal

The gate rendered one pending proposal. A person applies or refuses it:

```console
$ evogine approvals list --root .evogine/store
approval proposals: 1 total, 1 pending
  appr-retry-jitter-on-429 [pending]
    learning: retry-jitter-on-429
    trail: cleared: 8 binary flips (evidence: runs/gate/retry-jitter-on-429.json)
```

```console
$ evogine approvals approve appr-retry-jitter-on-429 \
    --by "Nihanth nihanthnaidu008@gmail.com" \
    --reason "replay evidence checks out" \
    --root .evogine/store
appr-retry-jitter-on-429 approved: record retry-jitter-on-429 is promoted
```

`approvals reject` is the other decision, and it is equally first-class:
it refuses the proposal and demotes the record, with a mandatory reason.

## Read the audit trail

The store's audit trail is `audit.jsonl` at the store root — append-only,
one JSON line per lifecycle event, sequence numbers derived from the live
line count. After the run and the approval:

```console
$ cat .evogine/store/audit.jsonl
{"seq":1,"ts":"2026-09-19T22:19:44.338746Z","event":"propose","record_id":"retry-jitter-on-429","from_status":null,"to_status":"ephemeral","reason":null,"run_ids":[],"pass_id":null}
{"seq":2,"ts":"2026-09-19T22:19:44.350070Z","event":"queue_replay","record_id":"retry-jitter-on-429","from_status":"ephemeral","to_status":"candidate","reason":null,"run_ids":["replay-retry-jitter-on-429-baseline","replay-retry-jitter-on-429-warm"],"pass_id":null}
{"seq":3,"ts":"2026-09-19T22:19:55.291613Z","event":"promote","record_id":"retry-jitter-on-429","from_status":"candidate","to_status":"promoted","reason":null,"run_ids":["replay-retry-jitter-on-429-baseline","replay-retry-jitter-on-429-warm"],"pass_id":null}
```

The store root is its own git repository — one lifecycle transition is
one commit, so the same history is visible with `git -C .evogine/store log`.

## Where to go next

- [User guide](user-guide.md) — store lifecycle, approvals, retrieval,
  gates and promotion, bench protocols, the proposal funnel, and
  cross-project export/import.
- [Operator runbook](operator-runbook.md) — backup and restore, sync
  daemon operation, doctor checks, and what the store looks like on disk.
- [Architecture](architecture.md) — the module map, the four federation
  invariants, and the refusal-state vocabulary.
