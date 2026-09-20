# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2026-09-19

First packaged release. Everything below shipped to the repository between the
initial v1 scaffold and the 4.0.0 packaging wave; the V2, V3, and V4 scope
documents define the milestones. Authentication, multi-user access, and
unattended promotion remain out of scope by design.

### Added

**Engine and learning (V2)**

- Git-backed learning store with an append-only audit trail (#2).
- Memory lifecycle policies (#5) and hybrid retrieval with an honest empty
  state (#7).
- Memory gate with a materiality rule and promotion telemetry (V2-1, V2-2) (#8).
- Evaluation discipline: pre-registered protocols, clustered confidence
  intervals, contamination gates (V2-7) (#9).
- Anti-gaming: per-decision trajectories, gold-sanity checks, judge
  separation (V2-8) (#10).
- Human-approval promotion boundary with an approvals CLI (V2-6) (#12).
- Federated memory with qualified identities and scope-dispatched retrieval (#26)
  and cross-project export/import with reviewed manifests (#28).

**Sandbox and execution**

- Docker-gated execution sandbox with a fail-closed code gate (#6).
- Hardening: seccomp default-on, wall-clock cross-check, provider failover,
  run directories with resume (#11).

**Swarm**

- Builder / verifier / reflector roles over a retrying model client (#4).
- Repository decomposer with evidence-based criticality scoring (#27).

**Benchmarking (V2/V3)**

- Warm-vs-cold harness with queue-time replay budgets (#3).
- Live-benchmark foundation: prompt pack, dollar ceiling, live case runner (#15).
- Non-mutating shadow-pass orchestrator (#18).
- Protocol-gated live bench with persistent evidence and spend reporting (#19),
  dispatched by a manual workflow (#22).

**Telemetry, console, and review (V3)**

- Concurrency-safe human review with pass-identity telemetry (#17).
- Proposal-funnel computation from telemetry (#21).
- Localhost-only read-only operator console (#16), the proposal-funnel panel
  with CLI parity (#23), and approval write paths over the human-approval
  boundary (#24).
- Console knowledge-injection panel over the reviewed-manifest import flow —
  manifest summaries before any action, refusals as visible panels, and one
  guarded confirm landing candidates — plus the stores-and-sync selector for
  read-only views over the local store and configured sync remotes.

**Sync and onboarding (V4)**

- Unified configuration with named profiles, `evogine init` and `evogine
  doctor` onboarding (#25).
- Git-native sync daemon with loopback-first binding and a read/query API (#29).

**Quality and packaging**

- Property and fuzz suites for the five V3 growth seams (#20).
- Single version source in `pyproject.toml` with a parity test against the
  package `__version__`.
- Tag-triggered release workflow building the sdist and wheel, smoke-testing
  the wheel in a fresh venv, and attaching artifacts to a GitHub Release.
- This changelog.

### Changed

- Package metadata: description, authors, keywords, classifiers, and project
  URLs completed; the console, server, docker, and dev extras are unchanged.
- CI completed and integrated: strict gates, coverage floor, full CLI surface,
  and an end-to-end offline loop (#13).

### Security

- No PyPI publishing is configured for this milestone, intentionally: no PyPI
  credentials exist for this repository and authentication is out of scope for
  4.0.0. Release artifacts (sdist and wheel) are attached to GitHub Releases
  only; adding PyPI publishing requires credentials and an ownership decision
  first.

[4.0.0]: https://github.com/nihanthnaidukalisetti-Liu/Evogine/releases/tag/v4.0.0
