"""End-to-end CLI tests for the live bench path: gated, fail-closed, offline.

The suite is strictly offline: the provider-key seam returns a dummy value
and the client seam is a fake — no network, no real keys, nothing spent.
The pins: the protocol-less live path does not exist; a missing key refuses
before any model call; pre-run validation refuses below-floor repeats, a
projected spend over the dollar ceiling, and a contaminated holdout before
the first dollar moves.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from conftest import make_record, replay_evidence

from evogine import cli
from evogine.bench.cases import default_corpus
from evogine.bench.gate_replay import replay_run_ids
from evogine.bench.harness import REPEAT_FLOOR
from evogine.bench.probes import content_tokens
from evogine.bench.prompts import DEFAULT_PROMPT_PACK
from evogine.bench.protocol import fingerprint_corpus
from evogine.engine.store import Engine
from evogine.swarm.clients import ModelClient, ModelRequest, ModelResponse

LIVE_SPEND_FLAGS = [
    "--model",
    "fake-model",
    "--expected-input-tokens",
    "1000",
    "--expected-output-tokens",
    "500",
    "--input-price-per-mtok-usd",
    "3.00",
    "--output-price-per-mtok-usd",
    "15.00",
]


def _protocol_yaml(tmp_path: Path, **overrides: object) -> Path:
    """A valid pre-registered protocol over the default corpus."""
    fields: dict[str, object] = {
        "name": "cli-live acceptance",
        "corpus_fingerprint": fingerprint_corpus(default_corpus()),
        "prompt_pack": DEFAULT_PROMPT_PACK.fingerprint,
        "repeats": 4,
        "budgets": {
            "max_per_proposal": 999,
            "max_per_case_family": 999,
            "max_per_run": 999,
        },
    }
    fields.update(overrides)
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(fields), encoding="utf-8")
    return path


class _FakeClient:
    """Fake ModelClient at the network seam: records requests, always answers."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text="no known fix for this failure",
            model="fake-model",
            input_tokens=1000,
            output_tokens=500,
        )


class _ClientSeam:
    """What the CLI resolved and built at the two live seams, plus the fake."""

    def __init__(self) -> None:
        self.resolved_key: str | None = None
        self.resolved_model: str | None = None
        self.client = _FakeClient()


def _patch_live_seams(monkeypatch: pytest.MonkeyPatch, *, with_key: bool) -> _ClientSeam:
    """Fake both live seams offline; a built client on a refused run fails the test."""
    seam = _ClientSeam()
    monkeypatch.setattr(
        cli, "_resolve_provider_key", lambda: "offline-fake-key" if with_key else None
    )

    def factory(*, api_key: str, model: str) -> ModelClient:
        seam.resolved_key = api_key
        seam.resolved_model = model
        return seam.client

    monkeypatch.setattr(cli, "_build_live_client", factory)
    return seam


def _contaminate_holdout(root: Path) -> None:
    """Promote a record quoting tokens unique to the locale family's context.

    The recall probe flags a record that quotes a held-out case's unique
    vocabulary, so the fixture builds exactly that from the probe's own
    token rules rather than hardcoding corpus wording.
    """
    corpus = default_corpus()
    holdout = next(case for case in corpus if case.family == "locale")
    others = [case for case in corpus if case.id != holdout.id]
    unique = content_tokens(holdout.task_context) - frozenset().union(
        *(content_tokens(case.task_context) for case in others)
    )
    assert unique, "the locale case must carry tokens no other case shares"
    engine = Engine(str(root))
    record = make_record("rec-contaminated", topic=" ".join(sorted(unique)[:5]))
    engine.propose(record)
    engine.queue_replay(record.id, replay_run_ids(record.id))
    engine.promote(record.id, replay_evidence(replay_run_ids(record.id)))


def test_resolve_provider_key_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolver accepts a real value and refuses an empty or absent one."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x" * 20)
    assert cli._resolve_provider_key() == "x" * 20
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert cli._resolve_provider_key() is None
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert cli._resolve_provider_key() is None


def test_live_without_a_protocol_does_not_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``bench --runner live`` without --protocol refuses before anything exists."""
    root = tmp_path / "store"
    seam = _patch_live_seams(monkeypatch, with_key=True)

    exit_code = cli.main(
        ["bench", "--root", str(root), "--runner", "live", *LIVE_SPEND_FLAGS]
    )

    assert exit_code == 1
    assert "--protocol" in capsys.readouterr().err
    assert seam.resolved_key is None  # no client was ever built
    assert not (root / "runs").exists()  # no run dir was opened


def test_live_refuses_a_missing_provider_key_before_any_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing key refuses the whole live run before any model call."""
    root = tmp_path / "store"
    protocol = _protocol_yaml(tmp_path)
    seam = _patch_live_seams(monkeypatch, with_key=False)

    exit_code = cli.main(
        [
            "bench",
            "--root",
            str(root),
            "--runner",
            "live",
            "--protocol",
            str(protocol),
            *LIVE_SPEND_FLAGS,
        ]
    )

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "provider key" in error
    assert "before any model call" in error
    assert seam.resolved_key is None
    assert not (root / "runs").exists()


def test_live_refuses_repeats_below_the_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A below-floor repeat count is refused on the live path too."""
    protocol = _protocol_yaml(tmp_path)
    seam = _patch_live_seams(monkeypatch, with_key=True)

    exit_code = cli.main(
        [
            "bench",
            "--root",
            str(tmp_path / "store"),
            "--runner",
            "live",
            "--protocol",
            str(protocol),
            "--repeats",
            "2",
            *LIVE_SPEND_FLAGS,
        ]
    )

    assert exit_code == 1
    assert f"below the {REPEAT_FLOOR}-repeat floor" in capsys.readouterr().err
    assert seam.resolved_key is None


def test_live_refuses_an_incomplete_spend_expectation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every spend knob is required: an absent one would project zero."""
    protocol = _protocol_yaml(tmp_path)
    seam = _patch_live_seams(monkeypatch, with_key=True)

    exit_code = cli.main(
        [
            "bench",
            "--root",
            str(tmp_path / "store"),
            "--runner",
            "live",
            "--protocol",
            str(protocol),
            "--expected-input-tokens",
            "1000",
            "--expected-output-tokens",
            "500",
            "--input-price-per-mtok-usd",
            "3.00",
            "--output-price-per-mtok-usd",
            "15.00",
        ]
    )

    assert exit_code == 1
    assert "--model" in capsys.readouterr().err
    assert seam.resolved_key is None


def test_live_refuses_a_garbage_price(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unparseable price is a declared-spend refusal, not a crash."""
    protocol = _protocol_yaml(tmp_path)
    flags = ["--model", "fake-model", *LIVE_SPEND_FLAGS[2:]]
    flags[flags.index("3.00")] = "not-a-number"
    seam = _patch_live_seams(monkeypatch, with_key=True)

    exit_code = cli.main(
        [
            "bench",
            "--root",
            str(tmp_path / "store"),
            "--runner",
            "live",
            "--protocol",
            str(protocol),
            *flags,
        ]
    )

    assert exit_code == 1
    assert "price_per_mtok_usd" in capsys.readouterr().err
    assert seam.resolved_key is None


def test_live_refuses_a_protocol_pin_on_a_different_prompt_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Live spend only follows the protocol's pinned prompt text."""
    protocol = _protocol_yaml(tmp_path, prompt_pack="pack-0001")
    seam = _patch_live_seams(monkeypatch, with_key=True)

    exit_code = cli.main(
        [
            "bench",
            "--root",
            str(tmp_path / "store"),
            "--runner",
            "live",
            "--protocol",
            str(protocol),
            *LIVE_SPEND_FLAGS,
        ]
    )

    assert exit_code == 1
    assert "prompt pack" in capsys.readouterr().err
    assert seam.resolved_key is None


def test_live_refuses_a_projection_over_the_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A ceiling below the projected spend refuses before any model call."""
    root = tmp_path / "store"
    protocol = _protocol_yaml(tmp_path)  # repeats 4 → 80 replays → $0.84 projected
    seam = _patch_live_seams(monkeypatch, with_key=True)

    exit_code = cli.main(
        [
            "bench",
            "--root",
            str(root),
            "--runner",
            "live",
            "--protocol",
            str(protocol),
            "--max-cost-usd",
            "0.01",
            *LIVE_SPEND_FLAGS,
        ]
    )

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "dollar ceiling exceeded" in error
    assert "refusing before spend" in error
    assert seam.resolved_key is None  # no client was ever built
    assert not (root / "runs").exists()  # no run dir was opened


def test_live_refuses_a_contaminated_holdout_before_spend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A holdout the store already quotes voids the run for zero dollars."""
    root = tmp_path / "store"
    protocol = _protocol_yaml(tmp_path, held_out_family="locale")
    _contaminate_holdout(root)
    seam = _patch_live_seams(monkeypatch, with_key=True)

    exit_code = cli.main(
        [
            "bench",
            "--root",
            str(root),
            "--runner",
            "live",
            "--protocol",
            str(protocol),
            *LIVE_SPEND_FLAGS,
        ]
    )

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "contaminated" in error
    assert "nothing was spent" in error
    assert seam.client.requests == []  # the fake client saw no traffic
    assert not (root / "runs").exists()


def test_live_run_with_a_fake_client_completes_the_full_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gated live path runs the whole schedule through the fake client."""
    root = tmp_path / "store"
    protocol = _protocol_yaml(tmp_path)  # repeats 4
    seam = _patch_live_seams(monkeypatch, with_key=True)

    exit_code = cli.main(
        [
            "bench",
            "--root",
            str(root),
            "--runner",
            "live",
            "--protocol",
            str(protocol),
            "--format",
            "json",
            *LIVE_SPEND_FLAGS,
        ]
    )

    assert exit_code == 0
    assert seam.resolved_key == "offline-fake-key"
    assert seam.resolved_model == "fake-model"
    assert len(seam.client.requests) == 80  # 10 cases x 2 arms x 4 repeats
