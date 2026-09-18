"""Offline seccomp units: default-on profile selection and runner boundary.

No Docker daemon is needed — the runner is exercised through the fake client
(so the contract holds at the docker-py call boundary), and profile
validation is pure JSON logic.
"""

from __future__ import annotations

import json

import pytest
from test_sandbox_hardening import FakeClient, FakeContainer

from evogine.sandbox.config import SandboxConfig
from evogine.sandbox.docker_runner import DockerSandbox, SandboxError
from evogine.sandbox.seccomp import (
    SeccompError,
    build_security_opts,
    load_profile_json,
    validate_profile_json,
)


def test_default_profile_is_bundled_and_deny_by_default() -> None:
    profile = validate_profile_json(load_profile_json(None))
    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    rules = profile["syscalls"]
    assert isinstance(rules, list) and rules


def test_vendored_default_pins_the_moby_upstream_shape() -> None:
    """The vendored file stays verbatim moby: no keys added or dropped."""
    vendored = json.loads(load_profile_json(None))
    assert set(vendored) == {"defaultAction", "defaultErrnoRet", "archMap", "syscalls"}


def test_allow_all_override_is_rejected() -> None:
    profile = json.dumps({"defaultAction": "SCMP_ACT_ALLOW", "syscalls": []})
    with pytest.raises(SeccompError, match="refusing to weaken"):
        load_profile_json(profile)


def test_malformed_override_is_rejected() -> None:
    with pytest.raises(SeccompError, match="not valid JSON"):
        load_profile_json("{not json")


def test_non_deny_by_default_override_is_rejected() -> None:
    profile = json.dumps({"defaultAction": "SCMP_ACT_KILL", "syscalls": [{"names": ["open"]}]})
    with pytest.raises(SeccompError, match="deny by default"):
        load_profile_json(profile)


def test_security_opts_carry_no_new_privileges_and_inline_profile() -> None:
    opts = build_security_opts(load_profile_json(None))
    assert opts[0] == "no-new-privileges:true"
    inline = json.loads(opts[1].removeprefix("seccomp="))
    assert inline["defaultAction"] == "SCMP_ACT_ERRNO"


def test_security_opts_reject_a_path_instead_of_a_profile() -> None:
    """Engine 25+ decodes the seccomp security-opt as JSON — a path 500s at start.

    Regression for the CI docker-gated run (API 1.48): a temp-file path made
    every container start fail with 'Decoding seccomp profile failed'. A path
    value must fail closed here, before any container exists.
    """
    with pytest.raises(SeccompError, match="not valid JSON"):
        build_security_opts("/tmp/evogine-seccomp-profile.json")


def test_runner_always_passes_a_seccomp_profile() -> None:
    """The boundary contract: every container start carries the profile file.

    This is the default-on pin — if profile wiring ever disappears from the
    runner, the offline suite fails here instead of the sandbox silently
    downgrading to daemon discretion.
    """
    container = FakeContainer()
    client = FakeClient(container)
    sandbox = DockerSandbox(client=client)
    result = sandbox.run("true")
    assert result.exit_code == 0
    opts = client.containers.last_kwargs["security_opt"]
    assert isinstance(opts, list) and len(opts) == 2
    assert opts[0] == "no-new-privileges:true"
    # The daemon decodes the profile from the security-opt itself (Engine 25+
    # dropped path support) — the captured value must be the JSON inline.
    inline = json.loads(str(opts[1]).removeprefix("seccomp="))
    assert inline["defaultAction"] == "SCMP_ACT_ERRNO"
    assert container.seccomp_profile_read is not None


def test_runner_honors_custom_profile_override() -> None:
    profile = json.dumps(
        {
            "defaultAction": "SCMP_ACT_ERRNO",
            "archMap": [],
            "syscalls": [{"names": ["exit_group"], "action": "SCMP_ACT_ALLOW"}],
        }
    )
    container = FakeContainer()
    client = FakeClient(container)
    sandbox = DockerSandbox(config=SandboxConfig(seccomp_profile_json=profile), client=client)
    sandbox.run("true")
    opts = client.containers.last_kwargs["security_opt"]
    assert isinstance(opts, list)
    assert opts[1].startswith("seccomp=")
    assert json.loads(container.seccomp_profile_read)["syscalls"] == [
        {"names": ["exit_group"], "action": "SCMP_ACT_ALLOW"}
    ]


def test_runner_fails_closed_when_profile_cannot_be_applied() -> None:
    """An inapplicable profile aborts the run before any container exists."""
    client = FakeClient(FakeContainer())
    client.containers.run_fails = True
    sandbox = DockerSandbox(client=client)
    with pytest.raises(SandboxError, match="container start failed"):
        sandbox.run("true")
