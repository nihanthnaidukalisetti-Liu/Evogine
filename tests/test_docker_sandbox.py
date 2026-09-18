"""Docker-gated isolation probes: the real container path.

Skipped cleanly — never failed — when no Docker daemon (or docker-py) is
available, so the offline suite stays green everywhere. Where a daemon
exists (GitHub ubuntu-latest runners have Docker), these run for real:
network blackhole, env-leak, and staging-traversal probes, plus an
end-to-end gate acceptance and rejection through the actual sandbox.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evogine.gates.code_gate import (
    CodeGate,
    DockerCaseEvaluator,
    EquivalenceClaim,
    EvalCase,
    GateStage,
)
from evogine.sandbox.docker_runner import DockerSandbox

pytestmark = pytest.mark.skipif(
    not DockerSandbox.available(),
    reason="Docker daemon unavailable — Docker-gated probes skip so the offline suite stays green",
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Comment-only edit of src/evogine/__init__.py (generated against the
#: scaffold content; applies to a HEAD archive).
COMMENT_ONLY_PATCH = """diff --git a/src/evogine/__init__.py b/src/evogine/__init__.py
index 2ef1563..858599d 100644
--- a/src/evogine/__init__.py
+++ b/src/evogine/__init__.py
@@ -10,3 +10,4 @@ __version__ = "0.1.0"
 from evogine.engine.records import LearningRecord

 __all__ = ["LearningRecord", "__version__"]
+# gate: comment-only refactor probe
"""

#: Breaks `import evogine` — the in-container suite must fail and the gate
#: must reject at the pytest stage (verified against an extracted tree).
BREAKING_PATCH = """diff --git a/src/evogine/__init__.py b/src/evogine/__init__.py
--- a/src/evogine/__init__.py
+++ b/src/evogine/__init__.py
@@ -10,3 +10,4 @@ __version__ = "0.1.0"
 from evogine.engine.records import LearningRecord

 __all__ = ["LearningRecord", "__version__"]
+raise RuntimeError("candidate is broken on purpose")
"""

#: Refuses to apply — the gate must reject at the staging stage.
TRAVERSAL_PATCH = """diff --git a/../../../evogine-traversal-escape b/../../../evogine-traversal-escape
new file mode 100644
--- /dev/null
+++ b/../../../evogine-traversal-escape
@@ -0,0 +1 @@
+pwned
"""


def test_network_is_blackholed() -> None:
    sandbox = DockerSandbox()
    result = sandbox.run(
        "python -c \"import socket; socket.create_connection(('93.184.216.34', 443),"
        " timeout=3); print('CONNECTED')\""
    )
    assert "CONNECTED" not in result.stdout
    assert result.exit_code != 0


def test_dns_is_blackholed() -> None:
    sandbox = DockerSandbox()
    result = sandbox.run(
        "python -c \"import socket; socket.gethostbyname('example.com'); print('RESOLVED')\""
    )
    assert "RESOLVED" not in result.stdout
    assert result.exit_code != 0


def test_host_environment_is_not_inherited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOGINE_PROBE_SECRET", "probe-secret-value-9f2c")
    sandbox = DockerSandbox()
    result = sandbox.run(
        "python -c \"import os, json; print(json.dumps(sorted(os.environ)))\""
    )
    assert result.exit_code == 0
    assert "EVOGINE_PROBE_SECRET" not in result.stdout
    combined = result.stdout + result.stderr
    assert "probe-secret-value-9f2c" not in combined


def test_allowlisted_env_reaches_container() -> None:
    sandbox = DockerSandbox()
    result = sandbox.run(
        "python -c \"import os; print(os.environ['EVOGINE_ALLOWED_VAR'])\"",
        env={"EVOGINE_ALLOWED_VAR": "gate-allowed"},
    )
    assert result.stdout.strip() == "gate-allowed"


def test_traversal_patch_rejected_end_to_end() -> None:
    canary = Path("/tmp/evogine-traversal-escape")
    canary.unlink(missing_ok=True)
    gate = CodeGate(DockerCaseEvaluator(DockerSandbox()), repo_root=REPO_ROOT)
    verdict = gate.evaluate_detailed(TRAVERSAL_PATCH, "HEAD")
    assert verdict.allowed is False
    assert verdict.stage is GateStage.STAGING
    assert not canary.exists()


def test_gate_accepts_clean_patch_through_real_sandbox() -> None:
    gate = CodeGate(
        DockerCaseEvaluator(DockerSandbox()),
        repo_root=REPO_ROOT,
        cases=[
            EvalCase(
                case_id="import-gate",
                command="python -c \"import evogine; print('ok')\"",
                env={"PYTHONPATH": "src"},
            )
        ],
    )
    verdict = gate.evaluate_detailed(
        COMMENT_ONLY_PATCH,
        "HEAD",
        claim=EquivalenceClaim.REFACTOR,
        claimed_files=["src/evogine/__init__.py"],
    )
    assert verdict.allowed, verdict.reason
    assert verdict.stage is GateStage.ACCEPTED


def test_gate_rejects_broken_patch_through_real_sandbox() -> None:
    gate = CodeGate(DockerCaseEvaluator(DockerSandbox()), repo_root=REPO_ROOT)
    verdict = gate.evaluate_detailed(BREAKING_PATCH, "HEAD")
    assert verdict.allowed is False
    assert verdict.stage is GateStage.PYTEST
