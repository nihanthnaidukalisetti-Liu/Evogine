"""Tests for the unified config module: load, validate, refuse, starter round-trip.

All offline: validation is pure pydantic over in-memory YAML text — no
network, no store, no provider key. Every bad-input test pins a refusal
state: the error names the file and the defect, never silently defaults.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from evogine.config import (
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_STORE_ROOT,
    ConfigError,
    default_config_path,
    load_config,
    starter_config_yaml,
)

MINIMAL_YAML = """\
default_profile: default
profiles:
  default:
    store:
      learning_root: .evogine/store
    provider:
      model: test-model
"""


def _write_config(tmp_path: Path, text: str) -> Path:
    config_path = tmp_path / DEFAULT_CONFIG_FILENAME
    config_path.write_text(text, encoding="utf-8")
    return config_path


def test_load_config_minimal_applies_every_default(tmp_path: Path) -> None:
    """A minimal file loads: optional sections default, required ones do not."""
    config = load_config(_write_config(tmp_path, MINIMAL_YAML))

    profile = config.profile()
    assert profile.store.learning_root == Path(".evogine/store")
    assert profile.provider.model == "test-model"
    assert profile.provider.api_key_env == "ANTHROPIC_API_KEY"
    assert str(profile.budget.max_run_cost_usd) == "25.00"  # the DollarBudget default
    assert profile.workers.min_workers == 1
    assert profile.workers.max_workers == 4
    assert profile.sync.endpoints == []
    assert profile.logging.level == "info"


def test_load_config_keeps_money_exact_as_decimal(tmp_path: Path) -> None:
    """A quoted dollar amount survives as an exact Decimal, digits intact."""
    # A full profile with an explicit two-decimal ceiling and a second profile.
    text = """\
default_profile: laptop
profiles:
  laptop:
    store:
      learning_root: /tmp/laptop-store
    provider:
      model: m
      api_key_env: MY_PROVIDER_KEY
    budget:
      max_run_cost_usd: "30.00"
    workers:
      min_workers: 2
      max_workers: 2
    sync:
      endpoints:
        - name: bench-daemon
          url: http://127.0.0.1:8787
    logging:
      level: debug
  bench:
    store:
      learning_root: /tmp/bench-store
    provider:
      model: m2
"""
    config = load_config(_write_config(tmp_path, text))

    laptop = config.profile("laptop")
    assert laptop.budget.max_run_cost_usd == Decimal("30.00")
    assert str(laptop.budget.max_run_cost_usd) == "30.00"  # not Decimal('30') drift
    assert laptop.provider.api_key_env == "MY_PROVIDER_KEY"
    assert laptop.workers.max_workers == 2
    assert laptop.sync.endpoints[0].name == "bench-daemon"
    # pydantic normalizes HttpUrl to its canonical form (bare host gains a
    # trailing slash); the probe and the report render the normalized URL.
    assert str(laptop.sync.endpoints[0].url) == "http://127.0.0.1:8787/"
    assert laptop.logging.level == "debug"

    # Resolution without a name uses default_profile; an explicit name resolves.
    assert config.profile() is config.profiles["laptop"]
    assert config.profile("bench").store.learning_root == Path("/tmp/bench-store")


def test_load_config_default_profile_must_exist(tmp_path: Path) -> None:
    """A default_profile naming no defined profile refuses at load."""
    text = MINIMAL_YAML.replace("default_profile: default", "default_profile: typo")
    with pytest.raises(ConfigError, match="default_profile 'typo' names no profile"):
        load_config(_write_config(tmp_path, text))


def test_load_config_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_load_config_refuses_malformed_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(_write_config(tmp_path, "default_profile: [unclosed"))


def test_load_config_refuses_a_non_mapping_document(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="YAML mapping at the top level"):
        load_config(_write_config(tmp_path, "- just\n- a\n- list\n"))


def test_load_config_refuses_an_empty_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="YAML mapping at the top level"):
        load_config(_write_config(tmp_path, ""))


def test_load_config_refuses_unknown_top_level_fields(tmp_path: Path) -> None:
    """A typo'd top-level key is config drift, refused (extra=forbid)."""
    text = MINIMAL_YAML + "default_profil: default\n"
    with pytest.raises(ConfigError, match="refused"):
        load_config(_write_config(tmp_path, text))


def test_load_config_refuses_unknown_profile_fields(tmp_path: Path) -> None:
    text = MINIMAL_YAML.replace(
        "      model: test-model", "      model: test-model\n      max_cost: 5"
    )
    with pytest.raises(ConfigError, match="refused"):
        load_config(_write_config(tmp_path, text))


def test_load_config_refuses_a_profile_missing_its_store(tmp_path: Path) -> None:
    text = MINIMAL_YAML.replace("    store:\n      learning_root: .evogine/store\n", "")
    with pytest.raises(ConfigError, match="refused"):
        load_config(_write_config(tmp_path, text))


def test_load_config_refuses_a_profile_missing_its_provider(tmp_path: Path) -> None:
    text = MINIMAL_YAML.replace("    provider:\n      model: test-model\n", "")
    with pytest.raises(ConfigError, match="refused"):
        load_config(_write_config(tmp_path, text))


@pytest.mark.parametrize("budget", ["abc", "0", "-5", ".inf", ".nan"])
def test_load_config_refuses_bad_budget_ceilings(tmp_path: Path, budget: str) -> None:
    """Non-numeric, non-positive, and non-finite ceilings all refuse."""
    text = MINIMAL_YAML + f"    budget:\n      max_run_cost_usd: {budget}\n"
    with pytest.raises(ConfigError, match="refused"):
        load_config(_write_config(tmp_path, text))


def test_load_config_refuses_worker_bounds_below_one(tmp_path: Path) -> None:
    text = MINIMAL_YAML + "    workers:\n      min_workers: 0\n      max_workers: 4\n"
    with pytest.raises(ConfigError, match="refused"):
        load_config(_write_config(tmp_path, text))


def test_load_config_refuses_max_workers_below_min(tmp_path: Path) -> None:
    text = MINIMAL_YAML + "    workers:\n      min_workers: 4\n      max_workers: 2\n"
    with pytest.raises(ConfigError, match="max_workers \\(2\\) must be >= min_workers \\(4\\)"):
        load_config(_write_config(tmp_path, text))


def test_load_config_refuses_an_unknown_profile_on_resolution(tmp_path: Path) -> None:
    """profile() refuses an unknown name — never a silent default fallback."""
    config = load_config(_write_config(tmp_path, MINIMAL_YAML))
    with pytest.raises(ConfigError, match="profile 'ghost' is not defined"):
        config.profile("ghost")


@pytest.mark.parametrize("name", ["", "bad name", "-leading", "traversal/x"])
def test_load_config_refuses_unsafe_profile_names(tmp_path: Path, name: str) -> None:
    """Profile names are locked to the store-id character class."""
    text = f"""\
default_profile: default
profiles:
  default:
    store:
      learning_root: .evogine/store
    provider:
      model: m
  {name!r}:
    store:
      learning_root: /tmp/x
    provider:
      model: m
"""
    with pytest.raises(ConfigError, match="not a safe profile id"):
        load_config(_write_config(tmp_path, text))


@pytest.mark.parametrize("bad_env", ["lower_case", "1LEADING", "with-dash", "", "with space"])
def test_load_config_refuses_an_invalid_api_key_env_name(
    tmp_path: Path, bad_env: str
) -> None:
    text = MINIMAL_YAML.replace(
        "      model: test-model", f'      model: test-model\n      api_key_env: "{bad_env}"'
    )
    with pytest.raises(ConfigError, match="not a valid environment variable name"):
        load_config(_write_config(tmp_path, text))


def test_load_config_refuses_non_http_sync_urls(tmp_path: Path) -> None:
    """Sync endpoints are http(s)-only — a file:// URL refuses at load."""
    text = (
        MINIMAL_YAML
        + "    sync:\n      endpoints:\n        - name: local\n          url: file:///etc/passwd\n"
    )
    with pytest.raises(ConfigError, match="refused"):
        load_config(_write_config(tmp_path, text))


def test_load_config_refuses_duplicate_sync_endpoint_names(tmp_path: Path) -> None:
    text = (
        MINIMAL_YAML
        + "    sync:\n      endpoints:\n"
        "        - name: local\n          url: http://127.0.0.1:1\n"
        "        - name: local\n          url: http://127.0.0.1:2\n"
    )
    with pytest.raises(ConfigError, match="duplicate sync endpoint name\\(s\\): local"):
        load_config(_write_config(tmp_path, text))


def test_load_config_refuses_an_unreadable_file(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, MINIMAL_YAML)
    config_path.chmod(0o000)
    try:
        with pytest.raises(ConfigError, match="could not be read"):
            load_config(config_path)
    finally:
        config_path.chmod(0o644)  # allow tmp cleanup even when running as root


def test_starter_config_round_trips_through_load_config(tmp_path: Path) -> None:
    """The starter init writes loads back unchanged: every knob present."""
    config = load_config(_write_config(tmp_path, starter_config_yaml()))

    assert config.default_profile == "default"
    profile = config.profile()
    assert profile.store.learning_root == Path(DEFAULT_STORE_ROOT)
    assert profile.provider.model == "REPLACE-WITH-PINNED-PROVIDER-MODEL"
    assert profile.provider.api_key_env == "ANTHROPIC_API_KEY"
    assert str(profile.budget.max_run_cost_usd) == "25.00"
    assert profile.sync.endpoints == []


def test_starter_config_embeds_a_custom_learning_root(tmp_path: Path) -> None:
    """init --root records the exact root it created in the written config."""
    custom = tmp_path / "stores" / "main"
    config = load_config(
        _write_config(tmp_path, starter_config_yaml(learning_root=str(custom)))
    )
    assert config.profile().store.learning_root == custom


def test_default_config_path_uses_the_working_directory(tmp_path: Path) -> None:
    assert default_config_path(tmp_path) == tmp_path / DEFAULT_CONFIG_FILENAME
    assert default_config_path().name == DEFAULT_CONFIG_FILENAME  # cwd-relative default
