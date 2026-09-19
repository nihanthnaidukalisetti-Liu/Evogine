"""Unified configuration: one YAML file, named profiles, one entry point.

Every productization knob lives here and nowhere else: store roots, provider
settings, budget ceilings (exact ``Decimal`` dollars), worker-pool bounds,
the forward-looking sync-endpoints section, and logging settings.
:func:`load_config` is the single loader — hand-parsing YAML anywhere else
would fork the validation contract, so nothing reads a config file directly.

Named profiles let one file serve several stores or machines with
``default_profile`` naming the one that applies when no name is passed.
Unknown profiles refuse, unknown fields refuse (config drift — the same
``extra="forbid"`` discipline as every other pydantic contract in this
repo), and money is a ``Decimal``: a float ceiling is how a $25.00 limit
becomes $24.999997 silently.

Secrets never live here: the provider key is read from the environment by
name (``api_key_env``), never stored in the file. ``evogine init`` writes
the env-var name, and ``evogine doctor`` reports presence without echoing
any value.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, model_validator

DEFAULT_CONFIG_FILENAME = "evogine.yaml"
"""The config file ``evogine init`` writes and every command reads by default."""

DEFAULT_STORE_ROOT = ".evogine/store"
"""The learning store root a fresh init creates when ``--root`` is not passed."""

# Profile names are rendered into reports and, in later waves, name paths
# and run directories — lock them to the same character class as store
# record ids (the staging-traversal lesson applied at the config boundary).
_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Environment variable names: uppercase with underscores, matching the
# convention every provider SDK documents (``ANTHROPIC_API_KEY`` and friends).
_ENV_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class ConfigError(Exception):
    """Refusal: the config file is missing, unreadable, or invalid.

    Every refusal names the file and the exact defect — a config that cannot
    be trusted must be visible, never silently defaulted around.
    """


class StoreSettings(BaseModel):
    """Where a profile's stores live."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    learning_root: Path
    """The git-backed learning store root. Relative paths resolve against
    the directory ``evogine`` runs from, matching every ``--root`` flag."""


class ProviderSettings(BaseModel):
    """Provider settings: the pinned model and where the key is NOT — disk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    """The provider model id, pinned deliberately. ``REPLACE-*`` placeholders
    are the repo's runbook convention (see bench-live-dispatch.yml): an
    unpinned model must refuse, never default into spending."""
    api_key_env: str = Field(default="ANTHROPIC_API_KEY")
    """The environment variable holding the provider key. The value is read
    from the environment at run time and never written to any file."""

    @model_validator(mode="after")
    def _env_name_is_a_name(self) -> ProviderSettings:
        if not _ENV_NAME_PATTERN.fullmatch(self.api_key_env):
            raise ValueError(
                f"api_key_env {self.api_key_env!r} is not a valid environment "
                "variable name (uppercase letters, digits, underscores; no leading digit)"
            )
        return self


class BudgetSettings(BaseModel):
    """Budget ceilings in exact dollars — the config-file twin of the bench's
    :class:`~evogine.bench.budgets.DollarBudget` (same $25.00 default)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_run_cost_usd: Decimal = Field(default=Decimal("25.00"), gt=0)

    @model_validator(mode="after")
    def _finite_ceiling(self) -> BudgetSettings:
        if not self.max_run_cost_usd.is_finite():
            raise ValueError("max_run_cost_usd must be a finite dollar amount")
        return self


class WorkerPoolSettings(BaseModel):
    """Bounded dynamic worker pool: ``1 <= min_workers <= max_workers``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_workers: int = Field(default=1, ge=1)
    max_workers: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _bounds_ordered(self) -> WorkerPoolSettings:
        if self.max_workers < self.min_workers:
            raise ValueError(
                f"max_workers ({self.max_workers}) must be >= min_workers ({self.min_workers})"
            )
        return self


class SyncEndpoint(BaseModel):
    """One sync-daemon endpoint: a daemon URL plus the store it hosts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    url: HttpUrl
    """Where the daemon answers. ``HttpUrl`` pins the scheme to http(s) — a
    ``file://`` or bare-host entry refuses at load, never at probe time."""

    store: str | None = None
    """The hosted store this remote serves — the name ``evogine sync``
    commands resolve against the daemon's registry. Optional so pre-sync
    configs load unchanged (``evogine doctor`` still probes the URL); a
    sync command against an endpoint without one refuses visibly — the
    client never guesses a store name."""


class SyncSettings(BaseModel):
    """Sync endpoints the config declares: ``evogine doctor`` probes their
    reachability; ``evogine sync`` pushes/pulls the hosted store each names."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoints: list[SyncEndpoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _names_unique(self) -> SyncSettings:
        names = [endpoint.name for endpoint in self.endpoints]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate sync endpoint name(s): {', '.join(duplicates)}")
        return self


class LoggingSettings(BaseModel):
    """Logging settings: level plus the sink format, consumed only by
    :func:`evogine.ops.logging_setup.configure_logging` — the one entry
    point that owns the root logger (no ``basicConfig`` anywhere else).
    The structured JSON sink is the V4 operations wave; ``text`` stays
    the default so interactive use is unchanged."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["debug", "info", "warning", "error"] = "info"
    format: Literal["text", "json"] = "text"


class ProfileSettings(BaseModel):
    """One named profile: every knob, with the optional sections defaulted.

    ``store`` and ``provider`` are required — a profile without a store root
    or a pinned model is incomplete and refuses, rather than half-working.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    store: StoreSettings
    provider: ProviderSettings
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    workers: WorkerPoolSettings = Field(default_factory=WorkerPoolSettings)
    sync: SyncSettings = Field(default_factory=SyncSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


class EvogineConfig(BaseModel):
    """The whole config file: named profiles plus the default profile's name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_profile: str = Field(min_length=1)
    profiles: dict[str, ProfileSettings] = Field(min_length=1)

    @model_validator(mode="after")
    def _profile_names_are_safe(self) -> EvogineConfig:
        for name in self.profiles:
            if not _PROFILE_NAME_PATTERN.fullmatch(name):
                raise ValueError(
                    f"profile name {name!r} is not a safe profile id (alphanumeric, "
                    "'.', '_', '-'; must start with a letter or digit)"
                )
        return self

    @model_validator(mode="after")
    def _default_profile_exists(self) -> EvogineConfig:
        if self.default_profile not in self.profiles:
            raise ValueError(
                f"default_profile {self.default_profile!r} names no profile in this file "
                f"(profiles: {', '.join(sorted(self.profiles))})"
            )
        return self

    def profile(self, name: str | None = None) -> ProfileSettings:
        """Resolve a profile by name; ``None`` means the default profile.

        An unknown name is config drift and refuses — it never falls back
        silently to the default.
        """
        resolved = self.default_profile if name is None else name
        if resolved not in self.profiles:
            raise ConfigError(
                f"profile {resolved!r} is not defined "
                f"(profiles: {', '.join(sorted(self.profiles))})"
            )
        return self.profiles[resolved]


def default_config_path(directory: Path | None = None) -> Path:
    """The config path in effect when no ``--config`` is passed: ``./evogine.yaml``."""
    return (directory if directory is not None else Path.cwd()) / DEFAULT_CONFIG_FILENAME


def starter_config_yaml(learning_root: str = DEFAULT_STORE_ROOT) -> str:
    """The starter config ``evogine init`` writes: every knob present and
    commented, ready to load back through :func:`load_config` unchanged."""
    return f"""\
# evogine configuration — one file, named profiles.
# Relative paths resolve against the directory evogine runs from.
# The provider key is NEVER stored in this file: it is read from the
# environment variable named by api_key_env.

default_profile: default

profiles:
  default:
    store:
      # The git-backed learning store (created by 'evogine init').
      learning_root: {learning_root}
    provider:
      # Pin the provider model deliberately — a REPLACE-* placeholder is the
      # runbook convention (see .github/workflows/bench-live-dispatch.yml):
      # an unpinned model must refuse, never default into spending.
      model: REPLACE-WITH-PINNED-PROVIDER-MODEL
      # The environment variable holding the provider key. The value is read
      # from the environment at run time and never written to disk.
      api_key_env: ANTHROPIC_API_KEY
    budget:
      # Dollar ceilings are exact Decimals; quote them to keep the digits.
      max_run_cost_usd: "25.00"
    workers:
      # Bounded dynamic worker pool: 1 <= min_workers <= max_workers.
      min_workers: 1
      max_workers: 4
    sync:
      # Sync endpoints (V4 Pillar C): 'evogine doctor' probes each URL's
      # reachability; 'evogine sync status|push|pull' moves knowledge and
      # rolls up spend across machines. 'store' names the hosted store the
      # endpoint serves — sync commands refuse an endpoint without one.
      endpoints: []
      # endpoints:
      #   - name: laptop
      #     url: http://127.0.0.1:8766
      #     store: alpha
    logging:
      level: info
      # 'json' renders one JSON object per line for log shippers and the
      # dashboard; 'text' (default) keeps human-oriented lines.
      format: text
"""


def load_config(path: str | Path) -> EvogineConfig:
    """Load and validate a config file — the one entry point.

    Every failure raises :class:`ConfigError` naming the file and the exact
    defect: a missing file, a YAML syntax error, a non-mapping document, or
    a validation refusal (unknown fields, bad bounds, an unresolvable
    default profile).
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(
            f"config {config_path} not found (or not a regular file) — create one "
            "with 'evogine init' or pass --config"
        )
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"config {config_path} could not be read: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config {config_path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"config {config_path} must be a YAML mapping at the top level "
            f"(got {type(data).__name__})"
        )
    try:
        return EvogineConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"config {config_path} refused:\n{exc}") from exc
