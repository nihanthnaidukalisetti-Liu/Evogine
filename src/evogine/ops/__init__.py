"""Operations surface (V4 productization): metrics, logging, migrations.

The :mod:`evogine.ops` package holds the productization modules the V4
scope's operations wave adds: in-process metrics
(:mod:`evogine.ops.metrics`), the one structured-logging entry point
(:mod:`evogine.ops.logging_setup`), store schema versioning and forward
migrations (:mod:`evogine.ops.migrations`), backup/restore
(:mod:`evogine.ops.backup`), compaction/indexing
(:mod:`evogine.ops.compaction`), and the scale benchmark
(:mod:`evogine.ops.benchmark`).

Import discipline: ``metrics``, ``logging_setup``, and ``migrations`` are
engine-free leaves — ``evogine.engine.store`` imports them, so this
package's ``__init__`` re-exports only those and never pulls the
engine-facing modules (``backup``, ``compaction``, ``benchmark``) into the
import cycle. Callers import those by full path.
"""
