"""Forward-only schema migrations for the Spren SQLite database.

The runner reads `<NN>__<slug>.py` files in this directory in order and applies
the unapplied ones in transactions; applied IDs are recorded in a `_migrations`
table keyed by the leading number prefix (e.g. `01`).
"""
from __future__ import annotations

from .runner import MigrationsRunner

__all__ = ["MigrationsRunner"]
