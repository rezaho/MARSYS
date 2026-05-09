"""Identifier factory for tracing.

ULID gives lexicographic-monotonic IDs that sort correctly by emission time
within a process and remain JSON-safe strings.
"""
from ulid import ULID


def new_id() -> str:
    """Return a 26-char uppercase Crockford-base32 ULID string.

    Monotonic-within-process via python-ulid's class-level ValueProvider.
    Drop-in replacement for ``str(uuid.uuid4())``.
    """
    return str(ULID())
