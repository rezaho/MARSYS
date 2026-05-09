"""Content-addressed message blob store for full-input trace capture.

When ``TracingConfig.capture_full_input`` is enabled, the tracing collector
captures every message sent to a model on each agent step and stores the
canonical-JSON-serialized form in a sidecar directory under the trace
output (``{output_dir}/messages/<sha256>.json``). Each step span carries
an ``input_messages_ref`` attribute pointing into the store; readers
resolve via :meth:`MessageStore.reconstruct`.

Identical messages — system prompts, repeated tool results, identical
user turns — store once across the whole output directory. A 50-step
append-only branch with a 4 KB system prompt persists ~25 KB of unique
message bytes instead of ~625 KB of duplicated history; forks share the
prefix automatically.

The shape of the ``input_messages_ref`` dict written into span
attributes::

    {
        "history": ["<sha256-hash>", "<sha256-hash>", ...],   # full ordered list
        "base":    "<sha256-of-prev-history-list>" | None,    # diff anchor
        "patch":   [{"op": "add", "path": "/-",                # JSON-Patch ops
                     "value": "<sha256-hash>"},
                    ...] | None,
    }

``history`` is the authoritative resolution surface — readers walk it,
look each hash up via :meth:`MessageStore.read_blob`, and rebuild the
list. ``base`` and ``patch`` are convenience fields for git-like diff
display; viewers that want "show me only the new turn" read ``patch``
directly.

Architecturally a peer to ``redactor.py``: configured via
``TracingConfig``, applied once during ``AgentStartEvent`` handling so
all downstream consumers (NDJSON writer, telemetry sinks) see the
already-attached ref attribute uniformly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


HASH_FIELDS_TO_EXCLUDE = ("message_id",)
"""Fields excluded from hash input to preserve content-addressing.

UUIDs poison dedup — two messages with identical content but different
``message_id`` should hash identically. The serialized blob still
includes ``message_id`` (so reads round-trip the original UUID); only the
hash input strips them.
"""


def compute_message_hash(message: Dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of a canonical-JSON message.

    Canonical form: keys sorted, no whitespace, ``ensure_ascii=False``.
    The hash input excludes ``message_id`` (a per-instance UUID) so two
    messages with identical content but different ids share a blob.

    Stable across Python versions and platforms.
    """
    canonical = {k: v for k, v in message.items() if k not in HASH_FIELDS_TO_EXCLUDE}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_history_list(history: List[str]) -> str:
    """Hash an ordered list of message hashes (the ``base`` field's anchor).

    Used to identify a specific resolved-history shape so a child branch
    forking off step ``k`` of a parent branch can reference the
    parent's history-at-step-k by a single value.
    """
    payload = json.dumps(history, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def diff_history(
    prev: Optional[List[str]],
    curr: List[str],
) -> List[Dict[str, Any]]:
    """Build a JSON-Patch (RFC 6902) op list transforming ``prev`` into ``curr``.

    Optimised for the common append-only case (``curr == prev + new``):
    emits one ``add`` op per appended hash. Falls back to a full
    ``replace`` op when histories diverge in non-append ways (rare —
    happens after compaction or memory pruning). Patch ``value`` fields
    are message hashes, never inlined message bodies.
    """
    if prev is None:
        return [{"op": "add", "path": "/-", "value": h} for h in curr]
    if len(curr) >= len(prev) and curr[: len(prev)] == prev:
        return [{"op": "add", "path": "/-", "value": h} for h in curr[len(prev):]]
    return [{"op": "replace", "path": "", "value": list(curr)}]


class MessageStore(ABC):
    """Abstract content-addressed message blob store.

    Two implementations ship in-tree:

    * :class:`InMemoryMessageStore` — for tests and short-lived sessions.
    * :class:`FilesystemMessageStore` — production default; one JSON file
      per unique message under ``{base_dir}/messages/<hash>.json``.

    External users can plug in S3-backed, Redis-backed, or otherwise-
    distributed stores by subclassing.
    """

    @abstractmethod
    def write_blob(self, message: Dict[str, Any]) -> str:
        """Write ``message`` to the store and return its content hash.

        Idempotent: a second call with the same content is a no-op.
        """

    @abstractmethod
    def read_blob(self, hash: str) -> Optional[Dict[str, Any]]:
        """Return the message stored under ``hash`` or ``None`` if absent."""

    def reconstruct(self, ref: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve an ``input_messages_ref`` dict to the full message list.

        Walks ``ref["history"]`` and looks each hash up via
        :meth:`read_blob`. Missing blobs are returned as ``None``
        entries — readers can filter or surface them separately.
        """
        history: List[str] = list(ref.get("history") or [])
        return [self.read_blob(h) for h in history]


class InMemoryMessageStore(MessageStore):
    """Process-local message store. No persistence; for tests."""

    def __init__(self) -> None:
        self._blobs: Dict[str, Dict[str, Any]] = {}

    @property
    def blob_count(self) -> int:
        return len(self._blobs)

    def write_blob(self, message: Dict[str, Any]) -> str:
        h = compute_message_hash(message)
        if h not in self._blobs:
            # Store a copy so callers mutating the dict afterwards don't corrupt the store.
            self._blobs[h] = json.loads(json.dumps(message, ensure_ascii=False))
        return h

    def read_blob(self, hash: str) -> Optional[Dict[str, Any]]:
        blob = self._blobs.get(hash)
        if blob is None:
            return None
        return json.loads(json.dumps(blob, ensure_ascii=False))


class FilesystemMessageStore(MessageStore):
    """Production default: ``{base_dir}/messages/<hash>.json`` per blob.

    Atomic writes via ``tempfile`` + ``os.replace`` so concurrent writers
    on the same hash never observe a partial file. Idempotent — a blob
    that already exists is left untouched (cheap stat, no rewrite).

    The ``base_dir`` is typically ``TracingConfig.output_dir``: messages
    sit beside the ``{trace_id}.ndjson`` files and dedup *across* traces
    in the same directory.
    """

    DEFAULT_SUBDIR = "messages"

    def __init__(
        self,
        base_dir: pathlib.Path,
        *,
        subdir: str = DEFAULT_SUBDIR,
    ) -> None:
        self._base_dir = pathlib.Path(base_dir)
        self._subdir = subdir
        self._messages_dir = self._base_dir / subdir
        # Created lazily on first write to avoid touching disk in the no-capture path.

    @property
    def messages_dir(self) -> pathlib.Path:
        return self._messages_dir

    def _ensure_dir(self) -> None:
        self._messages_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, hash: str) -> pathlib.Path:
        return self._messages_dir / f"{hash}.json"

    def write_blob(self, message: Dict[str, Any]) -> str:
        h = compute_message_hash(message)
        target = self._path(h)
        if target.exists():
            return h  # already persisted; cheap stat, no rewrite
        self._ensure_dir()
        # Atomic write: tempfile in same dir, then os.replace.
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{h}.", suffix=".json.tmp", dir=str(self._messages_dir)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(message, f, ensure_ascii=False, separators=(",", ":"))
                os.replace(tmp_path, target)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning("FilesystemMessageStore failed to write blob %s: %s", h, e)
        return h

    def read_blob(self, hash: str) -> Optional[Dict[str, Any]]:
        target = self._path(hash)
        if not target.exists():
            return None
        try:
            with target.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("FilesystemMessageStore failed to read blob %s: %s", hash, e)
            return None


def build_input_messages_ref(
    messages: List[Dict[str, Any]],
    *,
    store: MessageStore,
    prev_history: Optional[List[str]],
) -> Dict[str, Any]:
    """Hash & store ``messages``, return the span-attribute ref dict.

    Mutates nothing in ``messages`` — caller is responsible for deep-copy
    and redaction before invocation. Returns the canonical
    ``input_messages_ref`` payload (``history`` / ``base`` / ``patch``)
    described in this module's docstring.
    """
    history: List[str] = [store.write_blob(m) for m in messages]
    base = _hash_history_list(prev_history) if prev_history else None
    patch = diff_history(prev_history, history)
    return {"history": history, "base": base, "patch": patch}
