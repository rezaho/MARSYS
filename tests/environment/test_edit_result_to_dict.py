"""Tests for ``EditResult.to_dict()`` — the LLM-facing tool-result shape.

The dataclass is shared between write and edit operations. Before this
fix, ``to_dict()`` always emitted every field — so a successful write
returned ``{success: true, path: ..., lines_changed: N, hunks_applied:
0, hunks_total: 0, dry_run: false, preview: null, strategy_used: null,
diff_applied: null}``. LLMs reading this concluded "0 hunks applied,
no strategy used, no diff applied — looks like the write didn't take,
retry." Empirically observed: an agent looped 16 times on successful
``file_operations(write, ...)`` calls because the result shape signalled
failure despite ``success: true``.

This module verifies:
- Write-shaped results emit only the load-bearing fields
- Edit-shaped results retain the edit-specific fields they actually use
- Failure results carry ``error`` regardless
- Path is POSIX-normalized (forward slashes) on all OSes
"""
from __future__ import annotations

from pathlib import Path

from marsys.environment.file_operations.data_models import EditResult


class TestEditResultToDictWriteShape:
    """A write operation builds ``EditResult`` with only ``success``,
    ``path``, ``lines_changed`` set. Edit-specific fields stay at
    default. ``to_dict()`` must drop the defaulted edit fields so the
    LLM sees a clean write shape."""

    def test_write_result_omits_edit_specific_fields(self):
        result = EditResult(
            success=True,
            path=Path("./output/joke.md"),
            lines_changed=6,
        )
        d = result.to_dict()
        # Load-bearing for write — keep
        assert d["success"] is True
        assert d["path"] == "./output/joke.md"
        assert d["lines_changed"] == 6
        # Edit-specific — must NOT be in the dict (defaults).
        assert "hunks_applied" not in d
        assert "hunks_total" not in d
        assert "dry_run" not in d
        assert "preview" not in d
        assert "warnings" not in d
        assert "strategy_used" not in d
        assert "diff_applied" not in d
        # No ``error`` key when error is None — keeps the success
        # path uncluttered.
        assert "error" not in d

    def test_write_result_includes_error_when_present(self):
        """A failed write still surfaces the error string."""
        result = EditResult(
            success=False,
            path=Path("./output/joke.md"),
            error="Permission denied",
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["path"] == "./output/joke.md"
        assert d["error"] == "Permission denied"
        # Still no edit-specific defaults.
        assert "hunks_applied" not in d
        assert "strategy_used" not in d


class TestEditResultToDictEditShape:
    """A genuine edit populates ``hunks_applied``, ``hunks_total``,
    ``strategy_used`` etc. ``to_dict()`` must include them so the LLM
    sees the actual edit telemetry."""

    def test_edit_result_includes_hunks_when_total_nonzero(self):
        result = EditResult(
            success=True,
            path=Path("./code.py"),
            hunks_applied=3,
            hunks_total=3,
            lines_changed=12,
            strategy_used="unified_diff",
        )
        d = result.to_dict()
        assert d["hunks_applied"] == 3
        assert d["hunks_total"] == 3
        assert d["lines_changed"] == 12
        assert d["strategy_used"] == "unified_diff"

    def test_partial_edit_still_includes_hunk_counts(self):
        """A partial edit (some hunks failed) needs both ``applied``
        and ``total`` so the LLM can see what didn't apply."""
        result = EditResult(
            success=True,
            path=Path("./code.py"),
            hunks_applied=2,
            hunks_total=3,
            lines_changed=8,
        )
        d = result.to_dict()
        assert d["hunks_applied"] == 2
        assert d["hunks_total"] == 3

    def test_dry_run_emits_dry_run_flag_and_preview(self):
        result = EditResult(
            success=True,
            path=Path("./code.py"),
            dry_run=True,
            preview="would change line 5",
            lines_changed=1,
        )
        d = result.to_dict()
        assert d["dry_run"] is True
        assert d["preview"] == "would change line 5"

    def test_warnings_emitted_when_non_empty(self):
        result = EditResult(
            success=True,
            path=Path("./code.py"),
            lines_changed=4,
            warnings=["trailing whitespace normalized"],
        )
        d = result.to_dict()
        assert d["warnings"] == ["trailing whitespace normalized"]

    def test_warnings_not_emitted_when_empty(self):
        """Empty warnings list is the default — don't expose it as
        ``warnings: []`` because LLMs sometimes interpret empty
        collections as "something was supposed to be here but wasn't"."""
        result = EditResult(
            success=True,
            path=Path("./code.py"),
            lines_changed=4,
        )
        d = result.to_dict()
        assert "warnings" not in d


class TestEditResultToDictPathNormalization:
    """The path in the tool result must be POSIX-normalized regardless
    of host OS — the LLM passes ``./output/joke.md`` and reading back
    ``output\\joke.md`` (Windows) creates needless dissonance."""

    def test_posix_path_unchanged(self):
        result = EditResult(success=True, path=Path("./output/joke.md"))
        d = result.to_dict()
        assert d["path"] == "./output/joke.md"

    def test_windows_backslash_path_normalized(self):
        """Simulate a Path that stringifies with backslashes (Windows
        host). ``to_dict`` must replace them with forward slashes."""
        # Synthesize the Windows-style stringification — Path normalizes
        # to the host's separator at str() time, so we synthesize by
        # passing a string instead of a Path and checking the dict.
        result = EditResult(success=True)
        result.path = "output\\joke.md"  # type: ignore[assignment]
        d = result.to_dict()
        # Backslashes flipped, leading ``./`` restored so the agent
        # sees the same shape it sent.
        assert d["path"] == "./output/joke.md"

    def test_absolute_path_does_not_get_dot_prefix(self):
        """An absolute path stays absolute — don't prepend ``./``."""
        result = EditResult(success=True, path=Path("/tmp/abs.txt"))
        d = result.to_dict()
        assert d["path"] == "/tmp/abs.txt"

    def test_windows_drive_path_not_prefixed(self):
        result = EditResult(success=True)
        result.path = "C:\\Users\\file.txt"  # type: ignore[assignment]
        d = result.to_dict()
        # Backslashes flipped; drive-letter prefix preserved without
        # adding ``./``.
        assert d["path"] == "C:/Users/file.txt"

    def test_no_path_emits_no_path_key(self):
        """``path is None`` should not produce ``path: null``."""
        result = EditResult(success=False, error="something")
        d = result.to_dict()
        # path absent (None default) → no key
        assert "path" not in d


class TestEditResultToDictRegressionFromTrace:
    """Regression guard for the exact case observed in the trace of run
    01KSPWF8WHCT6DZEE2SD13H68H: a write returns ``{success: true, path:
    "working\\joke.md", hunks_applied: 0, hunks_total: 0, lines_changed:
    6, dry_run: false, preview: null, warnings: [], error: null,
    strategy_used: null, diff_applied: null}``. The LLM reads this and
    loops.

    Post-fix shape should be ``{success: true, path: "working/joke.md",
    lines_changed: 6}`` — clean, unambiguous success."""

    def test_trace_observed_shape(self):
        result = EditResult(success=True, lines_changed=6)
        result.path = "working\\joke.md"  # type: ignore[assignment]
        d = result.to_dict()
        # Exact post-fix shape — agent sent ``./working/joke.md``,
        # the response should show ``./working/joke.md`` (same format).
        assert d == {
            "success": True,
            "path": "./working/joke.md",
            "lines_changed": 6,
        }
