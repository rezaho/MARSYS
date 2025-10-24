"""
File editing with unified diff and search/replace support.

This module implements advanced editing strategies including unified diff
with flexible patching (inspired by Aider) for high success rates.
"""

import difflib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from .data_models import EditResult, EditFormat
from .config import FileOperationConfig

logger = logging.getLogger(__name__)


@dataclass
class ParsedHunk:
    """Represents a parsed diff hunk."""
    old_start: int  # Starting line in original (1-indexed)
    old_count: int  # Number of lines in original
    new_start: int  # Starting line in new (1-indexed)
    new_count: int  # Number of lines in new
    header: str  # The @@ header line
    context_before: List[str]  # Context lines before changes
    removed_lines: List[str]  # Lines to remove (-)
    added_lines: List[str]  # Lines to add (+)
    context_after: List[str]  # Context lines after changes


@dataclass
class ParsedDiff:
    """Represents a parsed unified diff."""
    file_path: str  # File path from diff
    hunks: List[ParsedHunk]  # List of hunks


class UnifiedDiffEditor:
    """
    Implements unified diff editing with flexible patching.

    Based on Aider's research showing 98% success rate with flexible patching.
    """

    def __init__(self, config: FileOperationConfig):
        """
        Initialize unified diff editor.

        Args:
            config: File operation configuration
        """
        self.config = config

    async def apply_diff(
        self,
        path: Path,
        patch: str,
        dry_run: bool = False,
        flexible: bool = None
    ) -> EditResult:
        """
        Apply unified diff patch to file.

        Args:
            path: File path to edit
            patch: Unified diff patch string
            dry_run: If True, don't apply changes, just return preview
            flexible: Enable flexible patching (default from config)

        Returns:
            EditResult with operation results
        """
        if flexible is None:
            flexible = self.config.enable_flexible_patching

        try:
            # Parse the unified diff
            parsed_diff = self.parse_unified_diff(patch)

            # Read current file content
            if not path.exists():
                return EditResult(
                    success=False,
                    error=f"File does not exist: {path}"
                )

            with open(path, 'r') as f:
                original_lines = f.readlines()

            # Apply patch with appropriate strategy
            if flexible and self.config.flexible_patch_max_attempts > 1:
                result = self._flexible_patch(original_lines, parsed_diff)
            else:
                result = self._exact_match_patch(original_lines, parsed_diff)

            if not result['success']:
                return EditResult(
                    success=False,
                    path=path,
                    error=result.get('error', 'Patch failed'),
                    hunks_total=len(parsed_diff.hunks)
                )

            new_lines = result['lines']
            hunks_applied = result['hunks_applied']
            strategy_used = result.get('strategy', 'exact_match')

            if dry_run:
                # Generate preview
                preview = self.generate_preview(original_lines, new_lines)
                return EditResult(
                    success=True,
                    path=path,
                    hunks_applied=hunks_applied,
                    hunks_total=len(parsed_diff.hunks),
                    lines_changed=self._count_changed_lines(original_lines, new_lines),
                    dry_run=True,
                    preview=preview,
                    strategy_used=strategy_used
                )

            # Apply changes
            with open(path, 'w') as f:
                f.writelines(new_lines)

            return EditResult(
                success=True,
                path=path,
                hunks_applied=hunks_applied,
                hunks_total=len(parsed_diff.hunks),
                lines_changed=self._count_changed_lines(original_lines, new_lines),
                dry_run=False,
                strategy_used=strategy_used,
                diff_applied=patch
            )

        except Exception as e:
            logger.error(f"Error applying diff to {path}: {e}", exc_info=True)
            return EditResult(
                success=False,
                path=path,
                error=str(e)
            )

    def parse_unified_diff(self, patch: str) -> ParsedDiff:
        """
        Parse unified diff format.

        Args:
            patch: Unified diff string

        Returns:
            ParsedDiff object

        Raises:
            ValueError: If patch format is invalid
        """
        lines = patch.split('\n')

        # Find file path from --- and +++ lines
        file_path = None
        for line in lines[:10]:  # Check first 10 lines
            if line.startswith('---'):
                file_path = line[4:].strip().split('\t')[0]
                if file_path.startswith('a/'):
                    file_path = file_path[2:]
                break

        if not file_path:
            file_path = "unknown"

        # Parse hunks
        hunks = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # Look for hunk header: @@ -old_start,old_count +new_start,new_count @@
            if line.startswith('@@'):
                hunk = self._parse_hunk(lines, i)
                if hunk:
                    hunks.append(hunk)
                    i += 1  # Will be incremented further in loop
                else:
                    i += 1
            else:
                i += 1

        if not hunks:
            raise ValueError("No valid hunks found in diff")

        return ParsedDiff(file_path=file_path, hunks=hunks)

    def _parse_hunk(self, lines: List[str], start_idx: int) -> Optional[ParsedHunk]:
        """Parse a single hunk from diff lines."""
        header = lines[start_idx]

        # Parse hunk header
        match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', header)
        if not match:
            return None

        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) else 1

        # Parse hunk content
        context_before = []
        removed_lines = []
        added_lines = []
        context_after = []

        in_change = False
        i = start_idx + 1

        while i < len(lines):
            line = lines[i]

            # Stop at next hunk or end
            if line.startswith('@@') or line.startswith('---') or line.startswith('+++'):
                break

            if line.startswith(' '):  # Context line
                if not in_change and not removed_lines and not added_lines:
                    context_before.append(line[1:] if len(line) > 1 else '')
                else:
                    in_change = True
                    context_after.append(line[1:] if len(line) > 1 else '')

            elif line.startswith('-'):  # Removed line
                in_change = True
                removed_lines.append(line[1:] if len(line) > 1 else '')
                context_after = []  # Reset context after

            elif line.startswith('+'):  # Added line
                in_change = True
                added_lines.append(line[1:] if len(line) > 1 else '')
                context_after = []  # Reset context after

            elif line.startswith('\\'):  # No newline indicator
                pass  # Ignore for now

            i += 1

        return ParsedHunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            header=header,
            context_before=context_before,
            removed_lines=removed_lines,
            added_lines=added_lines,
            context_after=context_after
        )

    def _exact_match_patch(
        self,
        original_lines: List[str],
        diff: ParsedDiff
    ) -> Dict[str, Any]:
        """
        Apply patch with exact matching (strict).

        Args:
            original_lines: Original file lines
            diff: Parsed diff

        Returns:
            Dict with success status and result lines
        """
        lines = original_lines.copy()
        hunks_applied = 0

        for hunk in diff.hunks:
            # Find the context in the file
            search_start = max(0, hunk.old_start - 10)
            search_end = min(len(lines), hunk.old_start + hunk.old_count + 10)

            # Look for exact match of context + removed lines
            found = False
            for line_num in range(search_start, search_end):
                if self._matches_hunk(lines, line_num, hunk):
                    # Apply the hunk
                    lines = self._apply_hunk(lines, line_num, hunk)
                    hunks_applied += 1
                    found = True
                    break

            if not found:
                return {
                    'success': False,
                    'error': f'Could not find exact match for hunk at line {hunk.old_start}',
                    'hunks_applied': hunks_applied
                }

        return {
            'success': True,
            'lines': lines,
            'hunks_applied': hunks_applied,
            'strategy': 'exact_match'
        }

    def _flexible_patch(
        self,
        original_lines: List[str],
        diff: ParsedDiff
    ) -> Dict[str, Any]:
        """
        Apply patch with flexible matching strategies.

        Tries progressively more permissive strategies:
        1. Exact match
        2. Whitespace normalization
        3. Context window adjustment
        4. Fuzzy matching

        Args:
            original_lines: Original file lines
            diff: Parsed diff

        Returns:
            Dict with success status and result lines
        """
        strategies = [
            ('exact_match', self._exact_match_patch),
            ('whitespace_normalized', self._whitespace_normalized_patch),
            ('adjusted_context', self._adjusted_context_patch),
            ('fuzzy_match', self._fuzzy_match_patch),
        ]

        for strategy_name, strategy_func in strategies[:self.config.flexible_patch_max_attempts]:
            result = strategy_func(original_lines, diff)
            if result['success']:
                result['strategy'] = strategy_name
                logger.debug(f"Patch succeeded with strategy: {strategy_name}")
                return result

        # All strategies failed
        return {
            'success': False,
            'error': 'Could not apply patch with any strategy',
            'hunks_applied': 0
        }

    def _whitespace_normalized_patch(
        self,
        original_lines: List[str],
        diff: ParsedDiff
    ) -> Dict[str, Any]:
        """Apply patch with whitespace normalization."""
        # Similar to exact_match but normalize whitespace
        lines = original_lines.copy()
        hunks_applied = 0

        for hunk in diff.hunks:
            search_start = max(0, hunk.old_start - 10)
            search_end = min(len(lines), hunk.old_start + hunk.old_count + 10)

            found = False
            for line_num in range(search_start, search_end):
                if self._matches_hunk_normalized(lines, line_num, hunk):
                    lines = self._apply_hunk(lines, line_num, hunk)
                    hunks_applied += 1
                    found = True
                    break

            if not found:
                return {'success': False, 'hunks_applied': hunks_applied}

        return {
            'success': True,
            'lines': lines,
            'hunks_applied': hunks_applied
        }

    def _adjusted_context_patch(
        self,
        original_lines: List[str],
        diff: ParsedDiff
    ) -> Dict[str, Any]:
        """Apply patch with adjusted context window."""
        # Try with different context sizes
        lines = original_lines.copy()
        hunks_applied = 0

        for hunk in diff.hunks:
            # Try wider search range
            search_start = max(0, hunk.old_start - 20)
            search_end = min(len(lines), hunk.old_start + hunk.old_count + 20)

            found = False
            for line_num in range(search_start, search_end):
                if self._matches_hunk_fuzzy(lines, line_num, hunk, threshold=0.8):
                    lines = self._apply_hunk(lines, line_num, hunk)
                    hunks_applied += 1
                    found = True
                    break

            if not found:
                return {'success': False, 'hunks_applied': hunks_applied}

        return {
            'success': True,
            'lines': lines,
            'hunks_applied': hunks_applied
        }

    def _fuzzy_match_patch(
        self,
        original_lines: List[str],
        diff: ParsedDiff
    ) -> Dict[str, Any]:
        """Apply patch with fuzzy matching."""
        lines = original_lines.copy()
        hunks_applied = 0

        for hunk in diff.hunks:
            # Very wide search
            search_start = 0
            search_end = len(lines)

            best_match = None
            best_score = 0

            for line_num in range(search_start, search_end):
                score = self._fuzzy_match_score(lines, line_num, hunk)
                if score > best_score:
                    best_score = score
                    best_match = line_num

            # Apply if score is good enough
            if best_match is not None and best_score > 0.6:
                lines = self._apply_hunk(lines, best_match, hunk)
                hunks_applied += 1
            else:
                return {'success': False, 'hunks_applied': hunks_applied}

        return {
            'success': True,
            'lines': lines,
            'hunks_applied': hunks_applied
        }

    def _matches_hunk(
        self,
        lines: List[str],
        line_num: int,
        hunk: ParsedHunk
    ) -> bool:
        """Check if hunk matches at given line number (exact)."""
        # Match removed lines
        for i, removed_line in enumerate(hunk.removed_lines):
            if line_num + i >= len(lines):
                return False
            if lines[line_num + i].rstrip('\n') != removed_line.rstrip('\n'):
                return False
        return True

    def _matches_hunk_normalized(
        self,
        lines: List[str],
        line_num: int,
        hunk: ParsedHunk
    ) -> bool:
        """Check if hunk matches with whitespace normalization."""
        for i, removed_line in enumerate(hunk.removed_lines):
            if line_num + i >= len(lines):
                return False
            if lines[line_num + i].strip() != removed_line.strip():
                return False
        return True

    def _matches_hunk_fuzzy(
        self,
        lines: List[str],
        line_num: int,
        hunk: ParsedHunk,
        threshold: float = 0.8
    ) -> bool:
        """Check if hunk matches with fuzzy matching."""
        score = self._fuzzy_match_score(lines, line_num, hunk)
        return score >= threshold

    def _fuzzy_match_score(
        self,
        lines: List[str],
        line_num: int,
        hunk: ParsedHunk
    ) -> float:
        """Calculate fuzzy match score for hunk at line."""
        if not hunk.removed_lines:
            return 1.0

        matches = 0
        total = len(hunk.removed_lines)

        for i, removed_line in enumerate(hunk.removed_lines):
            if line_num + i >= len(lines):
                break

            # Use difflib for similarity
            similarity = difflib.SequenceMatcher(
                None,
                lines[line_num + i].strip(),
                removed_line.strip()
            ).ratio()

            if similarity > 0.8:
                matches += 1

        return matches / total if total > 0 else 0

    def _apply_hunk(
        self,
        lines: List[str],
        line_num: int,
        hunk: ParsedHunk
    ) -> List[str]:
        """Apply a hunk at the specified line number."""
        # Remove old lines
        new_lines = lines[:line_num]

        # Add new lines
        for added_line in hunk.added_lines:
            if not added_line.endswith('\n'):
                added_line += '\n'
            new_lines.append(added_line)

        # Add remaining lines after the hunk
        skip_lines = len(hunk.removed_lines)
        new_lines.extend(lines[line_num + skip_lines:])

        return new_lines

    def generate_preview(
        self,
        original_lines: List[str],
        new_lines: List[str]
    ) -> str:
        """Generate a preview of changes."""
        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            lineterm='',
            n=3  # 3 lines of context
        )

        return '\n'.join(diff)

    def _count_changed_lines(
        self,
        original_lines: List[str],
        new_lines: List[str]
    ) -> int:
        """Count number of lines changed."""
        diff = list(difflib.unified_diff(original_lines, new_lines))
        # Count lines starting with + or - (excluding +++ and ---)
        changed = sum(1 for line in diff if line.startswith(('+', '-')) and not line.startswith(('+++', '---')))
        return changed


class SearchReplaceEditor:
    """Simple search/replace editor."""

    async def edit(
        self,
        path: Path,
        search: str,
        replace: str,
        replace_all: bool = False
    ) -> EditResult:
        """
        Apply search/replace edit.

        Args:
            path: File path
            search: Text to search for
            replace: Replacement text
            replace_all: If True, replace all occurrences

        Returns:
            EditResult
        """
        try:
            with open(path, 'r') as f:
                content = f.read()

            # Count occurrences
            count = content.count(search)

            if count == 0:
                return EditResult(
                    success=False,
                    path=path,
                    error=f"Search text not found: '{search[:50]}...'"
                )

            # Perform replacement
            if replace_all:
                new_content = content.replace(search, replace)
            else:
                # Replace only the first occurrence
                new_content = content.replace(search, replace, 1)

            # Write back
            with open(path, 'w') as f:
                f.write(new_content)

            return EditResult(
                success=True,
                path=path,
                lines_changed=count if replace_all else 1
            )

        except Exception as e:
            return EditResult(
                success=False,
                path=path,
                error=str(e)
            )
