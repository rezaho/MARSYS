"""The file-tool surface contract (no disk writes outside tmp).

Two halves of one security invariant:

1. ``FileOperationTools.get_tools()`` returns EXACTLY the 8 individual
   tools plus the ``file_operations`` dispatcher — consumers bind file
   access through a constructed (confinable) instance.
2. The module-level ``AVAILABLE_TOOLS`` registry carries NO
   host-filesystem-reaching defaults: ``RealToolExecutor`` builds its
   fallback registry from ``AVAILABLE_TOOLS`` at construction with
   name fall-through resolution, so an unconfined entry there is
   reachable even when callers bind confined tools onto their agents.
"""

from marsys.environment.file_operations.config import FileOperationConfig
from marsys.environment.file_operations.core import FileOperationTools
from marsys.environment.tools import AVAILABLE_TOOLS

EXPECTED_TOOL_NAMES = {
    "read_file",
    "write_file",
    "edit_file",
    "search_files",
    "get_file_structure",
    "read_section",
    "list_files",
    "create_directory",
    "file_operations",
}


def test_get_tools_returns_exactly_the_nine_names(tmp_path):
    tools = FileOperationTools(
        config=FileOperationConfig(base_directory=tmp_path)
    ).get_tools()
    assert set(tools.keys()) == EXPECTED_TOOL_NAMES
    assert all(callable(fn) for fn in tools.values())


def test_available_tools_has_no_host_filesystem_defaults():
    assert "file_operations" not in AVAILABLE_TOOLS
    assert "read_file" not in AVAILABLE_TOOLS
