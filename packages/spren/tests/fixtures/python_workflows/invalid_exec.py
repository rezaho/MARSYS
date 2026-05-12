"""Rejected — contains an exec(...) call.

The importer must NEVER execute the file's code. The companion test asserts
the side-effect-only marker file does not appear after this fixture is
parsed (the side effect would have run if exec actually fired).
"""
from pathlib import Path

# If the importer were to execute this module, the marker file would appear
# at /tmp/spren-importer-must-not-execute.txt. The test asserts the marker
# does NOT exist after parsing.
exec("Path('/tmp/spren-importer-must-not-execute.txt').write_text('boom')")
