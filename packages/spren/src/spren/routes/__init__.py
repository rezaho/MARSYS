"""HTTP route modules for the Spren sidecar.

Each router is constructed via a ``make_*`` factory taking dependencies
(auth, db) explicitly so there are no module-level globals.
"""
from __future__ import annotations
