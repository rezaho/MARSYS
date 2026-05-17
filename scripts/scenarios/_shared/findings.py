"""Findings collector for scenario journeys.

A finding is a structured observation about product behavior the user
should know about. Severity levels follow the implementation-reviewer
convention: critical / important / nit / info. The runner prints a
table to stdout and writes the full JSON to scenarios/output/.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Severity = Literal["critical", "important", "nit", "info"]


@dataclass
class Finding:
    severity: Severity
    surface: str
    summary: str
    detail: str | None = None
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None


@dataclass
class JourneyReport:
    name: str
    started_at: str
    finished_at: str | None = None
    duration_ms: float | None = None
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def count(self, severity: Severity) -> int:
        return sum(1 for f in self.findings if f.severity == severity)


class FindingsCollector:
    def __init__(self, journey_name: str) -> None:
        self.report = JourneyReport(
            name=journey_name,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._t0 = time.monotonic()

    def critical(self, surface: str, summary: str, **kwargs: Any) -> None:
        self.report.add(Finding(severity="critical", surface=surface, summary=summary, **kwargs))

    def important(self, surface: str, summary: str, **kwargs: Any) -> None:
        self.report.add(Finding(severity="important", surface=surface, summary=summary, **kwargs))

    def nit(self, surface: str, summary: str, **kwargs: Any) -> None:
        self.report.add(Finding(severity="nit", surface=surface, summary=summary, **kwargs))

    def info(self, surface: str, summary: str, **kwargs: Any) -> None:
        self.report.add(Finding(severity="info", surface=surface, summary=summary, **kwargs))

    def finalize(self) -> JourneyReport:
        self.report.finished_at = datetime.now(timezone.utc).isoformat()
        self.report.duration_ms = (time.monotonic() - self._t0) * 1000
        return self.report

    def render_summary(self, stream: Any = sys.stdout) -> None:
        r = self.report
        crit = r.count("critical")
        imp = r.count("important")
        nit = r.count("nit")
        info = r.count("info")
        print(f"\n=== {r.name} ===", file=stream)
        print(
            f"  critical={crit}  important={imp}  nit={nit}  info={info}  "
            f"duration={r.duration_ms:.0f}ms",
            file=stream,
        )
        for finding in r.findings:
            tag = f"[{finding.severity.upper():>9}]"
            print(f"  {tag} {finding.surface}: {finding.summary}", file=stream)
            if finding.detail:
                for line in finding.detail.splitlines():
                    print(f"      {line}", file=stream)

    def write_json(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = self.report.name.replace(" ", "_").replace("/", "_")
        path = output_dir / f"{slug}-{stamp}.json"
        payload = asdict(self.report)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
