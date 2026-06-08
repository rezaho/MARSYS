"""Configuration for the AG-UI event stream translator."""

from dataclasses import dataclass


@dataclass
class AGGUIConfig:
    """Configuration for the AG-UI translator.

    Off by default — the translator is an optional add-on consumed by AG-UI-speaking
    UIs (SSE clients, hosted control planes, third-party clients). Framework users running
    raw ``python my_workflow.py`` typically leave this off.

    Requires the optional ``aggui`` extras: ``pip install 'marsys[aggui]'``.
    """

    enabled: bool = False
    queue_max_size: int = 10000
