"""
Routing module for intelligent decision-making in the MARS coordination system.

The Router takes validation results and determines the next execution steps,
converting abstract actions into concrete execution instructions.
"""

from .router import Router
from .types import (
    RoutingDecision, 
    ExecutionStep, 
    StepType,
    BranchSpec,
    RoutingContext
)

__all__ = [
    'Router',
    'RoutingDecision',
    'ExecutionStep',
    'StepType',
    'BranchSpec',
    'RoutingContext'
]