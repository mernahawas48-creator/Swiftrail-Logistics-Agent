"""Backward-compatible re-export.

The real grounded Environment now lives in algorithms/environment.py (the
file the lab explicitly names for this). This module used to define its
own separate `Environment` class with the same name -- kept only as a
re-export so `planning_eval/*.py`'s existing `from planning.environment
import Environment` imports don't all need to change.
"""

from .algorithms.environment import Environment

__all__ = ["Environment"]
