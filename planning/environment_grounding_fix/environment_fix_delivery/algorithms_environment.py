"""Grounded Swiftrail environment.

The lab is explicit about this file: "algorithms/environment.py ships with
a deliberately fake evaluator... specifically so you replace it." So the
real grounded logic (Ganna's `SwiftrailGroundedValidator`, validating a
candidate plan against the live shipment/customer/invoice/hold/rate-exception
data) lives here now, under the same `Environment` name every existing
LATS/Reflexion/planning_router/orchestrator call site already imports --
nothing downstream needs to change its import.

The toolkit's original stochastic evaluator is kept, renamed to
`RandomEnvironment`, only for planning/cli.py's generic, non-Swiftrail demo
goal. It is never used for any Swiftrail sub-task and is never the thing
named `Environment`.
"""

from __future__ import annotations

import random

from ..models import EnvironmentFeedback
from ..swiftrail_validator import SwiftrailGroundedValidator


class Environment:
    """Validates a candidate plan against the real Swiftrail database.
    No randomized scoring anywhere in this class."""

    def __init__(self, *, shipment_id: int, employee_id: int, snapshot_provider=None):
        self._validator = SwiftrailGroundedValidator(
            shipment_id=shipment_id,
            employee_id=employee_id,
            snapshot_provider=snapshot_provider,
        )

    def evaluate(self, state: str) -> EnvironmentFeedback:
        return self._validator.evaluate(state)


class RandomEnvironment:
    """The reference toolkit's original stochastic evaluator, preserved
    only so planning/cli.py's generic demo (unrelated to any real Swiftrail
    request) still has something to run against."""

    def __init__(self, success_threshold: float = 0.6, rng: random.Random | None = None):
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError("success_threshold must be between zero and one")
        self.success_threshold = success_threshold
        self.rng = rng or random.Random()

    def evaluate(self, state: str) -> EnvironmentFeedback:
        del state  # This evaluator intentionally ignores the candidate contents.
        score = round(self.rng.betavariate(5.0, 2.0), 4)
        success = score >= self.success_threshold
        details = [] if success else ["The randomized evaluator rejected this attempt."]
        return EnvironmentFeedback(success=success, score=score, details=details)
