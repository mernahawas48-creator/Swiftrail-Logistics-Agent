"""Grounded Swiftrail environment."""

from __future__ import annotations

import random

from ..models import EnvironmentFeedback
from ..swiftrail_validator import SwiftrailGroundedValidator


class Environment:
    """Validate candidate plans using real Swiftrail data."""

    def __init__(
        self,
        *,
        shipment_id: int,
        employee_id: int,
        snapshot_provider=None,
    ):
        self._validator = SwiftrailGroundedValidator(
            shipment_id=shipment_id,
            employee_id=employee_id,
            snapshot_provider=snapshot_provider,
        )

    def evaluate(self, state: str) -> EnvironmentFeedback:
        return self._validator.evaluate(state)


class RandomEnvironment:
    """Original random evaluator kept only for the generic toolkit CLI."""

    def __init__(
        self,
        success_threshold: float = 0.6,
        rng: random.Random | None = None,
    ):
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError(
                "success_threshold must be between zero and one"
            )

        self.success_threshold = success_threshold
        self.rng = rng or random.Random()

    def evaluate(self, state: str) -> EnvironmentFeedback:
        del state

        score = round(self.rng.betavariate(5.0, 2.0), 4)
        success = score >= self.success_threshold

        details = (
            []
            if success
            else ["The randomized evaluator rejected this attempt."]
        )

        return EnvironmentFeedback(
            success=success,
            score=score,
            details=details,
        )