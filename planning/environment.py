"""Compatibility environment module replacing the reference randomized evaluator."""

from .models import EnvironmentFeedback
from .swiftrail_validator import SwiftrailGroundedValidator


class Environment:
    """Swiftrail grounded environment; no randomized scoring is used."""

    def __init__(self, *, shipment_id: int, employee_id: int, snapshot_provider=None):
        self.validator = SwiftrailGroundedValidator(
            shipment_id=shipment_id,
            employee_id=employee_id,
            snapshot_provider=snapshot_provider,
        )

    def evaluate(self, state: str) -> EnvironmentFeedback:
        return self.validator.evaluate(state)
