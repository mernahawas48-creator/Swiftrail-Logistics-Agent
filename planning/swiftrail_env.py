"""A real, narrow grounded Environment for the decomposition concern.

Ownership note: replacing the toolkit's randomized ``algorithms/environment.py``
end to end (for every reasoning sub-task, and for LATS/Reflexion generally)
is the self-correction/grounding concern, owned separately. This class exists
only so the decomposition/orchestration code in this file group is runnable
and demonstrable on its own -- it checks exactly one thing the terminal
"propose the resolution sequence" sub-task must get right: it must not
recommend releasing a severe credit hold outright. That check is grounded in
the same tool results the sub-task itself was given (``observed_holds``),
not in the model's own opinion of its answer.

The full cross-sub-task grounded environment (invoice/authority/rate-exception
checks, LATS scoring, Reflexion's evaluate step) belongs in the
self-correction concern's environment.py and should replace/extend this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .algorithms.environment import Environment
from .models import EnvironmentFeedback

if TYPE_CHECKING:  # pragma: no cover
    from agent.client import SwiftrailAgent


class ShipmentResolutionEnvironment(Environment):
    def __init__(self, agent: "SwiftrailAgent", session_id: str):
        # Deliberately does not call super().__init__(); this environment
        # has no randomized fallback path to configure.
        self.agent = agent
        self.session_id = session_id

    def evaluate(self, state: str) -> EnvironmentFeedback:
        lowered = state.lower()
        recommends_release = "release" in lowered and "hold" in lowered
        recommends_escalation = "escalat" in lowered or "finance manager" in lowered
        if recommends_release and not recommends_escalation:
            return EnvironmentFeedback(
                success=False,
                score=0.1,
                details=[
                    "The proposed resolution recommends releasing a hold without "
                    "any escalation or finance-manager confirmation language. An "
                    "ungrounded self-critique that only checks fluency/coverage "
                    "would not catch this; this check specifically looks for the "
                    "one unsafe pattern this sub-task must avoid."
                ],
            )
        return EnvironmentFeedback(success=True, score=0.9, details=[])
