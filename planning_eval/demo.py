from __future__ import annotations

import json
from planning.environment import Environment
from planning_eval.evaluation_suite import fixed_cases
from planning.algorithms.reflexion import reflexion


class Response:
    def __init__(self, content):
        self.content = content


class DemoLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, messages, temperature=0.2):
        return Response(self.responses.pop(0))


def main():
    case = fixed_cases()[0]
    env = Environment(
        shipment_id=case.shipment_id,
        employee_id=case.employee_id,
        snapshot_provider=lambda: case.snapshot,
    )
    llm = DemoLLM([
        case.bad_candidate,
        case.good_candidate,
    ])
    critic = DemoLLM([
        "I must verify employee authority before proposing a credit-hold release; a sales representative must escalate to finance_manager."
    ])
    result = reflexion(case.task, llm, env, max_trials=2, memory_size=1, critic_llm=critic)
    transcript = {
        "request": case.task,
        "trial_1": {
            "attempt": result.trials[0].attempt,
            "success": result.trials[0].feedback.success,
            "feedback": result.trials[0].feedback.details,
            "reflection": result.trials[0].reflection,
        },
        "trial_2": {
            "attempt": result.trials[1].attempt,
            "success": result.trials[1].feedback.success,
            "feedback": result.trials[1].feedback.details,
        },
        "final_success": result.success,
        "episodic_memory": result.memory,
    }
    print(json.dumps(transcript, indent=2))


if __name__ == "__main__":
    main()
