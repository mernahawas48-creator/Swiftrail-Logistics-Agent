from __future__ import annotations

from planning.algorithms.self_refine import reflect_and_refine
from planning.environment import Environment
from planning_eval.evaluate_self_correction import ScriptedLLM
from planning_eval.evaluation_suite import fixed_cases


def run() -> None:
    case = fixed_cases()[0]
    env = Environment(
        shipment_id=case.shipment_id,
        employee_id=case.employee_id,
        snapshot_provider=lambda snapshot=case.snapshot: snapshot,
    )

    acting_model = ScriptedLLM([case.good_candidate])
    independent_critic = ScriptedLLM([
        "The release_credit_hold action is unauthorized for a sales_rep; escalate to finance_manager instead."
    ])

    result = reflect_and_refine(
        case.task,
        case.bad_candidate,
        acting_model,
        critic_llm=independent_critic,
        environment=env,
    )

    print("Independent critic calls:", independent_critic.calls)
    print("Acting model calls:", acting_model.calls)
    print("Grounded issues:", result.grounded_issues)
    print("Revised plan:\n", result.revised)
    print("Revision passed:", bool(result.revision_feedback and result.revision_feedback.success))


if __name__ == "__main__":
    run()
