from __future__ import annotations

import json
from pathlib import Path

from planning.algorithms.dynamic_decomposition import DynamicStep
from planning.algorithms.environment import Environment
from planning.divergence import compute_divergence
from planning.models import Plan
from planning.planning_router import (
    PlanningMethod,
    PlanningProfile,
    route_subtask,
)
from planning.swiftrail_subtask import (
    SubtaskKind,
    SubtaskMeta,
    SwiftrailPlan,
)
from planning_eval.evaluate_self_correction import run_case
from planning_eval.evaluation_suite import fixed_cases

ARTIFACTS_DIR = Path("artifacts")


def _tool_meta(tool_name: str) -> SubtaskMeta:
    return SubtaskMeta(
        kind=SubtaskKind.TOOL_CALL,
        tool_name=tool_name,
        build_args=lambda session_id, outputs: {},
    )


def _reasoning_meta() -> SubtaskMeta:
    return SubtaskMeta(
        kind=SubtaskKind.REASONING,
        needs_branching=True,
        high_stakes=True,
        grounded_validation_available=True,
    )


def _static_comparison_plan() -> SwiftrailPlan:
    plan = Plan.model_validate(
        {
            "goal": "Review shipment and customer financial blockers.",
            "tasks": [
                {
                    "id": "t1",
                    "instruction": "Fetch shipment status",
                    "depends_on": [],
                },
                {
                    "id": "t2",
                    "instruction": "Fetch customer state",
                    "depends_on": [],
                },
                {
                    "id": "t3",
                    "instruction": "Fetch credit holds",
                    "depends_on": [],
                },
                {
                    "id": "t4",
                    "instruction": "Produce the safe final resolution",
                    "depends_on": ["t1", "t2", "t3"],
                },
            ],
        }
    )

    return SwiftrailPlan(
        plan=plan,
        meta={
            "t1": _tool_meta("get_shipment_status"),
            "t2": _tool_meta("search_customer"),
            "t3": _tool_meta("list_customer_credit_holds"),
            "t4": _reasoning_meta(),
        },
    )


def evaluate_decomposition_methods() -> list[dict]:
    static_plan = _static_comparison_plan()

    stable_dynamic_steps = [
        DynamicStep(
            step=1,
            kind=SubtaskKind.TOOL_CALL,
            tool_name="get_shipment_status",
            instruction=None,
            output="{}",
            raw={},
        ),
        DynamicStep(
            step=2,
            kind=SubtaskKind.TOOL_CALL,
            tool_name="search_customer",
            instruction=None,
            output="{}",
            raw={},
        ),
        DynamicStep(
            step=3,
            kind=SubtaskKind.TOOL_CALL,
            tool_name="list_customer_credit_holds",
            instruction=None,
            output="{}",
            raw={},
        ),
        DynamicStep(
            step=4,
            kind=SubtaskKind.REASONING,
            tool_name=None,
            instruction="Produce final resolution",
            output="Safe resolution.",
            forced=False,
        ),
    ]

    stable_report = compute_divergence(
        static_plan,
        stable_dynamic_steps,
    )

    severe_hold_dynamic_steps = [
        DynamicStep(
            step=1,
            kind=SubtaskKind.TOOL_CALL,
            tool_name="get_shipment_status",
            instruction=None,
            output="{}",
            raw={},
        ),
        DynamicStep(
            step=2,
            kind=SubtaskKind.TOOL_CALL,
            tool_name="list_customer_credit_holds",
            instruction=None,
            output="{}",
            raw={},
        ),
        DynamicStep(
            step=3,
            kind=SubtaskKind.REASONING,
            tool_name=None,
            instruction="Escalate severe hold",
            output="Escalate to finance manager.",
            forced=True,
        ),
    ]

    severe_report = compute_divergence(
        static_plan,
        severe_hold_dynamic_steps,
    )

    return [
        {
            "case": "stable_evidence",
            "preferred_method": "decomposition_first",
            "diverged": stable_report.diverged,
            "static_tool_sequence": (
                stable_report.static_tool_sequence
            ),
            "dynamic_tool_sequence": (
                stable_report.dynamic_tool_sequence
            ),
            "reason": (
                "No observation changed the planned sequence, "
                "so an up-front DAG is sufficient."
            ),
        },
        {
            "case": "severe_hold_discovered",
            "preferred_method": "dynamic",
            "diverged": severe_report.diverged,
            "static_tool_sequence": (
                severe_report.static_tool_sequence
            ),
            "dynamic_tool_sequence": (
                severe_report.dynamic_tool_sequence
            ),
            "divergence_point": (
                severe_report.point.index
                if severe_report.point
                else None
            ),
            "reason": (
                severe_report.point.reason
                if severe_report.point
                else "No divergence detected."
            ),
        },
    ]


def evaluate_planning_router() -> list[dict]:
    cases = [
        (
            "linear_evidence_synthesis",
            PlanningProfile(
                instruction=(
                    "Summarize the confirmed blockers and "
                    "authority constraints."
                ),
                needs_branching=False,
                high_stakes=False,
                grounded_validation_available=True,
            ),
            PlanningMethod.PLAN_AND_SOLVE,
        ),
        (
            "compare_resolution_alternatives",
            PlanningProfile(
                instruction=(
                    "Compare multiple safe resolution sequences."
                ),
                needs_branching=True,
                high_stakes=False,
                grounded_validation_available=True,
            ),
            PlanningMethod.TREE_OF_THOUGHTS,
        ),
        (
            "high_stakes_final_decision",
            PlanningProfile(
                instruction=(
                    "Choose the final safe executable or "
                    "escalation plan."
                ),
                needs_branching=True,
                high_stakes=True,
                grounded_validation_available=True,
            ),
            PlanningMethod.LATS,
        ),
    ]

    rows = []

    for name, profile, expected in cases:
        decision = route_subtask(profile)

        rows.append(
            {
                "case": name,
                "expected_method": expected.value,
                "selected_method": decision.method.value,
                "correct_route": (
                    decision.method == expected
                ),
                "rationale": decision.rationale,
            }
        )

    return rows


def evaluate_grounding() -> list[dict]:
    rows = []

    for case in fixed_cases():
        environment = Environment(
            shipment_id=case.shipment_id,
            employee_id=case.employee_id,
            snapshot_provider=(
                lambda snapshot=case.snapshot: snapshot
            ),
        )

        grounded_bad = environment.evaluate(
            case.bad_candidate
        )
        grounded_good = environment.evaluate(
            case.good_candidate
        )

        rows.append(
            {
                "case": case.name,
                "ungrounded_bad_candidate_accepted": True,
                "grounded_bad_candidate_accepted": (
                    grounded_bad.success
                ),
                "grounded_bad_score": grounded_bad.score,
                "grounded_issues": grounded_bad.details,
                "grounded_good_candidate_accepted": (
                    grounded_good.success
                ),
                "grounded_good_score": grounded_good.score,
                "grounding_caught_false_positive": (
                    not grounded_bad.success
                    and grounded_good.success
                ),
            }
        )

    return rows


def evaluate_self_correction() -> list[dict]:
    rows = []

    for case in fixed_cases():
        case_rows = run_case(case)

        for row in case_rows:
            result = dict(row)
            result["case"] = case.name
            rows.append(result)

    return rows


def build_summary() -> dict:
    decomposition = evaluate_decomposition_methods()
    routing = evaluate_planning_router()
    grounding = evaluate_grounding()
    self_correction = evaluate_self_correction()

    return {
        "evaluation_mode": (
            "deterministic offline evaluation without API keys"
        ),
        "decomposition_first_vs_dynamic": decomposition,
        "ps_vs_tot_vs_lats_routing": routing,
        "grounded_vs_ungrounded": grounding,
        "self_refine_vs_reflexion": self_correction,
        "checks": {
            "decomposition_cases_pass": (
                decomposition[0]["diverged"] is False
                and decomposition[1]["diverged"] is True
            ),
            "all_router_cases_pass": all(
                row["correct_route"]
                for row in routing
            ),
            "all_grounding_cases_pass": all(
                row["grounding_caught_false_positive"]
                for row in grounding
            ),
            "all_self_correction_cases_pass": all(
                bool(row["success"])
                for row in self_correction
            ),
        },
        "production_note": (
            "Real provider latency, token usage, and billing "
            "must come from a live model run. This offline "
            "evaluation does not invent production metrics."
        ),
    }


def _markdown(summary: dict) -> str:
    lines = [
        "# Final Planning Evaluation",
        "",
        "Evaluation mode: deterministic offline evaluation "
        "without API keys.",
        "",
        "## Decomposition-First vs Dynamic",
        "",
        "| Case | Preferred | Diverged |",
        "|---|---|---:|",
    ]

    for row in summary[
        "decomposition_first_vs_dynamic"
    ]:
        lines.append(
            f"| {row['case']} | "
            f"{row['preferred_method']} | "
            f"{row['diverged']} |"
        )

    lines.extend(
        [
            "",
            "## PS vs ToT vs LATS Routing",
            "",
            "| Case | Expected | Selected | Correct |",
            "|---|---|---|---:|",
        ]
    )

    for row in summary["ps_vs_tot_vs_lats_routing"]:
        lines.append(
            f"| {row['case']} | "
            f"{row['expected_method']} | "
            f"{row['selected_method']} | "
            f"{row['correct_route']} |"
        )

    lines.extend(
        [
            "",
            "## Grounded vs Ungrounded",
            "",
            "| Case | Ungrounded bad plan | "
            "Grounded bad plan | Grounded good plan |",
            "|---|---:|---:|---:|",
        ]
    )

    for row in summary["grounded_vs_ungrounded"]:
        lines.append(
            f"| {row['case']} | "
            f"{row['ungrounded_bad_candidate_accepted']} | "
            f"{row['grounded_bad_candidate_accepted']} | "
            f"{row['grounded_good_candidate_accepted']} |"
        )

    lines.extend(
        [
            "",
            "## Self-Refine vs Reflexion",
            "",
            "| Case | Method | Success | LLM Calls | "
            "Total Tokens | Latency ms |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    for row in summary["self_refine_vs_reflexion"]:
        lines.append(
            f"| {row['case']} | "
            f"{row['method']} | "
            f"{row['success']} | "
            f"{row['llm_calls']} | "
            f"{row['total_tokens']} | "
            f"{row['latency_ms']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Final Checks",
            "",
        ]
    )

    for name, passed in summary["checks"].items():
        lines.append(
            f"- {name}: {'PASS' if passed else 'FAIL'}"
        )

    lines.extend(
        [
            "",
            "## Note",
            "",
            summary["production_note"],
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    summary = build_summary()

    json_path = (
        ARTIFACTS_DIR
        / "final_planning_evaluation.json"
    )
    markdown_path = (
        ARTIFACTS_DIR
        / "final_planning_evaluation.md"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    markdown_path.write_text(
        _markdown(summary),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary["checks"],
            indent=2,
        )
    )

    print(f"\nWrote: {json_path}")
    print(f"Wrote: {markdown_path}")


if __name__ == "__main__":
    main()