from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from planning.environment import Environment
from planning.algorithms.reflexion import reflexion
from planning.algorithms.self_refine import reflect_and_refine
from planning_eval.metrics import RunMetrics
from planning_eval.evaluation_suite import fixed_cases


@dataclass
class Response:
    content: str


class ScriptedLLM:
    """Deterministic LLM double for reproducible evaluation without API keys."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @staticmethod
    def _tokens(messages) -> int:
        text = "\n".join(str(x) for x in messages)
        return max(1, len(text.split()))

    def invoke(self, messages, temperature=0.2):
        self.calls += 1
        self.input_tokens += self._tokens(messages)
        if not self.responses:
            raise RuntimeError("ScriptedLLM exhausted")
        value = self.responses.pop(0)
        self.output_tokens += max(1, len(str(value).split()))
        return Response(str(value))


def _env(case):
    return Environment(
        shipment_id=case.shipment_id,
        employee_id=case.employee_id,
        snapshot_provider=lambda snapshot=case.snapshot: snapshot,
    )


def _self_refine_responses(case):
    return [
        "The candidate contains an unauthorized action or misses required escalation based on the grounded evidence.",
        case.good_candidate,
    ]


def _reflexion_responses(case):
    return [
        case.bad_candidate,
        "I must verify employee authority and grounded shipment constraints before proposing state-changing actions.",
        case.good_candidate,
    ]


def run_case(case):
    rows = []

    llm = ScriptedLLM([case.good_candidate])
    critic_llm = ScriptedLLM([
        "The candidate contains an unauthorized action or misses required escalation based on grounded evidence."
    ])
    start = time.perf_counter()
    result = reflect_and_refine(
        case.task,
        case.bad_candidate,
        llm,
        critic_llm=critic_llm,
        environment=_env(case),
    )
    latency = (time.perf_counter() - start) * 1000
    rows.append(RunMetrics(
        method="Self-Refine (grounded)",
        success=bool(result.revision_feedback and result.revision_feedback.success),
        llm_calls=llm.calls + critic_llm.calls,
        input_tokens=llm.input_tokens + critic_llm.input_tokens,
        output_tokens=llm.output_tokens + critic_llm.output_tokens,
        latency_ms=latency,
    ).as_dict())

    llm = ScriptedLLM([case.bad_candidate, case.good_candidate])
    critic_llm = ScriptedLLM([
        "I must verify employee authority and grounded shipment constraints before proposing state-changing actions."
    ])
    start = time.perf_counter()
    result = reflexion(
        case.task,
        llm,
        _env(case),
        max_trials=2,
        memory_size=1,
        critic_llm=critic_llm,
    )
    latency = (time.perf_counter() - start) * 1000
    rows.append(RunMetrics(
        method="Reflexion (grounded)",
        success=result.success,
        llm_calls=llm.calls + critic_llm.calls,
        input_tokens=llm.input_tokens + critic_llm.input_tokens,
        output_tokens=llm.output_tokens + critic_llm.output_tokens,
        latency_ms=latency,
    ).as_dict())

    return rows


def main():
    all_rows = []
    for case in fixed_cases():
        for row in run_case(case):
            row["case"] = case.name
            # Illustrative local cost model; replace rates with the team's provider rates.
            row["estimated_cost_usd"] = round(
                (row["input_tokens"] / 1_000_000) * 0.15
                + (row["output_tokens"] / 1_000_000) * 0.60,
                6,
            )
            all_rows.append(row)

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    path = out / "person3_self_correction_metrics.json"
    path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    print(json.dumps(all_rows, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
