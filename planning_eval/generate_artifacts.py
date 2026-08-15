from __future__ import annotations

import json
from pathlib import Path

from planning.environment import Environment
from planning.reflexion import reflexion
from planning.self_refine import reflect_and_refine
from planning_eval.evaluation_suite import fixed_cases


class Response:
    def __init__(self, content: str):
        self.content = content


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @staticmethod
    def tokens(messages) -> int:
        return max(1, len("\n".join(str(x) for x in messages).split()))

    def invoke(self, messages, temperature=0.2):
        self.calls += 1
        self.input_tokens += self.tokens(messages)
        if not self.responses:
            raise RuntimeError("ScriptedLLM exhausted")
        content = str(self.responses.pop(0))
        self.output_tokens += max(1, len(content.split()))
        return Response(content)


def env_for(case):
    return Environment(
        shipment_id=case.shipment_id,
        employee_id=case.employee_id,
        snapshot_provider=lambda snapshot=case.snapshot: snapshot,
    )


def run():
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    summary = []

    for case in fixed_cases():
        env = env_for(case)

        # Grounded vs ungrounded evidence.
        grounded_bad = env.evaluate(case.bad_candidate)
        grounded_good = env.evaluate(case.good_candidate)
        ungrounded_accepts_bad = True  # Deliberate baseline: no DB/MCP evidence.
        grounding_trace = {
            "case": case.name,
            "candidate": case.bad_candidate,
            "ungrounded": {"accepted": ungrounded_accepts_bad, "reason": "No external state was checked."},
            "grounded": {
                "accepted": grounded_bad.success,
                "score": grounded_bad.score,
                "issues": grounded_bad.details,
                "evidence_keys": sorted(grounded_bad.evidence.keys()),
            },
        }
        (out / f"{case.name}_grounding.json").write_text(json.dumps(grounding_trace, indent=2), encoding="utf-8")

        # Self-Refine: bad draft -> independent critique -> revision -> grounded post-check.
        acting = ScriptedLLM([case.good_candidate])
        critic = ScriptedLLM(["The draft contains an unsafe state-changing action or misses required authority escalation."])
        result = reflect_and_refine(
            case.task,
            case.bad_candidate,
            acting,
            critic_llm=critic,
            environment=env,
        )
        sr_trace = {
            "method": "Self-Refine (grounded)",
            "case": case.name,
            "draft": result.draft,
            "critique": result.critique,
            "grounded_issues": result.grounded_issues,
            "revised": result.revised,
            "revision_validation": {
                "success": result.revision_feedback.success if result.revision_feedback else None,
                "score": result.revision_feedback.score if result.revision_feedback else None,
                "details": result.revision_feedback.details if result.revision_feedback else [],
            },
            "independent_critic": True,
            "llm_calls": acting.calls + critic.calls,
            "total_tokens": acting.input_tokens + acting.output_tokens + critic.input_tokens + critic.output_tokens,
        }
        (out / f"{case.name}_self_refine.json").write_text(json.dumps(sr_trace, indent=2), encoding="utf-8")
        summary.append({
            "case": case.name,
            "method": "Self-Refine (grounded)",
            "success": bool(result.revision_feedback and result.revision_feedback.success),
            "llm_calls": acting.calls + critic.calls,
            "total_tokens": acting.input_tokens + acting.output_tokens + critic.input_tokens + critic.output_tokens,
        })

        # Reflexion: intentionally fail once, carry reflection, then succeed.
        acting = ScriptedLLM([case.bad_candidate, case.good_candidate])
        critic = ScriptedLLM(["I must verify employee authority and grounded shipment constraints before proposing state-changing actions."])
        rx = reflexion(case.task, acting, env, max_trials=2, memory_size=1, critic_llm=critic)
        rx_trace = {
            "method": "Reflexion (grounded)",
            "case": case.name,
            "success": rx.success,
            "trials": [
                {
                    "number": t.number,
                    "attempt": t.attempt,
                    "feedback": {
                        "success": t.feedback.success,
                        "score": t.feedback.score,
                        "details": t.feedback.details,
                    },
                    "reflection": t.reflection,
                }
                for t in rx.trials
            ],
            "episodic_memory_after_run": rx.memory,
            "llm_calls": acting.calls + critic.calls,
            "total_tokens": acting.input_tokens + acting.output_tokens + critic.input_tokens + critic.output_tokens,
        }
        (out / f"{case.name}_reflexion.json").write_text(json.dumps(rx_trace, indent=2), encoding="utf-8")
        summary.append({
            "case": case.name,
            "method": "Reflexion (grounded)",
            "success": rx.success,
            "llm_calls": acting.calls + critic.calls,
            "total_tokens": acting.input_tokens + acting.output_tokens + critic.input_tokens + critic.output_tokens,
        })

    (out / "person3_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
