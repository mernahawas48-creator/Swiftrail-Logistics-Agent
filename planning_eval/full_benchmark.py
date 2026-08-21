from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from planning.algorithms.decomposition import (
    SwiftrailGeneratedPlan,
    decompose_blocked_shipment,
    execute_plan_swiftrail,
)
from planning.algorithms.dynamic_decomposition import (
    DynamicSwiftrailDecision,
    dynamic_decompose_blocked_shipment,
)
from planning.algorithms.environment import Environment, RandomEnvironment
from planning.algorithms.lats import LATSActionBatch, ValueEstimate, lats
from planning.algorithms.plan_and_solve import plan_and_solve
from planning.algorithms.reflexion import reflexion
from planning.algorithms.self_refine import reflect_and_refine
from planning.algorithms.tree_of_thoughts import (
    ThoughtCandidates,
    ThoughtEvaluation,
    tree_of_thoughts,
)
from planning.execution_adapter import SubtaskExecutionAdapter
from planning.swiftrail_subtask import SubtaskKind
from planning_eval.benchmark_cases import (
    BenchmarkCase,
    above_authority_rate_case,
    severe_hold_sales_rep_case,
    stable_minor_hold_case,
)

ARTIFACTS_DIR = Path("artifacts")
DEMO_PATH = Path("demo") / "planning_demo_transcript.md"

# Same explicit local cost model already used by the repo's self-correction
# evaluator. These are benchmark accounting rates, not a claim about current
# provider billing.
INPUT_USD_PER_MILLION = 0.15
OUTPUT_USD_PER_MILLION = 0.60


@dataclass
class TextResponse:
    content: str


@dataclass
class BenchmarkRow:
    group: str
    case: str
    method: str
    success: bool
    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    tool_calls: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _StructuredRunner:
    def __init__(self, owner: BenchmarkLLM, schema):
        self.owner = owner
        self.schema = schema

    def invoke(self, messages, temperature=0.2):
        return self.owner.structured(self.schema, messages, temperature)


class BenchmarkLLM:
    """Deterministic LLM double that runs the real planning loops.

    It records every prompt/structured-output call and uses a fixed token proxy
    so the benchmark is reproducible without an API key. The algorithm code is
    real; only model responses are scripted.
    """

    def __init__(
        self,
        case: BenchmarkCase,
        *,
        ps_output: str | None = None,
        dynamic_stop_early: bool = False,
        self_refine_revision: str | None = None,
        reflexion_attempts: list[str] | None = None,
    ):
        self.case = case
        self.ps_output = ps_output or case.good_candidate
        self.dynamic_stop_early = dynamic_stop_early
        self.self_refine_revision = self_refine_revision or case.good_candidate
        self.reflexion_attempts = list(reflexion_attempts or [])
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.trace: list[dict[str, Any]] = []
        self.dynamic_stage = 0

    @staticmethod
    def _message_text(messages) -> str:
        return "\n".join(str(item) for item in messages)

    @staticmethod
    def _tokens(text: str) -> int:
        # Stable local proxy used only for the offline benchmark.
        return max(1, len(re.findall(r"\w+|[^\w\s]", text)))

    def _record(self, kind: str, messages, output: Any) -> None:
        prompt = self._message_text(messages)
        if hasattr(output, "model_dump_json"):
            output_text = output.model_dump_json()
        else:
            output_text = str(output)
        self.calls += 1
        self.input_tokens += self._tokens(prompt)
        self.output_tokens += self._tokens(output_text)
        self.trace.append(
            {
                "call": self.calls,
                "kind": kind,
                "input_tokens": self._tokens(prompt),
                "output_tokens": self._tokens(output_text),
            }
        )

    def with_structured_output(self, schema, method="json_schema"):
        del method
        return _StructuredRunner(self, schema)

    def _static_plan(self) -> SwiftrailGeneratedPlan:
        tasks = [
            {
                "id": "t1",
                "instruction": "Fetch shipment status.",
                "depends_on": [],
                "kind": "tool_call",
                "tool_name": "fetch_shipment",
                "reasoning_role": None,
            },
            {
                "id": "t2",
                "instruction": "Fetch customer credit state.",
                "depends_on": [],
                "kind": "tool_call",
                "tool_name": "fetch_customer",
                "reasoning_role": None,
            },
            {
                "id": "t3",
                "instruction": "Fetch customer invoices.",
                "depends_on": [],
                "kind": "tool_call",
                "tool_name": "fetch_invoices",
                "reasoning_role": None,
            },
            {
                "id": "t4",
                "instruction": "Fetch customer credit holds.",
                "depends_on": [],
                "kind": "tool_call",
                "tool_name": "fetch_credit_holds",
                "reasoning_role": None,
            },
            {
                "id": "t5",
                "instruction": "Fetch the shipment rate exception.",
                "depends_on": [],
                "kind": "tool_call",
                "tool_name": "fetch_rate_exception",
                "reasoning_role": None,
            },
            {
                "id": "t6",
                "instruction": "Summarize confirmed blockers and authority constraints.",
                "depends_on": ["t1", "t2", "t3", "t4", "t5"],
                "kind": "reasoning",
                "tool_name": None,
                "reasoning_role": "linear",
            },
            {
                "id": "t7",
                "instruction": "Compare multiple safe resolution sequences and trade-offs.",
                "depends_on": ["t6"],
                "kind": "reasoning",
                "tool_name": None,
                "reasoning_role": "branching",
            },
            {
                "id": "t8",
                "instruction": "Choose the final safe executable or escalation plan.",
                "depends_on": ["t7"],
                "kind": "reasoning",
                "tool_name": None,
                "reasoning_role": "final",
            },
        ]
        return SwiftrailGeneratedPlan.model_validate(
            {"goal": self.case.task, "tasks": tasks}
        )

    def _dynamic_decision(self, prompt: str) -> DynamicSwiftrailDecision:
        del prompt
        stage = self.dynamic_stage
        self.dynamic_stage += 1

        if stage == 0:
            return DynamicSwiftrailDecision(
                done=False,
                kind=SubtaskKind.TOOL_CALL,
                tool_name="fetch_shipment",
                rationale="Start from the shipment state.",
            )
        if stage == 1:
            return DynamicSwiftrailDecision(
                done=False,
                kind=SubtaskKind.TOOL_CALL,
                tool_name="fetch_customer",
                rationale="Inspect customer credit state next.",
            )
        if stage == 2:
            return DynamicSwiftrailDecision(
                done=False,
                kind=SubtaskKind.TOOL_CALL,
                tool_name="fetch_credit_holds",
                rationale="Check hold severity before downstream action.",
            )
        if self.dynamic_stop_early:
            return DynamicSwiftrailDecision(
                done=True,
                kind=SubtaskKind.REASONING,
                instruction="Produce the safe resolution from the evidence gathered so far.",
                rationale="The adaptive planner believes it has enough evidence.",
            )
        if stage == 3:
            return DynamicSwiftrailDecision(
                done=False,
                kind=SubtaskKind.TOOL_CALL,
                tool_name="fetch_invoices",
                rationale="Check overdue invoices.",
            )
        if stage == 4:
            return DynamicSwiftrailDecision(
                done=False,
                kind=SubtaskKind.TOOL_CALL,
                tool_name="fetch_rate_exception",
                rationale="Check the current shipment rate exception.",
            )
        return DynamicSwiftrailDecision(
            done=True,
            kind=SubtaskKind.REASONING,
            instruction="Produce the final safe executable or escalation plan.",
            rationale="Required evidence has been gathered.",
        )

    def _candidate_pair(self, prompt: str) -> tuple[str, str]:
        good = self.case.good_candidate
        bad = self.case.bad_candidate

        # In the static-stability case, dynamic intentionally stops before two
        # required reads. Its generated plan therefore reflects only observed
        # evidence and is rejected by the grounded validator. This is the fixed
        # case that favors decomposition-first.
        if self.dynamic_stop_early and "Observed tool results" in prompt:
            good = (
                "ACTION: check_shipment\n"
                "ACTION: check_customer\n"
                "ACTION: check_credit_hold"
            )
        return bad, good

    def structured(self, schema, messages, temperature=0.2):
        del temperature
        prompt = self._message_text(messages)
        name = schema.__name__

        if name == "SwiftrailGeneratedPlan":
            result = self._static_plan()
        elif name == "DynamicSwiftrailDecision":
            result = self._dynamic_decision(prompt)
        elif name == "ThoughtCandidates":
            bad, good = self._candidate_pair(prompt)
            result = ThoughtCandidates(candidates=[bad, good])
        elif name == "ThoughtEvaluation":
            human = messages[-1][1] if messages and isinstance(messages[-1], tuple) else prompt
            _, good = self._candidate_pair(human)
            marker = "Candidate reasoning path:\n"
            candidate = human.split(marker, 1)[-1].split("\n\nScore this candidate", 1)[0].strip()
            if candidate == good.strip():
                result = ThoughtEvaluation(score=0.92, rationale="Best safe branch.")
            else:
                result = ThoughtEvaluation(score=0.25, rationale="Unsafe or incomplete branch.")
        elif name == "LATSActionBatch":
            bad, good = self._candidate_pair(prompt)
            result = LATSActionBatch.model_validate(
                {
                    "actions": [
                        {"action": "direct_resolution", "state": bad},
                        {"action": "safe_resolution", "state": good},
                    ]
                }
            )
        elif name == "ValueEstimate":
            match = re.search(r"Grounded environment score:\s*([0-9.]+)", prompt)
            score = float(match.group(1)) if match else 0.5
            result = ValueEstimate(score=max(0.0, min(1.0, score)))
        else:
            raise RuntimeError(f"Unsupported structured schema in benchmark: {name}")

        self._record(f"structured:{name}", messages, result)
        return result

    def invoke(self, messages, temperature=0.2):
        del temperature
        prompt = self._message_text(messages)

        if "Plan-and-Solve reasoner" in prompt:
            text = self.ps_output
        elif "branch-level LATS reflection" in prompt:
            text = (
                "I must follow the grounded authority failure and choose the "
                "safe escalation branch next."
            )
        elif "acting agent in a Reflexion loop" in prompt:
            if not self.reflexion_attempts:
                raise RuntimeError("Benchmark Reflexion responses exhausted")
            text = self.reflexion_attempts.pop(0)
        elif "concise first-person Reflexion memory" in prompt:
            text = (
                "I must carry the grounded authority and missing-observation "
                "failures into the next full trial before proposing any write."
            )
        elif "You are a separate critic" in prompt:
            text = (
                "The draft violates or incompletely covers the grounded Swiftrail "
                "authority and observation requirements."
            )
        elif "Revise a deliverable" in prompt:
            text = self.self_refine_revision
        else:
            # Only the explicit algorithm prompts above should reach invoke().
            text = self.case.good_candidate

        response = TextResponse(text)
        self._record("text", messages, text)
        return response


class FakeAgent:
    def __init__(self, case: BenchmarkCase, delay_s: float = 0.001):
        self.case = case
        self.delay_s = delay_s
        self.calls: list[str] = []

    async def call_tool(self, tool_name: str, args: dict) -> dict:
        del args
        self.calls.append(tool_name)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return self._payload(tool_name)

    def decode_tool_result(self, result):
        return result

    def _payload(self, tool_name: str) -> dict:
        snap = self.case.snapshot
        if tool_name == "get_shipment_status":
            data = {"shipment": snap.shipment}
        elif tool_name == "search_customer":
            data = {"customer": snap.customer}
        elif tool_name == "list_customer_invoices":
            data = {"customer_id": self.case.customer_id, "invoices": snap.invoices}
        elif tool_name == "list_customer_credit_holds":
            data = {
                "customer_id": self.case.customer_id,
                "holds": snap.holds,
                "active_holds": [h for h in snap.holds if h.get("status") == "active"],
            }
        elif tool_name == "get_shipment_rate_exception":
            data = {
                "shipment_id": self.case.shipment_id,
                "rate_exception": snap.rate_exceptions[-1] if snap.rate_exceptions else None,
            }
        else:
            raise AssertionError(f"Unexpected benchmark tool: {tool_name}")
        return {"success": True, "code": "BENCHMARK_OK", "message": "ok", "data": data}


class ToolCaller:
    def __init__(self, agent: FakeAgent):
        self.agent = agent

    async def __call__(self, tool_name: str, args: dict) -> dict:
        return await self.agent.call_tool(tool_name, {"request": args})


def _environment(case: BenchmarkCase) -> Environment:
    return Environment(
        shipment_id=case.shipment_id,
        employee_id=case.employee_id,
        snapshot_provider=lambda snapshot=case.snapshot: snapshot,
    )


def _cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        input_tokens / 1_000_000 * INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION,
        6,
    )


def _row(
    *,
    group: str,
    case: BenchmarkCase,
    method: str,
    success: bool,
    llm: BenchmarkLLM,
    latency_ms: float,
    tool_calls: int = 0,
    note: str = "",
) -> BenchmarkRow:
    return BenchmarkRow(
        group=group,
        case=case.name,
        method=method,
        success=success,
        llm_calls=llm.calls,
        input_tokens=llm.input_tokens,
        output_tokens=llm.output_tokens,
        total_tokens=llm.input_tokens + llm.output_tokens,
        latency_ms=round(latency_ms, 3),
        estimated_cost_usd=_cost(llm.input_tokens, llm.output_tokens),
        tool_calls=tool_calls,
        note=note,
    )


async def _run_static(case: BenchmarkCase) -> tuple[BenchmarkRow, dict[str, Any]]:
    llm = BenchmarkLLM(case, ps_output=case.good_candidate)
    agent = FakeAgent(case)
    env = _environment(case)
    started = time.perf_counter()
    plan = decompose_blocked_shipment(case.shipment_id, case.customer_id, llm)
    adapter = SubtaskExecutionAdapter(
        agent=agent,
        session_id="bench-session",
        llm=llm,
        environment=env,
    )
    outputs, _ = await execute_plan_swiftrail(plan, adapter)
    final = outputs[plan.terminal_tasks()[0]]
    feedback = env.evaluate(final)
    latency = (time.perf_counter() - started) * 1000
    routes = [
        item.routed.method.value
        for item in adapter.history
        if item.routed is not None
    ]
    row = _row(
        group="decomposition",
        case=case,
        method="Decomposition-first",
        success=feedback.success,
        llm=llm,
        latency_ms=latency,
        tool_calls=len(agent.calls),
        note="Whole DAG committed up front; independent reads execute in one batch.",
    )
    return row, {
        "final_output": final,
        "routes": routes,
        "tool_sequence": agent.calls,
        "feedback": feedback.model_dump(),
        "execution_batches": plan.execution_batches(),
    }


async def _run_dynamic(
    case: BenchmarkCase,
    *,
    stop_early: bool,
) -> tuple[BenchmarkRow, dict[str, Any]]:
    llm = BenchmarkLLM(case, dynamic_stop_early=stop_early)
    agent = FakeAgent(case)
    env = _environment(case)
    started = time.perf_counter()
    steps = await dynamic_decompose_blocked_shipment(
        shipment_id=case.shipment_id,
        customer_id=case.customer_id,
        session_id="bench-session",
        llm=llm,
        call_tool=ToolCaller(agent),
        environment=env,
    )
    final = steps[-1].output if steps else ""
    feedback = env.evaluate(final)
    latency = (time.perf_counter() - started) * 1000
    row = _row(
        group="decomposition",
        case=case,
        method="Dynamic decomposition",
        success=feedback.success,
        llm=llm,
        latency_ms=latency,
        tool_calls=len(agent.calls),
        note=(
            "Interleaved next-step decisions; grounded severe-hold rule may force early escalation."
        ),
    )
    return row, {
        "final_output": final,
        "tool_sequence": agent.calls,
        "forced_steps": [s.step for s in steps if s.forced],
        "steps": [
            {
                "step": s.step,
                "kind": s.kind.value,
                "tool_name": s.tool_name,
                "instruction": s.instruction,
                "forced": s.forced,
            }
            for s in steps
        ],
        "feedback": feedback.model_dump(),
    }


def _run_ps(case: BenchmarkCase, *, safe: bool) -> tuple[BenchmarkRow, dict[str, Any]]:
    llm = BenchmarkLLM(case, ps_output=case.good_candidate if safe else case.bad_candidate)
    env = _environment(case)
    started = time.perf_counter()
    output = plan_and_solve(case.task, llm)
    latency = (time.perf_counter() - started) * 1000
    feedback = env.evaluate(output)
    return _row(
        group="planning",
        case=case,
        method="Plan-and-Solve",
        success=feedback.success,
        llm=llm,
        latency_ms=latency,
        note="Single explicit plan/solve pass; no search.",
    ), {"output": output, "feedback": feedback.model_dump()}


def _run_tot(case: BenchmarkCase) -> tuple[BenchmarkRow, dict[str, Any]]:
    llm = BenchmarkLLM(case)
    env = _environment(case)
    started = time.perf_counter()
    thoughts = tree_of_thoughts(case.task, llm, depth=2, beam_width=2)
    best = max(thoughts, key=lambda thought: thought.score)
    latency = (time.perf_counter() - started) * 1000
    feedback = env.evaluate(best.state)
    return _row(
        group="planning",
        case=case,
        method="Tree of Thoughts",
        success=feedback.success,
        llm=llm,
        latency_ms=latency,
        note="Two-level beam search; model scores alternatives and prunes weaker branches.",
    ), {
        "output": best.state,
        "score": best.score,
        "feedback": feedback.model_dump(),
    }


def _run_lats(case: BenchmarkCase, *, grounded: bool) -> tuple[BenchmarkRow, dict[str, Any]]:
    llm = BenchmarkLLM(case)
    real_env = _environment(case)
    search_env = (
        real_env
        if grounded
        else RandomEnvironment(success_threshold=0.0, rng=random.Random(7))
    )
    started = time.perf_counter()
    result = lats(
        case.task,
        llm,
        search_env,
        iterations=2,
        n_actions=2,
    )
    latency = (time.perf_counter() - started) * 1000
    # Quality is always judged by the real source of truth, even when the
    # search itself deliberately uses the ungrounded toolkit environment.
    grounded_feedback = real_env.evaluate(result.output)
    label = "LATS grounded" if grounded else "LATS ungrounded"
    return _row(
        group="planning",
        case=case,
        method=label,
        success=grounded_feedback.success,
        llm=llm,
        latency_ms=latency,
        note=(
            "MCTS with real validator feedback."
            if grounded
            else "Toolkit-style randomized feedback; final quality rechecked by real validator."
        ),
    ), {
        "output": result.output,
        "search_reported_success": result.success,
        "search_score": result.best_score,
        "grounded_feedback": grounded_feedback.model_dump(),
    }


def _run_self_refine(
    case: BenchmarkCase,
    *,
    revision: str,
) -> tuple[BenchmarkRow, dict[str, Any]]:
    llm = BenchmarkLLM(case, self_refine_revision=revision)
    env = _environment(case)
    started = time.perf_counter()
    result = reflect_and_refine(
        case.task,
        case.bad_candidate,
        llm,
        critic_llm=llm,
        environment=env,
    )
    latency = (time.perf_counter() - started) * 1000
    success = bool(result.revision_feedback and result.revision_feedback.success)
    return _row(
        group="self_correction",
        case=case,
        method="Self-Refine",
        success=success,
        llm=llm,
        latency_ms=latency,
        note="One critique and one revision only.",
    ), {
        "draft": result.draft,
        "critique": result.critique,
        "revised": result.revised,
        "grounded_issues": result.grounded_issues,
        "revision_feedback": (
            result.revision_feedback.model_dump() if result.revision_feedback else None
        ),
    }


def _run_reflexion(case: BenchmarkCase) -> tuple[BenchmarkRow, dict[str, Any]]:
    llm = BenchmarkLLM(
        case,
        reflexion_attempts=[case.bad_candidate, case.good_candidate],
    )
    env = _environment(case)
    started = time.perf_counter()
    result = reflexion(
        case.task,
        llm,
        env,
        max_trials=2,
        memory_size=1,
        critic_llm=llm,
    )
    latency = (time.perf_counter() - started) * 1000
    return _row(
        group="self_correction",
        case=case,
        method="Reflexion",
        success=result.success,
        llm=llm,
        latency_ms=latency,
        note="Retries the whole task and carries one grounded reflection into trial 2.",
    ), {
        "output": result.output,
        "memory": result.memory,
        "trials": [
            {
                "number": trial.number,
                "attempt": trial.attempt,
                "success": trial.feedback.success,
                "score": trial.feedback.score,
                "issues": trial.feedback.details,
                "reflection": trial.reflection,
            }
            for trial in result.trials
        ],
    }


def _aggregate(rows: list[BenchmarkRow]) -> list[dict[str, Any]]:
    methods = []
    seen = set()
    for row in rows:
        if row.method not in seen:
            methods.append(row.method)
            seen.add(row.method)

    result = []
    for method in methods:
        selected = [r for r in rows if r.method == method]
        result.append(
            {
                "method": method,
                "success": f"{sum(r.success for r in selected)}/{len(selected)}",
                "success_rate": round(sum(r.success for r in selected) / len(selected), 3),
                "avg_llm_calls": round(mean(r.llm_calls for r in selected), 2),
                "avg_tokens": round(mean(r.total_tokens for r in selected), 1),
                "avg_latency_ms": round(mean(r.latency_ms for r in selected), 3),
                "avg_estimated_cost_usd": round(
                    mean(r.estimated_cost_usd for r in selected), 6
                ),
                "avg_tool_calls": round(mean(r.tool_calls for r in selected), 2),
            }
        )
    return result


def _markdown_table(aggregate: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Success | Avg. LLM calls | Avg. tokens | Avg. latency | Est. cost/run | Avg. tool calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['method']} | {row['success']} | {row['avg_llm_calls']} | "
            f"{row['avg_tokens']} | {row['avg_latency_ms']:.3f} ms | "
            f"${row['avg_estimated_cost_usd']:.6f} | {row['avg_tool_calls']} |"
        )
    return "\n".join(lines)


def _detail_markdown(rows: list[BenchmarkRow]) -> str:
    lines = [
        "| Group | Case | Method | Success | LLM calls | Tokens | Latency | Cost | Tool calls |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.group} | {row.case} | {row.method} | {row.success} | "
            f"{row.llm_calls} | {row.total_tokens} | {row.latency_ms:.3f} ms | "
            f"${row.estimated_cost_usd:.6f} | {row.tool_calls} |"
        )
    return "\n".join(lines)


def _build_demo(details: dict[str, Any], aggregate: list[dict[str, Any]]) -> str:
    static_stable = details["decomposition"]["stable_minor_hold"]["Decomposition-first"]
    dynamic_stable = details["decomposition"]["stable_minor_hold"]["Dynamic decomposition"]
    static_severe = details["decomposition"]["severe_hold_sales_rep"]["Decomposition-first"]
    dynamic_severe = details["decomposition"]["severe_hold_sales_rep"]["Dynamic decomposition"]
    planning_rate = details["planning"]["above_authority_rate"]
    self_hard = details["self_correction"]["severe_hold_sales_rep"]

    return f"""# Planning Demo Transcript

This transcript is generated from `python -m planning_eval.full_benchmark` using the fixed Swiftrail seed-data-shaped cases. It is deterministic and requires no API key.

## 1. Same request type: decomposition-first vs dynamic

### Stable blocked shipment — decomposition-first

Request: Review blocked shipment 2/customer 2 and resolve financial blockers safely.

Execution batches:

```text
{json.dumps(static_stable['execution_batches'])}
```

Tool sequence:

```text
{static_stable['tool_sequence']}
```

Reasoning routes observed in the DAG:

```text
{static_stable['routes']}
```

Grounded result: `{static_stable['feedback']['success']}`.

### Stable blocked shipment — dynamic

The dynamic planner stopped after:

```text
{dynamic_stable['tool_sequence']}
```

Its final candidate was rejected by the grounded validator because required evidence was still missing:

```text
{dynamic_stable['feedback']['details']}
```

This is the fixed case that favors decomposition-first: when the required reconnaissance is known and stable, committing the complete DAG avoids premature stopping.

### Severe hold — real divergence

Request: Resolve blocked shipment 3/customer 3 for a sales representative.

Decomposition-first committed this complete read sequence before seeing any result:

```text
{static_severe['tool_sequence']}
```

Dynamic decomposition instead moved the credit-hold lookup earlier and observed:

```text
{dynamic_severe['tool_sequence']}
```

Forced step(s): `{dynamic_severe['forced_steps']}`.

At that point the severe active hold triggers the deterministic safety branch, so the next step becomes finance-manager escalation instead of blindly continuing the remaining up-front sequence. Grounded result: `{dynamic_severe['feedback']['success']}`.

## 2. Plan-and-Solve, Tree of Thoughts, and LATS

Case: 25% pending rate exception on shipment 5 for a `sales_rep`.

### Plan-and-Solve

```text
{planning_rate['Plan-and-Solve']['output']}
```

Grounded success: `{planning_rate['Plan-and-Solve']['feedback']['success']}`.

### Tree of Thoughts

The beam search generates competing branches, evaluates them, prunes the unsafe direct-approval path, and returns:

```text
{planning_rate['Tree of Thoughts']['output']}
```

Grounded success: `{planning_rate['Tree of Thoughts']['feedback']['success']}`.

### LATS — ungrounded environment

The randomized environment reports search success on the first branch, but the real validator re-checks the selected output:

```text
{planning_rate['LATS ungrounded']['output']}
```

Search reported success: `{planning_rate['LATS ungrounded']['search_reported_success']}`.
Real grounded success: `{planning_rate['LATS ungrounded']['grounded_feedback']['success']}`.

This is the required failure that an ungrounded evaluator misses.

### LATS — grounded environment

The real Swiftrail validator rejects the unauthorized direct approval, LATS records a branch reflection, explores the safer branch, and returns:

```text
{planning_rate['LATS grounded']['output']}
```

Grounded success: `{planning_rate['LATS grounded']['grounded_feedback']['success']}`.

## 3. Self-Refine vs Reflexion

Case: severe active credit hold for a sales representative.

### Self-Refine

Draft:

```text
{self_hard['Self-Refine']['draft']}
```

Critique:

```text
{self_hard['Self-Refine']['critique']}
```

Single revision:

```text
{self_hard['Self-Refine']['revised']}
```

Grounded success after the one allowed revision: `{self_hard['Self-Refine']['revision_feedback']['success']}`.

### Reflexion

Trial 1 success: `{self_hard['Reflexion']['trials'][0]['success']}`.

Stored reflection:

```text
{self_hard['Reflexion']['trials'][0]['reflection']}
```

Trial 2 receives that episodic reflection and produces:

```text
{self_hard['Reflexion']['trials'][1]['attempt']}
```

Trial 2 success: `{self_hard['Reflexion']['trials'][1]['success']}`.

This is the fixed case where one Self-Refine revision is insufficient but Reflexion succeeds by carrying a grounded lesson across trials.

## 4. Cost / quality summary

{_markdown_table(aggregate)}

The benchmark uses the actual repository algorithms with deterministic scripted model responses. Token counts use a fixed local token proxy, latency is measured locally, and cost uses the repository's explicit illustrative accounting rates ($0.15/M input tokens and $0.60/M output tokens). No production-provider billing is claimed.
"""


async def run_benchmark() -> tuple[list[BenchmarkRow], dict[str, Any]]:
    rows: list[BenchmarkRow] = []
    details: dict[str, Any] = {
        "decomposition": {},
        "planning": {},
        "self_correction": {},
    }

    stable = stable_minor_hold_case()
    severe = severe_hold_sales_rep_case()
    rate = above_authority_rate_case()

    # Decomposition-first vs dynamic on the same recurring request type.
    for case, stop_early in [(stable, True), (severe, False)]:
        static_row, static_detail = await _run_static(case)
        dynamic_row, dynamic_detail = await _run_dynamic(case, stop_early=stop_early)
        rows.extend([static_row, dynamic_row])
        details["decomposition"][case.name] = {
            static_row.method: static_detail,
            dynamic_row.method: dynamic_detail,
        }

    # Planning algorithms: a linear case and a genuine lookahead case.
    for case in [stable, rate]:
        case_details = {}
        ps_row, ps_detail = _run_ps(case, safe=(case.reasoning_shape == "linear"))
        tot_row, tot_detail = _run_tot(case)
        lu_row, lu_detail = _run_lats(case, grounded=False)
        lg_row, lg_detail = _run_lats(case, grounded=True)
        rows.extend([ps_row, tot_row, lu_row, lg_row])
        case_details[ps_row.method] = ps_detail
        case_details[tot_row.method] = tot_detail
        case_details[lu_row.method] = lu_detail
        case_details[lg_row.method] = lg_detail
        details["planning"][case.name] = case_details

    # Easy correction: one revision is enough.
    sr_easy, sr_easy_detail = _run_self_refine(rate, revision=rate.good_candidate)
    rf_easy, rf_easy_detail = _run_reflexion(rate)
    rows.extend([sr_easy, rf_easy])
    details["self_correction"][rate.name] = {
        sr_easy.method: sr_easy_detail,
        rf_easy.method: rf_easy_detail,
    }

    # Hard correction: the single Self-Refine revision is still incomplete;
    # Reflexion carries the grounded lesson into a second whole-task trial.
    still_bad = (
        "ACTION: check_customer\n"
        "ACTION: check_credit_hold\n"
        "ACTION: escalate role=finance_manager"
    )
    sr_hard, sr_hard_detail = _run_self_refine(severe, revision=still_bad)
    rf_hard, rf_hard_detail = _run_reflexion(severe)
    rows.extend([sr_hard, rf_hard])
    details["self_correction"][severe.name] = {
        sr_hard.method: sr_hard_detail,
        rf_hard.method: rf_hard_detail,
    }

    return rows, details


def _required_checks(rows: list[BenchmarkRow], details: dict[str, Any]) -> dict[str, bool]:
    by = {(row.case, row.method): row for row in rows}
    return {
        "static_favored_case_present": (
            by[("stable_minor_hold", "Decomposition-first")].success
            and not by[("stable_minor_hold", "Dynamic decomposition")].success
        ),
        "dynamic_divergence_case_present": bool(
            details["decomposition"]["severe_hold_sales_rep"]["Dynamic decomposition"]["forced_steps"]
        ),
        "linear_case_favors_ps": by[("stable_minor_hold", "Plan-and-Solve")].success,
        "lookahead_case_needs_search": (
            not by[("above_authority_rate", "Plan-and-Solve")].success
            and by[("above_authority_rate", "Tree of Thoughts")].success
        ),
        "ungrounded_lats_misses_failure": (
            not by[("above_authority_rate", "LATS ungrounded")].success
            and by[("above_authority_rate", "LATS grounded")].success
        ),
        "reflexion_cross_trial_case": (
            not by[("severe_hold_sales_rep", "Self-Refine")].success
            and by[("severe_hold_sales_rep", "Reflexion")].success
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed offline Swiftrail planning benchmark."
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Write artifacts but suppress the markdown table on stdout.",
    )
    args = parser.parse_args()

    rows, details = asyncio.run(run_benchmark())
    aggregate = _aggregate(rows)
    checks = _required_checks(rows, details)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    DEMO_PATH.parent.mkdir(exist_ok=True)

    payload = {
        "benchmark_mode": "deterministic_offline_real_algorithm_loops",
        "token_accounting": "fixed local token proxy",
        "cost_model": {
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
        },
        "rows": [row.as_dict() for row in rows],
        "aggregate": aggregate,
        "required_checks": checks,
        "details": details,
    }

    json_path = ARTIFACTS_DIR / "full_planning_benchmark.json"
    md_path = ARTIFACTS_DIR / "full_planning_benchmark.md"
    summary_path = ARTIFACTS_DIR / "full_planning_benchmark_summary.json"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    markdown = (
        "# Full Planning Cost / Quality Benchmark\n\n"
        "## Aggregate comparison\n\n"
        + _markdown_table(aggregate)
        + "\n\n## Per-case runs\n\n"
        + _detail_markdown(rows)
        + "\n\n## Required-case checks\n\n"
        + "\n".join(
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in checks.items()
        )
        + "\n\n"
        "This benchmark executes the repository's real decomposition, PS, ToT, "
        "LATS, Self-Refine, and Reflexion loops with deterministic scripted model "
        "responses and fixed Swiftrail seed-data-shaped snapshots. Token counts "
        "are a stable local proxy; production provider latency/billing can be "
        "measured separately without changing the fixed cases.\n"
    )
    md_path.write_text(markdown, encoding="utf-8")
    DEMO_PATH.write_text(_build_demo(details, aggregate), encoding="utf-8")

    print(json.dumps(checks, indent=2))
    if not args.json_only:
        print("\n" + _markdown_table(aggregate))
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {DEMO_PATH}")

    if not all(checks.values()):
        raise SystemExit("One or more required benchmark checks failed.")


if __name__ == "__main__":
    main()