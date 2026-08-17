from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_mistralai import ChatMistralAI

from planning.algorithms.decomposition import (
    decompose_blocked_shipment,
    execute_plan_swiftrail,
)
from planning.algorithms.dynamic_decomposition import (
    dynamic_decompose_blocked_shipment,
)
from planning.algorithms.environment import Environment, RandomEnvironment
from planning.algorithms.lats import lats
from planning.algorithms.plan_and_solve import plan_and_solve
from planning.algorithms.reflexion import reflexion
from planning.algorithms.self_refine import reflect_and_refine
from planning.algorithms.tree_of_thoughts import tree_of_thoughts
from planning.execution_adapter import SubtaskExecutionAdapter
from planning_eval.benchmark_cases import (
    BenchmarkCase,
    above_authority_rate_case,
    severe_hold_sales_rep_case,
    stable_minor_hold_case,
)
from planning_eval.full_benchmark import FakeAgent, ToolCaller


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"

# Current Standard-tier prices for the models used by this project.
# If another model is selected, pass --input-price and --output-price explicitly.
KNOWN_STANDARD_PRICES: dict[str, tuple[float, float]] = {
    "mistral-small-latest": (0.15, 0.60),
}


ACTION_OUTPUT_CONTRACT = """Swiftrail benchmark output contract:
- Express every concrete check, escalation, approval, hold release, or shipment release as a plain machine-readable line beginning with ACTION:.
- Put ACTION: at the start of its own line. Do not wrap ACTION or the action name in Markdown and do not indent or number the ACTION line.
- Use only these action names when applicable: check_shipment, check_customer, check_invoices, check_credit_hold, check_rate_exception, escalate, release_credit_hold, approve_rate_exception, release_shipment.
- Use exactly ACTION: escalate role=finance_manager when finance approval/escalation is required. Do not invent aliases such as escalate_rate_exception or escalate_credit_hold.
- If observed evidence is already supplied, include the corresponding check_* ACTION lines in the final plan, but never claim that a write action succeeded unless a tool result proves it.
- Prose may explain the plan, but all concrete actions must also appear in ACTION lines.
"""


AUTHORITATIVE_POLICY = """Authoritative Swiftrail policy for these fixed benchmark cases:
- The employee role in the supplied snapshot is authoritative.
- A sales_rep must not release a severe active credit hold. A severe active hold requires ACTION: escalate role=finance_manager before any release.
- A sales_rep must not approve a pending rate exception above 15 percent. It requires ACTION: escalate role=finance_manager.
- Do not release a shipment while the customer remains on credit hold with overdue invoices.
- For a blocked shipment without a severe-stop condition, inspect shipment, customer, invoices, credit hold, and rate exception before recommending a resolution.
- Never invent a successful MCP write or database state transition.
"""


def _benchmark_problem(case: BenchmarkCase) -> str:
    snapshot = json.dumps(asdict(case.snapshot), indent=2, default=str)
    return (
        f"Task:\n{case.task}\n\n"
        "The following snapshot is already-observed ground truth for this fixed "
        "benchmark request. Do not say the evidence is missing.\n"
        f"{snapshot}\n\n"
        f"{AUTHORITATIVE_POLICY}\n\n"
        f"{ACTION_OUTPUT_CONTRACT}"
    )


@dataclass
class LiveBenchmarkRow:
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
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CallCounter(BaseCallbackHandler):
    """Count actual chat-model invocations made by LangChain."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        del serialized, messages, kwargs
        self.calls += 1


class _StructuredCompatibleRunner:
    def __init__(self, owner: "TextCompatibleLLM", runner, schema_name: str):
        self._owner = owner
        self._runner = runner
        self._schema_name = schema_name

    def invoke(self, messages, *args, **kwargs):
        augmented = self._owner._augment_structured(
            self._schema_name,
            messages,
        )
        result = self._runner.invoke(augmented, *args, **kwargs)

        # The dynamic schema says a completed request must end in reasoning,
        # not in one last tool read. Some live models still set done=true on
        # a tool call; normalize that schema violation without changing the
        # chosen tool or inventing a new step.
        if self._schema_name == "DynamicSwiftrailDecision":
            kind = getattr(result, "kind", None)
            kind_value = getattr(kind, "value", kind)
            if getattr(result, "done", False) and kind_value == "tool_call":
                result.done = False

        return result


class TextCompatibleLLM:
    """Adapter that keeps live Mistral calls compatible with the project.

    It also supplies one benchmark-wide output contract. The contract does
    not contain the answer to any case; it only makes the machine-readable
    ACTION syntax explicit so the grounded validator measures decision
    quality instead of Markdown/formatting differences between providers.
    """

    def __init__(self, model: ChatMistralAI):
        self._model = model

    @staticmethod
    def _message_text(messages) -> str:
        if isinstance(messages, list):
            return "\n".join(str(item) for item in messages)
        return str(messages)

    @staticmethod
    def _append_instruction(messages, instruction: str):
        if not isinstance(messages, list):
            return messages
        return [*messages, ("human", instruction)]

    def _augment_text(self, messages):
        prompt = self._message_text(messages)
        needs_contract = any(
            marker in prompt
            for marker in (
                "Plan-and-Solve reasoner",
                "acting agent in a Reflexion loop",
                "Revise a deliverable",
            )
        )
        if needs_contract:
            return self._append_instruction(
                messages,
                ACTION_OUTPUT_CONTRACT,
            )
        return messages

    def _augment_structured(self, schema_name: str, messages):
        if schema_name in {"ThoughtCandidates", "LATSActionBatch"}:
            return self._append_instruction(
                messages,
                ACTION_OUTPUT_CONTRACT,
            )

        if schema_name == "DynamicSwiftrailDecision":
            return self._append_instruction(
                messages,
                "Dynamic-step protocol: done=true is valid only when "
                "kind='reasoning'. Never set done=true on a tool_call, and "
                "do not deliberately repeat a tool already present in the "
                "Observed tool results.",
            )

        return messages

    def invoke(self, messages, *args, **kwargs):
        response = self._model.invoke(
            self._augment_text(messages),
            *args,
            **kwargs,
        )

        if isinstance(response.content, str):
            return response

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(
                "Mistral returned a non-text response that could not be "
                "normalized for the planning algorithms."
            )

        return SimpleNamespace(
            content=text,
            usage_metadata=getattr(response, "usage_metadata", None),
            response_metadata=getattr(response, "response_metadata", None),
        )

    def with_structured_output(self, schema, *args, **kwargs):
        runner = self._model.with_structured_output(
            schema,
            *args,
            **kwargs,
        )
        return _StructuredCompatibleRunner(
            self,
            runner,
            schema.__name__,
        )


@dataclass
class LiveTracker:
    llm: TextCompatibleLLM
    usage: UsageMetadataCallbackHandler
    counter: CallCounter


def _make_tracker(
    *,
    model_name: str,
    api_key: str,
    max_retries: int,
) -> LiveTracker:
    usage = UsageMetadataCallbackHandler()
    counter = CallCounter()

    model = ChatMistralAI(
        model=model_name,
        api_key=api_key,
        max_retries=max_retries,
        callbacks=[usage, counter],
    )

    return LiveTracker(
        llm=TextCompatibleLLM(model),
        usage=usage,
        counter=counter,
    )


def _token_totals(
    usage: UsageMetadataCallbackHandler,
) -> tuple[int, int, int]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for item in usage.usage_metadata.values():
        input_tokens += int(item.get("input_tokens", 0) or 0)
        output_tokens += int(item.get("output_tokens", 0) or 0)
        total_tokens += int(item.get("total_tokens", 0) or 0)

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return input_tokens, output_tokens, total_tokens


def _environment(case: BenchmarkCase) -> Environment:
    return Environment(
        shipment_id=case.shipment_id,
        employee_id=case.employee_id,
        snapshot_provider=lambda snapshot=case.snapshot: snapshot,
    )


def _cost(
    input_tokens: int,
    output_tokens: int,
    *,
    input_price: float,
    output_price: float,
) -> float:
    return round(
        (input_tokens / 1_000_000) * input_price
        + (output_tokens / 1_000_000) * output_price,
        8,
    )


def _row(
    *,
    group: str,
    case: BenchmarkCase,
    method: str,
    success: bool,
    tracker: LiveTracker,
    latency_ms: float,
    input_price: float,
    output_price: float,
    tool_calls: int = 0,
    error: str | None = None,
) -> LiveBenchmarkRow:
    input_tokens, output_tokens, total_tokens = _token_totals(tracker.usage)

    return LiveBenchmarkRow(
        group=group,
        case=case.name,
        method=method,
        success=success,
        llm_calls=tracker.counter.calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=round(latency_ms, 3),
        estimated_cost_usd=_cost(
            input_tokens,
            output_tokens,
            input_price=input_price,
            output_price=output_price,
        ),
        tool_calls=tool_calls,
        error=error,
    )


async def _run_static(
    case: BenchmarkCase,
    *,
    tracker: LiveTracker,
    input_price: float,
    output_price: float,
) -> tuple[LiveBenchmarkRow, dict[str, Any]]:
    agent = FakeAgent(case, delay_s=0.0)
    env = _environment(case)
    started = time.perf_counter()

    try:
        plan = decompose_blocked_shipment(
            case.shipment_id,
            case.customer_id,
            tracker.llm,
        )

        adapter = SubtaskExecutionAdapter(
            agent=agent,
            session_id="live-benchmark-session",
            llm=tracker.llm,
            environment=env,
        )

        outputs, _ = await execute_plan_swiftrail(plan, adapter)
        final = outputs[plan.terminal_tasks()[0]]
        feedback = env.evaluate(final)
        success = feedback.success

        detail = {
            "final_output": final,
            "feedback": feedback.model_dump(),
            "tool_sequence": agent.calls,
            "execution_batches": plan.execution_batches(),
            "routes": [
                item.routed.method.value
                for item in adapter.history
                if item.routed is not None
            ],
        }
        error = None

    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
        detail = {
            "error": error,
            "tool_sequence": agent.calls,
        }

    latency_ms = (time.perf_counter() - started) * 1000

    return _row(
        group="decomposition",
        case=case,
        method="Decomposition-first",
        success=success,
        tracker=tracker,
        latency_ms=latency_ms,
        input_price=input_price,
        output_price=output_price,
        tool_calls=len(agent.calls),
        error=error,
    ), detail


async def _run_dynamic(
    case: BenchmarkCase,
    *,
    tracker: LiveTracker,
    input_price: float,
    output_price: float,
) -> tuple[LiveBenchmarkRow, dict[str, Any]]:
    agent = FakeAgent(case, delay_s=0.0)
    env = _environment(case)
    started = time.perf_counter()

    try:
        steps = await dynamic_decompose_blocked_shipment(
            shipment_id=case.shipment_id,
            customer_id=case.customer_id,
            session_id="live-benchmark-session",
            llm=tracker.llm,
            call_tool=ToolCaller(agent),
            environment=env,
        )

        final = steps[-1].output if steps else ""
        feedback = env.evaluate(final)
        success = feedback.success

        detail = {
            "final_output": final,
            "feedback": feedback.model_dump(),
            "tool_sequence": agent.calls,
            "forced_steps": [step.step for step in steps if step.forced],
            "steps": [
                {
                    "step": step.step,
                    "kind": step.kind.value,
                    "tool_name": step.tool_name,
                    "instruction": step.instruction,
                    "forced": step.forced,
                }
                for step in steps
            ],
        }
        error = None

    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
        detail = {
            "error": error,
            "tool_sequence": agent.calls,
        }

    latency_ms = (time.perf_counter() - started) * 1000

    return _row(
        group="decomposition",
        case=case,
        method="Dynamic decomposition",
        success=success,
        tracker=tracker,
        latency_ms=latency_ms,
        input_price=input_price,
        output_price=output_price,
        tool_calls=len(agent.calls),
        error=error,
    ), detail


def _run_ps(
    case: BenchmarkCase,
    *,
    tracker: LiveTracker,
    input_price: float,
    output_price: float,
) -> tuple[LiveBenchmarkRow, dict[str, Any]]:
    env = _environment(case)
    started = time.perf_counter()

    try:
        output = plan_and_solve(_benchmark_problem(case), tracker.llm)
        feedback = env.evaluate(output)
        success = feedback.success
        detail = {
            "output": output,
            "feedback": feedback.model_dump(),
        }
        error = None

    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
        detail = {"error": error}

    latency_ms = (time.perf_counter() - started) * 1000

    return _row(
        group="planning",
        case=case,
        method="Plan-and-Solve",
        success=success,
        tracker=tracker,
        latency_ms=latency_ms,
        input_price=input_price,
        output_price=output_price,
        error=error,
    ), detail


def _run_tot(
    case: BenchmarkCase,
    *,
    tracker: LiveTracker,
    input_price: float,
    output_price: float,
) -> tuple[LiveBenchmarkRow, dict[str, Any]]:
    env = _environment(case)
    started = time.perf_counter()

    try:
        thoughts = tree_of_thoughts(
            _benchmark_problem(case),
            tracker.llm,
            depth=2,
            beam_width=2,
        )

        if not thoughts:
            raise RuntimeError("Tree of Thoughts returned no candidate paths.")

        best = max(thoughts, key=lambda thought: thought.score)
        feedback = env.evaluate(best.state)
        success = feedback.success

        detail = {
            "output": best.state,
            "model_score": best.score,
            "feedback": feedback.model_dump(),
        }
        error = None

    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
        detail = {"error": error}

    latency_ms = (time.perf_counter() - started) * 1000

    return _row(
        group="planning",
        case=case,
        method="Tree of Thoughts",
        success=success,
        tracker=tracker,
        latency_ms=latency_ms,
        input_price=input_price,
        output_price=output_price,
        error=error,
    ), detail


def _run_lats(
    case: BenchmarkCase,
    *,
    tracker: LiveTracker,
    grounded: bool,
    input_price: float,
    output_price: float,
) -> tuple[LiveBenchmarkRow, dict[str, Any]]:
    real_env = _environment(case)

    search_env = (
        real_env
        if grounded
        else RandomEnvironment(
            success_threshold=0.0,
            rng=random.Random(7),
        )
    )

    method = "LATS grounded" if grounded else "LATS ungrounded"
    started = time.perf_counter()

    try:
        result = lats(
            _benchmark_problem(case),
            tracker.llm,
            search_env,
            iterations=2,
            n_actions=2,
        )

        # Final quality is always judged against the real Swiftrail validator,
        # including for the intentionally ungrounded LATS baseline.
        feedback = real_env.evaluate(result.output)
        success = feedback.success

        detail = {
            "output": result.output,
            "search_reported_success": result.success,
            "search_score": result.best_score,
            "iterations": result.iterations,
            "feedback": feedback.model_dump(),
        }
        error = None

    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
        detail = {"error": error}

    latency_ms = (time.perf_counter() - started) * 1000

    return _row(
        group="planning",
        case=case,
        method=method,
        success=success,
        tracker=tracker,
        latency_ms=latency_ms,
        input_price=input_price,
        output_price=output_price,
        error=error,
    ), detail


def _run_self_refine(
    case: BenchmarkCase,
    *,
    tracker: LiveTracker,
    input_price: float,
    output_price: float,
) -> tuple[LiveBenchmarkRow, dict[str, Any]]:
    env = _environment(case)
    started = time.perf_counter()

    try:
        result = reflect_and_refine(
            _benchmark_problem(case),
            case.bad_candidate,
            tracker.llm,
            critic_llm=tracker.llm,
            environment=env,
        )

        success = bool(
            result.revision_feedback
            and result.revision_feedback.success
        )

        detail = {
            "draft": result.draft,
            "critique": result.critique,
            "revised": result.revised,
            "grounded_issues": result.grounded_issues,
            "revision_feedback": (
                result.revision_feedback.model_dump()
                if result.revision_feedback
                else None
            ),
        }
        error = None

    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
        detail = {"error": error}

    latency_ms = (time.perf_counter() - started) * 1000

    return _row(
        group="self_correction",
        case=case,
        method="Self-Refine",
        success=success,
        tracker=tracker,
        latency_ms=latency_ms,
        input_price=input_price,
        output_price=output_price,
        error=error,
    ), detail


def _run_reflexion(
    case: BenchmarkCase,
    *,
    tracker: LiveTracker,
    input_price: float,
    output_price: float,
) -> tuple[LiveBenchmarkRow, dict[str, Any]]:
    env = _environment(case)
    started = time.perf_counter()

    try:
        result = reflexion(
            _benchmark_problem(case),
            tracker.llm,
            env,
            max_trials=2,
            memory_size=1,
            critic_llm=tracker.llm,
        )

        success = result.success

        detail = {
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
        error = None

    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
        detail = {"error": error}

    latency_ms = (time.perf_counter() - started) * 1000

    return _row(
        group="self_correction",
        case=case,
        method="Reflexion",
        success=success,
        tracker=tracker,
        latency_ms=latency_ms,
        input_price=input_price,
        output_price=output_price,
        error=error,
    ), detail


def _aggregate(rows: list[LiveBenchmarkRow]) -> list[dict[str, Any]]:
    methods: list[str] = []

    for row in rows:
        if row.method not in methods:
            methods.append(row.method)

    aggregate: list[dict[str, Any]] = []

    for method in methods:
        selected = [row for row in rows if row.method == method]

        aggregate.append(
            {
                "method": method,
                "success": (
                    f"{sum(row.success for row in selected)}/"
                    f"{len(selected)}"
                ),
                "success_rate": round(
                    sum(row.success for row in selected)
                    / len(selected),
                    3,
                ),
                "avg_llm_calls": round(
                    mean(row.llm_calls for row in selected),
                    2,
                ),
                "avg_input_tokens": round(
                    mean(row.input_tokens for row in selected),
                    1,
                ),
                "avg_output_tokens": round(
                    mean(row.output_tokens for row in selected),
                    1,
                ),
                "avg_total_tokens": round(
                    mean(row.total_tokens for row in selected),
                    1,
                ),
                "avg_latency_ms": round(
                    mean(row.latency_ms for row in selected),
                    3,
                ),
                "avg_estimated_cost_usd": round(
                    mean(
                        row.estimated_cost_usd
                        for row in selected
                    ),
                    8,
                ),
                "avg_tool_calls": round(
                    mean(row.tool_calls for row in selected),
                    2,
                ),
            }
        )

    return aggregate


def _markdown_table(
    aggregate: list[dict[str, Any]],
) -> str:
    lines = [
        (
            "| Method | Success | Avg LLM calls | "
            "Avg input tokens | Avg output tokens | "
            "Avg total tokens | Avg latency | "
            "Estimated cost/run | Avg tool calls |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for row in aggregate:
        lines.append(
            f"| {row['method']} | "
            f"{row['success']} | "
            f"{row['avg_llm_calls']} | "
            f"{row['avg_input_tokens']} | "
            f"{row['avg_output_tokens']} | "
            f"{row['avg_total_tokens']} | "
            f"{row['avg_latency_ms']:.3f} ms | "
            f"${row['avg_estimated_cost_usd']:.8f} | "
            f"{row['avg_tool_calls']} |"
        )

    return "\n".join(lines)


def _detail_table(rows: list[LiveBenchmarkRow]) -> str:
    lines = [
        (
            "| Group | Case | Method | Success | LLM calls | "
            "Input tokens | Output tokens | Total tokens | "
            "Latency | Cost | Error |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
        ),
    ]

    for row in rows:
        error = (row.error or "").replace("|", "/")

        lines.append(
            f"| {row.group} | "
            f"{row.case} | "
            f"{row.method} | "
            f"{row.success} | "
            f"{row.llm_calls} | "
            f"{row.input_tokens} | "
            f"{row.output_tokens} | "
            f"{row.total_tokens} | "
            f"{row.latency_ms:.3f} ms | "
            f"${row.estimated_cost_usd:.8f} | "
            f"{error} |"
        )

    return "\n".join(lines)


async def run_live_benchmark(
    *,
    model_name: str,
    api_key: str,
    input_price: float,
    output_price: float,
    max_retries: int,
) -> tuple[list[LiveBenchmarkRow], dict[str, Any]]:
    rows: list[LiveBenchmarkRow] = []
    details: dict[str, Any] = {
        "decomposition": {},
        "planning": {},
        "self_correction": {},
    }

    stable = stable_minor_hold_case()
    severe = severe_hold_sales_rep_case()
    rate = above_authority_rate_case()

    def tracker() -> LiveTracker:
        return _make_tracker(
            model_name=model_name,
            api_key=api_key,
            max_retries=max_retries,
        )

    # 1) Decomposition-first vs dynamic on the same request type.
    for case in (stable, severe):
        static_row, static_detail = await _run_static(
            case,
            tracker=tracker(),
            input_price=input_price,
            output_price=output_price,
        )

        dynamic_row, dynamic_detail = await _run_dynamic(
            case,
            tracker=tracker(),
            input_price=input_price,
            output_price=output_price,
        )

        rows.extend([static_row, dynamic_row])

        details["decomposition"][case.name] = {
            static_row.method: static_detail,
            dynamic_row.method: dynamic_detail,
        }

    # 2) PS vs ToT vs LATS, including ungrounded vs grounded LATS.
    for case in (stable, rate):
        case_details: dict[str, Any] = {}

        ps_row, ps_detail = _run_ps(
            case,
            tracker=tracker(),
            input_price=input_price,
            output_price=output_price,
        )

        tot_row, tot_detail = _run_tot(
            case,
            tracker=tracker(),
            input_price=input_price,
            output_price=output_price,
        )

        lats_ungrounded_row, lats_ungrounded_detail = _run_lats(
            case,
            tracker=tracker(),
            grounded=False,
            input_price=input_price,
            output_price=output_price,
        )

        lats_grounded_row, lats_grounded_detail = _run_lats(
            case,
            tracker=tracker(),
            grounded=True,
            input_price=input_price,
            output_price=output_price,
        )

        rows.extend(
            [
                ps_row,
                tot_row,
                lats_ungrounded_row,
                lats_grounded_row,
            ]
        )

        case_details[ps_row.method] = ps_detail
        case_details[tot_row.method] = tot_detail
        case_details[lats_ungrounded_row.method] = (
            lats_ungrounded_detail
        )
        case_details[lats_grounded_row.method] = (
            lats_grounded_detail
        )

        details["planning"][case.name] = case_details

    # 3) Self-Refine vs Reflexion.
    for case in (rate, severe):
        self_refine_row, self_refine_detail = _run_self_refine(
            case,
            tracker=tracker(),
            input_price=input_price,
            output_price=output_price,
        )

        reflexion_row, reflexion_detail = _run_reflexion(
            case,
            tracker=tracker(),
            input_price=input_price,
            output_price=output_price,
        )

        rows.extend([self_refine_row, reflexion_row])

        details["self_correction"][case.name] = {
            self_refine_row.method: self_refine_detail,
            reflexion_row.method: reflexion_detail,
        }

    return rows, details


def _resolve_prices(
    *,
    model_name: str,
    input_price: float | None,
    output_price: float | None,
) -> tuple[float, float]:
    if input_price is not None and output_price is not None:
        return input_price, output_price

    known = KNOWN_STANDARD_PRICES.get(model_name)

    if known is None:
        raise ValueError(
            f"No built-in Standard-tier pricing is configured for "
            f"{model_name!r}. Pass both --input-price and --output-price "
            "as USD per 1M tokens."
        )

    return known


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Swiftrail planning cost/quality benchmark with a "
            "real Mistral model and provider-reported token usage."
        )
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Mistral model id. Defaults to MISTRAL_MODEL from .env, then "
            "mistral-small-latest."
        ),
    )

    parser.add_argument(
        "--input-price",
        type=float,
        default=None,
        help="USD per 1M input tokens. Required for unknown model ids.",
    )

    parser.add_argument(
        "--output-price",
        type=float,
        default=None,
        help="USD per 1M output tokens. Required for unknown model ids.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is missing from the root .env file."
        )

    model_name = (
        args.model
        or os.getenv("MISTRAL_MODEL")
        or "mistral-small-latest"
    )

    input_price, output_price = _resolve_prices(
        model_name=model_name,
        input_price=args.input_price,
        output_price=args.output_price,
    )

    rows, details = asyncio.run(
        run_live_benchmark(
            model_name=model_name,
            api_key=api_key,
            input_price=input_price,
            output_price=output_price,
            max_retries=args.max_retries,
        )
    )

    aggregate = _aggregate(rows)

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    payload = {
        "benchmark_mode": "live_mistral_fixed_swiftrail_snapshots",
        "model": model_name,
        "token_accounting": (
            "provider-reported LangChain AIMessage usage_metadata"
        ),
        "pricing": {
            "tier": "standard",
            "input_usd_per_million": input_price,
            "output_usd_per_million": output_price,
        },
        "scope_note": (
            "The LLM calls are live provider calls. Direct planning/self-correction "
            "methods receive the same fixed Swiftrail snapshot and authoritative "
            "policy text; decomposition methods still execute the fixed fake MCP "
            "tool observations. A common ACTION output contract is supplied so the "
            "grounded validator measures decision quality rather than provider-"
            "specific Markdown formatting. No database writes are performed."
        ),
        "rows": [row.as_dict() for row in rows],
        "aggregate": aggregate,
        "details": details,
    }

    json_path = ARTIFACTS_DIR / "live_planning_benchmark.json"
    markdown_path = ARTIFACTS_DIR / "live_planning_benchmark.md"

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown = (
        "# Live Planning Cost / Quality Benchmark\n\n"
        f"- Model: `{model_name}`\n"
        "- LLM calls: real Mistral API calls\n"
        "- Token counts: provider-reported usage metadata\n"
        "- Tool evidence: fixed Swiftrail seed-data-shaped snapshots\n"
        "- Evaluation format: shared plain ACTION-line contract\n"
        f"- Standard input price: ${input_price}/1M tokens\n"
        f"- Standard output price: ${output_price}/1M tokens\n\n"
        "## Aggregate comparison\n\n"
        + _markdown_table(aggregate)
        + "\n\n## Per-case runs\n\n"
        + _detail_table(rows)
        + "\n"
    )

    markdown_path.write_text(
        markdown,
        encoding="utf-8",
    )

    print(markdown)
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {markdown_path}")


if __name__ == "__main__":
    main()
