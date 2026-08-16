"""Decomposition Trace Logging.

``planning/cli.py`` (forked from the toolkit) already writes one JSON file
per run to ``artifacts/`` via ``save_artifact()``. This module reuses that
exact directory and file-naming convention -- it is not a second, parallel
logging system, just a payload shape specific to the decomposition/DAG
concern (which the generic CLI's payload never needed: tool calls, forced
safety overrides, divergence).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .divergence import DivergenceReport
from .execution_adapter import SubtaskExecution
from .swiftrail_subtask import SubtaskKind

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _execution_to_dict(exe: SubtaskExecution) -> dict[str, Any]:
    return {
        "task_id": exe.task_id,
        "kind": exe.kind.value,
        "tool_name": exe.tool_name,
        "duration_s": round(exe.duration_s, 4),
        "raw_tool_result": exe.raw_tool_result,
        "routed_method": exe.routed.method.value if exe.routed else None,
        "routed_rationale": exe.routed.routing_rationale if exe.routed else None,
        "routed_score": exe.routed.score if exe.routed else None,
        "output": exe.output,
    }


def log_decomposition_first_run(
    *,
    shipment_id: int,
    customer_id: int,
    plan_dump: dict[str, Any],
    execution_history: list[SubtaskExecution],
    final_result: str,
    total_llm_calls: int,
    total_tokens: int | None = None,
) -> Path:
    payload = {
        "method": "decomposition_first",
        "shipment_id": shipment_id,
        "customer_id": customer_id,
        "plan": plan_dump,
        "execution": [_execution_to_dict(e) for e in execution_history],
        "result": final_result,
        "metrics": {
            "total_llm_calls": total_llm_calls,
            "total_tokens": total_tokens,
            "total_latency_s": round(sum(e.duration_s for e in execution_history), 4),
            "tool_calls": sum(1 for e in execution_history if e.kind is SubtaskKind.TOOL_CALL),
        },
    }
    return _write(payload)


def log_dynamic_run(
    *,
    shipment_id: int,
    customer_id: int,
    steps: list,  # list[DynamicStep]
    final_result: str,
    total_llm_calls: int,
    total_tokens: int | None = None,
) -> Path:
    payload = {
        "method": "dynamic_decomposition",
        "shipment_id": shipment_id,
        "customer_id": customer_id,
        "steps": [
            {
                "step": s.step,
                "kind": s.kind.value,
                "tool_name": s.tool_name,
                "instruction": s.instruction,
                "forced_by_safety_rule": s.forced,
                "output": s.output,
                "raw": s.raw,
            }
            for s in steps
        ],
        "result": final_result,
        "metrics": {
            "total_llm_calls": total_llm_calls,
            "total_tokens": total_tokens,
            "tool_calls": sum(1 for s in steps if s.kind.value == "tool_call"),
            "forced_safety_overrides": sum(1 for s in steps if s.forced),
        },
    }
    return _write(payload)


def log_divergence(divergence: DivergenceReport) -> Path:
    return _write({"method": "divergence_comparison", "report": asdict(divergence)})


def _write(payload: dict[str, Any]) -> Path:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ARTIFACT_DIR / f"run-{payload.get('method', 'trace')}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
