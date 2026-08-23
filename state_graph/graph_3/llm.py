from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ACTION = re.compile(
    r"\bACTION\s*:\s*(dispute_review|payment_confirmation)\b",
    re.IGNORECASE,
)


def _run_lats(*args, **kwargs):
    from planning.algorithms.lats import lats

    return lats(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class PlanningFeedback:
    success: bool
    score: float
    details: list[str]
    evidence: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "score": self.score,
            "details": self.details,
            "evidence": self.evidence,
        }


def _action_from_candidate(candidate: str) -> str | None:
    match = _ACTION.search(candidate)
    return match.group(1).lower() if match else None


class CreditHoldPlanningEnvironment:
    """Ground LATS branches in the account facts already read through MCP."""

    def __init__(self, *, customer_claim: str | None, hold: dict[str, Any]) -> None:
        self.expected_action = (
            "dispute_review" if customer_claim else "payment_confirmation"
        )
        self.severity = str(hold.get("severity", "")).lower()

    def evaluate(self, candidate: str) -> PlanningFeedback:
        action = _action_from_candidate(candidate)
        issues: list[str] = []
        lowered = candidate.lower()
        if action is None:
            issues.append(
                "Candidate must contain exactly one ACTION marker with an allowed plan."
            )
        elif action != self.expected_action:
            issues.append(
                f"Observed account context requires {self.expected_action}, not {action}."
            )
        if self.severity == "severe" and not any(
            marker in lowered
            for marker in ("finance", "human approval", "admin approval", "escalat")
        ):
            issues.append(
                "A severe hold plan must preserve finance-manager HITL approval."
            )
        if "already released" in lowered or "release succeeded" in lowered:
            issues.append(
                "The plan cannot claim that the MCP release succeeded before execution."
            )
        return PlanningFeedback(
            success=not issues,
            score=0.95 if not issues else 0.1,
            details=issues,
            evidence={
                "expected_action": self.expected_action,
                "hold_severity": self.severity,
            },
        )


@dataclass(frozen=True, slots=True)
class LATSRemediationPlan:
    action: str
    narrative: str
    score: float
    iterations: int
    search_tree: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "narrative": self.narrative,
            "score": self.score,
            "iterations": self.iterations,
            "search_tree": list(self.search_tree),
        }


class MistralLATSRemediationPlanner:
    """Run bounded Mistral LATS search for Graph 3 remediation planning."""

    def __init__(
        self,
        llm: Any | None = None,
        *,
        runner: Callable[..., Any] | None = None,
    ) -> None:
        self._llm = llm
        self.runner = runner or _run_lats

    @property
    def llm(self) -> Any:
        if self._llm is None:
            project_root = Path(__file__).resolve().parents[2]
            from dotenv import load_dotenv

            load_dotenv(project_root / ".env")
            api_key = os.getenv("MISTRAL_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("MISTRAL_API_KEY is missing from the root .env file.")
            from langchain_mistralai import ChatMistralAI

            self._llm = ChatMistralAI(
                model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
                api_key=api_key,
                temperature=0.2,
                max_retries=2,
            )
        return self._llm

    def plan(
        self,
        *,
        invoices: list[dict[str, Any]],
        hold: dict[str, Any],
        overdue_amount: float,
        customer_claim: str | None,
    ) -> LATSRemediationPlan:
        environment = CreditHoldPlanningEnvironment(
            customer_claim=customer_claim,
            hold=hold,
        )
        task = f"""
Create a grounded credit-hold remediation plan from this observed MCP data.
Invoices: {json.dumps(invoices, default=str)}
Active hold: {json.dumps(hold, default=str)}
Overdue amount: {overdue_amount}
Customer claim: {json.dumps(customer_claim)}

Every candidate must end with exactly one marker:
ACTION: dispute_review
or
ACTION: payment_confirmation

Do not claim a tool already succeeded. A severe hold must preserve explicit
finance-manager human approval before release.
""".strip()
        result = self.runner(
            task,
            self.llm,
            environment,
            iterations=2,
            n_actions=2,
        )
        action = _action_from_candidate(result.output)
        if not result.success or action is None:
            raise RuntimeError(
                "Mistral LATS did not produce a grounded remediation plan."
            )
        tree: tuple[dict[str, Any], ...] = ()
        if getattr(result, "root", None) is not None:
            from planning.algorithms.lats import flatten_lats_tree

            tree = tuple(flatten_lats_tree(result.root))
        return LATSRemediationPlan(
            action=action,
            narrative=result.output.strip(),
            score=float(result.best_score),
            iterations=int(result.iterations),
            search_tree=tree,
        )
