"""Static vs Dynamic Divergence Handling.

Both decomposition methods are run against the *same* real request
(shipment_id, customer_id) in planning_eval's comparison suite. This module
answers the question a comparison table alone can't: not just "did dynamic
score higher", but *where exactly* did it do something decomposition-first
never would have, and why was that the safer choice.

Divergence is detected structurally (tool ids actually called and their
kind), not by diffing free text -- diffing prose is noisy and would flag
harmless rewordings as "divergence".
"""

from __future__ import annotations

from dataclasses import dataclass

from .algorithms.dynamic_decomposition import DynamicStep
from .swiftrail_subtask import SubtaskKind, SwiftrailPlan


@dataclass
class DivergencePoint:
    index: int
    static_next: str | None
    dynamic_next: str
    reason: str


@dataclass
class DivergenceReport:
    diverged: bool
    point: DivergencePoint | None
    static_tool_sequence: list[str]
    dynamic_tool_sequence: list[str]


def _static_tool_sequence(static_plan: SwiftrailPlan) -> list[str]:
    """The order decomposition-first would execute tool_call nodes in,
    reusing the toolkit's own topological batching (Plan.execution_batches)
    rather than re-deriving an order by hand."""

    ordered: list[str] = []
    for batch in static_plan.execution_batches():
        for task_id in batch:
            meta = static_plan.meta[task_id]
            if meta.kind is SubtaskKind.TOOL_CALL:
                ordered.append(meta.tool_name)  # type: ignore[arg-type]
    return ordered


def _dynamic_tool_sequence(dynamic_steps: list[DynamicStep]) -> list[str]:
    return [s.tool_name for s in dynamic_steps if s.kind is SubtaskKind.TOOL_CALL and s.tool_name]


def compute_divergence(
    static_plan: SwiftrailPlan,
    dynamic_steps: list[DynamicStep],
) -> DivergenceReport:
    static_seq = _static_tool_sequence(static_plan)
    dynamic_seq = _dynamic_tool_sequence(dynamic_steps)

    # 1) Did dynamic decomposition stop early relative to the static plan's
    #    full read-only reconnaissance? A forced escalation step cutting the
    #    loop short (see dynamic_decomposition._forced_next_step) is the
    #    clearest, most common form of real divergence for this request
    #    type: the static plan would still blindly run the remaining
    #    lookups (and its terminal reasoning task would still consider
    #    recommending release), while dynamic decomposition already
    #    stopped gathering evidence and moved straight to escalation.
    forced_step = next((s for s in dynamic_steps if s.forced), None)
    if forced_step is not None:
        cut_at = len(_dynamic_tool_sequence(dynamic_steps[: dynamic_steps.index(forced_step)]))
        if cut_at < len(static_seq):
            return DivergenceReport(
                diverged=True,
                point=DivergencePoint(
                    index=cut_at,
                    static_next=static_seq[cut_at],
                    dynamic_next="reasoning:escalate",
                    reason=(
                        "A severe active credit hold was observed after "
                        f"{cut_at} lookup(s); dynamic decomposition escalated "
                        "immediately instead of continuing the remaining "
                        f"planned lookups {static_seq[cut_at:]!r} that "
                        "decomposition-first would have executed anyway "
                        "before its single terminal reasoning task ever saw "
                        "the hold."
                    ),
                ),
                static_tool_sequence=static_seq,
                dynamic_tool_sequence=dynamic_seq,
            )

    # 2) Otherwise, compare sequences position by position -- e.g. dynamic
    # decomposition chose a different next lookup given an early result
    # (not just the forced-escalation case above).
    for i, (s, d) in enumerate(zip(static_seq, dynamic_seq)):
        if s != d:
            return DivergenceReport(
                diverged=True,
                point=DivergencePoint(
                    index=i,
                    static_next=s,
                    dynamic_next=d,
                    reason=f"Dynamic decomposition chose {d!r} where the static plan had committed to {s!r}.",
                ),
                static_tool_sequence=static_seq,
                dynamic_tool_sequence=dynamic_seq,
            )

    return DivergenceReport(
        diverged=False,
        point=None,
        static_tool_sequence=static_seq,
        dynamic_tool_sequence=dynamic_seq,
    )
