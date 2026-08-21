"""Runs all four context-management strategies against the fixed
long-context test suite and produces the comparison table the README
cites.

Run with: python -m context_eval.evaluate_strategies

What this measures honestly, and what it does not:
  - "Exact fact string retained": whether the planted critical fact is still
    present, verbatim, somewhere in the message list the strategy
    hands back. This is what actually matters for the failure mode
    (an agent that can't see the fact can't act on it), so it's
    checked at the string level rather than via an LLM judge -- an
    LLM judge would add a second, unrelated model cost to every run.
  - "Tokens": approximated via a 4-chars-per-token heuristic over the
    serialized message list. No model call is made, so this is a
    proxy for prompt size, not billed API tokens. Swap in a real
    tokenizer for the selected model if exact prompt-token estimates
    numbers are needed.
  - "Output tokens": provider-reported Mistral output tokens for recursive
    summarization. Deterministic strategies make no model call and report 0.
  - "Latency": wall-clock time of strategy.apply(). Recursive summarization
    therefore includes the live Mistral request; the other strategies measure
    local pruning overhead only.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from context_eval.strategies.recursive_summarization import RecursiveSummarization
from context_eval.strategies.sliding_window import SlidingWindow
from context_eval.strategies.tool_output_masking import ToolOutputMasking
from context_eval.strategies.zone_based_pruning import ZoneBasedPruning
from context_eval.test_transcripts import build_test_suite

RESULTS_DIR = Path(__file__).parent / "results"

STRATEGIES = {
    "sliding_window": SlidingWindow(max_messages=10),
    "tool_output_masking": ToolOutputMasking(keep_last_tool_outputs=3),
    "recursive_summarization": RecursiveSummarization(keep_last_messages=6),
    "zone_based_pruning": ZoneBasedPruning(keep_recent_messages=8),
}


def approx_tokens(messages: list[dict]) -> int:
    """4 chars/token heuristic. See module docstring for why this is a
    proxy and not a billed-token count."""
    text = json.dumps(messages)
    return max(1, len(text) // 4)


@dataclass
class RunResult:
    strategy: str
    transcript: str
    fact_recalled: bool
    input_tokens: int
    output_tokens: int
    latency_seconds: float


def run_one(strategy_name: str, strategy, transcript) -> RunResult:
    generator = getattr(strategy, "generator", None)
    reset_usage = getattr(generator, "reset_usage", None)
    if callable(reset_usage):
        reset_usage()

    start = time.perf_counter()
    pruned = strategy.apply(transcript.messages)
    elapsed = time.perf_counter() - start

    serialized = json.dumps(pruned)
    recalled = transcript.critical_fact in serialized

    # recursive_summarization (and any future strategy that calls a
    # real model) reports its own billed output tokens via the
    # generator's usage history. Strategies that never call a model
    # genuinely spend 0 output tokens -- that's the honest tradeoff
    # the comparison table is supposed to show.
    output_tokens = 0
    if generator is not None and hasattr(generator, "last_usage"):
        output_tokens = generator.last_usage.output_tokens

    return RunResult(
        strategy=strategy_name,
        transcript=transcript.name,
        fact_recalled=recalled,
        input_tokens=approx_tokens(pruned),
        output_tokens=output_tokens,
        latency_seconds=elapsed,
    )


def summarize(results: list[RunResult]) -> list[dict]:
    rows = []
    for name in STRATEGIES:
        runs = [r for r in results if r.strategy == name]
        n = len(runs)
        recalled = sum(1 for r in runs if r.fact_recalled)
        rows.append({
            "strategy": name,
            "fact_recalled": f"{recalled}/{n}",
            "avg_input_tokens": round(sum(r.input_tokens for r in runs) / n, 1),
            "avg_output_tokens": round(sum(r.output_tokens for r in runs) / n, 1),
            "avg_latency_ms": round(sum(r.latency_seconds for r in runs) / n * 1000, 3),
        })
    return rows


def render_markdown_table(rows: list[dict]) -> str:
    header = "| Strategy | Exact fact string retained | Avg. input tokens/run | Avg. output tokens/run | Avg. latency |\n"
    header += "| --- | --- | --- | --- | --- |\n"
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {row['fact_recalled']} | "
            f"{row['avg_input_tokens']} | {row['avg_output_tokens']} | "
            f"{row['avg_latency_ms']}ms |\n"
        )
    return "".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    suite = build_test_suite()

    all_results: list[RunResult] = []
    for transcript in suite:
        for name, strategy in STRATEGIES.items():
            all_results.append(run_one(name, strategy, transcript))

    rows = summarize(all_results)

    (RESULTS_DIR / "context_comparison.json").write_text(
        json.dumps(
            {
                "per_run": [asdict(r) for r in all_results],
                "summary": rows,
            },
            indent=2,
        )
    )

    table_md = render_markdown_table(rows)
    (RESULTS_DIR / "context_comparison.md").write_text(
        "# Context management strategy comparison\n\n"
        f"10 long-context transcripts (28-45 tool-noise turns each), one "
        f"critical fact planted early, re-asked at the final turn.\n\n"
        f"{table_md}"
    )

    print(table_md)
    print(f"\nWrote results to {RESULTS_DIR}/context_comparison.{{json,md}}")


if __name__ == "__main__":
    main()
