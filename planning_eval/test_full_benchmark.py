from __future__ import annotations

import asyncio

from planning_eval.full_benchmark import _required_checks, run_benchmark


def test_full_benchmark_covers_required_cases():
    rows, details = asyncio.run(run_benchmark())
    checks = _required_checks(rows, details)
    assert checks
    assert all(checks.values()), checks


def test_full_benchmark_contains_every_required_method():
    rows, _ = asyncio.run(run_benchmark())
    methods = {row.method for row in rows}
    assert {
        "Decomposition-first",
        "Dynamic decomposition",
        "Plan-and-Solve",
        "Tree of Thoughts",
        "LATS ungrounded",
        "LATS grounded",
        "Self-Refine",
        "Reflexion",
    }.issubset(methods)
