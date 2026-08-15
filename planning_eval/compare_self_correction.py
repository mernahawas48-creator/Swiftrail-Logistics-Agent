from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def main():
    path = Path("artifacts/person3_self_correction_metrics.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    print("| Method | Success | Avg calls | Avg tokens | Avg latency ms | Avg cost USD |")
    print("|---|---:|---:|---:|---:|---:|")
    for method, items in grouped.items():
        success = sum(bool(x["success"]) for x in items) / len(items)
        calls = sum(x["llm_calls"] for x in items) / len(items)
        tokens = sum(x["total_tokens"] for x in items) / len(items)
        latency = sum(x["latency_ms"] for x in items) / len(items)
        cost = sum(x["estimated_cost_usd"] for x in items) / len(items)
        print(f"| {method} | {success:.2f} | {calls:.1f} | {tokens:.1f} | {latency:.2f} | ${cost:.6f} |")


if __name__ == "__main__":
    main()
