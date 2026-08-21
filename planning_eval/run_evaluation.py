from __future__ import annotations

import json
from pathlib import Path

from planning_eval.evaluation_suite import run_grounded_suite


def main() -> None:
    rows = run_grounded_suite()
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    path = out / "person3_grounded_suite.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
