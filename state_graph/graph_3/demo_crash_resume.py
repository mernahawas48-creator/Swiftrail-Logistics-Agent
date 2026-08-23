"""
demo_crash_resume.py — proves checkpointing is real, not a log file.

Usage:
    python -m state_graph.demo_crash_resume

What it does:
  1. Seeds a fresh customer scenario.
  2. Launches a SUBPROCESS that starts a Graph 3 run and is configured to
     os._exit(137) (a hard kill, same as `kill -9`) the instant after
     'load_account_state' commits its checkpoint — before
     'build_remediation_plan' ever runs.
  3. Confirms the subprocess is really gone (nonzero/killed exit code).
  4. In THIS process, calls resume() on the run id and shows it continues
     from 'build_remediation_plan' onward — proving 'load_account_state'
     is not re-executed and no collected state (invoices, hold info) was
     lost.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from state_graph.graph_3 import mcp_tools
from state_graph.graph_3.checkpointer import Checkpointer
from state_graph.graph_3.graph3_credit_hold import graph

CUSTOMER_ID = "CRASH-DEMO-1"


def main():
    mcp_tools.seed_customer(CUSTOMER_ID, overdue_amount=4200.0, severity="severe")

    print(f"--- launching subprocess to start run for customer {CUSTOMER_ID} ---")
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT, CUSTOMER_ID],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    print(proc.stdout)
    if proc.stderr:
        print("stderr:", proc.stderr[-500:])
    print(f"subprocess exit code: {proc.returncode} (137 == hard-killed via os._exit(137))")

    run_id = proc.stdout.strip().splitlines()[-1]
    cp = Checkpointer()
    checkpoint = cp.latest_checkpoint(run_id)
    print(f"\n--- checkpoint on disk after kill ---")
    print(f"next node queued to run: {checkpoint['node_name']}  (load_account_state already completed)")
    print(f"state collected so far: invoices={len(checkpoint['state'].get('invoices', []))}, "
          f"hold_severity={checkpoint['state'].get('credit_hold', {}).get('severity')}")
    assert checkpoint["node_name"] == "build_remediation_plan", "expected load_account_state to have completed"

    print(f"\n--- resuming run {run_id} in THIS process (simulating server restart) ---")
    graph.resume(run_id)

    run = cp.get_run(run_id)
    history = cp.history(run_id)
    print(f"resumed run status: {run['status']}")
    print("node sequence after resume (note load_account_state runs exactly once):")
    for h in history:
        print(f"  seq={h['seq']:<3} node={h['node_name']}")


_SUBPROCESS_SCRIPT = """
import sys, uuid
sys.path.insert(0, ".")
from state_graph.graph_3.graph3_credit_hold import graph

graph.crash_after_node = "load_account_state"
customer_id = sys.argv[1] if len(sys.argv) > 1 else "CRASH-DEMO-1"
initial_state = {"customer_id": customer_id, "log": []}

# Print the run id FIRST (before driving, since the drive call will hard-exit
# mid-way and never return) so the parent process can pick it up.
run_id = str(uuid.uuid4())
print(run_id, flush=True)
graph.checkpointer.start_run(run_id, graph.name, "load_account_state", initial_state)
graph._drive(run_id, "load_account_state", initial_state)
"""

if __name__ == "__main__":
    main()
