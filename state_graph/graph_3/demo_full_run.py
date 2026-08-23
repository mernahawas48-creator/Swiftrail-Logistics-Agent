"""
demo_full_run.py — exercises every concern in one scenario, in-process.

Customer 3, $12,000 overdue, severe hold, claims one invoice is wrong
(mirrors the worked example in the Graph 3 design doc).

Path exercised:
  load_account_state
  -> build_remediation_plan (LATS picks dispute_review)
  -> await_customer_action (WAITS — real pause)
  -> [customer submits evidence, too short]
  -> review_evidence -> request_more_evidence -> await_customer_action (CYCLE)
  -> [customer submits real evidence]
  -> review_evidence -> wait_for_finance_admin (HITL — real pause)
  -> [admin approves via platform]
  -> execute_remediation_action -> forced MCP failure -> TICKET
  -> [admin resolves ticket via platform]
  -> execute_remediation_action (retried, succeeds) -> complete
"""
from __future__ import annotations

from state_graph.graph_3 import mcp_tools
from state_graph.graph_3.checkpointer import Checkpointer
from state_graph.graph_3.graph3_credit_hold import graph

cp = Checkpointer()


def show(run_id, label):
    run = cp.get_run(run_id)
    print(f"[{label}] status={run['status']:<14} current_node={run['current_node']}")


def main():
    mcp_tools.seed_customer("3", overdue_amount=12000.0, severity="severe")

    print("=== starting run ===")
    run_id = graph.start("load_account_state", {
        "customer_id": "3",
        "customer_claim": "One invoice is incorrect",
        "log": [],
    })
    show(run_id, "after start")

    print("\n=== customer submits weak evidence ===")
    graph.resume(run_id, extra_state={"customer_evidence": "nope"})
    show(run_id, "after weak evidence")
    task = cp.list_hitl_tasks("pending")
    print("HITL tasks pending:", len(task), "(expect 0 — we're still in the evidence cycle)")

    print("\n=== customer submits real evidence ===")
    graph.resume(run_id, extra_state={"customer_evidence": "Invoice INV-3-1 double-billed freight surcharge, see attached BOL."})
    show(run_id, "after real evidence")

    hitl_tasks = cp.list_hitl_tasks("pending")
    print(f"HITL tasks pending: {len(hitl_tasks)}")
    task = hitl_tasks[0]
    print(f"  reason: {task['reason']}")

    print("\n=== finance admin approves via platform ===")
    cp.decide_hitl_task(task["task_id"], "approve", decided_by="admin_jane")
    graph.resume(run_id, extra_state={"finance_decision": "approve", "_force_tool_failure": True})
    show(run_id, "after admin approval (tool forced to fail)")

    tickets = cp.list_tickets("open")
    print(f"Open tickets: {len(tickets)}")
    ticket = tickets[0]
    print(f"  error: {ticket['error_type']} — {ticket['error_message']}")

    print("\n=== admin investigates and resolves the ticket ===")
    cp.set_ticket_status(ticket["ticket_id"], "investigating")
    cp.set_ticket_status(ticket["ticket_id"], "resolved")
    graph.resume(run_id)  # no forced failure this time
    show(run_id, "after ticket resolved")

    final = cp.get_run(run_id)
    print(f"\nFinal status: {final['status']}")
    last_cp = cp.latest_checkpoint(run_id)
    print("Log:")
    for line in last_cp["state"]["log"]:
        print(" -", line)


if __name__ == "__main__":
    main()
