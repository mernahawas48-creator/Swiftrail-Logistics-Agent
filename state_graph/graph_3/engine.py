"""
engine.py — the generic state-graph runner shared by all three graphs.

This is deliberately NOT a DAG executor. A graph here is a dict of
node_name -> callable, where each node returns the name of the next node
(possibly one already visited — real cycles), or None to finish. Nodes can
raise three special signals:

  * HITLPause      — an expected pause: this node is not allowed to decide
                      alone, so the run stops, a task is opened for a human
                      admin, and the run will only continue once that admin
                      acts through the platform.
  * WaitForEvent    — an expected pause on something outside the model:
                      a customer hasn't replied yet. Distinct from HITL:
                      nothing needs an admin decision, the run is just
                      waiting on an external party.
  * NodeFailure     — an UNPLANNED failure: a tool call errored, a schema
                      didn't validate, the model returned something the
                      graph can't act on. This opens a ticket, distinct
                      from both of the above, and the run resumes from
                      the failed node (not the top) once the ticket is
                      resolved.

Every one of those three paths, and every ordinary transition, is written
through Checkpointer.save() BEFORE the engine looks at what to do next.
That's what makes `run_to_crash.py` a legitimate crash-and-resume proof:
the checkpoint for a completed node is durable on disk before the engine
ever starts the next one.
"""
from __future__ import annotations

import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from .checkpointer import Checkpointer

NodeFn = Callable[[dict], "NodeResult"]


@dataclass
class NodeResult:
    next_node: str | None   # None = graph finished
    state: dict


class HITLPause(Exception):
    """Raised by a node to stop the run and open a human decision task."""
    def __init__(self, reason: str, options: list[str]):
        super().__init__(reason)
        self.reason = reason
        self.options = options


class WaitForEvent(Exception):
    """Raised by a node to stop the run pending an external party's action
    (a customer reply), with no admin decision required."""
    def __init__(self, wait_key: str, reason: str):
        super().__init__(reason)
        self.wait_key = wait_key
        self.reason = reason


class NodeFailure(Exception):
    """Raised (or converted from any uncaught exception) to open a ticket
    instead of an HITL task. This is the UNPLANNED path."""
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


@dataclass
class StateGraph:
    name: str
    nodes: dict[str, NodeFn] = field(default_factory=dict)
    checkpointer: Checkpointer = field(default_factory=Checkpointer)
    # optional: simulate a hard process kill right after a given node commits
    # its checkpoint, for the crash-and-resume demo. Set via env var in the
    # demo script rather than in normal operation.
    crash_after_node: str | None = None

    def node(self, name: str):
        def _register(fn: NodeFn):
            self.nodes[name] = fn
            return fn
        return _register

    def start(self, initial_node: str, initial_state: dict) -> str:
        run_id = str(uuid.uuid4())
        self.checkpointer.start_run(run_id, self.name, initial_node, initial_state)
        self._drive(run_id, initial_node, initial_state)
        return run_id

    def resume(self, run_id: str, extra_state: dict | None = None) -> None:
        """Continue a paused/ticketed run from its last checkpoint. This is
        the ONLY way execution advances again — a HITL decision or a ticket
        resolution calls this, never a fresh start()."""
        cp = self.checkpointer.latest_checkpoint(run_id)
        if cp is None:
            raise ValueError(f"no checkpoint for run {run_id}")
        state = cp["state"]
        if extra_state:
            state.update(extra_state)
        self._drive(run_id, cp["node_name"], state, is_resume=True)

    def _drive(self, run_id: str, node_name: str, state: dict, is_resume: bool = False) -> None:
        # IMPORTANT: `current` is always the node about to EXECUTE, never a
        # node that already completed. That invariant is what makes resume
        # correct: the checkpoint we read on resume names the next node to
        # run (for a clean transition) or the SAME node that needs to be
        # re-entered (for a pause/failure, since that node's work wasn't
        # finished). Completed nodes are never re-run.
        current = node_name
        while True:
            fn = self.nodes.get(current)
            if fn is None:
                raise ValueError(f"unknown node '{current}' in graph '{self.name}'")

            try:
                result = fn(state)
            except HITLPause as pause:
                task_id = self.checkpointer.create_hitl_task(run_id, current, pause.reason, pause.options)
                state["_last_hitl_task_id"] = task_id
                self.checkpointer.save(run_id, current, state, status="paused_hitl")
                return
            except WaitForEvent as wait:
                state["_waiting_on"] = wait.wait_key
                self.checkpointer.save(run_id, current, state, status="paused_wait")
                return
            except NodeFailure as fail:
                ticket_id = self.checkpointer.create_ticket(run_id, current, fail.error_type, fail.message)
                state["_last_ticket_id"] = ticket_id
                self.checkpointer.save(run_id, current, state, status="ticketed")
                return
            except Exception as exc:
                ticket_id = self.checkpointer.create_ticket(
                    run_id, current, type(exc).__name__, f"{exc}\n{traceback.format_exc(limit=3)}"
                )
                state["_last_ticket_id"] = ticket_id
                self.checkpointer.save(run_id, current, state, status="ticketed")
                return

            # Node completed cleanly.
            state = result.state
            just_finished = current

            if result.next_node is None:
                self.checkpointer.save(run_id, just_finished, state, status="completed")
                return

            # Checkpoint names the NEXT node as the resume point — this is
            # the write that must land durably before we ever call the next
            # node's function, so a kill right after this line can never
            # cause `just_finished` to run twice.
            self.checkpointer.save(run_id, result.next_node, state, status="running")

            if self.crash_after_node and just_finished == self.crash_after_node:
                import os
                os._exit(137)  # simulate `kill -9`

            current = result.next_node
