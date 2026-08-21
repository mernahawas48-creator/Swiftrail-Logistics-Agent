from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    RUNNING = "running"
    WAITING_EXTERNAL = "waiting_external"
    WAITING_HITL = "waiting_hitl"
    WAITING_TICKET = "waiting_ticket"
    COMPLETED = "completed"


class HITLStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class TicketStatus(StrEnum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"


class NodeDirective(StrEnum):
    CONTINUE = "continue"
    WAIT_EXTERNAL = "wait_external"
    WAIT_HITL = "wait_hitl"
    COMPLETE = "complete"
