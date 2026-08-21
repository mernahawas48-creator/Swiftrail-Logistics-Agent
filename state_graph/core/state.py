from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from state_graph.core.types import RunStatus


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class SharedGraphState:
    """Serializable state envelope shared by every Swiftrail graph."""

    run_id: str
    graph_name: str
    current_node: str
    status: RunStatus = RunStatus.RUNNING
    revision: int = 0
    input_data: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    completed_nodes: list[str] = field(default_factory=list)
    transition_history: list[dict[str, Any]] = field(default_factory=list)
    resume_node: str | None = None
    failed_node: str | None = None
    hitl_task_id: str | None = None
    ticket_id: str | None = None
    error: dict[str, Any] | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def apply_updates(self, updates: dict[str, Any]) -> None:
        self.data.update(updates)
        self.updated_at = utc_now()

    def record_transition(self, source: str, target: str, event: str) -> None:
        self.transition_history.append(
            {
                "source": source,
                "target": target,
                "event": event,
                "at": utc_now(),
            }
        )
        self.current_node = target
        self.revision += 1
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SharedGraphState:
        data = dict(value)
        data["status"] = RunStatus(data["status"])
        return cls(**data)
