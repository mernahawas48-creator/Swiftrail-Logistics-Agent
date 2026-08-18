from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRecord:
    agent_id: str
    name: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, AgentRecord] = {}

    def register(self, agent_id: str, name: str, kind: str, **metadata: Any) -> AgentRecord:
        record = AgentRecord(agent_id, name, kind, metadata)
        self._agents[agent_id] = record
        return record

    def get(self, agent_id: str) -> AgentRecord | None:
        return self._agents.get(agent_id)

    def list(self) -> list[AgentRecord]:
        return list(self._agents.values())

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {"agent_id": a.agent_id, "name": a.name, "kind": a.kind, **a.metadata}
            for a in self._agents.values()
        ]
