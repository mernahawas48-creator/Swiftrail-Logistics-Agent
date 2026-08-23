from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp_server.agent_registry import AgentRegistry
from mcp_server.runtime_tool_manager import RuntimeToolManager
from state_graph.core.types import TicketStatus
from state_graph.graph_2.live import build_live_service


class AdminService:
    """Backend facade used by the admin UI.

    The UI should call this service instead of mutating registries, RAG files,
    or graph checkpoints directly. That keeps the platform connected to the
    same live MCP/RAG/state components used by the agents.
    """

    def __init__(
        self,
        tool_manager: RuntimeToolManager,
        agents: AgentRegistry,
        rag: Any | None = None,
        graph_factory: Callable[[], Any] | None = None,
    ):
        self.tool_manager = tool_manager
        self.agents = agents
        self.rag = rag
        self.graph_factory = graph_factory or build_live_service

    def agents_with_tools(self) -> list[dict[str, Any]]:
        return [
            {
                **record,
                "tools": sorted(
                    self.tool_manager.registry.tools_for(record["agent_id"])
                ),
            }
            for record in self.agents.as_dicts()
        ]

    def register_agent(
        self,
        agent_id: str,
        name: str,
        kind: str,
        tools: set[str] | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        """Register an agent and its initial runtime MCP permissions."""
        if self.agents.get(agent_id) is not None:
            raise ValueError(f"Agent already exists: {agent_id}")
        self.agents.register(agent_id, name, kind, **metadata)
        return self.tool_manager.register_agent(agent_id, tools)

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent and revoke its runtime tool permissions."""
        self._require_agent(agent_id)
        self.tool_manager.unregister_agent(agent_id)
        self.agents._agents.pop(agent_id, None)

    async def add_tool(self, agent_id: str, tool_name: str) -> dict[str, Any]:
        self._require_agent(agent_id)
        return await self.tool_manager.add_tool_to_agent(agent_id, tool_name)

    async def remove_tool(self, agent_id: str, tool_name: str) -> dict[str, Any]:
        self._require_agent(agent_id)
        return await self.tool_manager.remove_tool_from_agent(agent_id, tool_name)

    def list_rag_documents(self) -> list[dict[str, Any]]:
        if self.rag is None:
            return []
        return self.rag.list_documents()

    def add_rag_document(self, metadata: dict[str, Any], text: str) -> dict[str, Any]:
        return self._require_rag().add_document(metadata, text)

    def remove_rag_document(self, doc_id: str) -> None:
        self._require_rag().remove_document(doc_id)

    def update_rag_document(self, doc_id: str, text: str) -> dict[str, Any]:
        return self._require_rag().update_document(doc_id, text)

    def reindex_rag(self) -> dict[str, Any]:
        return self._require_rag().reindex()

    def pending_hitl_tasks(self) -> list[dict[str, Any]]:
        return self._graph().pending_hitl_tasks()

    def pending_failure_tickets(self) -> list[dict[str, Any]]:
        return self._graph().tickets(TicketStatus.OPEN)

    def resolve_hitl(
        self,
        task_id: str,
        decision: str,
        note: str | None = None,
        admin_employee_id: int = 3,
    ):
        return self._graph().resolve_hitl(
            task_id,
            approved=decision == "approve",
            note=note or "Resolved through the admin service.",
            admin_employee_id=admin_employee_id,
        )

    def investigate_failure(self, ticket_id: str):
        return self._graph().investigate_ticket(ticket_id)

    def resolve_failure(self, ticket_id: str, note: str):
        return self._graph().resolve_ticket(ticket_id, resolution_note=note)

    def _graph(self):
        return self.graph_factory()

    def _require_agent(self, agent_id: str) -> None:
        if self.agents.get(agent_id) is None:
            raise KeyError(f"Unknown agent: {agent_id}")

    def _require_rag(self):
        if self.rag is None:
            raise RuntimeError("RAG management is not configured for this platform")
        return self.rag
