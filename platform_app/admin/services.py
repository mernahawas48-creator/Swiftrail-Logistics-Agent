from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp_server.agent_registry import AgentRegistry
from mcp_server.runtime_tool_manager import RuntimeToolManager
from state_graph.graph_2.graph import RateExceptionGraph


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
        graph_factory: Callable[[], RateExceptionGraph] | None = None,
    ):
        self.tool_manager = tool_manager
        self.agents = agents
        self.rag = rag
        self.graph_factory = graph_factory or RateExceptionGraph

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
        return self._graph().checkpoints.list_tasks("hitl", "open")

    def pending_failure_tickets(self) -> list[dict[str, Any]]:
        return self._graph().checkpoints.list_tasks("failure", "open")

    def resolve_hitl(
        self,
        run_id: str,
        decision: str,
        note: str | None = None,
    ):
        return self._graph().resume(
            run_id,
            admin_decision=decision,
            admin_note=note,
        )

    def resolve_failure(self, run_id: str):
        return self._graph().resolve_failure(run_id)

    def _graph(self) -> RateExceptionGraph:
        return self.graph_factory()

    def _require_agent(self, agent_id: str) -> None:
        if self.agents.get(agent_id) is None:
            raise KeyError(f"Unknown agent: {agent_id}")

    def _require_rag(self):
        if self.rag is None:
            raise RuntimeError("RAG management is not configured for this platform")
        return self.rag
