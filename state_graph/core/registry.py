from __future__ import annotations

from state_graph.core.transitions import GraphDefinition


class GraphRegistry:
    """Runtime registry shared by the agent and platform layers."""

    def __init__(self) -> None:
        self._graphs: dict[str, GraphDefinition] = {}

    def register(self, graph: GraphDefinition) -> None:
        if graph.name in self._graphs:
            raise ValueError(f"Graph {graph.name} is already registered.")
        self._graphs[graph.name] = graph

    def get(self, name: str) -> GraphDefinition:
        try:
            return self._graphs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown graph: {name}") from exc

    def list(self) -> list[str]:
        return sorted(self._graphs)
