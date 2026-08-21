from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GraphDefinition:
    """Validated nodes and allowed transitions for one business graph."""

    name: str
    start_node: str
    nodes: dict[str, Any]
    transitions: dict[str, frozenset[str]]
    end_node: str = "END"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Graph name cannot be empty.")
        if self.start_node not in self.nodes:
            raise ValueError("start_node must reference a registered node.")

        known = set(self.nodes) | {self.end_node}
        for source, targets in self.transitions.items():
            if source not in self.nodes:
                raise ValueError(f"Unknown transition source: {source}")
            unknown = set(targets) - known
            if unknown:
                raise ValueError(
                    f"Transition {source} references unknown targets: {sorted(unknown)}"
                )

    def allows(self, source: str, target: str) -> bool:
        return target in self.transitions.get(source, frozenset())

    def require_transition(self, source: str, target: str) -> None:
        if not self.allows(source, target):
            raise ValueError(
                f"Graph {self.name} does not allow transition {source} -> {target}."
            )

    def has_cycle(self) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited or node == self.end_node:
                return False
            visiting.add(node)
            for target in self.transitions.get(node, frozenset()):
                if visit(target):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in self.nodes)
