from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from state_graph.core.state import SharedGraphState
from state_graph.core.types import NodeDirective


@dataclass(frozen=True, slots=True)
class NodeContext:
    services: dict[str, Any] = field(default_factory=dict)

    def require(self, name: str) -> Any:
        try:
            return self.services[name]
        except KeyError as exc:
            raise RuntimeError(f"Node service is not configured: {name}") from exc


@dataclass(frozen=True, slots=True)
class NodeResult:
    next_node: str
    updates: dict[str, Any] = field(default_factory=dict)
    directive: NodeDirective = NodeDirective.CONTINUE
    reason: str | None = None
    request: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["directive"] = self.directive.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NodeResult:
        data = dict(value)
        data["directive"] = NodeDirective(data["directive"])
        return cls(**data)


class GraphNode(Protocol):
    name: str

    def run(self, state: SharedGraphState, context: NodeContext) -> NodeResult: ...


@dataclass(slots=True)
class FunctionNode:
    name: str
    function: Callable[[SharedGraphState, NodeContext], NodeResult]

    def run(self, state: SharedGraphState, context: NodeContext) -> NodeResult:
        return self.function(state, context)


@dataclass(slots=True)
class ExternalWaitNode:
    """Expected pause while a customer or external system responds."""

    name: str
    resume_node: str
    reason: str
    request_builder: Callable[[SharedGraphState], dict[str, Any]]

    def run(self, state: SharedGraphState, context: NodeContext) -> NodeResult:
        del context
        return NodeResult(
            next_node=self.resume_node,
            directive=NodeDirective.WAIT_EXTERNAL,
            reason=self.reason,
            request=self.request_builder(state),
        )


@dataclass(slots=True)
class HITLNode:
    """Explicit admin-only pause; never routed through failure handling."""

    name: str
    resume_node: str
    reason_builder: Callable[[SharedGraphState], str]
    request_builder: Callable[[SharedGraphState], dict[str, Any]]

    def run(self, state: SharedGraphState, context: NodeContext) -> NodeResult:
        del context
        return NodeResult(
            next_node=self.resume_node,
            directive=NodeDirective.WAIT_HITL,
            reason=self.reason_builder(state),
            request=self.request_builder(state),
        )
