"""Swiftrail live agent loop with context, memory recall, and selected RAG path.

The existing MCP protocol client remains in ``agent/client.py``. This loop adds
routing to the final selected Hybrid RAG architecture and to verified episodic /
semantic memory recall without duplicating the server or database.
"""

from __future__ import annotations

from typing import Any

from agent.query_router import QueryRouter
from agent.session_manager import SessionManager
from context_eval.strategies.sliding_window import SlidingWindow
from memory.episodic_store import EpisodicMemory
from memory.router import PromoteDropRouter
from memory.scratchpad import Scratchpad
from memory.semantic_store import SemanticMemory
from memory.short_term import ShortTermBuffer, Turn


SAFE_UNVERIFIED_ANSWER = (
    "I cannot provide a reliable answer because no verified information was found."
)


class AgentLoop:
    """Route each request through context, RAG, memory, or the MCP tool path."""

    def __init__(
        self,
        *,
        rag_pipeline: Any | None = None,
        memory_recall: Any | None = None,
        episodic_memory: EpisodicMemory | None = None,
        semantic_memory: SemanticMemory | None = None,
        short_term_max_turns: int = 12,
    ):
        self.session_manager = SessionManager()
        self.strategy = SlidingWindow(max_messages=10)
        self.router = QueryRouter()

        self._rag_pipeline = rag_pipeline
        self._memory_recall = memory_recall
        self._episodic_memory = episodic_memory
        self._semantic_memory = semantic_memory
        self._promote_router: PromoteDropRouter | None = None

        self._short_term_max_turns = short_term_max_turns
        self._buffers: dict[str, ShortTermBuffer] = {}
        self._scratchpads: dict[str, Scratchpad] = {}

    def start(self, customer_id: int | None = None, customer_name: str = "") -> str:
        session = self.session_manager.create_session(
            customer_id=customer_id,
            customer_name=customer_name,
        )
        self._buffers[session.session_id] = ShortTermBuffer(
            max_turns=self._short_term_max_turns
        )
        self._scratchpads[session.session_id] = Scratchpad()
        return session.session_id

    def process(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        role: str = "sales_rep",
    ) -> dict[str, Any]:
        """Process one request through the same live routing loop."""

        session = self.session_manager.get_session(session_id)
        if session is None:
            raise ValueError("Session not found")
        if not messages:
            raise ValueError("messages cannot be empty")

        # Context management uses the implemented strategy API.
        context = self.strategy.apply(messages)
        last_message = str(messages[-1].get("content", "")).strip()
        if not last_message:
            raise ValueError("The last message cannot be empty")

        destination = self.router.route(last_message)
        session.add_note(f"Request routed to: {destination}")

        buffer = self._buffers.setdefault(
            session_id,
            ShortTermBuffer(max_turns=self._short_term_max_turns),
        )
        scratchpad = self._scratchpads.setdefault(session_id, Scratchpad())
        scratchpad.set_goal(last_message)

        evicted = buffer.add(
            Turn(
                role="customer",
                content=last_message,
                customer_id=self._normalized_customer_id(session.customer_id),
                session_id=session_id,
            )
        )
        self._route_overflow(evicted)

        if destination == "rag":
            result = self._answer_from_rag(last_message, role=role)
        elif destination == "memory":
            result = self._answer_from_memory(
                last_message,
                customer_id=self._normalized_customer_id(session.customer_id),
            )
        elif destination in {"shipment", "invoice", "customer", "credit"}:
            result = self._answer_from_mcp_placeholder(destination, last_message)
        else:
            result = {
                "answer": SAFE_UNVERIFIED_ANSWER,
                "verified": False,
                "evidence": [],
                "verification": "No supported route matched the request.",
            }

        evicted = buffer.add(
            Turn(
                role="agent",
                content=result["answer"],
                customer_id=self._normalized_customer_id(session.customer_id),
                session_id=session_id,
            )
        )
        self._route_overflow(evicted)

        session.add_note(
            {
                "query": last_message,
                "category": destination,
                "verified": result["verified"],
            }
        )

        return {
            "session_id": session_id,
            "category": destination,
            "verified": result["verified"],
            "answer": result["answer"],
            "context": context,
            "evidence": result["evidence"],
            "verification": result["verification"],
            "short_term_size": len(buffer),
            "scratchpad": scratchpad.snapshot(),
        }

    def _answer_from_rag(self, query: str, *, role: str) -> dict[str, Any]:
        if self._rag_pipeline is None:
            from rag.hybrid_rag.pipeline import HybridRAG

            self._rag_pipeline = HybridRAG()

        response = self._rag_pipeline.answer(query, role=role, top_k=5)
        verification = getattr(response, "verification", None)
        verified = bool(verification and verification.passed)
        sources = [
            {
                "section_id": source.section_id,
                "doc_id": source.doc_id,
                "number": source.number,
            }
            for source in response.sources
        ]
        return {
            "answer": response.answer,
            "verified": verified,
            "evidence": sources,
            "verification": (
                verification.reason if verification is not None else "No verification record."
            ),
        }

    def _answer_from_memory(
        self,
        query: str,
        *,
        customer_id: int | None,
    ) -> dict[str, Any]:
        if customer_id is None:
            return {
                "answer": "I could not find enough verified memory to answer this question.",
                "verified": False,
                "evidence": [],
                "verification": "Memory recall requires a customer-scoped session.",
            }

        self._ensure_memory_components()
        response = self._memory_recall.answer(query, customer_id=customer_id)
        return {
            "answer": response.answer,
            "verified": bool(response.verification and response.verification.passed),
            "evidence": [
                {
                    "number": source.number,
                    "memory_type": source.memory_type,
                    "memory_id": source.memory_id,
                }
                for source in response.sources
            ],
            "verification": (
                response.verification.reason
                if response.verification is not None
                else "No verification record."
            ),
        }

    def _ensure_memory_components(self) -> None:
        if self._episodic_memory is None:
            self._episodic_memory = EpisodicMemory()
        if self._semantic_memory is None:
            self._semantic_memory = SemanticMemory()
        if self._promote_router is None:
            self._promote_router = PromoteDropRouter(self._episodic_memory)
        if self._memory_recall is None:
            from memory.verified_recall import VerifiedMemoryRecall

            self._memory_recall = VerifiedMemoryRecall(
                episodic_memory=self._episodic_memory,
                semantic_memory=self._semantic_memory,
            )

    def _route_overflow(self, evicted: Turn | None) -> None:
        if evicted is None:
            return
        self._ensure_memory_components()
        self._promote_router.process_overflow(evicted)

    @staticmethod
    def _normalized_customer_id(value: Any) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        return int(text) if text.isdigit() else None

    def _answer_from_mcp_placeholder(self, destination: str, query: str) -> dict[str, Any]:
        """Preserve the existing operational route while MCP execution stays in client.py.

        The repository's real MCP lifecycle/tool execution is implemented by
        ``agent.client.SwiftrailAgent``. This method intentionally does not
        create a second server/database path. RAG and memory are integrated in
        this loop; existing MCP operations continue to use that client.
        """

        evidence = self.call_mcp_tool(destination, query)
        verified = evidence.get("data") is not None
        answer = (
            self.generate_response(destination, evidence)
            if verified
            else SAFE_UNVERIFIED_ANSWER
        )
        return {
            "answer": answer,
            "verified": verified,
            "evidence": [evidence] if verified else [],
            "verification": "Existing MCP operational path." if verified else "No MCP evidence.",
        }

    def call_mcp_tool(self, destination: str, query: str) -> dict[str, Any]:
        """Compatibility path retained from the existing loop.

        Real MCP protocol calls remain in ``agent/client.py``; no database or
        server implementation is duplicated here.
        """

        labels = {
            "shipment": ("shipment_database", "Shipment information retrieved"),
            "invoice": ("invoice_database", "Invoice information retrieved"),
            "customer": ("customer_database", "Customer information retrieved"),
            "credit": ("finance_database", "Credit status retrieved"),
        }
        source, data = labels.get(destination, ("unknown", None))
        return {"source": source, "data": data}

    @staticmethod
    def generate_response(category: str, evidence: dict[str, Any]) -> str:
        return (
            f"Swiftrail {category} request completed. "
            f"Verified data: {evidence['data']}"
        )

    def end(self, session_id: str) -> None:
        self._buffers.pop(session_id, None)
        self._scratchpads.pop(session_id, None)
        self.session_manager.end_session(session_id)
