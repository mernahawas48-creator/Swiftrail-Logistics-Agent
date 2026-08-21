from memory.episodic_store import Episode
from memory.semantic_store import SemanticFact
from memory.verification import MemoryVerifier


def _fact(**overrides):
    base = {
        "id": 1, "customer_id": 12, "fact_key": "customer_risk_level",
        "fact_value": "high_risk", "version": 2, "status": "active",
        "source_episode_id": 1, "conflict_reason": None, "superseded_by": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "expires_at": "2099-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return SemanticFact(**base)


def _episode(**overrides):
    base = {
        "id": 1, "customer_id": 12, "event_type": "credit_hold_placed",
        "content": {"event_type": "credit_hold_placed", "severity": "severe"},
        "source_session_id": "sess-1", "reason": "90+ days overdue on shipment 512",
        "created_at": "2026-01-01T00:00:00+00:00", "consolidated": False,
    }
    base.update(overrides)
    return Episode(**base)


def test_relevant_and_fresh_fact_passes():
    verifier = MemoryVerifier()
    summary = verifier.verify(
        "what is the credit risk level for this customer",
        [_fact()],
    )
    assert summary.passed is True
    assert summary.relevant is True
    assert summary.fresh is True


def test_off_topic_query_fails_relevance():
    verifier = MemoryVerifier()
    summary = verifier.verify(
        "what is the weather forecast for tomorrow",
        [_fact()],
    )
    assert summary.relevant is False
    assert summary.passed is False


def test_superseded_fact_fails_freshness_when_its_the_only_item():
    verifier = MemoryVerifier()
    stale_fact = _fact(status="superseded")
    summary = verifier.verify(
        "what is the credit risk level for this customer",
        [stale_fact],
    )
    assert summary.fresh is False
    assert summary.passed is False


def test_expired_fact_fails_freshness():
    verifier = MemoryVerifier()
    expired_fact = _fact(status="expired")
    summary = verifier.verify(
        "what is the credit risk level for this customer",
        [expired_fact],
    )
    assert summary.fresh is False


def test_no_items_fails_both_checks():
    verifier = MemoryVerifier()
    summary = verifier.verify("any open credit holds", [])
    assert summary.relevant is False
    assert summary.fresh is False
    assert summary.passed is False


def test_relevant_episode_passes():
    verifier = MemoryVerifier()
    summary = verifier.verify(
        "any credit hold issues on this shipment",
        [_episode()],
    )
    assert summary.passed is True


def test_mixed_items_pass_if_at_least_one_is_fresh():
    verifier = MemoryVerifier()
    summary = verifier.verify(
        "what is the credit risk level for this customer",
        [_fact(status="superseded", version=1), _fact(status="active", version=2)],
    )
    assert summary.fresh is True
    assert summary.passed is True
