from agent.session_manager import SessionManager


def test_session_lifecycle():
    manager = SessionManager()

    session = manager.create_session(
        customer_id="CUST001",
        customer_name="ABC Logistics",
    )
    session.add_note("Customer asked about shipment 500.")

    assert manager.get_session(session.session_id) is session
    assert session.customer_id == "CUST001"
    assert session.customer_name == "ABC Logistics"
    assert session.scratchpad == ["Customer asked about shipment 500."]
    assert session.session_id in manager.list_sessions()

    manager.end_session(session.session_id)
    assert manager.get_session(session.session_id) is None
