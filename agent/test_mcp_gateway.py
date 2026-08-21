import pytest

from agent.mcp_gateway import MCPGatewayError, StdioMCPGateway


def test_shipment_route_maps_query_id_to_status_tool():
    tool, request = StdioMCPGateway._build_request(
        "shipment",
        "show shipment #500",
        session_id="session-001",
        customer_id=3,
    )
    assert tool == "get_shipment_status"
    assert request == {"session_id": "session-001", "shipment_id": 500}


@pytest.mark.parametrize(
    ("destination", "expected_tool"),
    [
        ("customer", "search_customer"),
        ("invoice", "list_customer_invoices"),
        ("credit", "list_customer_credit_holds"),
    ],
)
def test_customer_scoped_routes_use_session_customer(destination, expected_tool):
    tool, request = StdioMCPGateway._build_request(
        destination,
        "show status",
        session_id="session-001",
        customer_id=3,
    )
    assert tool == expected_tool
    assert request == {"session_id": "session-001", "customer_id": 3}


def test_shipment_route_requires_an_identifier():
    with pytest.raises(MCPGatewayError, match="shipment ID"):
        StdioMCPGateway._build_request(
            "shipment",
            "show the shipment",
            session_id="session-001",
            customer_id=3,
        )
