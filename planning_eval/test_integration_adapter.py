from dataclasses import dataclass

from planning.shipment_exception_agent import ShipmentExceptionResolutionAgent


@dataclass
class FakeLLM:
    responses: list[str]

    def invoke(self, messages, temperature=0.2):
        if not self.responses:
            raise RuntimeError("no scripted response")
        return type("Response", (), {"content": self.responses.pop(0)})()


def test_extract_ids():
    assert ShipmentExceptionResolutionAgent.extract_ids(
        "Resolve shipment 3 for employee 1"
    ) == (3, 1)


def test_extract_ids_requires_both_ids():
    try:
        ShipmentExceptionResolutionAgent.extract_ids("Resolve shipment 3")
    except ValueError as exc:
        assert "shipment and employee IDs" in str(exc)
    else:
        raise AssertionError("expected ValueError")
