import pytest

from context_eval.strategies.zone_based_pruning import ZoneBasedPruning


def test_zone_pruning_keeps_system_zone_and_recent_messages():
    system = {"role": "system", "content": "ground all answers"}
    messages = [
        system,
        {"role": "employee", "content": "old"},
        {"role": "agent", "content": "middle"},
        {"role": "employee", "content": "recent"},
    ]

    result = ZoneBasedPruning(keep_recent_messages=2).apply(messages)

    assert result == [system, *messages[-2:]]


def test_zone_pruning_rejects_invalid_size():
    with pytest.raises(ValueError, match="at least 1"):
        ZoneBasedPruning(keep_recent_messages=0)
