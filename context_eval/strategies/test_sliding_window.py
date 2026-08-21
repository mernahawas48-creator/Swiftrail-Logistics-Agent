import pytest

from context_eval.strategies.sliding_window import SlidingWindow


def test_sliding_window_keeps_only_most_recent_messages():
    messages = [
        {"role": "employee", "content": "one"},
        {"role": "agent", "content": "two"},
        {"role": "employee", "content": "three"},
    ]

    assert SlidingWindow(max_messages=2).apply(messages) == messages[-2:]


def test_sliding_window_rejects_invalid_size():
    with pytest.raises(ValueError, match="at least 1"):
        SlidingWindow(max_messages=0)
