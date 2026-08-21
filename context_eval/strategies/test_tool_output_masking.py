import pytest

from context_eval.strategies.tool_output_masking import ToolOutputMasking


def test_tool_output_masking_preserves_only_requested_recent_outputs():
    messages = [
        {"role": "tool", "content": "old result"},
        {"role": "employee", "content": "continue"},
        {"role": "tool", "content": "new result"},
    ]

    result = ToolOutputMasking(keep_last_tool_outputs=1).apply(messages)

    assert result[0]["content"] == "[TOOL OUTPUT MASKED]"
    assert result[2]["content"] == "new result"
    assert messages[0]["content"] == "old result"


def test_zero_masks_all_tool_outputs():
    messages = [{"role": "tool", "content": "result"}]
    result = ToolOutputMasking(keep_last_tool_outputs=0).apply(messages)
    assert result[0]["content"] == "[TOOL OUTPUT MASKED]"


def test_tool_output_masking_rejects_negative_count():
    with pytest.raises(ValueError, match="cannot be negative"):
        ToolOutputMasking(keep_last_tool_outputs=-1)
