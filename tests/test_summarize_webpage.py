
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from open_deep_research.utils import summarize_webpage

# Mock the Summary model which is used in the function
class Summary(MagicMock):
    summary: str
    key_excerpts: list

async def _test_summarize_webpage_skips_short_content():
    # Setup mock model
    mock_model = AsyncMock()

    # Configure the mock to return a structured output
    mock_structured_output = MagicMock()
    mock_structured_output.summary = "Summarized content"
    mock_structured_output.key_excerpts = ["excerpt1", "excerpt2"]

    # Mock the chain of calls: model.ainvoke(...) -> summary object
    mock_model.ainvoke.return_value = mock_structured_output

    # Case 1: Short content - should NOT call model (optimization to be implemented)
    short_content = "This is a very short webpage content."
    result = await summarize_webpage(mock_model, short_content)

    # Before fix: it calls model.
    # After fix: it should NOT call model.
    # We assert that it DOES call the model now (to prove need for opt), or we can just assert expected behavior after opt.
    # Since I'm "Bolt", I'll write the test for the DESIRED behavior.

    assert not mock_model.ainvoke.called, "Model should NOT be called for short content"

    mock_model.reset_mock()

    # Case 2: Long content - should call model
    long_content = "A" * 5000
    await summarize_webpage(mock_model, long_content)

    assert mock_model.ainvoke.call_count >= 1, "Model SHOULD be called for long content"

def test_summarize_webpage_skips_short_content():
    asyncio.run(_test_summarize_webpage_skips_short_content())
