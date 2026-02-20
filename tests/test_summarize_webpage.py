from unittest.mock import AsyncMock, MagicMock

import pytest

from open_deep_research.utils import summarize_webpage


@pytest.mark.asyncio
async def test_summarize_webpage_skips_model_for_short_content():
    # Mock the model
    mock_model = AsyncMock()
    mock_response = MagicMock()
    mock_response.summary = "Summary"
    mock_response.key_excerpts = "Excerpts"
    mock_model.ainvoke.return_value = mock_response

    # Short content (< 3000 chars)
    short_content = "This is a very short webpage." * 10
    assert len(short_content) < 3000

    # Call summarize_webpage
    result = await summarize_webpage(mock_model, short_content)

    # Assert ainvoke was NOT called
    mock_model.ainvoke.assert_not_called()
    assert result == short_content

@pytest.mark.asyncio
async def test_summarize_webpage_calls_model_for_long_content():
    # Mock the model
    mock_model = AsyncMock()
    mock_response = MagicMock()
    mock_response.summary = "Summary"
    mock_response.key_excerpts = "Excerpts"
    mock_model.ainvoke.return_value = mock_response

    # Long content (> 3000 chars)
    long_content = "This is a long webpage." * 300
    assert len(long_content) > 3000

    # Call summarize_webpage
    result = await summarize_webpage(mock_model, long_content)

    # Assert ainvoke was called
    mock_model.ainvoke.assert_called()
    assert "<summary>" in result
