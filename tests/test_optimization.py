
import unittest
from unittest.mock import AsyncMock

from langchain_core.language_models import BaseChatModel

from open_deep_research.state import Summary
from open_deep_research.utils import summarize_webpage


class TestSummarizeWebpage(unittest.IsolatedAsyncioTestCase):
    async def test_summarize_webpage_calls_llm_for_short_content(self):
        """
        Verify that summarize_webpage does NOT call the LLM for short content.
        """
        mock_model = AsyncMock(spec=BaseChatModel)
        # Mock the return value of ainvoke
        mock_model.ainvoke.return_value = Summary(
            summary="This is a summary.",
            key_excerpts="These are key excerpts."
        )

        short_content = "This is a short content."
        result = await summarize_webpage(mock_model, short_content)

        # Verify LLM was NOT called
        mock_model.ainvoke.assert_not_called()
        self.assertIn(f"<summary>\n{short_content}\n</summary>", result)

    async def test_summarize_webpage_calls_llm_for_long_content(self):
        """
        Verify that LLM is called for long content.
        """
        mock_model = AsyncMock(spec=BaseChatModel)
        mock_model.ainvoke.return_value = Summary(
            summary="Long summary.",
            key_excerpts="Long excerpts."
        )

        long_content = "a" * 5000
        result = await summarize_webpage(mock_model, long_content)

        mock_model.ainvoke.assert_called_once()
        self.assertIn("<summary>\nLong summary.\n</summary>", result)

if __name__ == "__main__":
    unittest.main()
