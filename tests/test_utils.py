
import unittest
from unittest.mock import AsyncMock, MagicMock
from open_deep_research.utils import summarize_webpage

class TestSummarizeWebpage(unittest.IsolatedAsyncioTestCase):
    async def test_summarize_webpage_skips_short_content(self):
        # Setup mock model
        mock_model = AsyncMock()

        # Configure the mock to return a structured output
        mock_structured_output = MagicMock()
        mock_structured_output.summary = "Summarized content"
        mock_structured_output.key_excerpts = ["excerpt1", "excerpt2"]

        # Mock the chain of calls: model.ainvoke(...) -> summary object
        mock_model.ainvoke.return_value = mock_structured_output

        # Case 1: Short content - should NOT call model (optimization to be implemented)
        # Using a very short string clearly below any reasonable threshold (e.g. 3000 chars)
        short_content = "This is a very short webpage content."
        result = await summarize_webpage(mock_model, short_content)

        # Verify optimization: Model should NOT be called for short content
        self.assertFalse(mock_model.ainvoke.called, "Model should NOT be called for short content")

        # Case 2: Long content - should call model
        mock_model.reset_mock()
        long_content = "A" * 5000  # Above 3000 chars
        await summarize_webpage(mock_model, long_content)

        self.assertTrue(mock_model.ainvoke.called, "Model SHOULD be called for long content")

if __name__ == "__main__":
    unittest.main()
