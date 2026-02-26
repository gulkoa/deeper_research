import unittest
from unittest.mock import MagicMock, patch
from open_deep_research.utils import _get_summarization_model, _SUMMARIZATION_MODEL_CACHE

class TestUtilsPerf(unittest.TestCase):
    def setUp(self):
        # Clear cache before each test
        _SUMMARIZATION_MODEL_CACHE.clear()

    @patch("open_deep_research.utils.init_chat_model")
    def test_summarization_model_caching(self, mock_init_chat_model):
        # Configure mock to return a NEW mock chain each time called
        def create_mock_chain(*args, **kwargs):
            mock_model = MagicMock()
            mock_structured = MagicMock()
            mock_retry = MagicMock()
            mock_model.with_structured_output.return_value = mock_structured
            mock_structured.with_retry.return_value = mock_retry
            return mock_model

        mock_init_chat_model.side_effect = create_mock_chain

        # Call 1: Should initialize
        model1 = _get_summarization_model("openai:gpt-4o-mini", 1000, "fake-key", 3)

        self.assertEqual(mock_init_chat_model.call_count, 1)

        # Call 2: Should use cache (same args)
        model2 = _get_summarization_model("openai:gpt-4o-mini", 1000, "fake-key", 3)
        self.assertIs(model2, model1)
        self.assertEqual(mock_init_chat_model.call_count, 1)  # Count remains 1

        # Call 3: Different model -> Should initialize again
        model3 = _get_summarization_model("anthropic:claude-3-haiku", 1000, "fake-key", 3)
        self.assertIsNot(model3, model1)
        self.assertEqual(mock_init_chat_model.call_count, 2)

        # Call 4: Different API key -> Should initialize again
        model4 = _get_summarization_model("openai:gpt-4o-mini", 1000, "other-key", 3)
        self.assertIsNot(model4, model1)
        self.assertEqual(mock_init_chat_model.call_count, 3)

if __name__ == '__main__':
    unittest.main()
