
import unittest
from unittest.mock import patch, MagicMock
from open_deep_research.utils import _get_summarization_model

class TestUtilsPerf(unittest.TestCase):
    def test_get_summarization_model_caching(self):
        """
        Verify that _get_summarization_model returns the same instance for identical arguments
        and only calls init_chat_model once.
        """
        # Define test arguments
        model_name = "test-model"
        max_tokens = 100
        api_key = "test-key"
        max_retries = 2

        # Mock init_chat_model to track calls
        with patch('open_deep_research.utils.init_chat_model') as mock_init:
            # Mock the returned model object
            mock_model_instance = MagicMock()
            mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_model_instance

            # First call - should trigger initialization
            model1 = _get_summarization_model(model_name, max_tokens, api_key, max_retries)

            # Second call - should return cached instance
            model2 = _get_summarization_model(model_name, max_tokens, api_key, max_retries)

            # Verify object identity
            self.assertIs(model1, model2, "Cached model instance should be identical")

            # Verify init_chat_model was called exactly once
            mock_init.assert_called_once()

            # Verify call arguments
            mock_init.assert_called_with(
                model=model_name,
                max_tokens=max_tokens,
                api_key=api_key,
                tags=["langsmith:nostream"]
            )

    def test_get_summarization_model_different_args(self):
        """
        Verify that different arguments create different model instances.
        """
        with patch('open_deep_research.utils.init_chat_model') as mock_init:
            # Setup mocks for two different calls
            mock_model1 = MagicMock()
            mock_model2 = MagicMock()

            # Configure side_effect to return different mocks on subsequent calls
            # Note: The chain .with_structured_output().with_retry() needs to be mocked
            # This is tricky with side_effect on the end of a chain, so we'll just check call counts

            # First call
            _get_summarization_model("model-A", 100, "key", 1)

            # Second call with different model name
            _get_summarization_model("model-B", 100, "key", 1)

            # Should have called init twice
            self.assertEqual(mock_init.call_count, 2)

if __name__ == "__main__":
    unittest.main()
