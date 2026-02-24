import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to python path just in case, though usually handled by pytest/runner
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# We import inside test methods or after ensuring utils has the function to avoid import errors during collection if feasible
# But since this is a unit test for a specific module, we expect it to exist.
# However, I am adding the test BEFORE the implementation.
# So I will catch ImportError to allow the test file to be created without crashing pytest collection immediately
# if I were running it now. But I'll implement the function right after.

try:
    from open_deep_research.utils import _get_summarization_model, _SUMMARIZATION_MODEL_CACHE
except ImportError:
    # Function not implemented yet
    pass

class TestUtilsPerf(unittest.TestCase):
    def setUp(self):
        # Clear cache before each test
        if '_SUMMARIZATION_MODEL_CACHE' in globals():
             _SUMMARIZATION_MODEL_CACHE.clear()

    @patch("open_deep_research.utils.init_chat_model")
    def test_get_summarization_model_caching(self, mock_init):
        # Ensure we have the function imported
        from open_deep_research.utils import _get_summarization_model, _SUMMARIZATION_MODEL_CACHE

        # Clear cache
        _SUMMARIZATION_MODEL_CACHE.clear()

        # Setup mock return
        mock_model = MagicMock()
        mock_init.return_value = mock_model

        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured

        mock_retry = MagicMock()
        mock_structured.with_retry.return_value = mock_retry

        # Call first time
        model1 = _get_summarization_model("model1", 1000, "key1", 3)

        # Verify call chain
        mock_init.assert_called_with(
            model="model1",
            max_tokens=1000,
            api_key="key1",
            tags=["langsmith:nostream"]
        )
        mock_model.with_structured_output.assert_called()
        mock_structured.with_retry.assert_called_with(stop_after_attempt=3)

        # Call second time with same args
        model2 = _get_summarization_model("model1", 1000, "key1", 3)

        # Should be same object
        self.assertIs(model1, model2)

        # init_chat_model should be called only once
        mock_init.assert_called_once()

        # Call with different args
        model3 = _get_summarization_model("model2", 1000, "key1", 3)

        # Should call again
        self.assertEqual(mock_init.call_count, 2)
