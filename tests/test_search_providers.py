import os
import unittest
from langchain_core.messages import HumanMessage
from open_deep_research.deep_researcher import deep_researcher

class TestDeepResearchProviders(unittest.IsolatedAsyncioTestCase):

    async def run_research(self, search_api, research_model, other_models_provider="openai", max_tokens=10000):

        # Default support models to OpenAI GPT-4o-mini
        support_model = "openai:gpt-4o-mini"
        if other_models_provider == "anthropic":
            support_model = "anthropic:claude-3-haiku-20240307"

        # Explicitly set environment variables if max_tokens is set to a low value
        # This is a workaround because sometimes configurable doesn't seem to propagate correctly or defaults take precedence
        if max_tokens < 10000:
            os.environ["FINAL_REPORT_MODEL_MAX_TOKENS"] = str(max_tokens)
            os.environ["RESEARCH_MODEL_MAX_TOKENS"] = str(max_tokens)
            os.environ["SUMMARIZATION_MODEL_MAX_TOKENS"] = str(max_tokens)
            os.environ["COMPRESSION_MODEL_MAX_TOKENS"] = str(max_tokens)

        config = {
            "configurable": {
                "search_api": search_api,
                "research_model": research_model,
                "research_model_max_tokens": max_tokens,
                "max_researcher_iterations": 1,
                "max_concurrent_research_units": 1,
                "max_structured_output_retries": 1,
                "allow_clarification": False,
                "thread_id": "test_thread",
                "summarization_model": support_model,
                "summarization_model_max_tokens": max_tokens,
                "compression_model": support_model,
                "compression_model_max_tokens": max_tokens,
                "final_report_model": support_model,
                "final_report_model_max_tokens": max_tokens,
            }
        }

        inputs = {
            "messages": [HumanMessage(content="What is the capital of France?")]
        }

        try:
            result = await deep_researcher.ainvoke(inputs, config)
            return result
        finally:
            # Clean up env vars
            if max_tokens < 10000:
                os.environ.pop("FINAL_REPORT_MODEL_MAX_TOKENS", None)
                os.environ.pop("RESEARCH_MODEL_MAX_TOKENS", None)
                os.environ.pop("SUMMARIZATION_MODEL_MAX_TOKENS", None)
                os.environ.pop("COMPRESSION_MODEL_MAX_TOKENS", None)

    @unittest.skipIf(not os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY not set")
    @unittest.skipIf(not os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
    async def test_tavily_openai(self):
        result = await self.run_research("tavily", "openai:gpt-4o-mini")
        self.assertIn("final_report", result)
        self.assertIn("Paris", result["final_report"])

    @unittest.skipIf(not os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
    async def test_openai_native(self):
        result = await self.run_research("openai", "openai:gpt-4o-mini")
        self.assertIn("final_report", result)
        self.assertIn("Paris", result["final_report"])

    @unittest.skipIf(not os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set")
    async def test_anthropic_native(self):
        # Use Anthropic for support models as well
        result = await self.run_research(
            "anthropic",
            "anthropic:claude-3-haiku-20240307",
            other_models_provider="anthropic",
            max_tokens=4096
        )
        self.assertIn("final_report", result)
        self.assertIn("Paris", result["final_report"])

if __name__ == "__main__":
    unittest.main()
