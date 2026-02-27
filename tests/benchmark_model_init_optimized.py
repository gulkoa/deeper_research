
import time
import asyncio
from src.open_deep_research.utils import _get_summarization_model

async def benchmark_optimized_initialization():
    model_name = "openai:gpt-4o-mini"
    max_tokens = 1000
    api_key = "fake-key"
    max_retries = 3

    start_time = time.time()
    for _ in range(100):
        # This will use the cached model after the first call
        _ = _get_summarization_model(
            model_name=model_name,
            max_tokens=max_tokens,
            api_key=api_key,
            max_retries=max_retries
        )
    end_time = time.time()

    print(f"Total time for 100 calls (optimized): {end_time - start_time:.6f} seconds")
    print(f"Average time per call (optimized): {(end_time - start_time) / 100:.6f} seconds")

if __name__ == "__main__":
    asyncio.run(benchmark_optimized_initialization())
