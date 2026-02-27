
import time
from src.open_deep_research.utils import _get_summarization_model

def benchmark_optimized_initialization():
    model_name = "openai:gpt-4o-mini"
    max_tokens = 1000
    api_key = "fake-key"
    max_retries = 3

    # First call - should be slow (uncached)
    start_time_first = time.time()
    _ = _get_summarization_model(model_name, max_tokens, api_key, max_retries)
    end_time_first = time.time()
    print(f"First call (uncached): {end_time_first - start_time_first:.6f} seconds")

    # Subsequent calls - should be fast (cached)
    start_time_rest = time.time()
    for _ in range(99):
        _ = _get_summarization_model(model_name, max_tokens, api_key, max_retries)
    end_time_rest = time.time()

    print(f"Next 99 calls (cached): {end_time_rest - start_time_rest:.6f} seconds")
    print(f"Average time per cached call: {(end_time_rest - start_time_rest) / 99:.6f} seconds")

if __name__ == "__main__":
    benchmark_optimized_initialization()
