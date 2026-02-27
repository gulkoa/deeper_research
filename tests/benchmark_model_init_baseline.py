
import time
import asyncio
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

class Summary(BaseModel):
    summary: str
    key_excerpts: str

async def benchmark_initialization():
    model_name = "openai:gpt-4o-mini"
    max_tokens = 1000
    api_key = "fake-key" # We won't actually call the API, just initialize

    start_time = time.time()
    for _ in range(100):
        model = init_chat_model(
            model=model_name,
            max_tokens=max_tokens,
            api_key=api_key,
            tags=["langsmith:nostream"]
        ).with_structured_output(Summary).with_retry(
            stop_after_attempt=3
        )
    end_time = time.time()

    print(f"Total time for 100 initializations: {end_time - start_time:.4f} seconds")
    print(f"Average time per initialization: {(end_time - start_time) / 100:.4f} seconds")

if __name__ == "__main__":
    asyncio.run(benchmark_initialization())
