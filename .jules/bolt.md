## 2025-05-15 - Optimization of Summarization for Short Content
**Learning:** LLM summarization calls are expensive and slow. For short content (< 3000 chars), the "summary" is often just the content itself. Skipping the LLM call reduces latency significantly for these cases.
**Action:** Always check if content length warrants an LLM call before invoking it. Ensure the return value matches what the caller expects (in this case, raw string vs structured object were both handled by the caller).
