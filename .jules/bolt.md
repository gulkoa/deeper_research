## 2024-05-24 - SHA256 Deduplication Overhead
**Learning:** Using `hashlib.sha256` for in-memory string deduplication is significantly slower (approx 2x slower) than using Python's native `set` with string content directly.
**Action:** Always prefer `set()` for string deduplication unless cryptographic hashing is strictly required.
