## 2024-05-24 - SHA256 Deduplication Overhead
**Learning:** Using `hashlib.sha256` for in-memory string deduplication is significantly slower (approx 2x slower) than using Python's native `set` with string content directly.
**Action:** Always prefer `set()` for string deduplication unless cryptographic hashing is strictly required.

## 2025-01-28 - O(N^2) String Concatenation Bottleneck
**Learning:** Using `+=` to repeatedly append string chunks within a loop inside formatting functions (e.g. formatting search results) creates an O(N^2) time complexity bottleneck due to continuous reallocation of string objects in Python.
**Action:** Always replace `+=` inside large loops with a list-based string builder pattern (i.e. `"".join(list_of_strings)`) to ensure O(N) performance.
