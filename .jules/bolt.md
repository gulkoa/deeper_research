## 2026-02-18 - [String Concatenation Optimization]
**Learning:** Legacy code used `+=` for building large strings in loops, leading to O(N^2) complexity.
**Action:** Replaced with list accumulation and `"".join()` for O(N) performance. Applied to `deduplicate_and_format_sources` in `src/legacy/utils.py`.
