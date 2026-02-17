
import pytest
from legacy.utils import deduplicate_and_format_sources

def test_deduplicate_and_format_sources_basic():
    search_response = [
        {
            "query": "test",
            "results": [
                {
                    "title": "Title 1",
                    "url": "http://example.com/1",
                    "content": "Content 1",
                    "score": 1.0,
                    "raw_content": "Raw Content 1"
                }
            ]
        }
    ]

    result = deduplicate_and_format_sources(search_response)
    assert "Title 1" in result
    assert "http://example.com/1" in result
    assert "Content 1" in result
    assert "Raw Content 1" in result
    assert "Content from sources:" in result

def test_deduplicate_and_format_sources_deduplication():
    search_response = [
        {
            "query": "test",
            "results": [
                {
                    "title": "Title 1",
                    "url": "http://example.com/1",
                    "content": "Content 1",
                    "score": 1.0,
                    "raw_content": "Raw Content 1"
                },
                {
                    "title": "Title 1 Duplicate",
                    "url": "http://example.com/1",
                    "content": "Content 1 Duplicate",
                    "score": 0.9,
                    "raw_content": "Raw Content 1 Duplicate"
                }
            ]
        }
    ]

    # Default is keep_first
    result = deduplicate_and_format_sources(search_response, deduplication_strategy="keep_first")
    assert "Title 1" in result
    assert "Title 1 Duplicate" not in result

    # Test keep_last
    result = deduplicate_and_format_sources(search_response, deduplication_strategy="keep_last")
    assert "Title 1 Duplicate" in result
    # Note: simple string check might fail if both are present, but here we expect only one entry.
    # We can check count of URL occurrence or similar.
    assert result.count("URL: http://example.com/1") == 1

def test_deduplicate_and_format_sources_truncation():
    long_content = "a" * 1000
    search_response = [
        {
            "query": "test",
            "results": [
                {
                    "title": "Title 1",
                    "url": "http://example.com/1",
                    "content": "Content 1",
                    "score": 1.0,
                    "raw_content": long_content
                }
            ]
        }
    ]

    # max_tokens_per_source=10 -> approx 40 chars
    result = deduplicate_and_format_sources(search_response, max_tokens_per_source=10)
    assert "... [truncated]" in result
    assert len(result) < 1000

def test_deduplicate_and_format_sources_no_raw_content():
    search_response = [
        {
            "query": "test",
            "results": [
                {
                    "title": "Title 1",
                    "url": "http://example.com/1",
                    "content": "Content 1",
                    "score": 1.0,
                    "raw_content": None
                }
            ]
        }
    ]

    result = deduplicate_and_format_sources(search_response, include_raw_content=True)
    assert "Title 1" in result
    # Should handle None gracefully

def test_deduplicate_and_format_sources_exclude_raw_content():
    search_response = [
        {
            "query": "test",
            "results": [
                {
                    "title": "Title 1",
                    "url": "http://example.com/1",
                    "content": "Content 1",
                    "score": 1.0,
                    "raw_content": "Raw Content 1"
                }
            ]
        }
    ]

    result = deduplicate_and_format_sources(search_response, include_raw_content=False)
    assert "Title 1" in result
    assert "Raw Content 1" not in result
