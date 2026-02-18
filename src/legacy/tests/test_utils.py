import pytest
from legacy.utils import deduplicate_and_format_sources

def test_deduplicate_and_format_sources():
    search_response = [
        {
            'query': 'test query',
            'results': [
                {
                    'title': 'Test Title One',
                    'url': 'http://example.com/1',
                    'content': 'Content One',
                    'score': 0.9,
                    'raw_content': 'Raw Content One'
                },
                {
                    'title': 'Test Title Two',
                    'url': 'http://example.com/2',
                    'content': 'Content Two',
                    'score': 0.8,
                    'raw_content': 'Raw Content Two'
                },
                {
                    'title': 'Test Title One Duplicate',
                    'url': 'http://example.com/1',
                    'content': 'Content One Duplicate',
                    'score': 0.7,
                    'raw_content': 'Raw Content One Duplicate'
                }
            ]
        }
    ]

    formatted_output = deduplicate_and_format_sources(search_response, max_tokens_per_source=100, deduplication_strategy="keep_first")

    assert "Source: Test Title One" in formatted_output
    assert "URL: http://example.com/1" in formatted_output
    assert "Most relevant content from source: Content One" in formatted_output
    assert "Full source content limited to 100 tokens: Raw Content One" in formatted_output

    assert "Source: Test Title Two" in formatted_output
    assert "URL: http://example.com/2" in formatted_output

    # Check for duplicate removal (keep_first strategy)
    # The duplicate has a different title so we can check it's not present
    assert "Source: Test Title One Duplicate" not in formatted_output
    assert "Content One Duplicate" not in formatted_output

    # Check structure
    assert "Content from sources:" in formatted_output
    assert "="*80 in formatted_output

def test_deduplicate_and_format_sources_keep_last():
    search_response = [
        {
            'query': 'test query',
            'results': [
                {
                    'title': 'Test Title A',
                    'url': 'http://example.com/1',
                    'content': 'Content A',
                    'score': 0.9,
                    'raw_content': 'Raw Content A'
                },
                {
                    'title': 'Test Title B', # Same URL, different content/title
                    'url': 'http://example.com/1',
                    'content': 'Content B',
                    'score': 0.7,
                    'raw_content': 'Raw Content B'
                }
            ]
        }
    ]

    formatted_output = deduplicate_and_format_sources(search_response, max_tokens_per_source=100, deduplication_strategy="keep_last")

    # Check for duplicate removal (keep_last strategy)
    assert "Source: Test Title B" in formatted_output
    assert "Most relevant content from source: Content B" in formatted_output

    # "Test Title A" should NOT be in the output
    assert "Source: Test Title A" not in formatted_output
