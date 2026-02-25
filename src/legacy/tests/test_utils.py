
import pytest
from legacy.utils import deduplicate_and_format_sources, format_sections
from legacy.state import Section

def test_deduplicate_and_format_sources_basic():
    search_response = [
        {
            "query": "test query",
            "results": [
                {
                    "title": "Test Source 1",
                    "url": "http://example.com/1",
                    "content": "Content for source 1",
                    "score": 1.0,
                    "raw_content": "Raw content 1"
                },
                {
                    "title": "Test Source 2",
                    "url": "http://example.com/2",
                    "content": "Content for source 2",
                    "score": 0.9,
                    "raw_content": "Raw content 2"
                }
            ]
        }
    ]

    formatted = deduplicate_and_format_sources(search_response, max_tokens_per_source=100)

    expected_output = (
        "Content from sources:\n"
        f"{'='*80}\n"
        "Source: Test Source 1\n"
        f"{'-'*80}\n"
        "URL: http://example.com/1\n===\n"
        "Most relevant content from source: Content for source 1\n===\n"
        "Full source content limited to 100 tokens: Raw content 1\n\n"
        f"{'='*80}\n\n"
        f"{'='*80}\n"
        "Source: Test Source 2\n"
        f"{'-'*80}\n"
        "URL: http://example.com/2\n===\n"
        "Most relevant content from source: Content for source 2\n===\n"
        "Full source content limited to 100 tokens: Raw content 2\n\n"
        f"{'='*80}\n"
    )

    # Check that output matches expected format, ignoring minor whitespace differences at the end
    assert formatted.strip() == expected_output.strip()

def test_deduplicate_and_format_sources_deduplication():
    search_response = [
        {
            "query": "query 1",
            "results": [
                {
                    "title": "Test Source 1",
                    "url": "http://example.com/1",
                    "content": "Content 1",
                    "score": 1.0,
                    "raw_content": "Raw 1"
                }
            ]
        },
        {
            "query": "query 2",
            "results": [
                {
                    "title": "Test Source 1 Duplicate",
                    "url": "http://example.com/1",
                    "content": "Content 1 Duplicate",
                    "score": 0.9,
                    "raw_content": "Raw 1 Duplicate"
                }
            ]
        }
    ]

    formatted = deduplicate_and_format_sources(search_response, deduplication_strategy="keep_first")

    # Should only contain the first source
    assert "Test Source 1" in formatted
    assert "Test Source 1 Duplicate" not in formatted
    assert formatted.count("URL: http://example.com/1") == 1

def test_format_sections():
    sections = [
        Section(
            name="Introduction",
            description="Overview of the topic",
            research=False,
            content="This is the introduction."
        ),
        Section(
            name="Analysis",
            description="Deep dive",
            research=True,
            content="Detailed analysis here."
        )
    ]

    formatted = format_sections(sections)

    expected_part1 = (
        f"{'='*60}\n"
        "Section 1: Introduction\n"
        f"{'='*60}\n"
        "Description:\n"
        "Overview of the topic\n"
        "Requires Research: \n"
        "False\n\n"
        "Content:\n"
        "This is the introduction.\n"
    )

    expected_part2 = (
        f"{'='*60}\n"
        "Section 2: Analysis\n"
        f"{'='*60}\n"
        "Description:\n"
        "Deep dive\n"
        "Requires Research: \n"
        "True\n\n"
        "Content:\n"
        "Detailed analysis here.\n"
    )

    assert expected_part1.strip() in formatted
    assert expected_part2.strip() in formatted
