
import pytest
from langchain_core.documents import Document
from legacy.utils import stitch_documents_by_url

def test_stitch_documents_by_url_basic():
    docs = [
        Document(page_content="Content 1", metadata={"url": "http://example.com/1", "title": "Title 1"}),
        Document(page_content="Content 2", metadata={"url": "http://example.com/1", "title": "Title 1"}),
        Document(page_content="Content 3", metadata={"url": "http://example.com/2", "title": "Title 2"})
    ]

    stitched = stitch_documents_by_url(docs)

    # Should have 2 documents (one for each URL)
    assert len(stitched) == 2

    # Check document for URL 1
    doc1 = next(d for d in stitched if d.metadata["url"] == "http://example.com/1")
    assert "...Content 1..." in doc1.page_content
    assert "...Content 2..." in doc1.page_content
    assert doc1.metadata["title"] == "Title 1"

    # Check document for URL 2
    doc2 = next(d for d in stitched if d.metadata["url"] == "http://example.com/2")
    assert doc2.page_content == "...Content 3..."
    assert doc2.metadata["title"] == "Title 2"

def test_stitch_documents_by_url_deduplication():
    docs = [
        Document(page_content="Content 1", metadata={"url": "http://example.com/1"}),
        Document(page_content="Content 1", metadata={"url": "http://example.com/1"}), # Duplicate content
        Document(page_content="Content 2", metadata={"url": "http://example.com/1"})
    ]

    stitched = stitch_documents_by_url(docs)
    assert len(stitched) == 1

    content = stitched[0].page_content
    # "Content 1" should appear only once
    assert content.count("Content 1") == 1
    assert "Content 2" in content

def test_stitch_documents_by_url_empty():
    stitched = stitch_documents_by_url([])
    assert stitched == []

def test_stitch_documents_by_url_different_metadata_same_url():
    # If metadata differs but URL is same, it should use the first one's metadata
    docs = [
        Document(page_content="Content 1", metadata={"url": "http://example.com/1", "title": "Title A"}),
        Document(page_content="Content 2", metadata={"url": "http://example.com/1", "title": "Title B"})
    ]

    stitched = stitch_documents_by_url(docs)
    assert len(stitched) == 1
    assert stitched[0].metadata["title"] == "Title A"
