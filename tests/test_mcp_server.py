"""
Test suite for MCP Crawler Server
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import tempfile
from pathlib import Path
import hashlib
import numpy as np

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Import modules to test - UPDATE THIS TO YOUR ACTUAL FILENAME
from mcp_server import (  # Change 'mcp_server' to your actual filename
    MCPCrawlerServer,
    WebCrawler,
    ChromaDBManager,
    EmbeddingManager,
    Document,
    CrawlConfig,
    ContentCleaner,
    TextChunker
)


class TestEmbeddingManager:
    """Test embedding generation"""
    
    def test_embedding_initialization(self):
        """Test embedding manager initialization"""
        manager = EmbeddingManager()
        assert manager.model is not None
        assert manager.embedding_dim > 0
        # Remove device check - it's set conditionally in __init__
    
    def test_embed_texts(self):
        """Test text embedding generation"""
        manager = EmbeddingManager()
        texts = ["Hello world", "Test embedding"]
        embeddings = manager.embed_texts(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == manager.embedding_dim
        assert all(isinstance(e, list) for e in embeddings)
    
    def test_embed_query(self):
        """Test query embedding"""
        manager = EmbeddingManager()
        query = "What is machine learning?"
        embedding = manager.embed_query(query)
        
        assert len(embedding) == manager.embedding_dim
        assert isinstance(embedding, list)
    
    def test_embed_query_with_keywords(self):
        """Test query embedding with keyword emphasis"""
        manager = EmbeddingManager()
        query = "What is machine learning?"
        keywords = ["machine", "learning"]
        
        # Test with keywords
        embedding_with_keywords = manager.embed_query_with_keywords(query, keywords)
        assert len(embedding_with_keywords) == manager.embedding_dim
        
        # Test without keywords (should fall back to regular embedding)
        embedding_without = manager.embed_query_with_keywords(query)
        assert len(embedding_without) == manager.embedding_dim


class TestContentCleaner:
    """Test content cleaning functionality"""
    
    def test_extract_title(self):
        """Test title extraction from crawl result"""
        cleaner = ContentCleaner()
        
        # Test with metadata title
        mock_result = Mock()
        mock_result.metadata = {'title': 'Test Title'}
        title = cleaner.extract_title(mock_result, "https://example.com")
        assert title == 'Test Title'
        
        # Test with markdown title
        mock_result.metadata = {}
        mock_result.markdown = "# Markdown Title\n\nContent here"
        title = cleaner.extract_title(mock_result, "https://example.com")
        assert title == 'Markdown Title'
        
        # Test fallback to domain
        mock_result.markdown = "No title here"
        title = cleaner.extract_title(mock_result, "https://example.com/page")
        assert title == 'example.com'
    
    def test_clean_markdown_content(self):
        """Test markdown content cleaning"""
        cleaner = ContentCleaner()
        
        # Test navigation removal
        markdown = """[Skip to main content]
Navigation

From Wikipedia, the free encyclopedia

# Actual Content

This is the real content we want to keep.
It has multiple paragraphs.

And some more text."""
        
        cleaned = cleaner.clean_markdown_content(markdown)
        assert "[Skip to main content]" not in cleaned
        assert "Navigation" not in cleaned  # Removed as it's a navigation marker
        assert "From Wikipedia" in cleaned  # Kept as it's a content marker
        assert "Actual Content" in cleaned
        assert "real content we want" in cleaned
    
    def test_extract_code_blocks(self):
        """Test code block extraction"""
        cleaner = ContentCleaner()
        
        markdown = """Here's some text before code.

```python
def hello():
    print("Hello, world!")
```

And some text after.

```javascript
console.log("Test");
```

Final text."""
        
        code_blocks = cleaner.extract_code_blocks(markdown)
        
        assert len(code_blocks) == 2
        assert code_blocks[0]['language'] == 'python'
        assert 'def hello()' in code_blocks[0]['code']
        assert code_blocks[1]['language'] == 'javascript'
        assert 'console.log' in code_blocks[1]['code']


class TestTextChunker:
    """Test text chunking functionality"""
    
    def test_smart_chunk_markdown(self):
        """Test intelligent markdown chunking"""
        chunker = TextChunker()
        
        text = """# Header 1

This is the first paragraph with some content.

## Header 2

This is another section with more content that should be in a different chunk if it's long enough.

```python
def example():
    return "code block"
```

Final paragraph here."""
        
        chunks = chunker.smart_chunk_markdown(text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
        assert all('id' in chunk for chunk in chunks)
        assert all('summary' in chunk for chunk in chunks)
    
    def test_chunk_metadata(self):
        """Test chunk metadata creation"""
        chunker = TextChunker()
        
        text = """# Main Title

Some content here.

- List item 1
- List item 2

| Col1 | Col2 |
|------|------|
| A    | B    |
"""
        
        chunks = chunker.smart_chunk_markdown(text, chunk_size=200)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk['has_lists'] is True
        assert chunk['has_tables'] is True
        assert '# Main Title' in chunk['headers']


class TestChromaDBManager:
    """Test ChromaDB operations"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_initialization(self, temp_db):
        """Test ChromaDB manager initialization"""
        manager = ChromaDBManager(persist_directory=temp_db)
        assert manager.client is not None
        assert manager.collection is not None
        assert manager.collection.name == "web_documents"
        assert manager.embedding_manager is not None
    
    def test_add_documents(self, temp_db):
        """Test adding documents to database"""
        manager = ChromaDBManager(persist_directory=temp_db)
        
        # Create test documents with chunks
        docs = [
            Document(
                id="doc1",
                url="https://example.com/1",
                title="Test Document 1",
                content="This is test content about machine learning",
                markdown="# Test Document 1\n\nContent here",
                chunks=[
                    {
                        'id': 0,
                        'text': 'Test chunk content',
                        'start_pos': 0,
                        'end_pos': 100,
                        'headers': '# Test',
                        'summary': 'Test summary',
                        'word_count': 10,
                        'char_count': 50,
                        'code_blocks': 0,
                        'has_lists': False,
                        'has_tables': False
                    }
                ],
                timestamp=datetime.now()
            )
        ]
        
        manager.add_documents(docs)
        info = manager.get_collection_info()
        
        assert info["count"] == 1
    
    def test_search(self, temp_db):
        """Test semantic search functionality"""
        manager = ChromaDBManager(persist_directory=temp_db)
        
        # Add test document
        doc = Document(
            id="search_test",
            url="https://example.com/ml",
            title="Machine Learning Guide",
            content="Machine learning is a subset of artificial intelligence",
            markdown="# ML Guide",
            chunks=[
                {
                    'id': 0,
                    'text': 'Machine learning is a subset of artificial intelligence',
                    'start_pos': 0,
                    'end_pos': 100,
                    'headers': '# ML Guide',
                    'summary': 'ML intro',
                    'word_count': 8,
                    'char_count': 54,
                    'code_blocks': 0,
                    'has_lists': False,
                    'has_tables': False
                }
            ],
            timestamp=datetime.now()
        )
        
        manager.add_documents([doc])
        
        # Search
        results = manager.search("What is machine learning?", n_results=1)
        
        assert len(results) > 0
        assert results[0]["metadata"]["title"] == "Machine Learning Guide"
    
    def test_search_with_keywords(self, temp_db):
        """Test search with keyword emphasis"""
        manager = ChromaDBManager(persist_directory=temp_db)
        
        # Add test document
        doc = Document(
            id="keyword_test",
            url="https://example.com/test",
            title="Keyword Test",
            content="Python programming and machine learning",
            markdown="# Test",
            chunks=[
                {
                    'id': 0,
                    'text': 'Python programming and machine learning techniques',
                    'start_pos': 0,
                    'end_pos': 50,
                    'headers': '',
                    'summary': 'Python ML',
                    'word_count': 6,
                    'char_count': 50,
                    'code_blocks': 0,
                    'has_lists': False,
                    'has_tables': False
                }
            ],
            timestamp=datetime.now()
        )
        
        manager.add_documents([doc])
        
        # Search with keywords
        results = manager.search("programming", n_results=1, keywords=["Python"])
        
        assert len(results) > 0
    
    def test_get_document_chunks(self, temp_db):
        """Test retrieving all chunks for a document"""
        manager = ChromaDBManager(persist_directory=temp_db)
        
        # Add document with multiple chunks
        doc = Document(
            id="multi_chunk",
            url="https://example.com/multi",
            title="Multi Chunk Doc",
            content="Long content",
            markdown="# Long Doc",
            chunks=[
                {
                    'id': i,
                    'text': f'Chunk {i} content',
                    'start_pos': i * 100,
                    'end_pos': (i + 1) * 100,
                    'headers': '',
                    'summary': f'Chunk {i}',
                    'word_count': 3,
                    'char_count': 20,
                    'code_blocks': 0,
                    'has_lists': False,
                    'has_tables': False
                }
                for i in range(3)
            ],
            timestamp=datetime.now()
        )
        
        manager.add_documents([doc])
        
        # Get chunks
        chunks = manager.get_document_chunks("multi_chunk")
        
        assert len(chunks) == 3
        assert all('content' in chunk for chunk in chunks)
        assert chunks[0]['metadata']['chunk_index'] == 0


class TestWebCrawler:
    """Test web crawling functionality"""
    
    @pytest.mark.asyncio
    @patch('mcp_server.AsyncWebCrawler')  # Update module name here
    async def test_crawl_url(self, mock_crawler_class):
        """Test crawling a single URL"""
        # Mock crawler instance
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        
        # Mock crawl result
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "# Test Page\n\nTest content that is long enough to not be skipped. This needs to be more than 100 characters to pass the content length check."
        mock_result.metadata = {'title': 'Test Page'}
        
        mock_crawler.arun.return_value = mock_result
        
        # Test crawling
        config = CrawlConfig()
        crawler = WebCrawler(config)
        crawler.crawler = mock_crawler
        
        doc = await crawler.crawl_url("https://example.com")
        
        assert doc is not None
        assert doc.title == "Test Page"
        assert "Test content" in doc.content
        assert doc.url == "https://example.com"
        assert len(doc.chunks) > 0
    
    @pytest.mark.asyncio
    @patch('mcp_server.AsyncWebCrawler')  # Update module name here
    async def test_crawl_recursive(self, mock_crawler_class):
        """Test recursive crawling"""
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        
        # Mock results
        mock_result1 = Mock()
        mock_result1.success = True
        mock_result1.markdown = "# Page 1\n\nContent for page 1 with enough text to not be skipped. This needs to be more than 100 characters."
        mock_result1.metadata = {'title': 'Page 1'}
        mock_result1.links = [{'href': 'https://example.com/page2'}]
        
        mock_result2 = Mock()
        mock_result2.success = True
        mock_result2.markdown = "# Page 2\n\nContent for page 2 with enough text to not be skipped. This also needs to be more than 100 characters."
        mock_result2.metadata = {'title': 'Page 2'}
        mock_result2.links = []
        
        mock_crawler.arun.side_effect = [mock_result1, mock_result2]
        
        config = CrawlConfig(max_depth=1, max_pages=2)
        async with WebCrawler(config) as crawler:
            crawler.crawler = mock_crawler
            docs = await crawler.crawl_recursive("https://example.com")
        
        assert len(docs) <= 2
        assert all(isinstance(doc, Document) for doc in docs)


class TestMCPCrawlerServer:
    """Test main MCP server functionality"""
    
    @patch('chromadb.PersistentClient')
    def test_server_initialization(self, mock_chromadb):
        """Test server initialization"""
        server = MCPCrawlerServer()
        assert server.server is not None
        assert server.db_manager is not None
        assert server.crawl_config is not None
    
    @pytest.mark.asyncio
    @patch('mcp_server.WebCrawler')  # Update module name here
    @patch('chromadb.PersistentClient')
    async def test_crawl_website_tool(self, mock_chromadb, mock_crawler_class):
        """Test crawl_website tool"""
        server = MCPCrawlerServer()
        
        # Mock crawler
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        
        # Mock crawl results
        mock_docs = [
            Document(
                id="test1",
                url="https://example.com",
                title="Test Page",
                content="Test content",
                markdown="# Test",
                chunks=[{
                    'text': 'chunk1',
                    'summary': 'sum1',
                    'word_count': 10,
                    'headers': '',
                    'code_blocks': 0,
                    'has_lists': False,
                    'has_tables': False,
                    'start_pos': 0,
                    'end_pos': 10,
                    'char_count': 10,
                    'id': 0
                }],
                timestamp=datetime.now(),
                metadata={'word_count': 100}
            )
        ]
        mock_crawler.crawl_recursive.return_value = mock_docs
        mock_crawler.__aenter__.return_value = mock_crawler
        mock_crawler.__aexit__.return_value = None
        
        # Call tool
        result = await server._crawl_website({"url": "https://example.com", "max_pages": 1})
        
        assert len(result) == 1
        assert result[0].type == "text"
        assert "Successfully crawled" in result[0].text
    
    @pytest.mark.asyncio
    @patch('chromadb.PersistentClient')
    async def test_search_content_tool(self, mock_chromadb):
        """Test search_content tool"""
        server = MCPCrawlerServer()
        
        # Mock search results
        mock_results = [
            {
                'id': 'doc1_chunk_0',
                'content': 'Test content about Python',
                'metadata': {
                    'title': 'Python Guide',
                    'url': 'https://example.com/python',
                    'chunk_index': 0,
                    'chunk_total': 1,
                    'headers': '# Python'
                },
                'similarity': 0.85
            }
        ]
        
        with patch.object(server.db_manager, 'search', return_value=mock_results):
            result = await server._search_content({"query": "Python programming"})
        
        assert len(result) == 1
        assert result[0].type == "text"
        assert "Python Guide" in result[0].text
        assert "85.00%" in result[0].text  # Changed from "85%" to "85.00%"
    
    @pytest.mark.asyncio
    @patch('chromadb.PersistentClient')
    async def test_get_document_tool(self, mock_chromadb):
        """Test get_document tool"""
        server = MCPCrawlerServer()
        
        # Mock document chunks
        mock_chunks = [
            {
                'id': 'doc1_chunk_0',
                'content': 'First chunk content',
                'metadata': {
                    'title': 'Test Doc',
                    'url': 'https://example.com',
                    'chunk_index': 0
                }
            },
            {
                'id': 'doc1_chunk_1',
                'content': 'Second chunk content',
                'metadata': {
                    'title': 'Test Doc',
                    'url': 'https://example.com',
                    'chunk_index': 1
                }
            }
        ]
        
        with patch.object(server.db_manager, 'get_document_chunks', return_value=mock_chunks):
            result = await server._get_document({"doc_id": "doc1"})
        
        assert len(result) == 1
        assert result[0].type == "text"
        assert "Test Doc" in result[0].text
        assert "Chunk 1" in result[0].text
        assert "Chunk 2" in result[0].text
    
    @pytest.mark.asyncio
    @patch('chromadb.PersistentClient')
    async def test_get_crawl_stats_tool(self, mock_chromadb):
        """Test get_crawl_stats tool"""
        server = MCPCrawlerServer()
        
        # Mock collection info
        mock_info = {
            'name': 'web_documents',
            'count': 42,
            'metadata': {'hnsw:space': 'cosine'},
            'embedding_model': 'nomic-ai/nomic-embed-text-v1',
            'embedding_dimensions': 768
        }
        
        # Mock sample results
        mock_sample = {
            'metadatas': [
                {
                    'word_count': 100,
                    'has_code': True,
                    'code_blocks': 2,
                    'url': 'https://example.com/1'
                },
                {
                    'word_count': 200,
                    'has_code': False,
                    'code_blocks': 0,
                    'url': 'https://example.com/2'
                }
            ]
        }
        
        with patch.object(server.db_manager, 'get_collection_info', return_value=mock_info):
            with patch.object(server.db_manager.collection, 'get', return_value=mock_sample):
                result = await server._get_crawl_stats()
        
        assert len(result) == 1
        assert result[0].type == "text"
        assert "42" in result[0].text
        assert "150" in result[0].text  # Average words


# Integration test
@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_flow():
    """Test complete flow from crawl to search"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize server with temp database
        server = MCPCrawlerServer()
        server.db_manager = ChromaDBManager(persist_directory=tmpdir)
        
        # Create test document
        test_doc = Document(
            id=hashlib.md5("https://example.com".encode()).hexdigest(),
            url="https://example.com",
            title="E2E Test",
            content="End to end testing content about Python testing and web crawling",
            markdown="# E2E Test\n\nContent here",
            chunks=[
                {
                    'id': 0,
                    'text': 'End to end testing content about Python testing and web crawling',
                    'start_pos': 0,
                    'end_pos': 100,
                    'headers': '# E2E Test',
                    'summary': 'E2E testing content',
                    'word_count': 10,
                    'char_count': 65,
                    'code_blocks': 0,
                    'has_lists': False,
                    'has_tables': False
                }
            ],
            timestamp=datetime.now(),
            metadata={'word_count': 10, 'has_code': False}
        )
        
        # Mock crawling
        with patch('mcp_server.WebCrawler') as mock_crawler_class:  # Update module name
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value = mock_crawler
            mock_crawler.crawl_recursive.return_value = [test_doc]
            mock_crawler.__aenter__.return_value = mock_crawler
            mock_crawler.__aexit__.return_value = None
            
            # Crawl website
            result = await server._crawl_website({"url": "https://example.com", "max_pages": 1})
            assert "Successfully crawled" in result[0].text
        
        # Search knowledge
        search_results = await server._search_content({"query": "Python testing", "n_results": 1})
        assert "E2E Test" in search_results[0].text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])