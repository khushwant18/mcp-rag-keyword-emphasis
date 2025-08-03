import os
import json
import logging
import asyncio
import hashlib
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, field
import torch

# MCP imports
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import NoExtractionStrategy
from crawl4ai import BrowserConfig, CrawlerRunConfig, CacheMode

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# Embedding model imports
from sentence_transformers import SentenceTransformer
import numpy as np

# Additional utilities
from bs4 import BeautifulSoup
import aiohttp

from config import EmbeddingConfig, CrawlConfig, DatabaseConfig
from urllib.parse import urlparse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
browser_config = BrowserConfig(
        headless=True,
        verbose=False  # Critical: suppress browser output
    )


@dataclass
class Document:
    """Represents a scraped document"""
    id: str
    url: str
    title: str
    content: str
    markdown: str
    chunks: List[Dict[str, Any]]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingManager:
    """Manages embeddings using MixedBread AI model"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        logger.info(f"Initializing embedding model: {self.config.model_name}")
        self.model = SentenceTransformer(
            self.config.model_name, 
            trust_remote_code=self.config.trust_remote_code
        )
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        
    def _create_token_weights(
        self,
        tokens: List[str],
        keywords: List[str],
        importance_multiplier: float,
    ) -> torch.Tensor:
        """Create weights for tokens based on keywords."""
        weights = torch.ones(len(tokens))
        keywords_lower = [kw.lower().strip() for kw in keywords]
        
        for i, token in enumerate(tokens):
            # Clean token (remove special characters)
            clean_token = token.replace("▁", "").replace("Ġ", "").lower().strip()
            
            # Check if token matches any keyword
            for keyword in keywords_lower:
                if keyword in clean_token or clean_token in keyword:
                    weights[i] = importance_multiplier
                    break
        
        return weights
    
    def _weighted_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Apply weighted pooling to hidden states."""
        # Ensure weights is a tensor
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)
        
        weights = weights.to(hidden_states.device)
        mask_weights = attention_mask.float() * weights.unsqueeze(0)
        
        weighted_embeddings = hidden_states * mask_weights.unsqueeze(-1)
        pooled = weighted_embeddings.sum(dim=1) / (
            mask_weights.sum(dim=1, keepdim=True) + 1e-8
        )
        
        return pooled
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        # Add document prompt for storage
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query with retrieval prompt"""
        query_with_prompt = f"{query}"
        embedding = self.model.encode(query_with_prompt, convert_to_numpy=True)
        return embedding.tolist()

    def _cls_pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract CLS token representation (first token)."""
        return hidden_states[:, 0, :]
    
    def embed_query_with_keywords(
        self, 
        query: str, 
        keywords: List[str] = None, 
        importance_multiplier: float = None
    ) -> List[float]:
        """Generate embedding for a query with keyword emphasis using token weighting"""
        if not keywords:
            return self.embed_query(query)
        
        multiplier = importance_multiplier or self.config.keyword_importance_multiplier

        try:
            query_with_prompt = f"Represent this sentence for searching relevant passages: {query}"
            
            # Get the tokenizer from the model
            tokenizer = self.model.tokenizer
            
            # Tokenize the query
            inputs = tokenizer(
                query_with_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.max_seq_length,
                padding=True
            )
            # Get tokens
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            # Create weights based on keywords
            weights = self._create_token_weights(tokens, keywords, multiplier)
            # Get model outputs with hidden states
            with torch.no_grad():
                # Access the transformer model within sentence transformer
                transformer = self.model[0].auto_model
                outputs = transformer(**inputs)
                hidden_states = outputs.last_hidden_state
            
            # Apply weighted pooling
            try:
                weighted_embedding = self._weighted_pool(
                    hidden_states,
                    inputs["attention_mask"],
                    weights
                )
            except Exception as e:
                logger.error(f"Error in embed_query_with_keywords: {str(e)}")

            cls_embedding = self._cls_pool(hidden_states)
            ratio = self.config.weighted_cls_blend_ratio
            weighted_embedding = ratio * weighted_embedding + (1 - ratio) * cls_embedding
            
            return weighted_embedding[0].cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error in embed_query_with_keywords: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to regular embedding
            return self.embed_query(query)

class ContentCleaner:
    """Handles content cleaning and extraction"""
    
    @staticmethod
    def extract_title(result: Any, url: str) -> str:
        """Extract title from crawl result"""
        # Try to get title from result
        title_text = ""
        if hasattr(result, 'metadata') and result.metadata:
            title_text = result.metadata.get('title', '')
        
        if not title_text and hasattr(result, 'markdown') and result.markdown:
            # Try to extract title from markdown (usually first # heading)
            lines = result.markdown.split('\n')
            for line in lines[:20]:  # Check first 20 lines
                if line.strip().startswith('# '):
                    title_text = line.strip('# ').strip()
                    break
        
        if not title_text:
            title_text = urlparse(url).netloc
            
        return title_text
    
    @staticmethod
    def clean_markdown_content(markdown: str) -> str:
        """Clean markdown content by removing navigation and other non-content elements"""
        if not markdown:
            return ""
            
        # Common navigation patterns to skip
        navigation_markers = [
            "[Jump to content]",
            "Main menu",
            "Navigation",
            "[Main page]",
            "[Contents]",
            "[Random article]",
            "move to sidebar",
            "Skip to main content",
            "Skip to navigation",
            "Search form",
            "Table of contents"
        ]
        
        # Content markers that indicate real content
        content_markers = [
            "From Wikipedia",
            "Coordinates:",
            "This article is about",
            "may refer to",
            "is a",
            "was a",
            "are a",
            "refers to",
            "Introduction",
            "Overview",
            "Definition"
        ]
        
        lines = markdown.split('\n')
        content_start_idx = 0
        
        # Find where actual content starts
        for i, line in enumerate(lines):
            # Check if we've found actual content
            found_content = False
            for marker in content_markers:
                if marker in line:
                    content_start_idx = i
                    found_content = True
                    break
            
            if found_content:
                break
                
            # Skip navigation lines
            if any(nav_marker in line for nav_marker in navigation_markers):
                continue
                
            # If we find a substantial line (not navigation), use it
            if len(line.strip()) > 50 and not any(nav in line for nav in navigation_markers):
                content_start_idx = i
                break
        
        # Extract content from the start index
        if content_start_idx > 0:
            content = '\n'.join(lines[content_start_idx:])
        else:
            content = markdown
        
        # Additional cleaning
        # Remove multiple consecutive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove lines that are just links or references
        cleaned_lines = []
        for line in content.split('\n'):
            # Skip lines that are just navigation links
            if line.strip().startswith('[') and line.strip().endswith(']'):
                if len(line.strip()) < 50:  # Short bracketed text is likely navigation
                    continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def extract_code_blocks(markdown: str) -> List[Dict[str, Any]]:
        """Extract code blocks from markdown"""
        code_blocks = []
        
        # Pattern to match code blocks with optional language
        pattern = r'```(?:(\w+))?\n(.*?)```'
        matches = re.finditer(pattern, markdown, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or 'unknown'
            code = match.group(2).strip()
            
            # Get context around the code block
            start_pos = match.start()
            end_pos = match.end()
            
            # Find context before (up to 200 chars or previous paragraph)
            context_before_start = max(0, start_pos - 200)
            context_before = markdown[context_before_start:start_pos].strip()
            if '\n\n' in context_before:
                context_before = context_before.split('\n\n')[-1]
            
            # Find context after (up to 200 chars or next paragraph)
            context_after_end = min(len(markdown), end_pos + 200)
            context_after = markdown[end_pos:context_after_end].strip()
            if '\n\n' in context_after:
                context_after = context_after.split('\n\n')[0]
            
            code_blocks.append({
                'language': language,
                'code': code,
                'context_before': context_before,
                'context_after': context_after,
                'position': start_pos
            })
        
        return code_blocks


class TextChunker:
    """Handles intelligent text chunking"""
    
    @staticmethod
    def smart_chunk_markdown(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into chunks, respecting code blocks and paragraphs."""
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        chunk_id = 0
        
        while start < text_length:
            # Calculate the end position
            end = min(start + chunk_size, text_length)
            
            # Don't break if we're at the end
            if end >= text_length:
                chunk_text = text[start:]
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(TextChunker._create_chunk(chunk_text, chunk_id, start, text_length))
                break
            
            # Extract the chunk
            chunk = text[start:end]
            
            # Try to find a good break point
            # Priority 1: Code block boundary
            code_block_pos = chunk.rfind('```')
            if code_block_pos != -1 and code_block_pos > chunk_size * 0.3:
                # Check if we're in the middle of a code block
                code_blocks_before = chunk[:code_block_pos].count('```')
                if code_blocks_before % 2 == 0:  # Even number means we're not in a code block
                    end = start + code_block_pos
            
            # Priority 2: Section boundary (headers)
            elif '\n#' in chunk:
                # Find the last header
                header_positions = [m.start() for m in re.finditer(r'\n#+\s', chunk)]
                if header_positions:
                    last_header = header_positions[-1]
                    if last_header > chunk_size * 0.3:
                        end = start + last_header
            
            # Priority 3: Paragraph boundary
            elif '\n\n' in chunk:
                last_break = chunk.rfind('\n\n')
                if last_break > chunk_size * 0.3:
                    end = start + last_break
            
            # Priority 4: Sentence boundary
            elif '. ' in chunk:
                # Find the last sentence ending
                sentence_endings = [m.end() for m in re.finditer(r'\. (?=[A-Z])', chunk)]
                if sentence_endings:
                    last_sentence = sentence_endings[-1]
                    if last_sentence > chunk_size * 0.3:
                        end = start + last_sentence
            
            # Priority 5: Line boundary
            elif '\n' in chunk:
                last_newline = chunk.rfind('\n')
                if last_newline > chunk_size * 0.3:
                    end = start + last_newline
            
            # Extract the final chunk
            chunk_text = text[start:end]
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(TextChunker._create_chunk(chunk_text, chunk_id, start, end))
                chunk_id += 1
            
            # Move start position with overlap
            start = end - chunk_overlap if end < text_length else end
        
        return chunks
    
    @staticmethod
    def _create_chunk(text: str, chunk_id: int, start_pos: int, end_pos: int) -> Dict[str, Any]:
        """Create a chunk with metadata"""
        # Extract headers from the chunk
        headers = re.findall(r'^(#+)\s+(.+)$', text, re.MULTILINE)
        header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers[:5]]) if headers else ''
        
        # Count code blocks
        code_blocks = text.count('```') // 2
        
        # Extract first meaningful sentence as summary
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        summary = ''
        for sentence in sentences[:3]:  # Check first 3 sentences
            if len(sentence) > 20 and not sentence.startswith('['):
                summary = sentence
                break
        
        return {
            'id': chunk_id,
            'text': text.strip(),
            'start_pos': start_pos,
            'end_pos': end_pos,
            'headers': header_str,
            'summary': summary[:200] if summary else text[:200].strip(),
            'word_count': len(text.split()),
            'char_count': len(text),
            'code_blocks': code_blocks,
            'has_lists': bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE)),
            'has_tables': '|' in text and text.count('|') > 4
        }


class WebCrawler:
    """Handles web crawling using Crawl4AI"""
    
    def __init__(self, config: CrawlConfig = None):
        self.config = config or CrawlConfig()
        self.visited_urls = set()
        self.crawler = None
        self.content_cleaner = ContentCleaner()
        self.text_chunker = TextChunker()
        
    async def __aenter__(self):
        self.crawler = AsyncWebCrawler(config=browser_config)
        await self.crawler.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
    
    async def crawl_url(self, url: str, extract_links: bool = False) -> Optional[Document]:
        """Crawl a single URL and extract content"""
        if url in self.visited_urls:
            return None
            
        self.visited_urls.add(url)
        logger.info(f"Crawling: {url}")
        
        try:
            # Crawl4AI configuration - removed verbose parameter
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                verbose=False,  # Critical: suppress crawl output
                word_count_threshold=10,
                remove_overlay_elements=True,
                excluded_tags=self.config.excluded_tags,
                screenshot=False,
                pdf=False
            )
            result = await self.crawler.arun(
                url=url,
                config=run_config,
                extraction_strategy=NoExtractionStrategy()
            )
            
            # Check if crawl was successful
            if not result.success:
                error_msg = getattr(result, 'error_message', 'Unknown error')
                logger.error(f"Failed to crawl {url}: {error_msg}")
                return None
            
            # Extract and clean content
            title = self.content_cleaner.extract_title(result, url)
            cleaned_content = self.content_cleaner.clean_markdown_content(result.markdown)
            
            # Skip if content is too short
            if len(cleaned_content) < 100:
                logger.warning(f"Content too short for {url}, skipping")
                return None
            
            # Extract code blocks
            code_blocks = self.content_cleaner.extract_code_blocks(cleaned_content)
            
            # Chunk the content
            chunks = self.text_chunker.smart_chunk_markdown(
                cleaned_content, 
                chunk_size=self.config.chunk_size
            )
            
            # Generate document ID
            doc_id = hashlib.md5(url.encode()).hexdigest()
            
            # Create document
            document = Document(
                id=doc_id,
                url=url,
                title=title,
                content=cleaned_content[:10000],  # Store first 10k chars as main content
                markdown=result.markdown,  # Store original markdown
                chunks=chunks,
                timestamp=datetime.now(),
                metadata={
                    "word_count": len(cleaned_content.split()),
                    "char_count": len(cleaned_content),
                    "chunk_count": len(chunks),
                    "code_block_count": len(code_blocks),
                    "crawl4ai_word_count": result.metadata.get('word_count', 0) if hasattr(result, 'metadata') else 0,
                    "has_code": len(code_blocks) > 0
                }
            )
            
            # Store code blocks in metadata if present
            if code_blocks:
                document.metadata["code_examples"] = code_blocks[:5]  # Store first 5 code blocks
            
            # Extract links if requested
            if extract_links:
                links = self._extract_links(result, url)
                document.metadata["links"] = links[:10]  # Store first 10 links
                document.metadata["link_count"] = len(links)
                
            return document
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            return None
    
    def _extract_links(self, result: Any, base_url: str) -> List[str]:
        """Extract and filter links from crawl result"""
        links = []
        
        # Try to get links from result
        if hasattr(result, 'links') and result.links:
            for link in result.links:
                href = link.get('href', '') if isinstance(link, dict) else str(link)
                if href and href.startswith(('http://', 'https://')):
                    links.append(href)
                elif href and not href.startswith(('#', 'javascript:', 'mailto:')):
                    # Relative link - make it absolute
                    absolute_url = urljoin(base_url, href)
                    links.append(absolute_url)
        
        # Filter links based on config
        if not self.config.include_external:
            base_domain = urlparse(base_url).netloc
            links = [link for link in links if urlparse(link).netloc == base_domain]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
                
        return unique_links
    
    async def crawl_recursive(self, start_url: str, max_depth: Optional[int] = None) -> List[Document]:
        """Recursively crawl starting from a URL"""
        documents = []
        max_depth = max_depth or self.config.max_depth
        
        # Queue format: (url, depth)
        to_crawl = [(start_url, 0)]
        
        while to_crawl and len(documents) < self.config.max_pages:
            url, depth = to_crawl.pop(0)
            
            if url in self.visited_urls or depth > max_depth:
                continue
                
            doc = await self.crawl_url(url, extract_links=True)
            if doc:
                documents.append(doc)
                
                # Add linked pages to crawl queue if we haven't reached max depth
                if depth < max_depth and len(documents) < self.config.max_pages:
                    links = doc.metadata.get("links", [])
                    for link in links[:5]:  # Limit links per page
                        if link not in self.visited_urls:
                            to_crawl.append((link, depth + 1))
            
            # Rate limiting
            await asyncio.sleep(self.config.wait_time)
            
        return documents

class ChromaDBManager:
    """Manages ChromaDB vector database operations with embeddings"""
    
    def __init__(self, config: DatabaseConfig = None, embedding_config: EmbeddingConfig = None):
        self.config = config or DatabaseConfig()
        self.persist_directory = Path(self.config.persist_directory) 
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_config = embedding_config or EmbeddingConfig()
        
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=self.config.anonymized_telemetry,
                allow_reset=self.config.allow_reset
            )
        )
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(self.embedding_config)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric}
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector database with chunking and embeddings"""
        if not documents:
            return
        
        all_texts = []
        all_ids = []
        all_metadatas = []
        
        for doc in documents:
            # Process each chunk
            for i, chunk in enumerate(doc.chunks):
                chunk_id = f"{doc.id}_chunk_{i}"
                all_ids.append(chunk_id)
                all_texts.append(chunk['text'])
                
                # Create metadata for chunk
                chunk_metadata = {
                    "doc_id": doc.id,
                    "url": doc.url,
                    "title": doc.title,
                    "chunk_index": i,
                    "chunk_total": len(doc.chunks),
                    "chunk_summary": chunk['summary'],
                    "headers": chunk['headers'],
                    "word_count": chunk['word_count'],
                    "has_code": chunk['code_blocks'] > 0,
                    "has_lists": chunk['has_lists'],
                    "has_tables": chunk['has_tables'],
                    "timestamp": doc.timestamp.isoformat()
                }
                
                # Add document-level metadata
                for key, value in doc.metadata.items():
                    if key not in ["code_examples", "links"]:  # Skip large objects
                        chunk_metadata[f"doc_{key}"] = value
                
                all_metadatas.append(chunk_metadata)
        
        # Generate embeddings for all texts
        embeddings = self.embedding_manager.embed_texts(all_texts)
        
        # Add to ChromaDB in batches
        batch_size = self.config.batch_size
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i + batch_size]
            batch_texts = all_texts[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
        
        logger.info(f"Added {len(documents)} documents with {len(all_ids)} chunks to ChromaDB")
    
    def search(self, query: str, n_results: int = 5, keywords: List[str] = None, collection_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar document chunks using embeddings"""
        # Use specified collection or default
        if collection_name:
            collection = self.client.get_collection(collection_name)
        else:
            collection = self.collection
        # Generate query embedding with keywords if provided
        if keywords:
            query_embedding = self.embedding_manager.embed_query_with_keywords(query, keywords)
            # query_embedding = self.embedding_manager.embed_query(query)
        else:
            query_embedding = self.embedding_manager.embed_query(query)
        
        # The query_embedding should be a list at this point
        results = collection.query(
            query_embeddings=[query_embedding],  # ChromaDB expects a list of embeddings
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Rest of the method remains the same...
        
        # Convert results to more usable format
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                })
        
        return formatted_results
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=['documents', 'metadatas']
        )
        
        chunks = []
        for i in range(len(results['ids'])):
            chunks.append({
                'id': results['ids'][i],
                'content': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x['metadata'].get('chunk_index', 0))
        return chunks
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about all collections"""
        all_collections = []
        
        # Get list of all collections
        collections = self.client.list_collections()
        logger.info(f"Found {len(collections)} collections")
        
        for collection in collections:
            try:
                collection_info = {
                    "name": collection.name,
                    "count": collection.count(),
                    "metadata": collection.metadata
                }
            except Exception as e:
                logger.error(f"Error getting info for collection {collection}: {str(e)}")
                collection_info = {
                    "name": str(collection),
                    "count": 0,
                    "metadata": {},
                    "status": f"error: {str(e)}"
                }
            
            all_collections.append(collection_info)

        
        return {
            "total_collections": len(all_collections),
            "collections": all_collections
        }

class MCPCrawlerServer:
    """MCP Server for web crawling and content extraction"""
    
    def __init__(self, embedding_config: EmbeddingConfig = None, 
             crawl_config: CrawlConfig = None, 
             database_config: DatabaseConfig = None):

        self.embedding_config = embedding_config or EmbeddingConfig()
        self.crawl_config = crawl_config or CrawlConfig()
        self.database_config = database_config or DatabaseConfig()
    
        self.server = Server("web-crawler-mcp")
        self.db_manager =  ChromaDBManager(self.database_config, self.embedding_config)

        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all MCP handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools"""
            return [
                types.Tool(
                    name="crawl_website",
                    description="Crawl a website and extract content with intelligent chunking",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to start crawling from"
                            },
                            "max_pages": {
                                "type": "integer",
                                "description": "Maximum number of pages to crawl",
                                "default": 5
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum crawl depth",
                                "default": 2
                            },
                            "chunk_size": {
                                "type": "integer",
                                "description": "Target size for content chunks",
                                "default": 5000
                            },
                            "include_external": {
                                "type": "boolean",
                                "description": "Whether to include external links",
                                "default": False
                            }
                        },
                        "required": ["url"]
                    }
                ),
                types.Tool(
                    name="search_knowledge",  # Alias for search_content
                    description="Search the knowledge base using semantic search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 5
                            },
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "2-3 Keywords present in search query to emphasize in the search",
                                "default": []
                            },
                            "collection_name": {"type": "string", "description": "Collection to search in, put default as web_documents"},

                        },
                        "required": ["query", "collection_name"]
                    }
                ),
                types.Tool(
                    name="get_document",
                    description="Get all chunks for a specific document",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID to retrieve"
                            }
                        },
                        "required": ["doc_id"]
                    }
                ),
                types.Tool(
                    name="get_database_stats",  # Alias for get_crawl_stats
                    description="Get statistics about the vector database",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, 
            arguments: dict | None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool calls"""
            
            if name == "crawl_website":
                return await self._crawl_website(arguments)
            elif name == "search_knowledge":
                return await self._search_content(arguments)
            elif name == "get_document":
                return await self._get_document(arguments)
            elif name == "get_database_stats":
                return await self._get_crawl_stats()
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[types.Resource]:
            """List available resources"""
            return [
                types.Resource(
                    uri="crawl://stats",
                    name="Crawl Statistics",
                    description="Statistics about crawled content",
                    mimeType="application/json"
                ),
                types.Resource(
                    uri="crawl://recent",
                    name="Recent Crawls",
                    description="Recently crawled URLs",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a resource"""
            if uri == "crawl://stats":
                stats = self.db_manager.get_collection_info()
                return json.dumps(stats, indent=2)
            elif uri == "crawl://recent":
                # Get recent crawls from metadata
                results = self.db_manager.collection.get(
                    limit=10,
                    include=['metadatas']
                )
                recent_urls = []
                seen_urls = set()
                for metadata in results['metadatas']:
                    url = metadata.get('url')
                    if url and url not in seen_urls:
                        recent_urls.append({
                            'url': url,
                            'title': metadata.get('title', 'Untitled'),
                            'timestamp': metadata.get('timestamp', '')
                        })
                        seen_urls.add(url)
                return json.dumps(recent_urls[:5], indent=2)
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    async def _crawl_website(self, arguments: dict) -> list[types.TextContent]:
        """Handle website crawling"""
        url = arguments.get("url")
        if not url:
            raise ValueError("URL is required")
        
        # Update config with arguments
        config = CrawlConfig(
            max_pages=arguments.get("max_pages", 5),
            max_depth=arguments.get("max_depth", 2),
            chunk_size=arguments.get("chunk_size", 1000),
            include_external=arguments.get("include_external", False)
        )
        
        # Perform crawling
        async with WebCrawler(config) as crawler:
            documents = await crawler.crawl_recursive(url, max_depth=config.max_depth)
        
        if not documents:
            return [types.TextContent(
                type="text",
                text="No content could be extracted from the provided URL."
            )]
        
        # setup and Store documents in ChromaDB
        self.db_manager.add_documents(documents)
        
        # Prepare response
        total_chunks = sum(len(doc.chunks) for doc in documents)
        response_text = f"Successfully crawled {len(documents)} pages with {total_chunks} content chunks:\n\n"
        
        for doc in documents:
            response_text += f"📄 **{doc.title}**\n"
            response_text += f"   URL: {doc.url}\n"
            response_text += f"   Chunks: {len(doc.chunks)}\n"
            response_text += f"   Words: {doc.metadata.get('word_count', 0):,}\n"
            if doc.metadata.get('has_code'):
                response_text += f"   Code blocks: {doc.metadata.get('code_block_count', 0)}\n"
            response_text += "\n"
        
        response_text += f"\nTotal content stored: {total_chunks} chunks across {len(documents)} documents."
        
        return [types.TextContent(
            type="text",
            text=response_text
        )]
    
    async def _search_content(self, arguments: dict) -> list[types.TextContent]:
        """Handle content search"""
        query = arguments.get("query")
        if not query:
            raise ValueError("Query is required")
        
        n_results = arguments.get("n_results", 5)
        keywords = arguments.get("keywords", [])
        collection_name = arguments.get("collection_name", "web_documents")
        
        # Search in ChromaDB with keywords
        results = self.db_manager.search(query, n_results, keywords, collection_name)
        
        if not results:
            return [types.TextContent(
                type="text",
                text=f"No results found for query: '{query}'"
            )]
        
        # Format results
        response_text = f"🔍 Search results for: **{query}**\n\n"
        response_text += f"Found {len(results)} relevant chunks:\n\n"
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            content_preview = result['content'][:300].replace('\n', ' ')
            
            response_text += f"**{i}. {metadata.get('title', 'Untitled')}**\n"
            response_text += f"   📎 URL: {metadata.get('url', 'Unknown')}\n"
            response_text += f"   📊 Similarity: {result['similarity']:.2%}\n"
            response_text += f"   📝 Chunk: {metadata.get('chunk_index', 0) + 1}/{metadata.get('chunk_total', 1)}\n"
            
            if metadata.get('headers'):
                response_text += f"   📑 Headers: {metadata['headers']}\n"
            
            response_text += f"   💬 Preview: {content_preview}...\n\n"
        
        return [types.TextContent(
            type="text",
            text=response_text
        )]
    
    
    async def _get_document(self, arguments: dict) -> list[types.TextContent]:
        """Get all chunks for a document"""
        doc_id = arguments.get("doc_id")
        if not doc_id:
            raise ValueError("Document ID is required")
        
        # Get all chunks for the document
        chunks = self.db_manager.get_document_chunks(doc_id)
        
        if not chunks:
            return [types.TextContent(
                type="text",
                text=f"No document found with ID: {doc_id}"
            )]
        
        # Format response
        first_chunk_metadata = chunks[0]['metadata']
        response_text = f"📄 **{first_chunk_metadata.get('title', 'Untitled')}**\n"
        response_text += f"URL: {first_chunk_metadata.get('url', 'Unknown')}\n"
        response_text += f"Total chunks: {len(chunks)}\n\n"
        
        # Combine all chunks
        full_content = "\n\n---\n\n".join([
            f"### Chunk {chunk['metadata'].get('chunk_index', 0) + 1}\n{chunk['content']}"
            for chunk in chunks
        ])
        
        response_text += full_content
        
        return [types.TextContent(
            type="text",
            text=response_text
        )]
    
    async def _get_crawl_stats(self) -> list[types.TextContent]:
        """Get crawling statistics"""
        try:
            stats = self.db_manager.get_collection_info()
            
            response_text = "📊 **Database Statistics**\n\n"
            response_text += f"**Database Info:**\n"
            response_text += f"- Total collections: {stats['total_collections']}\n"
            response_text += f"- Database path: {self.db_manager.persist_directory}\n\n"
            
            response_text += f"**Collections:**\n"
            
            total_chunks_all = 0
            total_words_all = 0
            total_code_blocks_all = 0
            all_unique_domains = set()
            docs_with_code_all = 0
            
            for collection_info in stats['collections']:
                collection_name = collection_info['name']
                collection_count = collection_info['count']
                total_chunks_all += collection_count
                
                response_text += f"\n**{collection_name}:**\n"
                response_text += f"- Chunks: {collection_count}\n"
                response_text += f"- Status: {collection_info.get('status', 'active')}\n"
                
                # Get detailed stats for this collection if it has data
                if collection_count > 0:
                    try:
                        # Get the actual collection to analyze content
                        actual_collection = self.db_manager.client.get_collection(collection_name)
                        sample_results = actual_collection.get(
                            limit=min(100, collection_count),
                            include=['metadatas']
                        )
                        
                        # Analyze metadata for this collection
                        collection_words = 0
                        collection_code_blocks = 0
                        collection_domains = set()
                        collection_docs_with_code = 0
                        
                        for metadata in sample_results['metadatas']:
                            collection_words += metadata.get('word_count', 0)
                            total_words_all += metadata.get('word_count', 0)
                            
                            if metadata.get('has_code'):
                                collection_docs_with_code += 1
                                docs_with_code_all += 1
                                
                            code_blocks = metadata.get('code_blocks', 0)
                            collection_code_blocks += code_blocks
                            total_code_blocks_all += code_blocks
                            
                            url = metadata.get('url', '')
                            if url:
                                domain = urlparse(url).netloc
                                if domain:
                                    collection_domains.add(domain)
                                    all_unique_domains.add(domain)
                        
                        # Collection-specific stats
                        avg_words = collection_words / len(sample_results['metadatas']) if sample_results['metadatas'] else 0
                        response_text += f"- Avg words per chunk: {avg_words:.0f}\n"
                        response_text += f"- Chunks with code: {collection_docs_with_code}\n"
                        response_text += f"- Code blocks: {collection_code_blocks}\n"
                        response_text += f"- Domains: {len(collection_domains)}\n"
                        
                        if collection_domains:
                            response_text += f"- Domain(s): {', '.join(sorted(collection_domains))}\n"
                            
                    except Exception as e:
                        response_text += f"- Error analyzing: {str(e)}\n"
                else:
                    response_text += f"- No data\n"
            
            # Overall summary
            response_text += f"\n**Overall Summary:**\n"
            response_text += f"- Total chunks across all collections: {total_chunks_all}\n"
            response_text += f"- Total words: {total_words_all:,}\n"
            response_text += f"- Total chunks with code: {docs_with_code_all}\n"
            response_text += f"- Total code blocks: {total_code_blocks_all}\n"
            response_text += f"- Unique domains: {len(all_unique_domains)}\n"
            
            if all_unique_domains:
                response_text += f"\n**All Crawled Domains:**\n"
                for domain in sorted(all_unique_domains):
                    response_text += f"- {domain}\n"
            
            return [types.TextContent(
                type="text",
                text=response_text
            )]
            
        except Exception as e:
            logger.error(f"Error in _get_crawl_stats: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return [types.TextContent(
                type="text",
                text=f"Error getting database stats: {str(e)}"
            )]
    
    async def run(self):
        """Run the MCP server"""
        # Run the server using stdin/stdout streams
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="web-crawler-mcp",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


# Main entry point
def main():
    """Main entry point for the MCP server"""
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )
    
    # Create and run server
    server = MCPCrawlerServer()
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()