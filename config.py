import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from typing import Set

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = field(
        default_factory=lambda: os.getenv('MCP_EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1')
    )
    trust_remote_code: bool = True
    max_seq_length: Optional[int] = None
    keyword_importance_multiplier: float = 2.0
    weighted_cls_blend_ratio: float = 0.5
    device = "cpu"

@dataclass
class CrawlConfig:
    """Configuration for web crawling"""
    max_depth: int = 2
    max_pages: int = 10
    chunk_size: int = 1000
    chunk_overlap: int = 50
    include_external: bool = False
    wait_time: float = 1.0
    batch_size: int = 100
    excluded_tags: List[str] = field(default_factory=lambda: [
        'nav', 'header', 'footer', 'aside', 'script', 'style'
    ])
    headers: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (compatible; MCPCrawler/1.0)'
    })
    excluded_paths: Set[str] = field(default_factory=lambda: {
        '/internal', '/external', 'internal', 'external'
    })

@dataclass
class DatabaseConfig:
    """Configuration for ChromaDB"""
    persist_directory: str = field(
        default_factory=lambda: os.getenv('MCP_CHROMA_PATH', './chroma_db')
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv('MCP_COLLECTION_NAME', 'web_documents')
    )
    distance_metric: str = "cosine"
    anonymized_telemetry: bool = False
    allow_reset: bool = True
    batch_size: int = 100  # Add this for consistency