# MCP RAG Server with Advanced Keyword Emphasis

A professional, scalable Model Context Protocol (MCP) server featuring **intelligent keyword emphasis** for precision-focused semantic search and web content extraction.

**🎯 Unique Feature: Keyword Emphasis** - Unlike standard semantic search, this server allows you to emphasize specific keywords in your queries, giving you precise control over search focus and dramatically improving result relevance.

## Features

- **🎯 Advanced Keyword Emphasis**: Revolutionary token-level weighting for precision-focused semantic search
- **Intelligent Web Scraping**: Uses Crawl4AI for advanced web content extraction
- **Semantic Search with Keyword Emphasis**: Powered by nomic-ai/nomic-embed-text-v1 embedding model
- **Vector Storage**: ChromaDB for efficient vector storage and retrieval
- **Precision Query Control**: Focus search on specific aspects using keyword emphasis
- **Rate Limiting**: Built-in rate limiting for responsible crawling
- **Multiple Extraction Strategies**: Optimized extractors for articles, products, documentation, and forums
- **Docker Support**: Easy deployment with Docker and docker-compose
- **Async Architecture**: Fully asynchronous for optimal performance

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MCP Client    │────▶│   MCP Server    │────▶│   Crawl4AI      │
│  (LLM/Claude)   │     │                 │     │  (Web Scraper)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                          │
                               ▼                          ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   ChromaDB      │     │    Nomic AI     │
                        │ (Vector Store)  │◀────│   Embeddings    │
                        └─────────────────┘     └─────────────────┘
```

## Installation

### Using bash script

```bash
./scripts/setup_python_mcp.sh
```

### Using Poetry

```bash
poetry install
```
### Using pip

```bash
pip install -r requirements.txt
```

### Using Docker

```bash
docker compose up -d
```
or
```bash
./scripts/setup_docker_mcp.sh
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mcp-rag-server
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run the server**
   ```bash
   ./scripts/run_python_mcp.sh
   ```

4. **Connect with MCP client**
   ```bash
   # For Claude Desktop, add to config:
   {
   "mcpServers": {
         "web-rag": {
         "command": "mcp-rag-server/scripts/run_python_mcp.sh",
         "cwd": "mcp-rag-server",
         "env": {
            "MCP_CHROMA_PATH": "mcp-rag-server/data/chroma_db"
           }
         }
      }
   }
   ```

## Available Tools

### 1. `crawl_website`
Crawl a website and store documents in the vector database.

**Parameters:**
- `url` (str): Starting URL to crawl
- `max_pages` (int): Maximum number of pages to crawl (default: 5)
- `include_external` (bool): Whether to include external links (default: false)

**Example:**
```python
await crawl_website(
    url="https://example.com",
    max_pages=2,
    include_external=False
)
```

### 2. `search_knowledge`
Search the knowledge base using semantic search with optional keyword emphasis.

**Parameters:**
- `query` (str): Search query
- `n_results` (int): Number of results to return (default: 5)
- `keywords` (list): Keywords to emphasize in search (optional)

**Example:**
```python
results = await search_knowledge(
    query="What is feature scaling in machine learning?",
    n_results=5,
    keywords=["feature", "scaling"]
)
```

### 3. `get_database_stats`
Get statistics about the vector database.

**Example:**
```python
stats = await get_database_stats()
```

## Keyword Emphasis Feature

The server supports **keyword emphasis** in search queries, allowing users to focus on specific aspects of their search. This is particularly useful when:

- **Technical Documentation**: Emphasize specific APIs, functions, or concepts
- **Product Research**: Focus on features, pricing, or specifications  
- **Academic Content**: Highlight methodologies, results, or theories
- **Troubleshooting**: Emphasize error messages, solutions, or configurations

**How it works:**
- Keywords receive 2x importance weight in the embedding space
- Uses token-level weighting during embedding generation
- Combines weighted pooling with CLS token representation
- Gracefully falls back to standard search if keyword weighting fails

**Example Use Cases:**
```python
# Focus on performance aspects
search_knowledge(
    query="database optimization techniques",
    keywords=["database", "optimization"]
)

# Emphasize specific technologies
search_knowledge(
    query="React Vue Angular web development frameworks comparison",
    keywords=["React", "Vue", "Angular"]
)
```

## Development

### Running Tests

```bash
pytest tests/
```

## Deployment

### Docker Deployment

```bash
# Build and run with docker-compose
docker compose up -d

# View logs
docker compose logs -f mcp-rag-server
```

### Production Considerations

1. **Database Persistence**: Mount ChromaDB directory as volume
2. **Scaling**: Use multiple instances with shared ChromaDB
3. **Monitoring**: Enable comprehensive logging and metrics
4. **Security**: Implement authentication for MCP endpoints
5. **Rate Limiting**: Configure appropriate limits for your use case

## Troubleshooting

### Common Issues

1. **ChromaDB Connection Error**
   - Ensure ChromaDB service is running
   - Check connection settings in config

2. **Embedding Model Download**
   - First run downloads the model (~500MB)
   - Ensure sufficient disk space

3. **Rate Limiting**
   - Adjust `max_requests_per_minute` if getting blocked
   - Add delays between requests

4. **Memory Usage**
   - Large crawls can consume significant memory
   - Use smaller `max_pages` values
   - Enable chunking for large documents

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Crawl4AI](https://github.com/unclecode/crawl4ai) for web scraping
- [Nomic AI](https://www.nomic.ai/) for embedding models
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Model Context Protocol](https://modelcontextprotocol.io/) for the MCP specification