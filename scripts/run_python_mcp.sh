#!/bin/bash
set -e

# Silent script for MCP - no echo/output to stdout
# All setup should be done beforehand

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CHROMA_DB_PATH=./data/chroma_db
export PATH="$HOME/.local/bin:$PATH"

# Change to script directory
cd "$(dirname "$0")/.."

# Run the server directly with UV (assumes setup is already done)
if command -v uv &> /dev/null; then
    exec uv run python mcp_server.py
else
    # Fallback to system Python if UV not available
    exec python3 mcp_server.py
fi