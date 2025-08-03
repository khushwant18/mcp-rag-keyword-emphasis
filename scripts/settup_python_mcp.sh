#!/bin/bash
set -e

echo "=== MCP RAG Server Setup ==="

# Check if UV is installed, install if not
check_and_install_uv() {
    if command -v uv &> /dev/null; then
        echo "UV is already installed: $(uv --version)"
        return
    fi
    
    echo "UV not found. Installing UV..."
    
    # Install UV based on platform
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        # Windows
        if grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
            # WSL - use Linux method
            curl -LsSf https://astral.sh/uv/install.sh | sh
        else
            # Regular Windows
            echo "For Windows, please run this in PowerShell:"
            echo "powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
            echo "Then restart your terminal and run this script again."
            exit 1
        fi
    else
        # Linux and macOS
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    
    # Add UV to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Add to shell profile
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    else
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
    
    # Verify UV installation
    if ! command -v uv &> /dev/null; then
        echo "UV installation failed. Please install manually:"
        echo "Visit: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    
    echo "UV installed successfully: $(uv --version)"
}

# Main execution
check_and_install_uv

echo "=== Setting up MCP RAG Server ==="

# Create directories
mkdir -p data/chroma_db logs

# Install Python if needed (UV manages this)
echo "Ensuring Python is available..."
uv python install 3.11

# Install dependencies
echo "Installing dependencies..."
if [[ -f "pyproject.toml" ]]; then
    echo "pyproject.toml found. Installing dependencies directly..."
    uv pip install \
        "mcp[cli]>=1.0.0" \
        "crawl4ai>=0.3.0" \
        "chromadb>=0.4.0" \
        "sentence-transformers>=2.5.0" \
        "beautifulsoup4>=4.12.0" \
        "aiohttp>=3.9.0" \
        "numpy>=1.24.0" \
        "python-dotenv>=1.0.0" \
        "einops>=0.7.0" \
        "torch>=2.0.0" \
        "playwright>=1.40.0"
else
    echo "No pyproject.toml found. Please create one with your dependencies."
    exit 1
fi

# Download embedding model
echo "Downloading embedding model..."
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)"

echo "Setup complete!"
echo ""
echo "To run the MCP server, use:"
echo "  uv run python mcp_server.py"
echo ""
echo "Or add this to your MCP config:"
echo '  "command": "uv",'
echo '  "args": ["run", "python", "mcp_server.py"]'