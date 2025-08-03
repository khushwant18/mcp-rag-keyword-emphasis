#!/bin/bash
set -e

echo "====== Deploying MCP RAG Server with Docker ======"

# Clean up Docker resources first to free up space
echo "Cleaning up Docker resources..."
sudo docker system prune -f
sudo docker volume prune -f --filter "label!=keep"

# Stop and remove the existing container
echo "Stopping existing containers..."
sudo docker compose down --remove-orphans

# Build the new image
echo "Building Docker image (this may take several minutes)..."
sudo docker compose build --no-cache

# Start the container
echo "Starting container..."
sudo docker compose up -d

# Wait for container to be ready
echo "Waiting for container to initialize..."
for i in {1..30}; do
    if sudo docker compose ps | grep -q "mcp-rag-server.*Up"; then
        echo "[OK] Container is up and running"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "[ERROR] Timed out waiting for container to start"
        sudo docker compose logs mcp-rag-server
        exit 1
    fi
    
    echo -n "."
    sleep 2
done

# Look for specific log messages that indicate readiness
echo ""
echo "Checking for service readiness..."
sleep 10  # Give the application some time to initialize

# Show logs to verify proper startup
echo ""
echo "Recent logs:"
sudo docker compose logs --tail=20 mcp-rag-server

# Verify the service is responding
echo ""
echo "Verifying service health..."
if sudo docker compose exec mcp-rag-server ps aux | grep -q "python.*mcp_server.py"; then
    echo "[SUCCESS] Service is running properly"
    
    # Check for specific requirements
    if sudo docker compose exec mcp-rag-server python -c "import playwright" 2>/dev/null; then
        echo "[SUCCESS] Playwright is installed"
    else
        echo "[WARNING] Playwright may not be installed correctly"
    fi
    
    if sudo docker compose exec mcp-rag-server bash -c "ls -la /root/.cache/ms-playwright" 2>/dev/null; then
        echo "[SUCCESS] Playwright browsers are installed"
    else
        echo "[WARNING] Playwright browsers may not be installed"
        echo "Consider running: sudo docker compose exec mcp-rag-server python -m playwright install chromium"
    fi
    
    echo ""
    echo "===== Deployment Summary ====="
    echo "Status: Running"
    echo "Container: mcp-rag-server"
    echo "Port: 8000"
    echo ""
    echo "Commands:"
    echo "  - View logs: sudo docker compose logs -f mcp-rag-server"
    echo "  - Connect to container: sudo docker compose exec mcp-rag-server bash"
    echo "  - Stop service: sudo docker compose down"
    echo "  - Test service: python test_docker.py"
    echo "============================="
else
    echo "[ERROR] Service process not found"
    sudo docker compose logs mcp-rag-server
    exit 1
fi