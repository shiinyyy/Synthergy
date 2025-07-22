#!/bin/bash

# Test Docker container locally before Cloud Run deployment
set -e

PROJECT_ID="synthergy-0"
SERVICE_NAME="synthergy"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🧪 Testing Synthergy Docker container locally..."
echo "⚠️  Note: Testing linux/amd64 image on ARM Mac (performance may be slower)"
echo "💡 Alternative: Skip local test and deploy directly with './deploy.sh'"
echo ""

# Build the image with explicit platform
echo "🔨 Building Docker image for linux/amd64..."
docker build --platform linux/amd64 -t $IMAGE_NAME . || {
    echo "❌ Docker build failed"
    exit 1
}

echo "✅ Docker image built successfully"

# Test container startup
echo "🚀 Testing container startup..."
echo "Starting container on port 8080..."

# Kill any existing container on port 8080
docker ps -q --filter "publish=8080" | xargs -r docker kill 2>/dev/null || true

# Run container in background with explicit platform
CONTAINER_ID=$(docker run -d -p 8080:8080 --platform linux/amd64 $IMAGE_NAME)

echo "Container ID: $CONTAINER_ID"

# Wait for container to start (longer wait for cross-platform)
echo "⏳ Waiting for container to start (60 seconds - cross-platform emulation is slower)..."
sleep 60

# Test if container is responding
echo "🔍 Testing container health..."
if curl -s -f http://localhost:8080 > /dev/null; then
    echo "✅ Container is responding on port 8080"
    echo "🌐 Test URL: http://localhost:8080"
    
    # Show container logs
    echo ""
    echo "📋 Container logs (last 20 lines):"
    docker logs --tail 20 $CONTAINER_ID
    
    echo ""
    echo "✅ Local test completed successfully!"
    echo "🚀 Ready for Cloud Run deployment"
    echo ""
    echo "Commands:"
    echo "  View app: http://localhost:8080"
    echo "  Stop container: docker kill $CONTAINER_ID"
    echo "  Deploy: ./deploy.sh"
    
else
    echo "❌ Container not responding"
    echo "📋 Container logs:"
    docker logs $CONTAINER_ID
    
    echo "🔧 Troubleshooting:"
    echo "  1. Check if Streamlit is starting correctly"
    echo "  2. Verify port 8080 is exposed"  
    echo "  3. Check for configuration errors"
    echo "  4. Cross-platform emulation issues (ARM Mac running linux/amd64)"
    echo "  5. Try extending wait time or check: docker logs $CONTAINER_ID"
    
    # Cleanup
    docker kill $CONTAINER_ID
    exit 1
fi 