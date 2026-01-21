#!/bin/bash
# Build and run VEGA Docker environment

set -e

IMAGE_NAME="vega-reproduction"
CONTAINER_NAME="vega-env"

echo "=========================================="
echo "Building VEGA Reproduction Docker Image"
echo "=========================================="

# Build the Docker image
docker build -t ${IMAGE_NAME} .

echo "=========================================="
echo "Docker image built successfully!"
echo "=========================================="

# Check if container already exists
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Removing existing container..."
    docker rm -f ${CONTAINER_NAME}
fi

echo "=========================================="
echo "Starting VEGA container..."
echo "=========================================="

# Run container with GPU support (if available)
if command -v nvidia-smi &> /dev/null; then
    docker run -it --gpus all \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/tests:/workspace/tests \
        -v $(pwd)/results:/workspace/results \
        ${IMAGE_NAME}
else
    echo "Warning: GPU not detected, running without GPU support"
    docker run -it \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/tests:/workspace/tests \
        -v $(pwd)/results:/workspace/results \
        ${IMAGE_NAME}
fi
