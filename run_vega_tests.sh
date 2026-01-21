#!/bin/bash
# Run VEGA tests in Docker container

set -e

IMAGE_NAME="vega-light"
CONTAINER_NAME="vega-test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "VEGA Reproduction Test Suite"
echo "=========================================="

# Build lightweight Docker image
echo "[1/4] Building Docker image..."
docker build -t ${IMAGE_NAME} -f Dockerfile.light . 2>&1 | tail -5

# Remove existing container if exists
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Run tests
echo ""
echo "[2/4] Running VEGA simulator..."
docker run --rm --name ${CONTAINER_NAME} \
    -v "${SCRIPT_DIR}/tests:/workspace/tests" \
    -v "${SCRIPT_DIR}/results:/workspace/results" \
    ${IMAGE_NAME} \
    bash -c "cd /workspace && python3 tests/vega_simulator.py" 2>&1

echo ""
echo "[3/4] Compiling MatMul test (if compiler available)..."
docker run --rm --name ${CONTAINER_NAME} \
    -v "${SCRIPT_DIR}/tests:/workspace/tests" \
    -v "${SCRIPT_DIR}/results:/workspace/results" \
    ${IMAGE_NAME} \
    bash -c "
        cd /workspace/tests && \
        if command -v g++ &> /dev/null; then
            g++ -O2 -std=c++17 matmul_test.cpp -o /workspace/results/matmul_test && \
            echo 'MatMul test compiled successfully' && \
            /workspace/results/matmul_test
        else
            echo 'g++ not available, skipping compilation'
        fi
    " 2>&1

echo ""
echo "[4/4] Checking VEGA repository structure..."
docker run --rm --name ${CONTAINER_NAME} \
    -v "${SCRIPT_DIR}/tests:/workspace/tests" \
    -v "${SCRIPT_DIR}/results:/workspace/results" \
    ${IMAGE_NAME} \
    bash -c "
        cd /workspace/VEGA_AE && \
        echo 'VEGA_AE Directory Structure:' && \
        ls -la && \
        echo '' && \
        echo 'Dataset files:' && \
        ls -la dataset/ 2>/dev/null || echo 'Dataset directory not found' && \
        echo '' && \
        echo 'Scripts:' && \
        ls -la Scripts/ 2>/dev/null || echo 'Scripts directory not found'
    " 2>&1

echo ""
echo "=========================================="
echo "Test suite completed!"
echo "Results saved to: ${SCRIPT_DIR}/results/"
echo "=========================================="
