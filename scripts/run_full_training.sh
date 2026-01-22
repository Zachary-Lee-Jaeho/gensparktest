#!/bin/bash
# Full Neural Repair Model Training Script
# =========================================
# 
# This script runs the complete training pipeline for the neural repair model.
# 
# Usage:
#   # On GPU server (recommended):
#   ./scripts/run_full_training.sh --gpu
#
#   # On CPU (very slow, hours to days):
#   ./scripts/run_full_training.sh --cpu
#
#   # Resume from checkpoint:
#   ./scripts/run_full_training.sh --gpu --resume models/repair_model/checkpoint-500
#
#   # Using Docker:
#   docker build -f Dockerfile.unified -t vega-verified .
#   docker run --gpus all -v $(pwd)/models:/app/models vega-verified ./scripts/run_full_training.sh --gpu
#

set -e

# Default values
DEVICE="auto"
EPOCHS=10
BATCH_SIZE=8
MODEL="Salesforce/codet5-base"
OUTPUT_DIR="models/repair_model"
TRAIN_SIZE=1000
RESUME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            DEVICE="cuda"
            BATCH_SIZE=16
            shift
            ;;
        --cpu)
            DEVICE="cpu"
            BATCH_SIZE=4
            shift
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --train-size)
            TRAIN_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu           Use CUDA GPU (recommended)"
            echo "  --cpu           Use CPU only (slow)"
            echo "  --resume PATH   Resume from checkpoint"
            echo "  --epochs N      Number of epochs (default: 10)"
            echo "  --batch-size N  Batch size (default: 8 for GPU, 4 for CPU)"
            echo "  --model NAME    Base model (default: Salesforce/codet5-base)"
            echo "  --output DIR    Output directory (default: models/repair_model)"
            echo "  --train-size N  Training samples (default: 1000)"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "VEGA-Verified Neural Repair Model Training"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Device:      $DEVICE"
echo "  Model:       $MODEL"
echo "  Epochs:      $EPOCHS"
echo "  Batch Size:  $BATCH_SIZE"
echo "  Train Size:  $TRAIN_SIZE"
echo "  Output:      $OUTPUT_DIR"
if [ -n "$RESUME" ]; then
    echo "  Resume from: $RESUME"
fi
echo ""

# Check for GPU
if [ "$DEVICE" = "cuda" ]; then
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "WARNING: CUDA requested but not available!"
        echo "Falling back to CPU..."
        DEVICE="cpu"
        BATCH_SIZE=4
    else
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        echo "GPU detected: $GPU_NAME"
    fi
fi

# Estimate time
if [ "$DEVICE" = "cpu" ]; then
    EST_HOURS=$((TRAIN_SIZE * EPOCHS * 2 / 3600))
    echo ""
    echo "WARNING: CPU training estimated time: ~${EST_HOURS} hours"
    echo "Consider using --gpu on a GPU server for faster training."
    echo ""
fi

# Build command
CMD="python scripts/train_neural_repair.py"
CMD="$CMD --device $DEVICE"
CMD="$CMD --model $MODEL"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --train-size $TRAIN_SIZE"

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

if [ "$DEVICE" = "cuda" ]; then
    CMD="$CMD --fp16"
fi

echo "Running: $CMD"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
exec $CMD
