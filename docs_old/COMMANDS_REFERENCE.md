# VEGA-Verified: Complete Commands Reference

**Version**: 1.0.0  
**Last Updated**: 2026-01-22

This document provides a comprehensive reference for all commands available in VEGA-Verified, covering training, testing, evaluation, and deployment scenarios for both CPU and GPU environments.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Testing Commands](#testing-commands)
4. [Training Commands](#training-commands)
5. [Verification & Repair Commands](#verification--repair-commands)
6. [Experiment & Evaluation Commands](#experiment--evaluation-commands)
7. [Docker Commands](#docker-commands)
8. [Development Commands](#development-commands)

---

## Quick Start

### Minimal Test Run (CPU)
```bash
# Install dependencies
pip install -r requirements.txt

# Run all core tests
python -m pytest tests/test_phase1_infrastructure.py tests/test_phase2_complete.py -v

# Quick neural test (10 samples, 1 epoch)
python scripts/train_neural_repair.py --test-only
```

### Minimal Test Run (GPU)
```bash
# Install GPU dependencies (CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install transformers accelerate

# Quick neural test with GPU
python scripts/train_neural_repair.py --test-only --device cuda
```

---

## Environment Setup

### CPU Environment

```bash
# 1. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install PyTorch CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install transformers and training dependencies
pip install transformers accelerate datasets

# 5. Install VEGA-Verified as package
pip install -e .

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from src.verification import Verifier; print('Verifier: OK')"
python -c "from src.repair import NeuralRepairEngine; print('Neural: OK')"
```

### GPU Environment (NVIDIA CUDA 13.0)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install PyTorch with CUDA 13.0 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 4. Install transformers and training dependencies
pip install transformers accelerate datasets

# 5. Install VEGA-Verified
pip install -e .

# 6. Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### GPU Environment (Apple Silicon MPS)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch for MPS
pip install torch torchvision torchaudio

# 4. Verify MPS availability
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

---

## Testing Commands

### Unit Tests

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run specific unit test modules
python -m pytest tests/unit/test_verification.py -v
python -m pytest tests/unit/test_specification.py -v
python -m pytest tests/unit/test_bmc.py -v

# Run with coverage
python -m pytest tests/unit/ -v --cov=src --cov-report=html
```

### Integration Tests

```bash
# Run all integration tests
python -m pytest tests/integration/ -v

# Run specific integration test files
python -m pytest tests/integration/test_pipeline.py -v
python -m pytest tests/integration/test_verification_pipeline.py -v
python -m pytest tests/integration/test_hierarchical_verification.py -v
python -m pytest tests/integration/test_comprehensive_pipeline.py -v

# Run with verbose output
python -m pytest tests/integration/ -v -s
```

### Phase Tests (Core Functionality)

```bash
# Phase 1: Infrastructure tests (parsing, basic verification)
python -m pytest tests/test_phase1_infrastructure.py -v

# Phase 2: Complete integration tests (CGNR, neural, hierarchical)
python -m pytest tests/test_phase2_complete.py -v

# Both core phases
python -m pytest tests/test_phase1_infrastructure.py tests/test_phase2_complete.py -v
```

### All Tests Combined

```bash
# Run all tests
python -m pytest tests/ -v

# Run all tests with parallel execution (faster)
pip install pytest-xdist
python -m pytest tests/ -v -n auto

# Run all tests with detailed failure output
python -m pytest tests/ -v --tb=long

# Run tests matching a pattern
python -m pytest tests/ -v -k "verification"
python -m pytest tests/ -v -k "neural"
python -m pytest tests/ -v -k "repair"
```

### Quick Sanity Checks

```bash
# Check imports work
python -c "from src.verification import Verifier, SMTSolver"
python -c "from src.specification import Specification, SpecificationInferrer"
python -c "from src.repair import CGNREngine, NeuralRepairEngine"
python -c "from src.hierarchical import HierarchicalVerifier"
python -c "from src.integration import VEGAVerifiedPipeline"

# Check CLI works
python -m src.cli --help
python -m src.cli status
```

---

## Training Commands

### Quick Training Test (CPU)

```bash
# Minimal test: 10 samples, 1 epoch (< 1 minute)
python scripts/train_neural_repair.py --test-only

# Slightly larger test: 50 samples, 2 epochs
python scripts/train_neural_repair.py \
    --train-size 50 \
    --epochs 2 \
    --batch-size 4 \
    --device cpu
```

### Quick Training Test (GPU)

```bash
# Minimal GPU test
python scripts/train_neural_repair.py --test-only --device cuda

# Larger GPU test
python scripts/train_neural_repair.py \
    --train-size 100 \
    --epochs 3 \
    --batch-size 16 \
    --device cuda \
    --fp16
```

### Full Training (CPU) - SLOW

```bash
# WARNING: This can take hours on CPU!

# Full training with shell script
./scripts/run_full_training.sh --cpu --epochs 10

# Or direct Python command
python scripts/train_neural_repair.py \
    --device cpu \
    --model Salesforce/codet5-base \
    --epochs 10 \
    --batch-size 4 \
    --train-size 1000 \
    --output-dir models/repair_model
```

### Full Training (GPU) - RECOMMENDED

```bash
# Full training with shell script (recommended)
./scripts/run_full_training.sh --gpu --epochs 10

# Or direct Python command
python scripts/train_neural_repair.py \
    --device cuda \
    --model Salesforce/codet5-base \
    --epochs 10 \
    --batch-size 16 \
    --train-size 1000 \
    --output-dir models/repair_model \
    --fp16

# With larger model
python scripts/train_neural_repair.py \
    --device cuda \
    --model Salesforce/codet5-large \
    --epochs 10 \
    --batch-size 8 \
    --train-size 2000 \
    --fp16
```

### Resume Training from Checkpoint

```bash
# Resume from latest checkpoint
python scripts/train_neural_repair.py \
    --resume models/repair_model/checkpoint-latest \
    --epochs 5

# Resume from specific checkpoint
python scripts/train_neural_repair.py \
    --resume models/repair_model/checkpoint-500 \
    --device cuda \
    --epochs 10

# Using shell script
./scripts/run_full_training.sh --gpu --resume models/repair_model/checkpoint-500
```

### Training with Custom Parameters

```bash
# Small model (faster, less memory)
python scripts/train_neural_repair.py \
    --model Salesforce/codet5-small \
    --batch-size 32 \
    --epochs 15 \
    --device cuda

# Base model (balanced)
python scripts/train_neural_repair.py \
    --model Salesforce/codet5-base \
    --batch-size 16 \
    --epochs 10 \
    --device cuda

# Large model (best quality, more memory)
python scripts/train_neural_repair.py \
    --model Salesforce/codet5-large \
    --batch-size 4 \
    --epochs 10 \
    --device cuda \
    --fp16
```

### Training Script Options Reference

```
python scripts/train_neural_repair.py --help

Options:
  --device {auto,cpu,cuda,mps}  Device to use (default: auto)
  --model MODEL                  HuggingFace model name (default: Salesforce/codet5-base)
  --epochs N                     Number of training epochs (default: 5)
  --batch-size N                 Training batch size (default: 8)
  --train-size N                 Number of training samples (default: 100)
  --output-dir DIR               Output directory (default: models/repair_model)
  --resume CHECKPOINT            Resume from checkpoint path
  --fp16                         Use FP16 mixed precision (GPU only)
  --test-only                    Run minimal test (10 samples, 1 epoch)
  --learning-rate LR             Learning rate (default: 5e-5)
```

---

## Verification & Repair Commands

### Verify a Function

```bash
# Verify with specification file
vega-verify verify --code function.cpp --spec spec.json

# Alternative using Python module
python -m src.cli verify --code function.cpp --spec spec.json

# Auto-infer specification
vega-verify verify --code function.cpp --infer-spec --backend riscv

# With timeout
vega-verify verify --code function.cpp --spec spec.json --timeout 60000

# Verbose output
vega-verify verify --code function.cpp --spec spec.json -v
```

### Repair a Buggy Function

```bash
# Repair with CGNR
vega-verify repair --code buggy.cpp --spec spec.json

# Specify repair strategy
vega-verify repair --code buggy.cpp --spec spec.json --strategy template
vega-verify repair --code buggy.cpp --spec spec.json --strategy neural
vega-verify repair --code buggy.cpp --spec spec.json --strategy hybrid

# Limit iterations
vega-verify repair --code buggy.cpp --spec spec.json --max-iterations 10

# Save repaired code
vega-verify repair --code buggy.cpp --spec spec.json --save-repaired

# Specify backend
vega-verify repair --code buggy.cpp --spec spec.json --backend riscv
```

### Using Python API Directly

```python
# Verification
from src.verification import Verifier
from src.specification import Specification

verifier = Verifier(timeout_ms=30000)
spec = Specification(function_name="getRelocType")
result = verifier.verify(code, spec)
print(f"Verified: {result.is_verified()}")

# Repair
from src.repair import CGNREngine

cgnr = CGNREngine(verifier=verifier, max_iterations=5)
repair_result = cgnr.repair(buggy_code, spec)
print(f"Repaired: {repair_result.is_successful()}")
if repair_result.is_successful():
    print(repair_result.repaired_code)
```

---

## Experiment & Evaluation Commands

### Run All Experiments

```bash
# Full reproduction (all experiments)
vega-verify experiment --all

# Using shell script
./scripts/reproduce_experiments.sh --all
```

### Run Specific Experiments

```bash
# Verification experiment only
vega-verify experiment --experiment verification

# Repair experiment only
vega-verify experiment --experiment repair

# Comparison experiment (VEGA vs VEGA-Verified)
vega-verify experiment --experiment comparison

# Ablation study
vega-verify experiment --experiment ablation
```

### 🔥 학습된 Neural 모델로 실험 (중요!)

GPU 서버에서 모델을 학습한 후, 학습된 모델을 사용하여 실험을 실행합니다:

```bash
# 학습된 모델 경로 지정 (기본값: models/repair_model/final)
vega-verify experiment --experiment repair --model-path models/repair_model/final

# GPU에서 실행
vega-verify experiment --experiment repair --model-path models/repair_model/final --device cuda

# CPU에서 실행 (느림)
vega-verify experiment --experiment repair --model-path models/repair_model/final --device cpu

# 전체 실험 + 학습된 모델
vega-verify experiment --all --model-path models/repair_model/final --device cuda

# 큰 샘플 크기로 실험
vega-verify experiment --experiment repair --model-path models/repair_model/final --device cuda --sample-size 500
```

**참고**: `--model-path`를 지정하지 않으면 rule-based fallback을 사용합니다.

### Experiment with Specific Backends

```bash
# RISC-V backend only
vega-verify experiment --experiment verification --backend riscv

# ARM backend only
vega-verify experiment --experiment verification --backend arm

# x86 backend only
vega-verify experiment --experiment verification --backend x86

# All backends
vega-verify experiment --experiment verification --backend all
```

### Experiment with Sample Size

```bash
# Quick test (100 functions)
vega-verify experiment --experiment verification --sample-size 100

# Medium test (500 functions)
vega-verify experiment --experiment verification --sample-size 500

# Full evaluation (all functions)
vega-verify experiment --experiment verification --sample-size 0
```

### Reproducible Experiments

```bash
# Set random seed for reproducibility
vega-verify experiment --all --seed 42

# Using shell script with options
./scripts/reproduce_experiments.sh --all --seed 42 --sample-size 500
```

### Generate Reports

```bash
# Generate markdown report
vega-verify report --format markdown

# Generate JSON report
vega-verify report --format json

# Generate HTML report (if available)
vega-verify report --format html

# Specify output file
vega-verify report --format markdown --output results/report.md
```

### Show System Status

```bash
# Show current configuration and status
vega-verify status
python -m src.cli status
```

---

## Docker Commands

### Build Docker Images

```bash
# CPU Docker image
docker build -f Dockerfile.unified -t vega-verified:cpu .

# GPU Docker image (requires NVIDIA base)
docker build -f Dockerfile.gpu -t vega-verified:gpu .

# Light image (minimal dependencies)
docker build -f Dockerfile.light -t vega-verified:light .
```

### Run Tests in Docker

```bash
# Run all tests (CPU)
docker run --rm vega-verified:cpu python -m pytest tests/ -v

# Run core tests only
docker run --rm vega-verified:cpu python -m pytest \
    tests/test_phase1_infrastructure.py \
    tests/test_phase2_complete.py -v

# Run integration tests
docker run --rm vega-verified:cpu python -m pytest tests/integration/ -v

# Interactive shell for debugging
docker run -it --rm vega-verified:cpu /bin/bash
```

### Run Training in Docker (CPU)

```bash
# Quick training test
docker run --rm vega-verified:cpu \
    python scripts/train_neural_repair.py --test-only

# Full CPU training (mount volume for model output)
docker run --rm \
    -v $(pwd)/models:/app/models \
    vega-verified:cpu \
    ./scripts/run_full_training.sh --cpu --epochs 5
```

### Run Training in Docker (GPU)

```bash
# Quick GPU test
docker run --rm --gpus all vega-verified:gpu \
    python scripts/train_neural_repair.py --test-only --device cuda

# Full GPU training
docker run --rm --gpus all \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    vega-verified:gpu \
    ./scripts/run_full_training.sh --gpu --epochs 10

# Resume training from checkpoint
docker run --rm --gpus all \
    -v $(pwd)/models:/app/models \
    vega-verified:gpu \
    python scripts/train_neural_repair.py \
        --resume /app/models/repair_model/checkpoint-500 \
        --device cuda \
        --epochs 10
```

### Run Experiments in Docker

```bash
# Run all experiments (CPU)
docker run --rm \
    -v $(pwd)/results:/app/results \
    vega-verified:cpu \
    vega-verify experiment --all

# Run all experiments (GPU)
docker run --rm --gpus all \
    -v $(pwd)/results:/app/results \
    vega-verified:gpu \
    vega-verify experiment --all

# Run specific experiment
docker run --rm \
    -v $(pwd)/results:/app/results \
    vega-verified:cpu \
    vega-verify experiment --experiment verification --backend riscv
```

### Run Verification/Repair in Docker

```bash
# Verify a function
docker run --rm \
    -v $(pwd)/my_code:/app/input \
    vega-verified:cpu \
    vega-verify verify --code /app/input/function.cpp --infer-spec

# Repair a function
docker run --rm \
    -v $(pwd)/my_code:/app/input \
    -v $(pwd)/output:/app/output \
    vega-verified:cpu \
    vega-verify repair \
        --code /app/input/buggy.cpp \
        --spec /app/input/spec.json \
        --save-repaired
```

### Docker Compose (for complex setups)

```yaml
# docker-compose.yml
version: '3.8'
services:
  vega-verified:
    build:
      context: .
      dockerfile: Dockerfile.unified
    volumes:
      - ./models:/app/models
      - ./results:/app/results
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
    command: vega-verify experiment --all

  vega-verified-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    volumes:
      - ./models:/app/models
      - ./results:/app/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: ./scripts/run_full_training.sh --gpu
```

```bash
# Run with docker-compose
docker-compose up vega-verified
docker-compose up vega-verified-gpu
```

---

## Development Commands

### Code Quality

```bash
# Format code
pip install black isort
black src/ tests/
isort src/ tests/

# Type checking
pip install mypy
mypy src/

# Linting
pip install flake8
flake8 src/ tests/
```

### Documentation

```bash
# Generate API documentation (if using sphinx)
pip install sphinx sphinx-rtd-theme
cd docs && make html
```

### Package Management

```bash
# Install in development mode
pip install -e .

# Install all development dependencies
pip install -e ".[dev]"

# Update requirements
pip freeze > requirements.txt
```

### Debugging

```bash
# Run with Python debugger
python -m pdb -m pytest tests/test_phase1_infrastructure.py -v

# Run single test with verbose output
python -m pytest tests/integration/test_pipeline.py::TestPipelineConfig -v -s

# Check for memory leaks
pip install memory_profiler
python -m memory_profiler scripts/train_neural_repair.py --test-only
```

---

## Command Summary Table

| Task | CPU Command | GPU Command |
|------|-------------|-------------|
| **Quick Test** | `python scripts/train_neural_repair.py --test-only` | `python scripts/train_neural_repair.py --test-only --device cuda` |
| **Unit Tests** | `python -m pytest tests/unit/ -v` | Same |
| **Integration Tests** | `python -m pytest tests/integration/ -v` | Same |
| **All Tests** | `python -m pytest tests/ -v` | Same |
| **Full Training** | `./scripts/run_full_training.sh --cpu` | `./scripts/run_full_training.sh --gpu` |
| **Resume Training** | `--resume models/repair_model/checkpoint-X` | Same + `--device cuda` |
| **All Experiments** | `vega-verify experiment --all` | Same (with `--gpus all` in Docker) |
| **Verify Function** | `vega-verify verify --code f.cpp --spec s.json` | Same |
| **Repair Function** | `vega-verify repair --code f.cpp --spec s.json` | Same (neural uses GPU if available) |
| **Docker Tests** | `docker run vega-verified:cpu pytest tests/ -v` | `docker run --gpus all vega-verified:gpu pytest tests/ -v` |
| **Docker Training** | `docker run -v ./models:/app/models vega-verified:cpu ./scripts/run_full_training.sh --cpu` | `docker run --gpus all -v ./models:/app/models vega-verified:gpu ./scripts/run_full_training.sh --gpu` |

---

## Troubleshooting

### Common Issues

1. **CUDA not available**
   ```bash
   # Check CUDA installation
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Reinstall PyTorch with correct CUDA version
   pip uninstall torch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   ```

2. **Out of memory (GPU)**
   ```bash
   # Reduce batch size
   python scripts/train_neural_repair.py --batch-size 4 --device cuda
   
   # Use smaller model
   python scripts/train_neural_repair.py --model Salesforce/codet5-small
   ```

3. **Import errors**
   ```bash
   # Ensure package is installed
   pip install -e .
   
   # Check PYTHONPATH
   export PYTHONPATH="${PWD}:${PYTHONPATH}"
   ```

4. **Docker GPU not working**
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

---

## Environment Variables

```bash
# Optional configuration via environment variables
export VEGA_LOG_LEVEL=DEBUG           # Logging level
export VEGA_TIMEOUT_MS=60000          # Default timeout
export VEGA_OUTPUT_DIR=results        # Output directory
export CUDA_VISIBLE_DEVICES=0         # Limit GPU usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory management
```

---

## Further Reading

- [README.md](../README.md) - Project overview
- [IMPLEMENTATION_TASKS_100_PERCENT.md](./IMPLEMENTATION_TASKS_100_PERCENT.md) - Implementation details
- [IMPLEMENTATION_VS_DESIGN_REPORT.md](./IMPLEMENTATION_VS_DESIGN_REPORT.md) - Design comparison
