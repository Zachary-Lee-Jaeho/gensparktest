#!/bin/bash
# =============================================================================
# VEGA-Verified: Paper Reproduction Script
# =============================================================================
# This script reproduces all experiments from the VEGA-Verified paper.
#
# Usage:
#   ./scripts/reproduce_experiments.sh [--all|--quick|--verification|--repair]
#
# Options:
#   --all          Run all experiments (full reproduction)
#   --quick        Run quick verification (small sample)
#   --verification Run verification experiments only
#   --repair       Run repair experiments only
#   --comparison   Run VEGA vs VEGA-Verified comparison
#   --ablation     Run ablation study
#
# Requirements:
#   - Docker installed
#   - OR Python 3.8+ with dependencies installed
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${RESULTS_DIR}/reproduction_${TIMESTAMP}.log"

# Default settings
SAMPLE_SIZE=100
SEED=42
BACKENDS="all"
USE_DOCKER=false

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

print_banner() {
    echo "============================================================================="
    echo "  VEGA-Verified: Paper Reproduction"
    echo "  Timestamp: ${TIMESTAMP}"
    echo "============================================================================="
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        log_info "Python: ${PYTHON_VERSION}"
    else
        log_error "Python 3 not found!"
        exit 1
    fi
    
    # Check Z3
    if python3 -c "import z3" 2>/dev/null; then
        Z3_VERSION=$(python3 -c "import z3; print(z3.get_version_string())")
        log_info "Z3: ${Z3_VERSION}"
    else
        log_warning "Z3 not installed (pip install z3-solver)"
    fi
    
    # Check PyTorch
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        log_info "PyTorch: ${TORCH_VERSION}"
    else
        log_warning "PyTorch not installed"
    fi
    
    # Check VEGA-Verified installation
    if python3 -c "from src.cli import main" 2>/dev/null; then
        log_info "VEGA-Verified: Installed"
    else
        log_warning "VEGA-Verified not installed as package. Running from source."
    fi
    
    log_success "Dependency check complete"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create directories
    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${PROJECT_ROOT}/data"
    mkdir -p "${PROJECT_ROOT}/logs"
    
    # Set Python path
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
    
    log_success "Environment ready"
}

# =============================================================================
# Experiment Functions
# =============================================================================

run_verification_experiment() {
    log_info "Running Verification Experiment..."
    log_info "  Sample size: ${SAMPLE_SIZE}"
    log_info "  Backends: ${BACKENDS}"
    log_info "  Seed: ${SEED}"
    
    cd "${PROJECT_ROOT}"
    
    python3 -m src.cli experiment \
        --experiment verification \
        --backend "${BACKENDS}" \
        --sample-size "${SAMPLE_SIZE}" \
        --seed "${SEED}" \
        --output "${RESULTS_DIR}" \
        --verbose 2>&1 | tee -a "$LOG_FILE"
    
    log_success "Verification experiment complete"
}

run_repair_experiment() {
    log_info "Running Repair Experiment..."
    log_info "  Sample size: ${SAMPLE_SIZE}"
    log_info "  Backends: ${BACKENDS}"
    
    cd "${PROJECT_ROOT}"
    
    python3 -m src.cli experiment \
        --experiment repair \
        --backend "${BACKENDS}" \
        --sample-size "${SAMPLE_SIZE}" \
        --seed "${SEED}" \
        --output "${RESULTS_DIR}" \
        --verbose 2>&1 | tee -a "$LOG_FILE"
    
    log_success "Repair experiment complete"
}

run_comparison_experiment() {
    log_info "Running VEGA vs VEGA-Verified Comparison..."
    
    cd "${PROJECT_ROOT}"
    
    python3 -m src.cli experiment \
        --experiment comparison \
        --backend "${BACKENDS}" \
        --sample-size "${SAMPLE_SIZE}" \
        --seed "${SEED}" \
        --output "${RESULTS_DIR}" \
        --verbose 2>&1 | tee -a "$LOG_FILE"
    
    log_success "Comparison experiment complete"
}

run_ablation_study() {
    log_info "Running Ablation Study..."
    
    cd "${PROJECT_ROOT}"
    
    python3 -m src.cli experiment \
        --experiment ablation \
        --backend "${BACKENDS}" \
        --sample-size "${SAMPLE_SIZE}" \
        --seed "${SEED}" \
        --output "${RESULTS_DIR}" \
        --verbose 2>&1 | tee -a "$LOG_FILE"
    
    log_success "Ablation study complete"
}

run_all_experiments() {
    log_info "Running ALL experiments..."
    
    cd "${PROJECT_ROOT}"
    
    python3 -m src.cli experiment \
        --all \
        --backend "${BACKENDS}" \
        --sample-size "${SAMPLE_SIZE}" \
        --seed "${SEED}" \
        --output "${RESULTS_DIR}" \
        --verbose 2>&1 | tee -a "$LOG_FILE"
    
    log_success "All experiments complete"
}

run_quick_test() {
    log_info "Running Quick Test (10 samples)..."
    
    SAMPLE_SIZE=10
    
    cd "${PROJECT_ROOT}"
    
    python3 -m src.cli experiment \
        --experiment verification \
        --backend riscv \
        --sample-size 10 \
        --seed "${SEED}" \
        --output "${RESULTS_DIR}" 2>&1 | tee -a "$LOG_FILE"
    
    log_success "Quick test complete"
}

generate_report() {
    log_info "Generating reports..."
    
    cd "${PROJECT_ROOT}"
    
    # Generate Markdown report
    python3 -m src.cli report \
        --format markdown \
        --template paper \
        --output "${RESULTS_DIR}" 2>&1 | tee -a "$LOG_FILE"
    
    # Generate LaTeX tables
    python3 -m src.cli report \
        --format latex \
        --template paper \
        --output "${RESULTS_DIR}" 2>&1 | tee -a "$LOG_FILE"
    
    log_success "Reports generated in ${RESULTS_DIR}"
}

run_tests() {
    log_info "Running test suite..."
    
    cd "${PROJECT_ROOT}"
    
    python3 -m pytest tests/ \
        -v \
        --tb=short \
        -x \
        2>&1 | tee -a "$LOG_FILE"
    
    log_success "Tests complete"
}

# =============================================================================
# Docker Functions
# =============================================================================

run_with_docker() {
    log_info "Running with Docker..."
    
    # Build image if needed
    if ! docker image inspect vega-verified &>/dev/null; then
        log_info "Building Docker image..."
        cd "${PROJECT_ROOT}"
        docker build -f Dockerfile.unified -t vega-verified .
    fi
    
    # Run experiments
    docker run --rm \
        -v "${RESULTS_DIR}:/app/results" \
        vega-verified \
        vega-verify experiment --all \
        --sample-size "${SAMPLE_SIZE}" \
        --seed "${SEED}" \
        --output /app/results
    
    log_success "Docker run complete"
}

# =============================================================================
# Main
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all           Run all experiments (default)"
    echo "  --quick         Run quick test (10 samples)"
    echo "  --verification  Run verification experiments only"
    echo "  --repair        Run repair experiments only"
    echo "  --comparison    Run VEGA vs VEGA-Verified comparison"
    echo "  --ablation      Run ablation study"
    echo "  --tests         Run test suite"
    echo "  --report        Generate reports from existing results"
    echo "  --docker        Use Docker environment"
    echo ""
    echo "  --samples N     Number of samples (default: 100)"
    echo "  --seed N        Random seed (default: 42)"
    echo "  --backend B     Backend: riscv, arm, aarch64, x86, all (default: all)"
    echo ""
    echo "Examples:"
    echo "  $0 --all                    # Full reproduction"
    echo "  $0 --quick                  # Quick test"
    echo "  $0 --verification --samples 500"
    echo "  $0 --docker --all           # Run in Docker"
}

main() {
    print_banner
    
    # Parse arguments
    ACTION="all"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                ACTION="all"
                shift
                ;;
            --quick)
                ACTION="quick"
                shift
                ;;
            --verification)
                ACTION="verification"
                shift
                ;;
            --repair)
                ACTION="repair"
                shift
                ;;
            --comparison)
                ACTION="comparison"
                shift
                ;;
            --ablation)
                ACTION="ablation"
                shift
                ;;
            --tests)
                ACTION="tests"
                shift
                ;;
            --report)
                ACTION="report"
                shift
                ;;
            --docker)
                USE_DOCKER=true
                shift
                ;;
            --samples)
                SAMPLE_SIZE="$2"
                shift 2
                ;;
            --seed)
                SEED="$2"
                shift 2
                ;;
            --backend)
                BACKENDS="$2"
                shift 2
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Setup
    setup_environment
    
    if [ "$USE_DOCKER" = true ]; then
        run_with_docker
        exit 0
    fi
    
    check_dependencies
    
    # Run selected action
    case $ACTION in
        all)
            run_all_experiments
            generate_report
            ;;
        quick)
            run_quick_test
            ;;
        verification)
            run_verification_experiment
            ;;
        repair)
            run_repair_experiment
            ;;
        comparison)
            run_comparison_experiment
            ;;
        ablation)
            run_ablation_study
            ;;
        tests)
            run_tests
            ;;
        report)
            generate_report
            ;;
    esac
    
    # Summary
    echo ""
    echo "============================================================================="
    log_success "Reproduction complete!"
    echo "  Results: ${RESULTS_DIR}"
    echo "  Log: ${LOG_FILE}"
    echo "============================================================================="
}

main "$@"
