# VEGA-Verified: Semantically Verified Neural Compiler Backend Generation

[![Tests](https://img.shields.io/badge/tests-123%20passing-brightgreen)]()
[![Phase](https://img.shields.io/badge/phase-2%20complete-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

> **Paper Artifact**: This repository contains the implementation and reproduction materials for the VEGA-Verified system.

---

## ⚠️ IMPORTANT: Implementation Status & Limitations

**This section MUST be read before using the system or citing results.**

### Critical Mock/Placeholder Components

The following components operate in **mock/simulation mode** and do NOT represent fully functional implementations:

| Component | Status | Impact | Details |
|-----------|--------|--------|---------|
| **Neural Repair Model** | 🔴 Mock | Critical | `src/repair/model_finetuning.py` - Model NOT trained; uses template-based rules |
| **VEGA Model Adapter** | 🔴 Mock | Critical | `src/integration/vega_adapter.py` - Simulation mode; no real VEGA model |
| **Transformer Repair** | 🔴 Mock | Critical | `src/repair/neural_model.py` - Returns empty results without transformers |
| **Z3 SMT Verification** | 🟠 Conditional | Major | `src/verification/switch_verifier.py` - Falls back to pattern matching if Z3 unavailable |
| **CGNR Pipeline** | 🟠 Partial | Major | `src/integration/cgnr_pipeline.py` - Works but uses mock repair model |
| **Spec Validation** | 🟡 Placeholder | Minor | `src/specification/spec_language.py` - `validate()` always returns True |

### What IS Fully Implemented

| Component | Status | Files |
|-----------|--------|-------|
| Semantic Analyzer | ✅ Complete | `src/verification/semantic_analyzer.py` |
| IR to SMT Converter | ✅ Complete | `src/verification/ir_to_smt.py` |
| Training Data Generator | ✅ Complete | `src/repair/training_data.py` |
| Docker/LLVM Infrastructure | ✅ Complete | `docker/Dockerfile.llvm`, `docker/tools/` |
| Switch Verifier (Pattern) | ✅ Complete | `src/verification/switch_verifier.py` |
| Specification Language | ✅ Complete | `src/specification/spec_language.py` |

### Paper Writing Disclosure Requirements

When writing papers using this codebase, you **MUST** disclose:

1. **Limitations Section**: Neural repair operates in template-based mode (not trained)
2. **Experimental Setup**: GPU-free execution uses mock neural components  
3. **Threats to Validity**: Internal validity affected by mock-based evaluation

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [System Overview](#-system-overview)
- [Installation](#-installation)
- [CLI Usage](#-cli-usage)
- [Paper Reproduction](#-paper-reproduction)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [References](#-references)

---

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Build the unified Docker image
docker build -f Dockerfile.unified -t vega-verified .

# Run all experiments
docker run -it --rm -v $(pwd)/results:/app/results vega-verified \
    vega-verify experiment --all

# Interactive shell
docker run -it --rm vega-verified /bin/bash
```

### Using Python Directly

```bash
# Install dependencies
pip install -e .

# Check system status
vega-verify status

# Run quick test
vega-verify experiment --experiment verification --sample-size 10

# Run all experiments
vega-verify experiment --all
```

---

## 🔬 System Overview

VEGA-Verified extends the VEGA neural compiler backend generator with formal verification capabilities.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VEGA-Verified Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │    LLVM      │───▶│   Semantic   │───▶│     SMT      │          │
│  │  Extractor   │    │   Analyzer   │    │   Verifier   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Function    │    │  IR/Pattern  │    │ Counterexample│          │
│  │   Database   │    │  Recognition │    │  Extraction   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                             │                   │                   │
│                             ▼                   ▼                   │
│                      ┌──────────────────────────────┐              │
│                      │      CGNR Repair Loop        │              │
│                      │  (Counterexample-Guided      │              │
│                      │   Neural Repair)             │              │
│                      └──────────────────────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

| Phase | Component | Description |
|-------|-----------|-------------|
| **Phase 1** | LLVM Extraction | Extract functions from LLVM backends (RISCV, ARM, AArch64, X86) |
| **Phase 2.1** | Semantic Analyzer | Pattern recognition, CFG construction, symbolic execution |
| **Phase 2.2** | SMT Integration | Z3-based verification, Property DSL, counterexample extraction |
| **Phase 2.3** | Neural Repair | Training data generation, model fine-tuning interface |
| **Phase 2.4** | CGNR Pipeline | Counterexample-guided repair loop, end-to-end integration |

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- LLVM 18+ (for full functionality)
- Z3 Solver (optional, for SMT verification)

### Method 1: Docker (Full Environment)

```bash
# Build unified image with all dependencies
docker build -f Dockerfile.unified -t vega-verified .

# Verify installation
docker run --rm vega-verified vega-verify status
```

### Method 2: Local Installation

```bash
# Clone repository
git clone https://github.com/Zachary-Lee-Jaeho/gensparktest.git
cd gensparktest/webapp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install package
pip install -e .

# Install optional dependencies
pip install z3-solver  # SMT verification
pip install torch transformers  # Neural components

# Verify installation
vega-verify status
```

### Method 3: Development Installation

```bash
pip install -e ".[dev]"
pip install -e ".[neural]"
```

---

## 🖥️ CLI Usage

### Available Commands

```bash
# Show all commands
vega-verify --help

# System status
vega-verify status

# Extract functions from LLVM
vega-verify extract --llvm-source /path/to/llvm --backend riscv

# Verify a function
vega-verify verify --code function.cpp --spec spec.json

# Repair a buggy function
vega-verify repair --code buggy.cpp --spec spec.json --save-repaired

# Run experiments
vega-verify experiment --all
vega-verify experiment --experiment verification --backend riscv

# Generate reports
vega-verify report --format markdown
vega-verify report --format latex --template paper
```

### Examples

```bash
# Quick verification test
vega-verify experiment --experiment verification --sample-size 10

# Full RISCV backend evaluation
vega-verify experiment --experiment verification --backend riscv --sample-size 500

# VEGA vs VEGA-Verified comparison
vega-verify experiment --experiment comparison --sample-size 100

# Ablation study
vega-verify experiment --experiment ablation
```

---

## 📊 Paper Reproduction

### Quick Reproduction

```bash
# Using the reproduction script
./scripts/reproduce_experiments.sh --all

# Or with Docker
docker run -it --rm -v $(pwd)/results:/app/results vega-verified \
    ./scripts/reproduce_experiments.sh --all
```

### Step-by-Step Reproduction

```bash
# 1. Run verification experiments
vega-verify experiment --experiment verification --backend all --sample-size 500

# 2. Run repair experiments  
vega-verify experiment --experiment repair --sample-size 100

# 3. Run comparison (VEGA vs VEGA-Verified)
vega-verify experiment --experiment comparison

# 4. Run ablation study
vega-verify experiment --experiment ablation

# 5. Generate paper tables/figures
vega-verify report --format latex --template paper
```

### Expected Results

| Experiment | Metric | Expected Value |
|------------|--------|----------------|
| Verification | Accuracy | 75-85% |
| Repair | Success Rate | 60-75% |
| Comparison | Improvement over VEGA | +10-15pp |
| Ablation | SMT contribution | +15-20pp |

> **Note**: Results may vary due to mock components. See [Limitations](#-important-implementation-status--limitations) section.

### Reproduction Outputs

```
results/
├── experiments_YYYYMMDD_HHMMSS.json  # Raw results
├── report_paper.md                    # Markdown report
├── report_paper.tex                   # LaTeX tables
└── reproduction_YYYYMMDD_HHMMSS.log  # Execution log
```

---

## 📁 Project Structure

```
webapp/
├── Dockerfile.unified          # All-in-one Docker image
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── scripts/
│   └── reproduce_experiments.sh  # Paper reproduction script
├── src/
│   ├── cli.py                  # CLI entry point (vega-verify)
│   ├── main.py                 # Legacy entry point
│   ├── verification/
│   │   ├── semantic_analyzer.py  # Phase 2.1: Pattern recognition
│   │   ├── ir_to_smt.py          # Phase 2.2: IR → Z3 translation
│   │   ├── switch_verifier.py    # Switch statement verification
│   │   └── verifier.py           # Main verifier interface
│   ├── repair/
│   │   ├── training_data.py      # Phase 2.3: Training data generation
│   │   ├── model_finetuning.py   # Phase 2.3: Model fine-tuning
│   │   ├── neural_model.py       # Neural repair model
│   │   ├── cgnr.py               # CGNR algorithm
│   │   └── switch_repair.py      # Switch-specific repair
│   ├── integration/
│   │   ├── cgnr_pipeline.py      # Phase 2.4: End-to-end pipeline
│   │   └── vega_adapter.py       # VEGA model interface
│   ├── specification/
│   │   └── spec_language.py      # Formal specification DSL
│   ├── llvm_extraction/
│   │   └── ...                   # LLVM source extraction
│   └── utils/
│       └── ...                   # Utilities
├── tests/
│   ├── test_phase1_infrastructure.py
│   ├── test_phase2_complete.py
│   └── ...
├── data/
│   ├── llvm_functions_multi.json   # Extracted functions
│   ├── llvm_ground_truth.json      # Ground truth database
│   └── llvm_riscv_ast.json         # RISCV AST data
└── docker/
    ├── Dockerfile.llvm             # LLVM build environment
    └── tools/
        └── ast_extractor.cpp       # Clang LibTooling extractor
```

---

## 🧪 Development

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test files
python -m pytest tests/test_phase2_complete.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### Current Test Status

```
Tests: 123 passing
├── Phase 1 Infrastructure: 76 tests ✅
├── Phase 2 Complete: 47 tests ✅
└── Total: 123 tests ✅
```

---

## 📈 Data Statistics

### Extracted LLVM Functions

| Backend | Functions | Switch Statements |
|---------|-----------|-------------------|
| RISCV | 480 | 63 |
| ARM | 498 | 57 |
| AArch64 | 645 | 49 |
| X86 | 947 | 162 |
| **Total** | **2,570** | **331** |

---

## 🔗 References

1. Zhong et al., "VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model", CGO 2025
2. [LLVM Documentation](https://llvm.org/docs/)
3. [Z3 Solver Guide](https://microsoft.github.io/z3guide/)
4. Guo et al., "UniXcoder: Unified Cross-Modal Pre-training for Code Representation", ACL 2022

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- VEGA authors for the original neural compiler backend generation approach
- LLVM community for the compiler infrastructure
- Z3 team for the SMT solver

---

## 📧 Contact

For questions about this implementation, please open an issue on GitHub.
