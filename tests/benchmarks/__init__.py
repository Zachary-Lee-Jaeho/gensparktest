"""
VEGA-Verified Benchmark Suite.

This package provides comprehensive benchmarks for evaluating VEGA-Verified across
multiple processor architectures, including those from the original VEGA paper.

Benchmark Categories:
1. VEGA Paper Benchmarks: RISC-V, RI5CY, xCORE
2. Extended Benchmarks: ARM, AArch64, MIPS, x86-64, PowerPC

Key Features:
- Realistic compiler backend functions
- Formal specifications for each function
- Expected metrics for comparison
- Support for hierarchical verification testing
- Comparison framework for VEGA vs VEGA-Verified analysis
"""

from .processor_backends import (
    ProcessorFamily,
    ProcessorBackendBenchmark,
    get_all_benchmarks,
    get_vega_benchmarks,
    create_riscv_benchmark,
    create_arm_benchmark,
    create_aarch64_benchmark,
    create_mips_benchmark,
    create_x86_64_benchmark,
    create_powerpc_benchmark,
)

from .vega_paper_targets import (
    create_ri5cy_benchmark,
    create_xcore_benchmark,
    get_vega_paper_targets,
    get_vega_paper_metrics,
)

from .comparison_framework import (
    EvaluationMode,
    FunctionMetrics,
    ModuleMetrics,
    BackendMetrics,
    ComparisonResult,
    ComparisonFramework,
    run_vega_paper_comparison,
    run_extended_comparison,
)

__all__ = [
    # Enums and Classes
    'ProcessorFamily',
    'ProcessorBackendBenchmark',
    'EvaluationMode',
    'FunctionMetrics',
    'ModuleMetrics',
    'BackendMetrics',
    'ComparisonResult',
    'ComparisonFramework',
    
    # Registry Functions
    'get_all_benchmarks',
    'get_vega_benchmarks',
    'get_vega_paper_targets',
    'get_vega_paper_metrics',
    
    # Individual Benchmark Creators
    'create_riscv_benchmark',
    'create_arm_benchmark',
    'create_aarch64_benchmark',
    'create_mips_benchmark',
    'create_x86_64_benchmark',
    'create_powerpc_benchmark',
    'create_ri5cy_benchmark',
    'create_xcore_benchmark',
    
    # Comparison Functions
    'run_vega_paper_comparison',
    'run_extended_comparison',
]
