"""
Integration module for VEGA-Verified.

Provides adapters for integrating with VEGA and LLVM infrastructure.
"""

from .vega_adapter import VEGAAdapter, VEGAGenerationResult
from .llvm_adapter import LLVMAdapter, BackendInfo
from .experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentResult

__all__ = [
    'VEGAAdapter',
    'VEGAGenerationResult',
    'LLVMAdapter',
    'BackendInfo',
    'ExperimentRunner',
    'ExperimentConfig',
    'ExperimentResult',
]
