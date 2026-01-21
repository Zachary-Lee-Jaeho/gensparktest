"""
Integration module for VEGA-Verified.

Provides adapters for integrating with VEGA and LLVM infrastructure,
and the end-to-end verification pipeline.
"""

from .vega_adapter import VEGAAdapter, VEGAGenerationResult
from .llvm_adapter import LLVMAdapter, BackendInfo
from .experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentResult
from .pipeline import (
    VEGAVerifiedPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    StageResult,
    create_pipeline,
)

__all__ = [
    # VEGA Adapter
    'VEGAAdapter',
    'VEGAGenerationResult',
    # LLVM Adapter
    'LLVMAdapter',
    'BackendInfo',
    # Experiment Runner
    'ExperimentRunner',
    'ExperimentConfig',
    'ExperimentResult',
    # End-to-End Pipeline
    'VEGAVerifiedPipeline',
    'PipelineConfig',
    'PipelineResult',
    'PipelineStage',
    'StageResult',
    'create_pipeline',
]
