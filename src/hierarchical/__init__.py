"""
Hierarchical Modular Verification for VEGA-Verified.

This module implements three-level verification:
- Level 1: Function Verification
- Level 2: Module Verification  
- Level 3: Backend Integration Verification

Uses assume-guarantee reasoning and interface contracts for compositional soundness.
"""

from .interface_contract import InterfaceContract, Assumption, Guarantee
from .function_verify import FunctionVerifier, FunctionVerificationResult
from .module_verify import ModuleVerifier, ModuleVerificationResult
from .backend_verify import BackendVerifier, BackendVerificationResult
from .hierarchical_verifier import HierarchicalVerifier, HierarchicalResult, VerificationLevel

__all__ = [
    'InterfaceContract',
    'Assumption',
    'Guarantee',
    'FunctionVerifier',
    'FunctionVerificationResult',
    'ModuleVerifier',
    'ModuleVerificationResult',
    'BackendVerifier',
    'BackendVerificationResult',
    'HierarchicalVerifier',
    'HierarchicalResult',
    'VerificationLevel',
]
