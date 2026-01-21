"""
Verification module for VEGA-Verified.
Provides formal verification using SMT solving and bounded model checking.
"""

from .verifier import (
    Verifier,
    VerificationResult,
    VerificationStatus,
    Counterexample,
    BatchVerifier,
)
from .vcgen import VCGenerator, CFG, CFGNode, VerificationCondition
from .smt_solver import SMTSolver, SMTResult, SMTModel, SMTBuilder
from .bmc import (
    BoundedModelChecker,
    BMCResult,
    BMCCheckResult,
    BMCTrace,
    LoopAnalyzer,
    LoopInfo,
)

__all__ = [
    # Verifier
    "Verifier",
    "VerificationResult",
    "VerificationStatus",
    "Counterexample",
    "BatchVerifier",
    # VC Generation
    "VCGenerator",
    "CFG",
    "CFGNode",
    "VerificationCondition",
    # SMT Solving
    "SMTSolver",
    "SMTResult",
    "SMTModel",
    "SMTBuilder",
    # Bounded Model Checking
    "BoundedModelChecker",
    "BMCResult",
    "BMCCheckResult",
    "BMCTrace",
    "LoopAnalyzer",
    "LoopInfo",
]
