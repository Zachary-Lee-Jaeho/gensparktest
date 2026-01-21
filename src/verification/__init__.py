"""
Verification module for VEGA-Verified.
Provides formal verification using SMT solving.
"""

from .verifier import (
    Verifier,
    VerificationResult,
    VerificationStatus,
    Counterexample,
)
from .vcgen import VCGenerator
from .smt_solver import SMTSolver, SMTResult

__all__ = [
    "Verifier",
    "VerificationResult",
    "VerificationStatus",
    "Counterexample",
    "VCGenerator",
    "SMTSolver",
    "SMTResult",
]
