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
from .z3_backend import (
    Z3Verifier,
    Z3VerificationResult,
    CodeParser,
    CodeModel,
    SwitchCase,
    create_z3_verifier,
)
from .z3_semantic_analyzer import (
    Z3SemanticAnalyzer,
    SemanticVerificationResult,
    SemanticModel,
    ArchitectureType,
    create_semantic_analyzer,
)
from .integrated_verifier import (
    IntegratedVerifier,
    create_integrated_verifier,
)
from .switch_verifier import (
    SwitchVerifier,
    SwitchStatement,
    CaseMapping,
    SwitchStatementParser,
    VerificationResult as SwitchVerificationResult,
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
    # Z3 Backend
    "Z3Verifier",
    "Z3VerificationResult",
    "CodeParser",
    "CodeModel",
    "SwitchCase",
    "create_z3_verifier",
    # Semantic Analyzer
    "Z3SemanticAnalyzer",
    "SemanticVerificationResult",
    "SemanticModel",
    "ArchitectureType",
    "create_semantic_analyzer",
    # Integrated Verifier
    "IntegratedVerifier",
    "create_integrated_verifier",
    # Switch Verifier
    "SwitchVerifier",
    "SwitchStatement",
    "CaseMapping",
    "SwitchStatementParser",
    "SwitchVerificationResult",
]
