"""
Specification inference module for VEGA-Verified.
Automatically extracts formal specifications from reference implementations.

Phase 2 components:
- Symbolic execution for path condition extraction
- AST alignment for multi-reference comparison
- Enhanced specification inference
"""

from .spec_language import (
    Condition,
    ConditionType,
    Expression,
    Variable,
    Constant,
    BinaryOp,
    FunctionCall,
    Specification,
)
from .inferrer import SpecificationInferrer
from .condition_extract import ConditionExtractor
from .pattern_abstract import PatternAbstractor
from .symbolic_exec import (
    SymbolicExecutor,
    SymbolicState,
    SymbolicValue,
    ExecutionPath,
    extract_path_conditions,
)
from .alignment import (
    ASTAligner,
    ASTNode,
    AlignmentMapping,
    MultiAlignmentResult,
    AlignmentType,
    align_references,
)

__all__ = [
    # Core specification language
    "Condition",
    "ConditionType",
    "Expression",
    "Variable",
    "Constant",
    "BinaryOp",
    "FunctionCall",
    "Specification",
    # Inference
    "SpecificationInferrer",
    "ConditionExtractor",
    "PatternAbstractor",
    # Symbolic execution
    "SymbolicExecutor",
    "SymbolicState",
    "SymbolicValue",
    "ExecutionPath",
    "extract_path_conditions",
    # Alignment
    "ASTAligner",
    "ASTNode",
    "AlignmentMapping",
    "MultiAlignmentResult",
    "AlignmentType",
    "align_references",
]
