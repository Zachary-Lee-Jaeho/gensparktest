"""
Specification inference module for VEGA-Verified.
Automatically extracts formal specifications from reference implementations.
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

__all__ = [
    "Condition",
    "ConditionType",
    "Expression",
    "Variable",
    "Constant",
    "BinaryOp",
    "FunctionCall",
    "Specification",
    "SpecificationInferrer",
    "ConditionExtractor",
    "PatternAbstractor",
]
