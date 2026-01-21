"""
Parsing module for VEGA-Verified.

Provides C++ code parsing and analysis using tree-sitter.
"""

from .cpp_parser import CppParser, ParsedFunction, CFGNode, ControlFlowGraph
from .ast_utils import ASTNode, ASTVisitor, find_functions, extract_function_body

__all__ = [
    'CppParser',
    'ParsedFunction',
    'CFGNode',
    'ControlFlowGraph',
    'ASTNode',
    'ASTVisitor',
    'find_functions',
    'extract_function_body',
]
