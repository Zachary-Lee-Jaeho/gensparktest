"""
LLVM Source Code Extraction Module for VEGA-Verified.

This module provides tools to extract and analyze LLVM backend source code
without requiring a full LLVM build. It uses Python-based parsing (regex + 
tree-sitter when available) to extract function definitions, metadata, and
structure from LLVM source files.

Key Components:
- LLVMSourceFetcher: Downloads LLVM source from GitHub
- CppFunctionExtractor: Extracts C++ function definitions
- BackendAnalyzer: Analyzes backend module structure
- FunctionDatabase: Stores extracted functions with metadata

Usage:
    from src.llvm_extraction import LLVMExtractor
    
    extractor = LLVMExtractor()
    extractor.fetch_llvm_source(version="18.1.0")
    functions = extractor.extract_backend("RISCV")
"""

from .extractor import (
    LLVMSourceFetcher,
    CppFunctionExtractor,
    BackendAnalyzer,
    LLVMExtractor,
)

from .database import (
    FunctionRecord,
    ModuleRecord,
    BackendRecord,
    FunctionDatabase,
)

from .parser import (
    CppParser,
    FunctionSignature,
    SwitchCasePattern,
)

__all__ = [
    # Main extractor
    "LLVMExtractor",
    "LLVMSourceFetcher", 
    "CppFunctionExtractor",
    "BackendAnalyzer",
    # Database
    "FunctionRecord",
    "ModuleRecord", 
    "BackendRecord",
    "FunctionDatabase",
    # Parser
    "CppParser",
    "FunctionSignature",
    "SwitchCasePattern",
]
