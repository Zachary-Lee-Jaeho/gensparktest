"""
Clang AST Parser for VEGA-Verified.

Provides accurate C++ parsing using libclang bindings.
This replaces regex-based parsing with proper AST analysis.

Features:
1. Function extraction with full signature
2. Switch statement parsing
3. Control flow analysis
4. Type information extraction
5. Call graph construction
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for libclang availability
try:
    from clang.cindex import (
        Index, CursorKind, TypeKind, TranslationUnit,
        Cursor, Type, SourceLocation, SourceRange,
        Config
    )
    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False
    logger.warning("libclang not available. Install with: pip install libclang")


class NodeType(Enum):
    """AST node types."""
    FUNCTION = "function"
    SWITCH = "switch"
    CASE = "case"
    IF = "if"
    FOR = "for"
    WHILE = "while"
    RETURN = "return"
    CALL = "call"
    VARIABLE = "variable"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    MEMBER_ACCESS = "member_access"
    ARRAY_SUBSCRIPT = "array_subscript"
    UNKNOWN = "unknown"


@dataclass
class ASTNode:
    """Represents a node in the AST."""
    node_type: NodeType
    name: str = ""
    line: int = 0
    column: int = 0
    children: List['ASTNode'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type.value,
            "name": self.name,
            "line": self.line,
            "column": self.column,
            "children": [c.to_dict() for c in self.children],
            "attributes": self.attributes,
            "source_text": self.source_text[:200] if self.source_text else "",
        }


@dataclass
class FunctionInfo:
    """Information about a parsed function."""
    name: str
    return_type: str
    parameters: List[Tuple[str, str]]  # [(name, type), ...]
    body: str
    line_start: int
    line_end: int
    is_method: bool = False
    class_name: str = ""
    qualifiers: List[str] = field(default_factory=list)
    local_variables: List[Tuple[str, str]] = field(default_factory=list)
    called_functions: List[str] = field(default_factory=list)
    ast: Optional[ASTNode] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "return_type": self.return_type,
            "parameters": [{"name": n, "type": t} for n, t in self.parameters],
            "line_start": self.line_start,
            "line_end": self.line_end,
            "is_method": self.is_method,
            "class_name": self.class_name,
            "qualifiers": self.qualifiers,
            "local_variables": [{"name": n, "type": t} for n, t in self.local_variables],
            "called_functions": self.called_functions,
        }


@dataclass
class SwitchStatement:
    """Parsed switch statement."""
    condition_var: str
    cases: List[Dict[str, Any]]  # [{"value": ..., "statements": [...], "has_break": bool}, ...]
    has_default: bool
    default_statements: List[str]
    line_start: int
    line_end: int
    source_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_var": self.condition_var,
            "cases": self.cases,
            "has_default": self.has_default,
            "default_statements": self.default_statements,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


class ClangASTParser:
    """
    Parser using Clang AST for accurate C++ analysis.
    """
    
    def __init__(self, clang_path: Optional[str] = None):
        """
        Initialize the parser.
        
        Args:
            clang_path: Optional path to libclang.so
        """
        if not CLANG_AVAILABLE:
            raise ImportError(
                "libclang not available. Install with: pip install libclang"
            )
        
        # Try to configure libclang path
        if clang_path:
            Config.set_library_file(clang_path)
        
        self.index = Index.create()
        self.current_file = None
        self.functions: Dict[str, FunctionInfo] = {}
        self.switches: List[SwitchStatement] = []
    
    def parse_file(self, file_path: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Parse a C++ file.
        
        Args:
            file_path: Path to the C++ file
            args: Optional compiler arguments
            
        Returns:
            Dictionary with parsed information
        """
        self.current_file = Path(file_path)
        self.functions.clear()
        self.switches.clear()
        
        # Default arguments for LLVM-style code
        default_args = [
            '-x', 'c++',
            '-std=c++17',
            '-I/usr/include',
            '-I/usr/local/include',
        ]
        
        compile_args = (args or []) + default_args
        
        try:
            tu = self.index.parse(
                file_path,
                args=compile_args,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            
            # Check for errors
            diagnostics = list(tu.diagnostics)
            errors = [d for d in diagnostics if d.severity >= 3]
            
            if errors:
                logger.warning(f"Parse errors in {file_path}: {len(errors)} errors")
            
            # Walk the AST
            self._walk_ast(tu.cursor)
            
            return {
                "file": str(file_path),
                "functions": {name: f.to_dict() for name, f in self.functions.items()},
                "switches": [s.to_dict() for s in self.switches],
                "errors": [str(d) for d in errors],
            }
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return {"error": str(e)}
    
    def parse_code(self, code: str, filename: str = "input.cpp") -> Dict[str, Any]:
        """
        Parse C++ code from a string.
        
        Args:
            code: C++ source code
            filename: Virtual filename for the code
            
        Returns:
            Dictionary with parsed information
        """
        self.functions.clear()
        self.switches.clear()
        
        try:
            tu = self.index.parse(
                filename,
                args=['-x', 'c++', '-std=c++17'],
                unsaved_files=[(filename, code)],
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            
            self._walk_ast(tu.cursor)
            
            return {
                "functions": {name: f.to_dict() for name, f in self.functions.items()},
                "switches": [s.to_dict() for s in self.switches],
            }
            
        except Exception as e:
            logger.error(f"Failed to parse code: {e}")
            return {"error": str(e)}
    
    def _walk_ast(self, cursor: 'Cursor', depth: int = 0) -> None:
        """Walk the AST recursively."""
        kind = cursor.kind
        
        # Function definitions
        if kind == CursorKind.FUNCTION_DECL and cursor.is_definition():
            func_info = self._parse_function(cursor)
            if func_info:
                self.functions[func_info.name] = func_info
        
        # Method definitions
        elif kind == CursorKind.CXX_METHOD and cursor.is_definition():
            func_info = self._parse_function(cursor, is_method=True)
            if func_info:
                full_name = f"{func_info.class_name}::{func_info.name}"
                self.functions[full_name] = func_info
        
        # Switch statements (captured during function parsing)
        elif kind == CursorKind.SWITCH_STMT:
            switch_info = self._parse_switch(cursor)
            if switch_info:
                self.switches.append(switch_info)
        
        # Recurse into children
        for child in cursor.get_children():
            self._walk_ast(child, depth + 1)
    
    def _parse_function(self, cursor: 'Cursor', is_method: bool = False) -> Optional[FunctionInfo]:
        """Parse a function definition."""
        try:
            name = cursor.spelling
            return_type = cursor.result_type.spelling
            
            # Get parameters
            parameters = []
            for child in cursor.get_children():
                if child.kind == CursorKind.PARM_DECL:
                    param_name = child.spelling
                    param_type = child.type.spelling
                    parameters.append((param_name, param_type))
            
            # Get source location
            location = cursor.location
            extent = cursor.extent
            line_start = extent.start.line
            line_end = extent.end.line
            
            # Get class name for methods
            class_name = ""
            if is_method:
                parent = cursor.semantic_parent
                if parent and parent.kind == CursorKind.CLASS_DECL:
                    class_name = parent.spelling
            
            # Get qualifiers
            qualifiers = []
            if cursor.is_const_method():
                qualifiers.append("const")
            if cursor.is_static_method():
                qualifiers.append("static")
            if cursor.is_virtual_method():
                qualifiers.append("virtual")
            
            # Get function body and analyze it
            body = self._get_source_text(cursor)
            local_vars = []
            called_funcs = []
            
            for child in cursor.walk_preorder():
                # Local variables
                if child.kind == CursorKind.VAR_DECL:
                    var_name = child.spelling
                    var_type = child.type.spelling
                    local_vars.append((var_name, var_type))
                
                # Function calls
                elif child.kind == CursorKind.CALL_EXPR:
                    call_name = child.spelling
                    if call_name:
                        called_funcs.append(call_name)
            
            # Build AST
            ast = self._build_ast_node(cursor)
            
            return FunctionInfo(
                name=name,
                return_type=return_type,
                parameters=parameters,
                body=body,
                line_start=line_start,
                line_end=line_end,
                is_method=is_method,
                class_name=class_name,
                qualifiers=qualifiers,
                local_variables=local_vars,
                called_functions=list(set(called_funcs)),
                ast=ast,
            )
            
        except Exception as e:
            logger.error(f"Failed to parse function: {e}")
            return None
    
    def _parse_switch(self, cursor: 'Cursor') -> Optional[SwitchStatement]:
        """Parse a switch statement."""
        try:
            # Get condition variable
            condition_var = ""
            cases = []
            has_default = False
            default_statements = []
            
            children = list(cursor.get_children())
            
            # First child is typically the condition
            if children:
                condition = children[0]
                condition_var = self._get_source_text(condition)
            
            # Parse cases
            current_case = None
            current_statements = []
            has_break = False
            
            for child in cursor.walk_preorder():
                if child.kind == CursorKind.CASE_STMT:
                    # Save previous case
                    if current_case is not None:
                        cases.append({
                            "value": current_case,
                            "statements": current_statements,
                            "has_break": has_break,
                        })
                    
                    # Start new case
                    case_children = list(child.get_children())
                    if case_children:
                        current_case = self._get_source_text(case_children[0])
                    current_statements = []
                    has_break = False
                
                elif child.kind == CursorKind.DEFAULT_STMT:
                    # Save previous case
                    if current_case is not None:
                        cases.append({
                            "value": current_case,
                            "statements": current_statements,
                            "has_break": has_break,
                        })
                    
                    has_default = True
                    current_case = None
                    current_statements = []
                    has_break = False
                
                elif child.kind == CursorKind.BREAK_STMT:
                    has_break = True
                
                elif child.kind == CursorKind.RETURN_STMT:
                    stmt_text = self._get_source_text(child)
                    if current_case is not None:
                        current_statements.append(stmt_text)
                    elif has_default:
                        default_statements.append(stmt_text)
            
            # Save last case
            if current_case is not None:
                cases.append({
                    "value": current_case,
                    "statements": current_statements,
                    "has_break": has_break,
                })
            elif has_default and current_statements:
                default_statements = current_statements
            
            # Get source location
            extent = cursor.extent
            
            return SwitchStatement(
                condition_var=condition_var,
                cases=cases,
                has_default=has_default,
                default_statements=default_statements,
                line_start=extent.start.line,
                line_end=extent.end.line,
                source_text=self._get_source_text(cursor),
            )
            
        except Exception as e:
            logger.error(f"Failed to parse switch: {e}")
            return None
    
    def _build_ast_node(self, cursor: 'Cursor') -> ASTNode:
        """Build an ASTNode from a Clang cursor."""
        # Map Clang kinds to our types
        kind_map = {
            CursorKind.FUNCTION_DECL: NodeType.FUNCTION,
            CursorKind.CXX_METHOD: NodeType.FUNCTION,
            CursorKind.SWITCH_STMT: NodeType.SWITCH,
            CursorKind.CASE_STMT: NodeType.CASE,
            CursorKind.IF_STMT: NodeType.IF,
            CursorKind.FOR_STMT: NodeType.FOR,
            CursorKind.WHILE_STMT: NodeType.WHILE,
            CursorKind.RETURN_STMT: NodeType.RETURN,
            CursorKind.CALL_EXPR: NodeType.CALL,
            CursorKind.VAR_DECL: NodeType.VARIABLE,
            CursorKind.BINARY_OPERATOR: NodeType.BINARY_OP,
            CursorKind.UNARY_OPERATOR: NodeType.UNARY_OP,
            CursorKind.MEMBER_REF_EXPR: NodeType.MEMBER_ACCESS,
            CursorKind.ARRAY_SUBSCRIPT_EXPR: NodeType.ARRAY_SUBSCRIPT,
        }
        
        node_type = kind_map.get(cursor.kind, NodeType.UNKNOWN)
        
        # Get location
        location = cursor.location
        line = location.line if location.file else 0
        column = location.column if location.file else 0
        
        # Get attributes based on type
        attributes = {}
        
        if cursor.kind in (CursorKind.VAR_DECL, CursorKind.PARM_DECL):
            attributes["type"] = cursor.type.spelling
        
        if cursor.kind == CursorKind.CALL_EXPR:
            attributes["callee"] = cursor.spelling
        
        if cursor.kind == CursorKind.BINARY_OPERATOR:
            # Get operator from source
            src = self._get_source_text(cursor)
            for op in ['==', '!=', '<=', '>=', '&&', '||', '<<', '>>', '+=', '-=', '<', '>', '+', '-', '*', '/', '%', '&', '|', '^', '=']:
                if op in src:
                    attributes["operator"] = op
                    break
        
        # Build children
        children = []
        for child in cursor.get_children():
            child_node = self._build_ast_node(child)
            children.append(child_node)
        
        return ASTNode(
            node_type=node_type,
            name=cursor.spelling,
            line=line,
            column=column,
            children=children,
            attributes=attributes,
            source_text=self._get_source_text(cursor),
        )
    
    def _get_source_text(self, cursor: 'Cursor') -> str:
        """Get the source text for a cursor."""
        try:
            extent = cursor.extent
            start = extent.start
            end = extent.end
            
            if not start.file or not end.file:
                return ""
            
            # Read from file
            file_path = start.file.name
            
            if file_path and Path(file_path).exists():
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                if start.line == end.line:
                    return lines[start.line - 1][start.column - 1:end.column - 1]
                else:
                    result = lines[start.line - 1][start.column - 1:]
                    for i in range(start.line, end.line - 1):
                        result += lines[i]
                    result += lines[end.line - 1][:end.column - 1]
                    return result
            
            return ""
            
        except Exception:
            return ""
    
    def get_function(self, name: str) -> Optional[FunctionInfo]:
        """Get a parsed function by name."""
        return self.functions.get(name)
    
    def get_all_switches(self) -> List[SwitchStatement]:
        """Get all parsed switch statements."""
        return self.switches


class ClangSymbolicExecutor:
    """
    Symbolic executor using Clang AST for accurate path analysis.
    """
    
    def __init__(self):
        if not CLANG_AVAILABLE:
            raise ImportError("libclang required for ClangSymbolicExecutor")
        
        self.parser = ClangASTParser()
    
    def execute(
        self,
        code: str,
        function_name: str
    ) -> Dict[str, Any]:
        """
        Execute symbolic analysis on a function.
        
        Args:
            code: C++ source code
            function_name: Name of function to analyze
            
        Returns:
            Analysis results with paths and constraints
        """
        # Parse the code
        result = self.parser.parse_code(code)
        
        if "error" in result:
            return {"error": result["error"]}
        
        # Find the function
        func_info = None
        for name, info in self.parser.functions.items():
            if name == function_name or name.endswith(f"::{function_name}"):
                func_info = info
                break
        
        if not func_info or not func_info.ast:
            return {"error": f"Function {function_name} not found"}
        
        # Analyze paths through the AST
        paths = self._analyze_paths(func_info.ast)
        
        return {
            "function": function_name,
            "parameters": func_info.parameters,
            "paths": paths,
            "local_variables": func_info.local_variables,
            "called_functions": func_info.called_functions,
        }
    
    def _analyze_paths(self, ast: ASTNode) -> List[Dict[str, Any]]:
        """Analyze all paths through the AST."""
        paths = []
        current_path = {
            "constraints": [],
            "statements": [],
            "return_value": None,
        }
        
        self._walk_for_paths(ast, current_path, paths)
        
        return paths
    
    def _walk_for_paths(
        self,
        node: ASTNode,
        current_path: Dict[str, Any],
        all_paths: List[Dict[str, Any]]
    ) -> None:
        """Walk AST and collect paths."""
        if node.node_type == NodeType.IF:
            # Fork for both branches
            condition = node.source_text.split('{')[0] if '{' in node.source_text else ""
            
            # Then branch
            then_path = {
                "constraints": current_path["constraints"] + [condition],
                "statements": list(current_path["statements"]),
                "return_value": None,
            }
            
            # Else branch
            else_path = {
                "constraints": current_path["constraints"] + [f"!({condition})"],
                "statements": list(current_path["statements"]),
                "return_value": None,
            }
            
            # Process children
            for child in node.children:
                self._walk_for_paths(child, then_path, all_paths)
            
            all_paths.append(else_path)
        
        elif node.node_type == NodeType.SWITCH:
            # Fork for each case
            condition = node.attributes.get("condition", node.name)
            
            for child in node.children:
                if child.node_type == NodeType.CASE:
                    case_value = child.name
                    case_path = {
                        "constraints": current_path["constraints"] + [f"{condition} == {case_value}"],
                        "statements": list(current_path["statements"]),
                        "return_value": None,
                    }
                    self._walk_for_paths(child, case_path, all_paths)
        
        elif node.node_type == NodeType.RETURN:
            current_path["return_value"] = node.source_text
            current_path["statements"].append(node.source_text)
            all_paths.append(dict(current_path))
        
        else:
            # Regular statement
            if node.source_text:
                current_path["statements"].append(node.source_text)
            
            for child in node.children:
                self._walk_for_paths(child, current_path, all_paths)


def create_parser(clang_path: Optional[str] = None) -> Optional[ClangASTParser]:
    """
    Factory function to create a Clang parser.
    
    Returns None if libclang is not available.
    """
    if not CLANG_AVAILABLE:
        logger.warning("libclang not available")
        return None
    
    try:
        return ClangASTParser(clang_path)
    except Exception as e:
        logger.error(f"Failed to create parser: {e}")
        return None
