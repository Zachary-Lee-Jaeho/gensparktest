"""
AST Utilities for VEGA-Verified.

Provides helper functions for AST manipulation and traversal.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Generator
from enum import Enum
import re


class ASTNodeType(Enum):
    """AST node types."""
    ROOT = "root"
    FUNCTION = "function"
    CLASS = "class"
    STRUCT = "struct"
    NAMESPACE = "namespace"
    STATEMENT = "statement"
    EXPRESSION = "expression"
    DECLARATION = "declaration"
    IF = "if"
    ELSE = "else"
    FOR = "for"
    WHILE = "while"
    DO = "do"
    SWITCH = "switch"
    CASE = "case"
    RETURN = "return"
    BREAK = "break"
    CONTINUE = "continue"
    CALL = "call"
    ASSIGNMENT = "assignment"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    IDENTIFIER = "identifier"
    LITERAL = "literal"
    COMMENT = "comment"
    UNKNOWN = "unknown"


@dataclass
class ASTNode:
    """Generic AST node representation."""
    type: ASTNodeType
    text: str
    children: List['ASTNode'] = field(default_factory=list)
    parent: Optional['ASTNode'] = None
    start_line: int = 0
    end_line: int = 0
    start_col: int = 0
    end_col: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'ASTNode') -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)
    
    def find_children(self, node_type: ASTNodeType) -> List['ASTNode']:
        """Find all children of a given type."""
        return [c for c in self.children if c.type == node_type]
    
    def find_descendants(self, node_type: ASTNodeType) -> List['ASTNode']:
        """Find all descendants of a given type."""
        results = []
        
        def search(node: ASTNode):
            if node.type == node_type:
                results.append(node)
            for child in node.children:
                search(child)
        
        for child in self.children:
            search(child)
        
        return results
    
    def get_path_to_root(self) -> List['ASTNode']:
        """Get path from this node to root."""
        path = [self]
        node = self.parent
        while node:
            path.append(node)
            node = node.parent
        return path
    
    def depth(self) -> int:
        """Get depth of this node in the tree."""
        return len(self.get_path_to_root()) - 1
    
    def subtree_size(self) -> int:
        """Get size of subtree rooted at this node."""
        size = 1
        for child in self.children:
            size += child.subtree_size()
        return size
    
    def clone(self, deep: bool = True) -> 'ASTNode':
        """Clone this node."""
        new_node = ASTNode(
            type=self.type,
            text=self.text,
            start_line=self.start_line,
            end_line=self.end_line,
            start_col=self.start_col,
            end_col=self.end_col,
            metadata=dict(self.metadata)
        )
        
        if deep:
            for child in self.children:
                new_child = child.clone(deep=True)
                new_node.add_child(new_child)
        
        return new_node
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "text": self.text[:100] if len(self.text) > 100 else self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        text_preview = self.text[:50] if len(self.text) > 50 else self.text
        text_preview = text_preview.replace('\n', '\\n')
        return f"ASTNode({self.type.value}: {text_preview})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ASTVisitor:
    """
    AST visitor pattern implementation.
    
    Subclass and override visit_* methods to implement custom traversal.
    """
    
    def visit(self, node: ASTNode) -> Any:
        """Visit a node."""
        method_name = f"visit_{node.type.value}"
        method = getattr(self, method_name, self.generic_visit)
        return method(node)
    
    def generic_visit(self, node: ASTNode) -> None:
        """Default visitor that visits all children."""
        for child in node.children:
            self.visit(child)
    
    def visit_root(self, node: ASTNode) -> None:
        self.generic_visit(node)
    
    def visit_function(self, node: ASTNode) -> None:
        self.generic_visit(node)
    
    def visit_class(self, node: ASTNode) -> None:
        self.generic_visit(node)
    
    def visit_if(self, node: ASTNode) -> None:
        self.generic_visit(node)
    
    def visit_for(self, node: ASTNode) -> None:
        self.generic_visit(node)
    
    def visit_while(self, node: ASTNode) -> None:
        self.generic_visit(node)
    
    def visit_switch(self, node: ASTNode) -> None:
        self.generic_visit(node)
    
    def visit_return(self, node: ASTNode) -> None:
        self.generic_visit(node)
    
    def visit_call(self, node: ASTNode) -> None:
        self.generic_visit(node)
    
    def visit_assignment(self, node: ASTNode) -> None:
        self.generic_visit(node)
    
    def visit_statement(self, node: ASTNode) -> None:
        self.generic_visit(node)
    
    def visit_expression(self, node: ASTNode) -> None:
        self.generic_visit(node)


class FunctionCollector(ASTVisitor):
    """Visitor that collects all function definitions."""
    
    def __init__(self):
        self.functions: List[ASTNode] = []
    
    def visit_function(self, node: ASTNode) -> None:
        self.functions.append(node)
        self.generic_visit(node)


class CallCollector(ASTVisitor):
    """Visitor that collects all function calls."""
    
    def __init__(self):
        self.calls: List[ASTNode] = []
    
    def visit_call(self, node: ASTNode) -> None:
        self.calls.append(node)
        self.generic_visit(node)


class VariableCollector(ASTVisitor):
    """Visitor that collects all variable declarations."""
    
    def __init__(self):
        self.variables: Dict[str, str] = {}  # name -> type
    
    def visit_declaration(self, node: ASTNode) -> None:
        # Extract variable name and type from metadata
        if "var_name" in node.metadata and "var_type" in node.metadata:
            self.variables[node.metadata["var_name"]] = node.metadata["var_type"]
        self.generic_visit(node)


def find_functions(source: str) -> List[Dict[str, Any]]:
    """
    Find all function definitions in source code.
    
    Args:
        source: C++ source code
    
    Returns:
        List of function information dictionaries
    """
    functions = []
    
    # Pattern for function definitions
    func_pattern = r'''
        (?P<attrs>(?:\[\[[^\]]+\]\]\s*)*)
        (?P<template>template\s*<[^>]+>\s*)?
        (?P<inline>inline\s+)?
        (?P<static>static\s+)?
        (?P<virtual>virtual\s+)?
        (?P<return_type>(?:const\s+)?[\w:]+(?:\s*[*&])?\s+)
        (?:(?P<class>[\w:]+)::)?
        (?P<name>\w+)\s*
        \((?P<params>[^)]*)\)\s*
        (?P<const>const)?\s*
        (?P<noexcept>noexcept(?:\([^)]*\))?)?\s*
        (?P<override>override)?\s*
        (?P<final>final)?\s*
        (?=\{|;)
    '''
    
    for match in re.finditer(func_pattern, source, re.VERBOSE | re.MULTILINE):
        # Check if it's a definition (has body) or declaration
        pos_after = match.end()
        if pos_after < len(source) and source[pos_after] == '{':
            # Find matching closing brace
            brace_count = 1
            pos = pos_after + 1
            while pos < len(source) and brace_count > 0:
                if source[pos] == '{':
                    brace_count += 1
                elif source[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            body = source[pos_after + 1:pos - 1]
        else:
            body = None
        
        func_info = {
            "name": match.group("name"),
            "class_name": match.group("class"),
            "return_type": match.group("return_type").strip() if match.group("return_type") else "void",
            "params": match.group("params"),
            "is_const": bool(match.group("const")),
            "is_virtual": bool(match.group("virtual")),
            "is_static": bool(match.group("static")),
            "is_inline": bool(match.group("inline")),
            "is_override": bool(match.group("override")),
            "is_template": bool(match.group("template")),
            "has_body": body is not None,
            "body": body,
            "start_pos": match.start(),
            "end_pos": pos if body else match.end(),
            "line_start": source[:match.start()].count('\n') + 1,
        }
        
        if body:
            func_info["line_end"] = source[:pos].count('\n') + 1
        
        functions.append(func_info)
    
    return functions


def extract_function_body(source: str, function_name: str) -> Optional[str]:
    """
    Extract the body of a specific function.
    
    Args:
        source: C++ source code
        function_name: Name of function to extract
    
    Returns:
        Function body if found, None otherwise
    """
    functions = find_functions(source)
    
    for func in functions:
        if func["name"] == function_name:
            return func.get("body")
    
    return None


def get_function_signature(source: str, function_name: str) -> Optional[str]:
    """
    Get the signature of a specific function.
    
    Args:
        source: C++ source code
        function_name: Name of function
    
    Returns:
        Function signature if found, None otherwise
    """
    functions = find_functions(source)
    
    for func in functions:
        if func["name"] == function_name:
            class_prefix = f"{func['class_name']}::" if func["class_name"] else ""
            const_suffix = " const" if func["is_const"] else ""
            return f"{func['return_type']} {class_prefix}{func['name']}({func['params']}){const_suffix}"
    
    return None


def count_statements(body: str) -> int:
    """
    Count the number of statements in a function body.
    
    Args:
        body: Function body code
    
    Returns:
        Approximate statement count
    """
    # Remove string literals and comments
    cleaned = re.sub(r'"[^"]*"', '""', body)
    cleaned = re.sub(r"'[^']*'", "''", cleaned)
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Count semicolons (approximate statement count)
    return cleaned.count(';')


def count_branches(body: str) -> int:
    """
    Count branch statements in a function body.
    
    Args:
        body: Function body code
    
    Returns:
        Number of branch statements (if, switch, ternary)
    """
    if_count = len(re.findall(r'\bif\s*\(', body))
    switch_count = len(re.findall(r'\bswitch\s*\(', body))
    ternary_count = body.count('?')  # Approximate
    
    return if_count + switch_count + ternary_count


def count_loops(body: str) -> int:
    """
    Count loop statements in a function body.
    
    Args:
        body: Function body code
    
    Returns:
        Number of loop statements
    """
    for_count = len(re.findall(r'\bfor\s*\(', body))
    while_count = len(re.findall(r'\bwhile\s*\(', body))
    do_count = len(re.findall(r'\bdo\s*\{', body))
    
    return for_count + while_count + do_count


def extract_identifiers(code: str) -> List[str]:
    """
    Extract all identifiers from code.
    
    Args:
        code: C++ code snippet
    
    Returns:
        List of unique identifiers
    """
    # Remove string literals and comments
    cleaned = re.sub(r'"[^"]*"', '', code)
    cleaned = re.sub(r"'[^']*'", '', cleaned)
    cleaned = re.sub(r'//[^\n]*', '', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Extract identifiers
    identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', cleaned)
    
    # Remove keywords
    keywords = {
        'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
        'break', 'continue', 'return', 'goto', 'try', 'catch', 'throw',
        'class', 'struct', 'union', 'enum', 'namespace', 'template',
        'public', 'private', 'protected', 'virtual', 'override', 'final',
        'static', 'const', 'volatile', 'mutable', 'inline', 'extern',
        'auto', 'register', 'typedef', 'typename', 'using',
        'void', 'bool', 'char', 'short', 'int', 'long', 'float', 'double',
        'signed', 'unsigned', 'true', 'false', 'nullptr', 'NULL',
        'new', 'delete', 'this', 'sizeof', 'alignof', 'decltype',
        'constexpr', 'consteval', 'constinit', 'noexcept',
    }
    
    return list(set(id for id in identifiers if id not in keywords))


def extract_literals(code: str) -> Dict[str, List[str]]:
    """
    Extract literals from code.
    
    Args:
        code: C++ code snippet
    
    Returns:
        Dictionary with literal types as keys and lists of values
    """
    literals = {
        "integers": [],
        "floats": [],
        "strings": [],
        "chars": [],
    }
    
    # Integers
    literals["integers"] = re.findall(r'\b(0x[0-9a-fA-F]+|0b[01]+|0[0-7]+|\d+)\b', code)
    
    # Floats
    literals["floats"] = re.findall(r'\b(\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fFlL]?\b', code)
    
    # Strings
    literals["strings"] = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', code)
    
    # Characters
    literals["chars"] = re.findall(r"'([^'\\]|\\.)'", code)
    
    return literals


def compute_code_hash(code: str) -> str:
    """
    Compute a hash of code for comparison.
    
    Normalizes whitespace before hashing.
    
    Args:
        code: C++ code
    
    Returns:
        SHA256 hash of normalized code
    """
    import hashlib
    
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', code.strip())
    
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def diff_functions(func1: str, func2: str) -> List[tuple]:
    """
    Get differences between two function bodies.
    
    Args:
        func1: First function body
        func2: Second function body
    
    Returns:
        List of (operation, line) tuples
    """
    import difflib
    
    lines1 = func1.splitlines()
    lines2 = func2.splitlines()
    
    differ = difflib.unified_diff(lines1, lines2, lineterm='')
    
    diffs = []
    for line in differ:
        if line.startswith('+') and not line.startswith('+++'):
            diffs.append(('add', line[1:]))
        elif line.startswith('-') and not line.startswith('---'):
            diffs.append(('remove', line[1:]))
    
    return diffs
