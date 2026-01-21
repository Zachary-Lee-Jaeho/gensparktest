"""
C++ Parser using tree-sitter for VEGA-Verified.

Provides parsing and control flow graph construction for C++ code.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import re
import hashlib


class NodeType(Enum):
    """CFG node types."""
    ENTRY = "entry"
    EXIT = "exit"
    STATEMENT = "statement"
    BRANCH = "branch"
    LOOP_HEADER = "loop_header"
    SWITCH = "switch"
    CASE = "case"
    RETURN = "return"
    CALL = "call"
    ASSIGNMENT = "assignment"


@dataclass
class CFGNode:
    """Control Flow Graph node."""
    id: str
    node_type: NodeType
    code: str = ""
    line_start: int = 0
    line_end: int = 0
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # For branch nodes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_successor(self, node_id: str) -> None:
        """Add a successor node."""
        if node_id not in self.successors:
            self.successors.append(node_id)
    
    def add_predecessor(self, node_id: str) -> None:
        """Add a predecessor node."""
        if node_id not in self.predecessors:
            self.predecessors.append(node_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "code": self.code,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "successors": self.successors,
            "predecessors": self.predecessors,
            "condition": self.condition,
        }


@dataclass
class ControlFlowGraph:
    """Control Flow Graph for a function."""
    function_name: str
    nodes: Dict[str, CFGNode] = field(default_factory=dict)
    entry_node: Optional[str] = None
    exit_nodes: List[str] = field(default_factory=list)
    
    def add_node(self, node: CFGNode) -> None:
        """Add a node to the CFG."""
        self.nodes[node.id] = node
        
        if node.node_type == NodeType.ENTRY:
            self.entry_node = node.id
        elif node.node_type == NodeType.EXIT:
            self.exit_nodes.append(node.id)
    
    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add an edge between nodes."""
        if from_id in self.nodes and to_id in self.nodes:
            self.nodes[from_id].add_successor(to_id)
            self.nodes[to_id].add_predecessor(from_id)
    
    def get_paths(self, max_paths: int = 100) -> List[List[str]]:
        """Get all paths from entry to exit (bounded)."""
        if not self.entry_node or not self.exit_nodes:
            return []
        
        paths = []
        stack = [(self.entry_node, [self.entry_node])]
        visited_states = set()
        
        while stack and len(paths) < max_paths:
            node_id, path = stack.pop()
            
            if node_id in self.exit_nodes:
                paths.append(path)
                continue
            
            node = self.nodes.get(node_id)
            if not node:
                continue
            
            # Prevent infinite loops
            state = (node_id, tuple(path[-5:]))  # Last 5 nodes
            if state in visited_states:
                continue
            visited_states.add(state)
            
            for succ_id in node.successors:
                new_path = path + [succ_id]
                stack.append((succ_id, new_path))
        
        return paths
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "entry_node": self.entry_node,
            "exit_nodes": self.exit_nodes,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }
    
    def __str__(self) -> str:
        lines = [f"CFG: {self.function_name}"]
        lines.append(f"  Entry: {self.entry_node}")
        lines.append(f"  Exit: {self.exit_nodes}")
        lines.append(f"  Nodes ({len(self.nodes)}):")
        for node_id, node in self.nodes.items():
            lines.append(f"    {node_id} ({node.node_type.value}): -> {node.successors}")
        return "\n".join(lines)


@dataclass
class Parameter:
    """Function parameter."""
    name: str
    type_name: str
    is_const: bool = False
    is_reference: bool = False
    is_pointer: bool = False
    default_value: Optional[str] = None


@dataclass
class ParsedFunction:
    """Parsed function representation."""
    name: str
    return_type: str
    parameters: List[Parameter]
    body: str
    full_source: str
    class_name: Optional[str] = None
    is_const: bool = False
    line_start: int = 0
    line_end: int = 0
    cfg: Optional[ControlFlowGraph] = None
    local_variables: Dict[str, str] = field(default_factory=dict)  # name -> type
    called_functions: List[str] = field(default_factory=list)
    
    @property
    def qualified_name(self) -> str:
        """Get fully qualified function name."""
        if self.class_name:
            return f"{self.class_name}::{self.name}"
        return self.name
    
    @property
    def signature(self) -> str:
        """Get function signature."""
        params = ", ".join(
            f"{p.type_name}{' const' if p.is_const else ''}"
            f"{'&' if p.is_reference else ''}{'*' if p.is_pointer else ''} {p.name}"
            for p in self.parameters
        )
        const_suffix = " const" if self.is_const else ""
        return f"{self.return_type} {self.qualified_name}({params}){const_suffix}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "return_type": self.return_type,
            "parameters": [
                {"name": p.name, "type": p.type_name, "const": p.is_const,
                 "reference": p.is_reference, "pointer": p.is_pointer}
                for p in self.parameters
            ],
            "is_const": self.is_const,
            "class_name": self.class_name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "local_variables": self.local_variables,
            "called_functions": self.called_functions,
            "cfg": self.cfg.to_dict() if self.cfg else None,
        }


class CppParser:
    """
    C++ code parser for VEGA-Verified.
    
    Uses regex-based parsing as a fallback when tree-sitter is not available.
    Provides function extraction and CFG construction.
    """
    
    def __init__(self, use_tree_sitter: bool = True):
        """
        Initialize parser.
        
        Args:
            use_tree_sitter: Whether to use tree-sitter (falls back to regex if unavailable)
        """
        self.use_tree_sitter = use_tree_sitter
        self._tree_sitter_available = False
        self._parser = None
        
        if use_tree_sitter:
            self._init_tree_sitter()
    
    def _init_tree_sitter(self) -> None:
        """Initialize tree-sitter parser."""
        try:
            import tree_sitter_cpp as tscpp
            from tree_sitter import Language, Parser
            
            self._parser = Parser(Language(tscpp.language()))
            self._tree_sitter_available = True
        except ImportError:
            self._tree_sitter_available = False
    
    def parse(self, source: str) -> List[ParsedFunction]:
        """
        Parse C++ source code and extract functions.
        
        Args:
            source: C++ source code
        
        Returns:
            List of parsed functions
        """
        if self._tree_sitter_available and self._parser:
            return self._parse_with_tree_sitter(source)
        else:
            return self._parse_with_regex(source)
    
    def _parse_with_tree_sitter(self, source: str) -> List[ParsedFunction]:
        """Parse using tree-sitter."""
        tree = self._parser.parse(bytes(source, "utf8"))
        functions = []
        
        def visit_node(node, class_name=None):
            if node.type == "function_definition":
                func = self._extract_function_tree_sitter(node, source, class_name)
                if func:
                    functions.append(func)
            
            elif node.type == "class_specifier":
                # Extract class name
                name_node = None
                for child in node.children:
                    if child.type == "type_identifier":
                        name_node = child
                        break
                
                if name_node:
                    new_class_name = source[name_node.start_byte:name_node.end_byte]
                    for child in node.children:
                        visit_node(child, new_class_name)
            
            else:
                for child in node.children:
                    visit_node(child, class_name)
        
        visit_node(tree.root_node)
        return functions
    
    def _extract_function_tree_sitter(
        self,
        node,
        source: str,
        class_name: Optional[str] = None
    ) -> Optional[ParsedFunction]:
        """Extract function from tree-sitter node."""
        # Get function name and return type
        declarator = None
        return_type = "void"
        body_node = None
        
        for child in node.children:
            if child.type in ("type_identifier", "primitive_type", "qualified_identifier"):
                return_type = source[child.start_byte:child.end_byte]
            elif child.type == "function_declarator":
                declarator = child
            elif child.type == "compound_statement":
                body_node = child
        
        if not declarator:
            return None
        
        # Extract function name
        func_name = None
        params = []
        
        for child in declarator.children:
            if child.type in ("identifier", "field_identifier", "qualified_identifier"):
                func_name = source[child.start_byte:child.end_byte]
                # Handle qualified names
                if "::" in func_name:
                    parts = func_name.split("::")
                    class_name = parts[0]
                    func_name = parts[-1]
            elif child.type == "parameter_list":
                params = self._extract_parameters_tree_sitter(child, source)
        
        if not func_name:
            return None
        
        # Get body
        body = ""
        if body_node:
            body = source[body_node.start_byte:body_node.end_byte]
        
        # Calculate line numbers
        start_line = source[:node.start_byte].count('\n') + 1
        end_line = source[:node.end_byte].count('\n') + 1
        
        parsed_func = ParsedFunction(
            name=func_name,
            return_type=return_type,
            parameters=params,
            body=body,
            full_source=source[node.start_byte:node.end_byte],
            class_name=class_name,
            line_start=start_line,
            line_end=end_line,
        )
        
        # Build CFG
        if body_node:
            parsed_func.cfg = self._build_cfg(body_node, source, func_name)
            parsed_func.called_functions = self._extract_calls(body_node, source)
            parsed_func.local_variables = self._extract_locals(body_node, source)
        
        return parsed_func
    
    def _extract_parameters_tree_sitter(self, param_list, source: str) -> List[Parameter]:
        """Extract parameters from tree-sitter parameter list."""
        params = []
        
        for child in param_list.children:
            if child.type == "parameter_declaration":
                param_name = ""
                param_type = ""
                is_const = False
                is_ref = False
                is_ptr = False
                
                for subchild in child.children:
                    text = source[subchild.start_byte:subchild.end_byte]
                    
                    if subchild.type == "type_qualifier" and text == "const":
                        is_const = True
                    elif subchild.type in ("type_identifier", "primitive_type", "qualified_identifier"):
                        param_type = text
                    elif subchild.type == "identifier":
                        param_name = text
                    elif subchild.type == "reference_declarator":
                        is_ref = True
                        # Extract name from reference declarator
                        for rchild in subchild.children:
                            if rchild.type == "identifier":
                                param_name = source[rchild.start_byte:rchild.end_byte]
                    elif subchild.type == "pointer_declarator":
                        is_ptr = True
                        for pchild in subchild.children:
                            if pchild.type == "identifier":
                                param_name = source[pchild.start_byte:pchild.end_byte]
                
                if param_name or param_type:
                    params.append(Parameter(
                        name=param_name or f"param{len(params)}",
                        type_name=param_type,
                        is_const=is_const,
                        is_reference=is_ref,
                        is_pointer=is_ptr,
                    ))
        
        return params
    
    def _build_cfg(self, body_node, source: str, func_name: str) -> ControlFlowGraph:
        """Build control flow graph from function body."""
        cfg = ControlFlowGraph(function_name=func_name)
        
        # Create entry node
        entry = CFGNode(id="entry", node_type=NodeType.ENTRY)
        cfg.add_node(entry)
        
        # Process body
        node_counter = [0]  # Mutable counter for node IDs
        
        last_node_id = self._process_compound_statement(
            body_node, source, cfg, node_counter, "entry"
        )
        
        # Create exit node if not already created
        if not cfg.exit_nodes:
            exit_node = CFGNode(id="exit", node_type=NodeType.EXIT)
            cfg.add_node(exit_node)
            if last_node_id:
                cfg.add_edge(last_node_id, "exit")
        
        return cfg
    
    def _process_compound_statement(
        self,
        node,
        source: str,
        cfg: ControlFlowGraph,
        counter: List[int],
        prev_node_id: str
    ) -> str:
        """Process compound statement and return last node ID."""
        current_id = prev_node_id
        
        for child in node.children:
            if child.type == "{" or child.type == "}":
                continue
            
            current_id = self._process_statement(
                child, source, cfg, counter, current_id
            )
        
        return current_id
    
    def _process_statement(
        self,
        node,
        source: str,
        cfg: ControlFlowGraph,
        counter: List[int],
        prev_node_id: str
    ) -> str:
        """Process a statement and return new node ID."""
        code = source[node.start_byte:node.end_byte].strip()
        start_line = source[:node.start_byte].count('\n') + 1
        
        if node.type == "if_statement":
            return self._process_if_statement(node, source, cfg, counter, prev_node_id)
        
        elif node.type == "switch_statement":
            return self._process_switch_statement(node, source, cfg, counter, prev_node_id)
        
        elif node.type in ("for_statement", "while_statement", "do_statement"):
            return self._process_loop_statement(node, source, cfg, counter, prev_node_id)
        
        elif node.type == "return_statement":
            counter[0] += 1
            node_id = f"n{counter[0]}"
            cfg_node = CFGNode(
                id=node_id,
                node_type=NodeType.RETURN,
                code=code,
                line_start=start_line
            )
            cfg.add_node(cfg_node)
            cfg.add_edge(prev_node_id, node_id)
            
            # Return nodes go to exit
            if "exit" not in cfg.nodes:
                exit_node = CFGNode(id="exit", node_type=NodeType.EXIT)
                cfg.add_node(exit_node)
            cfg.add_edge(node_id, "exit")
            
            return node_id
        
        elif node.type == "compound_statement":
            return self._process_compound_statement(node, source, cfg, counter, prev_node_id)
        
        elif node.type in ("expression_statement", "declaration"):
            counter[0] += 1
            node_id = f"n{counter[0]}"
            
            # Determine if it's a call or assignment
            node_type = NodeType.STATEMENT
            if "=" in code:
                node_type = NodeType.ASSIGNMENT
            elif "(" in code:
                node_type = NodeType.CALL
            
            cfg_node = CFGNode(
                id=node_id,
                node_type=node_type,
                code=code,
                line_start=start_line
            )
            cfg.add_node(cfg_node)
            cfg.add_edge(prev_node_id, node_id)
            
            return node_id
        
        # Default: pass through
        return prev_node_id
    
    def _process_if_statement(
        self,
        node,
        source: str,
        cfg: ControlFlowGraph,
        counter: List[int],
        prev_node_id: str
    ) -> str:
        """Process if statement."""
        counter[0] += 1
        branch_id = f"n{counter[0]}"
        
        # Extract condition
        condition = ""
        for child in node.children:
            if child.type == "condition_clause":
                condition = source[child.start_byte:child.end_byte]
                break
        
        branch_node = CFGNode(
            id=branch_id,
            node_type=NodeType.BRANCH,
            code=f"if {condition}",
            condition=condition,
            line_start=source[:node.start_byte].count('\n') + 1
        )
        cfg.add_node(branch_node)
        cfg.add_edge(prev_node_id, branch_id)
        
        # Process then branch
        then_branch = None
        else_branch = None
        
        for child in node.children:
            if child.type == "compound_statement":
                if then_branch is None:
                    then_branch = child
                else:
                    else_branch = child
            elif child.type == "else_clause":
                for subchild in child.children:
                    if subchild.type == "compound_statement":
                        else_branch = subchild
        
        # Process then
        then_last = branch_id
        if then_branch:
            then_last = self._process_compound_statement(
                then_branch, source, cfg, counter, branch_id
            )
        
        # Process else
        else_last = branch_id
        if else_branch:
            else_last = self._process_compound_statement(
                else_branch, source, cfg, counter, branch_id
            )
        
        # Create merge node
        counter[0] += 1
        merge_id = f"n{counter[0]}"
        merge_node = CFGNode(
            id=merge_id,
            node_type=NodeType.STATEMENT,
            code="// merge"
        )
        cfg.add_node(merge_node)
        
        cfg.add_edge(then_last, merge_id)
        if else_branch:
            cfg.add_edge(else_last, merge_id)
        else:
            cfg.add_edge(branch_id, merge_id)  # No else, direct to merge
        
        return merge_id
    
    def _process_switch_statement(
        self,
        node,
        source: str,
        cfg: ControlFlowGraph,
        counter: List[int],
        prev_node_id: str
    ) -> str:
        """Process switch statement."""
        counter[0] += 1
        switch_id = f"n{counter[0]}"
        
        # Extract condition
        condition = ""
        for child in node.children:
            if child.type == "condition_clause":
                condition = source[child.start_byte:child.end_byte]
                break
        
        switch_node = CFGNode(
            id=switch_id,
            node_type=NodeType.SWITCH,
            code=f"switch {condition}",
            condition=condition,
            line_start=source[:node.start_byte].count('\n') + 1
        )
        cfg.add_node(switch_node)
        cfg.add_edge(prev_node_id, switch_id)
        
        # Process cases
        case_ends = []
        
        for child in node.children:
            if child.type == "compound_statement":
                for case_child in child.children:
                    if case_child.type in ("case_statement", "default_statement"):
                        counter[0] += 1
                        case_id = f"n{counter[0]}"
                        case_code = source[case_child.start_byte:case_child.end_byte].split('\n')[0]
                        
                        case_node = CFGNode(
                            id=case_id,
                            node_type=NodeType.CASE,
                            code=case_code,
                            line_start=source[:case_child.start_byte].count('\n') + 1
                        )
                        cfg.add_node(case_node)
                        cfg.add_edge(switch_id, case_id)
                        
                        # Process case body
                        last_id = case_id
                        for stmt in case_child.children:
                            if stmt.type not in ("case", ":", "default"):
                                last_id = self._process_statement(
                                    stmt, source, cfg, counter, last_id
                                )
                        
                        case_ends.append(last_id)
        
        # Create merge node
        counter[0] += 1
        merge_id = f"n{counter[0]}"
        merge_node = CFGNode(
            id=merge_id,
            node_type=NodeType.STATEMENT,
            code="// switch merge"
        )
        cfg.add_node(merge_node)
        
        for case_end in case_ends:
            cfg.add_edge(case_end, merge_id)
        
        return merge_id
    
    def _process_loop_statement(
        self,
        node,
        source: str,
        cfg: ControlFlowGraph,
        counter: List[int],
        prev_node_id: str
    ) -> str:
        """Process loop statement."""
        counter[0] += 1
        header_id = f"n{counter[0]}"
        
        code = source[node.start_byte:node.end_byte].split('{')[0].strip()
        
        header_node = CFGNode(
            id=header_id,
            node_type=NodeType.LOOP_HEADER,
            code=code,
            line_start=source[:node.start_byte].count('\n') + 1
        )
        cfg.add_node(header_node)
        cfg.add_edge(prev_node_id, header_id)
        
        # Process body
        body_last = header_id
        for child in node.children:
            if child.type == "compound_statement":
                body_last = self._process_compound_statement(
                    child, source, cfg, counter, header_id
                )
                break
        
        # Loop back edge
        cfg.add_edge(body_last, header_id)
        
        # Exit from loop
        counter[0] += 1
        exit_id = f"n{counter[0]}"
        exit_node = CFGNode(
            id=exit_id,
            node_type=NodeType.STATEMENT,
            code="// loop exit"
        )
        cfg.add_node(exit_node)
        cfg.add_edge(header_id, exit_id)
        
        return exit_id
    
    def _extract_calls(self, node, source: str) -> List[str]:
        """Extract called function names."""
        calls = []
        
        def visit(n):
            if n.type == "call_expression":
                for child in n.children:
                    if child.type in ("identifier", "field_expression"):
                        calls.append(source[child.start_byte:child.end_byte])
                        break
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return calls
    
    def _extract_locals(self, node, source: str) -> Dict[str, str]:
        """Extract local variable declarations."""
        locals_dict = {}
        
        def visit(n):
            if n.type == "declaration":
                var_type = ""
                var_name = ""
                
                for child in n.children:
                    if child.type in ("type_identifier", "primitive_type"):
                        var_type = source[child.start_byte:child.end_byte]
                    elif child.type == "init_declarator":
                        for subchild in child.children:
                            if subchild.type == "identifier":
                                var_name = source[subchild.start_byte:subchild.end_byte]
                    elif child.type == "identifier":
                        var_name = source[child.start_byte:child.end_byte]
                
                if var_name and var_type:
                    locals_dict[var_name] = var_type
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return locals_dict
    
    def _parse_with_regex(self, source: str) -> List[ParsedFunction]:
        """Fallback regex-based parsing."""
        functions = []
        
        # Pattern for function definitions
        func_pattern = r'''
            (?:(?P<return_type>[\w:]+(?:\s*[*&])?\s+))?
            (?:(?P<class>[\w:]+)::)?
            (?P<name>\w+)\s*
            \((?P<params>[^)]*)\)\s*
            (?P<const>const)?\s*
            \{
        '''
        
        for match in re.finditer(func_pattern, source, re.VERBOSE | re.MULTILINE):
            start = match.start()
            
            # Find matching closing brace
            brace_count = 1
            pos = match.end()
            while pos < len(source) and brace_count > 0:
                if source[pos] == '{':
                    brace_count += 1
                elif source[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            body = source[match.end():pos-1]
            full_source = source[start:pos]
            
            # Parse parameters
            params = []
            param_str = match.group('params').strip()
            if param_str:
                for param in param_str.split(','):
                    param = param.strip()
                    if param:
                        parts = param.rsplit(' ', 1)
                        if len(parts) == 2:
                            ptype, pname = parts
                            pname = pname.strip('*&')
                            params.append(Parameter(
                                name=pname,
                                type_name=ptype.strip(),
                                is_reference='&' in param,
                                is_pointer='*' in param,
                                is_const='const' in param
                            ))
            
            start_line = source[:start].count('\n') + 1
            end_line = source[:pos].count('\n') + 1
            
            func = ParsedFunction(
                name=match.group('name'),
                return_type=(match.group('return_type') or 'void').strip(),
                parameters=params,
                body=body,
                full_source=full_source,
                class_name=match.group('class'),
                is_const=bool(match.group('const')),
                line_start=start_line,
                line_end=end_line,
            )
            
            # Build simple CFG
            func.cfg = self._build_simple_cfg(body, func.name)
            
            functions.append(func)
        
        return functions
    
    def _build_simple_cfg(self, body: str, func_name: str) -> ControlFlowGraph:
        """Build a simple CFG using regex."""
        cfg = ControlFlowGraph(function_name=func_name)
        
        entry = CFGNode(id="entry", node_type=NodeType.ENTRY)
        cfg.add_node(entry)
        
        # Split by statements (simplified)
        statements = [s.strip() for s in body.split(';') if s.strip()]
        
        prev_id = "entry"
        for i, stmt in enumerate(statements):
            node_id = f"n{i+1}"
            
            # Determine node type
            node_type = NodeType.STATEMENT
            if stmt.startswith('return'):
                node_type = NodeType.RETURN
            elif stmt.startswith('if'):
                node_type = NodeType.BRANCH
            elif stmt.startswith(('for', 'while')):
                node_type = NodeType.LOOP_HEADER
            elif stmt.startswith('switch'):
                node_type = NodeType.SWITCH
            elif '=' in stmt:
                node_type = NodeType.ASSIGNMENT
            elif '(' in stmt:
                node_type = NodeType.CALL
            
            node = CFGNode(
                id=node_id,
                node_type=node_type,
                code=stmt,
            )
            cfg.add_node(node)
            cfg.add_edge(prev_id, node_id)
            
            if node_type == NodeType.RETURN:
                if "exit" not in cfg.nodes:
                    exit_node = CFGNode(id="exit", node_type=NodeType.EXIT)
                    cfg.add_node(exit_node)
                cfg.add_edge(node_id, "exit")
            
            prev_id = node_id
        
        # Add exit if not already present
        if "exit" not in cfg.nodes:
            exit_node = CFGNode(id="exit", node_type=NodeType.EXIT)
            cfg.add_node(exit_node)
            cfg.add_edge(prev_id, "exit")
        
        return cfg
    
    def parse_function(self, source: str, function_name: str) -> Optional[ParsedFunction]:
        """
        Parse a specific function from source code.
        
        Args:
            source: C++ source code
            function_name: Name of function to extract
        
        Returns:
            ParsedFunction if found, None otherwise
        """
        functions = self.parse(source)
        
        for func in functions:
            if func.name == function_name or func.qualified_name == function_name:
                return func
        
        return None
