"""
Symbolic Execution Engine for VEGA-Verified.

Performs symbolic execution on C++ code to extract path conditions,
variable constraints, and behavioral specifications.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from enum import Enum
from copy import deepcopy
import re
import logging

logger = logging.getLogger(__name__)

# Z3 availability check
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("Z3 not available. Symbolic execution will use string-based constraints only.")

# Clang AST Parser availability check
try:
    from ..parsing.clang_ast_parser import (
        ClangASTParser, 
        ClangSymbolicExecutor,
        FunctionInfo,
        SwitchStatement,
        ASTNode,
        NodeType,
        CLANG_AVAILABLE
    )
except ImportError:
    CLANG_AVAILABLE = False
    ClangASTParser = None
    logger.info("Clang AST parser not available. Using regex-based parsing.")


class SymbolicValueType(Enum):
    """Types of symbolic values."""
    CONCRETE = "concrete"
    SYMBOLIC = "symbolic"
    UNKNOWN = "unknown"


@dataclass
class SymbolicValue:
    """Represents a symbolic or concrete value."""
    name: str
    value_type: SymbolicValueType
    concrete_value: Optional[Any] = None
    constraints: List[str] = field(default_factory=list)
    depends_on: Set[str] = field(default_factory=set)
    
    @classmethod
    def concrete(cls, name: str, value: Any) -> 'SymbolicValue':
        """Create a concrete value."""
        return cls(
            name=name,
            value_type=SymbolicValueType.CONCRETE,
            concrete_value=value
        )
    
    @classmethod
    def symbolic(cls, name: str, depends_on: Optional[Set[str]] = None) -> 'SymbolicValue':
        """Create a symbolic value."""
        return cls(
            name=name,
            value_type=SymbolicValueType.SYMBOLIC,
            depends_on=depends_on or {name}
        )
    
    def is_concrete(self) -> bool:
        return self.value_type == SymbolicValueType.CONCRETE
    
    def is_symbolic(self) -> bool:
        return self.value_type == SymbolicValueType.SYMBOLIC
    
    def add_constraint(self, constraint: str) -> None:
        """Add a constraint to this value."""
        self.constraints.append(constraint)
    
    def to_smt(self) -> str:
        """Convert to SMT-LIB representation."""
        if self.is_concrete():
            return str(self.concrete_value)
        else:
            return self.name
    
    def __str__(self) -> str:
        if self.is_concrete():
            return f"{self.name} = {self.concrete_value}"
        else:
            return f"{self.name} (symbolic: {self.depends_on})"


@dataclass
class SymbolicState:
    """Represents the symbolic state at a program point."""
    variables: Dict[str, SymbolicValue] = field(default_factory=dict)
    path_condition: List[str] = field(default_factory=list)
    return_value: Optional[SymbolicValue] = None
    is_terminated: bool = False
    termination_reason: str = ""
    
    def copy(self) -> 'SymbolicState':
        """Create a deep copy of this state."""
        new_state = SymbolicState(
            variables=deepcopy(self.variables),
            path_condition=list(self.path_condition),
            return_value=deepcopy(self.return_value),
            is_terminated=self.is_terminated,
            termination_reason=self.termination_reason
        )
        return new_state
    
    def get_variable(self, name: str) -> Optional[SymbolicValue]:
        """Get a variable's symbolic value."""
        return self.variables.get(name)
    
    def set_variable(self, name: str, value: SymbolicValue) -> None:
        """Set a variable's value."""
        self.variables[name] = value
    
    def add_path_constraint(self, constraint: str) -> None:
        """Add a constraint to the path condition."""
        self.path_condition.append(constraint)
    
    def terminate(self, reason: str, return_val: Optional[SymbolicValue] = None) -> None:
        """Mark state as terminated."""
        self.is_terminated = True
        self.termination_reason = reason
        self.return_value = return_val
    
    def path_condition_smt(self) -> str:
        """Get path condition as SMT formula."""
        if not self.path_condition:
            return "true"
        
        if len(self.path_condition) == 1:
            return self.path_condition[0]
        
        return f"(and {' '.join(self.path_condition)})"
    
    def is_satisfiable(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if the current path condition is satisfiable using Z3.
        
        Returns:
            Tuple of (is_sat, model_dict) where model_dict contains
            variable assignments if satisfiable.
        """
        if not Z3_AVAILABLE:
            # Fallback: assume satisfiable if we can't check
            return True, None
        
        if not self.path_condition:
            return True, {}
        
        try:
            solver = z3.Solver()
            solver.set("timeout", 5000)  # 5 second timeout
            
            # Create Z3 variables for all symbolic values
            z3_vars = {}
            for name, sym_val in self.variables.items():
                if sym_val.is_symbolic():
                    z3_vars[name] = z3.Int(name)
            
            # Parse and add constraints
            for constraint in self.path_condition:
                z3_constraint = self._parse_constraint_to_z3(constraint, z3_vars)
                if z3_constraint is not None:
                    solver.add(z3_constraint)
            
            result = solver.check()
            
            if result == z3.sat:
                model = solver.model()
                assignments = {}
                for name, var in z3_vars.items():
                    try:
                        val = model.eval(var)
                        if val is not None:
                            assignments[name] = str(val)
                    except Exception:
                        pass
                return True, assignments
            elif result == z3.unsat:
                return False, None
            else:  # unknown
                return True, None  # Assume satisfiable on timeout
                
        except Exception as e:
            logger.warning(f"Z3 satisfiability check failed: {e}")
            return True, None
    
    def _parse_constraint_to_z3(self, constraint: str, z3_vars: Dict[str, Any]) -> Optional[Any]:
        """
        Parse an SMT-LIB style constraint to Z3 expression.
        
        Args:
            constraint: SMT-LIB format constraint string
            z3_vars: Dictionary of Z3 variable objects
            
        Returns:
            Z3 expression or None if parsing fails
        """
        if not Z3_AVAILABLE:
            return None
        
        try:
            constraint = constraint.strip()
            
            # Handle simple equality: (= var value)
            eq_match = re.match(r'\(=\s+(\w+)\s+(\w+(?:::\w+)?)\)', constraint)
            if eq_match:
                var_name = eq_match.group(1)
                value = eq_match.group(2)
                if var_name in z3_vars:
                    # Try to parse value as integer, otherwise use as enum constant
                    try:
                        return z3_vars[var_name] == int(value)
                    except ValueError:
                        # Create an integer constant for enum value
                        enum_val = z3.Int(value)
                        return z3_vars[var_name] == enum_val
            
            # Handle negation: (not (= var value))
            not_eq_match = re.match(r'\(not\s+\(=\s+(\w+)\s+(\w+(?:::\w+)?)\)\)', constraint)
            if not_eq_match:
                var_name = not_eq_match.group(1)
                value = not_eq_match.group(2)
                if var_name in z3_vars:
                    try:
                        return z3_vars[var_name] != int(value)
                    except ValueError:
                        enum_val = z3.Int(value)
                        return z3_vars[var_name] != enum_val
            
            # Handle conjunction: (and ...)
            if constraint.startswith('(and '):
                # Parse sub-constraints
                inner = constraint[5:-1]  # Remove "(and " and ")"
                sub_constraints = self._split_smt_terms(inner)
                z3_subs = []
                for sub in sub_constraints:
                    z3_sub = self._parse_constraint_to_z3(sub, z3_vars)
                    if z3_sub is not None:
                        z3_subs.append(z3_sub)
                if z3_subs:
                    return z3.And(*z3_subs)
            
            # Handle less than: (< var value) or (bvslt var value)
            lt_match = re.match(r'\((?:<|bvslt)\s+(\w+)\s+(\d+)\)', constraint)
            if lt_match:
                var_name = lt_match.group(1)
                value = int(lt_match.group(2))
                if var_name in z3_vars:
                    return z3_vars[var_name] < value
            
            # Handle greater than: (> var value) or (bvsgt var value)
            gt_match = re.match(r'\((?:>|bvsgt)\s+(\w+)\s+(\d+)\)', constraint)
            if gt_match:
                var_name = gt_match.group(1)
                value = int(gt_match.group(2))
                if var_name in z3_vars:
                    return z3_vars[var_name] > value
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to parse constraint '{constraint}': {e}")
            return None
    
    def _split_smt_terms(self, smt_string: str) -> List[str]:
        """Split SMT string into individual terms, respecting parentheses."""
        terms = []
        depth = 0
        current_term = ""
        
        for char in smt_string:
            if char == '(':
                depth += 1
                current_term += char
            elif char == ')':
                depth -= 1
                current_term += char
                if depth == 0 and current_term.strip():
                    terms.append(current_term.strip())
                    current_term = ""
            elif char == ' ' and depth == 0:
                if current_term.strip():
                    terms.append(current_term.strip())
                    current_term = ""
            else:
                current_term += char
        
        if current_term.strip():
            terms.append(current_term.strip())
        
        return terms
    
    def __str__(self) -> str:
        lines = ["SymbolicState:"]
        lines.append(f"  Variables ({len(self.variables)}):")
        for name, val in self.variables.items():
            lines.append(f"    {val}")
        lines.append(f"  Path condition: {self.path_condition_smt()}")
        if self.is_terminated:
            lines.append(f"  Terminated: {self.termination_reason}")
            if self.return_value:
                lines.append(f"  Return: {self.return_value}")
        return "\n".join(lines)


@dataclass
class ExecutionPath:
    """Represents a complete execution path."""
    path_id: int
    final_state: SymbolicState
    statements_executed: List[str] = field(default_factory=list)
    branches_taken: List[Tuple[str, bool]] = field(default_factory=list)
    
    @property
    def path_condition(self) -> str:
        return self.final_state.path_condition_smt()
    
    @property
    def return_value(self) -> Optional[SymbolicValue]:
        return self.final_state.return_value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "path_condition": self.path_condition,
            "return_value": str(self.return_value) if self.return_value else None,
            "statements": len(self.statements_executed),
            "branches": self.branches_taken,
        }


class SymbolicExecutor:
    """
    Symbolic execution engine for C++ code.
    
    Executes code symbolically to extract:
    - Path conditions for each execution path
    - Variable constraints
    - Return value expressions
    - Pre/post conditions
    """
    
    def __init__(
        self,
        max_depth: int = 50,
        max_paths: int = 100,
        timeout_ms: int = 30000,
        verbose: bool = False,
        use_clang: bool = True,  # Use Clang AST if available
        clang_path: Optional[str] = None
    ):
        """
        Initialize symbolic executor.
        
        Args:
            max_depth: Maximum execution depth
            max_paths: Maximum number of paths to explore
            timeout_ms: Execution timeout in milliseconds
            verbose: Enable verbose output
            use_clang: Use Clang AST parser if available
            clang_path: Optional path to libclang library
        """
        self.max_depth = max_depth
        self.max_paths = max_paths
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        
        # Clang AST parser setup
        self.use_clang = use_clang and CLANG_AVAILABLE
        self.clang_parser: Optional['ClangASTParser'] = None
        if self.use_clang:
            try:
                self.clang_parser = ClangASTParser(clang_path)
                logger.info("Clang AST parser initialized for symbolic execution")
            except Exception as e:
                logger.warning(f"Failed to initialize Clang parser: {e}. Using regex fallback.")
                self.use_clang = False
        
        # Execution state
        self.paths: List[ExecutionPath] = []
        self.path_counter = 0
        
        # Known function behaviors
        self.function_models: Dict[str, callable] = {
            "getTargetKind": self._model_getTargetKind,
            "getOperand": self._model_getOperand,
            "getImm": self._model_getImm,
            "getReg": self._model_getReg,
            "isReg": self._model_isReg,
            "isImm": self._model_isImm,
            "isExpr": self._model_isExpr,
        }
    
    def execute(
        self,
        code: str,
        function_name: str,
        parameters: List[Tuple[str, str]],  # [(name, type), ...]
        initial_constraints: Optional[List[str]] = None
    ) -> List[ExecutionPath]:
        """
        Symbolically execute a function.
        
        Args:
            code: Function body code
            function_name: Name of the function
            parameters: List of (parameter_name, parameter_type) tuples
            initial_constraints: Optional initial constraints on parameters
        
        Returns:
            List of execution paths
        """
        self.paths = []
        self.path_counter = 0
        
        # Create initial state
        initial_state = SymbolicState()
        
        # Initialize parameters as symbolic values
        for param_name, param_type in parameters:
            sym_val = SymbolicValue.symbolic(param_name)
            initial_state.set_variable(param_name, sym_val)
        
        # Add initial constraints
        if initial_constraints:
            for constraint in initial_constraints:
                initial_state.add_path_constraint(constraint)
        
        # Parse and execute
        statements = self._parse_statements(code, function_name)
        self._execute_statements(statements, initial_state, 0)
        
        return self.paths
    
    def _parse_statements(self, code: str, function_name: str = "") -> List[Dict[str, Any]]:
        """
        Parse code into statement list.
        
        Uses Clang AST parser if available, otherwise falls back to regex-based parsing.
        """
        # Try Clang AST parser first
        if self.use_clang and self.clang_parser:
            try:
                return self._parse_statements_clang(code, function_name)
            except Exception as e:
                logger.warning(f"Clang parsing failed, using regex fallback: {e}")
        
        # Fallback to regex-based parsing
        return self._parse_statements_regex(code)
    
    def _parse_statements_clang(self, code: str, function_name: str = "") -> List[Dict[str, Any]]:
        """Parse code using Clang AST parser."""
        if not self.clang_parser:
            raise RuntimeError("Clang parser not initialized")
        
        # Parse the code
        result = self.clang_parser.parse_code(code, "symbolic_input.cpp")
        
        if "error" in result:
            raise RuntimeError(f"Clang parse error: {result['error']}")
        
        statements = []
        
        # Extract switch statements - enhanced parsing
        for switch in self.clang_parser.switches:
            # If Clang didn't parse cases properly, fall back to regex on original code
            if not switch.cases:
                # Try to extract switch from original code using regex
                if self.verbose:
                    logger.info("Clang switch parsing incomplete, using regex fallback")
                # Fall back to full regex parsing for this code
                return self._parse_statements_regex(code)
                continue
            
            switch_stmt = {
                'type': 'switch',
                'expr': switch.condition_var,
                'cases': [],
                'default': None,
                'source': 'clang'
            }
            
            for case in switch.cases:
                case_value = case.get('value', '')
                case_stmts = case.get('statements', [])
                has_break = case.get('has_break', False)
                
                # Convert statements to our format
                case_body = []
                for stmt in case_stmts:
                    if 'return' in stmt.lower():
                        match = re.search(r'return\s+(.+?);', stmt)
                        if match:
                            case_body.append({
                                'type': 'return',
                                'value': match.group(1)
                            })
                    else:
                        case_body.append({
                            'type': 'statement',
                            'text': stmt
                        })
                
                switch_stmt['cases'].append({
                    'case': case_value,
                    'body': case_body,
                    'has_break': has_break
                })
            
            if switch.has_default:
                default_body = []
                for stmt in switch.default_statements:
                    if 'return' in stmt.lower():
                        match = re.search(r'return\s+(.+?);', stmt)
                        if match:
                            default_body.append({
                                'type': 'return',
                                'value': match.group(1)
                            })
                switch_stmt['default'] = default_body
            
            statements.append(switch_stmt)
        
        # Extract function information if available
        for func_name, func_info in self.clang_parser.functions.items():
            if function_name and function_name not in func_name:
                continue
            
            # Process AST nodes for additional statements
            if func_info.ast:
                self._extract_statements_from_ast(func_info.ast, statements)
        
        if self.verbose:
            logger.info(f"Clang parsed {len(statements)} statements")
        
        # If Clang didn't find much, fall back to regex
        return statements if statements else self._parse_statements_regex(code)
    
    def _parse_switch_from_source(self, source_text: str) -> Optional[Dict[str, Any]]:
        """Parse switch statement from source text using regex."""
        lines = source_text.split('\n')
        
        # Extract switch expression
        match = re.search(r'switch\s*\((.+?)\)', source_text)
        expr = match.group(1) if match else ""
        
        switch_stmt = {
            'type': 'switch',
            'expr': expr,
            'cases': [],
            'default': None
        }
        
        current_case = None
        case_body = []
        
        for line in lines:
            line = line.strip()
            
            # Parse case
            case_match = re.match(r'case\s+(\w+(?:::\w+)?)\s*:', line)
            if case_match:
                # Save previous case
                if current_case is not None:
                    switch_stmt['cases'].append({
                        'case': current_case,
                        'body': case_body
                    })
                
                current_case = case_match.group(1)
                case_body = []
                
                # Check for inline return
                return_match = re.search(r'return\s+(.+?);', line)
                if return_match:
                    case_body.append({
                        'type': 'return',
                        'value': return_match.group(1)
                    })
            
            elif line.startswith('default'):
                # Save previous case
                if current_case is not None:
                    switch_stmt['cases'].append({
                        'case': current_case,
                        'body': case_body
                    })
                
                current_case = None
                case_body = []
                
                # Check for inline return
                return_match = re.search(r'return\s+(.+?);', line)
                if return_match:
                    switch_stmt['default'] = [{
                        'type': 'return',
                        'value': return_match.group(1)
                    }]
            
            elif current_case is not None:
                return_match = re.search(r'return\s+(.+?);', line)
                if return_match:
                    case_body.append({
                        'type': 'return',
                        'value': return_match.group(1)
                    })
            
            elif switch_stmt['default'] is None and 'return' in line:
                return_match = re.search(r'return\s+(.+?);', line)
                if return_match:
                    switch_stmt['default'] = [{
                        'type': 'return',
                        'value': return_match.group(1)
                    }]
        
        # Save last case
        if current_case is not None:
            switch_stmt['cases'].append({
                'case': current_case,
                'body': case_body
            })
        
        return switch_stmt if switch_stmt['cases'] or switch_stmt['default'] else None
    
    def _extract_statements_from_ast(self, node: 'ASTNode', statements: List[Dict[str, Any]]) -> None:
        """Extract statements from AST node."""
        if not CLANG_AVAILABLE or node is None:
            return
        
        if node.node_type == NodeType.IF:
            # Extract condition from source
            condition = ""
            if 'condition' in node.attributes:
                condition = node.attributes['condition']
            elif '{' in node.source_text:
                condition = node.source_text.split('{')[0].replace('if', '').strip()
                condition = condition.strip('(').strip(')')
            
            if_stmt = {
                'type': 'if',
                'condition': condition,
                'then': [],
                'else': [],
                'source': 'clang'
            }
            
            # Process children for then/else branches
            for child in node.children:
                if child.node_type == NodeType.RETURN:
                    match = re.search(r'return\s+(.+?);', child.source_text)
                    if match:
                        if_stmt['then'].append({
                            'type': 'return',
                            'value': match.group(1)
                        })
            
            statements.append(if_stmt)
        
        elif node.node_type == NodeType.FOR or node.node_type == NodeType.WHILE:
            loop_stmt = {
                'type': 'loop',
                'loop_type': node.node_type.value,
                'body': [],
                'source': 'clang'
            }
            statements.append(loop_stmt)
        
        elif node.node_type == NodeType.RETURN:
            match = re.search(r'return\s+(.+?);', node.source_text)
            if match:
                statements.append({
                    'type': 'return',
                    'value': match.group(1),
                    'text': node.source_text,
                    'source': 'clang'
                })
        
        # Recurse into children
        for child in node.children:
            self._extract_statements_from_ast(child, statements)
    
    def _parse_statements_regex(self, code: str) -> List[Dict[str, Any]]:
        """Parse code using regex-based parsing (fallback)."""
        statements = []
        lines = code.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('//'):
                i += 1
                continue
            
            # Handle switch statements
            if line.startswith('switch'):
                switch_stmt, consumed = self._parse_switch(lines, i)
                statements.append(switch_stmt)
                i += consumed
            
            # Handle if statements
            elif line.startswith('if'):
                if_stmt, consumed = self._parse_if(lines, i)
                statements.append(if_stmt)
                i += consumed
            
            # Handle for/while loops
            elif line.startswith(('for', 'while')):
                loop_stmt, consumed = self._parse_loop(lines, i)
                statements.append(loop_stmt)
                i += consumed
            
            # Handle return
            elif line.startswith('return'):
                statements.append({
                    'type': 'return',
                    'text': line,
                    'value': self._extract_return_value(line)
                })
                i += 1
            
            # Handle declarations/assignments
            elif '=' in line and not line.startswith('if'):
                statements.append({
                    'type': 'assignment',
                    'text': line
                })
                i += 1
            
            # Handle function calls
            elif '(' in line and ');' in line:
                statements.append({
                    'type': 'call',
                    'text': line
                })
                i += 1
            
            else:
                statements.append({
                    'type': 'other',
                    'text': line
                })
                i += 1
        
        return statements
    
    def _parse_switch(self, lines: List[str], start: int) -> Tuple[Dict, int]:
        """Parse a switch statement."""
        text = lines[start].strip()
        
        # Extract switch expression
        match = re.search(r'switch\s*\((.+?)\)', text)
        expr = match.group(1) if match else ""
        
        # Find matching braces
        depth = 0
        end = start
        content_lines = []
        
        for i in range(start, len(lines)):
            line = lines[i]
            depth += line.count('{') - line.count('}')
            
            if i > start:
                content_lines.append(line.strip())
            
            if depth == 0 and i > start:
                end = i
                break
        
        # Parse cases
        cases = []
        current_case = None
        case_body = []
        
        for line in content_lines:
            if line.startswith('case ') or line.startswith('default'):
                if current_case is not None:
                    cases.append({
                        'case': current_case,
                        'body': case_body
                    })
                
                match = re.match(r'case\s+(\w+(?:::\w+)?)\s*:', line)
                if match:
                    current_case = match.group(1)
                else:
                    current_case = 'default'
                case_body = []
                
                # Check for inline return
                return_match = re.search(r'return\s+(.+?);', line)
                if return_match:
                    case_body.append({
                        'type': 'return',
                        'value': return_match.group(1)
                    })
            
            elif current_case is not None:
                if line.startswith('return'):
                    return_match = re.search(r'return\s+(.+?);', line)
                    if return_match:
                        case_body.append({
                            'type': 'return',
                            'value': return_match.group(1)
                        })
                elif line == 'break;':
                    case_body.append({'type': 'break'})
                elif line and line != '}':
                    case_body.append({
                        'type': 'statement',
                        'text': line
                    })
        
        if current_case is not None:
            cases.append({
                'case': current_case,
                'body': case_body
            })
        
        return {
            'type': 'switch',
            'expression': expr,
            'cases': cases,
            'text': '\n'.join([lines[i] for i in range(start, end + 1)])
        }, end - start + 1
    
    def _parse_if(self, lines: List[str], start: int) -> Tuple[Dict, int]:
        """Parse an if statement."""
        text = lines[start].strip()
        
        # Extract condition
        match = re.search(r'if\s*\((.+?)\)', text)
        condition = match.group(1) if match else ""
        
        # Find then branch
        depth = 0
        in_then = True
        then_lines = []
        else_lines = []
        end = start
        
        for i in range(start, len(lines)):
            line = lines[i]
            depth += line.count('{') - line.count('}')
            
            if 'else' in line and depth == 1:
                in_then = False
                continue
            
            if i > start:
                if in_then:
                    then_lines.append(line.strip())
                else:
                    else_lines.append(line.strip())
            
            if depth == 0 and i > start:
                end = i
                break
        
        return {
            'type': 'if',
            'condition': condition,
            'then_branch': then_lines,
            'else_branch': else_lines,
            'text': '\n'.join([lines[i] for i in range(start, end + 1)])
        }, end - start + 1
    
    def _parse_loop(self, lines: List[str], start: int) -> Tuple[Dict, int]:
        """Parse a loop statement."""
        text = lines[start].strip()
        
        loop_type = 'for' if text.startswith('for') else 'while'
        
        # Extract condition
        match = re.search(r'\((.+)\)', text)
        header = match.group(1) if match else ""
        
        # Find body
        depth = 0
        body_lines = []
        end = start
        
        for i in range(start, len(lines)):
            line = lines[i]
            depth += line.count('{') - line.count('}')
            
            if i > start:
                body_lines.append(line.strip())
            
            if depth == 0 and i > start:
                end = i
                break
        
        return {
            'type': 'loop',
            'loop_type': loop_type,
            'header': header,
            'body': body_lines,
            'text': '\n'.join([lines[i] for i in range(start, end + 1)])
        }, end - start + 1
    
    def _extract_return_value(self, stmt: str) -> str:
        """Extract return value expression."""
        match = re.search(r'return\s+(.+?)\s*;', stmt)
        return match.group(1) if match else ""
    
    def _execute_statements(
        self,
        statements: List[Dict],
        state: SymbolicState,
        depth: int
    ) -> None:
        """Execute a list of statements symbolically."""
        if depth >= self.max_depth or len(self.paths) >= self.max_paths:
            self._record_path(state, "max_depth_or_paths")
            return
        
        if state.is_terminated:
            self._record_path(state, state.termination_reason)
            return
        
        for stmt in statements:
            if state.is_terminated:
                break
            
            stmt_type = stmt.get('type', 'other')
            
            if stmt_type == 'switch':
                self._execute_switch(stmt, state, depth)
                return  # Switch creates multiple paths
            
            elif stmt_type == 'if':
                self._execute_if(stmt, state, depth)
                return  # If creates multiple paths
            
            elif stmt_type == 'return':
                self._execute_return(stmt, state)
            
            elif stmt_type == 'assignment':
                self._execute_assignment(stmt, state)
            
            elif stmt_type == 'loop':
                self._execute_loop(stmt, state, depth)
            
            elif stmt_type == 'call':
                self._execute_call(stmt, state)
        
        # End of statements reached
        if not state.is_terminated:
            self._record_path(state, "end_of_function")
    
    def _execute_switch(
        self,
        stmt: Dict,
        state: SymbolicState,
        depth: int
    ) -> None:
        """Execute switch statement, forking for each case."""
        expr = stmt.get('expression', '')
        cases = stmt.get('cases', [])
        
        for case in cases:
            case_value = case.get('case', '')
            case_body = case.get('body', [])
            
            # Fork state for this case
            case_state = state.copy()
            
            # Add path constraint
            if case_value != 'default':
                constraint = f"(= {expr} {case_value})"
                case_state.add_path_constraint(constraint)
            else:
                # Default case: negate all other cases
                other_cases = [c['case'] for c in cases if c['case'] != 'default']
                if other_cases:
                    negations = [f"(not (= {expr} {c}))" for c in other_cases]
                    constraint = f"(and {' '.join(negations)})" if len(negations) > 1 else negations[0]
                    case_state.add_path_constraint(constraint)
            
            # Execute case body
            for body_stmt in case_body:
                if body_stmt.get('type') == 'return':
                    return_val = body_stmt.get('value', '')
                    sym_val = self._evaluate_expression(return_val, case_state)
                    case_state.terminate('return', sym_val)
                    break
                elif body_stmt.get('type') == 'break':
                    break
            
            if not case_state.is_terminated:
                case_state.terminate('fallthrough')
            
            self._record_path(case_state, f"switch_case_{case_value}")
    
    def _execute_if(
        self,
        stmt: Dict,
        state: SymbolicState,
        depth: int
    ) -> None:
        """Execute if statement, forking for both branches."""
        condition = stmt.get('condition', '')
        then_branch = stmt.get('then_branch', [])
        else_branch = stmt.get('else_branch', [])
        
        # Convert condition to SMT
        smt_cond = self._condition_to_smt(condition, state)
        
        # Then branch
        then_state = state.copy()
        then_state.add_path_constraint(smt_cond)
        then_stmts = self._parse_statements('\n'.join(then_branch))
        self._execute_statements(then_stmts, then_state, depth + 1)
        
        # Else branch
        else_state = state.copy()
        else_state.add_path_constraint(f"(not {smt_cond})")
        if else_branch:
            else_stmts = self._parse_statements('\n'.join(else_branch))
            self._execute_statements(else_stmts, else_state, depth + 1)
        else:
            self._record_path(else_state, "if_else_implicit")
    
    def _execute_return(self, stmt: Dict, state: SymbolicState) -> None:
        """Execute return statement."""
        return_expr = stmt.get('value', '')
        sym_val = self._evaluate_expression(return_expr, state)
        state.terminate('return', sym_val)
    
    def _execute_assignment(self, stmt: Dict, state: SymbolicState) -> None:
        """Execute assignment statement."""
        text = stmt.get('text', '')
        
        # Parse assignment: type? var = expr;
        match = re.match(r'(?:(\w+)\s+)?(\w+)\s*=\s*(.+?)\s*;', text)
        if match:
            var_type = match.group(1)
            var_name = match.group(2)
            expr = match.group(3)
            
            sym_val = self._evaluate_expression(expr, state)
            state.set_variable(var_name, sym_val)
    
    def _execute_loop(
        self,
        stmt: Dict,
        state: SymbolicState,
        depth: int
    ) -> None:
        """Execute loop with bounded unrolling."""
        # For symbolic execution, we unroll loops a bounded number of times
        # or abstract them with invariants
        
        body = stmt.get('body', [])
        max_unroll = 3
        
        for i in range(max_unroll):
            body_stmts = self._parse_statements('\n'.join(body))
            
            for body_stmt in body_stmts:
                if body_stmt.get('type') == 'return':
                    self._execute_return(body_stmt, state)
                    return
                elif body_stmt.get('type') == 'assignment':
                    self._execute_assignment(body_stmt, state)
        
        # After bounded unrolling, record path
        state.add_path_constraint("(loop_bound_reached)")
    
    def _execute_call(self, stmt: Dict, state: SymbolicState) -> None:
        """Execute function call."""
        text = stmt.get('text', '')
        
        # Check for known function models
        for func_name, model in self.function_models.items():
            if func_name + '(' in text:
                model(text, state)
                return
    
    def _evaluate_expression(
        self,
        expr: str,
        state: SymbolicState
    ) -> SymbolicValue:
        """Evaluate an expression in the current state."""
        expr = expr.strip()
        
        # Check if it's a known variable
        if expr in state.variables:
            return state.variables[expr]
        
        # Check if it's a constant
        if expr.isdigit():
            return SymbolicValue.concrete(f"const_{expr}", int(expr))
        
        # Check for qualified constant (e.g., ELF::R_RISCV_NONE)
        if '::' in expr:
            return SymbolicValue.concrete(expr, expr)
        
        # Check for function call
        match = re.match(r'(\w+)\s*\((.+)\)', expr)
        if match:
            func_name = match.group(1)
            args = match.group(2)
            
            # Create symbolic value depending on function call
            depends_on = set()
            for var in state.variables:
                if var in args:
                    depends_on.add(var)
            
            return SymbolicValue.symbolic(
                f"{func_name}({args})",
                depends_on
            )
        
        # Otherwise, create symbolic value
        depends_on = set()
        for var in state.variables:
            if var in expr:
                depends_on.add(var)
        
        return SymbolicValue.symbolic(expr, depends_on)
    
    def _condition_to_smt(self, condition: str, state: SymbolicState) -> str:
        """Convert C++ condition to SMT-LIB format."""
        # Replace operators
        smt = condition
        smt = re.sub(r'\s*==\s*', ' ', smt)
        smt = re.sub(r'\s*!=\s*', ' ', smt)
        smt = re.sub(r'\s*&&\s*', ' ', smt)
        smt = re.sub(r'\s*\|\|\s*', ' ', smt)
        
        # Handle comparisons
        if '==' in condition:
            parts = condition.split('==')
            if len(parts) == 2:
                return f"(= {parts[0].strip()} {parts[1].strip()})"
        
        if '!=' in condition:
            parts = condition.split('!=')
            if len(parts) == 2:
                return f"(not (= {parts[0].strip()} {parts[1].strip()}))"
        
        if '<' in condition:
            parts = re.split(r'<(?!=)', condition)
            if len(parts) == 2:
                return f"(< {parts[0].strip()} {parts[1].strip()})"
        
        if '>' in condition:
            parts = re.split(r'>(?!=)', condition)
            if len(parts) == 2:
                return f"(> {parts[0].strip()} {parts[1].strip()})"
        
        # Default: return as-is
        return condition
    
    def _record_path(self, state: SymbolicState, reason: str) -> None:
        """Record a completed execution path, checking satisfiability with Z3."""
        # Check if path is satisfiable before recording
        is_sat, model = state.is_satisfiable()
        
        if not is_sat:
            if self.verbose:
                print(f"Path pruned (UNSAT): {reason}")
            return  # Don't record infeasible paths
        
        self.path_counter += 1
        path = ExecutionPath(
            path_id=self.path_counter,
            final_state=state.copy()
        )
        
        # Store model assignments if available
        if model:
            path.final_state.variables['_z3_model'] = SymbolicValue.concrete('_z3_model', model)
        
        self.paths.append(path)
        
        if self.verbose:
            print(f"Path {self.path_counter}: {reason} (SAT)")
            if state.return_value:
                print(f"  Return: {state.return_value}")
            print(f"  Condition: {state.path_condition_smt()}")
            if model:
                print(f"  Model: {model}")
    
    # Function models for common LLVM operations
    def _model_getTargetKind(self, text: str, state: SymbolicState) -> None:
        """Model for MCFixup::getTargetKind()."""
        match = re.search(r'(\w+)\.getTargetKind\(\)', text)
        if match:
            var_name = match.group(1)
            result_var = f"{var_name}_kind"
            state.set_variable(
                result_var,
                SymbolicValue.symbolic(result_var, {var_name})
            )
    
    def _model_getOperand(self, text: str, state: SymbolicState) -> None:
        """Model for MCInst::getOperand()."""
        match = re.search(r'(\w+)->?getOperand\((\d+)\)', text)
        if match:
            inst = match.group(1)
            idx = match.group(2)
            result_var = f"{inst}_operand_{idx}"
            state.set_variable(
                result_var,
                SymbolicValue.symbolic(result_var, {inst})
            )
    
    def _model_getImm(self, text: str, state: SymbolicState) -> None:
        """Model for MCOperand::getImm()."""
        match = re.search(r'(\w+)\.getImm\(\)', text)
        if match:
            op = match.group(1)
            result_var = f"{op}_imm"
            state.set_variable(
                result_var,
                SymbolicValue.symbolic(result_var, {op})
            )
    
    def _model_getReg(self, text: str, state: SymbolicState) -> None:
        """Model for MCOperand::getReg()."""
        match = re.search(r'(\w+)\.getReg\(\)', text)
        if match:
            op = match.group(1)
            result_var = f"{op}_reg"
            state.set_variable(
                result_var,
                SymbolicValue.symbolic(result_var, {op})
            )
    
    def _model_isReg(self, text: str, state: SymbolicState) -> None:
        """Model for MCOperand::isReg()."""
        pass  # Handled as boolean condition
    
    def _model_isImm(self, text: str, state: SymbolicState) -> None:
        """Model for MCOperand::isImm()."""
        pass  # Handled as boolean condition
    
    def _model_isExpr(self, text: str, state: SymbolicState) -> None:
        """Model for MCOperand::isExpr()."""
        pass  # Handled as boolean condition


def extract_path_conditions(
    code: str,
    function_name: str,
    parameters: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """
    Convenience function to extract path conditions from code.
    
    Returns:
        Dictionary with paths and extracted conditions
    """
    executor = SymbolicExecutor(verbose=False)
    paths = executor.execute(code, function_name, parameters)
    
    result = {
        "function": function_name,
        "parameters": parameters,
        "total_paths": len(paths),
        "paths": []
    }
    
    for path in paths:
        result["paths"].append({
            "id": path.path_id,
            "condition": path.path_condition,
            "return_value": str(path.return_value) if path.return_value else None,
        })
    
    return result
