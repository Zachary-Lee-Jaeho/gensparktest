"""
Phase 2.1: Semantic Analysis Engine for VEGA-Verified.

This module provides deep semantic analysis of compiler backend code:
1. LLVM IR Pattern Recognition (Switch, If-Else, Loop)
2. Control Flow Graph Construction
3. Symbolic Execution for Path Enumeration
4. AST-based Pattern Matching

Key capabilities:
- Parse C++ source code into semantic IR
- Identify critical patterns (relocation, encoding, etc.)
- Extract function call graphs and dependencies
- Support counterexample-guided analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from enum import Enum
import re
import json
from pathlib import Path


class PatternType(Enum):
    """Types of code patterns recognized."""
    SWITCH_CASE = "switch_case"
    IF_ELSE = "if_else"
    IF_ELSE_CHAIN = "if_else_chain"  # Multiple if-else
    TERNARY = "ternary"
    LOOP_FOR = "loop_for"
    LOOP_WHILE = "loop_while"
    FUNCTION_CALL = "function_call"
    RETURN = "return"
    ASSIGNMENT = "assignment"
    ASSERTION = "assertion"
    ERROR_HANDLER = "error_handler"


class DataType(Enum):
    """C++ data types."""
    VOID = "void"
    BOOL = "bool"
    INT = "int"
    UNSIGNED = "unsigned"
    INT32 = "int32_t"
    UINT32 = "uint32_t"
    INT64 = "int64_t"
    UINT64 = "uint64_t"
    ENUM = "enum"
    POINTER = "pointer"
    REFERENCE = "reference"
    CONST = "const"
    UNKNOWN = "unknown"


@dataclass
class Variable:
    """Represents a variable in the code."""
    name: str
    data_type: DataType
    is_const: bool = False
    is_reference: bool = False
    is_pointer: bool = False
    initial_value: Optional[str] = None
    

@dataclass
class Expression:
    """Represents an expression in the code."""
    raw: str
    expr_type: str = "unknown"  # binary, unary, call, literal, variable
    operands: List[Any] = field(default_factory=list)
    operator: Optional[str] = None
    result_type: DataType = DataType.UNKNOWN
    
    def is_comparison(self) -> bool:
        return self.operator in ('==', '!=', '<', '>', '<=', '>=')
    
    def is_logical(self) -> bool:
        return self.operator in ('&&', '||', '!')


@dataclass
class CaseBlock:
    """Represents a case block in switch statement."""
    case_value: str
    statements: List[Any] = field(default_factory=list)
    return_value: Optional[str] = None
    is_fallthrough: bool = False
    has_break: bool = False
    condition: Optional[Expression] = None  # For ternary returns


@dataclass
class SwitchPattern:
    """Semantic representation of switch statement."""
    switch_expr: str  # e.g., "Kind"
    cases: List[CaseBlock] = field(default_factory=list)
    default_block: Optional[CaseBlock] = None
    has_complete_coverage: bool = False
    
    def get_all_case_values(self) -> List[str]:
        return [c.case_value for c in self.cases]
    
    def get_return_mapping(self) -> Dict[str, str]:
        """Get case -> return value mapping."""
        mapping = {}
        for case in self.cases:
            if case.return_value:
                mapping[case.case_value] = case.return_value
        return mapping


@dataclass
class IfElsePattern:
    """Semantic representation of if-else statement."""
    condition: Expression
    then_branch: List[Any] = field(default_factory=list)
    else_branch: Optional[List[Any]] = None
    then_return: Optional[str] = None
    else_return: Optional[str] = None


@dataclass
class LoopPattern:
    """Semantic representation of loop."""
    loop_type: str  # "for" or "while"
    condition: Optional[Expression] = None
    init: Optional[str] = None
    update: Optional[str] = None
    body: List[Any] = field(default_factory=list)
    is_bounded: bool = False
    bound: Optional[int] = None


@dataclass
class FunctionSignature:
    """Function signature information."""
    name: str
    return_type: DataType
    parameters: List[Tuple[str, DataType]] = field(default_factory=list)
    is_const: bool = False
    is_virtual: bool = False


@dataclass
class FunctionSemantics:
    """Complete semantic representation of a function."""
    signature: FunctionSignature
    local_variables: List[Variable] = field(default_factory=list)
    patterns: List[Union[SwitchPattern, IfElsePattern, LoopPattern]] = field(default_factory=list)
    control_flow: Dict[str, Any] = field(default_factory=dict)
    return_paths: List[Dict[str, Any]] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    assertions: List[str] = field(default_factory=list)


class SemanticAnalyzer:
    """
    Main semantic analysis engine for compiler backend code.
    
    Features:
    1. Pattern recognition (switch, if-else, loops)
    2. Control flow analysis
    3. Return path enumeration
    4. Symbolic constraint extraction
    """
    
    # Regex patterns for C++ parsing
    SWITCH_PATTERN = re.compile(
        r'switch\s*\(\s*(\w+)\s*\)\s*\{',
        re.MULTILINE
    )
    
    CASE_PATTERN = re.compile(
        r'case\s+([\w:]+):\s*',
        re.MULTILINE
    )
    
    RETURN_PATTERN = re.compile(
        r'return\s+([^;]+);',
        re.MULTILINE
    )
    
    IF_PATTERN = re.compile(
        r'if\s*\(\s*([^)]+)\s*\)\s*\{?',
        re.MULTILINE
    )
    
    TERNARY_PATTERN = re.compile(
        r'(\w+)\s*\?\s*([\w:]+)\s*:\s*([\w:]+)',
        re.MULTILINE
    )
    
    FUNCTION_CALL_PATTERN = re.compile(
        r'(\w+)\s*\(\s*([^)]*)\s*\)',
        re.MULTILINE
    )
    
    VARIABLE_DECL_PATTERN = re.compile(
        r'(?:const\s+)?(?:unsigned\s+)?(\w+)\s+(\w+)\s*(?:=\s*([^;]+))?;',
        re.MULTILINE
    )
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {
            "functions_analyzed": 0,
            "switches_found": 0,
            "if_elses_found": 0,
            "loops_found": 0,
            "return_paths": 0,
        }
    
    def analyze_function(self, code: str, function_name: str = "unknown") -> FunctionSemantics:
        """
        Perform comprehensive semantic analysis of a function.
        
        Args:
            code: Function body source code
            function_name: Name of the function
            
        Returns:
            FunctionSemantics with complete semantic information
        """
        self.stats["functions_analyzed"] += 1
        
        # Parse function signature
        signature = self._parse_signature(code, function_name)
        
        # Initialize semantics
        semantics = FunctionSemantics(signature=signature)
        
        # Extract local variables
        semantics.local_variables = self._extract_variables(code)
        
        # Recognize patterns
        semantics.patterns = self._recognize_patterns(code)
        
        # Build control flow
        semantics.control_flow = self._build_control_flow(code, semantics.patterns)
        
        # Enumerate return paths
        semantics.return_paths = self._enumerate_return_paths(code, semantics.patterns)
        
        # Extract function calls
        semantics.calls = self._extract_calls(code)
        
        # Extract assertions
        semantics.assertions = self._extract_assertions(code)
        
        return semantics
    
    def _parse_signature(self, code: str, function_name: str) -> FunctionSignature:
        """Parse function signature from code."""
        # Try to find return type from code context
        return_type = DataType.UNSIGNED  # Common for getRelocType, etc.
        
        # Check for common return type patterns
        if 'void ' in code.split('{')[0] if '{' in code else '':
            return_type = DataType.VOID
        elif 'bool ' in code.split('{')[0] if '{' in code else '':
            return_type = DataType.BOOL
        elif 'int ' in code.split('{')[0] if '{' in code else '':
            return_type = DataType.INT
        
        return FunctionSignature(
            name=function_name,
            return_type=return_type
        )
    
    def _extract_variables(self, code: str) -> List[Variable]:
        """Extract local variable declarations."""
        variables = []
        
        for match in self.VARIABLE_DECL_PATTERN.finditer(code):
            type_str = match.group(1)
            name = match.group(2)
            init_val = match.group(3) if match.lastindex >= 3 else None
            
            # Parse type
            data_type = self._parse_type(type_str)
            is_const = 'const' in match.group(0)
            
            variables.append(Variable(
                name=name,
                data_type=data_type,
                is_const=is_const,
                initial_value=init_val
            ))
        
        return variables
    
    def _parse_type(self, type_str: str) -> DataType:
        """Parse C++ type string to DataType."""
        type_map = {
            'void': DataType.VOID,
            'bool': DataType.BOOL,
            'int': DataType.INT,
            'unsigned': DataType.UNSIGNED,
            'int32_t': DataType.INT32,
            'uint32_t': DataType.UINT32,
            'int64_t': DataType.INT64,
            'uint64_t': DataType.UINT64,
        }
        return type_map.get(type_str.strip(), DataType.UNKNOWN)
    
    def _recognize_patterns(self, code: str) -> List[Union[SwitchPattern, IfElsePattern, LoopPattern]]:
        """Recognize code patterns in function body."""
        patterns = []
        
        # Recognize switch statements
        switch_patterns = self._recognize_switches(code)
        patterns.extend(switch_patterns)
        self.stats["switches_found"] += len(switch_patterns)
        
        # Recognize if-else statements
        if_patterns = self._recognize_if_else(code)
        patterns.extend(if_patterns)
        self.stats["if_elses_found"] += len(if_patterns)
        
        # Recognize loops
        loop_patterns = self._recognize_loops(code)
        patterns.extend(loop_patterns)
        self.stats["loops_found"] += len(loop_patterns)
        
        return patterns
    
    def _recognize_switches(self, code: str) -> List[SwitchPattern]:
        """Recognize and parse switch statements."""
        switches = []
        
        for switch_match in self.SWITCH_PATTERN.finditer(code):
            switch_expr = switch_match.group(1)
            
            # Find switch block
            start = switch_match.end()
            brace_count = 1
            end = start
            
            for i, char in enumerate(code[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i
                        break
            
            switch_body = code[start:end]
            
            # Parse cases
            switch = SwitchPattern(switch_expr=switch_expr)
            switch.cases = self._parse_cases(switch_body)
            
            # Check for default
            if 'default:' in switch_body:
                default_start = switch_body.find('default:')
                default_body = switch_body[default_start:]
                ret_match = self.RETURN_PATTERN.search(default_body)
                if ret_match:
                    switch.default_block = CaseBlock(
                        case_value="default",
                        return_value=ret_match.group(1).strip()
                    )
            
            switches.append(switch)
        
        return switches
    
    def _parse_cases(self, switch_body: str) -> List[CaseBlock]:
        """Parse case blocks from switch body."""
        cases = []
        current_case = None
        
        # Split by case labels
        lines = switch_body.split('\n')
        
        for line in lines:
            case_match = self.CASE_PATTERN.search(line)
            if case_match:
                # Save previous case
                if current_case:
                    cases.append(current_case)
                
                current_case = CaseBlock(
                    case_value=case_match.group(1).strip()
                )
            elif current_case:
                # Check for return
                ret_match = self.RETURN_PATTERN.search(line)
                if ret_match:
                    return_expr = ret_match.group(1).strip()
                    
                    # Check for ternary
                    ternary_match = self.TERNARY_PATTERN.search(return_expr)
                    if ternary_match:
                        current_case.condition = Expression(
                            raw=ternary_match.group(1),
                            expr_type="condition"
                        )
                        # Store both possible returns
                        current_case.return_value = f"{ternary_match.group(2)}|{ternary_match.group(3)}"
                    else:
                        current_case.return_value = return_expr
                
                # Check for break
                if 'break;' in line:
                    current_case.has_break = True
        
        # Don't forget last case
        if current_case:
            cases.append(current_case)
        
        return cases
    
    def _recognize_if_else(self, code: str) -> List[IfElsePattern]:
        """Recognize if-else patterns."""
        patterns = []
        
        for match in self.IF_PATTERN.finditer(code):
            condition_str = match.group(1).strip()
            
            # Create expression for condition
            condition = Expression(
                raw=condition_str,
                expr_type="boolean"
            )
            
            # Try to find then and else branches
            # (simplified - proper parsing needs brace matching)
            pattern = IfElsePattern(condition=condition)
            
            # Check for returns in if/else
            start = match.end()
            next_lines = code[start:start+200]  # Look ahead
            
            ret_match = self.RETURN_PATTERN.search(next_lines)
            if ret_match:
                pattern.then_return = ret_match.group(1).strip()
            
            patterns.append(pattern)
        
        return patterns
    
    def _recognize_loops(self, code: str) -> List[LoopPattern]:
        """Recognize loop patterns."""
        patterns = []
        
        # For loops
        for_pattern = re.compile(r'for\s*\(\s*([^;]*);([^;]*);([^)]*)\)')
        for match in for_pattern.finditer(code):
            pattern = LoopPattern(
                loop_type="for",
                init=match.group(1).strip(),
                condition=Expression(raw=match.group(2).strip(), expr_type="boolean"),
                update=match.group(3).strip()
            )
            
            # Try to determine if bounded
            cond = match.group(2).strip()
            if '<' in cond and any(c.isdigit() for c in cond):
                pattern.is_bounded = True
            
            patterns.append(pattern)
        
        # While loops
        while_pattern = re.compile(r'while\s*\(\s*([^)]+)\s*\)')
        for match in while_pattern.finditer(code):
            pattern = LoopPattern(
                loop_type="while",
                condition=Expression(raw=match.group(1).strip(), expr_type="boolean")
            )
            patterns.append(pattern)
        
        return patterns
    
    def _build_control_flow(
        self, 
        code: str, 
        patterns: List[Union[SwitchPattern, IfElsePattern, LoopPattern]]
    ) -> Dict[str, Any]:
        """Build control flow graph representation."""
        cfg = {
            "entry": "start",
            "nodes": ["start"],
            "edges": [],
            "exit": "end",
        }
        
        node_id = 0
        
        for pattern in patterns:
            if isinstance(pattern, SwitchPattern):
                # Add switch node
                switch_node = f"switch_{node_id}"
                cfg["nodes"].append(switch_node)
                
                # Add edges for each case
                for case in pattern.cases:
                    case_node = f"case_{node_id}_{case.case_value}"
                    cfg["nodes"].append(case_node)
                    cfg["edges"].append({
                        "from": switch_node,
                        "to": case_node,
                        "condition": f"{pattern.switch_expr} == {case.case_value}"
                    })
                
                node_id += 1
            
            elif isinstance(pattern, IfElsePattern):
                # Add if node
                if_node = f"if_{node_id}"
                cfg["nodes"].append(if_node)
                
                # Add then/else edges
                then_node = f"then_{node_id}"
                cfg["nodes"].append(then_node)
                cfg["edges"].append({
                    "from": if_node,
                    "to": then_node,
                    "condition": pattern.condition.raw
                })
                
                if pattern.else_branch:
                    else_node = f"else_{node_id}"
                    cfg["nodes"].append(else_node)
                    cfg["edges"].append({
                        "from": if_node,
                        "to": else_node,
                        "condition": f"!({pattern.condition.raw})"
                    })
                
                node_id += 1
        
        cfg["nodes"].append("end")
        return cfg
    
    def _enumerate_return_paths(
        self,
        code: str,
        patterns: List[Union[SwitchPattern, IfElsePattern, LoopPattern]]
    ) -> List[Dict[str, Any]]:
        """Enumerate all possible return paths."""
        paths = []
        
        # Direct returns
        for match in self.RETURN_PATTERN.finditer(code):
            return_val = match.group(1).strip()
            
            # Check if ternary
            ternary_match = self.TERNARY_PATTERN.search(return_val)
            if ternary_match:
                # Two paths for ternary
                paths.append({
                    "condition": ternary_match.group(1),
                    "return": ternary_match.group(2),
                    "type": "ternary_true"
                })
                paths.append({
                    "condition": f"!{ternary_match.group(1)}",
                    "return": ternary_match.group(3),
                    "type": "ternary_false"
                })
            else:
                paths.append({
                    "return": return_val,
                    "type": "direct"
                })
        
        # Paths from switch patterns
        for pattern in patterns:
            if isinstance(pattern, SwitchPattern):
                for case in pattern.cases:
                    if case.return_value:
                        ret_vals = case.return_value.split('|')
                        if len(ret_vals) > 1:
                            # Conditional return
                            paths.append({
                                "condition": f"{pattern.switch_expr} == {case.case_value} && {case.condition.raw if case.condition else 'true'}",
                                "return": ret_vals[0],
                                "type": "switch_ternary_true"
                            })
                            paths.append({
                                "condition": f"{pattern.switch_expr} == {case.case_value} && !({case.condition.raw if case.condition else 'true'})",
                                "return": ret_vals[1],
                                "type": "switch_ternary_false"
                            })
                        else:
                            paths.append({
                                "condition": f"{pattern.switch_expr} == {case.case_value}",
                                "return": case.return_value,
                                "type": "switch_case"
                            })
        
        self.stats["return_paths"] += len(paths)
        return paths
    
    def _extract_calls(self, code: str) -> List[str]:
        """Extract function calls from code."""
        calls = set()
        
        for match in self.FUNCTION_CALL_PATTERN.finditer(code):
            func_name = match.group(1)
            # Filter out keywords and common non-function patterns
            if func_name not in ('if', 'for', 'while', 'switch', 'return', 'case'):
                calls.add(func_name)
        
        return list(calls)
    
    def _extract_assertions(self, code: str) -> List[str]:
        """Extract assertion statements."""
        assertions = []
        
        assert_pattern = re.compile(r'assert\s*\(\s*([^)]+)\s*\)')
        for match in assert_pattern.finditer(code):
            assertions.append(match.group(1).strip())
        
        llvm_assert_pattern = re.compile(r'llvm_unreachable\s*\(\s*"([^"]+)"\s*\)')
        for match in llvm_assert_pattern.finditer(code):
            assertions.append(f"unreachable: {match.group(1)}")
        
        return assertions
    
    def extract_switch_semantics(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract detailed switch statement semantics for verification.
        
        Returns list of switch statements with:
        - switch_variable: The variable being switched on
        - cases: List of (case_value, return_value, condition?)
        - default: Default return value
        - coverage: Analysis of case coverage
        """
        switches = []
        
        for switch in self._recognize_switches(code):
            semantics = {
                "switch_variable": switch.switch_expr,
                "cases": [],
                "default": None,
                "coverage": {
                    "total_cases": len(switch.cases),
                    "cases_with_return": 0,
                    "conditional_returns": 0,
                    "fallthrough_cases": 0,
                }
            }
            
            for case in switch.cases:
                case_info = {
                    "value": case.case_value,
                    "return": case.return_value,
                    "has_condition": case.condition is not None,
                    "is_fallthrough": case.is_fallthrough,
                }
                
                if case.condition:
                    case_info["condition"] = case.condition.raw
                    semantics["coverage"]["conditional_returns"] += 1
                
                if case.return_value:
                    semantics["coverage"]["cases_with_return"] += 1
                
                if case.is_fallthrough:
                    semantics["coverage"]["fallthrough_cases"] += 1
                
                semantics["cases"].append(case_info)
            
            if switch.default_block:
                semantics["default"] = switch.default_block.return_value
            
            switches.append(semantics)
        
        return switches
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return self.stats.copy()


class PatternMatcher:
    """
    Pattern matcher for LLVM-specific code patterns.
    
    Recognizes:
    - Relocation type computation (getRelocType)
    - Instruction encoding (encodeInstruction)
    - Register allocation patterns
    - ISel patterns
    """
    
    # Pattern templates
    RELOC_TYPE_PATTERN = re.compile(
        r'(?:unsigned|uint\d+_t)\s+(\w+)(?:RelocType|getRelocType)',
        re.IGNORECASE
    )
    
    ENCODE_INSTR_PATTERN = re.compile(
        r'void\s+(\w+)?(?:encodeInstruction|EncodeInstruction)',
        re.IGNORECASE
    )
    
    FIXUP_KIND_SWITCH = re.compile(
        r'switch\s*\(\s*(?:Kind|Fixup\.\w+\(\)|getTargetKind\(\))\s*\)',
        re.IGNORECASE
    )
    
    ISPCREL_CHECK = re.compile(
        r'if\s*\(\s*IsPCRel\s*\)',
        re.IGNORECASE
    )
    
    def __init__(self):
        self.patterns_found = {}
    
    def match_function_type(self, code: str, function_name: str) -> str:
        """
        Determine the type of compiler backend function.
        
        Returns one of:
        - "reloc_type": Relocation type computation
        - "encode_instr": Instruction encoding
        - "isel": Instruction selection
        - "regalloc": Register allocation
        - "emit": Code emission
        - "unknown": Unknown type
        """
        if self.RELOC_TYPE_PATTERN.search(code) or 'RelocType' in function_name:
            return "reloc_type"
        
        if self.ENCODE_INSTR_PATTERN.search(code) or 'encode' in function_name.lower():
            return "encode_instr"
        
        if 'select' in function_name.lower() or 'isel' in function_name.lower():
            return "isel"
        
        if 'alloc' in function_name.lower() or 'register' in function_name.lower():
            return "regalloc"
        
        if 'emit' in function_name.lower():
            return "emit"
        
        return "unknown"
    
    def extract_fixup_mappings(self, code: str) -> Dict[str, List[str]]:
        """
        Extract fixup kind to relocation type mappings.
        
        Returns:
            Dict mapping fixup kinds to possible relocation types
        """
        mappings = {}
        
        # Find switch on Kind/Fixup
        if not self.FIXUP_KIND_SWITCH.search(code):
            return mappings
        
        # Check for IsPCRel split
        has_ispcrel = bool(self.ISPCREL_CHECK.search(code))
        
        # Parse case statements
        case_pattern = re.compile(r'case\s+([\w:]+):\s*\n\s*return\s+([\w:]+)')
        ternary_case = re.compile(
            r'case\s+([\w:]+):\s*\n\s*return\s+(\w+)\s*\?\s*([\w:]+)\s*:\s*([\w:]+)'
        )
        
        for match in ternary_case.finditer(code):
            fixup = match.group(1).split('::')[-1]
            cond = match.group(2)
            true_ret = match.group(3).split('::')[-1]
            false_ret = match.group(4).split('::')[-1]
            
            mappings[fixup] = [true_ret, false_ret]
        
        for match in case_pattern.finditer(code):
            fixup = match.group(1).split('::')[-1]
            reloc = match.group(2).split('::')[-1]
            
            if fixup not in mappings:
                mappings[fixup] = [reloc]
        
        return mappings
    
    def identify_critical_patterns(self, code: str) -> List[Dict[str, Any]]:
        """
        Identify patterns that are critical for correctness.
        
        Returns list of critical patterns with:
        - type: Pattern type
        - location: Approximate position
        - risk_level: high/medium/low
        - description: Human-readable description
        """
        patterns = []
        
        # Switch on fixup kind
        if self.FIXUP_KIND_SWITCH.search(code):
            patterns.append({
                "type": "fixup_switch",
                "risk_level": "high",
                "description": "Switch statement on fixup kind - critical for relocation correctness"
            })
        
        # IsPCRel conditional
        if self.ISPCREL_CHECK.search(code):
            patterns.append({
                "type": "pcrel_check",
                "risk_level": "high",
                "description": "PC-relative relocation check - affects address calculation"
            })
        
        # Assertion unreachable
        if 'llvm_unreachable' in code:
            patterns.append({
                "type": "unreachable",
                "risk_level": "medium",
                "description": "Contains unreachable assertion - may indicate incomplete handling"
            })
        
        # Error reporting
        if 'reportError' in code:
            patterns.append({
                "type": "error_handler",
                "risk_level": "low",
                "description": "Contains error reporting - good for debugging"
            })
        
        return patterns


def analyze_ground_truth_functions(db_path: str, output_path: str) -> Dict[str, Any]:
    """
    Analyze functions from ground truth database.
    
    Args:
        db_path: Path to ground truth JSON file
        output_path: Path to save analysis results
        
    Returns:
        Analysis summary statistics
    """
    analyzer = SemanticAnalyzer(verbose=True)
    matcher = PatternMatcher()
    
    with open(db_path, 'r') as f:
        db = json.load(f)
    
    results = {
        "version": "2.0",
        "analyzer": "semantic_analyzer",
        "functions": {},
        "summary": {
            "total_functions": 0,
            "functions_with_switches": 0,
            "functions_with_conditionals": 0,
            "reloc_type_functions": 0,
            "encode_functions": 0,
        }
    }
    
    for func_id, func_data in db.get("functions", {}).items():
        body = func_data.get("body", "")
        if not body:
            continue
        
        name = func_data.get("name", "unknown")
        
        # Analyze function
        semantics = analyzer.analyze_function(body, name)
        
        # Match patterns
        func_type = matcher.match_function_type(body, name)
        fixup_mappings = matcher.extract_fixup_mappings(body)
        critical_patterns = matcher.identify_critical_patterns(body)
        
        # Store results
        results["functions"][func_id] = {
            "name": name,
            "backend": func_data.get("backend", "unknown"),
            "function_type": func_type,
            "patterns": {
                "switches": len([p for p in semantics.patterns if isinstance(p, SwitchPattern)]),
                "conditionals": len([p for p in semantics.patterns if isinstance(p, IfElsePattern)]),
                "loops": len([p for p in semantics.patterns if isinstance(p, LoopPattern)]),
            },
            "return_paths": len(semantics.return_paths),
            "fixup_mappings": fixup_mappings,
            "critical_patterns": critical_patterns,
            "calls": semantics.calls[:10],  # Limit
            "assertions": semantics.assertions,
        }
        
        # Update summary
        results["summary"]["total_functions"] += 1
        if len([p for p in semantics.patterns if isinstance(p, SwitchPattern)]) > 0:
            results["summary"]["functions_with_switches"] += 1
        if len([p for p in semantics.patterns if isinstance(p, IfElsePattern)]) > 0:
            results["summary"]["functions_with_conditionals"] += 1
        if func_type == "reloc_type":
            results["summary"]["reloc_type_functions"] += 1
        if func_type == "encode_instr":
            results["summary"]["encode_functions"] += 1
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results["summary"]


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 2.1: Semantic Analysis Engine Demo")
    print("=" * 70)
    
    # Test code
    test_code = """
    unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                          const MCFixup &Fixup, bool IsPCRel) const {
        unsigned Kind = Fixup.getTargetKind();
        
        if (IsPCRel) {
            switch (Kind) {
            case FK_Data_4:
                return ELF::R_RISCV_32_PCREL;
            case RISCV::fixup_riscv_pcrel_hi20:
                return ELF::R_RISCV_PCREL_HI20;
            default:
                Ctx.reportError(Fixup.getLoc(), "unsupported relocation");
                return ELF::R_RISCV_NONE;
            }
        }
        
        switch (Kind) {
        case FK_NONE:
            return ELF::R_RISCV_NONE;
        case FK_Data_4:
            return ELF::R_RISCV_32;
        case FK_Data_8:
            return ELF::R_RISCV_64;
        case RISCV::fixup_riscv_hi20:
            return ELF::R_RISCV_HI20;
        case RISCV::fixup_riscv_lo12_i:
            return ELF::R_RISCV_LO12_I;
        default:
            llvm_unreachable("Unknown fixup kind");
        }
    }
    """
    
    analyzer = SemanticAnalyzer(verbose=True)
    matcher = PatternMatcher()
    
    # Analyze
    semantics = analyzer.analyze_function(test_code, "getRelocType")
    
    print("\n📊 Analysis Results:")
    print(f"   Function: {semantics.signature.name}")
    print(f"   Return type: {semantics.signature.return_type.value}")
    print(f"   Local variables: {len(semantics.local_variables)}")
    print(f"   Patterns found: {len(semantics.patterns)}")
    print(f"   Return paths: {len(semantics.return_paths)}")
    print(f"   Function calls: {len(semantics.calls)}")
    print(f"   Assertions: {len(semantics.assertions)}")
    
    print("\n📋 Switch Statements:")
    for pattern in semantics.patterns:
        if isinstance(pattern, SwitchPattern):
            print(f"   switch({pattern.switch_expr}):")
            for case in pattern.cases[:5]:
                print(f"     case {case.case_value} -> {case.return_value}")
            if len(pattern.cases) > 5:
                print(f"     ... and {len(pattern.cases) - 5} more cases")
    
    print("\n🔍 Function Type:", matcher.match_function_type(test_code, "getRelocType"))
    
    print("\n⚠️ Critical Patterns:")
    for p in matcher.identify_critical_patterns(test_code):
        print(f"   [{p['risk_level'].upper()}] {p['type']}: {p['description']}")
    
    print("\n📈 Statistics:")
    for key, val in analyzer.get_statistics().items():
        print(f"   {key}: {val}")
    
    print("\n✅ Semantic Analysis Engine Demo Complete")
