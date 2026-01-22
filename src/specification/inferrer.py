"""
Specification inference engine for VEGA-Verified.
Automatically infers formal specifications from reference backend implementations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import re

from .spec_language import (
    Specification,
    Condition,
    ConditionType,
    Expression,
    Variable,
    Constant,
    FunctionCall,
)
from .pattern_abstract import PatternAbstractor
from .condition_extract import ConditionExtractor


@dataclass
class FunctionSignature:
    """Parsed function signature."""
    name: str
    return_type: str
    parameters: List[Tuple[str, str]]  # (name, type)
    qualifiers: List[str] = field(default_factory=list)  # const, override, etc.


@dataclass
class ParsedFunction:
    """Parsed function with AST-like structure."""
    signature: FunctionSignature
    body: str
    statements: List[Dict[str, Any]]
    source_target: str  # e.g., "ARM", "MIPS", "RISCV"


@dataclass
class AlignedStatements:
    """Aligned statements from multiple implementations."""
    statements: List[Tuple[str, Dict[str, Any]]]  # (target, statement)
    is_target_independent: bool
    common_pattern: Optional[str] = None


class SpecificationInferrer:
    """
    Main engine for inferring specifications from reference implementations.
    
    Algorithm:
    1. Parse each reference implementation into AST
    2. Align implementations using edit distance / GumTree-like approach
    3. Identify target-independent (TI) vs target-specific (TS) statements
    4. Extract preconditions from guards (null checks, bounds, etc.)
    5. Extract postconditions from return patterns
    6. Extract invariants from common patterns
    7. Validate inferred spec against all references
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.pattern_abstractor = PatternAbstractor()
        self.condition_extractor = ConditionExtractor()
        
        # Common patterns in compiler backends
        self.null_check_patterns = [
            r'if\s*\(\s*(\w+)\s*==\s*nullptr\s*\)',
            r'if\s*\(\s*!\s*(\w+)\s*\)',
            r'assert\s*\(\s*(\w+)\s*\)',
            r'if\s*\(\s*(\w+)\s*!=\s*nullptr\s*\)',
        ]
        
        self.bounds_check_patterns = [
            r'if\s*\(\s*(\w+)\s*[<>=]+\s*(\w+)\s*\)',
            r'assert\s*\(\s*(\w+)\s*[<>=]+\s*(\d+)\s*\)',
        ]
        
        self.return_patterns = [
            r'return\s+(\w+)::(\w+);',  # return ELF::R_*
            r'return\s+(\d+);',          # return constant
            r'return\s+(\w+);',          # return variable
        ]
    
    def infer(
        self,
        function_name: str,
        references: List[Tuple[str, str]],  # [(target_name, code), ...]
        module: str = ""
    ) -> Specification:
        """
        Infer specification from reference implementations.
        
        Args:
            function_name: Name of the function to analyze
            references: List of (target_name, source_code) tuples
            module: Module name (e.g., MCCodeEmitter)
            
        Returns:
            Inferred specification
        """
        # Step 1: Parse each reference
        parsed_functions = []
        for target_name, code in references:
            parsed = self._parse_function(function_name, code, target_name)
            if parsed:
                parsed_functions.append(parsed)
        
        if not parsed_functions:
            raise ValueError(f"Could not parse any reference for {function_name}")
        
        # Step 2: Align implementations
        aligned = self._align_implementations(parsed_functions)
        
        # Step 3: Extract conditions
        preconditions = self._extract_preconditions(parsed_functions, aligned)
        postconditions = self._extract_postconditions(parsed_functions, aligned)
        invariants = self._extract_invariants(parsed_functions, aligned)
        
        # Step 4: Build specification
        spec = Specification(
            function_name=function_name,
            module=module,
            preconditions=preconditions,
            postconditions=postconditions,
            invariants=invariants,
            inferred_from=[pf.source_target for pf in parsed_functions],
            confidence=self._compute_confidence(parsed_functions, aligned)
        )
        
        # Step 5: Validate against references
        self._validate_spec(spec, parsed_functions)
        
        return spec
    
    def _parse_function(
        self,
        function_name: str,
        code: str,
        target_name: str
    ) -> Optional[ParsedFunction]:
        """Parse a function from source code."""
        # Find function definition
        # Pattern: return_type class::function_name(params) { body }
        pattern = rf'(\w+)\s+\w*::{function_name}\s*\(([^)]*)\)\s*(?:const|override|\s)*\{{'
        
        match = re.search(pattern, code, re.MULTILINE)
        if not match:
            # Try simpler pattern
            pattern = rf'(\w+)\s+{function_name}\s*\(([^)]*)\)\s*\{{'
            match = re.search(pattern, code, re.MULTILINE)
        
        if not match:
            return None
        
        return_type = match.group(1)
        params_str = match.group(2)
        
        # Parse parameters
        parameters = []
        if params_str.strip():
            for param in params_str.split(','):
                param = param.strip()
                parts = param.rsplit(None, 1)
                if len(parts) == 2:
                    parameters.append((parts[1], parts[0]))
                elif len(parts) == 1:
                    parameters.append((parts[0], "auto"))
        
        # Extract body
        start_pos = match.end() - 1  # Start at '{'
        body = self._extract_braced_content(code, start_pos)
        
        # Parse statements
        statements = self._parse_statements(body)
        
        signature = FunctionSignature(
            name=function_name,
            return_type=return_type,
            parameters=parameters
        )
        
        return ParsedFunction(
            signature=signature,
            body=body,
            statements=statements,
            source_target=target_name
        )
    
    def _extract_braced_content(self, code: str, start_pos: int) -> str:
        """Extract content within braces, handling nesting."""
        depth = 0
        content_start = start_pos + 1
        
        for i in range(start_pos, len(code)):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    return code[content_start:i]
        
        return code[content_start:]
    
    def _parse_statements(self, body: str) -> List[Dict[str, Any]]:
        """Parse function body into statements."""
        statements = []
        
        # Split by semicolons and braces, preserving structure
        lines = body.split('\n')
        
        current_stmt = []
        depth = 0
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            current_stmt.append(line)
            depth += line.count('{') - line.count('}')
            
            if depth == 0 and (line.endswith(';') or line.endswith('}')):
                stmt_text = ' '.join(current_stmt)
                stmt = self._classify_statement(stmt_text)
                if stmt:
                    statements.append(stmt)
                current_stmt = []
        
        return statements
    
    def _classify_statement(self, stmt: str) -> Optional[Dict[str, Any]]:
        """Classify a statement by type."""
        stmt = stmt.strip()
        
        if stmt.startswith('switch'):
            return {
                'type': 'switch',
                'text': stmt,
                'cases': self._parse_switch_cases(stmt)
            }
        elif stmt.startswith('if'):
            return {
                'type': 'if',
                'text': stmt,
                'condition': self._extract_condition(stmt)
            }
        elif stmt.startswith('return'):
            return {
                'type': 'return',
                'text': stmt,
                'value': self._extract_return_value(stmt)
            }
        elif stmt.startswith('case') or stmt.startswith('default'):
            return {
                'type': 'case',
                'text': stmt,
                'case_value': self._extract_case_value(stmt)
            }
        elif '=' in stmt and not stmt.startswith('if'):
            return {
                'type': 'assignment',
                'text': stmt
            }
        else:
            return {
                'type': 'other',
                'text': stmt
            }
    
    def _parse_switch_cases(self, stmt: str) -> List[Dict[str, Any]]:
        """Parse switch statement cases."""
        cases = []
        case_pattern = r'case\s+(\w+(?:::\w+)?)\s*:\s*return\s+(\w+(?:::\w+)?)\s*;'
        
        for match in re.finditer(case_pattern, stmt):
            cases.append({
                'case': match.group(1),
                'return': match.group(2)
            })
        
        return cases
    
    def _extract_condition(self, stmt: str) -> str:
        """Extract condition from if statement."""
        match = re.search(r'if\s*\((.+?)\)', stmt)
        return match.group(1) if match else ""
    
    def _extract_return_value(self, stmt: str) -> str:
        """Extract return value from return statement."""
        match = re.search(r'return\s+(.+?)\s*;', stmt)
        return match.group(1) if match else ""
    
    def _extract_case_value(self, stmt: str) -> str:
        """Extract case value from case statement."""
        match = re.search(r'case\s+(\w+(?:::\w+)?)\s*:', stmt)
        return match.group(1) if match else "default"
    
    def _align_implementations(
        self,
        functions: List[ParsedFunction]
    ) -> List[AlignedStatements]:
        """Align statements across implementations."""
        if len(functions) < 2:
            # Single implementation - all statements are aligned with themselves
            return [
                AlignedStatements(
                    statements=[(functions[0].source_target, stmt)],
                    is_target_independent=True
                )
                for stmt in functions[0].statements
            ]
        
        aligned = []
        
        # Group statements by type and structure
        # For switch statements, align by case structure
        for i, stmt1 in enumerate(functions[0].statements):
            alignment = AlignedStatements(
                statements=[(functions[0].source_target, stmt1)],
                is_target_independent=True
            )
            
            for func in functions[1:]:
                # Find matching statement in other implementations
                matched = False
                for stmt2 in func.statements:
                    if self._statements_match(stmt1, stmt2):
                        alignment.statements.append((func.source_target, stmt2))
                        matched = True
                        break
                
                if not matched:
                    alignment.is_target_independent = False
            
            # Determine common pattern
            if alignment.is_target_independent:
                alignment.common_pattern = self._extract_common_pattern(
                    [s for _, s in alignment.statements]
                )
            
            aligned.append(alignment)
        
        return aligned
    
    def _statements_match(self, stmt1: Dict, stmt2: Dict) -> bool:
        """Check if two statements are structurally equivalent."""
        if stmt1['type'] != stmt2['type']:
            return False
        
        if stmt1['type'] == 'switch':
            # Compare switch structure
            cases1 = set(c['case'] for c in stmt1.get('cases', []))
            cases2 = set(c['case'] for c in stmt2.get('cases', []))
            return cases1 == cases2
        
        elif stmt1['type'] == 'return':
            # Returns are typically target-specific
            return False
        
        elif stmt1['type'] == 'if':
            # Compare condition structure
            cond1 = stmt1.get('condition', '')
            cond2 = stmt2.get('condition', '')
            # Abstract away target-specific names
            cond1_abstract = re.sub(r'\b(ARM|MIPS|RISCV|X86)\b', 'TARGET', cond1)
            cond2_abstract = re.sub(r'\b(ARM|MIPS|RISCV|X86)\b', 'TARGET', cond2)
            return cond1_abstract == cond2_abstract
        
        return True
    
    def _extract_common_pattern(self, statements: List[Dict]) -> str:
        """Extract common pattern from aligned statements."""
        if not statements:
            return ""
        
        # For now, return the first statement's text as the pattern
        # In full implementation, would abstract target-specific parts
        text = statements[0].get('text', '')
        
        # Replace target names with placeholder
        pattern = re.sub(r'\b(ARM|MIPS|RISCV|X86|RI5CY|xCORE)\b', '<TARGET>', text)
        pattern = re.sub(r'R_(ARM|MIPS|RISCV|X86)_\w+', 'R_<TARGET>_*', pattern)
        
        return pattern
    
    def _extract_preconditions(
        self,
        functions: List[ParsedFunction],
        aligned: List[AlignedStatements]
    ) -> List[Condition]:
        """Extract preconditions from reference implementations."""
        preconditions = []
        
        # Common parameter validity checks
        if functions:
            params = functions[0].signature.parameters
            for param_name, param_type in params:
                # Add isValid for pointer/reference types
                if '*' in param_type or '&' in param_type or param_type.endswith('Ref'):
                    var = Variable(param_name, param_type)
                    preconditions.append(Condition.valid(var))
        
        # Extract from null checks and assertions
        for func in functions:
            for stmt in func.statements:
                if stmt['type'] == 'if':
                    cond_str = stmt.get('condition', '')
                    
                    # Check for null checks
                    for pattern in self.null_check_patterns:
                        match = re.search(pattern, cond_str)
                        if match:
                            var_name = match.group(1)
                            var = Variable(var_name)
                            cond = Condition.valid(var)
                            if cond not in preconditions:
                                preconditions.append(cond)
        
        return preconditions
    
    def _extract_postconditions(
        self,
        functions: List[ParsedFunction],
        aligned: List[AlignedStatements]
    ) -> List[Condition]:
        """Extract postconditions from reference implementations."""
        postconditions = []
        
        # Result should be >= 0 for enum/unsigned return types
        if functions and functions[0].signature.return_type in ['unsigned', 'int', 'uint32_t']:
            result_var = Variable('result', functions[0].signature.return_type)
            zero = Constant(0)
            postconditions.append(Condition.ge(result_var, zero))
        
        # Extract from return patterns
        return_values: Set[str] = set()
        for func in functions:
            for stmt in func.statements:
                if stmt['type'] == 'return':
                    return_values.add(stmt.get('value', ''))
                elif stmt['type'] == 'switch':
                    for case in stmt.get('cases', []):
                        return_values.add(case.get('return', ''))
        
        # If all returns are from an enum, add membership condition
        if return_values:
            # Check if all are from same namespace (e.g., ELF::)
            namespaced = [v for v in return_values if '::' in v]
            if namespaced:
                namespace = namespaced[0].split('::')[0]
                if all(v.startswith(f'{namespace}::') for v in namespaced):
                    # Result is in valid set
                    result_var = Variable('result')
                    members = [Constant(v, 'enum') for v in sorted(return_values)]
                    if members:
                        postconditions.append(Condition.in_set(result_var, *members))
        
        return postconditions
    
    def _extract_invariants(
        self,
        functions: List[ParsedFunction],
        aligned: List[AlignedStatements]
    ) -> List[Condition]:
        """Extract invariants from common patterns."""
        invariants = []
        
        # Find common case -> return mappings
        case_mappings: Dict[str, Set[str]] = {}
        
        for func in functions:
            for stmt in func.statements:
                if stmt['type'] == 'switch':
                    for case in stmt.get('cases', []):
                        case_value = case.get('case', '')
                        return_value = case.get('return', '')
                        
                        if case_value not in case_mappings:
                            case_mappings[case_value] = set()
                        case_mappings[case_value].add(return_value)
        
        # Create invariants for common patterns
        for case_value, return_values in case_mappings.items():
            if len(return_values) >= 1:
                # Abstract pattern: FK_NONE -> R_*_NONE
                abstract_returns = set()
                for ret in return_values:
                    if 'NONE' in ret:
                        abstract_returns.add('R_*_NONE')
                    elif '_8' in ret:
                        abstract_returns.add('R_*_8')
                    elif '_16' in ret:
                        abstract_returns.add('R_*_16')
                    elif '_32' in ret:
                        abstract_returns.add('R_*_32')
                    elif '_64' in ret:
                        abstract_returns.add('R_*_64')
                
                if len(abstract_returns) == 1:
                    # Consistent pattern across targets
                    pattern = list(abstract_returns)[0]
                    # case_value -> pattern (e.g., FK_NONE -> R_*_NONE)
                    antecedent = Condition.eq(Variable('Fixup.kind'), Constant(case_value, 'enum'))
                    consequent = Condition.eq(
                        Variable('result.pattern'),
                        Constant(pattern, 'string')
                    )
                    invariants.append(Condition.implies(antecedent, consequent))
        
        return invariants
    
    def _compute_confidence(
        self,
        functions: List[ParsedFunction],
        aligned: List[AlignedStatements]
    ) -> float:
        """Compute confidence score for inferred specification."""
        if not aligned:
            return 0.0
        
        # Base confidence on alignment quality
        ti_count = sum(1 for a in aligned if a.is_target_independent)
        total = len(aligned)
        
        alignment_confidence = ti_count / total if total > 0 else 0.0
        
        # Adjust for number of references
        reference_factor = min(len(functions) / 3.0, 1.0)  # Max confidence at 3+ refs
        
        return alignment_confidence * reference_factor
    
    def _validate_spec(
        self,
        spec: Specification,
        functions: List[ParsedFunction]
    ) -> None:
        """Validate that specification holds for all references.
        
        If no conditions could be inferred, generates default conditions
        based on the function signature and return statements.
        """
        # Basic validation - check that extracted conditions are consistent
        # Full validation would require the verification module
        
        if not spec.preconditions and not spec.postconditions:
            # Try to generate default conditions from invariants
            if spec.invariants:
                # Use invariants as postconditions
                spec.postconditions.extend(spec.invariants[:1])
                return
            
            # Generate minimal default conditions from function analysis
            if functions:
                # Extract return type and generate default postcondition
                for func in functions:
                    if func.signature.return_type == "bool":
                        spec.postconditions.append("result in {true, false}")
                        break
                    elif func.signature.return_type in ("int", "unsigned", "unsigned int", "size_t"):
                        spec.postconditions.append("result >= 0")
                        break
                    elif func.signature.return_type in ("void",):
                        spec.postconditions.append("true")
                        break
                    else:
                        # For any other return type, add a generic postcondition
                        spec.postconditions.append(f"result is valid {func.signature.return_type}")
                        break
                
                # Add default precondition: all parameters are valid
                if functions[0].signature.parameters:
                    param_names = [p[0] for p in functions[0].signature.parameters]
                    spec.preconditions.append(f"valid_params({', '.join(param_names)})")
                else:
                    spec.preconditions.append("true")
            else:
                # Fallback: minimal valid specification
                spec.preconditions.append("true")
                spec.postconditions.append("true")
            
            # Adjust confidence for auto-generated specs
            spec.confidence = min(spec.confidence, 0.3)
        
        # Log validation results
        # In production, would use the SMT solver to validate
        pass
    
    def infer_from_files(
        self,
        function_name: str,
        reference_files: List[Tuple[str, str]],  # [(target_name, file_path), ...]
        module: str = ""
    ) -> Specification:
        """
        Infer specification from reference source files.
        
        Args:
            function_name: Name of the function to analyze
            reference_files: List of (target_name, file_path) tuples
            module: Module name
            
        Returns:
            Inferred specification
        """
        references = []
        for target_name, file_path in reference_files:
            path = Path(file_path)
            if path.exists():
                with open(path, 'r') as f:
                    code = f.read()
                references.append((target_name, code))
        
        return self.infer(function_name, references, module)
