"""
Condition extraction for specification inference.
Extracts preconditions, postconditions, and invariants from code patterns.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple
import re

from .spec_language import (
    Condition,
    ConditionType,
    Variable,
    Constant,
    FunctionCall,
    Expression,
)


@dataclass
class ExtractedCondition:
    """A condition extracted from code."""
    condition: Condition
    source: str  # Where it was extracted from
    confidence: float  # How confident we are in this extraction
    category: str  # precondition, postcondition, invariant


class ConditionExtractor:
    """
    Extracts formal conditions from code patterns.
    
    Supports extraction of:
    - Null/validity checks -> preconditions
    - Bounds checks -> preconditions/invariants
    - Return value constraints -> postconditions
    - Case mapping patterns -> invariants
    """
    
    # Patterns for different condition types
    NULL_CHECK_PATTERNS = [
        # if (ptr != nullptr)
        (r'if\s*\(\s*(\w+)\s*!=\s*nullptr\s*\)', 'positive_null'),
        # if (ptr == nullptr) return
        (r'if\s*\(\s*(\w+)\s*==\s*nullptr\s*\)', 'negative_null'),
        # if (!ptr)
        (r'if\s*\(\s*!\s*(\w+)\s*\)', 'falsy_check'),
        # if (ptr)
        (r'if\s*\(\s*(\w+)\s*\)', 'truthy_check'),
        # assert(ptr)
        (r'assert\s*\(\s*(\w+)\s*\)', 'assert_check'),
    ]
    
    BOUNDS_CHECK_PATTERNS = [
        # if (x < MAX)
        (r'if\s*\(\s*(\w+)\s*<\s*(\w+)\s*\)', 'upper_bound'),
        # if (x <= MAX)
        (r'if\s*\(\s*(\w+)\s*<=\s*(\w+)\s*\)', 'upper_bound_inclusive'),
        # if (x > MIN)
        (r'if\s*\(\s*(\w+)\s*>\s*(\w+)\s*\)', 'lower_bound'),
        # if (x >= MIN)
        (r'if\s*\(\s*(\w+)\s*>=\s*(\w+)\s*\)', 'lower_bound_inclusive'),
        # if (x >= MIN && x < MAX)
        (r'if\s*\(\s*(\w+)\s*>=\s*(\w+)\s*&&\s*\1\s*<\s*(\w+)\s*\)', 'range_check'),
    ]
    
    RETURN_PATTERNS = [
        # return ENUM::VALUE
        (r'return\s+(\w+)::(\w+)\s*;', 'enum_return'),
        # return expr ? a : b
        (r'return\s+(.+)\s*\?\s*(.+)\s*:\s*(.+)\s*;', 'ternary_return'),
        # return constant
        (r'return\s+(\d+)\s*;', 'constant_return'),
        # return variable
        (r'return\s+(\w+)\s*;', 'variable_return'),
    ]
    
    VALIDITY_PATTERNS = [
        # isValid(), isLegal(), etc.
        (r'(\w+)\.(isValid|isLegal|isInitialized)\(\)', 'method_validity'),
        # hasXXX()
        (r'(\w+)\.has(\w+)\(\)', 'has_property'),
        # getXXX() != invalid
        (r'(\w+)\.get(\w+)\(\)\s*!=\s*(\w+)', 'getter_validity'),
    ]
    
    def __init__(self):
        self.extracted_conditions: List[ExtractedCondition] = []
    
    def extract_all(
        self,
        statements: List[Dict[str, Any]],
        function_signature: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Condition], List[Condition], List[Condition]]:
        """
        Extract all conditions from statements.
        
        Args:
            statements: Parsed statements
            function_signature: Optional signature for parameter analysis
            
        Returns:
            Tuple of (preconditions, postconditions, invariants)
        """
        self.extracted_conditions = []
        
        for stmt in statements:
            self._extract_from_statement(stmt)
        
        # Add parameter validity if signature provided
        if function_signature:
            self._extract_from_signature(function_signature)
        
        # Categorize and deduplicate
        preconditions = []
        postconditions = []
        invariants = []
        
        seen_conditions: Set[str] = set()
        
        for ec in self.extracted_conditions:
            cond_str = str(ec.condition)
            if cond_str in seen_conditions:
                continue
            seen_conditions.add(cond_str)
            
            if ec.category == 'precondition':
                preconditions.append(ec.condition)
            elif ec.category == 'postcondition':
                postconditions.append(ec.condition)
            elif ec.category == 'invariant':
                invariants.append(ec.condition)
        
        return preconditions, postconditions, invariants
    
    def _extract_from_statement(self, stmt: Dict[str, Any]) -> None:
        """Extract conditions from a single statement."""
        stmt_type = stmt.get('type', 'other')
        text = stmt.get('text', '')
        
        if stmt_type == 'if':
            self._extract_from_if(stmt)
        elif stmt_type == 'switch':
            self._extract_from_switch(stmt)
        elif stmt_type == 'return':
            self._extract_from_return(stmt)
        
        # Also extract from raw text
        self._extract_from_text(text)
    
    def _extract_from_if(self, stmt: Dict[str, Any]) -> None:
        """Extract conditions from if statement."""
        condition_text = stmt.get('condition', '')
        
        # Check for null checks
        for pattern, check_type in self.NULL_CHECK_PATTERNS:
            match = re.search(pattern, stmt.get('text', ''))
            if match:
                var_name = match.group(1)
                var = Variable(var_name)
                
                if check_type in ('positive_null', 'truthy_check', 'assert_check'):
                    # Variable must be valid
                    cond = Condition.valid(var)
                else:
                    # Negated check - still implies validity expectation
                    cond = Condition.valid(var)
                
                self.extracted_conditions.append(ExtractedCondition(
                    condition=cond,
                    source=f"if statement: {condition_text}",
                    confidence=0.9,
                    category='precondition'
                ))
        
        # Check for bounds checks
        for pattern, check_type in self.BOUNDS_CHECK_PATTERNS:
            match = re.search(pattern, stmt.get('text', ''))
            if match:
                if check_type == 'range_check':
                    var_name, min_val, max_val = match.groups()
                    var = Variable(var_name)
                    lo = Variable(min_val) if not min_val.isdigit() else Constant(int(min_val))
                    hi = Variable(max_val) if not max_val.isdigit() else Constant(int(max_val))
                    cond = Condition.in_range(var, lo, hi)
                elif check_type in ('upper_bound', 'upper_bound_inclusive'):
                    var_name, bound = match.groups()
                    var = Variable(var_name)
                    bound_expr = Variable(bound) if not bound.isdigit() else Constant(int(bound))
                    if 'inclusive' in check_type:
                        cond = Condition.le(var, bound_expr)
                    else:
                        cond = Condition.lt(var, bound_expr)
                else:
                    var_name, bound = match.groups()
                    var = Variable(var_name)
                    bound_expr = Variable(bound) if not bound.isdigit() else Constant(int(bound))
                    if 'inclusive' in check_type:
                        cond = Condition.ge(var, bound_expr)
                    else:
                        cond = Condition.gt(var, bound_expr)
                
                self.extracted_conditions.append(ExtractedCondition(
                    condition=cond,
                    source=f"bounds check: {condition_text}",
                    confidence=0.85,
                    category='precondition'
                ))
    
    def _extract_from_switch(self, stmt: Dict[str, Any]) -> None:
        """Extract invariants from switch statement."""
        cases = stmt.get('cases', [])
        
        # Each case defines an invariant: case_value -> return_pattern
        for case in cases:
            case_value = case.get('case', '')
            return_value = case.get('return', '')
            
            if case_value and return_value:
                # Create invariant: Fixup.kind == case_value => result == return_value
                antecedent = Condition.eq(
                    Variable('Fixup.kind'),
                    Constant(case_value, 'enum')
                )
                consequent = Condition.eq(
                    Variable('result'),
                    Constant(return_value, 'enum')
                )
                cond = Condition.implies(antecedent, consequent)
                
                self.extracted_conditions.append(ExtractedCondition(
                    condition=cond,
                    source=f"switch case: {case_value} -> {return_value}",
                    confidence=0.95,
                    category='invariant'
                ))
        
        # Also extract that result must be one of the return values
        return_values = [case.get('return', '') for case in cases if case.get('return')]
        if return_values:
            result_var = Variable('result')
            members = [Constant(v, 'enum') for v in return_values]
            cond = Condition.in_set(result_var, *members)
            
            self.extracted_conditions.append(ExtractedCondition(
                condition=cond,
                source=f"switch return values: {return_values}",
                confidence=0.9,
                category='postcondition'
            ))
    
    def _extract_from_return(self, stmt: Dict[str, Any]) -> None:
        """Extract postconditions from return statement."""
        return_value = stmt.get('value', '')
        
        for pattern, return_type in self.RETURN_PATTERNS:
            match = re.match(pattern, f"return {return_value};")
            if match:
                if return_type == 'enum_return':
                    namespace, value = match.groups()
                    # Result must be in valid enum range
                    cond = Condition.ge(Variable('result'), Constant(0))
                    self.extracted_conditions.append(ExtractedCondition(
                        condition=cond,
                        source=f"enum return: {namespace}::{value}",
                        confidence=0.9,
                        category='postcondition'
                    ))
                
                elif return_type == 'constant_return':
                    const_val = int(match.group(1))
                    # Result equals constant
                    cond = Condition.eq(Variable('result'), Constant(const_val))
                    self.extracted_conditions.append(ExtractedCondition(
                        condition=cond,
                        source=f"constant return: {const_val}",
                        confidence=0.95,
                        category='postcondition'
                    ))
                
                elif return_type == 'ternary_return':
                    # Conditional return - extract both branches
                    condition, true_val, false_val = match.groups()
                    # This creates a conditional postcondition
                    # For now, just note that result is one of two values
                    pass
    
    def _extract_from_text(self, text: str) -> None:
        """Extract conditions from raw text."""
        # Check for validity patterns
        for pattern, check_type in self.VALIDITY_PATTERNS:
            for match in re.finditer(pattern, text):
                if check_type == 'method_validity':
                    obj_name, method = match.groups()
                    var = Variable(f"{obj_name}")
                    cond = Condition.valid(var)
                    
                    self.extracted_conditions.append(ExtractedCondition(
                        condition=cond,
                        source=f"validity check: {obj_name}.{method}()",
                        confidence=0.85,
                        category='precondition'
                    ))
    
    def _extract_from_signature(self, signature: Dict[str, Any]) -> None:
        """Extract conditions from function signature."""
        params = signature.get('parameters', [])
        return_type = signature.get('return_type', '')
        
        for param_name, param_type in params:
            # Pointer/reference types should be valid
            if '*' in param_type or '&' in param_type:
                var = Variable(param_name, param_type)
                cond = Condition.valid(var)
                
                self.extracted_conditions.append(ExtractedCondition(
                    condition=cond,
                    source=f"parameter type: {param_type} {param_name}",
                    confidence=0.8,
                    category='precondition'
                ))
            
            # Ref types (MCFixup &, etc.)
            if 'Ref' in param_type or param_type.endswith('&'):
                var = Variable(param_name, param_type)
                cond = Condition.valid(var)
                
                self.extracted_conditions.append(ExtractedCondition(
                    condition=cond,
                    source=f"reference parameter: {param_type} {param_name}",
                    confidence=0.85,
                    category='precondition'
                ))
        
        # Return type constraints
        if return_type in ('unsigned', 'uint32_t', 'uint64_t', 'size_t'):
            result_var = Variable('result', return_type)
            cond = Condition.ge(result_var, Constant(0))
            
            self.extracted_conditions.append(ExtractedCondition(
                condition=cond,
                source=f"unsigned return type: {return_type}",
                confidence=0.95,
                category='postcondition'
            ))
    
    def extract_preconditions(
        self,
        statements: List[Dict[str, Any]],
        signature: Optional[Dict[str, Any]] = None
    ) -> List[Condition]:
        """Extract only preconditions."""
        pre, _, _ = self.extract_all(statements, signature)
        return pre
    
    def extract_postconditions(
        self,
        statements: List[Dict[str, Any]],
        signature: Optional[Dict[str, Any]] = None
    ) -> List[Condition]:
        """Extract only postconditions."""
        _, post, _ = self.extract_all(statements, signature)
        return post
    
    def extract_invariants(
        self,
        statements: List[Dict[str, Any]],
        signature: Optional[Dict[str, Any]] = None
    ) -> List[Condition]:
        """Extract only invariants."""
        _, _, inv = self.extract_all(statements, signature)
        return inv
