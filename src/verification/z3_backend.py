"""
Z3 Backend for VEGA-Verified.

This module provides a comprehensive Z3-based verification backend that:
1. Translates compiler backend code patterns to Z3 formulas
2. Models switch statements, conditionals, and return mappings
3. Extracts meaningful counterexamples for CGNR repair

Key Features:
- Semantic modeling of switch/case statements (critical for getRelocType, etc.)
- IsPCRel condition tracking
- Fixup kind to relocation type mapping verification
- Counterexample extraction with concrete input/output values
"""

import z3
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import re
import time


class Z3BackendType(Enum):
    """Types of backend code patterns."""
    SWITCH_CASE = "switch_case"      # Switch statement with cases
    CONDITIONAL = "conditional"       # If-else pattern
    DIRECT_RETURN = "direct_return"   # Simple return
    MIXED = "mixed"                   # Combination


@dataclass
class SwitchCase:
    """Represents a case in a switch statement."""
    case_value: str
    return_value: str
    condition: Optional[str] = None  # Additional condition (e.g., IsPCRel)


@dataclass
class CodeModel:
    """Semantic model of compiler backend code."""
    function_name: str
    switch_expr: Optional[str] = None
    cases: List[SwitchCase] = field(default_factory=list)
    default_return: Optional[str] = None
    has_ispcrel: bool = False
    pcrel_cases: List[SwitchCase] = field(default_factory=list)
    non_pcrel_cases: List[SwitchCase] = field(default_factory=list)
    
    def get_all_case_values(self) -> Set[str]:
        """Get all case values."""
        values = {c.case_value for c in self.cases}
        values.update(c.case_value for c in self.pcrel_cases)
        values.update(c.case_value for c in self.non_pcrel_cases)
        return values
    
    def get_all_return_values(self) -> Set[str]:
        """Get all possible return values."""
        values = {c.return_value for c in self.cases}
        values.update(c.return_value for c in self.pcrel_cases)
        values.update(c.return_value for c in self.non_pcrel_cases)
        if self.default_return:
            values.add(self.default_return)
        return values


@dataclass
class Z3VerificationResult:
    """Result from Z3 verification."""
    verified: bool
    counterexample: Optional[Dict[str, Any]] = None
    time_ms: float = 0.0
    z3_stats: Optional[Dict[str, Any]] = None
    violated_property: Optional[str] = None


class CodeParser:
    """
    Parser for compiler backend code patterns.
    Extracts semantic information for Z3 modeling.
    """
    
    def __init__(self):
        # Patterns for parsing
        self.switch_pattern = re.compile(r'switch\s*\((\w+)\)\s*\{')
        self.case_return_pattern = re.compile(
            r'case\s+(\w+(?:::\w+)*)\s*:\s*(?:return\s+)?(\w+(?:::\w+)*)\s*;'
        )
        self.case_with_condition_pattern = re.compile(
            r'case\s+(\w+(?:::\w+)*)\s*:\s*return\s+(\w+)\s*\?\s*(\w+(?:::\w+)*)\s*:\s*(\w+(?:::\w+)*)\s*;'
        )
        self.if_ispcrel_pattern = re.compile(r'if\s*\(\s*IsPCRel\s*\)')
        self.default_pattern = re.compile(r'default\s*:\s*(?:return\s+)?(\w+(?:::\w+)*)\s*;')
    
    def parse(self, code: str, function_name: str = "unknown") -> CodeModel:
        """
        Parse compiler backend code into a semantic model.
        
        Args:
            code: Source code to parse
            function_name: Name of the function
            
        Returns:
            CodeModel representing the code semantics
        """
        model = CodeModel(function_name=function_name)
        
        # Check for IsPCRel pattern
        model.has_ispcrel = bool(self.if_ispcrel_pattern.search(code))
        
        # Find switch expression
        switch_match = self.switch_pattern.search(code)
        if switch_match:
            model.switch_expr = switch_match.group(1)
        
        # Split by IsPCRel if present
        if model.has_ispcrel:
            self._parse_with_ispcrel(code, model)
        else:
            self._parse_simple_switch(code, model)
        
        # Find default case
        default_match = self.default_pattern.search(code)
        if default_match:
            model.default_return = default_match.group(1)
        
        return model
    
    def _parse_simple_switch(self, code: str, model: CodeModel) -> None:
        """Parse a simple switch statement without IsPCRel."""
        # Find case-return patterns with ternary operators
        for match in self.case_with_condition_pattern.finditer(code):
            case_val = match.group(1)
            condition = match.group(2)
            true_ret = match.group(3)
            false_ret = match.group(4)
            
            # This is typically "IsPCRel ? pcrel_reloc : abs_reloc"
            if condition == "IsPCRel":
                model.pcrel_cases.append(SwitchCase(case_val, true_ret))
                model.non_pcrel_cases.append(SwitchCase(case_val, false_ret))
                model.has_ispcrel = True
            else:
                model.cases.append(SwitchCase(case_val, true_ret, condition))
        
        # Find simple case-return patterns
        for match in self.case_return_pattern.finditer(code):
            case_val = match.group(1)
            ret_val = match.group(2)
            
            # Skip if already parsed as conditional
            if any(c.case_value == case_val for c in model.pcrel_cases):
                continue
            if any(c.case_value == case_val for c in model.non_pcrel_cases):
                continue
            
            model.cases.append(SwitchCase(case_val, ret_val))
    
    def _parse_with_ispcrel(self, code: str, model: CodeModel) -> None:
        """Parse code with IsPCRel conditional blocks."""
        # Find the IsPCRel block
        if_match = self.if_ispcrel_pattern.search(code)
        if not if_match:
            return
        
        # Split into PCRel block and non-PCRel block
        # This is simplified - real implementation would need proper brace matching
        if_start = if_match.end()
        
        # Find cases in PCRel block (after "if (IsPCRel)")
        pcrel_section = code[if_start:]
        for match in self.case_return_pattern.finditer(pcrel_section[:pcrel_section.find('}') if '}' in pcrel_section else len(pcrel_section)]):
            model.pcrel_cases.append(SwitchCase(match.group(1), match.group(2)))
        
        # Find cases in main switch (typically after PCRel block)
        main_section = code[code.rfind('switch'):]
        for match in self.case_return_pattern.finditer(main_section):
            case_val = match.group(1)
            ret_val = match.group(2)
            # Skip if already in pcrel_cases
            if not any(c.case_value == case_val for c in model.pcrel_cases):
                model.non_pcrel_cases.append(SwitchCase(case_val, ret_val))


class Z3Verifier:
    """
    Z3-based verifier for compiler backend code.
    
    Verifies semantic correctness by:
    1. Building Z3 models of switch/case behavior
    2. Checking invariants (e.g., FK_Data_4 + IsPCRel → R_*_PC32)
    3. Extracting counterexamples when verification fails
    """
    
    def __init__(self, timeout_ms: int = 30000, verbose: bool = False):
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        self.parser = CodeParser()
        
        # Statistics
        self.stats = {
            "verifications": 0,
            "verified": 0,
            "failed": 0,
            "timeout": 0,
            "total_time_ms": 0.0,
        }
    
    def verify(
        self,
        code: str,
        spec: 'Specification',
        expected_mappings: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Z3VerificationResult:
        """
        Verify code against specification using Z3.
        
        Args:
            code: Source code to verify
            spec: Specification with preconditions/postconditions/invariants
            expected_mappings: Optional expected input→output mappings
            
        Returns:
            Z3VerificationResult with verification status and counterexample
        """
        self.stats["verifications"] += 1
        start_time = time.time()
        
        # Parse code into semantic model
        model = self.parser.parse(code, spec.function_name)
        
        if self.verbose:
            print(f"Parsed model: {model.function_name}")
            print(f"  Has IsPCRel: {model.has_ispcrel}")
            print(f"  Cases: {len(model.cases)}")
            print(f"  PCRel cases: {len(model.pcrel_cases)}")
            print(f"  Non-PCRel cases: {len(model.non_pcrel_cases)}")
        
        # Build Z3 model and verify
        try:
            result = self._verify_with_z3(model, spec, expected_mappings)
        except Exception as e:
            if self.verbose:
                print(f"Z3 verification error: {e}")
            result = Z3VerificationResult(
                verified=False,
                counterexample={"error": str(e)},
            )
        
        result.time_ms = (time.time() - start_time) * 1000
        self.stats["total_time_ms"] += result.time_ms
        
        if result.verified:
            self.stats["verified"] += 1
        else:
            self.stats["failed"] += 1
        
        return result
    
    def _verify_with_z3(
        self,
        model: CodeModel,
        spec: 'Specification',
        expected_mappings: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Z3VerificationResult:
        """Build Z3 model and check verification conditions."""
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        
        # Create Z3 variables
        kind = z3.Int('Kind')  # Fixup kind
        ispcrel = z3.Bool('IsPCRel')  # PC-relative flag
        result = z3.Int('result')  # Function return value
        
        # Create symbolic constants for case values and return values
        case_consts = {}
        ret_consts = {}
        
        for case_val in model.get_all_case_values():
            case_consts[case_val] = z3.Int(f'case_{case_val}')
        
        for ret_val in model.get_all_return_values():
            ret_consts[ret_val] = z3.Int(f'ret_{ret_val}')
        
        # Build constraints for each case
        case_constraints = []
        
        # PCRel-specific cases
        for case in model.pcrel_cases:
            if case.case_value in case_consts and case.return_value in ret_consts:
                constraint = z3.Implies(
                    z3.And(kind == case_consts[case.case_value], ispcrel),
                    result == ret_consts[case.return_value]
                )
                case_constraints.append(constraint)
        
        # Non-PCRel cases
        for case in model.non_pcrel_cases:
            if case.case_value in case_consts and case.return_value in ret_consts:
                constraint = z3.Implies(
                    z3.And(kind == case_consts[case.case_value], z3.Not(ispcrel)),
                    result == ret_consts[case.return_value]
                )
                case_constraints.append(constraint)
        
        # Simple cases (no IsPCRel dependency)
        for case in model.cases:
            if case.case_value in case_consts and case.return_value in ret_consts:
                constraint = z3.Implies(
                    kind == case_consts[case.case_value],
                    result == ret_consts[case.return_value]
                )
                case_constraints.append(constraint)
        
        # Default case
        if model.default_return and model.default_return in ret_consts:
            all_cases = z3.Or(*[kind == case_consts[cv] for cv in case_consts])
            constraint = z3.Implies(
                z3.Not(all_cases),
                result == ret_consts[model.default_return]
            )
            case_constraints.append(constraint)
        
        # Add code behavior constraints
        for constraint in case_constraints:
            solver.add(constraint)
        
        # Verify specification invariants
        from ..specification.spec_language import ConditionType, Variable, Constant
        
        for inv in spec.invariants:
            inv_formula = self._invariant_to_z3(inv, kind, ispcrel, result, case_consts, ret_consts)
            if inv_formula is not None:
                # Check if invariant can be violated
                solver.push()
                solver.add(z3.Not(inv_formula))
                
                check_result = solver.check()
                
                if check_result == z3.sat:
                    # Found counterexample
                    z3_model = solver.model()
                    counterexample = self._extract_counterexample(
                        z3_model, kind, ispcrel, result, case_consts, ret_consts
                    )
                    counterexample["violated_invariant"] = str(inv)
                    solver.pop()
                    return Z3VerificationResult(
                        verified=False,
                        counterexample=counterexample,
                        violated_property=str(inv),
                    )
                
                solver.pop()
        
        # Verify expected mappings if provided
        if expected_mappings:
            for input_key, expected_output in expected_mappings.items():
                # Parse input
                parts = input_key.split(',')
                kind_val = parts[0].strip()
                pcrel_val = len(parts) > 1 and 'pcrel' in parts[1].lower()
                
                if kind_val in case_consts:
                    solver.push()
                    solver.add(kind == case_consts[kind_val])
                    solver.add(ispcrel if pcrel_val else z3.Not(ispcrel))
                    
                    if expected_output in ret_consts:
                        solver.add(result != ret_consts[expected_output])
                    
                    check_result = solver.check()
                    
                    if check_result == z3.sat:
                        z3_model = solver.model()
                        counterexample = self._extract_counterexample(
                            z3_model, kind, ispcrel, result, case_consts, ret_consts
                        )
                        counterexample["expected"] = expected_output
                        solver.pop()
                        return Z3VerificationResult(
                            verified=False,
                            counterexample=counterexample,
                        )
                    
                    solver.pop()
        
        # All checks passed
        return Z3VerificationResult(verified=True)
    
    def _invariant_to_z3(
        self,
        invariant: 'Condition',
        kind: z3.ExprRef,
        ispcrel: z3.ExprRef,
        result: z3.ExprRef,
        case_consts: Dict[str, z3.ExprRef],
        ret_consts: Dict[str, z3.ExprRef]
    ) -> Optional[z3.ExprRef]:
        """Convert specification invariant to Z3 formula."""
        from ..specification.spec_language import ConditionType, Variable, Constant
        
        def expr_to_z3(expr):
            if isinstance(expr, Variable):
                name = expr.name.lower()
                if 'kind' in name or 'fixup' in name:
                    return kind
                elif 'pcrel' in name:
                    return ispcrel
                elif 'result' in name or 'return' in name:
                    return result
                else:
                    # Check if it's a known constant
                    for const_name, const_val in case_consts.items():
                        if expr.name == const_name or expr.name.endswith(const_name):
                            return const_val
                    for const_name, const_val in ret_consts.items():
                        if expr.name == const_name or expr.name.endswith(const_name):
                            return const_val
                    return z3.Int(expr.name)
            elif isinstance(expr, Constant):
                val = expr.value
                if isinstance(val, bool):
                    return z3.BoolVal(val)
                elif isinstance(val, str):
                    # Check if it's a known constant
                    if val in case_consts:
                        return case_consts[val]
                    if val in ret_consts:
                        return ret_consts[val]
                    return z3.Int(f'const_{val}')
                else:
                    try:
                        return z3.IntVal(int(val))
                    except:
                        return z3.Int(f'const_{val}')
            elif hasattr(expr, 'cond_type'):
                # Nested condition
                return self._condition_to_z3(expr, kind, ispcrel, result, case_consts, ret_consts)
            else:
                return z3.Int('unknown')
        
        return self._condition_to_z3(invariant, kind, ispcrel, result, case_consts, ret_consts)
    
    def _condition_to_z3(
        self,
        condition: 'Condition',
        kind: z3.ExprRef,
        ispcrel: z3.ExprRef,
        result: z3.ExprRef,
        case_consts: Dict[str, z3.ExprRef],
        ret_consts: Dict[str, z3.ExprRef]
    ) -> Optional[z3.ExprRef]:
        """Convert Condition to Z3 formula."""
        from ..specification.spec_language import ConditionType, Variable, Constant
        
        def expr_to_z3(expr):
            if isinstance(expr, Variable):
                name = expr.name.lower()
                if 'kind' in name or 'fixup' in name:
                    return kind
                elif 'pcrel' in name:
                    return ispcrel
                elif 'result' in name or 'return' in name:
                    return result
                else:
                    # Check if it's a known constant
                    for const_name, const_val in case_consts.items():
                        if expr.name == const_name or expr.name.endswith(const_name):
                            return const_val
                    for const_name, const_val in ret_consts.items():
                        if expr.name == const_name or expr.name.endswith(const_name):
                            return const_val
                    return z3.Int(expr.name)
            elif isinstance(expr, Constant):
                val = expr.value
                if isinstance(val, bool):
                    return z3.BoolVal(val)
                elif isinstance(val, str):
                    if val in case_consts:
                        return case_consts[val]
                    if val in ret_consts:
                        return ret_consts[val]
                    return z3.Int(f'const_{val}')
                else:
                    try:
                        return z3.IntVal(int(val))
                    except:
                        return z3.Int(f'const_{val}')
            elif hasattr(expr, 'cond_type'):
                return self._condition_to_z3(expr, kind, ispcrel, result, case_consts, ret_consts)
            else:
                return z3.Int('unknown')
        
        ctype = condition.cond_type
        operands = condition.operands
        
        try:
            if ctype == ConditionType.EQUALITY:
                return expr_to_z3(operands[0]) == expr_to_z3(operands[1])
            elif ctype == ConditionType.INEQUALITY:
                return expr_to_z3(operands[0]) != expr_to_z3(operands[1])
            elif ctype == ConditionType.LESS_THAN:
                return expr_to_z3(operands[0]) < expr_to_z3(operands[1])
            elif ctype == ConditionType.LESS_EQUAL:
                return expr_to_z3(operands[0]) <= expr_to_z3(operands[1])
            elif ctype == ConditionType.GREATER_THAN:
                return expr_to_z3(operands[0]) > expr_to_z3(operands[1])
            elif ctype == ConditionType.GREATER_EQUAL:
                return expr_to_z3(operands[0]) >= expr_to_z3(operands[1])
            elif ctype == ConditionType.IMPLIES:
                ant = self._condition_to_z3(operands[0], kind, ispcrel, result, case_consts, ret_consts)
                cons = self._condition_to_z3(operands[1], kind, ispcrel, result, case_consts, ret_consts)
                if ant is not None and cons is not None:
                    return z3.Implies(ant, cons)
            elif ctype == ConditionType.AND:
                clauses = [self._condition_to_z3(op, kind, ispcrel, result, case_consts, ret_consts) 
                          for op in operands]
                clauses = [c for c in clauses if c is not None]
                return z3.And(*clauses) if clauses else z3.BoolVal(True)
            elif ctype == ConditionType.OR:
                clauses = [self._condition_to_z3(op, kind, ispcrel, result, case_consts, ret_consts) 
                          for op in operands]
                clauses = [c for c in clauses if c is not None]
                return z3.Or(*clauses) if clauses else z3.BoolVal(False)
            elif ctype == ConditionType.NOT:
                inner = self._condition_to_z3(operands[0], kind, ispcrel, result, case_consts, ret_consts)
                return z3.Not(inner) if inner is not None else None
            elif ctype == ConditionType.IS_VALID:
                return expr_to_z3(operands[0]) != 0
        except Exception as e:
            if self.verbose:
                print(f"Error converting condition to Z3: {e}")
        
        return None
    
    def _extract_counterexample(
        self,
        z3_model: z3.ModelRef,
        kind: z3.ExprRef,
        ispcrel: z3.ExprRef,
        result: z3.ExprRef,
        case_consts: Dict[str, z3.ExprRef],
        ret_consts: Dict[str, z3.ExprRef]
    ) -> Dict[str, Any]:
        """Extract counterexample from Z3 model."""
        counterexample = {
            "input_values": {},
            "expected_output": None,
            "actual_output": None,
        }
        
        # Get Kind value
        kind_val = z3_model.eval(kind, model_completion=True)
        kind_name = None
        for name, const in case_consts.items():
            const_val = z3_model.eval(const, model_completion=True)
            if str(kind_val) == str(const_val):
                kind_name = name
                break
        counterexample["input_values"]["Kind"] = kind_name or str(kind_val)
        
        # Get IsPCRel value
        ispcrel_val = z3_model.eval(ispcrel, model_completion=True)
        counterexample["input_values"]["IsPCRel"] = z3.is_true(ispcrel_val)
        
        # Get result value
        result_val = z3_model.eval(result, model_completion=True)
        result_name = None
        for name, const in ret_consts.items():
            const_val = z3_model.eval(const, model_completion=True)
            if str(result_val) == str(const_val):
                result_name = name
                break
        counterexample["actual_output"] = result_name or str(result_val)
        
        return counterexample
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return self.stats.copy()


def create_z3_verifier(timeout_ms: int = 30000, verbose: bool = False) -> Z3Verifier:
    """Factory function to create a Z3 verifier."""
    return Z3Verifier(timeout_ms=timeout_ms, verbose=verbose)


# Quick test
if __name__ == "__main__":
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
        default:
            return ELF::R_RISCV_NONE;
        }
    }
    """
    
    parser = CodeParser()
    model = parser.parse(test_code, "getRelocType")
    
    print("Parsed Code Model:")
    print(f"  Function: {model.function_name}")
    print(f"  Has IsPCRel: {model.has_ispcrel}")
    print(f"  PCRel cases: {[(c.case_value, c.return_value) for c in model.pcrel_cases]}")
    print(f"  Non-PCRel cases: {[(c.case_value, c.return_value) for c in model.non_pcrel_cases]}")
    print(f"  Default: {model.default_return}")
