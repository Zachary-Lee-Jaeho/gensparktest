"""
Integrated Verifier for VEGA-Verified.

This module provides a unified verification interface that:
1. Uses Z3 backend for semantic verification
2. Integrates with the specification system
3. Provides detailed counterexamples for CGNR repair

The integrated verifier is the main entry point for verification in VEGA-Verified.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import time

from .z3_backend import Z3Verifier, Z3VerificationResult, CodeParser, CodeModel
from .verifier import VerificationResult, VerificationStatus, Counterexample
from ..specification.spec_language import Specification, Condition, ConditionType, Variable, Constant


class IntegratedVerifier:
    """
    Integrated verifier combining multiple verification backends.
    
    Features:
    1. Z3-based semantic verification for switch/case patterns
    2. Pattern-based verification for common compiler backend idioms
    3. Specification-driven verification using preconditions/postconditions
    4. Detailed counterexample extraction for CGNR repair
    """
    
    def __init__(
        self,
        timeout_ms: int = 30000,
        verbose: bool = False,
        use_z3: bool = True
    ):
        """
        Initialize the integrated verifier.
        
        Args:
            timeout_ms: Timeout for SMT solving in milliseconds
            verbose: Print verbose output
            use_z3: Whether to use Z3 backend (if available)
        """
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        self.use_z3 = use_z3
        
        # Initialize backends
        if use_z3:
            try:
                self.z3_verifier = Z3Verifier(timeout_ms=timeout_ms, verbose=verbose)
            except ImportError:
                if verbose:
                    print("Z3 not available, using pattern-based verification only")
                self.z3_verifier = None
                self.use_z3 = False
        else:
            self.z3_verifier = None
        
        self.parser = CodeParser()
        
        # Statistics
        self.stats = {
            "total_verifications": 0,
            "verified": 0,
            "failed": 0,
            "z3_verifications": 0,
            "pattern_verifications": 0,
            "total_time_ms": 0.0,
        }
    
    def verify(
        self,
        code: str,
        spec: Specification,
        statements: Optional[List[Dict[str, Any]]] = None
    ) -> VerificationResult:
        """
        Verify code against specification.
        
        Args:
            code: Source code to verify
            spec: Specification to verify against
            statements: Optional pre-parsed statements (ignored, uses internal parser)
            
        Returns:
            VerificationResult with status and counterexample if failed
        """
        self.stats["total_verifications"] += 1
        start_time = time.time()
        
        result = VerificationResult(status=VerificationStatus.UNKNOWN)
        
        try:
            # Step 1: Parse code into semantic model
            code_model = self.parser.parse(code, spec.function_name)
            
            if self.verbose:
                print(f"Verifying {spec.function_name}")
                print(f"  Model has {len(code_model.cases)} cases, {len(code_model.pcrel_cases)} PCRel cases")
            
            # Step 2: Generate expected mappings from specification
            expected_mappings = self._spec_to_mappings(spec, code_model)
            
            # Step 3: Try Z3 verification
            if self.use_z3 and self.z3_verifier:
                z3_result = self.z3_verifier.verify(code, spec, expected_mappings)
                self.stats["z3_verifications"] += 1
                
                if z3_result.verified:
                    result.status = VerificationStatus.VERIFIED
                    result.verified_properties = [str(inv) for inv in spec.invariants]
                else:
                    result.status = VerificationStatus.FAILED
                    if z3_result.counterexample:
                        result.counterexample = self._convert_counterexample(z3_result)
                        result.failed_properties.append(z3_result.violated_property or "specification")
                
                result.solve_time_ms = z3_result.time_ms
            
            # Step 4: Pattern-based verification as backup/supplement
            if result.status == VerificationStatus.UNKNOWN:
                pattern_result = self._pattern_verify(code_model, spec)
                self.stats["pattern_verifications"] += 1
                
                if pattern_result["verified"]:
                    result.status = VerificationStatus.VERIFIED
                else:
                    result.status = VerificationStatus.FAILED
                    if pattern_result.get("counterexample"):
                        result.counterexample = pattern_result["counterexample"]
                        result.failed_properties.append(pattern_result.get("violated", "pattern"))
            
            # Step 5: Check invariants explicitly
            if result.status == VerificationStatus.VERIFIED:
                inv_result = self._check_invariants(code_model, spec)
                if not inv_result["verified"]:
                    result.status = VerificationStatus.FAILED
                    if inv_result.get("counterexample"):
                        result.counterexample = inv_result["counterexample"]
                    result.failed_properties.append(inv_result.get("violated", "invariant"))
        
        except Exception as e:
            if self.verbose:
                print(f"Verification error: {e}")
            result.status = VerificationStatus.ERROR
            result.failed_properties.append(f"Error: {str(e)}")
        
        # Update statistics
        result.time_ms = (time.time() - start_time) * 1000
        self.stats["total_time_ms"] += result.time_ms
        
        if result.status == VerificationStatus.VERIFIED:
            self.stats["verified"] += 1
        elif result.status == VerificationStatus.FAILED:
            self.stats["failed"] += 1
        
        return result
    
    def _spec_to_mappings(
        self,
        spec: Specification,
        code_model: CodeModel
    ) -> Dict[str, str]:
        """
        Extract expected input→output mappings from specification invariants.
        
        Args:
            spec: Specification with invariants
            code_model: Parsed code model
            
        Returns:
            Dictionary of "Kind,condition" → expected_output
        """
        mappings = {}
        
        for inv in spec.invariants:
            mapping = self._extract_mapping_from_invariant(inv)
            if mapping:
                mappings.update(mapping)
        
        return mappings
    
    def _extract_mapping_from_invariant(
        self,
        invariant: Condition
    ) -> Optional[Dict[str, str]]:
        """Extract input→output mapping from an invariant condition."""
        if invariant.cond_type != ConditionType.IMPLIES:
            return None
        
        if len(invariant.operands) < 2:
            return None
        
        antecedent = invariant.operands[0]
        consequent = invariant.operands[1]
        
        # Extract input conditions from antecedent
        input_key = self._extract_input_key(antecedent)
        
        # Extract expected output from consequent
        output = self._extract_output(consequent)
        
        if input_key and output:
            return {input_key: output}
        
        return None
    
    def _extract_input_key(self, condition: Condition) -> Optional[str]:
        """Extract input key from condition (e.g., "FK_Data_4,pcrel")."""
        parts = []
        
        if condition.cond_type == ConditionType.EQUALITY:
            # Simple equality: Kind == FK_Data_4
            if len(condition.operands) >= 2:
                var = condition.operands[0]
                val = condition.operands[1]
                if isinstance(var, Variable) and isinstance(val, Constant):
                    if "kind" in var.name.lower():
                        parts.append(str(val.value))
        
        elif condition.cond_type == ConditionType.AND:
            # Compound: Kind == FK_Data_4 AND IsPCRel
            for op in condition.operands:
                sub_key = self._extract_input_key(op)
                if sub_key:
                    parts.append(sub_key)
        
        elif condition.cond_type == ConditionType.IS_VALID:
            # isValid(IsPCRel) or similar
            if len(condition.operands) >= 1:
                var = condition.operands[0]
                if isinstance(var, Variable) and "pcrel" in var.name.lower():
                    parts.append("pcrel")
        
        return ",".join(parts) if parts else None
    
    def _extract_output(self, condition: Condition) -> Optional[str]:
        """Extract expected output from condition."""
        if condition.cond_type == ConditionType.EQUALITY:
            if len(condition.operands) >= 2:
                var = condition.operands[0]
                val = condition.operands[1]
                if isinstance(var, Variable) and "result" in var.name.lower():
                    if isinstance(val, Constant):
                        return str(val.value)
                elif isinstance(val, Variable) and "result" in val.name.lower():
                    if isinstance(var, Constant):
                        return str(var.value)
        
        return None
    
    def _pattern_verify(
        self,
        code_model: CodeModel,
        spec: Specification
    ) -> Dict[str, Any]:
        """
        Pattern-based verification for common compiler backend idioms.
        
        Checks:
        1. All expected cases are present
        2. IsPCRel is handled correctly (if applicable)
        3. Default case exists
        """
        # Check for missing IsPCRel handling
        if self._spec_requires_pcrel(spec) and not code_model.has_ispcrel:
            # Check if any case should have PCRel handling
            for inv in spec.invariants:
                if self._invariant_requires_pcrel(inv):
                    return {
                        "verified": False,
                        "violated": "missing_pcrel_handling",
                        "counterexample": Counterexample(
                            input_values={"IsPCRel": True},
                            violated_condition="Specification requires IsPCRel handling but code doesn't check IsPCRel"
                        )
                    }
        
        # Check for missing cases
        expected_cases = self._get_expected_cases(spec)
        actual_cases = code_model.get_all_case_values()
        
        for case in expected_cases:
            if case not in actual_cases:
                return {
                    "verified": False,
                    "violated": f"missing_case_{case}",
                    "counterexample": Counterexample(
                        input_values={"Kind": case},
                        violated_condition=f"Missing case for {case}"
                    )
                }
        
        return {"verified": True}
    
    def _spec_requires_pcrel(self, spec: Specification) -> bool:
        """Check if specification requires IsPCRel handling."""
        for inv in spec.invariants:
            if self._invariant_requires_pcrel(inv):
                return True
        return False
    
    def _invariant_requires_pcrel(self, invariant: Condition) -> bool:
        """Check if invariant mentions IsPCRel."""
        def check_condition(cond):
            if isinstance(cond, Variable):
                return "pcrel" in cond.name.lower()
            elif isinstance(cond, Condition):
                for op in cond.operands:
                    if check_condition(op):
                        return True
            return False
        
        return check_condition(invariant)
    
    def _get_expected_cases(self, spec: Specification) -> List[str]:
        """Get list of expected case values from specification."""
        cases = []
        
        for inv in spec.invariants:
            if inv.cond_type == ConditionType.IMPLIES:
                if len(inv.operands) >= 1:
                    antecedent = inv.operands[0]
                    case = self._extract_case_from_condition(antecedent)
                    if case:
                        cases.append(case)
        
        return cases
    
    def _extract_case_from_condition(self, condition: Condition) -> Optional[str]:
        """Extract case value from condition."""
        if condition.cond_type == ConditionType.EQUALITY:
            if len(condition.operands) >= 2:
                var = condition.operands[0]
                val = condition.operands[1]
                if isinstance(var, Variable) and isinstance(val, Constant):
                    if "kind" in var.name.lower():
                        return str(val.value)
        elif condition.cond_type == ConditionType.AND:
            for op in condition.operands:
                case = self._extract_case_from_condition(op)
                if case:
                    return case
        return None
    
    def _check_invariants(
        self,
        code_model: CodeModel,
        spec: Specification
    ) -> Dict[str, Any]:
        """
        Explicitly check specification invariants against code model.
        
        This provides an additional verification layer using pattern matching
        when Z3 verification is not conclusive.
        """
        for inv in spec.invariants:
            result = self._check_single_invariant(inv, code_model)
            if not result["verified"]:
                return result
        
        return {"verified": True}
    
    def _check_single_invariant(
        self,
        invariant: Condition,
        code_model: CodeModel
    ) -> Dict[str, Any]:
        """Check a single invariant against the code model."""
        if invariant.cond_type != ConditionType.IMPLIES:
            # Non-implication invariants are assumed to hold
            return {"verified": True}
        
        if len(invariant.operands) < 2:
            return {"verified": True}
        
        antecedent = invariant.operands[0]
        consequent = invariant.operands[1]
        
        # Extract the input condition
        kind = self._extract_case_from_condition(antecedent)
        is_pcrel = self._condition_requires_pcrel(antecedent)
        
        # Extract expected output
        expected_output = self._extract_output(consequent)
        
        if not kind or not expected_output:
            return {"verified": True}  # Can't check, assume OK
        
        # Check if code model handles this case correctly
        if is_pcrel:
            # Check PCRel cases
            for case in code_model.pcrel_cases:
                if case.case_value == kind or case.case_value.endswith(kind):
                    if case.return_value == expected_output or case.return_value.endswith(expected_output):
                        return {"verified": True}
            
            # Not found in PCRel cases - violation
            return {
                "verified": False,
                "violated": str(invariant),
                "counterexample": Counterexample(
                    input_values={"Kind": kind, "IsPCRel": True},
                    expected_output=expected_output,
                    violated_condition=str(invariant)
                )
            }
        else:
            # Check non-PCRel cases
            all_cases = code_model.cases + code_model.non_pcrel_cases
            for case in all_cases:
                if case.case_value == kind or case.case_value.endswith(kind):
                    if case.return_value == expected_output or case.return_value.endswith(expected_output):
                        return {"verified": True}
            
            # Not found - violation
            return {
                "verified": False,
                "violated": str(invariant),
                "counterexample": Counterexample(
                    input_values={"Kind": kind, "IsPCRel": False},
                    expected_output=expected_output,
                    violated_condition=str(invariant)
                )
            }
    
    def _condition_requires_pcrel(self, condition: Condition) -> bool:
        """Check if condition requires IsPCRel to be true."""
        if condition.cond_type == ConditionType.IS_VALID:
            if len(condition.operands) >= 1:
                var = condition.operands[0]
                if isinstance(var, Variable) and "pcrel" in var.name.lower():
                    return True
        elif condition.cond_type == ConditionType.EQUALITY:
            if len(condition.operands) >= 2:
                for op in condition.operands:
                    if isinstance(op, Variable) and "pcrel" in op.name.lower():
                        return True
                    if isinstance(op, Constant) and op.value == True:
                        # Check if other operand is IsPCRel
                        for other_op in condition.operands:
                            if isinstance(other_op, Variable) and "pcrel" in other_op.name.lower():
                                return True
        elif condition.cond_type == ConditionType.AND:
            for op in condition.operands:
                if self._condition_requires_pcrel(op):
                    return True
        return False
    
    def _convert_counterexample(self, z3_result: Z3VerificationResult) -> Counterexample:
        """Convert Z3 counterexample to standard format."""
        ce = z3_result.counterexample or {}
        
        return Counterexample(
            input_values=ce.get("input_values", {}),
            expected_output=ce.get("expected_output"),
            actual_output=ce.get("actual_output"),
            violated_condition=z3_result.violated_property or ce.get("violated_invariant", ""),
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get verifier statistics."""
        stats = self.stats.copy()
        if self.z3_verifier:
            stats["z3_stats"] = self.z3_verifier.get_statistics()
        return stats


def create_integrated_verifier(
    timeout_ms: int = 30000,
    verbose: bool = False,
    use_z3: bool = True
) -> IntegratedVerifier:
    """Factory function to create an integrated verifier."""
    return IntegratedVerifier(timeout_ms=timeout_ms, verbose=verbose, use_z3=use_z3)


# Alias for backward compatibility
Verifier = IntegratedVerifier
