"""
Counterexample-Guided Neural Repair (CGNR) engine.
Core algorithm for VEGA-Verified that combines verification with neural repair.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time
import re

from ..specification.spec_language import Specification
from ..verification.verifier import Verifier, VerificationResult, VerificationStatus, Counterexample
from .fault_loc import FaultLocalizer, FaultLocation


class RepairStatus(Enum):
    """Status of repair attempt."""
    SUCCESS = "success"           # Repair succeeded, code verified
    PARTIAL = "partial"           # Some improvement but not fully verified
    FAILED = "failed"             # Repair failed
    MAX_ITERATIONS = "max_iter"   # Hit iteration limit


@dataclass
class RepairContext:
    """
    Context for neural repair.
    Contains all information needed by the repair model.
    """
    original_code: str
    counterexample: Counterexample
    fault_location: FaultLocation
    specification: Specification
    repair_history: List['RepairAttempt'] = field(default_factory=list)
    
    def to_prompt(self) -> str:
        """Convert to prompt for repair model."""
        return f"""[REPAIR TASK]
Function: {self.specification.function_name}

Original code with bug:
```cpp
{self.original_code}
```

Counterexample (inputs that cause wrong behavior):
- Input values: {self.counterexample.input_values}
- Expected output: {self.counterexample.expected_output}
- Actual output: {self.counterexample.actual_output}

Fault location (line {self.fault_location.line}):
{self.fault_location.statement}
Reason: {self.fault_location.reason}

Violated specification:
{self.counterexample.violated_condition}

Previous repair attempts: {len(self.repair_history)}

Generate the corrected code that satisfies the specification:
```cpp
"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_code": self.original_code,
            "counterexample": self.counterexample.to_dict() if self.counterexample else None,
            "fault_location": {
                "line": self.fault_location.line,
                "statement": self.fault_location.statement,
                "suspiciousness": self.fault_location.suspiciousness,
            } if self.fault_location else None,
            "specification": self.specification.function_name,
            "repair_attempts": len(self.repair_history),
        }


@dataclass
class RepairAttempt:
    """Record of a single repair attempt."""
    iteration: int
    code: str
    counterexample: Optional[Counterexample]
    verification_result: VerificationResult
    repair_strategy: str = ""
    time_ms: float = 0.0


@dataclass
class RepairResult:
    """Result of CGNR repair process."""
    status: RepairStatus
    repaired_code: str
    original_code: str
    
    # Verification result of final code
    final_verification: Optional[VerificationResult] = None
    
    # History of repair attempts
    attempts: List[RepairAttempt] = field(default_factory=list)
    
    # Timing
    total_time_ms: float = 0.0
    iterations: int = 0
    
    def is_successful(self) -> bool:
        """Check if repair was successful."""
        return self.status == RepairStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "is_successful": self.is_successful(),
            "verification_status": self.final_verification.status.value if self.final_verification else None,
        }


class CGNREngine:
    """
    Counterexample-Guided Neural Repair Engine.
    
    Main CGNR algorithm:
    1. Verify initial code against specification
    2. If verified, return success
    3. If failed, extract counterexample
    4. Localize fault using counterexample
    5. Build repair context
    6. Generate repair candidates using neural model
    7. Select best candidate
    8. Loop until verified or max iterations
    """
    
    DEFAULT_MAX_ITERATIONS = 5
    DEFAULT_BEAM_SIZE = 5
    
    def __init__(
        self,
        verifier: Optional[Verifier] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        beam_size: int = DEFAULT_BEAM_SIZE,
        use_neural_model: bool = True,
        verbose: bool = False
    ):
        self.verifier = verifier or Verifier()
        self.fault_localizer = FaultLocalizer()
        self.max_iterations = max_iterations
        self.beam_size = beam_size
        self.use_neural_model = use_neural_model
        self.verbose = verbose
        
        # Repair model (placeholder - would load actual model)
        self.repair_model = None
        if use_neural_model:
            self._init_repair_model()
    
    def _init_repair_model(self) -> None:
        """Initialize the neural repair model."""
        # In full implementation, would load fine-tuned model
        # For now, use rule-based repairs as fallback
        self.repair_model = RuleBasedRepairModel()
    
    def repair(
        self,
        code: str,
        spec: Specification,
        statements: Optional[List[Dict[str, Any]]] = None
    ) -> RepairResult:
        """
        Main CGNR repair algorithm.
        
        Args:
            code: Code to repair
            spec: Specification to satisfy
            statements: Optional pre-parsed statements
            
        Returns:
            RepairResult with repaired code and status
        """
        start_time = time.time()
        
        current_code = code
        history: List[RepairAttempt] = []
        
        for iteration in range(self.max_iterations):
            iter_start = time.time()
            
            if self.verbose:
                print(f"\n--- CGNR Iteration {iteration + 1} ---")
            
            # Step 1: Verify current code
            result = self.verifier.verify(current_code, spec, statements)
            
            if result.status == VerificationStatus.VERIFIED:
                # Success!
                return RepairResult(
                    status=RepairStatus.SUCCESS,
                    repaired_code=current_code,
                    original_code=code,
                    final_verification=result,
                    attempts=history,
                    total_time_ms=(time.time() - start_time) * 1000,
                    iterations=iteration + 1
                )
            
            if result.counterexample is None:
                if self.verbose:
                    print("No counterexample available, cannot proceed")
                break
            
            # Step 2: Localize fault
            fault_locs = self.fault_localizer.localize(
                current_code,
                result.counterexample
            )
            
            if not fault_locs:
                if self.verbose:
                    print("Could not localize fault")
                break
            
            primary_fault = fault_locs[0]
            
            if self.verbose:
                print(f"Counterexample: {result.counterexample.input_values}")
                print(f"Fault location: {primary_fault}")
            
            # Step 3: Build repair context
            context = RepairContext(
                original_code=current_code,
                counterexample=result.counterexample,
                fault_location=primary_fault,
                specification=spec,
                repair_history=history[-3:]  # Last 3 attempts
            )
            
            # Step 4: Generate repair candidates
            candidates = self._generate_repairs(context)
            
            if not candidates:
                if self.verbose:
                    print("No repair candidates generated")
                break
            
            # Step 5: Select best candidate
            best_candidate = self._select_best(candidates, spec)
            
            # Record attempt
            history.append(RepairAttempt(
                iteration=iteration + 1,
                code=best_candidate,
                counterexample=result.counterexample,
                verification_result=result,
                repair_strategy=getattr(self.repair_model, 'last_strategy', 'unknown'),
                time_ms=(time.time() - iter_start) * 1000
            ))
            
            current_code = best_candidate
        
        # Exhausted iterations or couldn't continue
        final_result = self.verifier.verify(current_code, spec, statements)
        
        status = RepairStatus.MAX_ITERATIONS
        if final_result.status == VerificationStatus.VERIFIED:
            status = RepairStatus.SUCCESS
        elif current_code != code:
            status = RepairStatus.PARTIAL
        else:
            status = RepairStatus.FAILED
        
        return RepairResult(
            status=status,
            repaired_code=current_code,
            original_code=code,
            final_verification=final_result,
            attempts=history,
            total_time_ms=(time.time() - start_time) * 1000,
            iterations=len(history)
        )
    
    def _generate_repairs(self, context: RepairContext) -> List[str]:
        """Generate repair candidates."""
        candidates = []
        
        # Use neural model if available
        if self.repair_model:
            candidates = self.repair_model.generate(context, self.beam_size)
        
        return candidates
    
    def _select_best(
        self,
        candidates: List[str],
        spec: Specification
    ) -> str:
        """Select best repair candidate."""
        # Try to verify each candidate
        for candidate in candidates:
            result = self.verifier.verify(candidate, spec)
            if result.status == VerificationStatus.VERIFIED:
                return candidate
        
        # If none verified, return first candidate
        return candidates[0] if candidates else ""


class RuleBasedRepairModel:
    """
    Rule-based repair model as fallback when neural model is not available.
    Implements common repair patterns for compiler backend code.
    """
    
    def __init__(self):
        self.last_strategy = ""
        
        # Common repair patterns
        self.repair_rules = [
            ("missing_pcrel_check", self._repair_missing_pcrel),
            ("wrong_reloc_size", self._repair_wrong_size),
            ("missing_case", self._repair_missing_case),
            ("wrong_return_value", self._repair_wrong_return),
        ]
    
    def generate(self, context: RepairContext, beam_size: int = 5) -> List[str]:
        """Generate repair candidates using rules."""
        candidates = []
        
        for rule_name, rule_fn in self.repair_rules:
            repaired = rule_fn(context)
            if repaired and repaired != context.original_code:
                candidates.append(repaired)
                self.last_strategy = rule_name
                
                if len(candidates) >= beam_size:
                    break
        
        # If no rules applied, try generic repairs
        if not candidates:
            candidates = self._generic_repairs(context)
        
        return candidates[:beam_size]
    
    def _repair_missing_pcrel(self, context: RepairContext) -> Optional[str]:
        """Repair missing PC-relative check."""
        code = context.original_code
        ce = context.counterexample
        
        # Check if IsPCRel is involved
        if 'IsPCRel' not in str(ce.input_values):
            return None
        
        # Find return statement that needs fixing
        fault_line = context.fault_location.statement
        
        if 'return' in fault_line and '?' not in fault_line:
            # Pattern: return X; -> return IsPCRel ? Y : X;
            match = re.search(r'return\s+(\w+::\w+)', fault_line)
            if match:
                current_val = match.group(1)
                
                # Generate PC-relative variant
                if '_32' in current_val:
                    pcrel_val = current_val.replace('_32', '_PC32')
                elif '_64' in current_val:
                    pcrel_val = current_val.replace('_64', '_PC64')
                else:
                    return None
                
                new_line = f"return IsPCRel ? {pcrel_val} : {current_val};"
                return code.replace(fault_line, new_line)
        
        return None
    
    def _repair_wrong_size(self, context: RepairContext) -> Optional[str]:
        """Repair wrong relocation size."""
        code = context.original_code
        ce = context.counterexample
        fault = context.fault_location
        
        # Look for size mismatch patterns
        expected = str(ce.expected_output) if ce.expected_output else ""
        actual = str(ce.actual_output) if ce.actual_output else ""
        
        # Try to fix size-related issues
        size_fixes = [
            ('_32', '_64'),
            ('_64', '_32'),
            ('_16', '_32'),
            ('_8', '_16'),
        ]
        
        for old_size, new_size in size_fixes:
            if old_size in fault.statement and new_size in expected:
                new_stmt = fault.statement.replace(old_size, new_size)
                return code.replace(fault.statement, new_stmt)
        
        return None
    
    def _repair_missing_case(self, context: RepairContext) -> Optional[str]:
        """Repair missing switch case."""
        code = context.original_code
        ce = context.counterexample
        
        # Check if a case is missing
        if 'Fixup_kind' in ce.input_values or 'Fixup.kind' in str(ce.input_values):
            # Find the switch statement
            switch_match = re.search(r'switch\s*\([^)]+\)\s*\{', code)
            if switch_match:
                # Try to add missing case before default
                default_match = re.search(r'default\s*:', code)
                if default_match:
                    missing_case = ce.input_values.get('Fixup_kind', 
                                  ce.input_values.get('Fixup.kind', ''))
                    if missing_case:
                        new_case = f"case {missing_case}: return /* TODO */;\n  "
                        insert_pos = default_match.start()
                        return code[:insert_pos] + new_case + code[insert_pos:]
        
        return None
    
    def _repair_wrong_return(self, context: RepairContext) -> Optional[str]:
        """Repair wrong return value."""
        code = context.original_code
        ce = context.counterexample
        fault = context.fault_location
        
        expected = str(ce.expected_output) if ce.expected_output else ""
        
        if expected and 'return' in fault.statement:
            # Try direct replacement
            match = re.search(r'return\s+(.+?)\s*;', fault.statement)
            if match:
                current_return = match.group(1)
                new_stmt = fault.statement.replace(current_return, expected)
                return code.replace(fault.statement, new_stmt)
        
        return None
    
    def _generic_repairs(self, context: RepairContext) -> List[str]:
        """Generate generic repair attempts."""
        candidates = []
        code = context.original_code
        fault = context.fault_location
        
        # Try commenting out suspicious line
        if fault.statement:
            commented = code.replace(fault.statement, f"// FIXME: {fault.statement}")
            candidates.append(commented)
        
        # Try adding assertion
        if fault.statement:
            assertion = f"assert(/* check condition */);\n  {fault.statement}"
            candidates.append(code.replace(fault.statement, assertion))
        
        return candidates
