"""
Integrated CGNR (Counterexample-Guided Neural Repair) Engine.

This module provides the main repair loop for VEGA-Verified:
1. Verify code against specification
2. If failed, extract counterexample
3. Generate repair candidates using neural/template models
4. Verify repairs
5. Repeat until verified or max iterations

The integrated CGNR combines:
- Z3-based verification for precise counterexamples
- Neural repair model for intelligent code fixes
- Template-based repair for common patterns
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time

from .neural_repair import (
    NeuralRepairModel, HybridRepairModel, RepairCandidate, 
    RepairContext, RepairPattern, create_neural_repair_model
)
from ..verification.integrated_verifier import IntegratedVerifier, create_integrated_verifier
from ..verification.verifier import VerificationResult, VerificationStatus, Counterexample
from ..specification.spec_language import Specification


class CGNRStatus(Enum):
    """Status of CGNR repair."""
    SUCCESS = "success"           # Repair found and verified
    PARTIAL = "partial"           # Some improvement but not fully verified
    FAILED = "failed"             # Could not find valid repair
    MAX_ITERATIONS = "max_iters"  # Hit iteration limit
    TIMEOUT = "timeout"           # Time limit exceeded


@dataclass
class CGNRAttempt:
    """Record of a single repair attempt."""
    iteration: int
    candidate: RepairCandidate
    verification_result: VerificationResult
    time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "candidate": self.candidate.to_dict(),
            "verified": self.verification_result.is_verified(),
            "time_ms": self.time_ms,
        }


@dataclass
class CGNRResult:
    """Result of CGNR repair process."""
    status: CGNRStatus
    original_code: str
    repaired_code: Optional[str] = None
    attempts: List[CGNRAttempt] = field(default_factory=list)
    final_verification: Optional[VerificationResult] = None
    total_time_ms: float = 0.0
    iterations: int = 0
    
    def is_successful(self) -> bool:
        """Check if repair was successful."""
        return self.status == CGNRStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "iterations": self.iterations,
            "successful": self.is_successful(),
            "total_time_ms": self.total_time_ms,
            "attempts": [a.to_dict() for a in self.attempts],
            "final_verified": self.final_verification.is_verified() if self.final_verification else False,
        }


class IntegratedCGNREngine:
    """
    Integrated CGNR engine combining verification and repair.
    
    Algorithm:
    1. Verify(code, spec) → result
    2. If verified: return SUCCESS
    3. Extract counterexample from result
    4. Build repair context
    5. Generate repair candidates
    6. For each candidate:
       a. Verify(candidate, spec)
       b. If verified: return SUCCESS with repaired code
    7. If no candidate verifies:
       a. Pick best candidate (highest confidence or most progress)
       b. Use as new code
       c. Go to step 1
    8. If max iterations: return MAX_ITERATIONS
    """
    
    DEFAULT_MAX_ITERATIONS = 5
    DEFAULT_NUM_CANDIDATES = 5
    
    def __init__(
        self,
        verifier: Optional[IntegratedVerifier] = None,
        repair_model: Optional[HybridRepairModel] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        timeout_ms: int = 60000,
        verbose: bool = False
    ):
        """
        Initialize CGNR engine.
        
        Args:
            verifier: Verifier instance (created if None)
            repair_model: Repair model (created if None)
            max_iterations: Maximum repair iterations
            num_candidates: Number of candidates per iteration
            timeout_ms: Total timeout for repair process
            verbose: Print verbose output
        """
        self.verifier = verifier or create_integrated_verifier(verbose=verbose)
        self.repair_model = repair_model or create_neural_repair_model(verbose=verbose)
        self.max_iterations = max_iterations
        self.num_candidates = num_candidates
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            "total_repairs": 0,
            "successful_repairs": 0,
            "failed_repairs": 0,
            "total_iterations": 0,
            "total_time_ms": 0.0,
            "pattern_counts": {},
        }
    
    def repair(
        self,
        code: str,
        spec: Specification,
        statements: Optional[List[Dict[str, Any]]] = None
    ) -> CGNRResult:
        """
        Attempt to repair code to satisfy specification.
        
        Args:
            code: Original code to repair
            spec: Specification to satisfy
            statements: Optional pre-parsed statements
            
        Returns:
            CGNRResult with repair status and repaired code if successful
        """
        self.stats["total_repairs"] += 1
        start_time = time.time()
        
        result = CGNRResult(
            status=CGNRStatus.FAILED,
            original_code=code
        )
        
        current_code = code
        repair_history: List[Dict[str, Any]] = []
        
        if self.verbose:
            print(f"Starting CGNR repair for {spec.function_name}")
            print(f"  Max iterations: {self.max_iterations}")
        
        for iteration in range(self.max_iterations):
            iteration_start = time.time()
            result.iterations = iteration + 1
            self.stats["total_iterations"] += 1
            
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.timeout_ms:
                result.status = CGNRStatus.TIMEOUT
                if self.verbose:
                    print(f"  Timeout after {elapsed_ms:.1f}ms")
                break
            
            if self.verbose:
                print(f"\n  Iteration {iteration + 1}")
            
            # Step 1: Verify current code
            verification_result = self.verifier.verify(current_code, spec)
            
            if verification_result.is_verified():
                # Success!
                result.status = CGNRStatus.SUCCESS
                result.repaired_code = current_code
                result.final_verification = verification_result
                self.stats["successful_repairs"] += 1
                
                if self.verbose:
                    print(f"    VERIFIED after {iteration + 1} iteration(s)")
                break
            
            if self.verbose:
                print(f"    Verification failed: {verification_result.failed_properties}")
                if verification_result.counterexample:
                    print(f"    Counterexample: {verification_result.counterexample}")
            
            # Step 2: Build repair context
            counterexample = verification_result.counterexample
            if counterexample is None:
                # No counterexample, create a generic one
                counterexample = Counterexample(
                    violated_condition="specification_violation"
                )
            
            context = RepairContext(
                original_code=current_code,
                counterexample={
                    "input_values": counterexample.input_values,
                    "expected_output": counterexample.expected_output,
                    "actual_output": counterexample.actual_output,
                },
                specification=spec,
                violated_property=counterexample.violated_condition,
                repair_history=repair_history
            )
            
            # Step 3: Generate repair candidates
            candidates = self.repair_model.repair(context, self.num_candidates)
            
            if self.verbose:
                print(f"    Generated {len(candidates)} repair candidates")
            
            if not candidates:
                if self.verbose:
                    print("    No repair candidates generated")
                continue
            
            # Step 4: Try each candidate
            best_candidate = None
            best_progress = -1
            
            for i, candidate in enumerate(candidates):
                if self.verbose:
                    print(f"    Trying candidate {i+1} (confidence: {candidate.confidence:.2f}, pattern: {candidate.pattern.value})")
                
                # Verify candidate
                candidate_result = self.verifier.verify(candidate.code, spec)
                
                attempt = CGNRAttempt(
                    iteration=iteration + 1,
                    candidate=candidate,
                    verification_result=candidate_result,
                    time_ms=(time.time() - iteration_start) * 1000
                )
                result.attempts.append(attempt)
                
                if candidate_result.is_verified():
                    # Found a valid repair!
                    result.status = CGNRStatus.SUCCESS
                    result.repaired_code = candidate.code
                    result.final_verification = candidate_result
                    self.stats["successful_repairs"] += 1
                    
                    # Track pattern
                    self.stats["pattern_counts"][candidate.pattern.value] = \
                        self.stats["pattern_counts"].get(candidate.pattern.value, 0) + 1
                    
                    if self.verbose:
                        print(f"    SUCCESS! Repair verified.")
                    break
                
                # Track progress (fewer failed properties = better)
                progress = len(verification_result.failed_properties) - len(candidate_result.failed_properties)
                if progress > best_progress:
                    best_progress = progress
                    best_candidate = candidate
            
            if result.status == CGNRStatus.SUCCESS:
                break
            
            # Step 5: Use best candidate for next iteration
            if best_candidate:
                current_code = best_candidate.code
                repair_history.append({
                    "iteration": iteration + 1,
                    "code": best_candidate.code,
                    "pattern": best_candidate.pattern.value,
                    "progress": best_progress,
                })
                
                if self.verbose:
                    print(f"    Using best candidate (progress: {best_progress})")
        
        # Finalize result
        result.total_time_ms = (time.time() - start_time) * 1000
        self.stats["total_time_ms"] += result.total_time_ms
        
        if result.status != CGNRStatus.SUCCESS:
            # Check if we made partial progress
            if result.attempts:
                last_attempt = result.attempts[-1]
                if len(last_attempt.verification_result.failed_properties) < \
                   len(self.verifier.verify(code, spec).failed_properties):
                    result.status = CGNRStatus.PARTIAL
                    result.repaired_code = last_attempt.candidate.code
            
            if result.status == CGNRStatus.FAILED:
                self.stats["failed_repairs"] += 1
        
        if self.verbose:
            print(f"\nCGNR completed: {result.status.value}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Time: {result.total_time_ms:.1f}ms")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get CGNR statistics."""
        stats = self.stats.copy()
        stats["verifier_stats"] = self.verifier.get_statistics()
        if hasattr(self.repair_model, 'neural_model'):
            stats["repair_model_stats"] = self.repair_model.neural_model.get_statistics()
        return stats


def create_cgnr_engine(
    max_iterations: int = 5,
    timeout_ms: int = 60000,
    verbose: bool = False
) -> IntegratedCGNREngine:
    """Factory function to create a CGNR engine."""
    return IntegratedCGNREngine(
        max_iterations=max_iterations,
        timeout_ms=timeout_ms,
        verbose=verbose
    )


# Alias for backward compatibility
CGNREngine = IntegratedCGNREngine


# Quick test
if __name__ == "__main__":
    from ..specification.spec_language import Condition, ConditionType, Variable, Constant
    
    # Test code with a bug (missing IsPCRel handling)
    buggy_code = """
    unsigned getRelocType(const MCFixup &Fixup, bool IsPCRel) const {
        unsigned Kind = Fixup.getTargetKind();
        
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
    
    # Specification requiring IsPCRel handling
    spec = Specification(
        function_name="getRelocType",
        module="ELFObjectWriter",
        preconditions=[
            Condition(ConditionType.IS_VALID, [Variable("Fixup")]),
        ],
        postconditions=[
            Condition(ConditionType.GREATER_EQUAL, [Variable("result"), Constant(0)]),
        ],
        invariants=[
            # FK_Data_4 + IsPCRel → R_RISCV_32_PCREL
            Condition(ConditionType.IMPLIES, [
                Condition(ConditionType.AND, [
                    Condition(ConditionType.EQUALITY, [Variable("Kind"), Constant("FK_Data_4")]),
                    Condition(ConditionType.EQUALITY, [Variable("IsPCRel"), Constant(True, const_type="bool")]),
                ]),
                Condition(ConditionType.EQUALITY, [Variable("result"), Constant("R_RISCV_32_PCREL")]),
            ]),
        ]
    )
    
    # Run CGNR
    engine = create_cgnr_engine(verbose=True)
    result = engine.repair(buggy_code, spec)
    
    print(f"\nResult: {result.status.value}")
    print(f"Iterations: {result.iterations}")
    if result.repaired_code:
        print("\nRepaired code:")
        print(result.repaired_code)
