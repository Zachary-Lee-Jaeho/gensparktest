"""
Main verifier for VEGA-Verified.
Orchestrates specification checking, VC generation, and SMT solving.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import time

from ..specification.spec_language import Specification, Condition
from .vcgen import VCGenerator, VerificationCondition
from .smt_solver import SMTSolver, SMTResult, SMTModel


class VerificationStatus(Enum):
    """Status of verification."""
    VERIFIED = "verified"     # All VCs passed
    FAILED = "failed"         # At least one VC failed
    TIMEOUT = "timeout"       # Solver timed out
    UNKNOWN = "unknown"       # Solver returned unknown
    ERROR = "error"           # Internal error


@dataclass
class Counterexample:
    """
    Counterexample from failed verification.
    
    Contains:
    - Input values that violate the specification
    - Expected vs actual output
    - Which condition was violated
    - Execution trace (if available)
    """
    input_values: Dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    actual_output: Any = None
    violated_condition: str = ""
    trace: List[str] = field(default_factory=list)
    
    # Additional context for repair
    fault_location: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_values": self.input_values,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "violated_condition": self.violated_condition,
            "trace": self.trace,
            "fault_location": self.fault_location,
        }
    
    def __str__(self) -> str:
        lines = ["Counterexample:"]
        lines.append(f"  Inputs: {self.input_values}")
        if self.expected_output:
            lines.append(f"  Expected: {self.expected_output}")
        if self.actual_output:
            lines.append(f"  Actual: {self.actual_output}")
        lines.append(f"  Violated: {self.violated_condition}")
        return "\n".join(lines)


@dataclass
class VerificationResult:
    """
    Result of verification.
    
    Contains:
    - Overall status
    - List of verified properties
    - Counterexample if failed
    - Timing information
    """
    status: VerificationStatus
    verified_properties: List[str] = field(default_factory=list)
    failed_properties: List[str] = field(default_factory=list)
    counterexample: Optional[Counterexample] = None
    
    # Timing
    time_ms: float = 0.0
    vcgen_time_ms: float = 0.0
    solve_time_ms: float = 0.0
    
    # Detailed results per VC
    vc_results: Dict[str, str] = field(default_factory=dict)
    
    def is_verified(self) -> bool:
        """Check if verification succeeded."""
        return self.status == VerificationStatus.VERIFIED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "verified_properties": self.verified_properties,
            "failed_properties": self.failed_properties,
            "counterexample": self.counterexample.to_dict() if self.counterexample else None,
            "time_ms": self.time_ms,
            "vcgen_time_ms": self.vcgen_time_ms,
            "solve_time_ms": self.solve_time_ms,
            "vc_results": self.vc_results,
        }


class Verifier:
    """
    Main verification engine for VEGA-Verified.
    
    Workflow:
    1. Parse code and build CFG
    2. Generate verification conditions from code + spec
    3. Check each VC using SMT solver
    4. If SAT (counterexample found), extract and return
    5. If UNSAT (all VCs valid), return verified
    """
    
    def __init__(
        self,
        timeout_ms: int = 30000,
        incremental: bool = True,
        verbose: bool = False
    ):
        self.timeout_ms = timeout_ms
        self.incremental = incremental
        self.verbose = verbose
        
        # Initialize components
        self.vcgen = VCGenerator()
        self.solver = SMTSolver(timeout_ms=timeout_ms, incremental=incremental)
    
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
            statements: Optional pre-parsed statements
            
        Returns:
            VerificationResult with status and details
        """
        start_time = time.time()
        
        result = VerificationResult(status=VerificationStatus.UNKNOWN)
        
        try:
            # Step 1: Generate verification conditions
            vcgen_start = time.time()
            vcs = self.vcgen.generate(code, spec, statements)
            result.vcgen_time_ms = (time.time() - vcgen_start) * 1000
            
            if self.verbose:
                print(f"Generated {len(vcs)} verification conditions")
            
            # Step 2: Check each VC
            solve_start = time.time()
            
            all_verified = True
            first_counterexample = None
            
            for vc in vcs:
                vc_result = self._check_vc(vc)
                result.vc_results[vc.name] = vc_result["status"]
                
                if vc_result["status"] == "verified":
                    result.verified_properties.append(vc.name)
                elif vc_result["status"] == "failed":
                    result.failed_properties.append(vc.name)
                    all_verified = False
                    
                    if first_counterexample is None:
                        first_counterexample = self._extract_counterexample(
                            vc, vc_result.get("model")
                        )
                elif vc_result["status"] == "timeout":
                    all_verified = False
                    result.status = VerificationStatus.TIMEOUT
                    break
            
            result.solve_time_ms = (time.time() - solve_start) * 1000
            
            # Step 3: Determine overall status
            if all_verified and result.status != VerificationStatus.TIMEOUT:
                result.status = VerificationStatus.VERIFIED
            elif result.failed_properties:
                result.status = VerificationStatus.FAILED
                result.counterexample = first_counterexample
            
        except Exception as e:
            result.status = VerificationStatus.ERROR
            result.failed_properties.append(f"Error: {str(e)}")
        
        result.time_ms = (time.time() - start_time) * 1000
        return result
    
    def _check_vc(self, vc: VerificationCondition) -> Dict[str, Any]:
        """
        Check a single verification condition.
        
        A VC is valid iff its negation is unsatisfiable.
        """
        if self.verbose:
            print(f"Checking VC: {vc.name}")
        
        self.solver.reset()
        
        # Check validity: VC is valid iff NOT(VC) is UNSAT
        smt_result, model = self.solver.check_valid(vc.formula)
        
        if smt_result == SMTResult.UNSAT:
            # VC is valid (negation unsatisfiable)
            return {"status": "verified"}
        elif smt_result == SMTResult.SAT:
            # VC is invalid (counterexample found)
            return {"status": "failed", "model": model}
        elif smt_result == SMTResult.TIMEOUT:
            return {"status": "timeout"}
        else:
            return {"status": "unknown"}
    
    def _extract_counterexample(
        self,
        vc: VerificationCondition,
        model: Optional[SMTModel]
    ) -> Counterexample:
        """Extract counterexample from SMT model."""
        ce = Counterexample()
        ce.violated_condition = vc.name
        
        if model:
            ce.input_values = model.assignments.copy()
            
            # Try to identify expected vs actual from model
            if "result" in model.assignments:
                ce.actual_output = model.assignments["result"]
        
        ce.trace.append(f"VC source: {vc.source}")
        
        return ce
    
    def verify_function(
        self,
        function_code: str,
        function_name: str,
        spec: Optional[Specification] = None,
        infer_spec: bool = True,
        references: Optional[List[Tuple[str, str]]] = None
    ) -> VerificationResult:
        """
        High-level function verification with optional spec inference.
        
        Args:
            function_code: Code of the function
            function_name: Name of the function
            spec: Optional pre-defined specification
            infer_spec: Whether to infer spec if not provided
            references: Reference implementations for inference
            
        Returns:
            VerificationResult
        """
        if spec is None and infer_spec and references:
            # Infer specification from references
            from ..specification import SpecificationInferrer
            inferrer = SpecificationInferrer()
            spec = inferrer.infer(function_name, references)
        
        if spec is None:
            # Create minimal spec
            spec = Specification(function_name=function_name)
        
        return self.verify(function_code, spec)
    
    def verify_incrementally(
        self,
        code: str,
        specs: List[Specification]
    ) -> List[VerificationResult]:
        """
        Verify code against multiple specifications incrementally.
        
        Useful for comparing different versions or checking multiple properties.
        
        Args:
            code: Source code
            specs: List of specifications to check
            
        Returns:
            List of results, one per specification
        """
        results = []
        
        for spec in specs:
            result = self.verify(code, spec)
            results.append(result)
            
            # Early exit if any verification fails (optional)
            # if not result.is_verified():
            #     break
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return {
            "solver_stats": self.solver.get_statistics(),
        }


class BatchVerifier:
    """
    Verifier for batch processing multiple functions.
    Useful for comparing VEGA vs VEGA-Verified on entire modules.
    """
    
    def __init__(self, verifier: Optional[Verifier] = None):
        self.verifier = verifier or Verifier()
        self.results: Dict[str, VerificationResult] = {}
    
    def verify_batch(
        self,
        functions: Dict[str, Tuple[str, Specification]],
        parallel: bool = False
    ) -> Dict[str, VerificationResult]:
        """
        Verify multiple functions.
        
        Args:
            functions: Dict of function_name -> (code, specification)
            parallel: Whether to verify in parallel (not implemented yet)
            
        Returns:
            Dict of function_name -> VerificationResult
        """
        self.results.clear()
        
        for func_name, (code, spec) in functions.items():
            result = self.verifier.verify(code, spec)
            self.results[func_name] = result
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of batch verification."""
        total = len(self.results)
        verified = sum(1 for r in self.results.values() if r.is_verified())
        failed = sum(1 for r in self.results.values() if r.status == VerificationStatus.FAILED)
        timeout = sum(1 for r in self.results.values() if r.status == VerificationStatus.TIMEOUT)
        
        total_time = sum(r.time_ms for r in self.results.values())
        
        return {
            "total_functions": total,
            "verified": verified,
            "failed": failed,
            "timeout": timeout,
            "verification_rate": verified / total if total > 0 else 0.0,
            "total_time_ms": total_time,
            "avg_time_ms": total_time / total if total > 0 else 0.0,
        }
