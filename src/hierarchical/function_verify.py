"""
Level 1: Function-level Verification.

Verifies individual functions against their specifications.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time

from ..verification.verifier import Verifier, VerificationResult, VerificationStatus
from ..specification.spec_language import Specification
from ..repair.cgnr import CGNREngine, RepairResult


class FunctionVerificationStatus(Enum):
    """Status of function verification."""
    PENDING = "pending"
    VERIFIED = "verified"
    REPAIRED = "repaired"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class FunctionVerificationResult:
    """Result of function-level verification."""
    function_name: str
    module_name: str
    status: FunctionVerificationStatus
    verification_result: Optional[VerificationResult] = None
    repair_result: Optional[RepairResult] = None
    original_code: str = ""
    verified_code: str = ""
    specification: Optional[Specification] = None
    time_ms: float = 0.0
    iterations: int = 0
    messages: List[str] = field(default_factory=list)
    
    def is_successful(self) -> bool:
        """Check if verification (with or without repair) succeeded."""
        return self.status in (FunctionVerificationStatus.VERIFIED, 
                               FunctionVerificationStatus.REPAIRED)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "module_name": self.module_name,
            "status": self.status.value,
            "time_ms": self.time_ms,
            "iterations": self.iterations,
            "messages": self.messages,
            "verification_result": self.verification_result.to_dict() if self.verification_result else None,
            "repair_result": self.repair_result.to_dict() if self.repair_result else None,
        }
    
    def __str__(self) -> str:
        status_emoji = {
            FunctionVerificationStatus.VERIFIED: "✓",
            FunctionVerificationStatus.REPAIRED: "⚡",
            FunctionVerificationStatus.FAILED: "✗",
            FunctionVerificationStatus.TIMEOUT: "⏱",
            FunctionVerificationStatus.PENDING: "○",
            FunctionVerificationStatus.ERROR: "!",
        }
        emoji = status_emoji.get(self.status, "?")
        return f"[{emoji}] {self.function_name}: {self.status.value} ({self.time_ms:.1f}ms)"


class FunctionVerifier:
    """
    Level 1 Verifier: Function-level verification.
    
    Verifies individual functions against specifications,
    with optional CGNR repair for failures.
    """
    
    def __init__(
        self,
        verifier: Optional[Verifier] = None,
        cgnr_engine: Optional[CGNREngine] = None,
        enable_repair: bool = True,
        max_repair_iterations: int = 5,
        timeout_ms: int = 30000,
        verbose: bool = False
    ):
        """
        Initialize function verifier.
        
        Args:
            verifier: Verification engine (creates default if None)
            cgnr_engine: CGNR repair engine (creates default if None and repair enabled)
            enable_repair: Whether to attempt repair on verification failure
            max_repair_iterations: Maximum repair iterations
            timeout_ms: Verification timeout in milliseconds
            verbose: Enable verbose output
        """
        self.verifier = verifier or Verifier(timeout_ms=timeout_ms)
        self.enable_repair = enable_repair
        self.max_repair_iterations = max_repair_iterations
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        
        # Initialize CGNR if repair is enabled
        if enable_repair:
            self.cgnr_engine = cgnr_engine or CGNREngine(
                verifier=self.verifier,
                max_iterations=max_repair_iterations,
                verbose=verbose
            )
        else:
            self.cgnr_engine = None
        
        # Statistics
        self.stats = {
            "total_verified": 0,
            "total_repaired": 0,
            "total_failed": 0,
            "total_timeout": 0,
            "total_time_ms": 0.0,
        }
    
    def verify(
        self,
        code: str,
        spec: Specification,
        module_name: str = "",
        attempt_repair: Optional[bool] = None
    ) -> FunctionVerificationResult:
        """
        Verify a function against its specification.
        
        Args:
            code: Function source code
            spec: Function specification
            module_name: Name of the containing module
            attempt_repair: Override enable_repair setting
        
        Returns:
            FunctionVerificationResult with verification status and details
        """
        start_time = time.time()
        
        should_repair = attempt_repair if attempt_repair is not None else self.enable_repair
        
        result = FunctionVerificationResult(
            function_name=spec.function_name,
            module_name=module_name or spec.module,
            status=FunctionVerificationStatus.PENDING,
            original_code=code,
            specification=spec
        )
        
        try:
            if self.verbose:
                print(f"[L1] Verifying function: {spec.function_name}")
            
            # Step 1: Verify against specification
            ver_result = self.verifier.verify(code, spec)
            result.verification_result = ver_result
            result.iterations = 1
            
            if ver_result.is_verified():
                # Verification succeeded
                result.status = FunctionVerificationStatus.VERIFIED
                result.verified_code = code
                result.messages.append("Direct verification successful")
                self.stats["total_verified"] += 1
                
                if self.verbose:
                    print(f"  [✓] Verified directly")
            
            elif ver_result.status == VerificationStatus.TIMEOUT:
                result.status = FunctionVerificationStatus.TIMEOUT
                result.messages.append(f"Verification timed out after {self.timeout_ms}ms")
                self.stats["total_timeout"] += 1
                
                if self.verbose:
                    print(f"  [⏱] Timeout")
            
            elif should_repair and self.cgnr_engine:
                # Step 2: Attempt repair using CGNR
                if self.verbose:
                    print(f"  [!] Verification failed, attempting repair...")
                
                repair_result = self.cgnr_engine.repair(code, spec)
                result.repair_result = repair_result
                result.iterations = 1 + repair_result.iterations
                
                if repair_result.is_successful():
                    result.status = FunctionVerificationStatus.REPAIRED
                    result.verified_code = repair_result.repaired_code
                    result.messages.append(
                        f"Repair successful after {repair_result.iterations} iterations"
                    )
                    self.stats["total_repaired"] += 1
                    
                    if self.verbose:
                        print(f"  [⚡] Repaired in {repair_result.iterations} iterations")
                else:
                    result.status = FunctionVerificationStatus.FAILED
                    result.messages.append("Repair failed")
                    if repair_result.counterexample:
                        result.messages.append(
                            f"Final counterexample: {repair_result.counterexample}"
                        )
                    self.stats["total_failed"] += 1
                    
                    if self.verbose:
                        print(f"  [✗] Repair failed")
            else:
                # No repair, mark as failed
                result.status = FunctionVerificationStatus.FAILED
                if ver_result.counterexample:
                    result.messages.append(
                        f"Counterexample found: {ver_result.counterexample}"
                    )
                self.stats["total_failed"] += 1
                
                if self.verbose:
                    print(f"  [✗] Verification failed (repair disabled)")
        
        except Exception as e:
            result.status = FunctionVerificationStatus.ERROR
            result.messages.append(f"Error: {str(e)}")
            
            if self.verbose:
                print(f"  [!] Error: {e}")
        
        finally:
            result.time_ms = (time.time() - start_time) * 1000
            self.stats["total_time_ms"] += result.time_ms
        
        return result
    
    def verify_batch(
        self,
        functions: List[tuple],  # List of (code, spec, module_name) tuples
        parallel: bool = False
    ) -> List[FunctionVerificationResult]:
        """
        Verify multiple functions.
        
        Args:
            functions: List of (code, spec, module_name) tuples
            parallel: Whether to verify in parallel (not yet implemented)
        
        Returns:
            List of verification results
        """
        results = []
        
        for i, (code, spec, module_name) in enumerate(functions):
            if self.verbose:
                print(f"\n[{i+1}/{len(functions)}]")
            
            result = self.verify(code, spec, module_name)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        total = (self.stats["total_verified"] + self.stats["total_repaired"] + 
                 self.stats["total_failed"] + self.stats["total_timeout"])
        
        return {
            **self.stats,
            "total_processed": total,
            "success_rate": (self.stats["total_verified"] + self.stats["total_repaired"]) / max(total, 1),
            "direct_verification_rate": self.stats["total_verified"] / max(total, 1),
            "repair_rate": self.stats["total_repaired"] / max(total, 1),
            "avg_time_ms": self.stats["total_time_ms"] / max(total, 1),
        }
    
    def reset_statistics(self) -> None:
        """Reset verification statistics."""
        self.stats = {
            "total_verified": 0,
            "total_repaired": 0,
            "total_failed": 0,
            "total_timeout": 0,
            "total_time_ms": 0.0,
        }
