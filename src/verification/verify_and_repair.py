"""
End-to-End Verify and Repair Pipeline for VEGA-Verified.

This module integrates:
- Phase 2: Switch Statement Verification (switch_verifier.py)
- Phase 3: Automatic Repair (switch_repair.py)

Pipeline:
1. Parse switch statements from code
2. Verify case-return mappings against ground truth
3. Detect missing cases using Z3
4. Generate repair candidates for errors
5. Apply repairs and re-verify
6. Report final results

Usage:
    from src.verification.verify_and_repair import VerifyAndRepairPipeline
    
    pipeline = VerifyAndRepairPipeline(backend="RISCV")
    result = pipeline.run(code)
    
    if result.needs_repair:
        repaired_code = result.repaired_code
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import sys
from pathlib import Path

# Direct imports using exec to avoid package issues
_current_dir = Path(__file__).parent
_switch_verifier_path = _current_dir / 'switch_verifier.py'
_switch_repair_path = _current_dir.parent / 'repair' / 'switch_repair.py'

# Execute switch_verifier.py to get its classes
exec(open(_switch_verifier_path).read())

# Import from switch_repair
sys.path.insert(0, str(_current_dir.parent / 'repair'))
from switch_repair import (
    SwitchRepairModel,
    SwitchRepairResult,
    SwitchRepairCandidate,
    SwitchRepairType,
)


class PipelineStatus(Enum):
    """Status of the verify-and-repair pipeline."""
    VERIFIED = "verified"  # Code passed verification
    REPAIRED = "repaired"  # Code was repaired and verified
    REPAIR_FAILED = "repair_failed"  # Repair attempted but re-verification failed
    ERROR = "error"  # Pipeline error


@dataclass
class PipelineResult:
    """Result of the verify-and-repair pipeline."""
    status: PipelineStatus
    original_code: str
    final_code: str
    
    # Verification results
    initial_verification: Optional[VerificationResult] = None
    final_verification: Optional[VerificationResult] = None
    
    # Repair results
    repairs_applied: List[SwitchRepairCandidate] = field(default_factory=list)
    repair_iterations: int = 0
    
    # Coverage
    initial_coverage: float = 0.0
    final_coverage: float = 0.0
    
    # Summary
    errors_fixed: int = 0
    cases_added: int = 0
    message: str = ""
    
    @property
    def needs_repair(self) -> bool:
        return self.status == PipelineStatus.REPAIRED
    
    @property
    def repaired_code(self) -> str:
        return self.final_code if self.needs_repair else self.original_code
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'needs_repair': self.needs_repair,
            'initial_verification': self.initial_verification.to_dict() if self.initial_verification else None,
            'final_verification': self.final_verification.to_dict() if self.final_verification else None,
            'repairs_applied': [r.to_dict() for r in self.repairs_applied],
            'repair_iterations': self.repair_iterations,
            'initial_coverage': f"{self.initial_coverage*100:.1f}%",
            'final_coverage': f"{self.final_coverage*100:.1f}%",
            'errors_fixed': self.errors_fixed,
            'cases_added': self.cases_added,
            'message': self.message,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "VERIFY AND REPAIR PIPELINE RESULT",
            "=" * 60,
            f"Status: {self.status.value.upper()}",
            f"Initial coverage: {self.initial_coverage*100:.1f}%",
            f"Final coverage: {self.final_coverage*100:.1f}%",
        ]
        
        if self.repairs_applied:
            lines.append(f"\nRepairs applied ({len(self.repairs_applied)}):")
            for i, repair in enumerate(self.repairs_applied, 1):
                lines.append(f"  {i}. {repair.description}")
        
        if self.errors_fixed:
            lines.append(f"\nErrors fixed: {self.errors_fixed}")
        if self.cases_added:
            lines.append(f"Cases added: {self.cases_added}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class VerifyAndRepairPipeline:
    """
    End-to-end pipeline for verifying and repairing switch statements.
    
    Features:
    - Automatic verification using Z3
    - Smart repair generation
    - Iterative refinement (up to max_iterations)
    - Detailed reporting
    """
    
    def __init__(
        self,
        backend: str = "RISCV",
        max_iterations: int = 3,
        verbose: bool = False
    ):
        self.backend = backend
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize components
        self.verifier = SwitchVerifier(verbose=verbose)
        self.parser = SwitchStatementParser()
        self.coverage_verifier = InputCoverageVerifier()
        self.repair_model = SwitchRepairModel(backend=backend, verbose=verbose)
    
    def run(
        self,
        code: str,
        function_name: str = "unknown",
        expected_mappings: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Run the complete verify-and-repair pipeline.
        
        Args:
            code: Function code to verify and repair
            function_name: Name of the function (for reporting)
            expected_mappings: Optional ground truth mappings
            
        Returns:
            PipelineResult with all details
        """
        if self.verbose:
            print(f"\n🔍 Starting pipeline for: {function_name}")
        
        # Use verifier's built-in mappings if not provided
        if expected_mappings is None and self.backend == "RISCV":
            expected_mappings = self.verifier.RISCV_RELOC_MAPPINGS
        
        # Step 1: Initial verification
        if self.verbose:
            print("  Step 1: Initial verification...")
        
        initial_result = self.verifier.verify_function(
            code=code,
            function_name=function_name,
            expected_mappings=expected_mappings
        )
        
        # Calculate initial coverage
        switches = self.parser.parse(code)
        initial_coverage = 0.0
        if switches and self.backend == "RISCV":
            coverage_result = self.coverage_verifier.verify_riscv_reloc_coverage(switches[0])
            initial_coverage = coverage_result['coverage_rate']
        
        if self.verbose:
            print(f"    Status: {initial_result.status.value}")
            print(f"    Coverage: {initial_coverage*100:.1f}%")
            print(f"    Incorrect: {len(initial_result.incorrect_mappings)}")
            print(f"    Missing: {len(initial_result.missing_cases)}")
        
        # Check if verification passed
        if (initial_result.status == VerificationStatus.VERIFIED and 
            not initial_result.incorrect_mappings and
            not initial_result.missing_cases):
            
            return PipelineResult(
                status=PipelineStatus.VERIFIED,
                original_code=code,
                final_code=code,
                initial_verification=initial_result,
                final_verification=initial_result,
                initial_coverage=initial_coverage,
                final_coverage=initial_coverage,
                message="Code verified successfully, no repairs needed"
            )
        
        # Step 2: Repair loop
        if self.verbose:
            print("  Step 2: Repair loop...")
        
        current_code = code
        all_repairs = []
        iteration = 0
        errors_fixed = 0
        cases_added = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if self.verbose:
                print(f"    Iteration {iteration}/{self.max_iterations}")
            
            # Get current verification result
            current_result = self.verifier.verify_function(
                code=current_code,
                function_name=function_name,
                expected_mappings=expected_mappings
            )
            
            # Check if we're done
            if (not current_result.incorrect_mappings and
                not current_result.missing_cases):
                break
            
            # Generate repairs
            repair_result = self.repair_model.repair_from_verification_result(
                current_code,
                current_result.to_dict()
            )
            
            if not repair_result.success or not repair_result.candidates:
                if self.verbose:
                    print("    No more repairs generated")
                break
            
            # Apply best repair
            best_repair = repair_result.best_repair
            if best_repair:
                current_code = best_repair.repaired_code
                all_repairs.append(best_repair)
                
                if best_repair.repair_type == SwitchRepairType.FIX_RETURN_VALUE:
                    errors_fixed += 1
                elif best_repair.repair_type == SwitchRepairType.ADD_MISSING_CASE:
                    cases_added += 1
                
                if self.verbose:
                    print(f"    Applied: {best_repair.description}")
        
        # Step 3: Final verification
        if self.verbose:
            print("  Step 3: Final verification...")
        
        final_result = self.verifier.verify_function(
            code=current_code,
            function_name=function_name,
            expected_mappings=expected_mappings
        )
        
        # Calculate final coverage
        final_switches = self.parser.parse(current_code)
        final_coverage = initial_coverage
        if final_switches and self.backend == "RISCV":
            final_coverage_result = self.coverage_verifier.verify_riscv_reloc_coverage(final_switches[0])
            final_coverage = final_coverage_result['coverage_rate']
        
        # Determine final status
        if (final_result.status == VerificationStatus.VERIFIED and
            not final_result.incorrect_mappings):
            final_status = PipelineStatus.REPAIRED
            message = f"Successfully repaired with {len(all_repairs)} fix(es)"
        elif all_repairs:
            final_status = PipelineStatus.REPAIR_FAILED
            message = f"Partial repair: {len(all_repairs)} fix(es) applied but verification still fails"
        else:
            final_status = PipelineStatus.REPAIR_FAILED
            message = "Could not generate repairs"
        
        if self.verbose:
            print(f"  Final status: {final_status.value}")
        
        return PipelineResult(
            status=final_status,
            original_code=code,
            final_code=current_code,
            initial_verification=initial_result,
            final_verification=final_result,
            repairs_applied=all_repairs,
            repair_iterations=iteration,
            initial_coverage=initial_coverage,
            final_coverage=final_coverage,
            errors_fixed=errors_fixed,
            cases_added=cases_added,
            message=message
        )
    
    def run_on_database(
        self,
        db_path: str,
        function_filter: Optional[str] = None,
        backend_filter: Optional[str] = None
    ) -> Dict[str, PipelineResult]:
        """
        Run pipeline on all functions in a database.
        
        Args:
            db_path: Path to llvm_functions_multi.json
            function_filter: Only process functions matching this pattern
            backend_filter: Only process functions from this backend
            
        Returns:
            Dict mapping function_id to PipelineResult
        """
        with open(db_path) as f:
            db = json.load(f)
        
        results = {}
        
        for func_id, func in db['functions'].items():
            # Apply filters
            if backend_filter and func.get('backend') != backend_filter:
                continue
            if function_filter and function_filter not in func.get('name', ''):
                continue
            
            # Skip functions without switch statements
            if not func.get('has_switch', False):
                continue
            
            code = func.get('body', '')
            if not code:
                continue
            
            result = self.run(
                code=code,
                function_name=func.get('full_name', func.get('name', 'unknown'))
            )
            results[func_id] = result
        
        return results


def create_pipeline(backend: str = "RISCV", verbose: bool = False) -> VerifyAndRepairPipeline:
    """Factory function to create a pipeline."""
    return VerifyAndRepairPipeline(backend=backend, verbose=verbose)


# Demo / Test
if __name__ == '__main__':
    print("=" * 70)
    print("Verify and Repair Pipeline Demo")
    print("=" * 70)
    
    # Create pipeline
    pipeline = VerifyAndRepairPipeline(backend="RISCV", verbose=True)
    
    # Test 1: Code that passes verification
    print("\n📝 Test 1: Code that passes verification")
    good_code = """
    switch (Kind) {
        case RISCV::fixup_riscv_hi20:
            return ELF::R_RISCV_HI20;
        case RISCV::fixup_riscv_lo12_i:
            return ELF::R_RISCV_LO12_I;
        default:
            return ELF::R_RISCV_NONE;
    }
    """
    
    result1 = pipeline.run(good_code, "test_good")
    print(result1.summary())
    
    # Test 2: Code with errors that needs repair
    print("\n📝 Test 2: Code with errors that needs repair")
    bad_code = """
    switch (Kind) {
        case RISCV::fixup_riscv_hi20:
            return ELF::R_RISCV_LO12_I;  // WRONG!
        case RISCV::fixup_riscv_lo12_i:
            return ELF::R_RISCV_LO12_I;
        default:
            return ELF::R_RISCV_NONE;
    }
    """
    
    result2 = pipeline.run(bad_code, "test_bad")
    print(result2.summary())
    
    if result2.needs_repair:
        print("\n🔧 Repaired code:")
        print(result2.repaired_code)
    
    # Test 3: Real LLVM code from database
    print("\n📝 Test 3: Real LLVM getRelocType")
    db_path = Path(__file__).parent.parent.parent / 'data' / 'llvm_functions_multi.json'
    
    if db_path.exists():
        with open(db_path) as f:
            db = json.load(f)
        
        for func_id, func in db['functions'].items():
            if 'getRelocType' in func['name'] and func['backend'] == 'RISCV':
                result3 = pipeline.run(
                    code=func.get('body', ''),
                    function_name=func.get('full_name', 'unknown')
                )
                print(result3.summary())
                break
    else:
        print(f"Database not found at {db_path}")
    
    print("\n" + "=" * 70)
    print("✅ Pipeline Demo Complete")
    print("=" * 70)
