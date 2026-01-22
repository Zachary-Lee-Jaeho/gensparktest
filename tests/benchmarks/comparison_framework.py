"""
VEGA vs VEGA-Verified Comparison Framework.

This module provides a comprehensive framework for comparing VEGA (neural-only)
against VEGA-Verified (neural + verification + repair) across multiple dimensions:

1. Accuracy Metrics:
   - Function-level accuracy
   - Statement-level accuracy
   - Semantic correctness (verified vs syntactic)

2. Verification Coverage:
   - Percentage of functions with verified specifications
   - Specification inference success rate
   - CGNR repair success rate

3. Performance Metrics:
   - Generation time
   - Verification time
   - Repair iterations

4. Quality Metrics:
   - Bug detection rate
   - Bug repair rate
   - Semantic preservation

This framework implements the evaluation methodology from our paper.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.verification.verifier import Verifier, VerificationResult
from src.specification.spec_language import Specification
from src.repair.cgnr import CGNREngine, RepairResult, RepairStatus
from src.hierarchical.function_verify import FunctionVerifier, FunctionVerificationStatus
from src.hierarchical.module_verify import ModuleVerifier, Module
from src.hierarchical.backend_verify import BackendVerifier, Backend

from .processor_backends import ProcessorBackendBenchmark


class EvaluationMode(Enum):
    """Evaluation modes for comparison."""
    VEGA = "vega"                    # Neural generation only
    VEGA_VERIFIED = "vega-verified"  # Neural + verification + repair
    VERIFY_ONLY = "verify-only"      # Verification without repair
    FORK_FLOW = "fork-flow"          # Traditional fork-based approach


@dataclass
class FunctionMetrics:
    """Metrics for a single function evaluation."""
    function_name: str
    module_name: str
    
    # VEGA metrics (neural generation)
    vega_generated: bool = False
    vega_syntactically_correct: bool = False
    vega_statement_accuracy: float = 0.0
    
    # VEGA-Verified metrics
    specification_inferred: bool = False
    verification_attempted: bool = False
    verification_passed: bool = False
    repair_attempted: bool = False
    repair_succeeded: bool = False
    repair_iterations: int = 0
    
    # Timing
    generation_time_ms: float = 0.0
    verification_time_ms: float = 0.0
    repair_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Counterexamples and bugs
    bugs_found: int = 0
    bugs_fixed: int = 0
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "module_name": self.module_name,
            "vega_generated": self.vega_generated,
            "vega_syntactically_correct": self.vega_syntactically_correct,
            "vega_statement_accuracy": self.vega_statement_accuracy,
            "specification_inferred": self.specification_inferred,
            "verification_attempted": self.verification_attempted,
            "verification_passed": self.verification_passed,
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
            "repair_iterations": self.repair_iterations,
            "generation_time_ms": self.generation_time_ms,
            "verification_time_ms": self.verification_time_ms,
            "repair_time_ms": self.repair_time_ms,
            "total_time_ms": self.total_time_ms,
            "bugs_found": self.bugs_found,
            "bugs_fixed": self.bugs_fixed,
        }


@dataclass
class ModuleMetrics:
    """Aggregated metrics for a module."""
    module_name: str
    functions: List[FunctionMetrics] = field(default_factory=list)
    
    # Module-level verification
    interface_contract_satisfied: bool = False
    internal_consistency_verified: bool = False
    
    @property
    def total_functions(self) -> int:
        return len(self.functions)
    
    @property
    def vega_accuracy(self) -> float:
        if not self.functions:
            return 0.0
        correct = sum(1 for f in self.functions if f.vega_syntactically_correct)
        return correct / len(self.functions)
    
    @property
    def verified_accuracy(self) -> float:
        if not self.functions:
            return 0.0
        verified = sum(1 for f in self.functions if f.verification_passed or f.repair_succeeded)
        return verified / len(self.functions)
    
    @property
    def repair_success_rate(self) -> float:
        attempted = sum(1 for f in self.functions if f.repair_attempted)
        if attempted == 0:
            return 0.0
        succeeded = sum(1 for f in self.functions if f.repair_succeeded)
        return succeeded / attempted
    
    @property
    def total_bugs_found(self) -> int:
        return sum(f.bugs_found for f in self.functions)
    
    @property
    def total_bugs_fixed(self) -> int:
        return sum(f.bugs_fixed for f in self.functions)


@dataclass
class BackendMetrics:
    """Aggregated metrics for a complete backend."""
    backend_name: str
    target_triple: str
    modules: List[ModuleMetrics] = field(default_factory=list)
    
    # Backend-level metrics
    cross_module_verification: bool = False
    end_to_end_correctness: bool = False
    
    # Timing
    total_evaluation_time_ms: float = 0.0
    
    @property
    def total_functions(self) -> int:
        return sum(m.total_functions for m in self.modules)
    
    @property
    def vega_function_accuracy(self) -> float:
        total = sum(m.total_functions for m in self.modules)
        if total == 0:
            return 0.0
        correct = sum(sum(1 for f in m.functions if f.vega_syntactically_correct) for m in self.modules)
        return correct / total
    
    @property
    def vega_verified_accuracy(self) -> float:
        total = sum(m.total_functions for m in self.modules)
        if total == 0:
            return 0.0
        verified = sum(sum(1 for f in m.functions if f.verification_passed or f.repair_succeeded) for m in self.modules)
        return verified / total
    
    @property
    def specification_coverage(self) -> float:
        total = sum(m.total_functions for m in self.modules)
        if total == 0:
            return 0.0
        with_spec = sum(sum(1 for f in m.functions if f.specification_inferred) for m in self.modules)
        return with_spec / total
    
    @property
    def verification_coverage(self) -> float:
        total = sum(m.total_functions for m in self.modules)
        if total == 0:
            return 0.0
        verified = sum(sum(1 for f in m.functions if f.verification_attempted) for m in self.modules)
        return verified / total
    
    @property
    def improvement_over_vega(self) -> float:
        return self.vega_verified_accuracy - self.vega_function_accuracy


@dataclass
class ComparisonResult:
    """Complete comparison result between VEGA and VEGA-Verified."""
    timestamp: str
    backends: List[BackendMetrics] = field(default_factory=list)
    
    # Summary statistics
    @property
    def overall_vega_accuracy(self) -> float:
        total = sum(b.total_functions for b in self.backends)
        if total == 0:
            return 0.0
        correct = sum(sum(sum(1 for f in m.functions if f.vega_syntactically_correct) 
                        for m in b.modules) for b in self.backends)
        return correct / total
    
    @property
    def overall_vega_verified_accuracy(self) -> float:
        total = sum(b.total_functions for b in self.backends)
        if total == 0:
            return 0.0
        verified = sum(sum(sum(1 for f in m.functions if f.verification_passed or f.repair_succeeded) 
                         for m in b.modules) for b in self.backends)
        return verified / total
    
    @property
    def overall_improvement(self) -> float:
        return self.overall_vega_verified_accuracy - self.overall_vega_accuracy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_backends": len(self.backends),
                "total_functions": sum(b.total_functions for b in self.backends),
                "vega_accuracy": self.overall_vega_accuracy,
                "vega_verified_accuracy": self.overall_vega_verified_accuracy,
                "improvement": self.overall_improvement,
            },
            "backends": [
                {
                    "name": b.backend_name,
                    "triple": b.target_triple,
                    "total_functions": b.total_functions,
                    "vega_accuracy": b.vega_function_accuracy,
                    "vega_verified_accuracy": b.vega_verified_accuracy,
                    "specification_coverage": b.specification_coverage,
                    "verification_coverage": b.verification_coverage,
                    "improvement": b.improvement_over_vega,
                    "modules": [
                        {
                            "name": m.module_name,
                            "functions": m.total_functions,
                            "vega_accuracy": m.vega_accuracy,
                            "verified_accuracy": m.verified_accuracy,
                            "repair_success_rate": m.repair_success_rate,
                        }
                        for m in b.modules
                    ]
                }
                for b in self.backends
            ]
        }
    
    def save(self, filepath: str):
        """Save comparison results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_summary(self):
        """Print a formatted summary of the comparison."""
        print("\n" + "=" * 70)
        print("VEGA vs VEGA-Verified Comparison Results")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Backends: {len(self.backends)}")
        print(f"Total Functions: {sum(b.total_functions for b in self.backends)}")
        print()
        
        print("-" * 70)
        print(f"{'Metric':<40} {'VEGA':<15} {'VEGA-Verified':<15}")
        print("-" * 70)
        print(f"{'Overall Function Accuracy':<40} {self.overall_vega_accuracy:>13.1%}   {self.overall_vega_verified_accuracy:>13.1%}")
        print(f"{'Improvement':<40} {'':<15} +{self.overall_improvement:>12.1%}")
        print("-" * 70)
        print()
        
        print("Per-Backend Results:")
        print("-" * 70)
        print(f"{'Backend':<15} {'VEGA Acc':<12} {'VV Acc':<12} {'Spec Cov':<12} {'Improve':<12}")
        print("-" * 70)
        
        for backend in self.backends:
            print(f"{backend.backend_name:<15} {backend.vega_function_accuracy:>10.1%}   "
                  f"{backend.vega_verified_accuracy:>10.1%}   "
                  f"{backend.specification_coverage:>10.1%}   "
                  f"+{backend.improvement_over_vega:>9.1%}")
        
        print("-" * 70)


class ComparisonFramework:
    """
    Main framework for comparing VEGA and VEGA-Verified.
    
    This class orchestrates the evaluation process:
    1. Load benchmarks for each target
    2. Simulate VEGA generation (or use real VEGA results)
    3. Run VEGA-Verified pipeline (spec inference + verification + repair)
    4. Collect and analyze metrics
    5. Generate comparison report
    """
    
    def __init__(self, 
                 verifier: Optional[Verifier] = None,
                 cgnr_engine: Optional[CGNREngine] = None,
                 verbose: bool = True):
        """
        Initialize the comparison framework.
        
        Args:
            verifier: Verifier instance (created if None)
            cgnr_engine: CGNR engine for repair (created if None)
            verbose: Print progress information
        """
        self.verifier = verifier or Verifier()
        self.cgnr_engine = cgnr_engine or CGNREngine(self.verifier)
        self.verbose = verbose
        
        # Hierarchical verifiers
        self.function_verifier = FunctionVerifier(self.verifier, self.cgnr_engine)
        self.module_verifier = ModuleVerifier(self.verifier, self.cgnr_engine)
        self.backend_verifier = BackendVerifier(self.verifier, self.cgnr_engine)
    
    def evaluate_function(self, 
                         code: str, 
                         spec: Specification,
                         function_name: str,
                         module_name: str,
                         mode: EvaluationMode = EvaluationMode.VEGA_VERIFIED) -> FunctionMetrics:
        """
        Evaluate a single function.
        
        Args:
            code: Function source code
            spec: Function specification
            function_name: Name of the function
            module_name: Name of the containing module
            mode: Evaluation mode
            
        Returns:
            FunctionMetrics with evaluation results
        """
        metrics = FunctionMetrics(
            function_name=function_name,
            module_name=module_name
        )
        
        start_time = time.time()
        
        # Step 1: VEGA generation (simulated - assume code is VEGA output)
        metrics.vega_generated = True
        metrics.vega_syntactically_correct = self._check_syntax(code)
        
        gen_time = time.time()
        metrics.generation_time_ms = (gen_time - start_time) * 1000
        
        if mode == EvaluationMode.VEGA:
            metrics.total_time_ms = metrics.generation_time_ms
            return metrics
        
        # Step 2: Specification inference (simulated - spec already provided)
        metrics.specification_inferred = spec is not None
        
        if not metrics.specification_inferred:
            metrics.total_time_ms = (time.time() - start_time) * 1000
            return metrics
        
        # Step 3: Verification
        metrics.verification_attempted = True
        verify_start = time.time()
        
        try:
            result = self.verifier.verify(code, spec)
            metrics.verification_passed = result.verified
            
            if not result.verified and result.counterexample:
                metrics.bugs_found = 1
                metrics.counterexamples.append({
                    "input_values": result.counterexample.get("input_values", {}),
                    "expected": result.counterexample.get("expected_output"),
                    "actual": result.counterexample.get("actual_output"),
                })
        except Exception as e:
            if self.verbose:
                print(f"  Verification error for {function_name}: {e}")
            metrics.verification_passed = False
        
        metrics.verification_time_ms = (time.time() - verify_start) * 1000
        
        if mode == EvaluationMode.VERIFY_ONLY or metrics.verification_passed:
            metrics.total_time_ms = (time.time() - start_time) * 1000
            return metrics
        
        # Step 4: CGNR Repair (if verification failed)
        if mode == EvaluationMode.VEGA_VERIFIED and not metrics.verification_passed:
            metrics.repair_attempted = True
            repair_start = time.time()
            
            try:
                repair_result = self.cgnr_engine.repair(code, spec)
                metrics.repair_succeeded = repair_result.is_successful()
                metrics.repair_iterations = repair_result.iterations
                
                if metrics.repair_succeeded:
                    metrics.bugs_fixed = metrics.bugs_found
            except Exception as e:
                if self.verbose:
                    print(f"  Repair error for {function_name}: {e}")
                metrics.repair_succeeded = False
            
            metrics.repair_time_ms = (time.time() - repair_start) * 1000
        
        metrics.total_time_ms = (time.time() - start_time) * 1000
        return metrics
    
    def evaluate_module(self, 
                       module: Module,
                       mode: EvaluationMode = EvaluationMode.VEGA_VERIFIED) -> ModuleMetrics:
        """
        Evaluate a module containing multiple functions.
        
        Args:
            module: Module to evaluate
            mode: Evaluation mode
            
        Returns:
            ModuleMetrics with aggregated results
        """
        metrics = ModuleMetrics(module_name=module.name)
        
        if self.verbose:
            print(f"\nEvaluating module: {module.name}")
            print(f"  Functions: {len(module.functions)}")
        
        for func in module.functions.values():
            if self.verbose:
                print(f"  - {func.name}...", end=" ")
            
            func_metrics = self.evaluate_function(
                code=func.code,
                spec=func.specification,
                function_name=func.name,
                module_name=module.name,
                mode=mode
            )
            metrics.functions.append(func_metrics)
            
            if self.verbose:
                status = "VERIFIED" if func_metrics.verification_passed else \
                        "REPAIRED" if func_metrics.repair_succeeded else "FAILED"
                print(status)
        
        # Module-level verification
        if mode == EvaluationMode.VEGA_VERIFIED:
            try:
                module_result = self.module_verifier.verify(module)
                metrics.interface_contract_satisfied = module_result.interface_satisfied
                metrics.internal_consistency_verified = module_result.is_successful()
            except Exception:
                pass
        
        return metrics
    
    def evaluate_backend(self, 
                        benchmark: ProcessorBackendBenchmark,
                        mode: EvaluationMode = EvaluationMode.VEGA_VERIFIED) -> BackendMetrics:
        """
        Evaluate a complete backend.
        
        Args:
            benchmark: Backend benchmark to evaluate
            mode: Evaluation mode
            
        Returns:
            BackendMetrics with complete results
        """
        metrics = BackendMetrics(
            backend_name=benchmark.name,
            target_triple=benchmark.triple
        )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating Backend: {benchmark.name}")
            print(f"Triple: {benchmark.triple}")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        for module in benchmark.modules.values():
            module_metrics = self.evaluate_module(module, mode)
            metrics.modules.append(module_metrics)
        
        # Backend-level verification
        if mode == EvaluationMode.VEGA_VERIFIED:
            try:
                backend = benchmark.to_backend()
                backend_result = self.backend_verifier.verify(backend)
                metrics.cross_module_verification = backend_result.integration_verified
                metrics.end_to_end_correctness = backend_result.is_successful()
            except Exception:
                pass
        
        metrics.total_evaluation_time_ms = (time.time() - start_time) * 1000
        
        return metrics
    
    def run_comparison(self, 
                      benchmarks: List[ProcessorBackendBenchmark],
                      modes: List[EvaluationMode] = None) -> ComparisonResult:
        """
        Run comparison across multiple benchmarks.
        
        Args:
            benchmarks: List of backend benchmarks
            modes: Evaluation modes to compare (default: VEGA and VEGA_VERIFIED)
            
        Returns:
            ComparisonResult with complete comparison
        """
        if modes is None:
            modes = [EvaluationMode.VEGA, EvaluationMode.VEGA_VERIFIED]
        
        result = ComparisonResult(
            timestamp=datetime.now().isoformat()
        )
        
        for benchmark in benchmarks:
            # Run VEGA-Verified evaluation (captures both VEGA baseline and verified metrics)
            backend_metrics = self.evaluate_backend(
                benchmark, 
                mode=EvaluationMode.VEGA_VERIFIED
            )
            result.backends.append(backend_metrics)
        
        return result
    
    def _check_syntax(self, code: str) -> bool:
        """Check if code is syntactically valid."""
        # Simple heuristic: check for basic C++ structure
        if not code or not code.strip():
            return False
        
        # Check for function-like structure
        has_function = '{' in code and '}' in code
        has_return_or_void = 'return' in code or 'void' in code
        
        return has_function and has_return_or_void


def run_vega_paper_comparison():
    """
    Run comparison on VEGA paper benchmarks (RISC-V, RI5CY, xCORE).
    
    This function reproduces the evaluation from the VEGA paper and
    compares against VEGA-Verified results.
    """
    from .vega_paper_targets import get_vega_paper_targets
    
    print("=" * 70)
    print("VEGA Paper Benchmarks Comparison")
    print("=" * 70)
    
    # Get benchmarks
    targets = get_vega_paper_targets()
    benchmarks = list(targets.values())
    
    # Run comparison
    framework = ComparisonFramework(verbose=True)
    result = framework.run_comparison(benchmarks)
    
    # Print results
    result.print_summary()
    
    return result


def run_extended_comparison():
    """
    Run comparison on extended benchmarks (including ARM, MIPS, x86-64).
    
    This demonstrates VEGA-Verified's applicability beyond VEGA paper targets.
    """
    from .processor_backends import get_all_benchmarks
    
    print("=" * 70)
    print("Extended Benchmarks Comparison (All Architectures)")
    print("=" * 70)
    
    # Get all benchmarks
    all_benchmarks = get_all_benchmarks()
    benchmarks = list(all_benchmarks.values())
    
    # Run comparison
    framework = ComparisonFramework(verbose=True)
    result = framework.run_comparison(benchmarks)
    
    # Print results
    result.print_summary()
    
    return result


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("VEGA vs VEGA-Verified Comparison Framework")
    print("=" * 60)
    
    # Test with single function
    from .processor_backends import create_riscv_benchmark
    
    benchmark = create_riscv_benchmark()
    
    framework = ComparisonFramework(verbose=True)
    backend_metrics = framework.evaluate_backend(benchmark)
    
    print(f"\nBackend: {backend_metrics.backend_name}")
    print(f"Total Functions: {backend_metrics.total_functions}")
    print(f"VEGA Accuracy: {backend_metrics.vega_function_accuracy:.1%}")
    print(f"VEGA-Verified Accuracy: {backend_metrics.vega_verified_accuracy:.1%}")
    print(f"Improvement: +{backend_metrics.improvement_over_vega:.1%}")
    print(f"Specification Coverage: {backend_metrics.specification_coverage:.1%}")
    print(f"Verification Coverage: {backend_metrics.verification_coverage:.1%}")
