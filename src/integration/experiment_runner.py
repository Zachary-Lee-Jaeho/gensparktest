"""
Experiment Runner for VEGA-Verified.

Orchestrates experiments comparing VEGA vs VEGA-Verified,
collecting metrics and generating reports.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
from datetime import datetime
import json
import time

from .vega_adapter import VEGAAdapter, VEGAMode, VEGAGenerationResult
from .llvm_adapter import LLVMAdapter, BackendInfo
from ..specification.spec_language import Specification
from ..specification.inferrer import SpecificationInferrer
from ..verification.verifier import Verifier, VerificationResult
from ..repair.cgnr import CGNREngine, RepairResult
from ..hierarchical.hierarchical_verifier import HierarchicalVerifier, HierarchicalResult
from ..utils.metrics import MetricsCollector


class ExperimentMode(Enum):
    """Experiment execution mode."""
    VEGA_ONLY = "vega"  # Original VEGA
    VEGA_VERIFIED = "vega-verified"  # VEGA + verification + repair
    COMPARISON = "comparison"  # Run both and compare


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    mode: ExperimentMode
    target: str
    output_dir: str = "results"
    
    # VEGA settings
    vega_accuracy: float = 0.715
    
    # Verification settings
    enable_verification: bool = True
    enable_repair: bool = True
    max_repair_iterations: int = 5
    verification_timeout_ms: int = 30000
    
    # Experiment scope
    modules: List[str] = field(default_factory=lambda: ["MCCodeEmitter", "ELFObjectWriter"])
    functions: List[str] = field(default_factory=list)  # Empty = all functions
    
    # Other settings
    seed: Optional[int] = None
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode.value,
            "target": self.target,
            "output_dir": self.output_dir,
            "vega_accuracy": self.vega_accuracy,
            "enable_verification": self.enable_verification,
            "enable_repair": self.enable_repair,
            "max_repair_iterations": self.max_repair_iterations,
            "verification_timeout_ms": self.verification_timeout_ms,
            "modules": self.modules,
            "functions": self.functions,
            "seed": self.seed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        return cls(
            name=data["name"],
            mode=ExperimentMode(data["mode"]),
            target=data["target"],
            output_dir=data.get("output_dir", "results"),
            vega_accuracy=data.get("vega_accuracy", 0.715),
            enable_verification=data.get("enable_verification", True),
            enable_repair=data.get("enable_repair", True),
            max_repair_iterations=data.get("max_repair_iterations", 5),
            verification_timeout_ms=data.get("verification_timeout_ms", 30000),
            modules=data.get("modules", ["MCCodeEmitter", "ELFObjectWriter"]),
            functions=data.get("functions", []),
            seed=data.get("seed"),
        )


@dataclass
class FunctionResult:
    """Result for a single function."""
    function_name: str
    module_name: str
    
    # Generation
    generation_result: Optional[VEGAGenerationResult] = None
    generated_code: str = ""
    
    # Specification
    specification: Optional[Specification] = None
    spec_inferred: bool = False
    
    # Verification
    verification_result: Optional[VerificationResult] = None
    is_verified: bool = False
    
    # Repair
    repair_result: Optional[RepairResult] = None
    is_repaired: bool = False
    final_code: str = ""
    
    # Timing
    generation_time_ms: float = 0.0
    spec_inference_time_ms: float = 0.0
    verification_time_ms: float = 0.0
    repair_time_ms: float = 0.0
    
    @property
    def total_time_ms(self) -> float:
        return (self.generation_time_ms + self.spec_inference_time_ms + 
                self.verification_time_ms + self.repair_time_ms)
    
    @property
    def is_successful(self) -> bool:
        return self.is_verified or self.is_repaired
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "module_name": self.module_name,
            "generated_code": self.generated_code[:500] if self.generated_code else "",
            "final_code": self.final_code[:500] if self.final_code else "",
            "is_verified": self.is_verified,
            "is_repaired": self.is_repaired,
            "is_successful": self.is_successful,
            "generation_time_ms": self.generation_time_ms,
            "verification_time_ms": self.verification_time_ms,
            "repair_time_ms": self.repair_time_ms,
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class ExperimentResult:
    """Complete result of an experiment."""
    config: ExperimentConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Function results by mode
    vega_results: Dict[str, FunctionResult] = field(default_factory=dict)
    vega_verified_results: Dict[str, FunctionResult] = field(default_factory=dict)
    
    # Summary statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    def compute_statistics(self) -> None:
        """Compute summary statistics."""
        self.statistics = {}
        
        # VEGA statistics
        if self.vega_results:
            total = len(self.vega_results)
            generated = sum(1 for r in self.vega_results.values() if r.generated_code)
            
            self.statistics["vega"] = {
                "total_functions": total,
                "generated": generated,
                "generation_rate": generated / max(total, 1),
                "avg_generation_time_ms": sum(
                    r.generation_time_ms for r in self.vega_results.values()
                ) / max(total, 1),
            }
        
        # VEGA-Verified statistics
        if self.vega_verified_results:
            total = len(self.vega_verified_results)
            verified = sum(1 for r in self.vega_verified_results.values() if r.is_verified)
            repaired = sum(1 for r in self.vega_verified_results.values() if r.is_repaired)
            successful = sum(1 for r in self.vega_verified_results.values() if r.is_successful)
            
            self.statistics["vega_verified"] = {
                "total_functions": total,
                "verified_directly": verified - repaired,  # Verified without repair
                "repaired": repaired,
                "successful": successful,
                "verification_rate": verified / max(total, 1),
                "repair_rate": repaired / max(total, 1),
                "success_rate": successful / max(total, 1),
                "avg_verification_time_ms": sum(
                    r.verification_time_ms for r in self.vega_verified_results.values()
                ) / max(total, 1),
                "avg_repair_time_ms": sum(
                    r.repair_time_ms for r in self.vega_verified_results.values()
                    if r.is_repaired
                ) / max(repaired, 1),
                "avg_total_time_ms": sum(
                    r.total_time_ms for r in self.vega_verified_results.values()
                ) / max(total, 1),
            }
        
        # Comparison statistics
        if self.vega_results and self.vega_verified_results:
            vega_acc = self.statistics["vega"]["generation_rate"]
            verified_rate = self.statistics["vega_verified"]["success_rate"]
            
            self.statistics["comparison"] = {
                "accuracy_improvement": verified_rate - vega_acc,
                "accuracy_improvement_percent": (verified_rate - vega_acc) / max(vega_acc, 0.01) * 100,
                "verification_coverage": self.statistics["vega_verified"]["verification_rate"],
                "semantic_correctness_guarantee": self.statistics["vega_verified"]["success_rate"],
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "statistics": self.statistics,
            "vega_results": {k: v.to_dict() for k, v in self.vega_results.items()},
            "vega_verified_results": {k: v.to_dict() for k, v in self.vega_verified_results.items()},
        }
    
    def save(self, path: str) -> None:
        """Save result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def generate_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            "EXPERIMENT REPORT",
            "=" * 70,
            f"Name: {self.config.name}",
            f"Target: {self.config.target}",
            f"Mode: {self.config.mode.value}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
        ]
        
        if "vega" in self.statistics:
            stats = self.statistics["vega"]
            lines.extend([
                "VEGA (Baseline):",
                f"  Functions: {stats['total_functions']}",
                f"  Generation Rate: {stats['generation_rate']:.1%}",
                f"  Avg Time: {stats['avg_generation_time_ms']:.1f}ms",
                "",
            ])
        
        if "vega_verified" in self.statistics:
            stats = self.statistics["vega_verified"]
            lines.extend([
                "VEGA-Verified:",
                f"  Functions: {stats['total_functions']}",
                f"  Verified Directly: {stats['verified_directly']}",
                f"  Repaired: {stats['repaired']}",
                f"  Success Rate: {stats['success_rate']:.1%}",
                f"  Avg Total Time: {stats['avg_total_time_ms']:.1f}ms",
                "",
            ])
        
        if "comparison" in self.statistics:
            comp = self.statistics["comparison"]
            lines.extend([
                "Comparison (VEGA-Verified vs VEGA):",
                f"  Accuracy Improvement: {comp['accuracy_improvement']:.1%} ({comp['accuracy_improvement_percent']:+.1f}%)",
                f"  Verification Coverage: {comp['verification_coverage']:.1%}",
                f"  Semantic Correctness: {comp['semantic_correctness_guarantee']:.1%}",
                "",
            ])
        
        lines.append("=" * 70)
        return "\n".join(lines)


class ExperimentRunner:
    """
    Orchestrates experiments comparing VEGA vs VEGA-Verified.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        llvm_adapter: Optional[LLVMAdapter] = None,
        vega_adapter: Optional[VEGAAdapter] = None,
        verbose: bool = True
    ):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
            llvm_adapter: LLVM adapter (creates default if None)
            vega_adapter: VEGA adapter (creates default if None)
            verbose: Enable verbose output
        """
        self.config = config
        self.verbose = verbose
        
        # Initialize adapters
        self.llvm_adapter = llvm_adapter or LLVMAdapter(verbose=verbose)
        self.vega_adapter = vega_adapter or VEGAAdapter(
            mode=VEGAMode.SIMULATION,
            accuracy_rate=config.vega_accuracy,
            target=config.target,
            verbose=verbose
        )
        
        # Initialize verification components
        self.verifier = Verifier(
            timeout_ms=config.verification_timeout_ms,
            verbose=verbose
        )
        
        self.spec_inferrer = SpecificationInferrer()
        
        if config.enable_repair:
            self.cgnr_engine = CGNREngine(
                verifier=self.verifier,
                max_iterations=config.max_repair_iterations,
                verbose=verbose
            )
        else:
            self.cgnr_engine = None
        
        # Set random seed if specified
        if config.seed is not None:
            import random
            random.seed(config.seed)
    
    def run(self) -> ExperimentResult:
        """
        Run the experiment.
        
        Returns:
            ExperimentResult with all results and statistics
        """
        result = ExperimentResult(
            config=self.config,
            start_time=datetime.now()
        )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"EXPERIMENT: {self.config.name}")
            print(f"Target: {self.config.target}")
            print(f"Mode: {self.config.mode.value}")
            print(f"{'='*60}\n")
        
        # Get backend info
        backend_info = self.llvm_adapter.get_backend_info(self.config.target)
        
        # Get functions to test
        functions = self._get_functions_to_test(backend_info)
        
        if self.verbose:
            print(f"Functions to test: {len(functions)}")
        
        # Run based on mode
        if self.config.mode == ExperimentMode.VEGA_ONLY:
            result.vega_results = self._run_vega(functions)
        
        elif self.config.mode == ExperimentMode.VEGA_VERIFIED:
            result.vega_verified_results = self._run_vega_verified(functions)
        
        elif self.config.mode == ExperimentMode.COMPARISON:
            result.vega_results = self._run_vega(functions)
            result.vega_verified_results = self._run_vega_verified(functions)
        
        result.end_time = datetime.now()
        result.compute_statistics()
        
        # Save results
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_path = output_dir / f"{self.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result.save(str(result_path))
        
        if self.verbose:
            print("\n" + result.generate_report())
            print(f"\nResults saved to: {result_path}")
        
        return result
    
    def _get_functions_to_test(
        self,
        backend_info: BackendInfo
    ) -> List[Tuple[str, str]]:
        """Get list of (function_name, module_name) to test."""
        functions = []
        
        for mod_name in self.config.modules:
            if mod_name not in backend_info.modules:
                continue
            
            module = backend_info.modules[mod_name]
            
            for func_name in module.functions:
                # Filter by specified functions if any
                if self.config.functions and func_name not in self.config.functions:
                    continue
                
                # Only include functions that VEGA can generate
                if self.vega_adapter.supports_function(func_name):
                    functions.append((func_name, mod_name))
        
        return functions
    
    def _run_vega(
        self,
        functions: List[Tuple[str, str]]
    ) -> Dict[str, FunctionResult]:
        """Run VEGA-only mode."""
        results = {}
        
        if self.verbose:
            print("\n--- VEGA Mode ---")
        
        for i, (func_name, mod_name) in enumerate(functions):
            if self.verbose:
                print(f"[{i+1}/{len(functions)}] Generating {func_name}...")
            
            result = FunctionResult(
                function_name=func_name,
                module_name=mod_name
            )
            
            # Generate
            gen_result = self.vega_adapter.generate(func_name, mod_name)
            result.generation_result = gen_result
            result.generated_code = gen_result.generated_code
            result.generation_time_ms = gen_result.generation_time_ms
            result.final_code = gen_result.generated_code
            
            results[func_name] = result
        
        return results
    
    def _run_vega_verified(
        self,
        functions: List[Tuple[str, str]]
    ) -> Dict[str, FunctionResult]:
        """Run VEGA-Verified mode."""
        results = {}
        
        if self.verbose:
            print("\n--- VEGA-Verified Mode ---")
        
        for i, (func_name, mod_name) in enumerate(functions):
            if self.verbose:
                print(f"\n[{i+1}/{len(functions)}] Processing {func_name}...")
            
            result = FunctionResult(
                function_name=func_name,
                module_name=mod_name
            )
            
            # Step 1: Generate with VEGA
            if self.verbose:
                print("  1. Generating code...")
            
            gen_result = self.vega_adapter.generate(func_name, mod_name)
            result.generation_result = gen_result
            result.generated_code = gen_result.generated_code
            result.generation_time_ms = gen_result.generation_time_ms
            
            # Step 2: Infer specification
            if self.config.enable_verification:
                if self.verbose:
                    print("  2. Inferring specification...")
                
                start_time = time.time()
                spec = self._infer_specification(func_name, mod_name)
                result.specification = spec
                result.spec_inferred = spec is not None
                result.spec_inference_time_ms = (time.time() - start_time) * 1000
            
            # Step 3: Verify
            if self.config.enable_verification and result.specification:
                if self.verbose:
                    print("  3. Verifying...")
                
                start_time = time.time()
                ver_result = self.verifier.verify(
                    result.generated_code,
                    result.specification
                )
                result.verification_result = ver_result
                result.is_verified = ver_result.is_verified()
                result.verification_time_ms = (time.time() - start_time) * 1000
                
                if result.is_verified:
                    result.final_code = result.generated_code
                    if self.verbose:
                        print("    ✓ Verified directly")
                
                # Step 4: Repair if needed
                elif self.config.enable_repair and self.cgnr_engine:
                    if self.verbose:
                        print("  4. Attempting repair...")
                    
                    start_time = time.time()
                    repair_result = self.cgnr_engine.repair(
                        result.generated_code,
                        result.specification
                    )
                    result.repair_result = repair_result
                    result.repair_time_ms = (time.time() - start_time) * 1000
                    
                    if repair_result.is_successful():
                        result.is_repaired = True
                        result.is_verified = True
                        result.final_code = repair_result.repaired_code
                        if self.verbose:
                            print(f"    ⚡ Repaired in {repair_result.iterations} iterations")
                    else:
                        result.final_code = result.generated_code
                        if self.verbose:
                            print("    ✗ Repair failed")
                else:
                    result.final_code = result.generated_code
                    if self.verbose:
                        print("    ✗ Verification failed")
            else:
                result.final_code = result.generated_code
            
            results[func_name] = result
        
        return results
    
    def _infer_specification(
        self,
        function_name: str,
        module_name: str
    ) -> Optional[Specification]:
        """Infer specification for a function."""
        # Get reference backends
        ref_backends = self.llvm_adapter.get_reference_backends(self.config.target)
        
        # For simulation, create a simple specification
        # In real implementation, would use spec_inferrer with actual references
        
        spec = Specification(
            function_name=function_name,
            module=module_name
        )
        
        # Add basic conditions based on function type
        if function_name == "getRelocType":
            from ..specification.spec_language import Condition, Variable, Constant
            spec.preconditions = [
                Condition.valid(Variable("Fixup")),
            ]
            spec.postconditions = [
                Condition.ge(Variable("result"), Constant(0)),
            ]
        
        elif function_name == "encodeInstruction":
            from ..specification.spec_language import Condition, Variable
            spec.preconditions = [
                Condition.valid(Variable("MI")),
            ]
        
        return spec


def run_experiment(
    name: str = "default",
    target: str = "RISCV",
    mode: str = "comparison",
    **kwargs
) -> ExperimentResult:
    """
    Convenience function to run an experiment.
    
    Args:
        name: Experiment name
        target: Target architecture
        mode: "vega", "vega-verified", or "comparison"
        **kwargs: Additional config options
    
    Returns:
        ExperimentResult
    """
    config = ExperimentConfig(
        name=name,
        mode=ExperimentMode(mode),
        target=target,
        **kwargs
    )
    
    runner = ExperimentRunner(config)
    return runner.run()
