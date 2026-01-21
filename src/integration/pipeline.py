"""
VEGA-Verified End-to-End Pipeline.

Integrates all components for automated verification and repair:
1. Specification Inference
2. Verification Condition Generation
3. SMT-based Verification
4. Counterexample-Guided Neural Repair
5. Hierarchical Verification
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time
import json
from pathlib import Path

from ..specification import SpecificationInferrer, Specification
from ..verification import Verifier, VCGenerator, BoundedModelChecker
from ..repair import CGNREngine, RepairResult, RepairStatus
from ..hierarchical import HierarchicalVerifier, HierarchicalResult
from ..parsing import CppParser


class PipelineStage(Enum):
    """Stages in the verification pipeline."""
    PARSE = "parse"
    SPEC_INFER = "spec_inference"
    VCGEN = "vcgen"
    VERIFY = "verify"
    REPAIR = "repair"
    HIERARCHICAL = "hierarchical"


@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    stage: PipelineStage
    success: bool
    time_ms: float
    output: Any = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "success": self.success,
            "time_ms": self.time_ms,
            "has_output": self.output is not None,
            "error": self.error,
        }


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    function_name: str
    stages: List[StageResult] = field(default_factory=list)
    final_status: str = "unknown"
    total_time_ms: float = 0.0
    
    # Final outputs
    specification: Optional[Specification] = None
    verification_result: Optional[Any] = None
    repair_result: Optional[RepairResult] = None
    hierarchical_result: Optional[HierarchicalResult] = None
    
    def is_verified(self) -> bool:
        return self.final_status == "verified"
    
    def is_repaired(self) -> bool:
        return self.final_status == "repaired"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "final_status": self.final_status,
            "total_time_ms": self.total_time_ms,
            "stages": [s.to_dict() for s in self.stages],
            "is_verified": self.is_verified(),
            "is_repaired": self.is_repaired(),
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== Pipeline Result for {self.function_name} ===",
            f"Final Status: {self.final_status}",
            f"Total Time: {self.total_time_ms:.2f} ms",
            "",
            "Stages:"
        ]
        
        for stage in self.stages:
            status = "✓" if stage.success else "✗"
            lines.append(f"  {status} {stage.stage.value}: {stage.time_ms:.2f} ms")
            if stage.error:
                lines.append(f"      Error: {stage.error}")
        
        return "\n".join(lines)


@dataclass
class PipelineConfig:
    """Configuration for the verification pipeline."""
    # Specification inference
    enable_spec_inference: bool = True
    min_references: int = 1
    
    # Verification
    verification_timeout_ms: int = 30000
    use_bmc: bool = True
    bmc_bound: int = 10
    
    # Repair
    enable_repair: bool = True
    max_repair_iterations: int = 5
    repair_beam_size: int = 5
    
    # Hierarchical verification
    enable_hierarchical: bool = True
    parallel_verification: bool = False
    
    # Output
    verbose: bool = False
    save_intermediate: bool = False
    output_dir: str = "results"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_spec_inference": self.enable_spec_inference,
            "verification_timeout_ms": self.verification_timeout_ms,
            "enable_repair": self.enable_repair,
            "max_repair_iterations": self.max_repair_iterations,
            "enable_hierarchical": self.enable_hierarchical,
        }


class VEGAVerifiedPipeline:
    """
    Main pipeline for VEGA-Verified.
    
    Orchestrates the complete verification workflow:
    1. Parse input code
    2. Infer specification from references (if available)
    3. Generate verification conditions
    4. Verify with SMT solver
    5. If failed, repair with CGNR
    6. Optionally run hierarchical verification
    
    Usage:
        pipeline = VEGAVerifiedPipeline()
        result = pipeline.verify_function(code, references=[...])
        
        if result.is_verified():
            print("Verification successful!")
        elif result.is_repaired():
            print(f"Repaired code: {result.repair_result.repaired_code}")
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.parser = CppParser()
        self.spec_inferrer = SpecificationInferrer()
        self.verifier = Verifier(timeout_ms=self.config.verification_timeout_ms)
        self.vcgen = VCGenerator()
        self.bmc = BoundedModelChecker() if self.config.use_bmc else None
        self.cgnr = CGNREngine(
            verifier=self.verifier,
            max_iterations=self.config.max_repair_iterations,
            beam_size=self.config.repair_beam_size,
            verbose=self.config.verbose
        ) if self.config.enable_repair else None
        self.hierarchical = HierarchicalVerifier() if self.config.enable_hierarchical else None
        
        # Statistics
        self.stats = {
            "total_runs": 0,
            "verified_count": 0,
            "repaired_count": 0,
            "failed_count": 0,
            "total_time_ms": 0.0,
        }
    
    def verify_function(
        self,
        code: str,
        function_name: str,
        specification: Optional[Specification] = None,
        references: Optional[List[Tuple[str, str]]] = None,
    ) -> PipelineResult:
        """
        Verify a single function.
        
        Args:
            code: The code to verify
            function_name: Name of the function
            specification: Optional pre-defined specification
            references: Optional reference implementations for spec inference
            
        Returns:
            PipelineResult with complete verification status
        """
        start_time = time.time()
        result = PipelineResult(function_name=function_name)
        
        self.stats["total_runs"] += 1
        
        try:
            # Stage 1: Parse
            parse_result = self._run_stage(
                PipelineStage.PARSE,
                lambda: self.parser.parse(code)
            )
            result.stages.append(parse_result)
            
            if not parse_result.success:
                result.final_status = "parse_error"
                return self._finalize_result(result, start_time)
            
            parsed_functions = parse_result.output or []
            
            # Stage 2: Specification Inference (if needed)
            spec = specification
            if spec is None and self.config.enable_spec_inference:
                if references and len(references) >= self.config.min_references:
                    spec_result = self._run_stage(
                        PipelineStage.SPEC_INFER,
                        lambda: self.spec_inferrer.infer(function_name, references)
                    )
                    result.stages.append(spec_result)
                    
                    if spec_result.success:
                        spec = spec_result.output
                else:
                    # Create minimal specification
                    spec = Specification(function_name=function_name)
            
            if spec is None:
                spec = Specification(function_name=function_name)
            
            result.specification = spec
            
            # Stage 3: Verification Condition Generation
            vcgen_result = self._run_stage(
                PipelineStage.VCGEN,
                lambda: self.vcgen.generate(code, spec)
            )
            result.stages.append(vcgen_result)
            
            # Stage 4: Verification
            verify_result = self._run_stage(
                PipelineStage.VERIFY,
                lambda: self.verifier.verify(code, spec)
            )
            result.stages.append(verify_result)
            result.verification_result = verify_result.output
            
            if verify_result.success and verify_result.output:
                verification = verify_result.output
                
                if verification.is_verified():
                    result.final_status = "verified"
                    self.stats["verified_count"] += 1
                    return self._finalize_result(result, start_time)
            
            # Stage 5: Repair (if verification failed)
            if self.config.enable_repair and self.cgnr:
                repair_result = self._run_stage(
                    PipelineStage.REPAIR,
                    lambda: self.cgnr.repair(code, spec)
                )
                result.stages.append(repair_result)
                
                if repair_result.success and repair_result.output:
                    result.repair_result = repair_result.output
                    
                    if repair_result.output.is_successful():
                        result.final_status = "repaired"
                        self.stats["repaired_count"] += 1
                        return self._finalize_result(result, start_time)
            
            # Stage 6: Hierarchical Verification (optional)
            if self.config.enable_hierarchical and self.hierarchical:
                hier_result = self._run_stage(
                    PipelineStage.HIERARCHICAL,
                    lambda: self.hierarchical.verify_function(code, spec)
                )
                result.stages.append(hier_result)
                result.hierarchical_result = hier_result.output
            
            # If we reach here, verification/repair didn't succeed
            result.final_status = "failed"
            self.stats["failed_count"] += 1
            
        except Exception as e:
            result.final_status = f"error: {str(e)}"
            self.stats["failed_count"] += 1
        
        return self._finalize_result(result, start_time)
    
    def verify_batch(
        self,
        functions: List[Tuple[str, str]],  # [(name, code), ...]
        specifications: Optional[Dict[str, Specification]] = None,
    ) -> Dict[str, PipelineResult]:
        """
        Verify multiple functions.
        
        Args:
            functions: List of (function_name, code) tuples
            specifications: Optional dict of function_name -> specification
            
        Returns:
            Dict mapping function_name to PipelineResult
        """
        specs = specifications or {}
        results = {}
        
        for name, code in functions:
            spec = specs.get(name)
            results[name] = self.verify_function(code, name, specification=spec)
        
        return results
    
    def _run_stage(
        self,
        stage: PipelineStage,
        func: callable
    ) -> StageResult:
        """Execute a pipeline stage with timing and error handling."""
        start = time.time()
        
        try:
            output = func()
            elapsed = (time.time() - start) * 1000
            
            return StageResult(
                stage=stage,
                success=True,
                time_ms=elapsed,
                output=output
            )
        
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            
            if self.config.verbose:
                import traceback
                traceback.print_exc()
            
            return StageResult(
                stage=stage,
                success=False,
                time_ms=elapsed,
                error=str(e)
            )
    
    def _finalize_result(
        self,
        result: PipelineResult,
        start_time: float
    ) -> PipelineResult:
        """Finalize result with timing and optional saving."""
        result.total_time_ms = (time.time() - start_time) * 1000
        self.stats["total_time_ms"] += result.total_time_ms
        
        if self.config.save_intermediate:
            self._save_result(result)
        
        return result
    
    def _save_result(self, result: PipelineResult) -> None:
        """Save result to output directory."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{result.function_name}_{int(time.time())}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total = self.stats["total_runs"]
        
        return {
            **self.stats,
            "verification_rate": self.stats["verified_count"] / total if total > 0 else 0,
            "repair_rate": self.stats["repaired_count"] / total if total > 0 else 0,
            "success_rate": (self.stats["verified_count"] + self.stats["repaired_count"]) / total if total > 0 else 0,
            "avg_time_ms": self.stats["total_time_ms"] / total if total > 0 else 0,
        }
    
    def reset_statistics(self) -> None:
        """Reset statistics."""
        self.stats = {
            "total_runs": 0,
            "verified_count": 0,
            "repaired_count": 0,
            "failed_count": 0,
            "total_time_ms": 0.0,
        }


def create_pipeline(
    config_path: Optional[str] = None,
    **kwargs
) -> VEGAVerifiedPipeline:
    """
    Factory function to create a configured pipeline.
    
    Args:
        config_path: Optional path to configuration file
        **kwargs: Override configuration options
        
    Returns:
        Configured VEGAVerifiedPipeline
    """
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                config_data = json.load(f)
            config = PipelineConfig.from_dict(config_data)
        else:
            config = PipelineConfig()
    else:
        config = PipelineConfig()
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return VEGAVerifiedPipeline(config)
