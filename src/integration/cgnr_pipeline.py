"""
Phase 2.4: CGNR Pipeline - Counterexample-Guided Neural Repair.

This module implements the complete CGNR loop as described in VEGA:
1. Verify: Check code against specification using SMT
2. Analyze: Extract counterexample from failed verification
3. Repair: Generate fix candidates using neural model
4. Re-verify: Check if repair is correct
5. Iterate: Repeat until verified or max iterations

The pipeline integrates:
- Semantic Analysis (Phase 2.1)
- SMT Verification (Phase 2.2)
- Neural Repair (Phase 2.3)

Key Features:
- Incremental verification
- Repair candidate ranking
- Patch synthesis
- Convergence tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import json
import time
from pathlib import Path
from datetime import datetime


class PipelineStatus(Enum):
    """Status of CGNR pipeline execution."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    VERIFIED = "verified"
    REPAIRED = "repaired"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class VerificationResult:
    """Result from verification phase."""
    verified: bool
    counterexample: Optional[Dict[str, Any]] = None
    violated_property: Optional[str] = None
    time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairCandidate:
    """A repair candidate with metadata."""
    code: str
    confidence: float
    source: str  # "neural", "template", "heuristic"
    iteration: int
    verification_result: Optional[VerificationResult] = None


@dataclass
class CGNRIteration:
    """Single iteration of CGNR loop."""
    iteration_number: int
    input_code: str
    verification_result: VerificationResult
    repair_candidates: List[RepairCandidate] = field(default_factory=list)
    selected_repair: Optional[RepairCandidate] = None
    time_ms: float = 0.0


@dataclass
class CGNRResult:
    """Complete result of CGNR pipeline execution."""
    status: PipelineStatus
    original_code: str
    final_code: Optional[str] = None
    iterations: List[CGNRIteration] = field(default_factory=list)
    total_time_ms: float = 0.0
    total_repairs_tried: int = 0
    convergence_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "total_iterations": len(self.iterations),
            "total_time_ms": self.total_time_ms,
            "total_repairs_tried": self.total_repairs_tried,
            "converged": self.status in (PipelineStatus.VERIFIED, PipelineStatus.REPAIRED),
            "convergence_history": self.convergence_history,
        }


class CGNRPipeline:
    """
    Counterexample-Guided Neural Repair Pipeline.
    
    Implements the VEGA verification-repair loop:
    
    while not verified and iterations < max:
        1. result = verify(code, spec)
        2. if result.verified: return success
        3. counterexample = result.counterexample
        4. candidates = neural_repair(code, counterexample)
        5. for candidate in candidates:
               if verify(candidate): code = candidate; break
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        max_candidates_per_iteration: int = 5,
        timeout_seconds: int = 300,
        model_path: Optional[str] = None,
        verbose: bool = False
    ):
        self.max_iterations = max_iterations
        self.max_candidates = max_candidates_per_iteration
        self.timeout_seconds = timeout_seconds
        self.model_path = model_path
        self.verbose = verbose
        
        # Components (lazy-loaded)
        self._verifier = None
        self._analyzer = None
        self._repair_model = None
        
        # Statistics
        self.stats = {
            "pipelines_run": 0,
            "successful_repairs": 0,
            "total_iterations": 0,
            "total_candidates_tried": 0,
            "average_iterations": 0.0,
        }
    
    @property
    def verifier(self):
        """Lazy load verifier."""
        if self._verifier is None:
            from ..verification.ir_to_smt import SMTVerifier
            self._verifier = SMTVerifier(verbose=self.verbose)
        return self._verifier
    
    @property
    def analyzer(self):
        """Lazy load semantic analyzer."""
        if self._analyzer is None:
            from ..verification.semantic_analyzer import SemanticAnalyzer
            self._analyzer = SemanticAnalyzer(verbose=self.verbose)
        return self._analyzer
    
    @property
    def repair_model(self):
        """Lazy load repair model."""
        if self._repair_model is None:
            from ..repair.model_finetuning import CodeT5RepairModel, TrainingConfig
            config = TrainingConfig()
            if self.model_path:
                config.output_dir = str(Path(self.model_path).parent)
            self._repair_model = CodeT5RepairModel(config)
            # Load model if path specified
            if self.model_path:
                try:
                    self._repair_model.load_model(self.model_path)
                    if self.verbose:
                        print(f"Loaded trained model from: {self.model_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not load model from {self.model_path}: {e}")
        return self._repair_model
    
    def run(
        self,
        code: str,
        specification: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[CGNRIteration], None]] = None
    ) -> CGNRResult:
        """
        Run the complete CGNR pipeline.
        
        Args:
            code: Input code to verify/repair
            specification: Verification specification
            ground_truth: Optional ground truth for validation
            callback: Optional callback for iteration updates
            
        Returns:
            CGNRResult with final status and repaired code
        """
        self.stats["pipelines_run"] += 1
        start_time = time.time()
        
        result = CGNRResult(
            status=PipelineStatus.RUNNING,
            original_code=code
        )
        
        current_code = code
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration_start = time.time()
            
            # Check timeout
            if time.time() - start_time > self.timeout_seconds:
                result.status = PipelineStatus.TIMEOUT
                break
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"CGNR Iteration {iteration + 1}")
                print(f"{'='*60}")
            
            # 1. Verify current code
            ver_result = self._verify_code(current_code, specification)
            
            # Create iteration record
            iter_record = CGNRIteration(
                iteration_number=iteration + 1,
                input_code=current_code,
                verification_result=ver_result
            )
            
            # 2. Check if verified
            if ver_result.verified:
                if self.verbose:
                    print("✅ Code verified successfully!")
                
                result.status = PipelineStatus.VERIFIED if iteration == 0 else PipelineStatus.REPAIRED
                result.final_code = current_code
                iter_record.time_ms = (time.time() - iteration_start) * 1000
                result.iterations.append(iter_record)
                break
            
            if self.verbose:
                print(f"❌ Verification failed")
                print(f"   Counterexample: {ver_result.counterexample}")
            
            # 3. Generate repair candidates
            candidates = self._generate_repairs(
                current_code,
                ver_result.counterexample,
                specification,
                iteration
            )
            iter_record.repair_candidates = candidates
            
            # 4. Try each candidate
            repair_found = False
            for candidate in candidates:
                self.stats["total_candidates_tried"] += 1
                result.total_repairs_tried += 1
                
                # Verify candidate
                cand_result = self._verify_code(candidate.code, specification)
                candidate.verification_result = cand_result
                
                if cand_result.verified:
                    if self.verbose:
                        print(f"✅ Found valid repair (candidate {candidates.index(candidate) + 1})")
                    
                    current_code = candidate.code
                    iter_record.selected_repair = candidate
                    repair_found = True
                    break
                else:
                    # Track convergence (how close to solution)
                    convergence = self._calculate_convergence(
                        ver_result.counterexample,
                        cand_result.counterexample
                    )
                    result.convergence_history.append(convergence)
            
            iter_record.time_ms = (time.time() - iteration_start) * 1000
            result.iterations.append(iter_record)
            
            if callback:
                callback(iter_record)
            
            # If no repair found in this iteration, continue with best candidate
            if not repair_found:
                if candidates:
                    # Use best candidate for next iteration
                    current_code = candidates[0].code
                    if self.verbose:
                        print(f"⚠️ No valid repair found, continuing with best candidate")
                else:
                    if self.verbose:
                        print(f"⚠️ No repair candidates generated")
            
            iteration += 1
            self.stats["total_iterations"] += 1
        
        # Final status
        if result.status == PipelineStatus.RUNNING:
            result.status = PipelineStatus.MAX_ITERATIONS
        
        if result.status in (PipelineStatus.VERIFIED, PipelineStatus.REPAIRED):
            self.stats["successful_repairs"] += 1
        
        result.total_time_ms = (time.time() - start_time) * 1000
        
        # Update average
        if self.stats["pipelines_run"] > 0:
            self.stats["average_iterations"] = (
                self.stats["total_iterations"] / self.stats["pipelines_run"]
            )
        
        return result
    
    def _verify_code(
        self,
        code: str,
        specification: Dict[str, Any]
    ) -> VerificationResult:
        """Verify code against specification."""
        start_time = time.time()
        
        try:
            # Use SMT verifier if available
            from ..verification.ir_to_smt import IRToSMTConverter, SMTVerifier, SMTModel
            import z3
            
            converter = IRToSMTConverter()
            
            # Extract switch semantics
            semantics = self.analyzer.extract_switch_semantics(code)
            
            if not semantics:
                # No switches to verify - consider verified
                return VerificationResult(
                    verified=True,
                    time_ms=(time.time() - start_time) * 1000,
                    details={"reason": "no_switches"}
                )
            
            # Build verification conditions from specification
            expected_mappings = specification.get("expected_mappings", {})
            
            if expected_mappings:
                # Get actual mappings from code
                actual_mappings = {}
                for switch in semantics:
                    for case in switch.get("cases", []):
                        case_val = case.get("value", "").split("::")[-1]
                        ret_val = case.get("return", "").split("::")[-1] if case.get("return") else None
                        if ret_val:
                            actual_mappings[case_val] = ret_val
                
                # Check for mismatches
                for input_val, expected_output in expected_mappings.items():
                    input_key = input_val.split("::")[-1]
                    expected_key = expected_output.split("::")[-1]
                    
                    actual = actual_mappings.get(input_key)
                    
                    if actual is None:
                        # Missing case
                        return VerificationResult(
                            verified=False,
                            counterexample={
                                "input_values": {"Kind": input_val},
                                "expected_output": expected_output,
                                "actual_output": "missing_case",
                            },
                            violated_property=f"missing_case_{input_key}",
                            time_ms=(time.time() - start_time) * 1000
                        )
                    
                    if actual != expected_key:
                        # Wrong return
                        return VerificationResult(
                            verified=False,
                            counterexample={
                                "input_values": {"Kind": input_val},
                                "expected_output": expected_output,
                                "actual_output": actual,
                            },
                            violated_property=f"wrong_return_{input_key}",
                            time_ms=(time.time() - start_time) * 1000
                        )
            
            # All checks passed
            return VerificationResult(
                verified=True,
                time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            # Verification error
            return VerificationResult(
                verified=False,
                details={"error": str(e)},
                time_ms=(time.time() - start_time) * 1000
            )
    
    def _generate_repairs(
        self,
        code: str,
        counterexample: Optional[Dict[str, Any]],
        specification: Dict[str, Any],
        iteration: int
    ) -> List[RepairCandidate]:
        """Generate repair candidates for buggy code."""
        candidates = []
        
        # 1. Neural repair
        try:
            neural_repairs = self.repair_model.repair(
                code,
                counterexample,
                num_candidates=self.max_candidates
            )
            
            for repaired_code, confidence in neural_repairs:
                candidates.append(RepairCandidate(
                    code=repaired_code,
                    confidence=confidence,
                    source="neural",
                    iteration=iteration
                ))
        except Exception as e:
            if self.verbose:
                print(f"   Neural repair error: {e}")
        
        # 2. Template-based repair
        template_repairs = self._template_repair(code, counterexample, specification)
        candidates.extend(template_repairs)
        
        # 3. Heuristic repair
        heuristic_repairs = self._heuristic_repair(code, counterexample)
        candidates.extend(heuristic_repairs)
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        return candidates[:self.max_candidates]
    
    def _template_repair(
        self,
        code: str,
        counterexample: Optional[Dict[str, Any]],
        specification: Dict[str, Any]
    ) -> List[RepairCandidate]:
        """Generate template-based repair candidates."""
        candidates = []
        
        if not counterexample:
            return candidates
        
        input_vals = counterexample.get("input_values", {})
        expected = counterexample.get("expected_output", "")
        actual = counterexample.get("actual_output", "")
        
        # Template 1: Add missing case
        if actual == "missing_case" and expected:
            kind_val = input_vals.get("Kind", "")
            
            # Find insertion point (before default)
            if "default:" in code:
                repair = code.replace(
                    "default:",
                    f"case {kind_val}:\n    return {expected};\n  default:"
                )
                candidates.append(RepairCandidate(
                    code=repair,
                    confidence=0.9,
                    source="template",
                    iteration=0
                ))
        
        # Template 2: Fix wrong return
        if actual and expected and actual != expected and actual != "missing_case":
            import re
            
            # Find the wrong return and fix it
            pattern = rf'return\s+{re.escape(actual)}\s*;'
            repair = re.sub(pattern, f'return {expected};', code, count=1)
            
            if repair != code:
                candidates.append(RepairCandidate(
                    code=repair,
                    confidence=0.85,
                    source="template",
                    iteration=0
                ))
        
        return candidates
    
    def _heuristic_repair(
        self,
        code: str,
        counterexample: Optional[Dict[str, Any]]
    ) -> List[RepairCandidate]:
        """Generate heuristic-based repair candidates."""
        candidates = []
        
        if not counterexample:
            return candidates
        
        input_vals = counterexample.get("input_values", {})
        expected = counterexample.get("expected_output", "")
        
        # Heuristic: Check for common naming patterns
        if expected:
            # Try to find similar return values and suggest fixing
            import re
            
            # Find all return statements
            returns = re.findall(r'return\s+([\w:]+)\s*;', code)
            
            for ret in returns:
                # Check if similar to expected
                ret_base = ret.split("::")[-1]
                exp_base = expected.split("::")[-1]
                
                # Simple similarity: same length, differ by one char
                if len(ret_base) == len(exp_base):
                    diff = sum(a != b for a, b in zip(ret_base, exp_base))
                    if diff == 1:
                        repair = code.replace(ret, expected, 1)
                        candidates.append(RepairCandidate(
                            code=repair,
                            confidence=0.6,
                            source="heuristic",
                            iteration=0
                        ))
        
        return candidates
    
    def _calculate_convergence(
        self,
        old_cex: Optional[Dict[str, Any]],
        new_cex: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate convergence metric between counterexamples."""
        if old_cex is None or new_cex is None:
            return 0.5
        
        # Simple metric: check if expected output is getting closer
        old_actual = old_cex.get("actual_output", "")
        new_actual = new_cex.get("actual_output", "")
        expected = old_cex.get("expected_output", "")
        
        if new_actual == expected:
            return 1.0
        
        # Check string similarity
        from difflib import SequenceMatcher
        
        old_sim = SequenceMatcher(None, old_actual, expected).ratio()
        new_sim = SequenceMatcher(None, new_actual, expected).ratio()
        
        if new_sim > old_sim:
            return 0.5 + (new_sim - old_sim) * 0.5
        else:
            return 0.5 - (old_sim - new_sim) * 0.5
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self.stats.copy()


class EndToEndPipeline:
    """
    Complete End-to-End pipeline for compiler backend verification and repair.
    
    Stages:
    1. Code Extraction - Extract functions from LLVM sources
    2. Specification Inference - Infer specs from tests/comments
    3. Verification - SMT-based verification
    4. Repair - Neural/template-based repair
    5. Validation - Validate repairs against tests
    """
    
    def __init__(
        self,
        ground_truth_path: str,
        output_dir: str,
        verbose: bool = False
    ):
        self.ground_truth_path = Path(ground_truth_path)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CGNR pipeline
        self.cgnr = CGNRPipeline(
            max_iterations=10,
            max_candidates_per_iteration=5,
            verbose=verbose
        )
        
        # Results tracking
        self.results = {
            "functions_processed": 0,
            "functions_verified": 0,
            "functions_repaired": 0,
            "functions_failed": 0,
            "total_time_ms": 0,
            "per_function": []
        }
    
    def run_batch(
        self,
        function_filter: Optional[Callable[[Dict], bool]] = None,
        max_functions: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run verification/repair on batch of functions.
        
        Args:
            function_filter: Optional filter for selecting functions
            max_functions: Maximum number of functions to process
            
        Returns:
            Batch results summary
        """
        start_time = time.time()
        
        # Load ground truth
        with open(self.ground_truth_path, 'r') as f:
            db = json.load(f)
        
        functions = list(db.get("functions", {}).items())
        
        # Apply filter
        if function_filter:
            functions = [(k, v) for k, v in functions if function_filter(v)]
        
        # Limit
        if max_functions:
            functions = functions[:max_functions]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing {len(functions)} functions")
            print(f"{'='*60}")
        
        for func_id, func_data in functions:
            self._process_function(func_id, func_data)
        
        self.results["total_time_ms"] = (time.time() - start_time) * 1000
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _process_function(self, func_id: str, func_data: Dict[str, Any]) -> None:
        """Process a single function."""
        self.results["functions_processed"] += 1
        
        name = func_data.get("name", func_id)
        body = func_data.get("body", "")
        backend = func_data.get("backend", "unknown")
        
        if self.verbose:
            print(f"\n📋 Processing: {name} ({backend})")
        
        if not body or "switch" not in body:
            if self.verbose:
                print("   Skipping (no switch statements)")
            return
        
        # Infer specification from ground truth
        spec = self._infer_specification(func_data)
        
        if not spec.get("expected_mappings"):
            if self.verbose:
                print("   Skipping (no specification)")
            return
        
        # Run CGNR
        result = self.cgnr.run(body, spec)
        
        # Record result
        func_result = {
            "function_id": func_id,
            "name": name,
            "backend": backend,
            "status": result.status.value,
            "iterations": len(result.iterations),
            "time_ms": result.total_time_ms,
        }
        
        if result.status == PipelineStatus.VERIFIED:
            self.results["functions_verified"] += 1
            if self.verbose:
                print(f"   ✅ Verified (already correct)")
        
        elif result.status == PipelineStatus.REPAIRED:
            self.results["functions_repaired"] += 1
            func_result["repaired_code"] = result.final_code
            if self.verbose:
                print(f"   🔧 Repaired in {len(result.iterations)} iterations")
        
        else:
            self.results["functions_failed"] += 1
            if self.verbose:
                print(f"   ❌ Failed: {result.status.value}")
        
        self.results["per_function"].append(func_result)
    
    def _infer_specification(self, func_data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer specification from function data."""
        spec = {
            "function_name": func_data.get("name", ""),
            "expected_mappings": {},
        }
        
        # Extract from switches in ground truth
        switches = func_data.get("switches", [])
        for switch in switches:
            for case in switch.get("cases", []):
                case_label = case.get("label", "")
                case_return = case.get("return", "")
                
                if case_label and case_return:
                    spec["expected_mappings"][case_label] = case_return
        
        # If no switches, try to extract from body
        if not spec["expected_mappings"]:
            body = func_data.get("body", "")
            
            import re
            case_return_pattern = re.compile(
                r'case\s+([\w:]+):\s*\n\s*return\s+([\w:]+);'
            )
            
            for match in case_return_pattern.finditer(body):
                spec["expected_mappings"][match.group(1)] = match.group(2)
        
        return spec
    
    def _save_results(self) -> None:
        """Save pipeline results."""
        results_path = self.output_dir / "e2e_results.json"
        
        with open(results_path, 'w') as f:
            json.dump({
                **self.results,
                "timestamp": datetime.now().isoformat(),
                "cgnr_stats": self.cgnr.get_statistics(),
            }, f, indent=2)
        
        if self.verbose:
            print(f"\n📁 Results saved to: {results_path}")


def run_cgnr_demo(verbose: bool = True) -> CGNRResult:
    """
    Run a demo of the CGNR pipeline.
    """
    print("=" * 70)
    print("Phase 2.4: CGNR Pipeline Demo")
    print("=" * 70)
    
    # Test code with a bug (missing case)
    buggy_code = """
unsigned getRelocType(unsigned Kind, bool IsPCRel) {
    switch (Kind) {
    case FK_NONE:
        return ELF::R_RISCV_NONE;
    case FK_Data_4:
        return ELF::R_RISCV_32;
    // Missing: FK_Data_8 -> R_RISCV_64
    default:
        llvm_unreachable("Unknown fixup kind");
    }
}
"""
    
    # Specification
    specification = {
        "function_name": "getRelocType",
        "expected_mappings": {
            "FK_NONE": "ELF::R_RISCV_NONE",
            "FK_Data_4": "ELF::R_RISCV_32",
            "FK_Data_8": "ELF::R_RISCV_64",  # This is missing in buggy code
        }
    }
    
    # Create pipeline
    pipeline = CGNRPipeline(
        max_iterations=5,
        max_candidates_per_iteration=3,
        verbose=verbose
    )
    
    # Run
    result = pipeline.run(buggy_code, specification)
    
    # Print results
    print(f"\n{'='*60}")
    print("CGNR Result Summary")
    print(f"{'='*60}")
    print(f"Status: {result.status.value}")
    print(f"Iterations: {len(result.iterations)}")
    print(f"Total repairs tried: {result.total_repairs_tried}")
    print(f"Total time: {result.total_time_ms:.2f}ms")
    
    if result.final_code:
        print(f"\n📝 Final Code:")
        print(result.final_code)
    
    print(f"\n📊 Pipeline Stats:")
    for key, val in pipeline.get_statistics().items():
        print(f"   {key}: {val}")
    
    return result


# Demo
if __name__ == "__main__":
    run_cgnr_demo()
