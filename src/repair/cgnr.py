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
from .repair_model import RuleBasedRepairModel

import logging

logger = logging.getLogger(__name__)


# Try to import neural repair engine
try:
    from .neural_repair_engine import NeuralRepairEngine, NeuralRepairConfig, create_repair_engine
    NEURAL_ENGINE_AVAILABLE = True
except ImportError:
    NEURAL_ENGINE_AVAILABLE = False
    logger.warning("Neural repair engine not available. Using rule-based repair only.")


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
        model_name: str = "Salesforce/codet5-base",
        model_path: Optional[str] = None,
        verbose: bool = False
    ):
        self.verifier = verifier or Verifier()
        self.fault_localizer = FaultLocalizer()
        self.max_iterations = max_iterations
        self.beam_size = beam_size
        self.use_neural_model = use_neural_model
        self.model_name = model_name
        self.model_path = model_path
        self.verbose = verbose
        
        # Repair models
        self.neural_engine = None
        self.rule_based_model = RuleBasedRepairModel()
        
        if use_neural_model:
            self._init_neural_model()
    
    def _init_neural_model(self) -> None:
        """Initialize the neural repair model."""
        if not NEURAL_ENGINE_AVAILABLE:
            logger.warning("Neural engine not available, using rule-based fallback")
            return
        
        try:
            config = NeuralRepairConfig(
                model_name=self.model_name,
                model_path=self.model_path,
                num_return_sequences=self.beam_size,
            )
            self.neural_engine = NeuralRepairEngine(config)
            
            # Try to load the model
            if self.model_path or self._check_model_available():
                loaded = self.neural_engine.load(self.model_path)
                if loaded:
                    logger.info(f"Neural repair model loaded: {self.model_name}")
                else:
                    logger.warning("Failed to load neural model, using rule-based fallback")
                    self.neural_engine = None
            else:
                logger.info("Neural model not loaded (no GPU or model not cached)")
                self.neural_engine = None
        except Exception as e:
            logger.warning(f"Failed to initialize neural model: {e}")
            self.neural_engine = None
    
    def _check_model_available(self) -> bool:
        """Check if neural model dependencies are available."""
        try:
            import torch
            # Check if GPU available or model is cached
            return torch.cuda.is_available() or self.model_path is not None
        except ImportError:
            return False
    
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
        """
        Generate repair candidates using hybrid approach.
        
        Priority:
        1. Neural model (if available and loaded)
        2. Rule-based model (fallback)
        """
        candidates = []
        
        # Try neural model first
        if self.neural_engine and self.neural_engine.is_available():
            try:
                # Convert counterexample to format expected by neural engine
                cex_dict = {
                    "input_values": context.counterexample.input_values if context.counterexample else {},
                    "expected_output": context.counterexample.expected_output if context.counterexample else None,
                    "actual_output": context.counterexample.actual_output if context.counterexample else None,
                }
                
                neural_candidates = self.neural_engine.repair(
                    buggy_code=context.original_code,
                    counterexample=cex_dict,
                    num_candidates=self.beam_size
                )
                
                # Extract code from RepairCandidate objects
                for nc in neural_candidates:
                    if nc.code and nc.code != context.original_code:
                        candidates.append(nc.code)
                
                if self.verbose and candidates:
                    logger.info(f"Neural engine generated {len(candidates)} candidates")
                    
            except Exception as e:
                logger.warning(f"Neural repair failed: {e}")
        
        # Augment with rule-based repairs
        rule_candidates = self.rule_based_model.generate(context, self.beam_size)
        
        # Add rule-based candidates that aren't duplicates
        seen = set(candidates)
        for rc in rule_candidates:
            if rc not in seen and rc != context.original_code:
                candidates.append(rc)
                seen.add(rc)
        
        if self.verbose:
            logger.info(f"Total repair candidates: {len(candidates)}")
        
        return candidates[:self.beam_size * 2]  # Return more candidates for better selection
    
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



# RuleBasedRepairModel is now imported from repair_model.py
