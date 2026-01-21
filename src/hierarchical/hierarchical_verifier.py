"""
Hierarchical Verifier: Unified interface for all verification levels.

Provides a single entry point for hierarchical verification,
orchestrating L1 (Function), L2 (Module), and L3 (Backend) verification.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import time
import json

from .function_verify import (
    FunctionVerifier, FunctionVerificationResult, FunctionVerificationStatus
)
from .module_verify import (
    ModuleVerifier, ModuleVerificationResult, ModuleVerificationStatus, Module
)
from .backend_verify import (
    BackendVerifier, BackendVerificationResult, BackendVerificationStatus, Backend
)
from .interface_contract import InterfaceContract
from ..specification.spec_language import Specification


class VerificationLevel(Enum):
    """Verification level."""
    FUNCTION = 1
    MODULE = 2
    BACKEND = 3


@dataclass
class HierarchicalResult:
    """Combined result from hierarchical verification."""
    level: VerificationLevel
    backend_result: Optional[BackendVerificationResult] = None
    module_results: Dict[str, ModuleVerificationResult] = field(default_factory=dict)
    function_results: Dict[str, FunctionVerificationResult] = field(default_factory=dict)
    total_time_ms: float = 0.0
    
    @property
    def is_successful(self) -> bool:
        """Check if highest-level verification succeeded."""
        if self.level == VerificationLevel.BACKEND and self.backend_result:
            return self.backend_result.is_successful()
        elif self.level == VerificationLevel.MODULE:
            return all(r.is_successful() for r in self.module_results.values())
        else:
            return all(r.is_successful() for r in self.function_results.values())
    
    @property
    def verified_count(self) -> int:
        """Count of verified items at the primary level."""
        if self.level == VerificationLevel.BACKEND and self.backend_result:
            return self.backend_result.modules_verified
        elif self.level == VerificationLevel.MODULE:
            return sum(1 for r in self.module_results.values() if r.is_successful())
        else:
            return sum(1 for r in self.function_results.values() if r.is_successful())
    
    @property
    def total_count(self) -> int:
        """Total count of items at the primary level."""
        if self.level == VerificationLevel.BACKEND and self.backend_result:
            return self.backend_result.total_modules
        elif self.level == VerificationLevel.MODULE:
            return len(self.module_results)
        else:
            return len(self.function_results)
    
    @property
    def total_functions_verified(self) -> int:
        """Total functions verified across all levels."""
        if self.backend_result:
            return self.backend_result.functions_verified
        
        total = sum(r.verified_count for r in self.module_results.values())
        if total == 0:
            total = sum(1 for r in self.function_results.values() if r.is_successful())
        return total
    
    @property
    def total_functions(self) -> int:
        """Total functions across all levels."""
        if self.backend_result:
            return self.backend_result.total_functions
        
        total = sum(r.total_count for r in self.module_results.values())
        if total == 0:
            total = len(self.function_results)
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "level": self.level.name,
            "is_successful": self.is_successful,
            "verified_count": self.verified_count,
            "total_count": self.total_count,
            "total_functions_verified": self.total_functions_verified,
            "total_functions": self.total_functions,
            "total_time_ms": self.total_time_ms,
        }
        
        if self.backend_result:
            result["backend"] = self.backend_result.to_dict()
        
        if self.module_results:
            result["modules"] = {
                name: r.to_dict() for name, r in self.module_results.items()
            }
        
        if self.function_results:
            result["functions"] = {
                name: r.to_dict() for name, r in self.function_results.items()
            }
        
        return result
    
    def save(self, path: str) -> None:
        """Save result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "HIERARCHICAL VERIFICATION SUMMARY",
            "=" * 60,
            f"Verification Level: {self.level.name}",
            f"Overall Status: {'SUCCESS' if self.is_successful else 'INCOMPLETE'}",
            f"Total Time: {self.total_time_ms:.1f}ms",
            "",
            "Results:",
            f"  {self.level.name}: {self.verified_count}/{self.total_count}",
            f"  Functions: {self.total_functions_verified}/{self.total_functions} " 
            f"({self.total_functions_verified/max(self.total_functions, 1)*100:.1f}%)",
        ]
        
        if self.backend_result:
            lines.append("")
            lines.append("Backend Details:")
            lines.append(f"  Integration: {'✓' if self.backend_result.integration_verified else '✗'}")
            
            if self.backend_result.end_to_end_properties:
                lines.append("  End-to-End Properties:")
                for prop, verified in self.backend_result.end_to_end_properties.items():
                    lines.append(f"    {'✓' if verified else '✗'} {prop}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def __str__(self) -> str:
        status = "✓" if self.is_successful else "✗"
        return (f"[{status}] Hierarchical ({self.level.name}): "
                f"{self.verified_count}/{self.total_count} verified, "
                f"{self.total_functions_verified}/{self.total_functions} functions, "
                f"{self.total_time_ms:.1f}ms")


class HierarchicalVerifier:
    """
    Unified hierarchical verification engine.
    
    Orchestrates verification at all three levels:
    - Level 1 (Function): Individual function verification
    - Level 2 (Module): Module-level consistency and contracts
    - Level 3 (Backend): Backend-wide integration and end-to-end properties
    
    Supports both bottom-up (L1 -> L2 -> L3) and targeted verification.
    """
    
    def __init__(
        self,
        enable_repair: bool = True,
        max_repair_iterations: int = 5,
        parallel_verification: bool = False,
        max_workers: int = 4,
        timeout_ms: int = 600000,  # 10 minutes
        verbose: bool = False
    ):
        """
        Initialize hierarchical verifier.
        
        Args:
            enable_repair: Enable CGNR repair for failed verifications
            max_repair_iterations: Maximum repair iterations per function
            parallel_verification: Enable parallel verification
            max_workers: Number of parallel workers
            timeout_ms: Total verification timeout
            verbose: Enable verbose output
        """
        self.enable_repair = enable_repair
        self.max_repair_iterations = max_repair_iterations
        self.parallel_verification = parallel_verification
        self.max_workers = max_workers
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        
        # Initialize verifiers
        self.function_verifier = FunctionVerifier(
            enable_repair=enable_repair,
            max_repair_iterations=max_repair_iterations,
            verbose=verbose
        )
        
        self.module_verifier = ModuleVerifier(
            function_verifier=self.function_verifier,
            enable_repair=enable_repair,
            verbose=verbose
        )
        
        self.backend_verifier = BackendVerifier(
            module_verifier=self.module_verifier,
            enable_repair=enable_repair,
            verbose=verbose
        )
        
        # Statistics
        self.stats = {
            "verifications_run": 0,
            "backends_verified": 0,
            "modules_verified": 0,
            "functions_verified": 0,
            "total_time_ms": 0.0,
        }
    
    def verify_function(
        self,
        code: str,
        spec: Specification,
        module_name: str = ""
    ) -> HierarchicalResult:
        """
        Level 1: Verify a single function.
        
        Args:
            code: Function source code
            spec: Function specification
            module_name: Optional module name
        
        Returns:
            HierarchicalResult with function verification result
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n[Hierarchical L1] Verifying function: {spec.function_name}")
        
        result = HierarchicalResult(level=VerificationLevel.FUNCTION)
        
        func_result = self.function_verifier.verify(code, spec, module_name)
        result.function_results[spec.function_name] = func_result
        
        result.total_time_ms = (time.time() - start_time) * 1000
        
        self.stats["verifications_run"] += 1
        if func_result.is_successful():
            self.stats["functions_verified"] += 1
        self.stats["total_time_ms"] += result.total_time_ms
        
        return result
    
    def verify_module(self, module: Module) -> HierarchicalResult:
        """
        Level 2: Verify a module.
        
        Args:
            module: Module to verify
        
        Returns:
            HierarchicalResult with module verification result
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n[Hierarchical L2] Verifying module: {module.name}")
        
        result = HierarchicalResult(level=VerificationLevel.MODULE)
        
        mod_result = self.module_verifier.verify(module)
        result.module_results[module.name] = mod_result
        
        # Copy function results
        result.function_results = dict(mod_result.function_results)
        
        result.total_time_ms = (time.time() - start_time) * 1000
        
        self.stats["verifications_run"] += 1
        if mod_result.is_successful():
            self.stats["modules_verified"] += 1
        self.stats["functions_verified"] += mod_result.verified_count
        self.stats["total_time_ms"] += result.total_time_ms
        
        return result
    
    def verify_backend(self, backend: Backend) -> HierarchicalResult:
        """
        Level 3: Verify an entire backend.
        
        Args:
            backend: Backend to verify
        
        Returns:
            HierarchicalResult with backend verification result
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n[Hierarchical L3] Verifying backend: {backend.name}")
        
        result = HierarchicalResult(level=VerificationLevel.BACKEND)
        
        backend_result = self.backend_verifier.verify(backend)
        result.backend_result = backend_result
        
        # Copy module and function results
        result.module_results = dict(backend_result.module_results)
        
        for mod_result in result.module_results.values():
            result.function_results.update(mod_result.function_results)
        
        result.total_time_ms = (time.time() - start_time) * 1000
        
        self.stats["verifications_run"] += 1
        if backend_result.is_successful():
            self.stats["backends_verified"] += 1
        self.stats["modules_verified"] += backend_result.modules_verified
        self.stats["functions_verified"] += backend_result.functions_verified
        self.stats["total_time_ms"] += result.total_time_ms
        
        return result
    
    def verify(
        self,
        target: Union[tuple, Module, Backend],
        level: Optional[VerificationLevel] = None
    ) -> HierarchicalResult:
        """
        Unified verification interface.
        
        Automatically determines verification level based on target type,
        or uses specified level.
        
        Args:
            target: Item to verify (tuple for function, Module, or Backend)
            level: Optional explicit verification level
        
        Returns:
            HierarchicalResult
        """
        if isinstance(target, Backend):
            return self.verify_backend(target)
        elif isinstance(target, Module):
            return self.verify_module(target)
        elif isinstance(target, tuple) and len(target) >= 2:
            code, spec = target[0], target[1]
            module_name = target[2] if len(target) > 2 else ""
            return self.verify_function(code, spec, module_name)
        else:
            raise ValueError(f"Unknown target type: {type(target)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined verification statistics."""
        return {
            **self.stats,
            "function_stats": self.function_verifier.get_statistics(),
            "module_stats": self.module_verifier.get_statistics(),
            "backend_stats": self.backend_verifier.get_statistics(),
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = {
            "verifications_run": 0,
            "backends_verified": 0,
            "modules_verified": 0,
            "functions_verified": 0,
            "total_time_ms": 0.0,
        }
        self.function_verifier.reset_statistics()
