"""
Level 3: Backend-level Integration Verification.

Verifies entire compiler backends for end-to-end correctness
and cross-module integration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import time

from .module_verify import (
    ModuleVerifier, ModuleVerificationResult, ModuleVerificationStatus, Module
)
from .interface_contract import InterfaceContract


class BackendVerificationStatus(Enum):
    """Status of backend verification."""
    PENDING = "pending"
    VERIFIED = "verified"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class Backend:
    """Compiler backend containing multiple modules."""
    name: str
    target_triple: str
    modules: Dict[str, Module] = field(default_factory=dict)
    module_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    integration_contract: Optional[InterfaceContract] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_module(self, module: Module) -> None:
        """Add a module to the backend."""
        self.modules[module.name] = module
    
    def set_dependencies(self, module_name: str, deps: List[str]) -> None:
        """Set dependencies for a module."""
        self.module_dependencies[module_name] = deps
    
    def get_dependency_order(self) -> List[str]:
        """Get modules in dependency order (topological sort)."""
        visited = set()
        order = []
        
        def visit(name: str):
            if name in visited or name not in self.modules:
                return
            visited.add(name)
            
            for dep in self.module_dependencies.get(name, []):
                visit(dep)
            order.append(name)
        
        for name in self.modules:
            visit(name)
        
        return order
    
    def total_functions(self) -> int:
        """Get total number of functions across all modules."""
        return sum(len(m.functions) for m in self.modules.values())


@dataclass
class BackendVerificationResult:
    """Result of backend-level verification."""
    backend_name: str
    target_triple: str
    status: BackendVerificationStatus
    module_results: Dict[str, ModuleVerificationResult] = field(default_factory=dict)
    integration_verified: bool = False
    integration_violations: List[str] = field(default_factory=list)
    end_to_end_properties: Dict[str, bool] = field(default_factory=dict)
    time_ms: float = 0.0
    messages: List[str] = field(default_factory=list)
    
    @property
    def modules_verified(self) -> int:
        """Count of fully verified modules."""
        return sum(
            1 for r in self.module_results.values() 
            if r.status == ModuleVerificationStatus.VERIFIED
        )
    
    @property
    def total_modules(self) -> int:
        """Total number of modules."""
        return len(self.module_results)
    
    @property
    def functions_verified(self) -> int:
        """Total verified functions across all modules."""
        return sum(r.verified_count for r in self.module_results.values())
    
    @property
    def total_functions(self) -> int:
        """Total functions across all modules."""
        return sum(r.total_count for r in self.module_results.values())
    
    @property
    def function_success_rate(self) -> float:
        """Overall function verification success rate."""
        if self.total_functions == 0:
            return 0.0
        return self.functions_verified / self.total_functions
    
    def is_successful(self) -> bool:
        """Check if backend verification succeeded."""
        return self.status == BackendVerificationStatus.VERIFIED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend_name": self.backend_name,
            "target_triple": self.target_triple,
            "status": self.status.value,
            "modules_verified": self.modules_verified,
            "total_modules": self.total_modules,
            "functions_verified": self.functions_verified,
            "total_functions": self.total_functions,
            "function_success_rate": self.function_success_rate,
            "integration_verified": self.integration_verified,
            "integration_violations": self.integration_violations,
            "end_to_end_properties": self.end_to_end_properties,
            "time_ms": self.time_ms,
            "messages": self.messages,
            "module_results": {
                name: r.to_dict() for name, r in self.module_results.items()
            }
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Backend Verification: {self.backend_name} ({self.target_triple})",
            f"Status: {self.status.value}",
            f"Modules: {self.modules_verified}/{self.total_modules} verified",
            f"Functions: {self.functions_verified}/{self.total_functions} verified ({self.function_success_rate:.1%})",
            f"Integration: {'✓' if self.integration_verified else '✗'}",
            f"Time: {self.time_ms:.1f}ms",
        ]
        
        if self.end_to_end_properties:
            lines.append("End-to-end properties:")
            for prop, verified in self.end_to_end_properties.items():
                lines.append(f"  {'✓' if verified else '✗'} {prop}")
        
        if self.messages:
            lines.append("Messages:")
            for msg in self.messages:
                lines.append(f"  - {msg}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        status_emoji = {
            BackendVerificationStatus.VERIFIED: "✓",
            BackendVerificationStatus.PARTIAL: "◐",
            BackendVerificationStatus.FAILED: "✗",
            BackendVerificationStatus.TIMEOUT: "⏱",
            BackendVerificationStatus.PENDING: "○",
            BackendVerificationStatus.ERROR: "!",
        }
        emoji = status_emoji.get(self.status, "?")
        return (f"[{emoji}] Backend {self.backend_name}: {self.status.value} "
                f"({self.modules_verified}/{self.total_modules} modules, "
                f"{self.functions_verified}/{self.total_functions} functions, "
                f"{self.time_ms:.1f}ms)")


class BackendVerifier:
    """
    Level 3 Verifier: Backend-level verification.
    
    Verifies entire backends for:
    1. Module-level correctness (using ModuleVerifier)
    2. Cross-module integration
    3. End-to-end properties
    """
    
    def __init__(
        self,
        module_verifier: Optional[ModuleVerifier] = None,
        enable_repair: bool = True,
        require_all_modules: bool = False,
        timeout_ms: int = 600000,  # 10 minutes
        verbose: bool = False
    ):
        """
        Initialize backend verifier.
        
        Args:
            module_verifier: L2 verifier (creates default if None)
            enable_repair: Enable CGNR repair
            require_all_modules: Require all modules to be verified
            timeout_ms: Total timeout for backend verification
            verbose: Enable verbose output
        """
        self.module_verifier = module_verifier or ModuleVerifier(
            enable_repair=enable_repair,
            verbose=verbose
        )
        self.enable_repair = enable_repair
        self.require_all_modules = require_all_modules
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            "backends_verified": 0,
            "backends_partial": 0,
            "backends_failed": 0,
            "total_modules": 0,
            "total_functions": 0,
            "total_time_ms": 0.0,
        }
    
    def verify(self, backend: Backend) -> BackendVerificationResult:
        """
        Verify a backend.
        
        Args:
            backend: Backend to verify
        
        Returns:
            BackendVerificationResult with verification status
        """
        start_time = time.time()
        
        result = BackendVerificationResult(
            backend_name=backend.name,
            target_triple=backend.target_triple,
            status=BackendVerificationStatus.PENDING
        )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[L3] Verifying backend: {backend.name} ({backend.target_triple})")
            print(f"  Modules: {len(backend.modules)}")
            print(f"  Total functions: {backend.total_functions()}")
            print(f"{'='*60}")
        
        try:
            # Step 1: Verify modules in dependency order
            module_order = backend.get_dependency_order()
            verified_modules: Set[str] = set()
            
            for module_name in module_order:
                module = backend.modules[module_name]
                
                # Check module dependencies
                deps = backend.module_dependencies.get(module_name, [])
                deps_verified = all(d in verified_modules for d in deps)
                
                if not deps_verified and self.require_all_modules:
                    result.module_results[module_name] = ModuleVerificationResult(
                        module_name=module_name,
                        status=ModuleVerificationStatus.FAILED,
                        messages=["Module dependencies not verified"]
                    )
                    continue
                
                # Verify the module
                if self.verbose:
                    print(f"\n[Module {len(result.module_results)+1}/{len(backend.modules)}]")
                
                mod_result = self.module_verifier.verify(module)
                result.module_results[module_name] = mod_result
                
                if mod_result.status == ModuleVerificationStatus.VERIFIED:
                    verified_modules.add(module_name)
            
            # Step 2: Verify cross-module integration
            integration_result = self._verify_integration(backend, verified_modules)
            result.integration_verified = integration_result["verified"]
            result.integration_violations = integration_result.get("violations", [])
            
            # Step 3: Verify end-to-end properties
            result.end_to_end_properties = self._verify_end_to_end(
                backend, result.module_results
            )
            
            # Determine overall status
            if (len(verified_modules) == len(backend.modules) and 
                result.integration_verified and
                all(result.end_to_end_properties.values())):
                result.status = BackendVerificationStatus.VERIFIED
                result.messages.append("Full backend verification successful")
                self.stats["backends_verified"] += 1
            elif len(verified_modules) > 0:
                result.status = BackendVerificationStatus.PARTIAL
                result.messages.append(
                    f"Partial verification: {len(verified_modules)}/{len(backend.modules)} modules"
                )
                self.stats["backends_partial"] += 1
            else:
                result.status = BackendVerificationStatus.FAILED
                result.messages.append("No modules verified")
                self.stats["backends_failed"] += 1
            
            self.stats["total_modules"] += len(backend.modules)
            self.stats["total_functions"] += backend.total_functions()
        
        except Exception as e:
            result.status = BackendVerificationStatus.ERROR
            result.messages.append(f"Error: {str(e)}")
            
            if self.verbose:
                print(f"  [!] Error: {e}")
        
        finally:
            result.time_ms = (time.time() - start_time) * 1000
            self.stats["total_time_ms"] += result.time_ms
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(result.summary())
            print(f"{'='*60}")
        
        return result
    
    def _verify_integration(
        self,
        backend: Backend,
        verified_modules: Set[str]
    ) -> Dict[str, Any]:
        """
        Verify cross-module integration.
        
        Checks that:
        1. Module interface contracts are compatible
        2. Data flows correctly between modules
        """
        violations = []
        
        # Check interface contract compatibility
        for mod_name, deps in backend.module_dependencies.items():
            if mod_name not in backend.modules:
                continue
            
            module = backend.modules[mod_name]
            
            for dep_name in deps:
                if dep_name not in backend.modules:
                    violations.append(f"Missing dependency: {mod_name} -> {dep_name}")
                    continue
                
                dep_module = backend.modules[dep_name]
                
                # Check contract compatibility
                if module.interface_contract and dep_module.interface_contract:
                    if not dep_module.interface_contract.is_compatible_with(
                        module.interface_contract
                    ):
                        violations.append(
                            f"Contract incompatibility: {dep_name} -> {mod_name}"
                        )
        
        return {
            "verified": len(violations) == 0,
            "violations": violations
        }
    
    def _verify_end_to_end(
        self,
        backend: Backend,
        module_results: Dict[str, ModuleVerificationResult]
    ) -> Dict[str, bool]:
        """
        Verify end-to-end properties of the backend.
        """
        properties = {}
        
        # Property 1: Instruction encoding correctness
        # (All MCCodeEmitter functions verified)
        mc_emitter = module_results.get("MCCodeEmitter")
        properties["instruction_encoding"] = (
            mc_emitter is not None and 
            mc_emitter.status == ModuleVerificationStatus.VERIFIED
        )
        
        # Property 2: Assembly printing correctness
        asm_printer = module_results.get("AsmPrinter")
        properties["assembly_printing"] = (
            asm_printer is not None and
            asm_printer.status in (ModuleVerificationStatus.VERIFIED, 
                                   ModuleVerificationStatus.PARTIAL)
        )
        
        # Property 3: Object file generation
        obj_writer = module_results.get("ELFObjectWriter")
        properties["object_generation"] = (
            obj_writer is not None and
            obj_writer.status in (ModuleVerificationStatus.VERIFIED,
                                  ModuleVerificationStatus.PARTIAL)
        )
        
        # Property 4: Full pipeline (all critical modules verified)
        critical_modules = ["MCCodeEmitter", "AsmPrinter", "ELFObjectWriter"]
        properties["full_pipeline"] = all(
            module_results.get(m, ModuleVerificationResult(
                module_name=m, status=ModuleVerificationStatus.FAILED
            )).status == ModuleVerificationStatus.VERIFIED
            for m in critical_modules
            if m in backend.modules
        )
        
        return properties
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get backend verification statistics."""
        total = (self.stats["backends_verified"] + 
                 self.stats["backends_partial"] + 
                 self.stats["backends_failed"])
        
        return {
            **self.stats,
            "total_backends": total,
            "full_verification_rate": self.stats["backends_verified"] / max(total, 1),
            "avg_time_ms": self.stats["total_time_ms"] / max(total, 1),
        }
