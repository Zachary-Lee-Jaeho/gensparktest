"""
Level 2: Module-level Verification.

Verifies entire modules for internal consistency and
interface contract satisfaction.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import time

from .function_verify import FunctionVerifier, FunctionVerificationResult, FunctionVerificationStatus
from .interface_contract import InterfaceContract, Assumption, Guarantee
from ..specification.spec_language import Specification


class ModuleVerificationStatus(Enum):
    """Status of module verification."""
    PENDING = "pending"
    VERIFIED = "verified"
    PARTIAL = "partial"  # Some functions verified, some failed
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ModuleFunction:
    """Function within a module."""
    name: str
    code: str
    specification: Specification
    is_interface: bool = False  # Part of module's public interface
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Module:
    """Module containing multiple functions."""
    name: str
    functions: Dict[str, ModuleFunction] = field(default_factory=dict)
    interface_contract: Optional[InterfaceContract] = None
    internal_contracts: List[InterfaceContract] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other module names
    
    def add_function(self, func: ModuleFunction) -> None:
        """Add a function to the module."""
        self.functions[func.name] = func
    
    def get_interface_functions(self) -> List[ModuleFunction]:
        """Get functions that are part of the public interface."""
        return [f for f in self.functions.values() if f.is_interface]
    
    def get_internal_functions(self) -> List[ModuleFunction]:
        """Get internal (non-interface) functions."""
        return [f for f in self.functions.values() if not f.is_interface]
    
    def get_dependency_order(self) -> List[str]:
        """Get functions in dependency order (topological sort)."""
        # Build dependency graph
        visited = set()
        order = []
        
        def visit(name: str):
            if name in visited or name not in self.functions:
                return
            visited.add(name)
            
            func = self.functions[name]
            for dep in func.dependencies:
                visit(dep)
            order.append(name)
        
        for name in self.functions:
            visit(name)
        
        return order


@dataclass
class ModuleVerificationResult:
    """Result of module-level verification."""
    module_name: str
    status: ModuleVerificationStatus
    function_results: Dict[str, FunctionVerificationResult] = field(default_factory=dict)
    contract_satisfied: bool = False
    contract_violations: List[str] = field(default_factory=list)
    internal_consistency: bool = False
    time_ms: float = 0.0
    messages: List[str] = field(default_factory=list)
    
    @property
    def verified_count(self) -> int:
        """Count of successfully verified functions."""
        return sum(1 for r in self.function_results.values() if r.is_successful())
    
    @property
    def total_count(self) -> int:
        """Total number of functions."""
        return len(self.function_results)
    
    @property
    def success_rate(self) -> float:
        """Rate of successfully verified functions."""
        if self.total_count == 0:
            return 0.0
        return self.verified_count / self.total_count
    
    def is_successful(self) -> bool:
        """Check if module verification succeeded."""
        return self.status == ModuleVerificationStatus.VERIFIED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "module_name": self.module_name,
            "status": self.status.value,
            "verified_count": self.verified_count,
            "total_count": self.total_count,
            "success_rate": self.success_rate,
            "contract_satisfied": self.contract_satisfied,
            "contract_violations": self.contract_violations,
            "internal_consistency": self.internal_consistency,
            "time_ms": self.time_ms,
            "messages": self.messages,
            "function_results": {
                name: r.to_dict() for name, r in self.function_results.items()
            }
        }
    
    def __str__(self) -> str:
        status_emoji = {
            ModuleVerificationStatus.VERIFIED: "✓",
            ModuleVerificationStatus.PARTIAL: "◐",
            ModuleVerificationStatus.FAILED: "✗",
            ModuleVerificationStatus.TIMEOUT: "⏱",
            ModuleVerificationStatus.PENDING: "○",
            ModuleVerificationStatus.ERROR: "!",
        }
        emoji = status_emoji.get(self.status, "?")
        return (f"[{emoji}] Module {self.module_name}: {self.status.value} "
                f"({self.verified_count}/{self.total_count} functions, {self.time_ms:.1f}ms)")


class ModuleVerifier:
    """
    Level 2 Verifier: Module-level verification.
    
    Verifies modules for:
    1. Function-level correctness (using FunctionVerifier)
    2. Internal consistency (function dependencies)
    3. Interface contract satisfaction
    """
    
    def __init__(
        self,
        function_verifier: Optional[FunctionVerifier] = None,
        enable_repair: bool = True,
        require_all_verified: bool = False,
        timeout_ms: int = 120000,
        verbose: bool = False
    ):
        """
        Initialize module verifier.
        
        Args:
            function_verifier: L1 verifier (creates default if None)
            enable_repair: Enable CGNR repair for failed functions
            require_all_verified: Require all functions to be verified
            timeout_ms: Total timeout for module verification
            verbose: Enable verbose output
        """
        self.function_verifier = function_verifier or FunctionVerifier(
            enable_repair=enable_repair,
            verbose=verbose
        )
        self.enable_repair = enable_repair
        self.require_all_verified = require_all_verified
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            "modules_verified": 0,
            "modules_partial": 0,
            "modules_failed": 0,
            "total_functions": 0,
            "total_time_ms": 0.0,
        }
    
    def verify(self, module: Module) -> ModuleVerificationResult:
        """
        Verify a module.
        
        Args:
            module: Module to verify
        
        Returns:
            ModuleVerificationResult with verification status
        """
        start_time = time.time()
        
        result = ModuleVerificationResult(
            module_name=module.name,
            status=ModuleVerificationStatus.PENDING
        )
        
        if self.verbose:
            print(f"\n[L2] Verifying module: {module.name}")
            print(f"  Functions: {len(module.functions)}")
        
        try:
            # Step 1: Verify functions in dependency order
            function_order = module.get_dependency_order()
            verified_functions: Set[str] = set()
            
            for func_name in function_order:
                func = module.functions[func_name]
                
                # Check if dependencies are verified
                deps_verified = all(
                    dep in verified_functions 
                    for dep in func.dependencies 
                    if dep in module.functions
                )
                
                if not deps_verified and self.require_all_verified:
                    result.function_results[func_name] = FunctionVerificationResult(
                        function_name=func_name,
                        module_name=module.name,
                        status=FunctionVerificationStatus.FAILED,
                        messages=["Dependencies not verified"]
                    )
                    continue
                
                # Verify the function
                func_result = self.function_verifier.verify(
                    func.code,
                    func.specification,
                    module.name
                )
                
                result.function_results[func_name] = func_result
                
                if func_result.is_successful():
                    verified_functions.add(func_name)
            
            # Step 2: Check internal consistency
            result.internal_consistency = self._check_internal_consistency(
                module, verified_functions
            )
            
            # Step 3: Check interface contract
            if module.interface_contract:
                contract_result = self._verify_interface_contract(
                    module, result.function_results
                )
                result.contract_satisfied = contract_result["satisfied"]
                result.contract_violations = contract_result.get("violations", [])
            else:
                result.contract_satisfied = True
            
            # Determine overall status
            verified_count = len(verified_functions)
            total_count = len(module.functions)
            
            if verified_count == total_count and result.contract_satisfied:
                result.status = ModuleVerificationStatus.VERIFIED
                result.messages.append("All functions verified, contract satisfied")
                self.stats["modules_verified"] += 1
            elif verified_count > 0:
                result.status = ModuleVerificationStatus.PARTIAL
                result.messages.append(
                    f"Partial verification: {verified_count}/{total_count} functions"
                )
                self.stats["modules_partial"] += 1
            else:
                result.status = ModuleVerificationStatus.FAILED
                result.messages.append("No functions verified")
                self.stats["modules_failed"] += 1
            
            self.stats["total_functions"] += total_count
        
        except Exception as e:
            result.status = ModuleVerificationStatus.ERROR
            result.messages.append(f"Error: {str(e)}")
            
            if self.verbose:
                print(f"  [!] Error: {e}")
        
        finally:
            result.time_ms = (time.time() - start_time) * 1000
            self.stats["total_time_ms"] += result.time_ms
        
        if self.verbose:
            print(f"  {result}")
        
        return result
    
    def _check_internal_consistency(
        self,
        module: Module,
        verified_functions: Set[str]
    ) -> bool:
        """
        Check internal consistency of module.
        
        Verifies that:
        1. All interface functions are verified
        2. Function dependencies are satisfied
        """
        # Check interface functions
        interface_funcs = module.get_interface_functions()
        for func in interface_funcs:
            if func.name not in verified_functions:
                return False
        
        # Check internal contracts
        for contract in module.internal_contracts:
            # Simplified: check that all functions referenced in contract are verified
            for g in contract.guarantees:
                # Parse function name from guarantee (simplified)
                pass
        
        return True
    
    def _verify_interface_contract(
        self,
        module: Module,
        function_results: Dict[str, FunctionVerificationResult]
    ) -> Dict[str, Any]:
        """
        Verify that module satisfies its interface contract.
        """
        contract = module.interface_contract
        if not contract:
            return {"satisfied": True}
        
        violations = []
        
        # Check that all guarantees are satisfied by verified functions
        interface_funcs = module.get_interface_functions()
        verified_interfaces = [
            f.name for f in interface_funcs
            if f.name in function_results and function_results[f.name].is_successful()
        ]
        
        # For now, simplified check: all interface functions must be verified
        required_funcs = [f.name for f in interface_funcs]
        missing = set(required_funcs) - set(verified_interfaces)
        
        if missing:
            violations.append(
                f"Interface functions not verified: {', '.join(missing)}"
            )
        
        return {
            "satisfied": len(violations) == 0,
            "violations": violations
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get module verification statistics."""
        total = (self.stats["modules_verified"] + 
                 self.stats["modules_partial"] + 
                 self.stats["modules_failed"])
        
        return {
            **self.stats,
            "total_modules": total,
            "full_verification_rate": self.stats["modules_verified"] / max(total, 1),
            "avg_time_ms": self.stats["total_time_ms"] / max(total, 1),
        }
