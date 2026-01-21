"""
Interface Contracts for Assume-Guarantee Reasoning.

Defines interface contracts that enable compositional verification
across module boundaries.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import json


class ContractType(Enum):
    """Types of interface contracts."""
    FUNCTION = "function"
    MODULE = "module"
    BACKEND = "backend"


@dataclass
class Assumption:
    """
    Assumption about module inputs/environment.
    
    Represents what a module assumes about its environment
    for the guarantee to hold.
    """
    name: str
    description: str
    condition: str  # SMT-LIB format condition
    scope: str = "input"  # input, state, or environment
    
    def to_smt(self) -> str:
        """Convert assumption to SMT-LIB format."""
        return f"; Assumption: {self.name}\n{self.condition}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "scope": self.scope
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Assumption':
        return cls(
            name=data["name"],
            description=data["description"],
            condition=data["condition"],
            scope=data.get("scope", "input")
        )


@dataclass
class Guarantee:
    """
    Guarantee about module outputs/effects.
    
    Represents what a module guarantees about its behavior
    when assumptions are met.
    """
    name: str
    description: str
    condition: str  # SMT-LIB format condition
    scope: str = "output"  # output, state, or effect
    
    def to_smt(self) -> str:
        """Convert guarantee to SMT-LIB format."""
        return f"; Guarantee: {self.name}\n{self.condition}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "scope": self.scope
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Guarantee':
        return cls(
            name=data["name"],
            description=data["description"],
            condition=data["condition"],
            scope=data.get("scope", "output")
        )


@dataclass
class InterfaceContract:
    """
    Interface Contract for modular verification.
    
    An interface contract specifies:
    - Assumptions (A): What the module expects from its environment
    - Guarantees (G): What the module provides when assumptions hold
    - Dependencies: Other modules this contract depends on
    
    The assume-guarantee rule:
        If module M satisfies A → G, and environment E satisfies A,
        then composition E || M satisfies G.
    """
    name: str
    module_name: str
    contract_type: ContractType
    assumptions: List[Assumption] = field(default_factory=list)
    guarantees: List[Guarantee] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_assumption(self, assumption: Assumption) -> None:
        """Add an assumption to the contract."""
        self.assumptions.append(assumption)
    
    def add_guarantee(self, guarantee: Guarantee) -> None:
        """Add a guarantee to the contract."""
        self.guarantees.append(guarantee)
    
    def add_dependency(self, module_name: str) -> None:
        """Add a module dependency."""
        if module_name not in self.dependencies:
            self.dependencies.append(module_name)
    
    def to_smt(self) -> str:
        """
        Convert contract to SMT-LIB format.
        
        Returns the verification condition:
            (and assumptions) => (and guarantees)
        """
        lines = [
            f"; Interface Contract: {self.name}",
            f"; Module: {self.module_name}",
            f"; Type: {self.contract_type.value}",
            ""
        ]
        
        # Assumptions
        if self.assumptions:
            lines.append("; === ASSUMPTIONS ===")
            assumption_conds = []
            for a in self.assumptions:
                lines.append(a.to_smt())
                assumption_conds.append(f"({a.condition})")
            
            if len(assumption_conds) == 1:
                lines.append(f"(define-fun assumptions () Bool {assumption_conds[0]})")
            else:
                lines.append(f"(define-fun assumptions () Bool (and {' '.join(assumption_conds)}))")
        else:
            lines.append("(define-fun assumptions () Bool true)")
        
        lines.append("")
        
        # Guarantees
        if self.guarantees:
            lines.append("; === GUARANTEES ===")
            guarantee_conds = []
            for g in self.guarantees:
                lines.append(g.to_smt())
                guarantee_conds.append(f"({g.condition})")
            
            if len(guarantee_conds) == 1:
                lines.append(f"(define-fun guarantees () Bool {guarantee_conds[0]})")
            else:
                lines.append(f"(define-fun guarantees () Bool (and {' '.join(guarantee_conds)}))")
        else:
            lines.append("(define-fun guarantees () Bool true)")
        
        lines.append("")
        
        # Verification condition: assumptions => guarantees
        lines.append("; === VERIFICATION CONDITION ===")
        lines.append("(assert (not (=> (assumptions) (guarantees))))")
        lines.append("(check-sat)")
        
        return "\n".join(lines)
    
    def is_compatible_with(self, other: 'InterfaceContract') -> bool:
        """
        Check if this contract is compatible with another.
        
        Two contracts are compatible if:
        1. This contract's guarantees satisfy the other's assumptions
        2. Dependencies are properly ordered
        """
        # Check if our guarantees satisfy their assumptions
        our_guarantee_names = {g.name for g in self.guarantees}
        their_assumption_names = {a.name for a in other.assumptions}
        
        # Simplified check: names should match
        # In full implementation, would do semantic checking
        required = their_assumption_names - our_guarantee_names
        
        return len(required) == 0
    
    def merge_with(self, other: 'InterfaceContract') -> 'InterfaceContract':
        """
        Merge two contracts for compositional reasoning.
        
        Creates a new contract where:
        - Assumptions are the union of both assumptions (minus satisfied ones)
        - Guarantees are the union of both guarantees
        """
        merged = InterfaceContract(
            name=f"{self.name}_{other.name}",
            module_name=f"{self.module_name}+{other.module_name}",
            contract_type=ContractType.MODULE
        )
        
        # Collect guarantees from both
        guarantee_names = set()
        for g in self.guarantees + other.guarantees:
            if g.name not in guarantee_names:
                merged.guarantees.append(g)
                guarantee_names.add(g.name)
        
        # Assumptions: exclude those satisfied by guarantees
        for a in self.assumptions + other.assumptions:
            if a.name not in guarantee_names:
                merged.assumptions.append(a)
        
        # Merge dependencies
        merged.dependencies = list(set(self.dependencies + other.dependencies))
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "module_name": self.module_name,
            "contract_type": self.contract_type.value,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "guarantees": [g.to_dict() for g in self.guarantees],
            "dependencies": self.dependencies,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterfaceContract':
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            module_name=data["module_name"],
            contract_type=ContractType(data["contract_type"]),
            assumptions=[Assumption.from_dict(a) for a in data.get("assumptions", [])],
            guarantees=[Guarantee.from_dict(g) for g in data.get("guarantees", [])],
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {})
        )
    
    def save(self, path: str) -> None:
        """Save contract to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'InterfaceContract':
        """Load contract from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"InterfaceContract: {self.name}",
            f"  Module: {self.module_name}",
            f"  Type: {self.contract_type.value}",
            f"  Assumptions ({len(self.assumptions)}):"
        ]
        for a in self.assumptions:
            lines.append(f"    - {a.name}: {a.description}")
        
        lines.append(f"  Guarantees ({len(self.guarantees)}):")
        for g in self.guarantees:
            lines.append(f"    - {g.name}: {g.description}")
        
        if self.dependencies:
            lines.append(f"  Dependencies: {', '.join(self.dependencies)}")
        
        return "\n".join(lines)


# Predefined contract templates for LLVM backend modules
def create_mc_code_emitter_contract(target: str) -> InterfaceContract:
    """Create contract for MCCodeEmitter module."""
    contract = InterfaceContract(
        name=f"{target}_MCCodeEmitter_IFC",
        module_name="MCCodeEmitter",
        contract_type=ContractType.MODULE
    )
    
    # Assumptions
    contract.add_assumption(Assumption(
        name="valid_instruction",
        description="Input instruction is well-formed",
        condition="(and (>= opcode 0) (< opcode MAX_OPCODE))"
    ))
    contract.add_assumption(Assumption(
        name="valid_operands",
        description="All operands are valid for the instruction",
        condition="(forall ((i Int)) (=> (and (>= i 0) (< i num_operands)) (valid_operand (operand i))))"
    ))
    
    # Guarantees
    contract.add_guarantee(Guarantee(
        name="correct_encoding",
        description="Emitted bytes correctly encode the instruction",
        condition="(= (decode (emit MI)) MI)"
    ))
    contract.add_guarantee(Guarantee(
        name="encoding_size",
        description="Encoding size matches instruction specification",
        condition="(= (size (emit MI)) (expected_size MI))"
    ))
    
    return contract


def create_asm_printer_contract(target: str) -> InterfaceContract:
    """Create contract for AsmPrinter module."""
    contract = InterfaceContract(
        name=f"{target}_AsmPrinter_IFC",
        module_name="AsmPrinter",
        contract_type=ContractType.MODULE
    )
    
    # Assumptions
    contract.add_assumption(Assumption(
        name="valid_mc_inst",
        description="MCInst is well-formed",
        condition="(valid_mcinst MI)"
    ))
    
    # Guarantees
    contract.add_guarantee(Guarantee(
        name="parseable_output",
        description="Output can be parsed back to equivalent instruction",
        condition="(equivalent (parse (print MI)) MI)"
    ))
    contract.add_guarantee(Guarantee(
        name="correct_mnemonic",
        description="Mnemonic matches opcode",
        condition="(= (mnemonic (print MI)) (expected_mnemonic (opcode MI)))"
    ))
    
    contract.add_dependency("MCCodeEmitter")
    
    return contract


def create_elf_object_writer_contract(target: str) -> InterfaceContract:
    """Create contract for ELFObjectWriter module."""
    contract = InterfaceContract(
        name=f"{target}_ELFObjectWriter_IFC",
        module_name="ELFObjectWriter",
        contract_type=ContractType.MODULE
    )
    
    # Assumptions
    contract.add_assumption(Assumption(
        name="valid_fixup",
        description="Fixup is valid for the target",
        condition="(valid_fixup Fixup Target)"
    ))
    
    # Guarantees
    contract.add_guarantee(Guarantee(
        name="correct_reloc_type",
        description="Relocation type is correct for fixup kind",
        condition="(correct_reloc_mapping (getRelocType Fixup Target IsPCRel))"
    ))
    
    contract.add_dependency("MCCodeEmitter")
    contract.add_dependency("AsmPrinter")
    
    return contract
