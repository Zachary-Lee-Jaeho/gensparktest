"""
Integration tests for Hierarchical Verification (Phase 4).

Tests the complete hierarchical verification workflow:
- Level 1: Function-level verification
- Level 2: Module-level verification with contract satisfaction
- Level 3: Backend-level verification with cross-module integration
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hierarchical import (
    HierarchicalVerifier,
    FunctionVerifier,
    ModuleVerifier,
    BackendVerifier,
    HierarchicalResult,
    VerificationLevel,
)
from src.hierarchical.interface_contract import (
    InterfaceContract,
    ContractType,
    Assumption,
    Guarantee,
    create_mc_code_emitter_contract,
    create_asm_printer_contract,
    create_elf_object_writer_contract,
)
from src.hierarchical.function_verify import (
    FunctionVerificationResult,
    FunctionVerificationStatus,
)
from src.hierarchical.module_verify import (
    Module,
    ModuleFunction,
    ModuleVerificationResult,
    ModuleVerificationStatus,
)
from src.hierarchical.backend_verify import (
    Backend,
    BackendVerificationResult,
    BackendVerificationStatus,
)
from src.specification import Specification
from src.specification.spec_language import Condition, Variable, Constant, ConditionType


class TestInterfaceContract:
    """Tests for Interface Contract system."""
    
    def test_assumption_creation(self):
        """Test creating an assumption."""
        assumption = Assumption(
            name="valid_input",
            description="Input must be valid",
            condition="(>= x 0)",
            scope="input"
        )
        
        assert assumption.name == "valid_input"
        assert "(>= x 0)" in assumption.to_smt()
    
    def test_guarantee_creation(self):
        """Test creating a guarantee."""
        guarantee = Guarantee(
            name="correct_output",
            description="Output is correctly computed",
            condition="(= result (+ x y))",
            scope="output"
        )
        
        assert guarantee.name == "correct_output"
        assert "(= result" in guarantee.to_smt()
    
    def test_contract_creation(self):
        """Test creating an interface contract."""
        contract = InterfaceContract(
            name="test_contract",
            module_name="TestModule",
            contract_type=ContractType.MODULE
        )
        
        contract.add_assumption(Assumption(
            name="pre1",
            description="First precondition",
            condition="(> x 0)"
        ))
        
        contract.add_guarantee(Guarantee(
            name="post1",
            description="First postcondition",
            condition="(>= result 0)"
        ))
        
        assert len(contract.assumptions) == 1
        assert len(contract.guarantees) == 1
        assert contract.contract_type == ContractType.MODULE
    
    def test_contract_to_smt(self):
        """Test converting contract to SMT-LIB format."""
        contract = InterfaceContract(
            name="test",
            module_name="Test",
            contract_type=ContractType.FUNCTION
        )
        
        contract.add_assumption(Assumption("a1", "desc", "(> x 0)"))
        contract.add_guarantee(Guarantee("g1", "desc", "(>= y 0)"))
        
        smt = contract.to_smt()
        
        assert "assumptions" in smt
        assert "guarantees" in smt
        assert "check-sat" in smt
    
    def test_contract_serialization(self):
        """Test contract serialization/deserialization."""
        contract = InterfaceContract(
            name="test",
            module_name="Test",
            contract_type=ContractType.MODULE
        )
        contract.add_assumption(Assumption("a1", "desc", "true"))
        
        # Serialize
        data = contract.to_dict()
        
        # Deserialize
        restored = InterfaceContract.from_dict(data)
        
        assert restored.name == contract.name
        assert len(restored.assumptions) == len(contract.assumptions)
    
    def test_contract_compatibility(self):
        """Test contract compatibility checking."""
        provider = InterfaceContract(
            name="provider",
            module_name="Provider",
            contract_type=ContractType.MODULE
        )
        provider.add_guarantee(Guarantee("data_valid", "Data is valid", "true"))
        
        consumer = InterfaceContract(
            name="consumer",
            module_name="Consumer",
            contract_type=ContractType.MODULE
        )
        consumer.add_assumption(Assumption("data_valid", "Data must be valid", "true"))
        
        assert provider.is_compatible_with(consumer)
    
    def test_contract_merge(self):
        """Test merging two contracts."""
        c1 = InterfaceContract("c1", "M1", ContractType.MODULE)
        c1.add_assumption(Assumption("a1", "desc", "true"))
        c1.add_guarantee(Guarantee("g1", "desc", "true"))
        
        c2 = InterfaceContract("c2", "M2", ContractType.MODULE)
        c2.add_assumption(Assumption("g1", "desc", "true"))  # Satisfied by c1
        c2.add_guarantee(Guarantee("g2", "desc", "true"))
        
        merged = c1.merge_with(c2)
        
        # g1 should not be in merged assumptions (satisfied by c1's guarantee)
        assumption_names = [a.name for a in merged.assumptions]
        assert "g1" not in assumption_names
        assert "a1" in assumption_names
    
    def test_predefined_contracts(self):
        """Test predefined contract templates."""
        mc_contract = create_mc_code_emitter_contract("RISCV")
        assert mc_contract.module_name == "MCCodeEmitter"
        assert len(mc_contract.assumptions) > 0
        assert len(mc_contract.guarantees) > 0
        
        asm_contract = create_asm_printer_contract("RISCV")
        assert "MCCodeEmitter" in asm_contract.dependencies
        
        elf_contract = create_elf_object_writer_contract("RISCV")
        assert len(elf_contract.dependencies) >= 2


class TestFunctionVerifier:
    """Tests for Level 1: Function Verification."""
    
    @pytest.fixture
    def verifier(self):
        return FunctionVerifier(enable_repair=True, verbose=False)
    
    @pytest.fixture
    def simple_spec(self):
        return Specification(
            function_name="testFunc",
            preconditions=[Condition(ConditionType.IS_VALID, Variable("x"))],
            postconditions=[Condition(ConditionType.GREATER_EQUAL, Variable("result"), Constant(0))]
        )
    
    @pytest.fixture
    def simple_code(self):
        return """
        unsigned testFunc(int x) {
            if (x < 0) return 0;
            return x * 2;
        }
        """
    
    def test_verifier_creation(self, verifier):
        """Test verifier creation."""
        assert verifier.verifier is not None
        assert verifier.enable_repair is True
    
    def test_verify_simple_function(self, verifier, simple_code, simple_spec):
        """Test verifying a simple function."""
        result = verifier.verify(simple_code, simple_spec, "TestModule")
        
        assert result.function_name == "testFunc"
        assert result.module_name == "TestModule"
        assert result.time_ms > 0
    
    def test_verification_result_attributes(self, verifier, simple_code, simple_spec):
        """Test verification result attributes."""
        result = verifier.verify(simple_code, simple_spec)
        
        assert hasattr(result, 'status')
        assert hasattr(result, 'original_code')
        assert hasattr(result, 'specification')
        assert result.to_dict() is not None
    
    def test_batch_verification(self, verifier, simple_code, simple_spec):
        """Test batch verification."""
        functions = [
            (simple_code, simple_spec, "Module1"),
            (simple_code, simple_spec, "Module2"),
        ]
        
        results = verifier.verify_batch(functions)
        
        assert len(results) == 2
    
    def test_statistics_tracking(self, verifier, simple_code, simple_spec):
        """Test statistics tracking."""
        verifier.reset_statistics()
        
        verifier.verify(simple_code, simple_spec)
        
        stats = verifier.get_statistics()
        assert stats["total_processed"] >= 1


class TestModuleVerifier:
    """Tests for Level 2: Module Verification."""
    
    @pytest.fixture
    def verifier(self):
        return ModuleVerifier(enable_repair=True, verbose=False)
    
    @pytest.fixture
    def simple_module(self):
        module = Module(name="TestModule")
        
        # Add functions
        func1 = ModuleFunction(
            name="func1",
            code="unsigned func1() { return 0; }",
            specification=Specification(function_name="func1"),
            is_interface=True
        )
        
        func2 = ModuleFunction(
            name="func2",
            code="unsigned func2() { return func1() + 1; }",
            specification=Specification(function_name="func2"),
            dependencies=["func1"]
        )
        
        module.add_function(func1)
        module.add_function(func2)
        
        return module
    
    def test_module_creation(self, simple_module):
        """Test module creation."""
        assert simple_module.name == "TestModule"
        assert len(simple_module.functions) == 2
    
    def test_module_function_ordering(self, simple_module):
        """Test dependency ordering."""
        order = simple_module.get_dependency_order()
        
        # func1 should come before func2
        assert order.index("func1") < order.index("func2")
    
    def test_interface_function_filtering(self, simple_module):
        """Test filtering interface functions."""
        interface_funcs = simple_module.get_interface_functions()
        internal_funcs = simple_module.get_internal_functions()
        
        assert len(interface_funcs) == 1
        assert interface_funcs[0].name == "func1"
        assert len(internal_funcs) == 1
    
    def test_module_verification(self, verifier, simple_module):
        """Test module verification."""
        result = verifier.verify(simple_module)
        
        assert result.module_name == "TestModule"
        assert result.total_count == 2
        assert result.time_ms > 0
    
    def test_module_with_contract(self, verifier):
        """Test module with interface contract."""
        module = Module(name="ContractModule")
        
        module.add_function(ModuleFunction(
            name="api_func",
            code="int api_func(int x) { return x > 0 ? x : 0; }",
            specification=Specification(function_name="api_func"),
            is_interface=True
        ))
        
        contract = InterfaceContract(
            name="ContractModule_IFC",
            module_name="ContractModule",
            contract_type=ContractType.MODULE
        )
        contract.add_assumption(Assumption("valid_input", "Input is valid", "true"))
        contract.add_guarantee(Guarantee("positive_result", "Result >= 0", "(>= result 0)"))
        
        module.interface_contract = contract
        
        result = verifier.verify(module)
        
        assert result.module_name == "ContractModule"


class TestBackendVerifier:
    """Tests for Level 3: Backend Verification."""
    
    @pytest.fixture
    def verifier(self):
        return BackendVerifier(enable_repair=True, verbose=False)
    
    @pytest.fixture
    def simple_backend(self):
        backend = Backend(
            name="TestBackend",
            target_triple="test-unknown-linux-gnu"
        )
        
        # Create modules
        emitter = Module(name="MCCodeEmitter")
        emitter.add_function(ModuleFunction(
            name="encodeInstruction",
            code="void encodeInstruction(const MCInst &MI) { }",
            specification=Specification(function_name="encodeInstruction"),
            is_interface=True
        ))
        emitter.interface_contract = create_mc_code_emitter_contract("Test")
        
        printer = Module(name="AsmPrinter")
        printer.add_function(ModuleFunction(
            name="emitInstruction",
            code="void emitInstruction(const MachineInstr *MI) { }",
            specification=Specification(function_name="emitInstruction"),
            is_interface=True
        ))
        printer.interface_contract = create_asm_printer_contract("Test")
        
        backend.add_module(emitter)
        backend.add_module(printer)
        backend.set_dependencies("AsmPrinter", ["MCCodeEmitter"])
        
        return backend
    
    def test_backend_creation(self, simple_backend):
        """Test backend creation."""
        assert simple_backend.name == "TestBackend"
        assert len(simple_backend.modules) == 2
    
    def test_backend_dependency_order(self, simple_backend):
        """Test module dependency ordering."""
        order = simple_backend.get_dependency_order()
        
        # MCCodeEmitter should come before AsmPrinter
        assert order.index("MCCodeEmitter") < order.index("AsmPrinter")
    
    def test_backend_verification(self, verifier, simple_backend):
        """Test backend verification."""
        result = verifier.verify(simple_backend)
        
        assert result.backend_name == "TestBackend"
        assert result.total_modules == 2
        assert result.time_ms > 0
    
    def test_end_to_end_properties(self, verifier, simple_backend):
        """Test end-to-end property verification."""
        result = verifier.verify(simple_backend)
        
        assert "instruction_encoding" in result.end_to_end_properties
        assert "assembly_printing" in result.end_to_end_properties
    
    def test_result_summary(self, verifier, simple_backend):
        """Test result summary generation."""
        result = verifier.verify(simple_backend)
        
        summary = result.summary()
        
        assert "TestBackend" in summary
        assert "Modules:" in summary


class TestHierarchicalVerifier:
    """Tests for the unified Hierarchical Verifier."""
    
    @pytest.fixture
    def verifier(self):
        return HierarchicalVerifier(enable_repair=True, verbose=False)
    
    @pytest.fixture
    def simple_spec(self):
        return Specification(function_name="testFunc")
    
    def test_hierarchical_verifier_creation(self, verifier):
        """Test hierarchical verifier creation."""
        assert verifier.function_verifier is not None
        assert verifier.module_verifier is not None
        assert verifier.backend_verifier is not None
    
    def test_verify_function(self, verifier, simple_spec):
        """Test function-level verification through hierarchical verifier."""
        code = "int testFunc() { return 42; }"
        
        result = verifier.verify_function(code, simple_spec)
        
        assert isinstance(result, HierarchicalResult)
        assert result.level == VerificationLevel.FUNCTION
    
    def test_verify_module(self, verifier):
        """Test module-level verification through hierarchical verifier."""
        module = Module(name="TestModule")
        module.add_function(ModuleFunction(
            name="func",
            code="int func() { return 0; }",
            specification=Specification(function_name="func")
        ))
        
        result = verifier.verify_module(module)
        
        assert result.level == VerificationLevel.MODULE
    
    def test_verify_backend(self, verifier):
        """Test backend-level verification through hierarchical verifier."""
        backend = Backend(name="TestBackend", target_triple="test-linux-gnu")
        
        module = Module(name="TestModule")
        module.add_function(ModuleFunction(
            name="func",
            code="int func() { return 0; }",
            specification=Specification(function_name="func")
        ))
        backend.add_module(module)
        
        result = verifier.verify_backend(backend)
        
        assert result.level == VerificationLevel.BACKEND
    
    def test_auto_level_detection(self, verifier, simple_spec):
        """Test automatic verification level detection."""
        # Function-level (code string)
        result1 = verifier.verify("int f() { return 0; }")
        assert result1.level == VerificationLevel.FUNCTION
        
        # Module-level
        module = Module(name="M")
        result2 = verifier.verify(module)
        assert result2.level == VerificationLevel.MODULE
        
        # Backend-level
        backend = Backend(name="B", target_triple="test")
        result3 = verifier.verify(backend)
        assert result3.level == VerificationLevel.BACKEND
    
    def test_statistics(self, verifier, simple_spec):
        """Test statistics collection."""
        verifier.reset_statistics()
        
        code = "int testFunc() { return 0; }"
        verifier.verify_function(code, simple_spec)
        
        stats = verifier.get_statistics()
        
        assert stats["verifications_run"] >= 1


class TestIntegrationScenarios:
    """End-to-end integration scenarios."""
    
    def test_complete_riscv_backend_verification(self):
        """Test complete RISC-V backend verification scenario."""
        # Create a RISC-V-like backend
        backend = Backend(
            name="RISCV",
            target_triple="riscv64-unknown-linux-gnu"
        )
        
        # MCCodeEmitter module
        mc_emitter = Module(name="MCCodeEmitter")
        mc_emitter.add_function(ModuleFunction(
            name="encodeInstruction",
            code="""
            void RISCVMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                                       raw_ostream &OS) const {
                uint32_t Binary = getBinaryCodeForInstr(MI);
                support::endian::write<uint32_t>(OS, Binary, support::little);
            }
            """,
            specification=Specification(function_name="encodeInstruction"),
            is_interface=True
        ))
        mc_emitter.interface_contract = create_mc_code_emitter_contract("RISCV")
        
        # ELFObjectWriter module
        elf_writer = Module(name="ELFObjectWriter")
        elf_writer.add_function(ModuleFunction(
            name="getRelocType",
            code="""
            unsigned RISCVELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                                        const MCValue &Target,
                                                        bool IsPCRel) const {
                switch (Fixup.getTargetKind()) {
                case FK_NONE: return ELF::R_RISCV_NONE;
                case FK_Data_4: return ELF::R_RISCV_32;
                case FK_Data_8: return ELF::R_RISCV_64;
                default: llvm_unreachable("Unknown fixup!");
                }
            }
            """,
            specification=Specification(function_name="getRelocType"),
            is_interface=True
        ))
        elf_writer.interface_contract = create_elf_object_writer_contract("RISCV")
        
        backend.add_module(mc_emitter)
        backend.add_module(elf_writer)
        backend.set_dependencies("ELFObjectWriter", ["MCCodeEmitter"])
        
        # Verify
        verifier = HierarchicalVerifier(verbose=False)
        result = verifier.verify_backend(backend)
        
        # Check results
        assert result.backend_name == "RISCV"
        assert result.total_modules == 2
        assert result.total_functions >= 2
        
        # Print summary
        print("\n" + result.to_dict().__str__())


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
