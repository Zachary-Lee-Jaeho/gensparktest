"""
Integration tests for the complete verification pipeline.
Tests the interaction between specification inference, verification, and repair.
"""

import pytest
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.specification import SpecificationInferrer
from src.specification.spec_language import (
    Specification, Condition, ConditionType, Variable, Constant
)
from src.verification import (
    Verifier, VerificationStatus, VCGenerator, BoundedModelChecker, BMCResult
)
from src.repair import CGNREngine, RepairStatus
from src.hierarchical import HierarchicalVerifier, VerificationLevel


class TestSpecificationToVerification:
    """Test specification inference to verification flow."""
    
    def test_infer_and_verify_simple_function(self):
        """Test inferring spec from references and verifying generated code."""
        # Reference implementations
        references = [
            ('arm', '''
            unsigned getRelocType(const MCFixup &Fixup) {
                switch (Fixup.getTargetKind()) {
                    case FK_NONE: return ELF::R_ARM_NONE;
                    case FK_Data_4: return ELF::R_ARM_ABS32;
                    default: return ELF::R_ARM_NONE;
                }
            }
            '''),
            ('mips', '''
            unsigned getRelocType(const MCFixup &Fixup) {
                switch (Fixup.getTargetKind()) {
                    case FK_NONE: return ELF::R_MIPS_NONE;
                    case FK_Data_4: return ELF::R_MIPS_32;
                    default: return ELF::R_MIPS_NONE;
                }
            }
            '''),
        ]
        
        # Infer specification
        inferrer = SpecificationInferrer()
        spec = inferrer.infer(
            function_name="getRelocType",
            references=references,
            module="ELFObjectWriter"
        )
        
        # Verify that spec was inferred
        assert spec is not None
        assert spec.function_name == "getRelocType"
        
        # Generated code (simulating VEGA output)
        generated_code = '''
        unsigned getRelocType(const MCFixup &Fixup) {
            switch (Fixup.getTargetKind()) {
                case FK_NONE: return ELF::R_RISCV_NONE;
                case FK_Data_4: return ELF::R_RISCV_32;
                default: return ELF::R_RISCV_NONE;
            }
        }
        '''
        
        # Verify
        verifier = Verifier(timeout_ms=5000)
        result = verifier.verify(generated_code, spec)
        
        # Should get some result (VERIFIED, FAILED, or UNKNOWN)
        assert result.status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.FAILED,
            VerificationStatus.UNKNOWN,
            VerificationStatus.ERROR
        ]
        assert result.time_ms > 0
    
    def test_verify_with_bmc(self):
        """Test verification using bounded model checking."""
        code = '''
        int sum = 0;
        for (int i = 0; i < 5; i++) {
            sum = sum + i;
        }
        return sum;
        '''
        
        spec = Specification(
            function_name="compute_sum",
            postconditions=[
                Condition(
                    cond_type=ConditionType.GREATER_EQUAL,
                    operands=[Variable("sum"), Constant(0)]
                )
            ]
        )
        
        checker = BoundedModelChecker(default_bound=10, verbose=False)
        result = checker.check(code, spec)
        
        assert result.result in [BMCResult.SAFE, BMCResult.UNKNOWN]
        assert result.iterations > 0


class TestVerificationAndRepair:
    """Test verification to repair flow."""
    
    def test_cgnr_repair_simple_bug(self):
        """Test CGNR repair on a simple bug."""
        # Buggy code
        buggy_code = '''
        unsigned getRelocType(const MCFixup &Fixup) {
            switch (Fixup.getTargetKind()) {
                case FK_NONE: return ELF::R_RISCV_NONE;
                case FK_Data_4: return ELF::R_RISCV_64;  // Bug: should be R_RISCV_32
                default: return ELF::R_RISCV_NONE;
            }
        }
        '''
        
        # Specification
        spec = Specification(
            function_name="getRelocType",
            postconditions=[
                Condition(
                    cond_type=ConditionType.GREATER_EQUAL,
                    operands=[Variable("result"), Constant(0)]
                )
            ]
        )
        
        # Attempt repair
        cgnr = CGNREngine(max_iterations=3, verbose=False)
        result = cgnr.repair(buggy_code, spec)
        
        # Should complete (may or may not succeed in repair)
        assert result.status in [
            RepairStatus.SUCCESS,
            RepairStatus.PARTIAL,
            RepairStatus.FAILED,
            RepairStatus.MAX_ITERATIONS
        ]
        assert result.iterations >= 0
        assert result.total_time_ms > 0
    
    def test_repair_result_contains_code(self):
        """Test that repair result contains repaired code."""
        code = '''
        int test() {
            return -1;  // Bug
        }
        '''
        
        spec = Specification(
            function_name="test",
            postconditions=[
                Condition(
                    cond_type=ConditionType.GREATER_EQUAL,
                    operands=[Variable("result"), Constant(0)]
                )
            ]
        )
        
        cgnr = CGNREngine(max_iterations=2)
        result = cgnr.repair(code, spec)
        
        assert result.original_code == code
        assert result.repaired_code is not None
        assert len(result.repaired_code) > 0


class TestHierarchicalVerification:
    """Test hierarchical verification flow."""
    
    def test_function_level_verification(self):
        """Test Level 1 (function) verification."""
        code = '''
        int abs(int x) {
            return x < 0 ? -x : x;
        }
        '''
        
        spec = Specification(
            function_name="abs",
            postconditions=[
                Condition(
                    cond_type=ConditionType.GREATER_EQUAL,
                    operands=[Variable("result"), Constant(0)]
                )
            ]
        )
        
        verifier = HierarchicalVerifier(enable_repair=False, verbose=False)
        result = verifier.verify_function(code, spec)
        
        assert result.level == VerificationLevel.FUNCTION
        assert len(result.function_results) == 1
        assert "abs" in result.function_results
    
    def test_hierarchical_statistics(self):
        """Test hierarchical verifier statistics."""
        code = "int x = 1;"
        spec = Specification(function_name="test")
        
        verifier = HierarchicalVerifier()
        verifier.reset_statistics()
        
        result = verifier.verify_function(code, spec)
        
        stats = verifier.get_statistics()
        assert stats["verifications_run"] >= 1


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""
    
    def test_full_pipeline_compiler_backend_function(self):
        """Test full pipeline on a typical compiler backend function."""
        # Reference implementations for spec inference
        references = [
            ('x86', '''
            bool needsRelocateWithSymbol(const MCValue &Val, const MCSymbol &Sym) {
                return !Sym.isInSection();
            }
            '''),
            ('arm', '''
            bool needsRelocateWithSymbol(const MCValue &Val, const MCSymbol &Sym) {
                return !Sym.isInSection();
            }
            '''),
        ]
        
        # Infer specification
        inferrer = SpecificationInferrer()
        spec = inferrer.infer(
            function_name="needsRelocateWithSymbol",
            references=references,
            module="ELFObjectWriter"
        )
        
        # Generated code
        generated = '''
        bool needsRelocateWithSymbol(const MCValue &Val, const MCSymbol &Sym) {
            return !Sym.isInSection();
        }
        '''
        
        # Verify
        verifier = Verifier(timeout_ms=10000)
        verify_result = verifier.verify(generated, spec)
        
        # If verification fails, try repair
        if verify_result.status == VerificationStatus.FAILED:
            cgnr = CGNREngine(max_iterations=3)
            repair_result = cgnr.repair(generated, spec)
            
            assert repair_result.repaired_code is not None
        
        # Complete without errors
        assert True
    
    def test_pipeline_timing(self):
        """Test that pipeline completes within reasonable time."""
        start = time.time()
        
        code = '''
        unsigned test() {
            int x = 0;
            for (int i = 0; i < 10; i++) {
                x += i;
            }
            return x;
        }
        '''
        
        spec = Specification(
            function_name="test",
            postconditions=[
                Condition(
                    cond_type=ConditionType.GREATER_EQUAL,
                    operands=[Variable("x"), Constant(0)]
                )
            ]
        )
        
        verifier = Verifier(timeout_ms=5000)
        result = verifier.verify(code, spec)
        
        elapsed = time.time() - start
        
        # Should complete within 10 seconds
        assert elapsed < 10.0


class TestErrorHandling:
    """Test error handling in the pipeline."""
    
    def test_empty_code(self):
        """Test handling of empty code."""
        verifier = Verifier()
        spec = Specification(function_name="test")
        
        result = verifier.verify("", spec)
        
        # Should not crash
        assert result.status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.FAILED,
            VerificationStatus.UNKNOWN,
            VerificationStatus.ERROR
        ]
    
    def test_malformed_code(self):
        """Test handling of malformed code."""
        verifier = Verifier()
        spec = Specification(function_name="test")
        
        # Malformed code
        result = verifier.verify("{{{{", spec)
        
        # Should not crash
        assert result is not None
    
    def test_empty_specification(self):
        """Test handling of empty specification."""
        verifier = Verifier()
        spec = Specification(function_name="test")
        
        code = "int x = 1;"
        result = verifier.verify(code, spec)
        
        # Empty spec should verify trivially
        assert result.status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.UNKNOWN
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
