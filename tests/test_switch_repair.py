"""
Tests for Switch Repair Module and End-to-End Pipeline.

Tests both:
- switch_repair.py: SwitchRepairModel
- verify_and_repair.py: VerifyAndRepairPipeline
"""

import pytest
import json
import sys
from pathlib import Path

# Add src to path
_src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(_src_path / 'repair'))
sys.path.insert(0, str(_src_path / 'verification'))

# Import by executing (for switch_repair and switch_verifier)
exec(open(_src_path / 'repair' / 'switch_repair.py').read())
exec(open(_src_path / 'verification' / 'switch_verifier.py').read())


class TestSwitchRepairModel:
    """Tests for SwitchRepairModel."""
    
    def test_fix_incorrect_mapping(self):
        """Test fixing an incorrect case-return mapping."""
        model = SwitchRepairModel(backend="RISCV")
        
        code = """
        switch (Kind) {
            case RISCV::fixup_riscv_hi20:
                return ELF::R_RISCV_LO12_I;
        }
        """
        
        candidate = model.repair_incorrect_mapping(
            code=code,
            case_name="fixup_riscv_hi20",
            actual_return="R_RISCV_LO12_I",
            expected_return="R_RISCV_HI20"
        )
        
        assert candidate.repair_type == SwitchRepairType.FIX_RETURN_VALUE
        assert candidate.confidence >= 0.9
        assert "R_RISCV_HI20" in candidate.repaired_code
        assert "R_RISCV_LO12_I" not in candidate.repaired_code.split("fixup_riscv_hi20")[1].split("case")[0]
    
    def test_add_missing_case(self):
        """Test adding a missing case."""
        model = SwitchRepairModel(backend="RISCV")
        
        code = """
        switch (Kind) {
            case RISCV::fixup_riscv_hi20:
                return ELF::R_RISCV_HI20;
            default:
                return ELF::R_RISCV_NONE;
        }
        """
        
        candidate = model.add_missing_case(
            code=code,
            case_name="fixup_riscv_lo12_i",
            return_value="R_RISCV_LO12_I"
        )
        
        assert candidate.repair_type == SwitchRepairType.ADD_MISSING_CASE
        assert "fixup_riscv_lo12_i" in candidate.repaired_code
        assert "R_RISCV_LO12_I" in candidate.repaired_code
    
    def test_repair_from_verification_result(self):
        """Test repair generation from verification result."""
        model = SwitchRepairModel(backend="RISCV")
        
        code = """
        switch (Kind) {
            case RISCV::fixup_riscv_hi20:
                return ELF::R_RISCV_LO12_I;
            default:
                return ELF::R_RISCV_NONE;
        }
        """
        
        verification_result = {
            'incorrect_mappings': [
                {'case': 'fixup_riscv_hi20', 'expected': 'R_RISCV_HI20', 'actual': 'R_RISCV_LO12_I'}
            ],
            'missing_cases': ['fixup_riscv_branch'],
            'duplicate_cases': []
        }
        
        result = model.repair_from_verification_result(code, verification_result)
        
        assert result.success
        assert len(result.candidates) == 2  # One fix + one add
        assert result.best_repair is not None
    
    def test_generate_complete_switch(self):
        """Test generating a complete switch statement."""
        model = SwitchRepairModel(backend="RISCV")
        
        cases = {
            'fixup_riscv_hi20': 'R_RISCV_HI20',
            'fixup_riscv_lo12_i': 'R_RISCV_LO12_I',
        }
        
        switch_code = model.generate_complete_switch(
            switch_variable="Kind",
            cases=cases,
            include_default=True
        )
        
        assert "switch (Kind)" in switch_code
        assert "fixup_riscv_hi20" in switch_code
        assert "R_RISCV_HI20" in switch_code
        assert "default:" in switch_code


class TestVerifyAndRepairIntegration:
    """Tests for verifier + repair integration."""
    
    def test_verifier_to_repair_flow(self):
        """Test that verifier output feeds into repair."""
        # Run verifier
        verifier = SwitchVerifier()
        
        code = """
        switch (Kind) {
            case RISCV::fixup_riscv_hi20:
                return ELF::R_RISCV_LO12_I;
        }
        """
        
        result = verifier.verify_function(
            code=code,
            function_name="test",
            expected_mappings={'fixup_riscv_hi20': 'R_RISCV_HI20'}
        )
        
        # Feed to repair
        repair_model = SwitchRepairModel(backend="RISCV")
        repair_result = repair_model.repair_from_verification_result(
            code,
            result.to_dict()
        )
        
        assert result.status == VerificationStatus.FAILED
        assert len(result.incorrect_mappings) == 1
        assert repair_result.success
        assert "R_RISCV_HI20" in repair_result.best_repair.repaired_code


class TestRealLLVMCode:
    """Tests with actual LLVM code."""
    
    @pytest.fixture
    def llvm_db(self):
        """Load LLVM function database."""
        db_path = Path(__file__).parent.parent / 'data' / 'llvm_functions_multi.json'
        if not db_path.exists():
            pytest.skip("LLVM database not available")
        with open(db_path) as f:
            return json.load(f)
    
    def test_real_getreloctype_verification(self, llvm_db):
        """Test verification of real RISCV getRelocType."""
        for func in llvm_db['functions'].values():
            if 'getRelocType' in func['name'] and func['backend'] == 'RISCV':
                verifier = SwitchVerifier()
                result = verifier.verify_reloc_type(func.get('body', ''), 'RISCV')
                
                # Real code should pass
                assert result.status == VerificationStatus.VERIFIED
                assert len(result.incorrect_mappings) == 0
                assert result.cases_total >= 25
                
                print(f"\nReal LLVM verification:")
                print(f"  Status: {result.status.value}")
                print(f"  Cases: {result.cases_total}")
                return
        
        pytest.fail("RISCV getRelocType not found")
    
    def test_real_code_no_repair_needed(self, llvm_db):
        """Test that real LLVM code doesn't need repair."""
        for func in llvm_db['functions'].values():
            if 'getRelocType' in func['name'] and func['backend'] == 'RISCV':
                code = func.get('body', '')
                
                # Verify
                verifier = SwitchVerifier()
                ver_result = verifier.verify_reloc_type(code, 'RISCV')
                
                # Try to repair (should not generate fixes for correct code)
                repair_model = SwitchRepairModel(backend="RISCV")
                repair_result = repair_model.repair_from_verification_result(
                    code,
                    ver_result.to_dict()
                )
                
                # Should have no incorrect mapping repairs
                fix_repairs = [c for c in repair_result.candidates 
                              if c.repair_type == SwitchRepairType.FIX_RETURN_VALUE]
                assert len(fix_repairs) == 0, "Real LLVM code should not need fixes"
                
                print(f"\nReal code repair test:")
                print(f"  Fix repairs needed: {len(fix_repairs)}")
                return
        
        pytest.fail("RISCV getRelocType not found")
    
    def test_coverage_verification(self, llvm_db):
        """Test coverage verification of real code."""
        for func in llvm_db['functions'].values():
            if 'getRelocType' in func['name'] and func['backend'] == 'RISCV':
                code = func.get('body', '')
                
                parser = SwitchStatementParser()
                switches = parser.parse(code)
                
                if switches:
                    coverage_verifier = InputCoverageVerifier()
                    result = coverage_verifier.verify_riscv_reloc_coverage(switches[0])
                    
                    # Should have high coverage
                    assert result['coverage_rate'] >= 0.9
                    assert result['z3_verification']['status'] == 'complete'
                    
                    print(f"\nCoverage test:")
                    print(f"  Rate: {result['coverage_rate']*100:.1f}%")
                    print(f"  Z3: {result['z3_verification']['status']}")
                return
        
        pytest.fail("RISCV getRelocType not found")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
