"""
Tests for Switch Statement Verifier

Tests that the verifier actually works - not just "toy level".
Includes tests for:
- Basic case-return parsing
- Fall-through case handling
- Ternary operator conditions
- Error detection
"""

import pytest
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'verification'))

# Import by executing (to avoid relative import issues)
exec(open(Path(__file__).parent.parent / 'src' / 'verification' / 'switch_verifier.py').read())


class TestSwitchVerifierBasic:
    """Basic functionality tests."""
    
    def test_parser_extracts_cases(self):
        """Test that parser can extract case-return pairs."""
        code = """
        switch (Kind) {
            case RISCV::fixup_hi20:
                return ELF::R_RISCV_HI20;
            case RISCV::fixup_lo12:
                return ELF::R_RISCV_LO12_I;
        }
        """
        parser = SwitchStatementParser()
        switches = parser.parse(code)
        
        assert len(switches) == 1
        assert len(switches[0].cases) == 2
        assert switches[0].variable == "Kind"
    
    def test_verifier_detects_correct_mappings(self):
        """Test that verifier confirms correct mappings."""
        code = """
        switch (Kind) {
            case RISCV::fixup_riscv_hi20:
                return ELF::R_RISCV_HI20;
            case RISCV::fixup_riscv_lo12_i:
                return ELF::R_RISCV_LO12_I;
        }
        """
        verifier = SwitchVerifier()
        result = verifier.verify_function(
            code=code,
            function_name="test",
            expected_mappings={
                'fixup_riscv_hi20': 'R_RISCV_HI20',
                'fixup_riscv_lo12_i': 'R_RISCV_LO12_I',
            }
        )
        
        assert result.cases_verified == 2
        assert len(result.incorrect_mappings) == 0
    
    def test_verifier_detects_incorrect_mappings(self):
        """Test that verifier catches wrong return values."""
        code = """
        switch (Kind) {
            case RISCV::fixup_riscv_hi20:
                return ELF::R_RISCV_LO12_I;
        }
        """
        verifier = SwitchVerifier()
        result = verifier.verify_function(
            code=code,
            function_name="test",
            expected_mappings={
                'fixup_riscv_hi20': 'R_RISCV_HI20',  # Expected HI20, got LO12_I
            }
        )
        
        assert len(result.incorrect_mappings) == 1
        assert result.status == VerificationStatus.FAILED
    
    def test_verifier_detects_missing_cases(self):
        """Test that verifier finds missing expected cases."""
        code = """
        switch (Kind) {
            case RISCV::fixup_riscv_hi20:
                return ELF::R_RISCV_HI20;
        }
        """
        verifier = SwitchVerifier()
        result = verifier.verify_function(
            code=code,
            function_name="test",
            expected_mappings={
                'fixup_riscv_hi20': 'R_RISCV_HI20',
                'fixup_riscv_lo12_i': 'R_RISCV_LO12_I',  # This is missing
            }
        )
        
        assert 'fixup_riscv_lo12_i' in result.missing_cases


class TestFallthroughCases:
    """Tests for fall-through case handling."""
    
    def test_fallthrough_parsing(self):
        """Test that parser correctly handles fall-through cases."""
        code = """
        switch (Kind) {
            case FK_Data_4:
            case FK_PCRel_4:
                return ELF::R_RISCV_32_PCREL;
        }
        """
        parser = SwitchStatementParser()
        switches = parser.parse(code)
        
        assert len(switches) == 1
        # Should parse both cases
        assert len(switches[0].cases) == 2
        
        # Check fall-through tracking
        case_names = [c.case_value for c in switches[0].cases]
        assert 'FK_Data_4' in case_names
        assert 'FK_PCRel_4' in case_names
        
        # FK_Data_4 should be marked as fallthrough
        fk_data = next(c for c in switches[0].cases if c.case_value == 'FK_Data_4')
        assert fk_data.is_fallthrough == True
    
    def test_fallthrough_verification(self):
        """Test that fall-through cases are verified correctly."""
        code = """
        switch (Kind) {
            case FK_Data_4:
            case FK_PCRel_4:
                return ELF::R_RISCV_32_PCREL;
            case FK_Data_8:
                return ELF::R_RISCV_64;
        }
        """
        verifier = SwitchVerifier()
        result = verifier.verify_function(
            code=code,
            function_name="test_fallthrough",
            expected_mappings={
                'FK_Data_4': ['R_RISCV_32_PCREL', 'R_RISCV_32'],
                'FK_PCRel_4': ['R_RISCV_32_PCREL'],
                'FK_Data_8': 'R_RISCV_64',
            }
        )
        
        # Should verify all cases
        assert result.cases_verified == 3
        assert result.fallthrough_cases >= 1
        assert len(result.incorrect_mappings) == 0
    
    def test_fallthrough_groups_detected(self):
        """Test that fall-through groups are tracked."""
        code = """
        switch (Kind) {
            case A:
            case B:
            case C:
                return X;
        }
        """
        parser = SwitchStatementParser()
        switches = parser.parse(code)
        
        assert len(switches[0].fallthrough_groups) >= 1
        # Should have a group containing A, B, C
        all_in_group = False
        for group in switches[0].fallthrough_groups:
            if 'A' in group and 'B' in group and 'C' in group:
                all_in_group = True
        assert all_in_group, "Should detect A, B, C as a fall-through group"


class TestTernaryConditions:
    """Tests for ternary operator handling."""
    
    def test_ternary_parsing(self):
        """Test that parser correctly handles ternary returns."""
        code = """
        switch (Kind) {
            case FK_PCRel_4:
                return IsPCRel ? ELF::R_RISCV_PLT32 : ELF::R_RISCV_32_PCREL;
        }
        """
        parser = SwitchStatementParser()
        switches = parser.parse(code)
        
        assert len(switches) == 1
        assert len(switches[0].cases) == 1
        
        case = switches[0].cases[0]
        assert case.has_condition == True
        assert case.condition is not None
        assert 'R_RISCV_PLT32' in case.condition.true_value
        assert 'R_RISCV_32_PCREL' in case.condition.false_value
    
    def test_ternary_verification(self):
        """Test that ternary cases are verified correctly."""
        code = """
        switch (Kind) {
            case FK_PCRel_4:
                return VK_PLT ? ELF::R_RISCV_PLT32 : ELF::R_RISCV_32_PCREL;
        }
        """
        verifier = SwitchVerifier()
        result = verifier.verify_function(
            code=code,
            function_name="test_ternary",
            expected_mappings={
                'FK_PCRel_4': ['R_RISCV_PLT32', 'R_RISCV_32_PCREL'],
            }
        )
        
        assert result.cases_verified == 1
        assert result.ternary_cases == 1
        assert len(result.incorrect_mappings) == 0
    
    def test_ternary_with_fallthrough(self):
        """Test combined fall-through and ternary."""
        code = """
        switch (Kind) {
            case FK_Data_4:
            case FK_PCRel_4:
                return VK_PLT ? ELF::R_RISCV_PLT32 : ELF::R_RISCV_32_PCREL;
        }
        """
        verifier = SwitchVerifier()
        result = verifier.verify_function(
            code=code,
            function_name="test_combined",
            expected_mappings={
                'FK_Data_4': ['R_RISCV_PLT32', 'R_RISCV_32_PCREL'],
                'FK_PCRel_4': ['R_RISCV_PLT32', 'R_RISCV_32_PCREL'],
            }
        )
        
        assert result.cases_verified == 2
        assert result.fallthrough_cases >= 1
        assert result.ternary_cases >= 1


class TestRealLLVMCode:
    """Tests with actual extracted LLVM code."""
    
    @pytest.fixture
    def llvm_db(self):
        """Load LLVM function database."""
        db_path = Path(__file__).parent.parent / 'data' / 'llvm_functions_multi.json'
        if not db_path.exists():
            pytest.skip("LLVM database not available")
        with open(db_path) as f:
            return json.load(f)
    
    def test_riscv_getreloctype_parsing(self, llvm_db):
        """Test parsing of real RISCV getRelocType."""
        for func in llvm_db['functions'].values():
            if 'getRelocType' in func['name'] and func['backend'] == 'RISCV':
                parser = SwitchStatementParser()
                switches = parser.parse(func.get('body', ''))
                
                assert len(switches) >= 1, "Should find at least one switch"
                assert len(switches[0].cases) >= 20, f"Should have many cases, got {len(switches[0].cases)}"
                print(f"\nRISCV getRelocType: {len(switches[0].cases)} cases parsed")
                return
        
        pytest.fail("RISCV getRelocType not found")
    
    def test_riscv_getreloctype_verification(self, llvm_db):
        """Test verification of real RISCV getRelocType."""
        for func in llvm_db['functions'].values():
            if 'getRelocType' in func['name'] and func['backend'] == 'RISCV':
                verifier = SwitchVerifier()
                result = verifier.verify_reloc_type(func.get('body', ''), 'RISCV')
                
                # Should have high verification rate
                assert result.cases_total >= 20, f"Should parse many cases, got {result.cases_total}"
                assert result.success_rate >= 0.9, f"Should have >90% success rate, got {result.success_rate}"
                
                # Should not have incorrect mappings
                assert len(result.incorrect_mappings) == 0, f"Should have no incorrect: {result.incorrect_mappings}"
                
                # Should detect fall-through and ternary cases
                assert result.fallthrough_cases >= 1, "Should detect fall-through cases"
                assert result.ternary_cases >= 1, "Should detect ternary cases"
                
                print(f"\nRISCV getRelocType verification:")
                print(f"  Cases: {result.cases_total}")
                print(f"  Verified: {result.cases_verified}")
                print(f"  Fall-through: {result.fallthrough_cases}")
                print(f"  Ternary: {result.ternary_cases}")
                print(f"  Success rate: {result.success_rate*100:.1f}%")
                return
        
        pytest.fail("RISCV getRelocType not found")
    
    def test_multiple_backends_parsing(self, llvm_db):
        """Test parsing works for multiple backends."""
        backends_tested = set()
        parser = SwitchStatementParser()
        
        for func in llvm_db['functions'].values():
            if 'getRelocType' in func['name']:
                backend = func['backend']
                if backend not in backends_tested:
                    switches = parser.parse(func.get('body', ''))
                    if switches and switches[0].cases:
                        backends_tested.add(backend)
                        print(f"  {backend}: {len(switches[0].cases)} cases")
        
        assert len(backends_tested) >= 3, f"Should work for multiple backends, got {backends_tested}"


class TestErrorDetection:
    """Tests for error detection capabilities."""
    
    def test_detects_swapped_values(self):
        """Test detection of swapped return values."""
        # Intentionally wrong: HI20 returns LO12 and vice versa
        code = """
        switch (Kind) {
            case RISCV::fixup_riscv_hi20:
                return ELF::R_RISCV_LO12_I;
            case RISCV::fixup_riscv_lo12_i:
                return ELF::R_RISCV_HI20;
        }
        """
        verifier = SwitchVerifier()
        result = verifier.verify_function(
            code=code,
            function_name="SwappedValues",
            expected_mappings={
                'fixup_riscv_hi20': 'R_RISCV_HI20',
                'fixup_riscv_lo12_i': 'R_RISCV_LO12_I',
            }
        )
        
        # Should detect both are wrong
        assert len(result.incorrect_mappings) == 2
        assert result.status == VerificationStatus.FAILED
    
    def test_detects_typo_in_return_value(self):
        """Test detection of typo in return value."""
        code = """
        switch (Kind) {
            case RISCV::fixup_riscv_branch:
                return ELF::R_RISCV_BRAMCH;
        }
        """
        verifier = SwitchVerifier()
        result = verifier.verify_function(
            code=code,
            function_name="Typo",
            expected_mappings={
                'fixup_riscv_branch': 'R_RISCV_BRANCH',  # BRANCH not BRAMCH
            }
        )
        
        assert len(result.incorrect_mappings) == 1
    
    def test_passes_correct_code(self):
        """Test that correct code passes."""
        code = """
        switch (Kind) {
            case RISCV::fixup_riscv_hi20:
                return ELF::R_RISCV_HI20;
            case RISCV::fixup_riscv_lo12_i:
                return ELF::R_RISCV_LO12_I;
            case RISCV::fixup_riscv_branch:
                return ELF::R_RISCV_BRANCH;
        }
        """
        verifier = SwitchVerifier()
        result = verifier.verify_function(
            code=code,
            function_name="Correct",
            expected_mappings={
                'fixup_riscv_hi20': 'R_RISCV_HI20',
                'fixup_riscv_lo12_i': 'R_RISCV_LO12_I',
                'fixup_riscv_branch': 'R_RISCV_BRANCH',
            }
        )
        
        assert len(result.incorrect_mappings) == 0
        assert result.cases_verified == 3


class TestInputCoverage:
    """Tests for input coverage verification."""
    
    def test_coverage_check(self):
        """Test that coverage verifier works."""
        code = """
        switch (Kind) {
            case A:
                return X;
            case B:
                return Y;
            default:
                return Z;
        }
        """
        parser = SwitchStatementParser()
        switches = parser.parse(code)
        
        coverage_verifier = InputCoverageVerifier()
        result = coverage_verifier.verify_coverage(
            switches[0],
            all_possible_inputs=['A', 'B', 'C', 'D'],
            has_default=True
        )
        
        assert result['handled'] == 2
        assert 'C' in result['missing']
        assert 'D' in result['missing']
        assert result['is_complete'] == True  # Because has default
    
    def test_riscv_full_coverage(self, llvm_db=None):
        """Test RISCV getRelocType has 100% coverage."""
        db_path = Path(__file__).parent.parent / 'data' / 'llvm_functions_multi.json'
        if not db_path.exists():
            pytest.skip("LLVM database not available")
        
        with open(db_path) as f:
            db = json.load(f)
        
        for func in db['functions'].values():
            if 'getRelocType' in func['name'] and func['backend'] == 'RISCV':
                parser = SwitchStatementParser()
                switches = parser.parse(func.get('body', ''))
                
                if switches:
                    coverage_verifier = InputCoverageVerifier()
                    result = coverage_verifier.verify_riscv_reloc_coverage(switches[0])
                    
                    # Should have 100% coverage
                    assert result['coverage_rate'] == 1.0, f"Expected 100% coverage, got {result['coverage_rate']*100}%"
                    assert len(result['missing']) == 0, f"Missing: {result['missing']}"
                    
                    # Z3 should prove completeness
                    if result['z3_verification']:
                        assert result['z3_verification']['status'] == 'complete'
                    
                    print(f"\nRISCV coverage: {result['covered']}/{result['total_required']} = 100%")
                    return
        
        pytest.fail("RISCV getRelocType not found")
    
    def test_z3_detects_missing_case(self):
        """Test that Z3 correctly identifies missing cases."""
        # Create a switch with intentionally missing case
        code = """
        switch (Kind) {
            case fixup_riscv_hi20:
                return R_RISCV_HI20;
            case fixup_riscv_lo12_i:
                return R_RISCV_LO12_I;
        }
        """
        parser = SwitchStatementParser()
        switches = parser.parse(code)
        
        coverage_verifier = InputCoverageVerifier()
        result = coverage_verifier.verify_coverage(
            switches[0],
            all_possible_inputs=['fixup_riscv_hi20', 'fixup_riscv_lo12_i', 'fixup_riscv_branch'],
            has_default=False
        )
        
        # Should detect missing fixup_riscv_branch
        assert 'fixup_riscv_branch' in result['missing']
        assert result['is_complete'] == False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
