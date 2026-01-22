"""
Tests for Switch Statement Verifier

Tests that the verifier actually works - not just "toy level".
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
                assert result.success_rate >= 0.7, f"Should have >70% success rate, got {result.success_rate}"
                
                # Should not have incorrect mappings
                assert len(result.incorrect_mappings) == 0, f"Should have no incorrect: {result.incorrect_mappings}"
                
                print(f"\nRISCV getRelocType verification:")
                print(f"  Cases: {result.cases_total}")
                print(f"  Verified: {result.cases_verified}")
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


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
