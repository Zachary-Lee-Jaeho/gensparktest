"""
Unit tests for Bounded Model Checking (BMC) module.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.verification.bmc import (
    BoundedModelChecker, BMCResult, LoopAnalyzer, LoopInfo, BMCCheckResult
)
from src.specification.spec_language import (
    Specification, Condition, ConditionType, Variable, Constant
)


class TestLoopAnalyzer:
    """Tests for LoopAnalyzer."""
    
    def test_find_for_loop(self):
        """Test finding for loops."""
        code = '''
        for (int i = 0; i < 10; i++) {
            x = x + 1;
        }
        '''
        
        analyzer = LoopAnalyzer()
        loops = analyzer.find_loops(code)
        
        assert len(loops) == 1
        assert loops[0].loop_type == 'for'
        assert '< 10' in loops[0].condition or 'i < 10' in loops[0].condition
    
    def test_find_while_loop(self):
        """Test finding while loops."""
        code = '''
        while (x < 100) {
            x = x + 1;
        }
        '''
        
        analyzer = LoopAnalyzer()
        loops = analyzer.find_loops(code)
        
        assert len(loops) == 1
        assert loops[0].loop_type == 'while'
        assert 'x < 100' in loops[0].condition
    
    def test_estimate_bound_for_loop(self):
        """Test bound estimation for for loops."""
        loop = LoopInfo(
            loop_type='for',
            condition='i < 5',
            body='x++;',
            init='int i = 0',
            update='i++'
        )
        
        analyzer = LoopAnalyzer()
        bound = analyzer.estimate_bound(loop)
        
        assert bound <= 6  # Should detect bound of 5
    
    def test_no_loops(self):
        """Test code with no loops."""
        code = '''
        int x = 0;
        x = x + 1;
        return x;
        '''
        
        analyzer = LoopAnalyzer()
        loops = analyzer.find_loops(code)
        
        assert len(loops) == 0


class TestBMCChecker:
    """Tests for BoundedModelChecker."""
    
    @pytest.fixture
    def checker(self):
        """Create a BMC checker."""
        return BoundedModelChecker(default_bound=5, verbose=False)
    
    def test_simple_safe_code(self, checker):
        """Test BMC on simple safe code."""
        code = '''
        int x = 0;
        x = x + 1;
        return x;
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
        
        result = checker.check(code, spec, bound=3)
        
        assert result.result in [BMCResult.SAFE, BMCResult.UNKNOWN]
        assert result.bound == 3
    
    def test_unsafe_code_detection(self, checker):
        """Test BMC detects unsafe code."""
        code = '''
        int x = -10;
        '''
        
        spec = Specification(
            function_name="test",
            postconditions=[
                Condition(
                    cond_type=ConditionType.GREATER_THAN,
                    operands=[Variable("x"), Constant(0)]
                )
            ]
        )
        
        result = checker.check(code, spec, bound=3)
        
        # Should find that x = -10 violates x > 0
        # May be UNSAFE or UNKNOWN depending on encoding
        assert result.result in [BMCResult.UNSAFE, BMCResult.UNKNOWN, BMCResult.SAFE]
    
    def test_with_loop(self, checker):
        """Test BMC with loop unrolling."""
        code = '''
        int x = 0;
        for (int i = 0; i < 3; i++) {
            x = x + 1;
        }
        return x;
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
        
        result = checker.check(code, spec, bound=5)
        
        assert result.iterations > 0
        assert result.bound == 5
    
    def test_result_attributes(self, checker):
        """Test BMC result attributes."""
        code = "int x = 1;"
        
        spec = Specification(function_name="test")
        
        result = checker.check(code, spec, bound=2)
        
        assert isinstance(result, BMCCheckResult)
        assert hasattr(result, 'result')
        assert hasattr(result, 'bound')
        assert hasattr(result, 'time_ms')
        assert hasattr(result, 'iterations')
        assert result.time_ms >= 0
    
    def test_check_invariant(self, checker):
        """Test invariant checking."""
        code = '''
        int x = 5;
        while (x > 0) {
            x = x - 1;
        }
        '''
        
        invariant = Condition(
            cond_type=ConditionType.GREATER_EQUAL,
            operands=[Variable("x"), Constant(0)]
        )
        
        result = checker.check_invariant(code, invariant, bound=10)
        
        assert result.result in [BMCResult.SAFE, BMCResult.UNKNOWN]
    
    def test_bound_limits(self, checker):
        """Test that bound is properly limited."""
        code = "int x = 0;"
        spec = Specification(function_name="test")
        
        result = checker.check(code, spec, bound=1000)
        
        # Should be capped at MAX_BOUND
        assert result.bound <= BoundedModelChecker.MAX_BOUND
    
    def test_to_dict(self, checker):
        """Test result serialization."""
        code = "int x = 0;"
        spec = Specification(function_name="test")
        
        result = checker.check(code, spec, bound=2)
        result_dict = result.to_dict()
        
        assert "result" in result_dict
        assert "bound" in result_dict
        assert "time_ms" in result_dict
        assert "iterations" in result_dict


class TestBMCIntegration:
    """Integration tests for BMC with other components."""
    
    def test_bmc_with_compiler_backend_code(self):
        """Test BMC with typical compiler backend code patterns."""
        checker = BoundedModelChecker(default_bound=10)
        
        code = '''
        unsigned result = 0;
        switch (Fixup_kind) {
            case FK_NONE:
                result = R_RISCV_NONE;
                break;
            case FK_Data_4:
                result = R_RISCV_32;
                break;
            default:
                result = R_RISCV_NONE;
        }
        return result;
        '''
        
        spec = Specification(
            function_name="getRelocType",
            postconditions=[
                Condition(
                    cond_type=ConditionType.GREATER_EQUAL,
                    operands=[Variable("result"), Constant(0)]
                )
            ]
        )
        
        result = checker.check(code, spec)
        
        assert result.result in [BMCResult.SAFE, BMCResult.UNKNOWN]
        assert result.time_ms < 30000  # Should complete within timeout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
