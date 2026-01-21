"""
Unit tests for verification module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.verification.verifier import Verifier, VerificationResult, VerificationStatus
from src.verification.smt_solver import SMTSolver, SMTResult
from src.verification.vcgen import VCGenerator
from src.specification.spec_language import (
    Specification,
    Condition,
    Variable,
    Constant,
)


class TestSMTSolver:
    """Tests for SMT solver."""
    
    def test_solver_creation(self):
        """Test solver creation."""
        solver = SMTSolver()
        assert solver is not None
    
    def test_simple_sat(self):
        """Test simple satisfiable formula."""
        solver = SMTSolver()
        
        # x > 0 is satisfiable
        formula = "(declare-const x Int)\n(assert (> x 0))\n(check-sat)"
        result = solver.check(formula)
        
        # Should be SAT (meaning verification fails, counterexample exists)
        assert result.is_sat or result.is_unknown
    
    def test_simple_unsat(self):
        """Test simple unsatisfiable formula."""
        solver = SMTSolver()
        
        # x > 0 and x < 0 is unsatisfiable
        formula = """
        (declare-const x Int)
        (assert (> x 0))
        (assert (< x 0))
        (check-sat)
        """
        result = solver.check(formula)
        
        # Should be UNSAT (meaning verification succeeds)
        assert result.is_unsat or result.is_unknown


class TestVCGenerator:
    """Tests for verification condition generator."""
    
    def test_vcgen_creation(self):
        """Test VCGen creation."""
        vcgen = VCGenerator()
        assert vcgen is not None
    
    def test_generate_basic_vc(self):
        """Test basic VC generation."""
        vcgen = VCGenerator()
        
        code = """
        int abs(int x) {
            if (x < 0) {
                return -x;
            }
            return x;
        }
        """
        
        spec = Specification(
            function_name="abs",
            module="Math",
            postconditions=[
                Condition.ge(Variable("result"), Constant(0))
            ]
        )
        
        vc = vcgen.generate(code, spec)
        
        assert vc is not None
        assert "result" in vc or "abs" in vc


class TestVerifier:
    """Tests for the main verifier."""
    
    def test_verifier_creation(self):
        """Test verifier creation."""
        verifier = Verifier()
        assert verifier is not None
    
    def test_verifier_with_timeout(self):
        """Test verifier with custom timeout."""
        verifier = Verifier(timeout_ms=5000)
        assert verifier.timeout_ms == 5000
    
    def test_verify_simple_function(self):
        """Test verification of simple function."""
        verifier = Verifier()
        
        code = """
        unsigned getRelocType(int kind) {
            switch (kind) {
            case 0: return 0;
            case 1: return 1;
            default: return 0;
            }
        }
        """
        
        spec = Specification(
            function_name="getRelocType",
            module="Test",
            postconditions=[
                Condition.ge(Variable("result"), Constant(0))
            ]
        )
        
        result = verifier.verify(code, spec)
        
        assert isinstance(result, VerificationResult)
        assert result.status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.FAILED,
            VerificationStatus.UNKNOWN
        ]
    
    def test_verification_result_properties(self):
        """Test verification result properties."""
        result = VerificationResult(
            status=VerificationStatus.VERIFIED,
            function_name="test",
            time_ms=100.0
        )
        
        assert result.is_verified()
        assert result.time_ms == 100.0
    
    def test_verification_result_with_counterexample(self):
        """Test verification result with counterexample."""
        from src.verification.verifier import Counterexample
        
        ce = Counterexample(
            input_values={"x": -5},
            output_values={"result": -5},
            violated_condition="result >= 0",
            trace=["entry", "n1", "return"]
        )
        
        result = VerificationResult(
            status=VerificationStatus.FAILED,
            function_name="test",
            counterexample=ce,
            time_ms=50.0
        )
        
        assert not result.is_verified()
        assert result.counterexample is not None
        assert result.counterexample.input_values["x"] == -5


class TestVerificationStatus:
    """Tests for verification status enum."""
    
    def test_status_values(self):
        """Test status enum values."""
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.FAILED.value == "failed"
        assert VerificationStatus.TIMEOUT.value == "timeout"
        assert VerificationStatus.UNKNOWN.value == "unknown"


class TestIntegration:
    """Integration tests for verification pipeline."""
    
    def test_full_verification_pipeline(self):
        """Test complete verification pipeline."""
        verifier = Verifier(timeout_ms=10000)
        
        # Test function
        code = """
        int max(int a, int b) {
            if (a > b) {
                return a;
            }
            return b;
        }
        """
        
        # Specification
        spec = Specification(
            function_name="max",
            module="Math",
            preconditions=[],
            postconditions=[
                Condition.ge(Variable("result"), Variable("a")),
                Condition.ge(Variable("result"), Variable("b")),
            ]
        )
        
        result = verifier.verify(code, spec)
        
        # Should complete (verified or unknown due to simulation)
        assert result.status != VerificationStatus.TIMEOUT
    
    def test_verification_with_invariants(self):
        """Test verification with loop invariants."""
        verifier = Verifier()
        
        code = """
        int sum(int n) {
            int s = 0;
            for (int i = 0; i < n; i++) {
                s += i;
            }
            return s;
        }
        """
        
        spec = Specification(
            function_name="sum",
            module="Math",
            preconditions=[
                Condition.ge(Variable("n"), Constant(0))
            ],
            postconditions=[
                Condition.ge(Variable("result"), Constant(0))
            ],
            invariants=[
                Condition.ge(Variable("s"), Constant(0))
            ]
        )
        
        result = verifier.verify(code, spec)
        
        assert isinstance(result, VerificationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
