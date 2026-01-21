"""
Unit tests for specification module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.specification.spec_language import (
    Specification,
    Condition,
    Variable,
    Constant,
    ConditionType,
)


class TestCondition:
    """Tests for Condition class."""
    
    def test_eq_condition(self):
        """Test equality condition creation."""
        cond = Condition.eq(Variable("x"), Constant(5))
        
        assert cond.condition_type == ConditionType.EQ
        assert cond.left.name == "x"
        assert cond.right.value == 5
    
    def test_ne_condition(self):
        """Test not-equal condition creation."""
        cond = Condition.ne(Variable("x"), Constant(0))
        
        assert cond.condition_type == ConditionType.NE
    
    def test_lt_condition(self):
        """Test less-than condition creation."""
        cond = Condition.lt(Variable("x"), Constant(10))
        
        assert cond.condition_type == ConditionType.LT
    
    def test_ge_condition(self):
        """Test greater-equal condition creation."""
        cond = Condition.ge(Variable("result"), Constant(0))
        
        assert cond.condition_type == ConditionType.GE
    
    def test_valid_condition(self):
        """Test validity condition."""
        cond = Condition.valid(Variable("ptr"))
        
        assert cond.condition_type == ConditionType.VALID
    
    def test_implies_condition(self):
        """Test implication condition."""
        antecedent = Condition.eq(Variable("x"), Constant(0))
        consequent = Condition.eq(Variable("y"), Constant(0))
        cond = Condition.implies(antecedent, consequent)
        
        assert cond.condition_type == ConditionType.IMPLIES
    
    def test_and_condition(self):
        """Test conjunction condition."""
        c1 = Condition.eq(Variable("x"), Constant(0))
        c2 = Condition.eq(Variable("y"), Constant(0))
        cond = Condition.and_(c1, c2)
        
        assert cond.condition_type == ConditionType.AND
    
    def test_or_condition(self):
        """Test disjunction condition."""
        c1 = Condition.eq(Variable("x"), Constant(0))
        c2 = Condition.eq(Variable("y"), Constant(0))
        cond = Condition.or_(c1, c2)
        
        assert cond.condition_type == ConditionType.OR
    
    def test_to_smt_eq(self):
        """Test SMT conversion for equality."""
        cond = Condition.eq(Variable("x"), Constant(5))
        smt = cond.to_smt()
        
        assert "(= x 5)" in smt
    
    def test_to_smt_implies(self):
        """Test SMT conversion for implication."""
        antecedent = Condition.eq(Variable("x"), Constant(0))
        consequent = Condition.eq(Variable("y"), Constant(0))
        cond = Condition.implies(antecedent, consequent)
        smt = cond.to_smt()
        
        assert "=>" in smt


class TestVariable:
    """Tests for Variable class."""
    
    def test_variable_creation(self):
        """Test variable creation."""
        var = Variable("x", "Int")
        
        assert var.name == "x"
        assert var.var_type == "Int"
    
    def test_variable_to_smt(self):
        """Test variable SMT conversion."""
        var = Variable("counter", "Int")
        smt = var.to_smt()
        
        assert smt == "counter"


class TestConstant:
    """Tests for Constant class."""
    
    def test_int_constant(self):
        """Test integer constant."""
        const = Constant(42)
        
        assert const.value == 42
        assert const.const_type == "int"
    
    def test_bool_constant(self):
        """Test boolean constant."""
        const = Constant(True, "bool")
        
        assert const.value is True
        assert const.const_type == "bool"
    
    def test_string_constant(self):
        """Test string constant."""
        const = Constant("hello", "string")
        
        assert const.value == "hello"
    
    def test_enum_constant(self):
        """Test enum constant."""
        const = Constant("FK_NONE", "enum")
        
        assert const.value == "FK_NONE"
        assert const.const_type == "enum"
    
    def test_constant_to_smt_int(self):
        """Test SMT conversion for int."""
        const = Constant(42)
        smt = const.to_smt()
        
        assert smt == "42"
    
    def test_constant_to_smt_bool(self):
        """Test SMT conversion for bool."""
        const = Constant(True, "bool")
        smt = const.to_smt()
        
        assert smt == "true"


class TestSpecification:
    """Tests for Specification class."""
    
    def test_specification_creation(self):
        """Test specification creation."""
        spec = Specification(
            function_name="getRelocType",
            module="ELFObjectWriter"
        )
        
        assert spec.function_name == "getRelocType"
        assert spec.module == "ELFObjectWriter"
    
    def test_specification_with_conditions(self):
        """Test specification with pre/post conditions."""
        spec = Specification(
            function_name="test",
            module="TestModule",
            preconditions=[
                Condition.valid(Variable("input"))
            ],
            postconditions=[
                Condition.ge(Variable("result"), Constant(0))
            ],
            invariants=[
                Condition.ne(Variable("divisor"), Constant(0))
            ]
        )
        
        assert len(spec.preconditions) == 1
        assert len(spec.postconditions) == 1
        assert len(spec.invariants) == 1
    
    def test_specification_to_smt(self):
        """Test full SMT generation."""
        spec = Specification(
            function_name="abs",
            module="Math",
            preconditions=[
                Condition.valid(Variable("x"))
            ],
            postconditions=[
                Condition.ge(Variable("result"), Constant(0))
            ]
        )
        
        smt = spec.to_smt()
        
        assert "abs" in smt
        assert "Precondition" in smt
        assert "Postcondition" in smt
    
    def test_specification_to_json(self):
        """Test JSON serialization."""
        spec = Specification(
            function_name="test",
            module="TestModule"
        )
        
        json_data = spec.to_json()
        
        assert json_data["function_name"] == "test"
        assert json_data["module"] == "TestModule"
    
    def test_specification_from_json(self):
        """Test JSON deserialization."""
        json_data = {
            "function_name": "test",
            "module": "TestModule",
            "preconditions": [],
            "postconditions": [],
            "invariants": []
        }
        
        spec = Specification.from_json(json_data)
        
        assert spec.function_name == "test"
        assert spec.module == "TestModule"


class TestSpecificationSaveLoad:
    """Tests for specification save/load functionality."""
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading specification."""
        spec = Specification(
            function_name="getRelocType",
            module="ELFObjectWriter",
            preconditions=[
                Condition.valid(Variable("Fixup"))
            ],
            postconditions=[
                Condition.ge(Variable("result"), Constant(0))
            ]
        )
        
        # Save
        save_path = tmp_path / "spec.json"
        spec.save(str(save_path))
        
        assert save_path.exists()
        
        # Load
        loaded_spec = Specification.load(str(save_path))
        
        assert loaded_spec.function_name == spec.function_name
        assert loaded_spec.module == spec.module
        assert len(loaded_spec.preconditions) == 1
        assert len(loaded_spec.postconditions) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
