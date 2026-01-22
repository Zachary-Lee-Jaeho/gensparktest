"""
Formal specification language for compiler backend functions.
Supports preconditions, postconditions, and invariants.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod
import json


class ConditionType(Enum):
    """Types of conditions in specifications."""
    EQUALITY = "eq"           # a == b
    INEQUALITY = "neq"        # a != b
    LESS_THAN = "lt"          # a < b
    LESS_EQUAL = "le"         # a <= b
    GREATER_THAN = "gt"       # a > b
    GREATER_EQUAL = "ge"      # a >= b
    IS_VALID = "valid"        # isValid(x)
    IS_IN_RANGE = "range"     # lo <= x <= hi
    IS_IN_SET = "in_set"      # x in {a, b, c}
    IMPLIES = "implies"       # a => b
    AND = "and"               # a && b
    OR = "or"                 # a || b
    NOT = "not"               # !a
    FORALL = "forall"         # forall x: P(x)
    EXISTS = "exists"         # exists x: P(x)


@dataclass
class Expression(ABC):
    """Base class for expressions in specifications."""
    
    @abstractmethod
    def to_smt(self) -> str:
        """Convert to SMT-LIB format."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Expression":
        """Create expression from dictionary."""
        expr_type = data.get("type")
        if expr_type == "variable":
            return Variable(name=data["name"], var_type=data.get("var_type", "int"))
        elif expr_type == "constant":
            return Constant(value=data["value"], const_type=data.get("const_type", "int"))
        elif expr_type == "binary_op":
            return BinaryOp(
                op=data["op"],
                left=Expression.from_dict(data["left"]),
                right=Expression.from_dict(data["right"])
            )
        elif expr_type == "function_call":
            return FunctionCall(
                name=data["name"],
                args=[Expression.from_dict(arg) for arg in data.get("args", [])]
            )
        else:
            raise ValueError(f"Unknown expression type: {expr_type}")


@dataclass
class Variable(Expression):
    """Variable in specification."""
    name: str
    var_type: str = "int"
    
    def to_smt(self) -> str:
        return self.name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "variable",
            "name": self.name,
            "var_type": self.var_type
        }
    
    def __str__(self) -> str:
        return self.name


@dataclass
class Constant(Expression):
    """Constant value in specification."""
    value: Any
    const_type: str = "int"
    
    def to_smt(self) -> str:
        if self.const_type == "bool":
            return "true" if self.value else "false"
        elif self.const_type == "string":
            return f'"{self.value}"'
        return str(self.value)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "constant",
            "value": self.value,
            "const_type": self.const_type
        }
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass
class BinaryOp(Expression):
    """Binary operation expression."""
    op: str  # +, -, *, /, %, &, |, ^, <<, >>
    left: Expression
    right: Expression
    
    def to_smt(self) -> str:
        op_map = {
            "+": "bvadd", "-": "bvsub", "*": "bvmul", "/": "bvsdiv",
            "%": "bvsrem", "&": "bvand", "|": "bvor", "^": "bvxor",
            "<<": "bvshl", ">>": "bvashr"
        }
        smt_op = op_map.get(self.op, self.op)
        return f"({smt_op} {self.left.to_smt()} {self.right.to_smt()})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "binary_op",
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict()
        }
    
    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class FunctionCall(Expression):
    """Function call expression."""
    name: str
    args: List[Expression] = field(default_factory=list)
    
    def to_smt(self) -> str:
        if not self.args:
            return f"({self.name})"
        args_str = " ".join(arg.to_smt() for arg in self.args)
        return f"({self.name} {args_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "function_call",
            "name": self.name,
            "args": [arg.to_dict() for arg in self.args]
        }
    
    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"


@dataclass
class Condition:
    """
    Formal condition in specification.
    Can be atomic (comparison, validity check) or composite (and, or, implies).
    """
    cond_type: ConditionType
    operands: List[Union[Expression, "Condition"]] = field(default_factory=list)
    quantified_var: Optional[str] = None  # For forall/exists
    
    def to_smt(self) -> str:
        """Convert condition to SMT-LIB format."""
        if self.cond_type == ConditionType.EQUALITY:
            return f"(= {self.operands[0].to_smt()} {self.operands[1].to_smt()})"
        
        elif self.cond_type == ConditionType.INEQUALITY:
            return f"(not (= {self.operands[0].to_smt()} {self.operands[1].to_smt()}))"
        
        elif self.cond_type == ConditionType.LESS_THAN:
            return f"(bvslt {self.operands[0].to_smt()} {self.operands[1].to_smt()})"
        
        elif self.cond_type == ConditionType.LESS_EQUAL:
            return f"(bvsle {self.operands[0].to_smt()} {self.operands[1].to_smt()})"
        
        elif self.cond_type == ConditionType.GREATER_THAN:
            return f"(bvsgt {self.operands[0].to_smt()} {self.operands[1].to_smt()})"
        
        elif self.cond_type == ConditionType.GREATER_EQUAL:
            return f"(bvsge {self.operands[0].to_smt()} {self.operands[1].to_smt()})"
        
        elif self.cond_type == ConditionType.IS_VALID:
            return f"(isValid {self.operands[0].to_smt()})"
        
        elif self.cond_type == ConditionType.IS_IN_RANGE:
            # lo <= x <= hi
            var, lo, hi = self.operands[0], self.operands[1], self.operands[2]
            return f"(and (bvsle {lo.to_smt()} {var.to_smt()}) (bvsle {var.to_smt()} {hi.to_smt()}))"
        
        elif self.cond_type == ConditionType.IS_IN_SET:
            var = self.operands[0]
            set_members = self.operands[1:]
            clauses = [f"(= {var.to_smt()} {m.to_smt()})" for m in set_members]
            return f"(or {' '.join(clauses)})"
        
        elif self.cond_type == ConditionType.IMPLIES:
            return f"(=> {self.operands[0].to_smt()} {self.operands[1].to_smt()})"
        
        elif self.cond_type == ConditionType.AND:
            if len(self.operands) == 0:
                return "true"
            elif len(self.operands) == 1:
                return self.operands[0].to_smt()
            return f"(and {' '.join(op.to_smt() for op in self.operands)})"
        
        elif self.cond_type == ConditionType.OR:
            if len(self.operands) == 0:
                return "false"
            elif len(self.operands) == 1:
                return self.operands[0].to_smt()
            return f"(or {' '.join(op.to_smt() for op in self.operands)})"
        
        elif self.cond_type == ConditionType.NOT:
            return f"(not {self.operands[0].to_smt()})"
        
        elif self.cond_type == ConditionType.FORALL:
            return f"(forall (({self.quantified_var} Int)) {self.operands[0].to_smt()})"
        
        elif self.cond_type == ConditionType.EXISTS:
            return f"(exists (({self.quantified_var} Int)) {self.operands[0].to_smt()})"
        
        else:
            raise ValueError(f"Unknown condition type: {self.cond_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "type": self.cond_type.value,
            "operands": []
        }
        
        for op in self.operands:
            if isinstance(op, Condition):
                result["operands"].append({"condition": op.to_dict()})
            else:
                result["operands"].append({"expression": op.to_dict()})
        
        if self.quantified_var:
            result["quantified_var"] = self.quantified_var
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        """Create condition from dictionary."""
        cond_type = ConditionType(data["type"])
        operands = []
        
        for op in data.get("operands", []):
            if "condition" in op:
                operands.append(Condition.from_dict(op["condition"]))
            elif "expression" in op:
                operands.append(Expression.from_dict(op["expression"]))
        
        return cls(
            cond_type=cond_type,
            operands=operands,
            quantified_var=data.get("quantified_var")
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.cond_type == ConditionType.EQUALITY:
            return f"{self.operands[0]} == {self.operands[1]}"
        elif self.cond_type == ConditionType.INEQUALITY:
            return f"{self.operands[0]} != {self.operands[1]}"
        elif self.cond_type == ConditionType.LESS_THAN:
            return f"{self.operands[0]} < {self.operands[1]}"
        elif self.cond_type == ConditionType.LESS_EQUAL:
            return f"{self.operands[0]} <= {self.operands[1]}"
        elif self.cond_type == ConditionType.GREATER_THAN:
            return f"{self.operands[0]} > {self.operands[1]}"
        elif self.cond_type == ConditionType.GREATER_EQUAL:
            return f"{self.operands[0]} >= {self.operands[1]}"
        elif self.cond_type == ConditionType.IS_VALID:
            return f"isValid({self.operands[0]})"
        elif self.cond_type == ConditionType.IS_IN_RANGE:
            return f"{self.operands[1]} <= {self.operands[0]} <= {self.operands[2]}"
        elif self.cond_type == ConditionType.IS_IN_SET:
            members = ", ".join(str(m) for m in self.operands[1:])
            return f"{self.operands[0]} in {{{members}}}"
        elif self.cond_type == ConditionType.IMPLIES:
            return f"{self.operands[0]} => {self.operands[1]}"
        elif self.cond_type == ConditionType.AND:
            return " && ".join(str(op) for op in self.operands)
        elif self.cond_type == ConditionType.OR:
            return " || ".join(str(op) for op in self.operands)
        elif self.cond_type == ConditionType.NOT:
            return f"!({self.operands[0]})"
        elif self.cond_type == ConditionType.FORALL:
            return f"forall {self.quantified_var}: {self.operands[0]}"
        elif self.cond_type == ConditionType.EXISTS:
            return f"exists {self.quantified_var}: {self.operands[0]}"
        else:
            return f"Condition({self.cond_type}, {self.operands})"
    
    # Convenience constructors
    @classmethod
    def eq(cls, left: Expression, right: Expression) -> "Condition":
        return cls(ConditionType.EQUALITY, [left, right])
    
    @classmethod
    def neq(cls, left: Expression, right: Expression) -> "Condition":
        return cls(ConditionType.INEQUALITY, [left, right])
    
    @classmethod
    def lt(cls, left: Expression, right: Expression) -> "Condition":
        return cls(ConditionType.LESS_THAN, [left, right])
    
    @classmethod
    def le(cls, left: Expression, right: Expression) -> "Condition":
        return cls(ConditionType.LESS_EQUAL, [left, right])
    
    @classmethod
    def gt(cls, left: Expression, right: Expression) -> "Condition":
        return cls(ConditionType.GREATER_THAN, [left, right])
    
    @classmethod
    def ge(cls, left: Expression, right: Expression) -> "Condition":
        return cls(ConditionType.GREATER_EQUAL, [left, right])
    
    @classmethod
    def valid(cls, expr: Expression) -> "Condition":
        return cls(ConditionType.IS_VALID, [expr])
    
    @classmethod
    def in_range(cls, var: Expression, lo: Expression, hi: Expression) -> "Condition":
        return cls(ConditionType.IS_IN_RANGE, [var, lo, hi])
    
    @classmethod
    def in_set(cls, var: Expression, *members: Expression) -> "Condition":
        return cls(ConditionType.IS_IN_SET, [var, *members])
    
    @classmethod
    def implies(cls, antecedent: "Condition", consequent: "Condition") -> "Condition":
        return cls(ConditionType.IMPLIES, [antecedent, consequent])
    
    @classmethod
    def and_(cls, *conditions: "Condition") -> "Condition":
        return cls(ConditionType.AND, list(conditions))
    
    @classmethod
    def or_(cls, *conditions: "Condition") -> "Condition":
        return cls(ConditionType.OR, list(conditions))
    
    @classmethod
    def not_(cls, condition: "Condition") -> "Condition":
        return cls(ConditionType.NOT, [condition])


@dataclass
class Specification:
    """
    Complete formal specification for a compiler backend function.
    
    Contains:
    - Preconditions: Conditions that must hold before function execution
    - Postconditions: Conditions that must hold after function execution
    - Invariants: Conditions that must hold throughout (for loops, etc.)
    """
    function_name: str
    preconditions: List[Condition] = field(default_factory=list)
    postconditions: List[Condition] = field(default_factory=list)
    invariants: List[Condition] = field(default_factory=list)
    
    # Metadata
    module: str = ""
    inferred_from: List[str] = field(default_factory=list)  # Reference targets
    confidence: float = 1.0  # Inference confidence
    
    def to_smt(self) -> str:
        """Convert entire specification to SMT-LIB assertions."""
        smt_parts = []
        
        # Preconditions
        if self.preconditions:
            pre_cond = Condition.and_(*self.preconditions)
            smt_parts.append(f"; Preconditions\n(assert {pre_cond.to_smt()})")
        
        # Postconditions
        if self.postconditions:
            post_cond = Condition.and_(*self.postconditions)
            smt_parts.append(f"; Postconditions\n(assert {post_cond.to_smt()})")
        
        # Invariants
        for inv in self.invariants:
            smt_parts.append(f"; Invariant\n(assert {inv.to_smt()})")
        
        return "\n\n".join(smt_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "function_name": self.function_name,
            "module": self.module,
            "preconditions": [c.to_dict() for c in self.preconditions],
            "postconditions": [c.to_dict() for c in self.postconditions],
            "invariants": [c.to_dict() for c in self.invariants],
            "inferred_from": self.inferred_from,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Specification":
        """Create specification from dictionary."""
        return cls(
            function_name=data["function_name"],
            module=data.get("module", ""),
            preconditions=[Condition.from_dict(c) for c in data.get("preconditions", [])],
            postconditions=[Condition.from_dict(c) for c in data.get("postconditions", [])],
            invariants=[Condition.from_dict(c) for c in data.get("invariants", [])],
            inferred_from=data.get("inferred_from", []),
            confidence=data.get("confidence", 1.0),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Specification":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, filepath: str) -> None:
        """Save specification to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> "Specification":
        """Load specification from file."""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())
    
    def validate(self, code: str, timeout_ms: int = 30000) -> bool:
        """
        Validate that the specification holds for given code.
        
        Uses the verification engine to check if the code satisfies
        all preconditions, postconditions, and invariants.
        
        Args:
            code: C++ code to validate against this specification
            timeout_ms: Verification timeout in milliseconds
            
        Returns:
            True if the code satisfies the specification, False otherwise
        """
        from src.verification.verifier import Verifier, VerificationStatus
        
        try:
            verifier = Verifier(timeout_ms=timeout_ms)
            result = verifier.verify(code, self)
            return result.status == VerificationStatus.VERIFIED
        except Exception as e:
            # Log the error but return False (spec not validated)
            import logging
            logging.warning(f"Specification validation failed with error: {e}")
            return False
    
    def validate_with_result(self, code: str, timeout_ms: int = 30000):
        """
        Validate specification and return full verification result.
        
        Args:
            code: C++ code to validate against this specification
            timeout_ms: Verification timeout in milliseconds
            
        Returns:
            VerificationResult with status, counterexample, etc.
        """
        from src.verification.verifier import Verifier
        
        verifier = Verifier(timeout_ms=timeout_ms)
        return verifier.verify(code, self)
    
    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [f"Specification for {self.function_name}"]
        
        if self.preconditions:
            lines.append("\nPreconditions:")
            for pre in self.preconditions:
                lines.append(f"  - {pre}")
        
        if self.postconditions:
            lines.append("\nPostconditions:")
            for post in self.postconditions:
                lines.append(f"  - {post}")
        
        if self.invariants:
            lines.append("\nInvariants:")
            for inv in self.invariants:
                lines.append(f"  - {inv}")
        
        return "\n".join(lines)
