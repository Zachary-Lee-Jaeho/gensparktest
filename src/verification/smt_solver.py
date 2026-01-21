"""
SMT solver interface for VEGA-Verified.
Provides abstraction over Z3 solver for verification condition checking.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import time

# Try to import z3, provide fallback if not available
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    z3 = None


class SMTResult(Enum):
    """Result of SMT solving."""
    SAT = "sat"           # Satisfiable - counterexample exists
    UNSAT = "unsat"       # Unsatisfiable - property holds
    UNKNOWN = "unknown"   # Solver couldn't determine
    TIMEOUT = "timeout"   # Solver timed out


@dataclass
class SMTModel:
    """Model (counterexample) from SAT result."""
    assignments: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, var_name: str, default: Any = None) -> Any:
        """Get value of a variable in the model."""
        return self.assignments.get(var_name, default)
    
    def __str__(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.assignments.items())


class SMTSolver:
    """
    Wrapper around Z3 SMT solver.
    
    Supports:
    - Bitvector theory (for integer operations)
    - Uninterpreted functions (for abstraction)
    - Arrays (for memory modeling)
    - Incremental solving
    """
    
    def __init__(self, timeout_ms: int = 30000, incremental: bool = True):
        if not Z3_AVAILABLE:
            raise ImportError(
                "Z3 solver is not available. Install with: pip install z3-solver"
            )
        
        self.timeout_ms = timeout_ms
        self.incremental = incremental
        
        # Create solver
        self.solver = z3.Solver()
        self.solver.set("timeout", timeout_ms)
        
        # Track declared variables
        self.variables: Dict[str, z3.ExprRef] = {}
        self.functions: Dict[str, z3.FuncDeclRef] = {}
        
        # Statistics
        self.stats = {
            "queries": 0,
            "sat_count": 0,
            "unsat_count": 0,
            "timeout_count": 0,
            "total_time_ms": 0.0,
        }
    
    def reset(self) -> None:
        """Reset the solver state."""
        self.solver.reset()
        self.variables.clear()
        self.functions.clear()
    
    def push(self) -> None:
        """Push a new scope for incremental solving."""
        self.solver.push()
    
    def pop(self) -> None:
        """Pop a scope for incremental solving."""
        self.solver.pop()
    
    def declare_var(
        self,
        name: str,
        var_type: str = "int",
        bits: int = 32
    ) -> z3.ExprRef:
        """
        Declare a variable.
        
        Args:
            name: Variable name
            var_type: Type (int, bool, bitvec)
            bits: Bit width for bitvector
            
        Returns:
            Z3 expression for the variable
        """
        if name in self.variables:
            return self.variables[name]
        
        if var_type == "bool":
            var = z3.Bool(name)
        elif var_type == "int":
            var = z3.Int(name)
        elif var_type == "bitvec":
            var = z3.BitVec(name, bits)
        elif var_type == "real":
            var = z3.Real(name)
        else:
            # Default to int
            var = z3.Int(name)
        
        self.variables[name] = var
        return var
    
    def declare_function(
        self,
        name: str,
        arg_types: List[str],
        return_type: str
    ) -> z3.FuncDeclRef:
        """
        Declare an uninterpreted function.
        
        Args:
            name: Function name
            arg_types: List of argument types
            return_type: Return type
            
        Returns:
            Z3 function declaration
        """
        if name in self.functions:
            return self.functions[name]
        
        def type_to_z3_sort(t: str) -> z3.SortRef:
            if t == "bool":
                return z3.BoolSort()
            elif t == "int":
                return z3.IntSort()
            elif t == "real":
                return z3.RealSort()
            else:
                return z3.IntSort()
        
        arg_sorts = [type_to_z3_sort(t) for t in arg_types]
        ret_sort = type_to_z3_sort(return_type)
        
        func = z3.Function(name, *arg_sorts, ret_sort)
        self.functions[name] = func
        return func
    
    def add_constraint(self, constraint: z3.ExprRef) -> None:
        """Add a constraint to the solver."""
        self.solver.add(constraint)
    
    def add_constraints(self, constraints: List[z3.ExprRef]) -> None:
        """Add multiple constraints to the solver."""
        for c in constraints:
            self.solver.add(c)
    
    def check(self, *assumptions: z3.ExprRef) -> Tuple[SMTResult, Optional[SMTModel]]:
        """
        Check satisfiability.
        
        Args:
            assumptions: Optional assumptions for this check only
            
        Returns:
            Tuple of (result, model if SAT)
        """
        self.stats["queries"] += 1
        start_time = time.time()
        
        try:
            if assumptions:
                result = self.solver.check(*assumptions)
            else:
                result = self.solver.check()
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["total_time_ms"] += elapsed_ms
            
            if result == z3.sat:
                self.stats["sat_count"] += 1
                model = self._extract_model()
                return SMTResult.SAT, model
            
            elif result == z3.unsat:
                self.stats["unsat_count"] += 1
                return SMTResult.UNSAT, None
            
            else:
                self.stats["timeout_count"] += 1
                return SMTResult.UNKNOWN, None
        
        except z3.Z3Exception as e:
            if "timeout" in str(e).lower():
                self.stats["timeout_count"] += 1
                return SMTResult.TIMEOUT, None
            raise
    
    def _extract_model(self) -> SMTModel:
        """Extract model from SAT result."""
        z3_model = self.solver.model()
        assignments = {}
        
        for var_name, var in self.variables.items():
            try:
                value = z3_model.eval(var, model_completion=True)
                # Convert Z3 value to Python
                if z3.is_int_value(value):
                    assignments[var_name] = value.as_long()
                elif z3.is_true(value):
                    assignments[var_name] = True
                elif z3.is_false(value):
                    assignments[var_name] = False
                elif z3.is_rational_value(value):
                    assignments[var_name] = float(value.as_decimal(10))
                else:
                    assignments[var_name] = str(value)
            except Exception:
                # Variable not in model
                pass
        
        return SMTModel(assignments=assignments)
    
    def check_valid(self, formula: z3.ExprRef) -> Tuple[SMTResult, Optional[SMTModel]]:
        """
        Check if a formula is valid (always true).
        
        A formula is valid iff its negation is unsatisfiable.
        
        Args:
            formula: Formula to check
            
        Returns:
            Tuple of (UNSAT if valid, SAT with counterexample if invalid)
        """
        self.push()
        self.add_constraint(z3.Not(formula))
        result, model = self.check()
        self.pop()
        return result, model
    
    def check_implication(
        self,
        antecedent: z3.ExprRef,
        consequent: z3.ExprRef
    ) -> Tuple[SMTResult, Optional[SMTModel]]:
        """
        Check if antecedent => consequent is valid.
        
        Args:
            antecedent: Left side of implication
            consequent: Right side of implication
            
        Returns:
            UNSAT if implication holds, SAT with counterexample otherwise
        """
        return self.check_valid(z3.Implies(antecedent, consequent))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solver statistics."""
        return {
            **self.stats,
            "z3_stats": str(self.solver.statistics()) if Z3_AVAILABLE else {}
        }


class SMTBuilder:
    """
    Helper class for building SMT formulas.
    """
    
    @staticmethod
    def from_condition(
        condition: 'Condition',  # From spec_language
        solver: SMTSolver
    ) -> z3.ExprRef:
        """
        Convert a Condition to Z3 expression.
        
        Args:
            condition: Condition from spec_language
            solver: SMT solver for variable declarations
            
        Returns:
            Z3 expression
        """
        from ..specification.spec_language import ConditionType
        
        ctype = condition.cond_type
        operands = condition.operands
        
        def expr_to_z3(expr) -> z3.ExprRef:
            """Convert Expression to Z3."""
            from ..specification.spec_language import Variable, Constant, BinaryOp, FunctionCall
            
            if isinstance(expr, Variable):
                return solver.declare_var(expr.name, expr.var_type)
            elif isinstance(expr, Constant):
                if expr.const_type == "bool":
                    return z3.BoolVal(expr.value)
                elif expr.const_type == "int":
                    return z3.IntVal(expr.value)
                else:
                    return z3.IntVal(int(expr.value) if str(expr.value).isdigit() else 0)
            elif isinstance(expr, BinaryOp):
                left = expr_to_z3(expr.left)
                right = expr_to_z3(expr.right)
                op_map = {
                    "+": lambda a, b: a + b,
                    "-": lambda a, b: a - b,
                    "*": lambda a, b: a * b,
                    "/": lambda a, b: a / b,
                    "%": lambda a, b: a % b,
                    "&": lambda a, b: a & b,
                    "|": lambda a, b: a | b,
                }
                return op_map.get(expr.op, lambda a, b: a)(left, right)
            elif isinstance(expr, FunctionCall):
                # Use uninterpreted function
                func = solver.declare_function(expr.name, ["int"] * len(expr.args), "int")
                args = [expr_to_z3(arg) for arg in expr.args]
                return func(*args) if args else func
            else:
                # Nested condition
                return SMTBuilder.from_condition(expr, solver)
        
        if ctype == ConditionType.EQUALITY:
            return expr_to_z3(operands[0]) == expr_to_z3(operands[1])
        
        elif ctype == ConditionType.INEQUALITY:
            return expr_to_z3(operands[0]) != expr_to_z3(operands[1])
        
        elif ctype == ConditionType.LESS_THAN:
            return expr_to_z3(operands[0]) < expr_to_z3(operands[1])
        
        elif ctype == ConditionType.LESS_EQUAL:
            return expr_to_z3(operands[0]) <= expr_to_z3(operands[1])
        
        elif ctype == ConditionType.GREATER_THAN:
            return expr_to_z3(operands[0]) > expr_to_z3(operands[1])
        
        elif ctype == ConditionType.GREATER_EQUAL:
            return expr_to_z3(operands[0]) >= expr_to_z3(operands[1])
        
        elif ctype == ConditionType.IS_VALID:
            # isValid(x) -> x != 0 (simplified)
            var = expr_to_z3(operands[0])
            return var != 0 if hasattr(var, '__ne__') else z3.BoolVal(True)
        
        elif ctype == ConditionType.IS_IN_RANGE:
            var = expr_to_z3(operands[0])
            lo = expr_to_z3(operands[1])
            hi = expr_to_z3(operands[2])
            return z3.And(var >= lo, var <= hi)
        
        elif ctype == ConditionType.IS_IN_SET:
            var = expr_to_z3(operands[0])
            members = [expr_to_z3(m) for m in operands[1:]]
            return z3.Or(*[var == m for m in members])
        
        elif ctype == ConditionType.IMPLIES:
            ant = SMTBuilder.from_condition(operands[0], solver)
            cons = SMTBuilder.from_condition(operands[1], solver)
            return z3.Implies(ant, cons)
        
        elif ctype == ConditionType.AND:
            clauses = [SMTBuilder.from_condition(op, solver) for op in operands]
            return z3.And(*clauses) if clauses else z3.BoolVal(True)
        
        elif ctype == ConditionType.OR:
            clauses = [SMTBuilder.from_condition(op, solver) for op in operands]
            return z3.Or(*clauses) if clauses else z3.BoolVal(False)
        
        elif ctype == ConditionType.NOT:
            inner = SMTBuilder.from_condition(operands[0], solver)
            return z3.Not(inner)
        
        else:
            # Default to true
            return z3.BoolVal(True)
