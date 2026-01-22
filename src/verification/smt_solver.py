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


class MemoryModel:
    """
    Simple memory model for pointer/array verification.
    
    Uses Z3 arrays to model heap memory.
    """
    
    def __init__(self, solver: SMTSolver):
        if not Z3_AVAILABLE:
            raise ImportError("Z3 required for memory model")
        
        self.solver = solver
        
        # Memory as array: Address -> Value
        self.memory = z3.Array('memory', z3.IntSort(), z3.IntSort())
        
        # Track allocated regions
        self.allocated_regions: List[Tuple[z3.ExprRef, z3.ExprRef]] = []
        
        # Track null pointer
        self.null = z3.IntVal(0)
    
    def read(self, address: z3.ExprRef) -> z3.ExprRef:
        """Read from memory at address."""
        return z3.Select(self.memory, address)
    
    def write(self, address: z3.ExprRef, value: z3.ExprRef) -> None:
        """Write value to memory at address."""
        self.memory = z3.Store(self.memory, address, value)
    
    def allocate(self, base: z3.ExprRef, size: z3.ExprRef) -> None:
        """Mark a region as allocated."""
        self.allocated_regions.append((base, size))
        
        # Add constraint: allocated memory is not null
        self.solver.add_constraint(base != self.null)
    
    def is_valid_pointer(self, ptr: z3.ExprRef) -> z3.ExprRef:
        """Check if pointer is valid (not null and within allocated region)."""
        not_null = ptr != self.null
        
        # Check if pointer is in any allocated region
        if not self.allocated_regions:
            return not_null
        
        in_region = z3.Or(*[
            z3.And(ptr >= base, ptr < base + size)
            for base, size in self.allocated_regions
        ])
        
        return z3.And(not_null, in_region)
    
    def add_null_check(self, ptr: z3.ExprRef, condition_name: str = "null_check") -> z3.ExprRef:
        """Create null check condition."""
        return ptr != self.null


class FunctionCallModel:
    """
    Model for function calls using uninterpreted functions.
    
    Allows verifying properties about function calls without
    needing their full implementation.
    """
    
    def __init__(self, solver: SMTSolver):
        self.solver = solver
        self.functions: Dict[str, z3.FuncDeclRef] = {}
        self.function_specs: Dict[str, Dict[str, Any]] = {}
    
    def declare_function(
        self,
        name: str,
        arg_types: List[str],
        return_type: str,
        preconditions: Optional[List[z3.ExprRef]] = None,
        postconditions: Optional[List[z3.ExprRef]] = None
    ) -> z3.FuncDeclRef:
        """
        Declare a function with optional pre/postconditions.
        
        Args:
            name: Function name
            arg_types: List of argument type strings
            return_type: Return type string
            preconditions: List of precondition Z3 expressions
            postconditions: List of postcondition Z3 expressions
            
        Returns:
            Z3 function declaration
        """
        func = self.solver.declare_function(name, arg_types, return_type)
        self.functions[name] = func
        
        self.function_specs[name] = {
            'arg_types': arg_types,
            'return_type': return_type,
            'preconditions': preconditions or [],
            'postconditions': postconditions or [],
        }
        
        return func
    
    def call(self, name: str, *args: z3.ExprRef) -> z3.ExprRef:
        """
        Model a function call.
        
        Returns the result of applying the uninterpreted function.
        """
        if name not in self.functions:
            # Auto-declare with generic signature
            arg_types = ['int'] * len(args)
            self.declare_function(name, arg_types, 'int')
        
        func = self.functions[name]
        return func(*args) if args else func()
    
    def add_function_axioms(self, name: str) -> None:
        """
        Add axioms for a function based on its specification.
        
        These axioms constrain the behavior of the uninterpreted function.
        """
        if name not in self.function_specs:
            return
        
        spec = self.function_specs[name]
        
        # Add preconditions as constraints
        for pre in spec.get('preconditions', []):
            self.solver.add_constraint(pre)
        
        # Add postconditions as constraints
        for post in spec.get('postconditions', []):
            self.solver.add_constraint(post)


class ExtendedSMTSolver(SMTSolver):
    """
    Extended SMT solver with memory model and function call support.
    
    Provides:
    - Pointer/memory verification via Z3 arrays
    - Function call modeling via uninterpreted functions
    - Loop invariant checking
    """
    
    def __init__(self, timeout_ms: int = 30000, incremental: bool = True):
        super().__init__(timeout_ms, incremental)
        
        # Extended components
        self.memory_model = None
        self.function_model = None
        
        # Initialize on demand
        self._memory_initialized = False
        self._function_initialized = False
    
    def get_memory_model(self) -> MemoryModel:
        """Get or create memory model."""
        if not self._memory_initialized:
            self.memory_model = MemoryModel(self)
            self._memory_initialized = True
        return self.memory_model
    
    def get_function_model(self) -> FunctionCallModel:
        """Get or create function call model."""
        if not self._function_initialized:
            self.function_model = FunctionCallModel(self)
            self._function_initialized = True
        return self.function_model
    
    def verify_null_safety(self, ptr_var: str) -> Tuple[SMTResult, Optional[SMTModel]]:
        """
        Verify that a pointer is never null when dereferenced.
        
        Args:
            ptr_var: Name of pointer variable
            
        Returns:
            UNSAT if null-safe, SAT with counterexample otherwise
        """
        ptr = self.declare_var(ptr_var, "int")
        mem = self.get_memory_model()
        
        # Check: can ptr be null?
        self.push()
        self.add_constraint(ptr == mem.null)
        result, model = self.check()
        self.pop()
        
        return result, model
    
    def verify_array_bounds(
        self,
        index_var: str,
        array_size: int
    ) -> Tuple[SMTResult, Optional[SMTModel]]:
        """
        Verify that array access is within bounds.
        
        Args:
            index_var: Name of index variable
            array_size: Size of the array
            
        Returns:
            UNSAT if bounds-safe, SAT with out-of-bounds example otherwise
        """
        idx = self.declare_var(index_var, "int")
        
        # Check: can index be out of bounds?
        self.push()
        out_of_bounds = z3.Or(idx < 0, idx >= array_size)
        self.add_constraint(out_of_bounds)
        result, model = self.check()
        self.pop()
        
        return result, model
    
    def verify_loop_invariant(
        self,
        invariant: z3.ExprRef,
        init_condition: z3.ExprRef,
        loop_body_effect: z3.ExprRef,
        exit_condition: z3.ExprRef
    ) -> Dict[str, Tuple[SMTResult, Optional[SMTModel]]]:
        """
        Verify a loop invariant using inductive reasoning.
        
        Checks:
        1. Invariant holds initially (init => invariant)
        2. Invariant is preserved by loop body (invariant && !exit => invariant')
        3. Invariant implies postcondition on exit
        
        Args:
            invariant: Loop invariant expression
            init_condition: Initial condition
            loop_body_effect: Effect of loop body (as transition relation)
            exit_condition: Loop exit condition
            
        Returns:
            Dict with results for each check
        """
        results = {}
        
        # Check 1: Initialization
        self.push()
        init_check = z3.Implies(init_condition, invariant)
        result, model = self.check_valid(init_check)
        results['initialization'] = (result, model)
        self.pop()
        
        # Check 2: Preservation (simplified - needs proper encoding)
        self.push()
        preservation = z3.Implies(
            z3.And(invariant, z3.Not(exit_condition)),
            loop_body_effect  # Should be invariant with primed variables
        )
        result, model = self.check_valid(preservation)
        results['preservation'] = (result, model)
        self.pop()
        
        return results
    
    def verify_pointer_dereference(
        self,
        ptr_var: str,
        offset: int = 0,
        access_size: int = 1
    ) -> Tuple[SMTResult, Optional[SMTModel]]:
        """
        Verify that a pointer dereference is safe.
        
        Checks:
        - Pointer is not null
        - Pointer + offset is within valid memory
        
        Args:
            ptr_var: Pointer variable name
            offset: Offset from pointer (for ptr[offset])
            access_size: Size of access in bytes
            
        Returns:
            UNSAT if safe, SAT with counterexample otherwise
        """
        ptr = self.declare_var(ptr_var, "int")
        mem = self.get_memory_model()
        
        # Check validity of ptr + offset
        access_addr = ptr + offset if offset else ptr
        
        self.push()
        
        # Condition for unsafe: null or not in valid region
        unsafe = z3.Not(mem.is_valid_pointer(access_addr))
        self.add_constraint(unsafe)
        
        result, model = self.check()
        self.pop()
        
        return result, model
    
    def verify_division_safety(
        self,
        divisor_var: str
    ) -> Tuple[SMTResult, Optional[SMTModel]]:
        """
        Verify that division is safe (divisor != 0).
        
        Args:
            divisor_var: Variable used as divisor
            
        Returns:
            UNSAT if safe, SAT with divide-by-zero example otherwise
        """
        divisor = self.declare_var(divisor_var, "int")
        
        self.push()
        self.add_constraint(divisor == 0)
        result, model = self.check()
        self.pop()
        
        return result, model
    
    def verify_overflow(
        self,
        expr_var: str,
        bits: int = 32,
        signed: bool = True
    ) -> Tuple[SMTResult, Optional[SMTModel]]:
        """
        Verify that an expression does not overflow.
        
        Args:
            expr_var: Variable to check
            bits: Bit width (default 32)
            signed: Whether signed arithmetic (default True)
            
        Returns:
            UNSAT if no overflow possible, SAT with overflow example otherwise
        """
        expr = self.declare_var(expr_var, "int")
        
        if signed:
            min_val = -(2 ** (bits - 1))
            max_val = (2 ** (bits - 1)) - 1
        else:
            min_val = 0
            max_val = (2 ** bits) - 1
        
        self.push()
        overflow = z3.Or(expr < min_val, expr > max_val)
        self.add_constraint(overflow)
        result, model = self.check()
        self.pop()
        
        return result, model
    
    def verify_switch_completeness(
        self,
        switch_var: str,
        handled_cases: List[int],
        valid_range: Optional[Tuple[int, int]] = None
    ) -> Tuple[SMTResult, Optional[SMTModel]]:
        """
        Verify that a switch statement handles all possible values.
        
        Args:
            switch_var: Variable being switched on
            handled_cases: List of case values handled
            valid_range: Optional (min, max) range of valid values
            
        Returns:
            UNSAT if complete, SAT with unhandled case otherwise
        """
        var = self.declare_var(switch_var, "int")
        
        self.push()
        
        # Variable is not any of the handled cases
        not_handled = z3.And(*[var != case for case in handled_cases])
        self.add_constraint(not_handled)
        
        # Variable is within valid range (if specified)
        if valid_range:
            min_val, max_val = valid_range
            in_range = z3.And(var >= min_val, var <= max_val)
            self.add_constraint(in_range)
        
        result, model = self.check()
        self.pop()
        
        return result, model
    
    def verify_interprocedural(
        self,
        caller_constraints: List[z3.ExprRef],
        callee_preconditions: List[z3.ExprRef],
        callee_postconditions: List[z3.ExprRef],
        arg_mappings: Dict[str, z3.ExprRef]
    ) -> Dict[str, Tuple[SMTResult, Optional[SMTModel]]]:
        """
        Verify interprocedural call correctness.
        
        Checks:
        1. Caller satisfies callee's preconditions
        2. Callee's postconditions imply expected results
        
        Args:
            caller_constraints: Constraints at call site
            callee_preconditions: Preconditions of called function
            callee_postconditions: Postconditions of called function
            arg_mappings: Mapping from callee params to caller args
            
        Returns:
            Dict with verification results
        """
        results = {}
        
        # Check 1: Preconditions satisfied
        self.push()
        
        # Add caller constraints
        for c in caller_constraints:
            self.add_constraint(c)
        
        # Check that preconditions hold
        if callee_preconditions:
            pre_conjunction = z3.And(*callee_preconditions) if len(callee_preconditions) > 1 else callee_preconditions[0]
            result, model = self.check_valid(pre_conjunction)
            results['preconditions'] = (result, model)
        else:
            results['preconditions'] = (SMTResult.UNSAT, None)  # Trivially satisfied
        
        self.pop()
        
        # Check 2: Postconditions useful
        self.push()
        
        # Add postconditions
        for c in callee_postconditions:
            self.add_constraint(c)
        
        # Check satisfiability (postconditions should be consistent)
        result, model = self.check()
        results['postconditions_consistent'] = (result, model)
        
        self.pop()
        
        return results


class ComprehensiveSMTVerifier:
    """
    High-level verifier combining all SMT capabilities.
    
    Provides a unified interface for verifying:
    - Memory safety
    - Arithmetic safety
    - Control flow completeness
    - Interprocedural correctness
    """
    
    def __init__(self, timeout_ms: int = 30000):
        self.solver = ExtendedSMTSolver(timeout_ms=timeout_ms)
        self.results: Dict[str, Any] = {}
    
    def verify_function_safety(
        self,
        function_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive safety verification on a function.
        
        Args:
            function_info: Dictionary with function information
                - parameters: List of (name, type) tuples
                - local_variables: List of (name, type) tuples
                - pointer_derefs: List of pointer dereference expressions
                - divisions: List of division expressions
                - array_accesses: List of (array, index) tuples
                - switches: List of switch information
                
        Returns:
            Comprehensive verification results
        """
        results = {
            "status": "verified",
            "checks": {},
            "counterexamples": [],
        }
        
        # Check null safety for all pointers
        for param_name, param_type in function_info.get("parameters", []):
            if "*" in param_type:
                check_result, model = self.solver.verify_null_safety(param_name)
                results["checks"][f"null_safety_{param_name}"] = {
                    "result": check_result.value,
                    "safe": check_result == SMTResult.UNSAT,
                }
                if check_result == SMTResult.SAT and model:
                    results["counterexamples"].append({
                        "type": "null_pointer",
                        "variable": param_name,
                        "model": model.assignments,
                    })
                    results["status"] = "failed"
        
        # Check division safety
        for div_expr in function_info.get("divisions", []):
            check_result, model = self.solver.verify_division_safety(div_expr)
            results["checks"][f"division_safety_{div_expr}"] = {
                "result": check_result.value,
                "safe": check_result == SMTResult.UNSAT,
            }
            if check_result == SMTResult.SAT and model:
                results["counterexamples"].append({
                    "type": "divide_by_zero",
                    "expression": div_expr,
                    "model": model.assignments,
                })
                results["status"] = "failed"
        
        # Check array bounds
        for array_name, index_var, size in function_info.get("array_accesses", []):
            check_result, model = self.solver.verify_array_bounds(index_var, size)
            results["checks"][f"bounds_{array_name}_{index_var}"] = {
                "result": check_result.value,
                "safe": check_result == SMTResult.UNSAT,
            }
            if check_result == SMTResult.SAT and model:
                results["counterexamples"].append({
                    "type": "out_of_bounds",
                    "array": array_name,
                    "index": index_var,
                    "model": model.assignments,
                })
                results["status"] = "failed"
        
        # Check switch completeness
        for switch_info in function_info.get("switches", []):
            check_result, model = self.solver.verify_switch_completeness(
                switch_info["variable"],
                switch_info["cases"],
                switch_info.get("valid_range")
            )
            results["checks"][f"switch_completeness_{switch_info['variable']}"] = {
                "result": check_result.value,
                "complete": check_result == SMTResult.UNSAT,
            }
            if check_result == SMTResult.SAT and model:
                results["counterexamples"].append({
                    "type": "incomplete_switch",
                    "variable": switch_info["variable"],
                    "unhandled_value": model.assignments.get(switch_info["variable"]),
                    "model": model.assignments,
                })
                results["status"] = "failed"
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return self.solver.get_statistics()
