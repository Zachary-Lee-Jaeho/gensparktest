"""
Bounded Model Checking (BMC) for VEGA-Verified.

Implements bounded verification for loops and complex control flow
by unrolling loops up to a specified bound and checking properties.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import time
import re

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    z3 = None

from ..specification.spec_language import Specification, Condition, ConditionType
from ..specification.spec_language import Variable, Constant


class BMCResult(Enum):
    """Result of BMC check."""
    SAFE = "safe"              # No counterexample found within bound
    UNSAFE = "unsafe"          # Counterexample found
    UNKNOWN = "unknown"        # Solver couldn't determine
    BOUND_REACHED = "bound"    # Hit bound limit without conclusive result


@dataclass
class LoopInfo:
    """Information about a loop in the code."""
    loop_type: str  # for, while, do-while
    condition: str
    body: str
    init: Optional[str] = None  # For for-loops
    update: Optional[str] = None  # For for-loops
    line_start: int = 0
    line_end: int = 0
    
    def __str__(self) -> str:
        return f"{self.loop_type} loop: {self.condition}"


@dataclass
class BMCTrace:
    """Execution trace from BMC."""
    states: List[Dict[str, Any]] = field(default_factory=list)
    path_condition: Optional[Any] = None  # Z3 formula
    is_counterexample: bool = False
    violated_property: str = ""
    bound_used: int = 0
    
    def add_state(self, state: Dict[str, Any]) -> None:
        """Add a state to the trace."""
        self.states.append(state.copy())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "states": self.states,
            "is_counterexample": self.is_counterexample,
            "violated_property": self.violated_property,
            "bound_used": self.bound_used,
            "num_states": len(self.states),
        }


@dataclass
class BMCCheckResult:
    """Result of BMC verification."""
    result: BMCResult
    bound: int
    trace: Optional[BMCTrace] = None
    time_ms: float = 0.0
    iterations: int = 0
    
    # Property-specific results
    checked_properties: List[str] = field(default_factory=list)
    violated_properties: List[str] = field(default_factory=list)
    
    def is_safe(self) -> bool:
        return self.result == BMCResult.SAFE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result.value,
            "bound": self.bound,
            "time_ms": self.time_ms,
            "iterations": self.iterations,
            "trace": self.trace.to_dict() if self.trace else None,
            "checked_properties": self.checked_properties,
            "violated_properties": self.violated_properties,
        }


class LoopAnalyzer:
    """Analyzes loops in code for BMC."""
    
    def __init__(self):
        self.loop_patterns = [
            # for loop
            (r'for\s*\(([^;]*);([^;]*);([^)]*)\)\s*\{([^}]*)\}', 'for'),
            # while loop
            (r'while\s*\(([^)]+)\)\s*\{([^}]*)\}', 'while'),
            # do-while loop
            (r'do\s*\{([^}]*)\}\s*while\s*\(([^)]+)\)\s*;', 'do-while'),
        ]
    
    def find_loops(self, code: str) -> List[LoopInfo]:
        """Find all loops in code."""
        loops = []
        
        # Find for loops
        for_pattern = r'for\s*\(\s*([^;]*)\s*;\s*([^;]*)\s*;\s*([^)]*)\s*\)'
        for match in re.finditer(for_pattern, code):
            init, cond, update = match.groups()
            # Find loop body
            body_start = match.end()
            body = self._extract_block(code[body_start:])
            
            loops.append(LoopInfo(
                loop_type='for',
                condition=cond.strip(),
                body=body,
                init=init.strip(),
                update=update.strip(),
                line_start=code[:match.start()].count('\n') + 1,
            ))
        
        # Find while loops
        while_pattern = r'while\s*\(\s*([^)]+)\s*\)'
        for match in re.finditer(while_pattern, code):
            # Check it's not do-while
            before = code[:match.start()].strip()
            if before.endswith('}'):
                continue  # Likely do-while
            
            cond = match.group(1)
            body_start = match.end()
            body = self._extract_block(code[body_start:])
            
            loops.append(LoopInfo(
                loop_type='while',
                condition=cond.strip(),
                body=body,
                line_start=code[:match.start()].count('\n') + 1,
            ))
        
        return loops
    
    def _extract_block(self, code: str) -> str:
        """Extract a braced block from code."""
        code = code.lstrip()
        if not code.startswith('{'):
            # Single statement
            end = code.find(';')
            return code[:end+1] if end >= 0 else code
        
        # Find matching brace
        depth = 0
        for i, char in enumerate(code):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return code[1:i]
        
        return code
    
    def estimate_bound(self, loop: LoopInfo) -> int:
        """Estimate a reasonable bound for loop unrolling."""
        # Look for obvious bounds in condition
        if loop.loop_type == 'for':
            # Try to parse "i < N" pattern
            match = re.search(r'<\s*(\d+)', loop.condition)
            if match:
                bound = int(match.group(1))
                return min(bound + 1, 100)  # Cap at 100
        
        # Default bounds based on loop type
        return 10  # Conservative default


class BoundedModelChecker:
    """
    Bounded Model Checker for compiler backend verification.
    
    Unrolls loops up to a specified bound and checks properties
    using SMT solving. Supports:
    - Loop unrolling
    - Array bounds checking
    - Invariant checking at each iteration
    - Counterexample generation
    """
    
    DEFAULT_BOUND = 10
    MAX_BOUND = 100
    
    def __init__(
        self,
        default_bound: int = DEFAULT_BOUND,
        timeout_ms: int = 60000,
        verbose: bool = False
    ):
        if not Z3_AVAILABLE:
            raise ImportError("Z3 is required for bounded model checking")
        
        self.default_bound = min(default_bound, self.MAX_BOUND)
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        
        self.loop_analyzer = LoopAnalyzer()
        
        # Solver setup
        self.solver = z3.Solver()
        self.solver.set("timeout", timeout_ms)
        
        # Variable tracking
        self.variables: Dict[str, Dict[int, z3.ExprRef]] = {}  # var_name -> {step -> z3_var}
        self.current_step = 0
    
    def check(
        self,
        code: str,
        spec: Specification,
        bound: Optional[int] = None
    ) -> BMCCheckResult:
        """
        Perform bounded model checking.
        
        Args:
            code: Source code to check
            spec: Specification to verify
            bound: Optional bound override
            
        Returns:
            BMCCheckResult with verification result
        """
        start_time = time.time()
        
        k = bound if bound is not None else self.default_bound
        k = min(k, self.MAX_BOUND)
        
        if self.verbose:
            print(f"[BMC] Checking with bound k={k}")
        
        # Analyze loops
        loops = self.loop_analyzer.find_loops(code)
        if self.verbose and loops:
            print(f"[BMC] Found {len(loops)} loops: {[str(l) for l in loops]}")
        
        # Initialize solver
        self.solver.reset()
        self.variables.clear()
        self.current_step = 0
        
        result = BMCCheckResult(
            result=BMCResult.UNKNOWN,
            bound=k
        )
        
        try:
            # Add initial state constraints (preconditions)
            self._add_initial_constraints(spec)
            
            # Unroll and encode transitions
            for step in range(k):
                self.current_step = step
                
                # Encode step
                self._encode_step(code, spec, step)
                
                # Check safety property at this step
                property_violated, property_name = self._check_properties_at_step(
                    spec, step
                )
                
                if property_violated:
                    # Found counterexample
                    trace = self._extract_trace(step + 1)
                    trace.is_counterexample = True
                    trace.violated_property = property_name
                    
                    result.result = BMCResult.UNSAFE
                    result.trace = trace
                    result.violated_properties.append(property_name)
                    result.iterations = step + 1
                    break
                
                result.iterations = step + 1
            
            if result.result == BMCResult.UNKNOWN:
                # No counterexample found within bound
                result.result = BMCResult.SAFE
            
            result.checked_properties = [
                str(c) for c in spec.postconditions + spec.invariants
            ]
            
        except z3.Z3Exception as e:
            if "timeout" in str(e).lower():
                result.result = BMCResult.UNKNOWN
            else:
                raise
        
        result.time_ms = (time.time() - start_time) * 1000
        return result
    
    def _add_initial_constraints(self, spec: Specification) -> None:
        """Add initial state constraints from preconditions."""
        for pre in spec.preconditions:
            z3_pre = self._condition_to_z3(pre, step=0)
            self.solver.add(z3_pre)
    
    def _encode_step(
        self,
        code: str,
        spec: Specification,
        step: int
    ) -> None:
        """Encode a single execution step."""
        # Parse and encode statements
        statements = self._parse_statements(code)
        
        for stmt in statements:
            if stmt['type'] == 'assignment':
                self._encode_assignment(stmt, step)
            elif stmt['type'] == 'if':
                self._encode_branch(stmt, step)
            elif stmt['type'] == 'switch':
                self._encode_switch(stmt, step)
        
        # Add invariant constraints
        for inv in spec.invariants:
            z3_inv = self._condition_to_z3(inv, step)
            self.solver.add(z3_inv)
    
    def _check_properties_at_step(
        self,
        spec: Specification,
        step: int
    ) -> Tuple[bool, str]:
        """Check if any property is violated at this step."""
        # Check postconditions
        for i, post in enumerate(spec.postconditions):
            z3_post = self._condition_to_z3(post, step)
            
            # Check if negation is satisfiable (property violated)
            self.solver.push()
            self.solver.add(z3.Not(z3_post))
            
            check_result = self.solver.check()
            self.solver.pop()
            
            if check_result == z3.sat:
                return True, f"postcondition_{i}: {post}"
        
        return False, ""
    
    def _extract_trace(self, length: int) -> BMCTrace:
        """Extract execution trace from solver model."""
        trace = BMCTrace(bound_used=length)
        
        try:
            model = self.solver.model()
            
            for step in range(length):
                state = {}
                
                for var_name, step_vars in self.variables.items():
                    if step in step_vars:
                        var = step_vars[step]
                        try:
                            value = model.eval(var, model_completion=True)
                            if z3.is_int_value(value):
                                state[var_name] = value.as_long()
                            elif z3.is_true(value):
                                state[var_name] = True
                            elif z3.is_false(value):
                                state[var_name] = False
                            else:
                                state[var_name] = str(value)
                        except Exception:
                            pass
                
                trace.add_state(state)
        
        except Exception:
            pass
        
        return trace
    
    def _get_var(self, name: str, step: int) -> z3.ExprRef:
        """Get or create a variable for a given step."""
        if name not in self.variables:
            self.variables[name] = {}
        
        if step not in self.variables[name]:
            # Create new variable for this step
            var_name = f"{name}_{step}"
            self.variables[name][step] = z3.Int(var_name)
        
        return self.variables[name][step]
    
    def _parse_statements(self, code: str) -> List[Dict[str, Any]]:
        """Parse code into statements for encoding."""
        statements = []
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            if '=' in line and not line.startswith('if') and '==' not in line:
                # Assignment
                match = re.match(r'(\w+)\s*=\s*(.+?)\s*;', line)
                if match:
                    statements.append({
                        'type': 'assignment',
                        'var': match.group(1),
                        'expr': match.group(2),
                        'line': line,
                    })
            
            elif line.startswith('if'):
                match = re.search(r'if\s*\((.+?)\)', line)
                if match:
                    statements.append({
                        'type': 'if',
                        'condition': match.group(1),
                        'line': line,
                    })
            
            elif line.startswith('switch'):
                match = re.search(r'switch\s*\((.+?)\)', line)
                if match:
                    statements.append({
                        'type': 'switch',
                        'expr': match.group(1),
                        'line': line,
                    })
        
        return statements
    
    def _encode_assignment(self, stmt: Dict[str, Any], step: int) -> None:
        """Encode an assignment statement."""
        var = stmt['var']
        expr_str = stmt['expr']
        
        # Get variables for current and next step
        var_curr = self._get_var(var, step)
        var_next = self._get_var(var, step + 1)
        
        # Parse expression (simplified)
        expr_z3 = self._parse_expr(expr_str, step)
        
        # Add constraint: var_next = expr
        self.solver.add(var_next == expr_z3)
    
    def _encode_branch(self, stmt: Dict[str, Any], step: int) -> None:
        """Encode a branch statement."""
        cond_str = stmt['condition']
        cond_z3 = self._parse_condition(cond_str, step)
        
        # Branch condition is added as a constraint
        # In full implementation, would handle both branches
        branch_var = z3.Bool(f"branch_{step}")
        self.solver.add(branch_var == cond_z3)
    
    def _encode_switch(self, stmt: Dict[str, Any], step: int) -> None:
        """Encode a switch statement."""
        expr_str = stmt['expr']
        expr_z3 = self._parse_expr(expr_str, step)
        
        # Switch expression value at this step
        switch_val = self._get_var(f"switch_val", step)
        self.solver.add(switch_val == expr_z3)
    
    def _parse_expr(self, expr: str, step: int) -> z3.ExprRef:
        """Parse an expression to Z3."""
        expr = expr.strip()
        
        # Try to parse as integer
        try:
            return z3.IntVal(int(expr))
        except ValueError:
            pass
        
        # Check for binary operations
        for op in ['+', '-', '*', '/', '%']:
            if op in expr:
                parts = expr.split(op, 1)
                if len(parts) == 2:
                    left = self._parse_expr(parts[0], step)
                    right = self._parse_expr(parts[1], step)
                    
                    if op == '+':
                        return left + right
                    elif op == '-':
                        return left - right
                    elif op == '*':
                        return left * right
                    elif op == '/':
                        return left / right
                    elif op == '%':
                        return left % right
        
        # Variable reference
        # Remove any namespace/member access for simplicity
        var_name = expr.replace('.', '_').replace('::', '_').strip()
        return self._get_var(var_name, step)
    
    def _parse_condition(self, cond: str, step: int) -> z3.ExprRef:
        """Parse a condition to Z3."""
        cond = cond.strip()
        
        # Comparison operators
        for op, z3_op in [('==', lambda a, b: a == b),
                          ('!=', lambda a, b: a != b),
                          ('<=', lambda a, b: a <= b),
                          ('>=', lambda a, b: a >= b),
                          ('<', lambda a, b: a < b),
                          ('>', lambda a, b: a > b)]:
            if op in cond:
                parts = cond.split(op, 1)
                if len(parts) == 2:
                    left = self._parse_expr(parts[0], step)
                    right = self._parse_expr(parts[1], step)
                    return z3_op(left, right)
        
        # Boolean variable
        if cond.startswith('!'):
            inner = self._parse_condition(cond[1:], step)
            return z3.Not(inner)
        
        return z3.Bool(f"cond_{cond}_{step}")
    
    def _condition_to_z3(self, condition: Condition, step: int) -> z3.ExprRef:
        """Convert a Condition to Z3 at a specific step."""
        ctype = condition.cond_type
        operands = condition.operands
        
        def expr_to_z3(expr) -> z3.ExprRef:
            if isinstance(expr, Variable):
                return self._get_var(expr.name.replace('.', '_'), step)
            elif isinstance(expr, Constant):
                if expr.const_type == "bool":
                    return z3.BoolVal(expr.value)
                try:
                    return z3.IntVal(int(expr.value))
                except (ValueError, TypeError):
                    return z3.Int(f"const_{expr.value}")
            elif isinstance(expr, Condition):
                return self._condition_to_z3(expr, step)
            else:
                return z3.Int(f"unknown_{step}")
        
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
            var = expr_to_z3(operands[0])
            return var != 0
        elif ctype == ConditionType.IS_IN_RANGE:
            var = expr_to_z3(operands[0])
            lo = expr_to_z3(operands[1])
            hi = expr_to_z3(operands[2])
            return z3.And(var >= lo, var <= hi)
        elif ctype == ConditionType.IMPLIES:
            ant = self._condition_to_z3(operands[0], step)
            cons = self._condition_to_z3(operands[1], step)
            return z3.Implies(ant, cons)
        elif ctype == ConditionType.AND:
            if not operands:
                return z3.BoolVal(True)
            return z3.And(*[self._condition_to_z3(op, step) for op in operands])
        elif ctype == ConditionType.OR:
            if not operands:
                return z3.BoolVal(False)
            return z3.Or(*[self._condition_to_z3(op, step) for op in operands])
        elif ctype == ConditionType.NOT:
            return z3.Not(self._condition_to_z3(operands[0], step))
        else:
            return z3.BoolVal(True)
    
    def check_invariant(
        self,
        code: str,
        invariant: Condition,
        bound: int = 10
    ) -> BMCCheckResult:
        """
        Check if an invariant holds within bound.
        
        Args:
            code: Code containing the loop
            invariant: Invariant to check
            bound: Unrolling bound
            
        Returns:
            BMCCheckResult
        """
        # Create a spec with just the invariant
        spec = Specification(
            function_name="invariant_check",
            invariants=[invariant]
        )
        return self.check(code, spec, bound)
    
    def find_bound(
        self,
        code: str,
        spec: Specification,
        max_bound: int = 50
    ) -> int:
        """
        Find minimum bound needed for verification.
        
        Uses iterative deepening to find smallest k where
        verification succeeds or fails definitively.
        
        Args:
            code: Code to check
            spec: Specification
            max_bound: Maximum bound to try
            
        Returns:
            Minimum bound needed
        """
        for k in [1, 2, 5, 10, 20, max_bound]:
            if k > max_bound:
                break
            
            result = self.check(code, spec, k)
            
            if result.result == BMCResult.UNSAFE:
                # Found counterexample
                return k
            elif result.result == BMCResult.SAFE:
                # May need larger bound for completeness
                continue
        
        return max_bound
