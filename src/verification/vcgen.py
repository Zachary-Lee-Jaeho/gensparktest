"""
Verification Condition Generator for VEGA-Verified.
Generates VCs from code and specifications using weakest precondition calculus.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import re

# Try to import z3
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    z3 = None

from ..specification.spec_language import Specification, Condition, Variable, Constant


@dataclass
class CFGNode:
    """Node in control flow graph."""
    id: int
    node_type: str  # entry, exit, statement, branch, switch
    content: str = ""
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    
    # For branch nodes
    condition: Optional[str] = None
    true_branch: Optional[int] = None
    false_branch: Optional[int] = None
    
    # For switch nodes
    cases: Dict[str, int] = field(default_factory=dict)  # case_value -> successor_id
    default_branch: Optional[int] = None
    
    # Computed during verification
    wp: Optional[Any] = None  # Weakest precondition (Z3 formula)


@dataclass
class CFG:
    """Control Flow Graph for a function."""
    function_name: str
    nodes: Dict[int, CFGNode] = field(default_factory=dict)
    entry_id: int = 0
    exit_id: int = -1
    
    def add_node(self, node: CFGNode) -> None:
        """Add a node to the CFG."""
        self.nodes[node.id] = node
    
    def get_node(self, node_id: int) -> Optional[CFGNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def reverse_postorder(self) -> List[CFGNode]:
        """Get nodes in reverse postorder (for backward analysis)."""
        visited = set()
        order = []
        
        def dfs(node_id: int):
            if node_id in visited or node_id not in self.nodes:
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            for succ in node.successors:
                dfs(succ)
            order.append(node)
        
        dfs(self.entry_id)
        return order  # Already in reverse postorder due to DFS


@dataclass
class VerificationCondition:
    """A verification condition to be checked."""
    name: str
    formula: Any  # Z3 formula
    source: str  # Where the VC came from
    
    def __str__(self) -> str:
        return f"VC({self.name}): {self.formula}"


class CFGBuilder:
    """Builds CFG from parsed code."""
    
    def __init__(self):
        self.node_counter = 0
    
    def build(self, function_name: str, statements: List[Dict[str, Any]]) -> CFG:
        """
        Build CFG from parsed statements.
        
        Args:
            function_name: Name of the function
            statements: Parsed statements
            
        Returns:
            Control flow graph
        """
        cfg = CFG(function_name=function_name)
        self.node_counter = 0
        
        # Create entry node
        entry = self._create_node("entry")
        cfg.entry_id = entry.id
        cfg.add_node(entry)
        
        # Create exit node
        exit_node = self._create_node("exit")
        cfg.exit_id = exit_node.id
        cfg.add_node(exit_node)
        
        # Build nodes from statements
        prev_node = entry
        for stmt in statements:
            node = self._statement_to_node(stmt)
            cfg.add_node(node)
            
            # Connect to previous
            prev_node.successors.append(node.id)
            node.predecessors.append(prev_node.id)
            
            if stmt.get('type') == 'switch':
                # Switch node - connect cases to exit (simplified)
                for case_val, case_node_id in node.cases.items():
                    node.successors.append(cfg.exit_id)
                if node.default_branch:
                    node.successors.append(node.default_branch)
                prev_node = node
            elif stmt.get('type') == 'return':
                # Return connects to exit
                node.successors.append(cfg.exit_id)
                exit_node.predecessors.append(node.id)
                prev_node = node
            else:
                prev_node = node
        
        # Connect last statement to exit if not already
        if prev_node.id != cfg.exit_id and cfg.exit_id not in prev_node.successors:
            prev_node.successors.append(cfg.exit_id)
            exit_node.predecessors.append(prev_node.id)
        
        return cfg
    
    def _create_node(self, node_type: str, content: str = "") -> CFGNode:
        """Create a new CFG node."""
        node = CFGNode(
            id=self.node_counter,
            node_type=node_type,
            content=content
        )
        self.node_counter += 1
        return node
    
    def _statement_to_node(self, stmt: Dict[str, Any]) -> CFGNode:
        """Convert a statement to a CFG node."""
        stmt_type = stmt.get('type', 'other')
        text = stmt.get('text', '')
        
        if stmt_type == 'switch':
            node = self._create_node("switch", text)
            # Add cases
            for case in stmt.get('cases', []):
                case_val = case.get('case', 'default')
                node.cases[case_val] = self.node_counter  # Will point to next node
            return node
        
        elif stmt_type == 'if':
            node = self._create_node("branch", text)
            node.condition = stmt.get('condition', '')
            return node
        
        elif stmt_type == 'return':
            node = self._create_node("statement", text)
            return node
        
        else:
            node = self._create_node("statement", text)
            return node


class VCGenerator:
    """
    Generates verification conditions from code and specifications.
    
    Uses weakest precondition calculus:
    - wp(skip, Q) = Q
    - wp(x := e, Q) = Q[e/x]
    - wp(S1; S2, Q) = wp(S1, wp(S2, Q))
    - wp(if b then S1 else S2, Q) = (b => wp(S1, Q)) && (!b => wp(S2, Q))
    - wp(switch(e) {...}, Q) = AND of (e == case_i => wp(case_i, Q))
    """
    
    def __init__(self):
        self.cfg_builder = CFGBuilder()
        
        if not Z3_AVAILABLE:
            raise ImportError("Z3 is required for verification condition generation")
    
    def generate(
        self,
        code: str,
        spec: Specification,
        statements: Optional[List[Dict[str, Any]]] = None
    ) -> List[VerificationCondition]:
        """
        Generate verification conditions.
        
        Args:
            code: Source code
            spec: Specification to verify against
            statements: Optional pre-parsed statements
            
        Returns:
            List of verification conditions
        """
        # Parse code if statements not provided
        if statements is None:
            statements = self._parse_code(code)
        
        # Build CFG
        cfg = self.cfg_builder.build(spec.function_name, statements)
        
        # Compute weakest preconditions
        self._compute_wp(cfg, spec)
        
        # Generate VCs
        vcs = []
        
        # Main VC: Pre => wp(body, Post)
        entry_wp = cfg.get_node(cfg.entry_id).wp
        if entry_wp is not None:
            pre_formula = self._spec_to_z3(spec.preconditions)
            
            vc = VerificationCondition(
                name="main",
                formula=z3.Implies(pre_formula, entry_wp),
                source=f"Pre => wp(body, Post) for {spec.function_name}"
            )
            vcs.append(vc)
        
        # Invariant VCs
        for i, inv in enumerate(spec.invariants):
            inv_formula = self._condition_to_z3(inv)
            vc = VerificationCondition(
                name=f"invariant_{i}",
                formula=inv_formula,
                source=f"Invariant {i}: {inv}"
            )
            vcs.append(vc)
        
        return vcs
    
    def _parse_code(self, code: str) -> List[Dict[str, Any]]:
        """Parse code into statements (simplified)."""
        statements = []
        
        # Split by lines and basic parsing
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            if line.startswith('switch'):
                # Find the full switch block
                statements.append({
                    'type': 'switch',
                    'text': line,
                    'cases': self._parse_switch_cases(code)
                })
            elif line.startswith('if'):
                match = re.search(r'if\s*\((.+?)\)', line)
                statements.append({
                    'type': 'if',
                    'text': line,
                    'condition': match.group(1) if match else ''
                })
            elif line.startswith('return'):
                match = re.search(r'return\s+(.+?)\s*;', line)
                statements.append({
                    'type': 'return',
                    'text': line,
                    'value': match.group(1) if match else ''
                })
            elif '=' in line and not line.startswith('if'):
                statements.append({
                    'type': 'assignment',
                    'text': line
                })
        
        return statements
    
    def _parse_switch_cases(self, code: str) -> List[Dict[str, str]]:
        """Parse switch cases from code."""
        cases = []
        case_pattern = r'case\s+(\w+(?:::\w+)?)\s*:\s*return\s+(\w+(?:::\w+)?)\s*;'
        
        for match in re.finditer(case_pattern, code):
            cases.append({
                'case': match.group(1),
                'return': match.group(2)
            })
        
        return cases
    
    def _compute_wp(self, cfg: CFG, spec: Specification) -> None:
        """Compute weakest preconditions through the CFG."""
        # Start from exit with postcondition
        exit_node = cfg.get_node(cfg.exit_id)
        if exit_node:
            exit_node.wp = self._spec_to_z3(spec.postconditions)
        
        # Backward propagation through reverse postorder
        for node in cfg.reverse_postorder():
            if node.id == cfg.exit_id:
                continue  # Already set
            
            if node.node_type == "switch":
                node.wp = self._wp_switch(node, cfg, spec)
            elif node.node_type == "branch":
                node.wp = self._wp_branch(node, cfg)
            elif node.node_type == "statement":
                node.wp = self._wp_statement(node, cfg)
            elif node.node_type == "entry":
                # Entry node gets wp from first real node
                if node.successors:
                    succ = cfg.get_node(node.successors[0])
                    node.wp = succ.wp if succ else z3.BoolVal(True)
    
    def _wp_switch(
        self,
        node: CFGNode,
        cfg: CFG,
        spec: Specification
    ) -> z3.ExprRef:
        """Compute wp for switch statement."""
        # For each case, we need: switch_expr == case_val => wp(case_body)
        clauses = []
        
        # Get exit wp as default continuation
        exit_node = cfg.get_node(cfg.exit_id)
        post_wp = exit_node.wp if exit_node and exit_node.wp else z3.BoolVal(True)
        
        # Parse switch expression
        switch_var = z3.Int("Fixup_kind")  # Simplified
        
        for case_val, _ in node.cases.items():
            # Create clause: switch_var == case_val => post_wp
            case_const = z3.Int(f"case_{case_val}")
            clause = z3.Implies(switch_var == case_const, post_wp)
            clauses.append(clause)
        
        return z3.And(*clauses) if clauses else post_wp
    
    def _wp_branch(self, node: CFGNode, cfg: CFG) -> z3.ExprRef:
        """Compute wp for branch (if) statement."""
        # wp(if b then S1 else S2, Q) = (b => wp(S1, Q)) && (!b => wp(S2, Q))
        
        # Simplified: condition as boolean variable
        cond = z3.Bool(f"cond_{node.id}")
        
        # Get wp from successors
        true_wp = z3.BoolVal(True)
        false_wp = z3.BoolVal(True)
        
        for succ_id in node.successors:
            succ = cfg.get_node(succ_id)
            if succ and succ.wp:
                if node.true_branch == succ_id:
                    true_wp = succ.wp
                elif node.false_branch == succ_id:
                    false_wp = succ.wp
                else:
                    # Default: use for both
                    true_wp = succ.wp
                    false_wp = succ.wp
        
        return z3.And(
            z3.Implies(cond, true_wp),
            z3.Implies(z3.Not(cond), false_wp)
        )
    
    def _wp_statement(self, node: CFGNode, cfg: CFG) -> z3.ExprRef:
        """Compute wp for regular statement."""
        # Get wp from successor
        if node.successors:
            succ = cfg.get_node(node.successors[0])
            if succ and succ.wp:
                return succ.wp
        
        return z3.BoolVal(True)
    
    def _spec_to_z3(self, conditions: List[Condition]) -> z3.ExprRef:
        """Convert specification conditions to Z3 formula."""
        if not conditions:
            return z3.BoolVal(True)
        
        z3_conditions = [self._condition_to_z3(c) for c in conditions]
        return z3.And(*z3_conditions)
    
    def _condition_to_z3(self, condition: Condition) -> z3.ExprRef:
        """Convert a single condition to Z3 formula."""
        from ..specification.spec_language import ConditionType
        
        ctype = condition.cond_type
        operands = condition.operands
        
        def expr_to_z3(expr) -> z3.ExprRef:
            if isinstance(expr, Variable):
                # Simplified: all variables are integers
                return z3.Int(expr.name.replace('.', '_'))
            elif isinstance(expr, Constant):
                if expr.const_type == "bool":
                    return z3.BoolVal(expr.value)
                else:
                    # Try to parse as int, otherwise use symbolic
                    try:
                        return z3.IntVal(int(expr.value))
                    except (ValueError, TypeError):
                        return z3.Int(f"const_{expr.value}")
            elif isinstance(expr, Condition):
                return self._condition_to_z3(expr)
            else:
                return z3.Int("unknown")
        
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
            # isValid(x) -> x != 0
            var = expr_to_z3(operands[0])
            return var != 0
        
        elif ctype == ConditionType.IS_IN_RANGE:
            var = expr_to_z3(operands[0])
            lo = expr_to_z3(operands[1])
            hi = expr_to_z3(operands[2])
            return z3.And(var >= lo, var <= hi)
        
        elif ctype == ConditionType.IS_IN_SET:
            var = expr_to_z3(operands[0])
            members = [expr_to_z3(m) for m in operands[1:]]
            return z3.Or(*[var == m for m in members]) if members else z3.BoolVal(False)
        
        elif ctype == ConditionType.IMPLIES:
            ant = self._condition_to_z3(operands[0])
            cons = self._condition_to_z3(operands[1])
            return z3.Implies(ant, cons)
        
        elif ctype == ConditionType.AND:
            if not operands:
                return z3.BoolVal(True)
            return z3.And(*[self._condition_to_z3(op) for op in operands])
        
        elif ctype == ConditionType.OR:
            if not operands:
                return z3.BoolVal(False)
            return z3.Or(*[self._condition_to_z3(op) for op in operands])
        
        elif ctype == ConditionType.NOT:
            return z3.Not(self._condition_to_z3(operands[0]))
        
        else:
            return z3.BoolVal(True)
