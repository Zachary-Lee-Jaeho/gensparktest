"""
Fault localization for CGNR.
Identifies likely locations of bugs based on counterexamples.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import re


@dataclass
class FaultLocation:
    """A potential fault location."""
    line: int
    column: int = 0
    statement: str = ""
    suspiciousness: float = 0.0  # 0.0 to 1.0
    relevant_vars: List[str] = field(default_factory=list)
    reason: str = ""
    
    def __str__(self) -> str:
        return f"Line {self.line} (susp: {self.suspiciousness:.2f}): {self.statement}"


class FaultLocalizer:
    """
    Localizes faults in code based on counterexamples.
    
    Techniques:
    1. Spectrum-based fault localization (SBFL)
    2. Counterexample-driven analysis
    3. Statement-level suspiciousness ranking
    """
    
    def __init__(self):
        # Patterns that often indicate bugs
        self.suspicious_patterns = [
            (r'\breturn\s+\w+::\w+', 0.8),  # Return of enum value - often wrong
            (r'\bcase\s+\w+:', 0.7),         # Case statement
            (r'\?.*:', 0.6),                  # Ternary operator
            (r'\bif\s*\(', 0.5),              # Conditionals
        ]
    
    def localize(
        self,
        code: str,
        counterexample: 'Counterexample',  # From verifier
        statements: Optional[List[Dict[str, Any]]] = None
    ) -> List[FaultLocation]:
        """
        Localize potential faults based on counterexample.
        
        Args:
            code: Source code
            counterexample: Counterexample from verification
            statements: Optional parsed statements
            
        Returns:
            List of FaultLocation sorted by suspiciousness (descending)
        """
        locations = []
        lines = code.split('\n')
        
        # Variables involved in counterexample
        ce_vars = set(counterexample.input_values.keys()) if counterexample else set()
        violated = counterexample.violated_condition if counterexample else ""
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            # Compute suspiciousness
            susp = self._compute_suspiciousness(line, ce_vars, violated)
            
            if susp > 0.1:  # Threshold
                # Find relevant variables
                vars_in_line = self._extract_variables(line)
                relevant = list(vars_in_line & ce_vars) if ce_vars else list(vars_in_line)
                
                locations.append(FaultLocation(
                    line=i,
                    statement=line,
                    suspiciousness=susp,
                    relevant_vars=relevant,
                    reason=self._get_reason(line, ce_vars, violated)
                ))
        
        # Sort by suspiciousness
        locations.sort(key=lambda x: x.suspiciousness, reverse=True)
        
        return locations
    
    def _compute_suspiciousness(
        self,
        line: str,
        ce_vars: Set[str],
        violated: str
    ) -> float:
        """Compute suspiciousness score for a line."""
        susp = 0.0
        
        # Pattern-based scoring
        for pattern, base_score in self.suspicious_patterns:
            if re.search(pattern, line):
                susp = max(susp, base_score)
        
        # Variable involvement
        line_vars = self._extract_variables(line)
        if ce_vars and line_vars:
            overlap = len(line_vars & ce_vars) / len(ce_vars)
            susp += 0.3 * overlap
        
        # Violated condition involvement
        if violated:
            # Check if line relates to violated condition
            violated_terms = set(re.findall(r'\b\w+\b', violated))
            line_terms = set(re.findall(r'\b\w+\b', line))
            if violated_terms & line_terms:
                susp += 0.2
        
        # Return statements are often the bug location
        if 'return' in line:
            susp += 0.1
        
        return min(susp, 1.0)  # Cap at 1.0
    
    def _extract_variables(self, line: str) -> Set[str]:
        """Extract variable names from a line of code."""
        # Remove string literals
        line = re.sub(r'"[^"]*"', '', line)
        line = re.sub(r"'[^']*'", '', line)
        
        # Find identifiers
        identifiers = set(re.findall(r'\b([a-zA-Z_]\w*)\b', line))
        
        # Remove keywords
        keywords = {
            'if', 'else', 'switch', 'case', 'default', 'return', 'break',
            'continue', 'for', 'while', 'do', 'const', 'static', 'void',
            'int', 'unsigned', 'bool', 'true', 'false', 'nullptr'
        }
        
        return identifiers - keywords
    
    def _get_reason(
        self,
        line: str,
        ce_vars: Set[str],
        violated: str
    ) -> str:
        """Get human-readable reason for suspicion."""
        reasons = []
        
        if 'return' in line and '::' in line:
            reasons.append("Return value selection")
        
        if 'case' in line:
            reasons.append("Case branch")
        
        line_vars = self._extract_variables(line)
        involved = line_vars & ce_vars if ce_vars else set()
        if involved:
            reasons.append(f"Uses CE vars: {involved}")
        
        if '?' in line:
            reasons.append("Conditional expression")
        
        return "; ".join(reasons) if reasons else "General suspicion"
    
    def localize_from_trace(
        self,
        code: str,
        execution_trace: List[str]
    ) -> List[FaultLocation]:
        """
        Localize faults using execution trace.
        
        Args:
            code: Source code
            execution_trace: List of executed statements
            
        Returns:
            List of FaultLocation
        """
        # This would use more sophisticated SBFL techniques
        # For now, use simple heuristic: last statements in trace are more suspicious
        
        locations = []
        lines = code.split('\n')
        
        # Find lines that appear in trace
        trace_set = set(execution_trace)
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line in trace_set:
                # Suspiciousness based on position in trace
                try:
                    pos = execution_trace.index(line)
                    # Later in trace = more suspicious
                    susp = 0.3 + 0.7 * (pos / len(execution_trace))
                except ValueError:
                    susp = 0.5
                
                locations.append(FaultLocation(
                    line=i,
                    statement=line,
                    suspiciousness=susp,
                    reason="Appeared in execution trace"
                ))
        
        locations.sort(key=lambda x: x.suspiciousness, reverse=True)
        return locations
