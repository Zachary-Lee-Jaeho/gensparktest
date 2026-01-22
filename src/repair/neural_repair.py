"""
Neural Repair Model for VEGA-Verified.

This module implements neural-guided code repair using:
1. Template-based repair for common compiler backend bugs
2. Pattern-based repair using learned heuristics
3. LLM-based repair for complex cases (when available)

The repair model works with counterexamples from Z3 verification to:
- Identify the specific bug pattern
- Generate targeted repair candidates
- Rank candidates by confidence

Key Bug Patterns in Compiler Backends:
1. Missing IsPCRel check (FK_Data_4 without PC-relative handling)
2. Wrong relocation type (R_*_32 instead of R_*_64)
3. Missing case in switch statement
4. Wrong return value mapping
5. Missing break statement
6. Off-by-one errors in encoding
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
import difflib


class RepairPattern(Enum):
    """Common repair patterns for compiler backend bugs."""
    MISSING_PCREL_CHECK = "missing_pcrel_check"
    WRONG_RELOC_SIZE = "wrong_reloc_size"
    MISSING_CASE = "missing_case"
    WRONG_RETURN = "wrong_return"
    MISSING_BREAK = "missing_break"
    OFF_BY_ONE = "off_by_one"
    MISSING_NULL_CHECK = "missing_null_check"
    WRONG_CONDITION = "wrong_condition"
    GENERIC = "generic"


@dataclass
class RepairCandidate:
    """A candidate repair."""
    code: str
    confidence: float
    pattern: RepairPattern
    description: str
    changes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "confidence": self.confidence,
            "pattern": self.pattern.value,
            "description": self.description,
            "changes": self.changes,
        }


@dataclass
class RepairContext:
    """Context for repair generation."""
    original_code: str
    counterexample: Dict[str, Any]
    specification: Any  # Specification
    violated_property: Optional[str] = None
    repair_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_prompt(self) -> str:
        """Format context as prompt for LLM-based repair."""
        prompt = f"""[REPAIR TASK]
Original code:
{self.original_code}

Counterexample:
Input: {self.counterexample.get('input_values', {})}
Expected: {self.counterexample.get('expected_output', 'unknown')}
Actual: {self.counterexample.get('actual_output', 'unknown')}

Violated property: {self.violated_property or 'specification violation'}

Previous repair attempts: {len(self.repair_history)}

Please fix the code to handle this case correctly.
"""
        return prompt


class NeuralRepairModel:
    """
    Neural repair model for compiler backend code.
    
    Uses a combination of:
    1. Pattern matching for known bug types
    2. Template-based repair for common patterns
    3. Heuristic-guided code transformation
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            "repairs_attempted": 0,
            "repairs_successful": 0,
            "patterns_matched": {},
        }
        
        # Known relocation mappings for different architectures
        self.reloc_mappings = {
            "RISCV": {
                "FK_NONE": "R_RISCV_NONE",
                "FK_Data_1": "R_RISCV_NONE",
                "FK_Data_2": "R_RISCV_NONE",
                "FK_Data_4": "R_RISCV_32",
                "FK_Data_8": "R_RISCV_64",
                "FK_Data_4_PCREL": "R_RISCV_32_PCREL",
            },
            "ARM": {
                "FK_NONE": "R_ARM_NONE",
                "FK_Data_1": "R_ARM_ABS8",
                "FK_Data_2": "R_ARM_ABS16",
                "FK_Data_4": "R_ARM_ABS32",
            },
            "AArch64": {
                "FK_NONE": "R_AARCH64_NONE",
                "FK_Data_2": "R_AARCH64_ABS16",
                "FK_Data_4": "R_AARCH64_ABS32",
                "FK_Data_8": "R_AARCH64_ABS64",
            },
            "MIPS": {
                "FK_NONE": "R_MIPS_NONE",
                "FK_Data_4": "R_MIPS_32",
                "FK_Data_8": "R_MIPS_64",
            },
            "X86_64": {
                "FK_NONE": "R_X86_64_NONE",
                "FK_Data_1": "R_X86_64_8",
                "FK_Data_2": "R_X86_64_16",
                "FK_Data_4": "R_X86_64_32",
                "FK_Data_8": "R_X86_64_64",
            },
        }
    
    def generate_repairs(
        self,
        context: RepairContext,
        num_candidates: int = 5
    ) -> List[RepairCandidate]:
        """
        Generate repair candidates for the given context.
        
        Args:
            context: Repair context with counterexample
            num_candidates: Maximum number of candidates to generate
            
        Returns:
            List of repair candidates sorted by confidence
        """
        self.stats["repairs_attempted"] += 1
        candidates = []
        
        # Analyze counterexample to determine bug pattern
        pattern = self._identify_pattern(context)
        
        if self.verbose:
            print(f"Identified bug pattern: {pattern.value}")
        
        # Generate repairs based on pattern
        if pattern == RepairPattern.MISSING_PCREL_CHECK:
            candidates.extend(self._repair_missing_pcrel(context))
        elif pattern == RepairPattern.WRONG_RELOC_SIZE:
            candidates.extend(self._repair_wrong_size(context))
        elif pattern == RepairPattern.MISSING_CASE:
            candidates.extend(self._repair_missing_case(context))
        elif pattern == RepairPattern.WRONG_RETURN:
            candidates.extend(self._repair_wrong_return(context))
        else:
            candidates.extend(self._repair_generic(context))
        
        # Track pattern
        self.stats["patterns_matched"][pattern.value] = \
            self.stats["patterns_matched"].get(pattern.value, 0) + 1
        
        # Sort by confidence and limit
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates[:num_candidates]
    
    def _identify_pattern(self, context: RepairContext) -> RepairPattern:
        """Identify the bug pattern from counterexample."""
        ce = context.counterexample
        input_vals = ce.get("input_values", {})
        expected = ce.get("expected_output")
        actual = ce.get("actual_output")
        
        # Check for IsPCRel-related bug
        if input_vals.get("IsPCRel", False):
            if expected and actual:
                if "PCREL" in str(expected) and "PCREL" not in str(actual):
                    return RepairPattern.MISSING_PCREL_CHECK
        
        # Check for size mismatch
        if expected and actual:
            exp_str = str(expected)
            act_str = str(actual)
            
            # Check for 32 vs 64 mismatch
            if ("64" in exp_str and "32" in act_str) or \
               ("32" in exp_str and "64" in act_str):
                return RepairPattern.WRONG_RELOC_SIZE
            
            # Check for wrong return value
            if exp_str != act_str:
                return RepairPattern.WRONG_RETURN
        
        # Check for missing case
        kind = input_vals.get("Kind", "")
        if kind and "case " + kind not in context.original_code:
            return RepairPattern.MISSING_CASE
        
        return RepairPattern.GENERIC
    
    def _repair_missing_pcrel(self, context: RepairContext) -> List[RepairCandidate]:
        """Generate repairs for missing IsPCRel check."""
        candidates = []
        ce = context.counterexample
        input_vals = ce.get("input_values", {})
        expected = ce.get("expected_output")
        kind = input_vals.get("Kind", "FK_Data_4")
        
        code = context.original_code
        
        # Strategy 1: Add IsPCRel ternary operator to existing case
        if f"case {kind}" in code:
            # Find the case and its return
            case_pattern = rf'case\s+{re.escape(kind)}\s*:\s*return\s+(\w+(?:::\w+)*)\s*;'
            match = re.search(case_pattern, code)
            
            if match:
                old_return = match.group(1)
                new_return = f"IsPCRel ? ELF::{expected} : {old_return}"
                
                new_code = re.sub(
                    case_pattern,
                    f"case {kind}: return {new_return};",
                    code
                )
                
                candidates.append(RepairCandidate(
                    code=new_code,
                    confidence=0.9,
                    pattern=RepairPattern.MISSING_PCREL_CHECK,
                    description=f"Added IsPCRel check for {kind}",
                    changes=[f"Changed return for case {kind} to use IsPCRel ternary"]
                ))
        
        # Strategy 2: Add IsPCRel block before switch
        if "if (IsPCRel)" not in code:
            # Find the switch statement
            switch_match = re.search(r'(switch\s*\([^)]+\)\s*\{)', code)
            
            if switch_match:
                pcrel_block = f"""if (IsPCRel) {{
        switch (Kind) {{
        case {kind}:
            return ELF::{expected};
        default:
            break;
        }}
    }}
    
    """
                new_code = code[:switch_match.start()] + pcrel_block + code[switch_match.start():]
                
                candidates.append(RepairCandidate(
                    code=new_code,
                    confidence=0.85,
                    pattern=RepairPattern.MISSING_PCREL_CHECK,
                    description="Added IsPCRel block before switch",
                    changes=["Added if (IsPCRel) block with PCRel-specific handling"]
                ))
        
        # Strategy 3: Add case to existing IsPCRel block
        if "if (IsPCRel)" in code:
            pcrel_section = code[code.find("if (IsPCRel)"):]
            if f"case {kind}" not in pcrel_section[:pcrel_section.find("}")] if "}" in pcrel_section else pcrel_section:
                # Find position to insert new case
                insert_pos = code.find("if (IsPCRel)")
                switch_start = code.find("switch", insert_pos)
                if switch_start > insert_pos:
                    case_insert = code.find("{", switch_start) + 1
                    new_case = f"\n        case {kind}:\n            return ELF::{expected};"
                    new_code = code[:case_insert] + new_case + code[case_insert:]
                    
                    candidates.append(RepairCandidate(
                        code=new_code,
                        confidence=0.88,
                        pattern=RepairPattern.MISSING_PCREL_CHECK,
                        description=f"Added {kind} case to IsPCRel block",
                        changes=[f"Added case {kind} in IsPCRel switch"]
                    ))
        
        return candidates
    
    def _repair_wrong_size(self, context: RepairContext) -> List[RepairCandidate]:
        """Generate repairs for wrong relocation size."""
        candidates = []
        ce = context.counterexample
        expected = ce.get("expected_output")
        actual = ce.get("actual_output")
        
        code = context.original_code
        
        if expected and actual:
            # Simple substitution
            new_code = code.replace(str(actual), str(expected))
            
            if new_code != code:
                candidates.append(RepairCandidate(
                    code=new_code,
                    confidence=0.95,
                    pattern=RepairPattern.WRONG_RELOC_SIZE,
                    description=f"Changed {actual} to {expected}",
                    changes=[f"Replaced {actual} with {expected}"]
                ))
        
        return candidates
    
    def _repair_missing_case(self, context: RepairContext) -> List[RepairCandidate]:
        """Generate repairs for missing switch case."""
        candidates = []
        ce = context.counterexample
        input_vals = ce.get("input_values", {})
        expected = ce.get("expected_output")
        kind = input_vals.get("Kind", "")
        
        code = context.original_code
        
        if kind and expected:
            # Find position to insert new case (before default)
            default_match = re.search(r'default\s*:', code)
            
            if default_match:
                new_case = f"case {kind}:\n        return ELF::{expected};\n    "
                insert_pos = default_match.start()
                new_code = code[:insert_pos] + new_case + code[insert_pos:]
                
                candidates.append(RepairCandidate(
                    code=new_code,
                    confidence=0.9,
                    pattern=RepairPattern.MISSING_CASE,
                    description=f"Added missing case {kind}",
                    changes=[f"Added case {kind}: return ELF::{expected};"]
                ))
            else:
                # No default, find last case and add after
                case_matches = list(re.finditer(r'case\s+\w+(?:::\w+)*\s*:.*?;', code, re.DOTALL))
                if case_matches:
                    last_case = case_matches[-1]
                    new_case = f"\n    case {kind}:\n        return ELF::{expected};"
                    insert_pos = last_case.end()
                    new_code = code[:insert_pos] + new_case + code[insert_pos:]
                    
                    candidates.append(RepairCandidate(
                        code=new_code,
                        confidence=0.85,
                        pattern=RepairPattern.MISSING_CASE,
                        description=f"Added missing case {kind}",
                        changes=[f"Added case {kind}: return ELF::{expected};"]
                    ))
        
        return candidates
    
    def _repair_wrong_return(self, context: RepairContext) -> List[RepairCandidate]:
        """Generate repairs for wrong return value."""
        candidates = []
        ce = context.counterexample
        input_vals = ce.get("input_values", {})
        expected = ce.get("expected_output")
        actual = ce.get("actual_output")
        kind = input_vals.get("Kind", "")
        
        code = context.original_code
        
        if kind and expected and actual:
            # Find the specific case and replace return
            case_pattern = rf'(case\s+{re.escape(kind)}\s*:.*?return\s+){re.escape(str(actual))}(\s*;)'
            
            match = re.search(case_pattern, code, re.DOTALL)
            if match:
                new_code = code[:match.start()] + match.group(1) + f"ELF::{expected}" + match.group(2) + code[match.end():]
                
                candidates.append(RepairCandidate(
                    code=new_code,
                    confidence=0.92,
                    pattern=RepairPattern.WRONG_RETURN,
                    description=f"Fixed return value for case {kind}",
                    changes=[f"Changed return from {actual} to ELF::{expected}"]
                ))
        
        return candidates
    
    def _repair_generic(self, context: RepairContext) -> List[RepairCandidate]:
        """Generate generic repairs using heuristics."""
        candidates = []
        ce = context.counterexample
        
        code = context.original_code
        
        # Add FIXME comment at likely fault location
        if "switch" in code:
            switch_pos = code.find("switch")
            new_code = code[:switch_pos] + "// FIXME: Check specification\n    " + code[switch_pos:]
            
            candidates.append(RepairCandidate(
                code=new_code,
                confidence=0.3,
                pattern=RepairPattern.GENERIC,
                description="Added FIXME comment for manual review",
                changes=["Added FIXME comment"]
            ))
        
        # Try adding assertion
        func_start = code.find("{")
        if func_start > 0:
            assertion = "\n    assert(Fixup.isValid() && \"Invalid fixup\");"
            new_code = code[:func_start+1] + assertion + code[func_start+1:]
            
            candidates.append(RepairCandidate(
                code=new_code,
                confidence=0.25,
                pattern=RepairPattern.GENERIC,
                description="Added input validation assertion",
                changes=["Added assert for input validation"]
            ))
        
        return candidates
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repair statistics."""
        return self.stats.copy()


class HybridRepairModel:
    """
    Hybrid repair model combining neural and rule-based approaches.
    
    Priority:
    1. Template-based repair for known patterns (high confidence)
    2. Pattern-based repair using learned heuristics
    3. Generic transformation as fallback
    """
    
    def __init__(self, verbose: bool = False):
        self.neural_model = NeuralRepairModel(verbose=verbose)
        self.verbose = verbose
        
        # Additional repair strategies
        self.repair_templates = {
            # (pattern, condition) -> repair template
            ("FK_Data_4", "IsPCRel"): "IsPCRel ? ELF::R_{arch}_32_PCREL : ELF::R_{arch}_32",
            ("FK_Data_8", "IsPCRel"): "IsPCRel ? ELF::R_{arch}_64_PCREL : ELF::R_{arch}_64",
        }
    
    def repair(
        self,
        context: RepairContext,
        num_candidates: int = 5
    ) -> List[RepairCandidate]:
        """
        Generate repair candidates using hybrid approach.
        
        Args:
            context: Repair context with counterexample
            num_candidates: Maximum candidates to generate
            
        Returns:
            Sorted list of repair candidates
        """
        all_candidates = []
        
        # Get neural model repairs
        neural_candidates = self.neural_model.generate_repairs(context, num_candidates)
        all_candidates.extend(neural_candidates)
        
        # Try template-based repairs
        template_candidates = self._apply_templates(context)
        all_candidates.extend(template_candidates)
        
        # Deduplicate and sort
        seen_codes = set()
        unique_candidates = []
        for c in all_candidates:
            code_hash = hash(c.code.strip())
            if code_hash not in seen_codes:
                seen_codes.add(code_hash)
                unique_candidates.append(c)
        
        unique_candidates.sort(key=lambda c: c.confidence, reverse=True)
        return unique_candidates[:num_candidates]
    
    def _apply_templates(self, context: RepairContext) -> List[RepairCandidate]:
        """Apply repair templates for known patterns."""
        candidates = []
        ce = context.counterexample
        input_vals = ce.get("input_values", {})
        kind = input_vals.get("Kind", "")
        ispcrel = input_vals.get("IsPCRel", False)
        
        # Detect architecture from code
        arch = self._detect_architecture(context.original_code)
        
        for (pattern_kind, pattern_cond), template in self.repair_templates.items():
            if kind == pattern_kind or kind.endswith(pattern_kind):
                if (pattern_cond == "IsPCRel" and ispcrel) or pattern_cond is None:
                    # Apply template
                    repair_expr = template.format(arch=arch)
                    
                    # Find and replace the return for this case
                    code = context.original_code
                    case_pattern = rf'(case\s+{re.escape(kind)}\s*:\s*return\s+)(\w+(?:::\w+)*)(\s*;)'
                    
                    match = re.search(case_pattern, code)
                    if match:
                        new_code = code[:match.start()] + match.group(1) + repair_expr + match.group(3) + code[match.end():]
                        
                        candidates.append(RepairCandidate(
                            code=new_code,
                            confidence=0.88,
                            pattern=RepairPattern.MISSING_PCREL_CHECK,
                            description=f"Applied template repair for {kind}",
                            changes=[f"Applied template: {template}"]
                        ))
        
        return candidates
    
    def _detect_architecture(self, code: str) -> str:
        """Detect target architecture from code."""
        if "RISCV" in code or "riscv" in code.lower():
            return "RISCV"
        elif "AArch64" in code or "aarch64" in code.lower():
            return "AARCH64"
        elif "ARM" in code:
            return "ARM"
        elif "MIPS" in code or "mips" in code.lower():
            return "MIPS"
        elif "X86" in code or "x86" in code.lower():
            return "X86_64"
        else:
            return "RISCV"  # Default


def create_neural_repair_model(verbose: bool = False) -> HybridRepairModel:
    """Factory function to create a neural repair model."""
    return HybridRepairModel(verbose=verbose)


# Quick test
if __name__ == "__main__":
    # Test repair
    test_code = """
    unsigned getRelocType(const MCFixup &Fixup, bool IsPCRel) const {
        unsigned Kind = Fixup.getTargetKind();
        
        switch (Kind) {
        case FK_NONE:
            return ELF::R_RISCV_NONE;
        case FK_Data_4:
            return ELF::R_RISCV_32;
        case FK_Data_8:
            return ELF::R_RISCV_64;
        default:
            return ELF::R_RISCV_NONE;
        }
    }
    """
    
    context = RepairContext(
        original_code=test_code,
        counterexample={
            "input_values": {"Kind": "FK_Data_4", "IsPCRel": True},
            "expected_output": "R_RISCV_32_PCREL",
            "actual_output": "R_RISCV_32",
        },
        specification=None,
        violated_property="IsPCRel implies PC-relative relocation"
    )
    
    model = create_neural_repair_model(verbose=True)
    candidates = model.repair(context)
    
    print("\nRepair Candidates:")
    for i, c in enumerate(candidates):
        print(f"\n{i+1}. Confidence: {c.confidence:.2f}")
        print(f"   Pattern: {c.pattern.value}")
        print(f"   Description: {c.description}")
        print(f"   Changes: {c.changes}")
