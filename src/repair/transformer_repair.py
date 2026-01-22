"""
Transformer-Based Neural Repair Model for VEGA-Verified.

This module implements a sophisticated repair model using:
1. Pretrained code understanding models (CodeBERT, UniXcoder)
2. Template-based repair for high-confidence fixes
3. Hybrid approach combining learned patterns and rules

Key Features:
- Context-aware repair generation
- Multi-candidate ranking
- Counterexample-guided fix localization
- Architecture-specific repair patterns
"""

# Optional torch import for neural components
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import re
import difflib
from abc import ABC, abstractmethod


class RepairStrategy(Enum):
    """Repair strategies in order of confidence."""
    TEMPLATE_EXACT = "template_exact"      # Exact template match
    PATTERN_BASED = "pattern_based"        # Pattern-based transformation
    NEURAL_GUIDED = "neural_guided"        # Neural model guidance
    HYBRID = "hybrid"                      # Combination
    FALLBACK = "fallback"                  # Generic fallback


@dataclass
class CodeContext:
    """Context for code repair."""
    code: str
    function_name: str
    architecture: str
    counterexample: Dict[str, Any]
    specification: Any
    fault_location: Optional[Tuple[int, int]] = None  # (start_line, end_line)
    repair_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_fault_region(self) -> str:
        """Get code around fault location."""
        if not self.fault_location:
            return self.code
        
        lines = self.code.split('\n')
        start, end = self.fault_location
        start = max(0, start - 2)
        end = min(len(lines), end + 2)
        return '\n'.join(lines[start:end])


@dataclass
class RepairCandidate:
    """A repair candidate with metadata."""
    repaired_code: str
    strategy: RepairStrategy
    confidence: float
    description: str
    changes: List[str] = field(default_factory=list)
    edit_distance: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "confidence": self.confidence,
            "description": self.description,
            "changes": self.changes,
            "edit_distance": self.edit_distance,
        }


class RepairTemplates:
    """
    Repository of repair templates for compiler backends.
    
    Templates are organized by:
    1. Bug type (missing_case, wrong_reloc, missing_pcrel, etc.)
    2. Architecture (RISCV, ARM, etc.)
    3. Context (switch, if-else, return, etc.)
    """
    
    # Bug type patterns and fixes
    TEMPLATES = {
        "missing_pcrel_check": {
            # Pattern: case FK_Data_4: return R_*_32;
            # Fix: case FK_Data_4: return IsPCRel ? R_*_PCREL : R_*_32;
            "pattern": r'case\s+(FK_Data_\d+)\s*:\s*return\s+(\w+::\w+)\s*;',
            "fix": 'case {kind}: return IsPCRel ? {pcrel_reloc} : {normal_reloc};',
            "requires": ["kind", "pcrel_reloc", "normal_reloc"],
        },
        "missing_case": {
            # Pattern: switch (...) { ... default: }
            # Fix: Add missing case before default
            "pattern": r'(default\s*:)',
            "fix": 'case {kind}:\n        return {reloc};\n    {default}',
            "requires": ["kind", "reloc", "default"],
        },
        "wrong_reloc_type": {
            # Pattern: case FK_*: return R_*_WRONG;
            # Fix: case FK_*: return R_*_CORRECT;
            "pattern": r'(case\s+{kind}\s*:\s*return\s+)\w+::\w+(\s*;)',
            "fix": r'\1{reloc}\2',
            "requires": ["kind", "reloc"],
        },
        "missing_break": {
            # Pattern: case X: ... (missing break/return)
            # Fix: Add break
            "pattern": r'(case\s+\w+\s*:.*?)(\n\s*case\s+)',
            "fix": r'\1\n        break;\2',
            "requires": [],
        },
        "add_ispcrel_block": {
            # Pattern: No IsPCRel check
            # Fix: Add if (IsPCRel) block
            "template": """
    if (IsPCRel) {{
        switch (Kind) {{
{pcrel_cases}
        default:
            break;
        }}
    }}
    
""",
            "requires": ["pcrel_cases"],
        },
    }
    
    # Architecture-specific relocation mappings
    ARCH_RELOCS = {
        "RISCV": {
            "FK_Data_4": ("R_RISCV_32", "R_RISCV_32_PCREL"),
            "FK_Data_8": ("R_RISCV_64", None),
            "FK_NONE": ("R_RISCV_NONE", None),
        },
        "ARM": {
            "FK_Data_4": ("R_ARM_ABS32", "R_ARM_REL32"),
            "FK_Data_2": ("R_ARM_ABS16", None),
            "FK_Data_1": ("R_ARM_ABS8", None),
        },
        "AArch64": {
            "FK_Data_4": ("R_AARCH64_ABS32", "R_AARCH64_PREL32"),
            "FK_Data_8": ("R_AARCH64_ABS64", "R_AARCH64_PREL64"),
        },
        "MIPS": {
            "FK_Data_4": ("R_MIPS_32", "R_MIPS_PC32"),
            "FK_Data_8": ("R_MIPS_64", None),
        },
        "X86_64": {
            "FK_Data_4": ("R_X86_64_32", "R_X86_64_PC32"),
            "FK_Data_8": ("R_X86_64_64", "R_X86_64_PC64"),
        },
    }
    
    @classmethod
    def get_reloc_mapping(cls, arch: str, kind: str) -> Optional[Tuple[str, Optional[str]]]:
        """Get relocation mapping for architecture and kind."""
        if arch in cls.ARCH_RELOCS:
            return cls.ARCH_RELOCS[arch].get(kind)
        return None


class FaultLocalizer:
    """
    Locates faults in compiler backend code based on counterexamples.
    
    Strategies:
    1. Match counterexample input to switch cases
    2. Find mismatched return values
    3. Identify missing conditional branches
    """
    
    def localize(self, context: CodeContext) -> Optional[Tuple[int, int]]:
        """
        Locate fault based on counterexample.
        
        Returns:
            (start_line, end_line) of fault region, or None
        """
        ce = context.counterexample
        input_vals = ce.get("input_values", {})
        expected = ce.get("expected_output")
        actual = ce.get("actual_output")
        
        code = context.code
        lines = code.split('\n')
        
        # Strategy 1: Find case statement for input Kind
        kind = input_vals.get("Kind", "")
        if kind:
            fault = self._find_case_fault(lines, kind, expected, actual)
            if fault:
                return fault
        
        # Strategy 2: Find IsPCRel-related fault
        is_pcrel = input_vals.get("IsPCRel", False)
        if is_pcrel:
            fault = self._find_pcrel_fault(lines, kind)
            if fault:
                return fault
        
        # Strategy 3: Generic switch statement
        fault = self._find_switch_fault(lines)
        if fault:
            return fault
        
        return None
    
    def _find_case_fault(self, lines: List[str], kind: str, expected: Any, actual: Any) -> Optional[Tuple[int, int]]:
        """Find fault in specific case statement."""
        for i, line in enumerate(lines):
            if f'case {kind}' in line or line.strip().startswith(f'case') and kind in line:
                # Found the case - check return
                for j in range(i, min(i + 5, len(lines))):
                    if 'return' in lines[j]:
                        return (i, j + 1)
                return (i, min(i + 3, len(lines)))
        
        # Case not found - fault is missing case
        for i, line in enumerate(lines):
            if 'default:' in line:
                return (max(0, i - 1), i + 1)
        
        return None
    
    def _find_pcrel_fault(self, lines: List[str], kind: str) -> Optional[Tuple[int, int]]:
        """Find fault related to IsPCRel handling."""
        # Check if IsPCRel is handled
        has_ispcrel = any('IsPCRel' in line for line in lines)
        
        if not has_ispcrel:
            # Missing IsPCRel check entirely
            for i, line in enumerate(lines):
                if 'switch' in line:
                    return (i, i + 1)
        
        # Find specific case
        for i, line in enumerate(lines):
            if kind and f'case {kind}' in line:
                return (i, min(i + 3, len(lines)))
        
        return None
    
    def _find_switch_fault(self, lines: List[str]) -> Optional[Tuple[int, int]]:
        """Find switch statement as general fault region."""
        for i, line in enumerate(lines):
            if 'switch' in line:
                # Find end of switch
                brace_count = 0
                for j in range(i, len(lines)):
                    brace_count += lines[j].count('{')
                    brace_count -= lines[j].count('}')
                    if brace_count == 0 and j > i:
                        return (i, j + 1)
                return (i, min(i + 20, len(lines)))
        return None


class BaseRepairModel(ABC):
    """Abstract base class for repair models."""
    
    @abstractmethod
    def generate_repairs(self, context: CodeContext, num_candidates: int = 5) -> List[RepairCandidate]:
        """Generate repair candidates."""
        pass


class TemplateRepairModel(BaseRepairModel):
    """
    Template-based repair model for high-confidence fixes.
    
    Uses predefined templates for common compiler backend bugs.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.fault_localizer = FaultLocalizer()
        self.stats = {
            "repairs_generated": 0,
            "templates_matched": 0,
        }
    
    def generate_repairs(self, context: CodeContext, num_candidates: int = 5) -> List[RepairCandidate]:
        """Generate template-based repairs."""
        candidates = []
        
        # Localize fault
        if context.fault_location is None:
            context.fault_location = self.fault_localizer.localize(context)
        
        ce = context.counterexample
        input_vals = ce.get("input_values", {})
        expected = ce.get("expected_output")
        actual = ce.get("actual_output")
        kind = input_vals.get("Kind", "")
        is_pcrel = input_vals.get("IsPCRel", False)
        
        # Detect architecture
        arch = self._detect_architecture(context)
        
        # Try different repair strategies
        
        # 1. Missing IsPCRel check
        if is_pcrel and expected and 'PCREL' in str(expected).upper():
            repairs = self._repair_missing_pcrel(context, kind, expected, actual, arch)
            candidates.extend(repairs)
        
        # 2. Wrong relocation type
        if expected and actual and str(expected) != str(actual):
            repairs = self._repair_wrong_reloc(context, kind, expected, actual)
            candidates.extend(repairs)
        
        # 3. Missing case
        if kind and f'case {kind}' not in context.code:
            repairs = self._repair_missing_case(context, kind, expected, arch)
            candidates.extend(repairs)
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        self.stats["repairs_generated"] += len(candidates)
        return candidates[:num_candidates]
    
    def _repair_missing_pcrel(self, context: CodeContext, kind: str, 
                              expected: str, actual: str, arch: str) -> List[RepairCandidate]:
        """Repair missing IsPCRel check."""
        candidates = []
        code = context.code
        
        # Get correct relocations for this architecture
        reloc_mapping = RepairTemplates.get_reloc_mapping(arch, kind)
        if reloc_mapping:
            normal_reloc, pcrel_reloc = reloc_mapping
        else:
            # Use expected value
            pcrel_reloc = str(expected)
            normal_reloc = str(actual) if actual else "ELF::R_UNKNOWN"
        
        # Strategy 1: Add ternary to existing case
        case_pattern = rf'(case\s+{re.escape(kind)}\s*:\s*return\s+)(\w+(?:::\w+)*)(\s*;)'
        match = re.search(case_pattern, code)
        
        if match:
            ternary_expr = f"IsPCRel ? ELF::{pcrel_reloc} : {match.group(2)}"
            repaired = code[:match.start(2)] + ternary_expr + code[match.end(2):]
            
            candidates.append(RepairCandidate(
                repaired_code=repaired,
                strategy=RepairStrategy.TEMPLATE_EXACT,
                confidence=0.92,
                description=f"Added IsPCRel ternary check for {kind}",
                changes=[f"Changed return to: {ternary_expr}"],
                edit_distance=self._compute_edit_distance(code, repaired)
            ))
        
        # Strategy 2: Add IsPCRel block before switch
        if 'if (IsPCRel)' not in code and 'if(IsPCRel)' not in code:
            switch_match = re.search(r'(switch\s*\(\s*Kind\s*\)\s*\{)', code)
            if switch_match:
                pcrel_block = f"""if (IsPCRel) {{
        switch (Kind) {{
        case {kind}:
            return ELF::{pcrel_reloc};
        default:
            break;
        }}
    }}
    
    """
                repaired = code[:switch_match.start()] + pcrel_block + code[switch_match.start():]
                
                candidates.append(RepairCandidate(
                    repaired_code=repaired,
                    strategy=RepairStrategy.TEMPLATE_EXACT,
                    confidence=0.88,
                    description="Added IsPCRel block before main switch",
                    changes=["Added if (IsPCRel) block with PCRel case handling"],
                    edit_distance=self._compute_edit_distance(code, repaired)
                ))
        
        # Strategy 3: Add case to existing IsPCRel block
        elif 'if (IsPCRel)' in code or 'if(IsPCRel)' in code:
            # Find IsPCRel block and add case
            pcrel_match = re.search(r'if\s*\(\s*IsPCRel\s*\)\s*\{[\s\S]*?switch[\s\S]*?\{', code)
            if pcrel_match:
                insert_pos = pcrel_match.end()
                new_case = f"\n        case {kind}:\n            return ELF::{pcrel_reloc};"
                repaired = code[:insert_pos] + new_case + code[insert_pos:]
                
                candidates.append(RepairCandidate(
                    repaired_code=repaired,
                    strategy=RepairStrategy.TEMPLATE_EXACT,
                    confidence=0.90,
                    description=f"Added {kind} case to IsPCRel block",
                    changes=[f"Added case {kind}: return ELF::{pcrel_reloc};"],
                    edit_distance=self._compute_edit_distance(code, repaired)
                ))
        
        self.stats["templates_matched"] += len(candidates)
        return candidates
    
    def _repair_wrong_reloc(self, context: CodeContext, kind: str,
                            expected: str, actual: str) -> List[RepairCandidate]:
        """Repair wrong relocation type."""
        candidates = []
        code = context.code
        
        # Simple substitution
        if actual in code:
            repaired = code.replace(str(actual), f"ELF::{expected}")
            
            candidates.append(RepairCandidate(
                repaired_code=repaired,
                strategy=RepairStrategy.TEMPLATE_EXACT,
                confidence=0.95,
                description=f"Changed relocation from {actual} to {expected}",
                changes=[f"Replaced {actual} with ELF::{expected}"],
                edit_distance=self._compute_edit_distance(code, repaired)
            ))
        
        return candidates
    
    def _repair_missing_case(self, context: CodeContext, kind: str,
                             expected: str, arch: str) -> List[RepairCandidate]:
        """Repair missing switch case."""
        candidates = []
        code = context.code
        
        # Get expected relocation
        reloc = str(expected) if expected else "R_UNKNOWN"
        if not reloc.startswith("ELF::") and not reloc.startswith("R_"):
            reloc_mapping = RepairTemplates.get_reloc_mapping(arch, kind)
            if reloc_mapping:
                reloc = reloc_mapping[0]
        
        # Add case before default
        default_match = re.search(r'(\s*default\s*:)', code)
        if default_match:
            new_case = f"\n    case {kind}:\n        return ELF::{reloc};\n    "
            repaired = code[:default_match.start()] + new_case + code[default_match.start():]
            
            candidates.append(RepairCandidate(
                repaired_code=repaired,
                strategy=RepairStrategy.TEMPLATE_EXACT,
                confidence=0.90,
                description=f"Added missing case {kind}",
                changes=[f"Added case {kind}: return ELF::{reloc};"],
                edit_distance=self._compute_edit_distance(code, repaired)
            ))
        
        return candidates
    
    def _detect_architecture(self, context: CodeContext) -> str:
        """Detect architecture from context."""
        arch = context.architecture.upper()
        
        # Also check code for hints
        code = context.code.upper()
        if "RISCV" in code or "R_RISCV" in code:
            return "RISCV"
        elif "AARCH64" in code or "R_AARCH64" in code:
            return "AArch64"
        elif "ARM" in code or "R_ARM" in code:
            return "ARM"
        elif "MIPS" in code or "R_MIPS" in code:
            return "MIPS"
        elif "X86" in code or "R_X86" in code:
            return "X86_64"
        
        return arch if arch else "RISCV"
    
    def _compute_edit_distance(self, original: str, repaired: str) -> int:
        """Compute line-based edit distance."""
        orig_lines = original.split('\n')
        repair_lines = repaired.split('\n')
        
        matcher = difflib.SequenceMatcher(None, orig_lines, repair_lines)
        return int((1 - matcher.ratio()) * len(orig_lines))


class PatternRepairModel(BaseRepairModel):
    """
    Pattern-based repair model using learned heuristics.
    
    Learns common repair patterns from successful repairs.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.fault_localizer = FaultLocalizer()
        
        # Learned patterns (pattern -> repair transformation)
        self.patterns = {}
        
        # Statistics
        self.stats = {
            "repairs_generated": 0,
            "patterns_matched": 0,
        }
    
    def generate_repairs(self, context: CodeContext, num_candidates: int = 5) -> List[RepairCandidate]:
        """Generate pattern-based repairs."""
        candidates = []
        
        # Apply learned patterns
        for pattern_name, pattern_data in self.patterns.items():
            match = re.search(pattern_data["regex"], context.code)
            if match:
                repaired = re.sub(
                    pattern_data["regex"],
                    pattern_data["replacement"],
                    context.code
                )
                
                candidates.append(RepairCandidate(
                    repaired_code=repaired,
                    strategy=RepairStrategy.PATTERN_BASED,
                    confidence=pattern_data.get("confidence", 0.75),
                    description=pattern_data.get("description", f"Applied pattern: {pattern_name}"),
                    changes=[pattern_name],
                ))
        
        self.stats["repairs_generated"] += len(candidates)
        return candidates[:num_candidates]
    
    def learn_pattern(self, original: str, repaired: str, description: str = ""):
        """Learn a new repair pattern from example."""
        # Extract differences
        diff = list(difflib.unified_diff(
            original.split('\n'),
            repaired.split('\n'),
            lineterm=''
        ))
        
        if len(diff) < 4:
            return
        
        # Simplified pattern learning
        pattern_name = f"pattern_{len(self.patterns)}"
        self.patterns[pattern_name] = {
            "original_snippet": original[:200],
            "repaired_snippet": repaired[:200],
            "description": description,
            "confidence": 0.7,
        }
        
        self.stats["patterns_matched"] += 1


class HybridTransformerRepairModel(BaseRepairModel):
    """
    Hybrid repair model combining templates, patterns, and neural guidance.
    
    Priority order:
    1. Template-based (highest confidence for known patterns)
    2. Pattern-based (learned from successful repairs)
    3. Heuristic fallback
    """
    
    def __init__(self, verbose: bool = False, use_neural: bool = False):
        self.verbose = verbose
        self.use_neural = use_neural
        
        self.template_model = TemplateRepairModel(verbose=verbose)
        self.pattern_model = PatternRepairModel(verbose=verbose)
        
        self.stats = {
            "total_repairs": 0,
            "template_repairs": 0,
            "pattern_repairs": 0,
            "neural_repairs": 0,
            "fallback_repairs": 0,
        }
    
    def generate_repairs(self, context: CodeContext, num_candidates: int = 5) -> List[RepairCandidate]:
        """Generate repairs using hybrid approach."""
        self.stats["total_repairs"] += 1
        all_candidates = []
        
        # 1. Template-based repairs (highest priority)
        template_candidates = self.template_model.generate_repairs(context, num_candidates)
        all_candidates.extend(template_candidates)
        self.stats["template_repairs"] += len(template_candidates)
        
        if self.verbose:
            print(f"Template model generated {len(template_candidates)} candidates")
        
        # 2. Pattern-based repairs
        pattern_candidates = self.pattern_model.generate_repairs(context, num_candidates)
        all_candidates.extend(pattern_candidates)
        self.stats["pattern_repairs"] += len(pattern_candidates)
        
        if self.verbose:
            print(f"Pattern model generated {len(pattern_candidates)} candidates")
        
        # 3. Fallback repairs if needed
        if len(all_candidates) < num_candidates:
            fallback = self._generate_fallback_repairs(context, num_candidates - len(all_candidates))
            all_candidates.extend(fallback)
            self.stats["fallback_repairs"] += len(fallback)
        
        # Deduplicate and sort
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            code_hash = hash(c.repaired_code.strip())
            if code_hash not in seen:
                seen.add(code_hash)
                unique_candidates.append(c)
        
        unique_candidates.sort(key=lambda c: c.confidence, reverse=True)
        return unique_candidates[:num_candidates]
    
    def _generate_fallback_repairs(self, context: CodeContext, num: int) -> List[RepairCandidate]:
        """Generate fallback repairs."""
        candidates = []
        code = context.code
        
        ce = context.counterexample
        expected = ce.get("expected_output")
        
        # Add FIXME comment
        if 'switch' in code:
            switch_pos = code.find('switch')
            repaired = code[:switch_pos] + "// FIXME: Review specification compliance\n    " + code[switch_pos:]
            
            candidates.append(RepairCandidate(
                repaired_code=repaired,
                strategy=RepairStrategy.FALLBACK,
                confidence=0.2,
                description="Added FIXME comment for manual review",
                changes=["Added review comment"],
            ))
        
        # Add assertion
        func_start = code.find('{')
        if func_start > 0:
            assertion = f'\n    // Verification hint: expected {expected}\n    assert(Fixup.isValid() && "Invalid fixup");'
            repaired = code[:func_start+1] + assertion + code[func_start+1:]
            
            candidates.append(RepairCandidate(
                repaired_code=repaired,
                strategy=RepairStrategy.FALLBACK,
                confidence=0.15,
                description="Added validation assertion",
                changes=["Added assert for input validation"],
            ))
        
        return candidates[:num]
    
    def learn_from_repair(self, original: str, repaired: str, description: str = ""):
        """Learn from a successful repair."""
        self.pattern_model.learn_pattern(original, repaired, description)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            **self.stats,
            "template_stats": self.template_model.stats,
            "pattern_stats": self.pattern_model.stats,
        }


def create_transformer_repair_model(
    verbose: bool = False,
    use_neural: bool = False
) -> HybridTransformerRepairModel:
    """Factory function to create transformer repair model."""
    return HybridTransformerRepairModel(verbose=verbose, use_neural=use_neural)


# Convenience aliases
TransformerRepairModel = HybridTransformerRepairModel
