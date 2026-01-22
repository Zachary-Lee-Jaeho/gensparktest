"""
Switch Statement Repair Module for VEGA-Verified.

This module implements automatic repair for switch statement errors
detected by the SwitchVerifier in Phase 2.

Key Repair Capabilities:
1. Fix incorrect case-return mappings
2. Add missing switch cases
3. Handle fall-through case errors
4. Fix ternary condition issues
5. Generate complete switch statements from specifications

Integration with Phase 2:
- Takes VerificationResult from switch_verifier.py
- Uses RISCV_RELOC_MAPPINGS as ground truth
- Generates repair candidates with confidence scores
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import re
import difflib


class SwitchRepairType(Enum):
    """Types of switch statement repairs."""
    FIX_RETURN_VALUE = "fix_return_value"
    ADD_MISSING_CASE = "add_missing_case"
    FIX_FALLTHROUGH = "fix_fallthrough"
    FIX_TERNARY = "fix_ternary"
    ADD_DEFAULT = "add_default"
    REMOVE_DUPLICATE = "remove_duplicate"
    REORDER_CASES = "reorder_cases"


@dataclass
class SwitchRepairCandidate:
    """A candidate repair for a switch statement."""
    repair_type: SwitchRepairType
    original_code: str
    repaired_code: str
    confidence: float
    description: str
    affected_cases: List[str] = field(default_factory=list)
    diff: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'repair_type': self.repair_type.value,
            'confidence': self.confidence,
            'description': self.description,
            'affected_cases': self.affected_cases,
            'diff': self.diff,
        }
    
    def compute_diff(self):
        """Compute unified diff between original and repaired code."""
        original_lines = self.original_code.splitlines(keepends=True)
        repaired_lines = self.repaired_code.splitlines(keepends=True)
        diff_lines = difflib.unified_diff(
            original_lines, repaired_lines,
            fromfile='original', tofile='repaired'
        )
        self.diff = ''.join(diff_lines)


@dataclass
class SwitchRepairResult:
    """Result of switch statement repair."""
    success: bool
    candidates: List[SwitchRepairCandidate]
    best_repair: Optional[SwitchRepairCandidate]
    verification_passed: bool = False
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'num_candidates': len(self.candidates),
            'best_repair': self.best_repair.to_dict() if self.best_repair else None,
            'verification_passed': self.verification_passed,
            'message': self.message,
        }


class SwitchRepairModel:
    """
    Automatic repair model for switch statement errors.
    
    Works with:
    - VerificationResult from switch_verifier.py
    - Ground truth mappings (RISCV_RELOC_MAPPINGS)
    - Counterexamples from Z3 coverage verification
    """
    
    # Ground truth mappings for RISCV (from switch_verifier.py)
    RISCV_GROUND_TRUTH = {
        'fixup_riscv_hi20': 'R_RISCV_HI20',
        'fixup_riscv_lo12_i': 'R_RISCV_LO12_I',
        'fixup_riscv_lo12_s': 'R_RISCV_LO12_S',
        'fixup_riscv_pcrel_hi20': 'R_RISCV_PCREL_HI20',
        'fixup_riscv_pcrel_lo12_i': 'R_RISCV_PCREL_LO12_I',
        'fixup_riscv_pcrel_lo12_s': 'R_RISCV_PCREL_LO12_S',
        'fixup_riscv_got_hi20': 'R_RISCV_GOT_HI20',
        'fixup_riscv_tls_got_hi20': 'R_RISCV_TLS_GOT_HI20',
        'fixup_riscv_tls_gd_hi20': 'R_RISCV_TLS_GD_HI20',
        'fixup_riscv_tlsdesc_hi20': 'R_RISCV_TLSDESC_HI20',
        'fixup_riscv_tlsdesc_load_lo12': 'R_RISCV_TLSDESC_LOAD_LO12',
        'fixup_riscv_tlsdesc_add_lo12': 'R_RISCV_TLSDESC_ADD_LO12',
        'fixup_riscv_tlsdesc_call': 'R_RISCV_TLSDESC_CALL',
        'fixup_riscv_jal': 'R_RISCV_JAL',
        'fixup_riscv_branch': 'R_RISCV_BRANCH',
        'fixup_riscv_rvc_jump': 'R_RISCV_RVC_JUMP',
        'fixup_riscv_rvc_branch': 'R_RISCV_RVC_BRANCH',
        'fixup_riscv_call': 'R_RISCV_CALL_PLT',
        'fixup_riscv_call_plt': 'R_RISCV_CALL_PLT',
        'fixup_riscv_relax': 'R_RISCV_RELAX',
        'fixup_riscv_align': 'R_RISCV_ALIGN',
        'fixup_riscv_tprel_hi20': 'R_RISCV_TPREL_HI20',
        'fixup_riscv_tprel_lo12_i': 'R_RISCV_TPREL_LO12_I',
        'fixup_riscv_tprel_lo12_s': 'R_RISCV_TPREL_LO12_S',
        'fixup_riscv_tprel_add': 'R_RISCV_TPREL_ADD',
        'FK_Data_1': 'R_RISCV_NONE',  # Error case
        'FK_Data_2': 'R_RISCV_NONE',  # Error case
        'FK_Data_4': 'R_RISCV_32',
        'FK_Data_8': 'R_RISCV_64',
        'FK_PCRel_4': 'R_RISCV_32_PCREL',
    }
    
    # Template for new case
    CASE_TEMPLATE = """    case {namespace}{case_name}:
      return {return_namespace}{return_value};"""
    
    # Template for error case
    ERROR_CASE_TEMPLATE = """    case {namespace}{case_name}:
      Ctx.reportError(Fixup.getLoc(), "{error_message}");
      return {return_namespace}{return_value};"""
    
    def __init__(self, backend: str = "RISCV", verbose: bool = False):
        self.backend = backend
        self.verbose = verbose
        self.ground_truth = self.RISCV_GROUND_TRUTH if backend == "RISCV" else {}
    
    def repair_incorrect_mapping(
        self,
        code: str,
        case_name: str,
        actual_return: str,
        expected_return: str
    ) -> SwitchRepairCandidate:
        """
        Fix an incorrect case-return mapping.
        
        Example:
            case fixup_riscv_hi20:
                return R_RISCV_LO12_I;  // WRONG
            ->
            case fixup_riscv_hi20:
                return R_RISCV_HI20;    // CORRECT
        """
        # Find the case in code
        pattern = re.compile(
            rf'(case\s+[\w:]*{re.escape(case_name)}:\s*\n\s*return\s+)([^;]+)(;)',
            re.MULTILINE
        )
        
        def replace_return(match):
            prefix = match.group(1)
            old_return = match.group(2)
            suffix = match.group(3)
            
            # Preserve namespace if present
            if '::' in old_return and '::' not in expected_return:
                namespace = old_return.rsplit('::', 1)[0] + '::'
                new_return = namespace + expected_return
            else:
                new_return = expected_return
            
            return f"{prefix}{new_return}{suffix}"
        
        repaired_code = pattern.sub(replace_return, code)
        
        candidate = SwitchRepairCandidate(
            repair_type=SwitchRepairType.FIX_RETURN_VALUE,
            original_code=code,
            repaired_code=repaired_code,
            confidence=0.95,  # High confidence for direct mapping fix
            description=f"Fix return value: {case_name} -> {expected_return} (was {actual_return})",
            affected_cases=[case_name]
        )
        candidate.compute_diff()
        
        return candidate
    
    def add_missing_case(
        self,
        code: str,
        case_name: str,
        return_value: Optional[str] = None,
        is_error_case: bool = False,
        error_message: str = ""
    ) -> SwitchRepairCandidate:
        """
        Add a missing case to the switch statement.
        
        Example:
            switch (Kind) {
                case A: return X;
                case B: return Y;
            }
            ->
            switch (Kind) {
                case A: return X;
                case B: return Y;
                case C: return Z;  // NEW
            }
        """
        # Determine return value from ground truth if not provided
        if return_value is None:
            normalized = case_name.split('::')[-1] if '::' in case_name else case_name
            return_value = self.ground_truth.get(normalized, 'R_RISCV_NONE')
        
        # Determine namespace based on existing code
        namespace = ""
        return_namespace = ""
        
        if "RISCV::" in code:
            namespace = "RISCV::"
        if "ELF::" in code:
            return_namespace = "ELF::"
        
        # Find insertion point (before default or at end of switch)
        default_pattern = re.compile(r'(\s*)(default:)', re.MULTILINE)
        default_match = default_pattern.search(code)
        
        if is_error_case:
            new_case = self.ERROR_CASE_TEMPLATE.format(
                namespace=namespace,
                case_name=case_name,
                error_message=error_message or f"{case_name} not supported",
                return_namespace=return_namespace,
                return_value=return_value
            )
        else:
            new_case = self.CASE_TEMPLATE.format(
                namespace=namespace,
                case_name=case_name,
                return_namespace=return_namespace,
                return_value=return_value
            )
        
        if default_match:
            # Insert before default
            indent = default_match.group(1)
            insertion_point = default_match.start()
            repaired_code = code[:insertion_point] + new_case + "\n" + code[insertion_point:]
        else:
            # Find the last case and insert after it
            last_case_pattern = re.compile(r'(case\s+[\w:]+:.*?return\s+[^;]+;)', re.DOTALL)
            matches = list(last_case_pattern.finditer(code))
            if matches:
                last_match = matches[-1]
                insertion_point = last_match.end()
                repaired_code = code[:insertion_point] + "\n" + new_case + code[insertion_point:]
            else:
                # Fallback: insert at end of switch
                repaired_code = code.rstrip('}') + new_case + "\n}"
        
        candidate = SwitchRepairCandidate(
            repair_type=SwitchRepairType.ADD_MISSING_CASE,
            original_code=code,
            repaired_code=repaired_code,
            confidence=0.85,
            description=f"Add missing case: {case_name} -> {return_value}",
            affected_cases=[case_name]
        )
        candidate.compute_diff()
        
        return candidate
    
    def fix_fallthrough_error(
        self,
        code: str,
        case_name: str,
        should_fallthrough: bool,
        target_case: Optional[str] = None
    ) -> SwitchRepairCandidate:
        """
        Fix fall-through case errors.
        
        If should_fallthrough=True: Remove return statement to enable fall-through
        If should_fallthrough=False: Add return statement to prevent fall-through
        """
        if should_fallthrough and target_case:
            # Convert to fall-through
            pattern = re.compile(
                rf'(case\s+[\w:]*{re.escape(case_name)}:\s*)\n\s*return\s+[^;]+;',
                re.MULTILINE
            )
            repaired_code = pattern.sub(r'\1', code)
            description = f"Enable fall-through: {case_name} -> {target_case}"
        else:
            # Add explicit return
            return_value = self.ground_truth.get(case_name.split('::')[-1], 'R_RISCV_NONE')
            pattern = re.compile(
                rf'(case\s+[\w:]*{re.escape(case_name)}:)\s*\n\s*(case)',
                re.MULTILINE
            )
            namespace = "ELF::" if "ELF::" in code else ""
            replacement = rf'\1\n      return {namespace}{return_value};\n    \2'
            repaired_code = pattern.sub(replacement, code)
            description = f"Prevent fall-through: {case_name} -> explicit return"
        
        candidate = SwitchRepairCandidate(
            repair_type=SwitchRepairType.FIX_FALLTHROUGH,
            original_code=code,
            repaired_code=repaired_code,
            confidence=0.80,
            description=description,
            affected_cases=[case_name]
        )
        candidate.compute_diff()
        
        return candidate
    
    def repair_from_verification_result(
        self,
        code: str,
        verification_result: Dict[str, Any]
    ) -> SwitchRepairResult:
        """
        Generate repairs based on VerificationResult from switch_verifier.
        
        Args:
            code: Original function code
            verification_result: Result dict from SwitchVerifier.verify_function().to_dict()
            
        Returns:
            SwitchRepairResult with all candidate repairs
        """
        candidates = []
        
        # 1. Fix incorrect mappings
        for incorrect in verification_result.get('incorrect_mappings', []):
            case = incorrect.get('case', '')
            expected = incorrect.get('expected', '')
            actual = incorrect.get('actual', '')
            
            if isinstance(expected, list):
                expected = expected[0]  # Use first expected value
            if isinstance(actual, list):
                actual = actual[0]
            
            candidate = self.repair_incorrect_mapping(
                code, case, actual, expected
            )
            candidates.append(candidate)
            code = candidate.repaired_code  # Apply repair for chaining
        
        # 2. Add missing cases
        for missing in verification_result.get('missing_cases', []):
            candidate = self.add_missing_case(code, missing)
            candidates.append(candidate)
            code = candidate.repaired_code
        
        # 3. Handle duplicates (if any)
        for duplicate in verification_result.get('duplicate_cases', []):
            # Remove duplicate - keep first occurrence
            pattern = re.compile(
                rf'(case\s+[\w:]*{re.escape(duplicate)}:.*?return[^;]+;.*?)(case\s+[\w:]*{re.escape(duplicate)}:.*?return[^;]+;)',
                re.DOTALL
            )
            repaired_code = pattern.sub(r'\1', code)
            
            if repaired_code != code:
                candidate = SwitchRepairCandidate(
                    repair_type=SwitchRepairType.REMOVE_DUPLICATE,
                    original_code=code,
                    repaired_code=repaired_code,
                    confidence=0.90,
                    description=f"Remove duplicate case: {duplicate}",
                    affected_cases=[duplicate]
                )
                candidate.compute_diff()
                candidates.append(candidate)
                code = repaired_code
        
        # Determine best repair
        best_repair = None
        if candidates:
            # Sort by confidence
            candidates.sort(key=lambda c: c.confidence, reverse=True)
            best_repair = candidates[0]
        
        return SwitchRepairResult(
            success=len(candidates) > 0,
            candidates=candidates,
            best_repair=best_repair,
            verification_passed=False,  # Would need re-verification
            message=f"Generated {len(candidates)} repair candidate(s)"
        )
    
    def generate_complete_switch(
        self,
        switch_variable: str = "Kind",
        cases: Optional[Dict[str, str]] = None,
        include_default: bool = True,
        backend: str = "RISCV"
    ) -> str:
        """
        Generate a complete switch statement from scratch.
        
        Useful for:
        - Generating reference implementations
        - Creating test fixtures
        - Bootstrapping new backends
        """
        if cases is None:
            cases = self.ground_truth
        
        lines = [f"switch ({switch_variable}) {{"]
        
        namespace = "RISCV::" if backend == "RISCV" else ""
        return_namespace = "ELF::"
        
        for case_name, return_value in sorted(cases.items()):
            if case_name.startswith('FK_'):
                # Generic fixup - no namespace
                case_line = f"  case {case_name}:"
            else:
                case_line = f"  case {namespace}{case_name}:"
            
            lines.append(case_line)
            lines.append(f"    return {return_namespace}{return_value};")
        
        if include_default:
            lines.append("  default:")
            lines.append('    llvm_unreachable("Unknown fixup kind!");')
        
        lines.append("}")
        
        return "\n".join(lines)


def create_switch_repair_model(backend: str = "RISCV") -> SwitchRepairModel:
    """Factory function to create a SwitchRepairModel."""
    return SwitchRepairModel(backend=backend)


# Demo / Test
if __name__ == '__main__':
    print("=" * 70)
    print("Switch Repair Model Demo")
    print("=" * 70)
    
    # Create model
    model = SwitchRepairModel(backend="RISCV")
    
    # Test 1: Fix incorrect mapping
    print("\n📝 Test 1: Fix incorrect mapping")
    test_code = """
    switch (Kind) {
        case RISCV::fixup_riscv_hi20:
            return ELF::R_RISCV_LO12_I;  // WRONG!
        case RISCV::fixup_riscv_lo12_i:
            return ELF::R_RISCV_LO12_I;
    }
    """
    
    candidate = model.repair_incorrect_mapping(
        test_code,
        case_name="fixup_riscv_hi20",
        actual_return="R_RISCV_LO12_I",
        expected_return="R_RISCV_HI20"
    )
    
    print(f"Repair type: {candidate.repair_type.value}")
    print(f"Confidence: {candidate.confidence}")
    print(f"Description: {candidate.description}")
    print(f"\nDiff:\n{candidate.diff}")
    
    # Test 2: Add missing case
    print("\n📝 Test 2: Add missing case")
    test_code2 = """
    switch (Kind) {
        case RISCV::fixup_riscv_hi20:
            return ELF::R_RISCV_HI20;
        default:
            llvm_unreachable("Unknown fixup!");
    }
    """
    
    candidate2 = model.add_missing_case(
        test_code2,
        case_name="fixup_riscv_lo12_i",
        return_value="R_RISCV_LO12_I"
    )
    
    print(f"Repair type: {candidate2.repair_type.value}")
    print(f"Confidence: {candidate2.confidence}")
    print(f"Description: {candidate2.description}")
    print(f"\nDiff:\n{candidate2.diff}")
    
    # Test 3: Repair from verification result
    print("\n📝 Test 3: Repair from verification result")
    mock_verification_result = {
        'incorrect_mappings': [
            {'case': 'fixup_riscv_branch', 'expected': 'R_RISCV_BRANCH', 'actual': 'R_RISCV_JAL'}
        ],
        'missing_cases': ['fixup_riscv_call'],
        'duplicate_cases': []
    }
    
    test_code3 = """
    switch (Kind) {
        case RISCV::fixup_riscv_hi20:
            return ELF::R_RISCV_HI20;
        case RISCV::fixup_riscv_branch:
            return ELF::R_RISCV_JAL;
        default:
            llvm_unreachable("Unknown fixup!");
    }
    """
    
    result = model.repair_from_verification_result(test_code3, mock_verification_result)
    
    print(f"Success: {result.success}")
    print(f"Candidates: {len(result.candidates)}")
    if result.best_repair:
        print(f"Best repair: {result.best_repair.description}")
    
    # Test 4: Generate complete switch
    print("\n📝 Test 4: Generate complete switch (first 20 lines)")
    complete_switch = model.generate_complete_switch(
        switch_variable="Kind",
        cases={
            'fixup_riscv_hi20': 'R_RISCV_HI20',
            'fixup_riscv_lo12_i': 'R_RISCV_LO12_I',
            'fixup_riscv_branch': 'R_RISCV_BRANCH',
        }
    )
    print(complete_switch)
    
    print("\n" + "=" * 70)
    print("✅ Switch Repair Model Demo Complete")
    print("=" * 70)
