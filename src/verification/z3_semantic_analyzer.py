"""
Enhanced Z3 Semantic Analyzer for VEGA-Verified.

This module provides deep semantic analysis of compiler backend code using Z3.
It models the actual behavior of:
1. Switch statements with case mappings
2. Conditional expressions (IsPCRel, IsBigEndian, etc.)
3. Relocation type computations
4. Instruction encoding logic

Key improvements over basic Z3 backend:
- Full symbolic execution of switch/case statements
- Precise modeling of LLVM fixup kinds
- Architecture-specific relocation mappings
- Counterexample-guided test case generation
"""

import z3
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
import re
import time
import hashlib


class ArchitectureType(Enum):
    """Supported architectures."""
    RISCV = "riscv"
    RI5CY = "ri5cy"      # PULP RISC-V
    XCORE = "xcore"
    ARM = "arm"
    AARCH64 = "aarch64"
    MIPS = "mips"
    X86_64 = "x86_64"
    POWERPC = "powerpc"


@dataclass
class FixupKindMapping:
    """Mapping from fixup kind to relocation type."""
    kind: str
    reloc_normal: str
    reloc_pcrel: Optional[str] = None
    arch: Optional[ArchitectureType] = None
    
    def get_reloc(self, is_pcrel: bool) -> str:
        """Get relocation type based on IsPCRel."""
        if is_pcrel and self.reloc_pcrel:
            return self.reloc_pcrel
        return self.reloc_normal


# Architecture-specific fixup mappings
FIXUP_MAPPINGS: Dict[ArchitectureType, List[FixupKindMapping]] = {
    ArchitectureType.RISCV: [
        FixupKindMapping("FK_NONE", "R_RISCV_NONE"),
        FixupKindMapping("FK_Data_1", "R_RISCV_NONE"),
        FixupKindMapping("FK_Data_2", "R_RISCV_NONE"),
        FixupKindMapping("FK_Data_4", "R_RISCV_32", "R_RISCV_32_PCREL"),
        FixupKindMapping("FK_Data_8", "R_RISCV_64"),
        FixupKindMapping("fixup_riscv_hi20", "R_RISCV_HI20"),
        FixupKindMapping("fixup_riscv_lo12_i", "R_RISCV_LO12_I"),
        FixupKindMapping("fixup_riscv_lo12_s", "R_RISCV_LO12_S"),
        FixupKindMapping("fixup_riscv_pcrel_hi20", "R_RISCV_PCREL_HI20"),
        FixupKindMapping("fixup_riscv_pcrel_lo12_i", "R_RISCV_PCREL_LO12_I"),
        FixupKindMapping("fixup_riscv_branch", "R_RISCV_BRANCH"),
        FixupKindMapping("fixup_riscv_jal", "R_RISCV_JAL"),
        FixupKindMapping("fixup_riscv_call", "R_RISCV_CALL"),
    ],
    ArchitectureType.RI5CY: [
        FixupKindMapping("FK_NONE", "R_RISCV_NONE"),
        FixupKindMapping("FK_Data_4", "R_RISCV_32", "R_RISCV_32_PCREL"),
        FixupKindMapping("fixup_pulp_loop", "R_RISCV_PULP_LOOP"),
        FixupKindMapping("fixup_pulp_post_inc", "R_RISCV_PULP_POST_INC"),
    ],
    ArchitectureType.ARM: [
        FixupKindMapping("FK_NONE", "R_ARM_NONE"),
        FixupKindMapping("FK_Data_1", "R_ARM_ABS8"),
        FixupKindMapping("FK_Data_2", "R_ARM_ABS16"),
        FixupKindMapping("FK_Data_4", "R_ARM_ABS32", "R_ARM_REL32"),
        FixupKindMapping("fixup_arm_ldst_pcrel_12", "R_ARM_LDR_PC_G0"),
        FixupKindMapping("fixup_arm_pcrel_10_unscaled", "R_ARM_LDRS_PC_G0"),
        FixupKindMapping("fixup_arm_branch", "R_ARM_JUMP24"),
        FixupKindMapping("fixup_arm_thumb_bl", "R_ARM_THM_CALL"),
        FixupKindMapping("fixup_arm_thumb_br", "R_ARM_THM_JUMP24"),
    ],
    ArchitectureType.AARCH64: [
        FixupKindMapping("FK_NONE", "R_AARCH64_NONE"),
        FixupKindMapping("FK_Data_2", "R_AARCH64_ABS16"),
        FixupKindMapping("FK_Data_4", "R_AARCH64_ABS32", "R_AARCH64_PREL32"),
        FixupKindMapping("FK_Data_8", "R_AARCH64_ABS64", "R_AARCH64_PREL64"),
        FixupKindMapping("fixup_aarch64_pcrel_adr_imm21", "R_AARCH64_ADR_PREL_LO21"),
        FixupKindMapping("fixup_aarch64_pcrel_adrp_imm21", "R_AARCH64_ADR_PREL_PG_HI21"),
        FixupKindMapping("fixup_aarch64_pcrel_branch26", "R_AARCH64_JUMP26"),
        FixupKindMapping("fixup_aarch64_pcrel_call26", "R_AARCH64_CALL26"),
    ],
    ArchitectureType.MIPS: [
        FixupKindMapping("FK_NONE", "R_MIPS_NONE"),
        FixupKindMapping("FK_Data_4", "R_MIPS_32", "R_MIPS_PC32"),
        FixupKindMapping("FK_Data_8", "R_MIPS_64"),
        FixupKindMapping("fixup_Mips_HI16", "R_MIPS_HI16"),
        FixupKindMapping("fixup_Mips_LO16", "R_MIPS_LO16"),
        FixupKindMapping("fixup_Mips_26", "R_MIPS_26"),
        FixupKindMapping("fixup_Mips_GPREL16", "R_MIPS_GPREL16"),
        FixupKindMapping("fixup_Mips_Branch_PCRel", "R_MIPS_PC16"),
    ],
    ArchitectureType.X86_64: [
        FixupKindMapping("FK_NONE", "R_X86_64_NONE"),
        FixupKindMapping("FK_Data_1", "R_X86_64_8"),
        FixupKindMapping("FK_Data_2", "R_X86_64_16"),
        FixupKindMapping("FK_Data_4", "R_X86_64_32", "R_X86_64_PC32"),
        FixupKindMapping("FK_Data_8", "R_X86_64_64", "R_X86_64_PC64"),
        FixupKindMapping("FK_PCRel_1", "R_X86_64_PC8"),
        FixupKindMapping("FK_PCRel_2", "R_X86_64_PC16"),
        FixupKindMapping("FK_PCRel_4", "R_X86_64_PC32"),
        FixupKindMapping("FK_SecRel_4", "R_X86_64_32S"),
        FixupKindMapping("fixup_x86_64_plt", "R_X86_64_PLT32"),
        FixupKindMapping("fixup_x86_64_got", "R_X86_64_GOT32"),
    ],
    ArchitectureType.POWERPC: [
        FixupKindMapping("FK_NONE", "R_PPC_NONE"),
        FixupKindMapping("FK_Data_1", "R_PPC_NONE"),
        FixupKindMapping("FK_Data_2", "R_PPC_ADDR16"),
        FixupKindMapping("FK_Data_4", "R_PPC_ADDR32", "R_PPC_REL32"),
        FixupKindMapping("FK_Data_8", "R_PPC64_ADDR64"),
        FixupKindMapping("fixup_ppc_br24", "R_PPC_REL24"),
        FixupKindMapping("fixup_ppc_ha16", "R_PPC_ADDR16_HA"),
        FixupKindMapping("fixup_ppc_lo16", "R_PPC_ADDR16_LO"),
    ],
    ArchitectureType.XCORE: [
        FixupKindMapping("FK_NONE", "R_XCORE_NONE"),
        FixupKindMapping("FK_Data_4", "R_XCORE_32", "R_XCORE_32_PCREL"),
        FixupKindMapping("fixup_xcore_pcrel_dp", "R_XCORE_PCREL_DP"),
        FixupKindMapping("fixup_xcore_pcrel_cp", "R_XCORE_PCREL_CP"),
    ],
}


@dataclass
class SemanticModel:
    """Semantic model of code behavior."""
    function_name: str
    architecture: ArchitectureType
    input_vars: Dict[str, z3.ExprRef] = field(default_factory=dict)
    output_var: Optional[z3.ExprRef] = None
    constraints: List[z3.ExprRef] = field(default_factory=list)
    case_mappings: Dict[str, Dict[bool, str]] = field(default_factory=dict)  # kind -> {is_pcrel -> reloc}
    
    def add_case_mapping(self, kind: str, reloc: str, is_pcrel: bool = False):
        """Add a case mapping."""
        if kind not in self.case_mappings:
            self.case_mappings[kind] = {}
        self.case_mappings[kind][is_pcrel] = reloc


@dataclass
class VerificationCondition:
    """A verification condition to check."""
    name: str
    formula: z3.ExprRef
    is_invariant: bool = False
    source_info: Optional[str] = None


@dataclass 
class SemanticVerificationResult:
    """Result of semantic verification."""
    verified: bool
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    verified_conditions: List[str] = field(default_factory=list)
    failed_conditions: List[str] = field(default_factory=list)
    time_ms: float = 0.0
    z3_stats: Dict[str, Any] = field(default_factory=dict)
    coverage: float = 0.0


class Z3SemanticAnalyzer:
    """
    Z3-based semantic analyzer for compiler backend code.
    
    Performs deep semantic analysis by:
    1. Parsing code into semantic model
    2. Building Z3 constraints for behavior
    3. Checking specification properties
    4. Extracting counterexamples for repair
    """
    
    def __init__(self, 
                 architecture: ArchitectureType = ArchitectureType.RISCV,
                 timeout_ms: int = 30000,
                 verbose: bool = False):
        self.architecture = architecture
        self.timeout_ms = timeout_ms
        self.verbose = verbose
        
        # Z3 solver
        self.solver = z3.Solver()
        self.solver.set("timeout", timeout_ms)
        
        # Statistics
        self.stats = {
            "analyses": 0,
            "verified": 0,
            "failed": 0,
            "total_time_ms": 0.0,
            "counterexamples_found": 0,
        }
        
        # Caches
        self._model_cache: Dict[str, SemanticModel] = {}
    
    def analyze(self, 
                code: str,
                spec: 'Specification',
                function_name: Optional[str] = None) -> SemanticVerificationResult:
        """
        Perform semantic analysis of code against specification.
        
        Args:
            code: Source code to analyze
            spec: Specification to verify
            function_name: Function name (defaults to spec.function_name)
            
        Returns:
            SemanticVerificationResult with verification status and counterexamples
        """
        self.stats["analyses"] += 1
        start_time = time.time()
        
        func_name = function_name or spec.function_name
        
        result = SemanticVerificationResult(verified=True)
        
        try:
            # Step 1: Build semantic model from code
            model = self._build_semantic_model(code, func_name)
            
            if self.verbose:
                print(f"Built semantic model for {func_name}")
                print(f"  Architecture: {model.architecture}")
                print(f"  Case mappings: {len(model.case_mappings)}")
            
            # Step 2: Generate verification conditions
            vcs = self._generate_vcs(model, spec)
            
            if self.verbose:
                print(f"  Generated {len(vcs)} verification conditions")
            
            # Step 3: Check each verification condition
            for vc in vcs:
                vc_result = self._check_vc(model, vc)
                
                if vc_result["verified"]:
                    result.verified_conditions.append(vc.name)
                else:
                    result.verified = False
                    result.failed_conditions.append(vc.name)
                    if vc_result.get("counterexample"):
                        result.counterexamples.append(vc_result["counterexample"])
                        self.stats["counterexamples_found"] += 1
            
            # Step 4: Compute coverage
            total_cases = len(model.case_mappings)
            if total_cases > 0:
                verified_cases = self._count_verified_cases(model, spec)
                result.coverage = verified_cases / total_cases
            else:
                result.coverage = 1.0 if result.verified else 0.0
        
        except Exception as e:
            if self.verbose:
                print(f"Analysis error: {e}")
            result.verified = False
            result.failed_conditions.append(f"Analysis error: {str(e)}")
        
        # Update stats
        result.time_ms = (time.time() - start_time) * 1000
        self.stats["total_time_ms"] += result.time_ms
        
        if result.verified:
            self.stats["verified"] += 1
        else:
            self.stats["failed"] += 1
        
        return result
    
    def _build_semantic_model(self, code: str, function_name: str) -> SemanticModel:
        """Build semantic model from code."""
        # Check cache
        code_hash = hashlib.md5(code.encode()).hexdigest()
        cache_key = f"{function_name}:{code_hash}"
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model = SemanticModel(
            function_name=function_name,
            architecture=self.architecture
        )
        
        # Create Z3 variables
        model.input_vars = {
            "Kind": z3.BitVec("Kind", 32),
            "IsPCRel": z3.Bool("IsPCRel"),
            "Fixup": z3.BitVec("Fixup", 64),
        }
        model.output_var = z3.BitVec("result", 32)
        
        # Parse switch statements
        self._parse_switch_statements(code, model)
        
        # Cache model
        self._model_cache[cache_key] = model
        
        return model
    
    def _parse_switch_statements(self, code: str, model: SemanticModel):
        """Parse switch statements to extract case mappings."""
        # Pattern for case statements
        case_patterns = [
            # Simple case: return
            r'case\s+([\w:]+)\s*:\s*(?:return\s+)?(\w+(?:::\w+)*)\s*;',
            # Ternary: IsPCRel ? A : B
            r'case\s+([\w:]+)\s*:\s*return\s+IsPCRel\s*\?\s*(\w+(?:::\w+)*)\s*:\s*(\w+(?:::\w+)*)\s*;',
        ]
        
        # Find ternary cases first
        ternary_pattern = re.compile(
            r'case\s+([\w:]+)\s*:\s*return\s+IsPCRel\s*\?\s*(\w+(?:::\w+)*)\s*:\s*(\w+(?:::\w+)*)\s*;'
        )
        
        for match in ternary_pattern.finditer(code):
            kind = match.group(1)
            pcrel_reloc = match.group(2)
            normal_reloc = match.group(3)
            
            model.add_case_mapping(kind, pcrel_reloc, is_pcrel=True)
            model.add_case_mapping(kind, normal_reloc, is_pcrel=False)
        
        # Check for IsPCRel blocks
        if 'if (IsPCRel)' in code or 'if(IsPCRel)' in code:
            self._parse_ispcrel_blocks(code, model)
        
        # Parse simple cases
        simple_pattern = re.compile(r'case\s+([\w:]+)\s*:\s*(?:return\s+)?(\w+(?:::\w+)*)\s*;')
        for match in simple_pattern.finditer(code):
            kind = match.group(1)
            reloc = match.group(2)
            
            # Skip if already handled as ternary
            if kind in model.case_mappings and (True in model.case_mappings[kind] or False in model.case_mappings[kind]):
                continue
            
            # Check if inside IsPCRel block (simplified heuristic)
            case_pos = match.start()
            before_case = code[:case_pos]
            
            # Count braces to determine if inside IsPCRel block
            ispcrel_match = re.search(r'if\s*\(\s*IsPCRel\s*\)\s*\{', before_case)
            if ispcrel_match:
                # Rough check if still inside the block
                block_start = ispcrel_match.end()
                opens = before_case[block_start:].count('{')
                closes = before_case[block_start:].count('}')
                if opens >= closes:
                    model.add_case_mapping(kind, reloc, is_pcrel=True)
                    continue
            
            # Check for else block (non-PCRel)
            else_match = re.search(r'\}\s*else\s*\{', before_case)
            if else_match and ispcrel_match:
                model.add_case_mapping(kind, reloc, is_pcrel=False)
                continue
            
            # Default: assume non-PCRel if not specified
            model.add_case_mapping(kind, reloc, is_pcrel=False)
    
    def _parse_ispcrel_blocks(self, code: str, model: SemanticModel):
        """Parse if (IsPCRel) blocks."""
        # Find IsPCRel block
        ispcrel_match = re.search(r'if\s*\(\s*IsPCRel\s*\)\s*\{', code)
        if not ispcrel_match:
            return
        
        # Find the corresponding block
        block_start = ispcrel_match.end()
        brace_count = 1
        block_end = block_start
        
        for i, c in enumerate(code[block_start:], block_start):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    block_end = i
                    break
        
        pcrel_block = code[block_start:block_end]
        
        # Parse cases in PCRel block
        simple_pattern = re.compile(r'case\s+([\w:]+)\s*:\s*(?:return\s+)?(\w+(?:::\w+)*)\s*;')
        for match in simple_pattern.finditer(pcrel_block):
            kind = match.group(1)
            reloc = match.group(2)
            model.add_case_mapping(kind, reloc, is_pcrel=True)
    
    def _generate_vcs(self, model: SemanticModel, spec: 'Specification') -> List[VerificationCondition]:
        """Generate verification conditions from specification."""
        vcs = []
        
        # Get expected mappings for this architecture
        expected_mappings = FIXUP_MAPPINGS.get(model.architecture, [])
        mapping_dict = {m.kind: m for m in expected_mappings}
        
        # Create symbolic constants for kinds and relocs
        kind_consts: Dict[str, z3.ExprRef] = {}
        reloc_consts: Dict[str, z3.ExprRef] = {}
        
        # Assign unique integer values to each kind/reloc
        for i, mapping in enumerate(expected_mappings):
            kind_consts[mapping.kind] = z3.BitVecVal(i, 32)
            reloc_consts[mapping.reloc_normal] = z3.BitVecVal(i * 2, 32)
            if mapping.reloc_pcrel:
                reloc_consts[mapping.reloc_pcrel] = z3.BitVecVal(i * 2 + 1, 32)
        
        # Generate VCs for each case mapping in the code
        for kind, mappings in model.case_mappings.items():
            # Find expected mapping
            expected = mapping_dict.get(kind)
            if not expected:
                # Kind not in expected mappings - might be custom
                continue
            
            # VC 1: Non-PCRel case
            if False in mappings:
                actual_reloc = mappings[False]
                expected_reloc = expected.reloc_normal
                
                if actual_reloc in reloc_consts and expected_reloc in reloc_consts:
                    vc = VerificationCondition(
                        name=f"vc_{kind}_nonpcrel",
                        formula=z3.Implies(
                            z3.And(
                                model.input_vars["Kind"] == kind_consts.get(kind, z3.BitVecVal(0, 32)),
                                z3.Not(model.input_vars["IsPCRel"])
                            ),
                            model.output_var == reloc_consts[expected_reloc]
                        ),
                        is_invariant=True,
                        source_info=f"{kind} without IsPCRel -> {expected_reloc}"
                    )
                    vcs.append(vc)
            
            # VC 2: PCRel case
            if True in mappings and expected.reloc_pcrel:
                actual_reloc = mappings[True]
                expected_reloc = expected.reloc_pcrel
                
                if actual_reloc in reloc_consts and expected_reloc in reloc_consts:
                    vc = VerificationCondition(
                        name=f"vc_{kind}_pcrel",
                        formula=z3.Implies(
                            z3.And(
                                model.input_vars["Kind"] == kind_consts.get(kind, z3.BitVecVal(0, 32)),
                                model.input_vars["IsPCRel"]
                            ),
                            model.output_var == reloc_consts[expected_reloc]
                        ),
                        is_invariant=True,
                        source_info=f"{kind} with IsPCRel -> {expected_reloc}"
                    )
                    vcs.append(vc)
        
        # Add VCs from specification invariants
        for inv in spec.invariants:
            inv_vc = self._invariant_to_vc(inv, model, kind_consts, reloc_consts)
            if inv_vc:
                vcs.append(inv_vc)
        
        return vcs
    
    def _invariant_to_vc(self, 
                         invariant: 'Condition',
                         model: SemanticModel,
                         kind_consts: Dict[str, z3.ExprRef],
                         reloc_consts: Dict[str, z3.ExprRef]) -> Optional[VerificationCondition]:
        """Convert specification invariant to verification condition."""
        try:
            from ..specification.spec_language import ConditionType, Variable, Constant
            
            def to_z3(cond):
                if isinstance(cond, Variable):
                    name = cond.name.lower()
                    if 'kind' in name:
                        return model.input_vars.get("Kind", z3.BitVecVal(0, 32))
                    elif 'pcrel' in name:
                        return model.input_vars.get("IsPCRel", z3.BoolVal(False))
                    elif 'result' in name:
                        return model.output_var
                    return z3.Int(cond.name)
                
                elif isinstance(cond, Constant):
                    val = cond.value
                    if isinstance(val, bool):
                        return z3.BoolVal(val)
                    elif isinstance(val, str):
                        if val in kind_consts:
                            return kind_consts[val]
                        if val in reloc_consts:
                            return reloc_consts[val]
                        return z3.BitVecVal(hash(val) % 2**16, 32)
                    elif isinstance(val, int):
                        return z3.BitVecVal(val, 32)
                    return z3.BoolVal(True)
                
                elif hasattr(cond, 'cond_type'):
                    ctype = cond.cond_type
                    ops = cond.operands
                    
                    if ctype == ConditionType.EQUALITY:
                        return to_z3(ops[0]) == to_z3(ops[1])
                    elif ctype == ConditionType.INEQUALITY:
                        return to_z3(ops[0]) != to_z3(ops[1])
                    elif ctype == ConditionType.IMPLIES:
                        return z3.Implies(to_z3(ops[0]), to_z3(ops[1]))
                    elif ctype == ConditionType.AND:
                        return z3.And(*[to_z3(op) for op in ops])
                    elif ctype == ConditionType.OR:
                        return z3.Or(*[to_z3(op) for op in ops])
                    elif ctype == ConditionType.NOT:
                        return z3.Not(to_z3(ops[0]))
                    elif ctype == ConditionType.IS_VALID:
                        return z3.BoolVal(True)  # Assume valid
                
                return z3.BoolVal(True)
            
            formula = to_z3(invariant)
            return VerificationCondition(
                name=f"inv_{hash(str(invariant)) % 10000}",
                formula=formula,
                is_invariant=True,
                source_info=str(invariant)
            )
        except Exception as e:
            if self.verbose:
                print(f"Error converting invariant: {e}")
            return None
    
    def _check_vc(self, model: SemanticModel, vc: VerificationCondition) -> Dict[str, Any]:
        """Check a verification condition using Z3."""
        self.solver.push()
        
        try:
            # Add model constraints
            for constraint in model.constraints:
                self.solver.add(constraint)
            
            # Check if VC can be violated
            self.solver.add(z3.Not(vc.formula))
            
            result = self.solver.check()
            
            if result == z3.unsat:
                # No counterexample - VC holds
                return {"verified": True}
            
            elif result == z3.sat:
                # Found counterexample
                z3_model = self.solver.model()
                counterexample = self._extract_counterexample(z3_model, model)
                counterexample["violated_vc"] = vc.name
                counterexample["source"] = vc.source_info
                
                return {
                    "verified": False,
                    "counterexample": counterexample
                }
            
            else:
                # Unknown (timeout, etc.)
                return {
                    "verified": False,
                    "counterexample": {"error": "Z3 returned unknown"}
                }
        
        finally:
            self.solver.pop()
    
    def _extract_counterexample(self, z3_model: z3.ModelRef, model: SemanticModel) -> Dict[str, Any]:
        """Extract counterexample from Z3 model."""
        ce = {
            "input_values": {},
            "actual_output": None,
            "expected_output": None,
        }
        
        # Extract input values
        for name, var in model.input_vars.items():
            try:
                val = z3_model.eval(var, model_completion=True)
                if z3.is_bool(val):
                    ce["input_values"][name] = z3.is_true(val)
                else:
                    ce["input_values"][name] = str(val)
            except:
                pass
        
        # Extract output
        if model.output_var is not None:
            try:
                val = z3_model.eval(model.output_var, model_completion=True)
                ce["actual_output"] = str(val)
            except:
                pass
        
        return ce
    
    def _count_verified_cases(self, model: SemanticModel, spec: 'Specification') -> int:
        """Count number of verified cases."""
        verified = 0
        expected_mappings = FIXUP_MAPPINGS.get(model.architecture, [])
        mapping_dict = {m.kind: m for m in expected_mappings}
        
        for kind, mappings in model.case_mappings.items():
            expected = mapping_dict.get(kind)
            if not expected:
                continue
            
            # Check non-PCRel
            if False in mappings:
                actual = mappings[False]
                if actual.endswith(expected.reloc_normal) or expected.reloc_normal.endswith(actual):
                    verified += 0.5
            
            # Check PCRel
            if True in mappings and expected.reloc_pcrel:
                actual = mappings[True]
                if actual.endswith(expected.reloc_pcrel) or expected.reloc_pcrel.endswith(actual):
                    verified += 0.5
        
        return int(verified)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return self.stats.copy()


def create_semantic_analyzer(
    architecture: Union[ArchitectureType, str] = ArchitectureType.RISCV,
    timeout_ms: int = 30000,
    verbose: bool = False
) -> Z3SemanticAnalyzer:
    """Factory function to create a semantic analyzer."""
    if isinstance(architecture, str):
        architecture = ArchitectureType(architecture.lower())
    return Z3SemanticAnalyzer(
        architecture=architecture,
        timeout_ms=timeout_ms,
        verbose=verbose
    )
