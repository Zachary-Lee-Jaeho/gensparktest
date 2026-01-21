"""
Pattern abstraction for specification inference.
Generalizes concrete values to abstract patterns.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Set, Optional, Tuple
import re


@dataclass
class AbstractPattern:
    """An abstracted pattern from concrete implementations."""
    pattern_type: str  # switch, mapping, guard, etc.
    template: str
    variables: List[str]
    constraints: List[str]
    concrete_instances: List[Dict[str, Any]]


class PatternAbstractor:
    """
    Abstracts concrete code patterns into generalized templates.
    
    Key abstractions:
    - Target names: ARM, MIPS, RISCV -> <TARGET>
    - Register names: X0, R0, A0 -> <REG>
    - Relocation types: R_ARM_*, R_MIPS_* -> R_<TARGET>_*
    - Fixup kinds: preserved across targets
    """
    
    # Target name patterns
    TARGET_NAMES = ['ARM', 'MIPS', 'RISCV', 'X86', 'AArch64', 'RI5CY', 'xCORE', 'PULP']
    
    # Common relocation type patterns
    RELOC_PATTERNS = [
        (r'R_(ARM|MIPS|RISCV|X86|AARCH64)_NONE', 'R_<TARGET>_NONE'),
        (r'R_(ARM|MIPS|RISCV|X86|AARCH64)_8', 'R_<TARGET>_8'),
        (r'R_(ARM|MIPS|RISCV|X86|AARCH64)_16', 'R_<TARGET>_16'),
        (r'R_(ARM|MIPS|RISCV|X86|AARCH64)_32', 'R_<TARGET>_32'),
        (r'R_(ARM|MIPS|RISCV|X86|AARCH64)_64', 'R_<TARGET>_64'),
        (r'R_(ARM|MIPS|RISCV|X86|AARCH64)_PC32', 'R_<TARGET>_PC32'),
        (r'R_(ARM|MIPS|RISCV|X86|AARCH64)_(\w+)', 'R_<TARGET>_<RELOC_TYPE>'),
    ]
    
    # Common fixup kinds (target-independent)
    FIXUP_KINDS = [
        'FK_NONE', 'FK_Data_1', 'FK_Data_2', 'FK_Data_4', 'FK_Data_8',
        'FK_PCRel_1', 'FK_PCRel_2', 'FK_PCRel_4', 'FK_PCRel_8',
        'FK_GPRel_1', 'FK_GPRel_2', 'FK_GPRel_4',
        'FK_SecRel_1', 'FK_SecRel_2', 'FK_SecRel_4', 'FK_SecRel_8',
    ]
    
    def __init__(self):
        self.target_pattern = re.compile(
            r'\b(' + '|'.join(self.TARGET_NAMES) + r')\b'
        )
    
    def abstract(
        self,
        statements: List[Dict[str, Any]],
        target_name: str
    ) -> List[AbstractPattern]:
        """
        Abstract a list of statements into patterns.
        
        Args:
            statements: Parsed statements from a function
            target_name: Name of the target (for abstraction)
            
        Returns:
            List of abstracted patterns
        """
        patterns = []
        
        for stmt in statements:
            pattern = self._abstract_statement(stmt, target_name)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _abstract_statement(
        self,
        stmt: Dict[str, Any],
        target_name: str
    ) -> Optional[AbstractPattern]:
        """Abstract a single statement."""
        stmt_type = stmt.get('type', 'other')
        
        if stmt_type == 'switch':
            return self._abstract_switch(stmt, target_name)
        elif stmt_type == 'if':
            return self._abstract_if(stmt, target_name)
        elif stmt_type == 'return':
            return self._abstract_return(stmt, target_name)
        elif stmt_type == 'case':
            return self._abstract_case(stmt, target_name)
        else:
            return self._abstract_other(stmt, target_name)
    
    def _abstract_switch(
        self,
        stmt: Dict[str, Any],
        target_name: str
    ) -> AbstractPattern:
        """Abstract a switch statement."""
        cases = stmt.get('cases', [])
        
        # Abstract each case
        abstracted_cases = []
        for case in cases:
            case_value = case.get('case', '')
            return_value = case.get('return', '')
            
            # Abstract return value
            abstract_return = self._abstract_value(return_value, target_name)
            
            abstracted_cases.append({
                'case': case_value,  # Fixup kinds are usually target-independent
                'return': abstract_return,
                'original_return': return_value
            })
        
        # Build template
        template_parts = ['switch (Fixup.getTargetKind()) {']
        for ac in abstracted_cases:
            template_parts.append(f"  case {ac['case']}: return {ac['return']};")
        template_parts.append('}')
        template = '\n'.join(template_parts)
        
        return AbstractPattern(
            pattern_type='switch',
            template=template,
            variables=['<TARGET>'],
            constraints=[f"case {ac['case']} -> {ac['return']}" for ac in abstracted_cases],
            concrete_instances=[{
                'target': target_name,
                'cases': cases
            }]
        )
    
    def _abstract_if(
        self,
        stmt: Dict[str, Any],
        target_name: str
    ) -> AbstractPattern:
        """Abstract an if statement."""
        condition = stmt.get('condition', '')
        text = stmt.get('text', '')
        
        # Abstract target-specific parts
        abstract_cond = self._abstract_value(condition, target_name)
        abstract_text = self._abstract_value(text, target_name)
        
        return AbstractPattern(
            pattern_type='if',
            template=abstract_text,
            variables=['<TARGET>'],
            constraints=[abstract_cond],
            concrete_instances=[{
                'target': target_name,
                'condition': condition,
                'text': text
            }]
        )
    
    def _abstract_return(
        self,
        stmt: Dict[str, Any],
        target_name: str
    ) -> AbstractPattern:
        """Abstract a return statement."""
        value = stmt.get('value', '')
        abstract_value = self._abstract_value(value, target_name)
        
        return AbstractPattern(
            pattern_type='return',
            template=f'return {abstract_value};',
            variables=['<TARGET>'],
            constraints=[],
            concrete_instances=[{
                'target': target_name,
                'value': value
            }]
        )
    
    def _abstract_case(
        self,
        stmt: Dict[str, Any],
        target_name: str
    ) -> AbstractPattern:
        """Abstract a case statement."""
        case_value = stmt.get('case_value', '')
        text = stmt.get('text', '')
        abstract_text = self._abstract_value(text, target_name)
        
        return AbstractPattern(
            pattern_type='case',
            template=abstract_text,
            variables=['<TARGET>'],
            constraints=[case_value],
            concrete_instances=[{
                'target': target_name,
                'case': case_value,
                'text': text
            }]
        )
    
    def _abstract_other(
        self,
        stmt: Dict[str, Any],
        target_name: str
    ) -> AbstractPattern:
        """Abstract any other statement."""
        text = stmt.get('text', '')
        abstract_text = self._abstract_value(text, target_name)
        
        return AbstractPattern(
            pattern_type='other',
            template=abstract_text,
            variables=['<TARGET>'],
            constraints=[],
            concrete_instances=[{
                'target': target_name,
                'text': text
            }]
        )
    
    def _abstract_value(self, value: str, target_name: str) -> str:
        """Abstract target-specific values to templates."""
        result = value
        
        # Replace target name
        result = re.sub(
            rf'\b{target_name}\b',
            '<TARGET>',
            result,
            flags=re.IGNORECASE
        )
        
        # Replace relocation types
        for pattern, replacement in self.RELOC_PATTERNS:
            result = re.sub(pattern, replacement, result)
        
        # Replace class names with target prefix
        result = re.sub(
            rf'\b({target_name})(\w+)(?=::|\.|->)',
            r'<TARGET>\2',
            result,
            flags=re.IGNORECASE
        )
        
        return result
    
    def find_common_pattern(
        self,
        patterns: List[AbstractPattern]
    ) -> Optional[AbstractPattern]:
        """
        Find common pattern across multiple abstracted patterns.
        
        Args:
            patterns: Patterns from different targets
            
        Returns:
            Common pattern if found, None otherwise
        """
        if not patterns:
            return None
        
        if len(patterns) == 1:
            return patterns[0]
        
        # Group by pattern type
        by_type: Dict[str, List[AbstractPattern]] = {}
        for p in patterns:
            if p.pattern_type not in by_type:
                by_type[p.pattern_type] = []
            by_type[p.pattern_type].append(p)
        
        # For each type, find common structure
        common_patterns = []
        for ptype, type_patterns in by_type.items():
            if len(type_patterns) == len(patterns):
                # All patterns have this type - good candidate
                common = self._merge_patterns(type_patterns)
                if common:
                    common_patterns.append(common)
        
        return common_patterns[0] if common_patterns else None
    
    def _merge_patterns(
        self,
        patterns: List[AbstractPattern]
    ) -> Optional[AbstractPattern]:
        """Merge multiple patterns into a common pattern."""
        if not patterns:
            return None
        
        # For switch patterns, merge case mappings
        if patterns[0].pattern_type == 'switch':
            return self._merge_switch_patterns(patterns)
        
        # For other types, use the template if they match
        templates = set(p.template for p in patterns)
        if len(templates) == 1:
            # All templates match
            merged_instances = []
            for p in patterns:
                merged_instances.extend(p.concrete_instances)
            
            return AbstractPattern(
                pattern_type=patterns[0].pattern_type,
                template=patterns[0].template,
                variables=patterns[0].variables,
                constraints=list(set(c for p in patterns for c in p.constraints)),
                concrete_instances=merged_instances
            )
        
        return None
    
    def _merge_switch_patterns(
        self,
        patterns: List[AbstractPattern]
    ) -> AbstractPattern:
        """Merge switch patterns from different targets."""
        # Collect all case mappings
        all_cases: Dict[str, Set[str]] = {}  # case -> {abstract_returns}
        all_instances = []
        
        for p in patterns:
            for instance in p.concrete_instances:
                all_instances.append(instance)
                for case in instance.get('cases', []):
                    case_value = case.get('case', '')
                    return_value = case.get('return', '')
                    
                    # Abstract the return
                    abstract_return = self._abstract_value(
                        return_value,
                        instance.get('target', '')
                    )
                    
                    if case_value not in all_cases:
                        all_cases[case_value] = set()
                    all_cases[case_value].add(abstract_return)
        
        # Build merged template
        template_parts = ['switch (Fixup.getTargetKind()) {']
        constraints = []
        
        for case_value in sorted(all_cases.keys()):
            returns = all_cases[case_value]
            if len(returns) == 1:
                # Consistent mapping
                abstract_return = list(returns)[0]
                template_parts.append(f"  case {case_value}: return {abstract_return};")
                constraints.append(f"{case_value} -> {abstract_return}")
            else:
                # Variable mapping
                template_parts.append(f"  case {case_value}: return <TARGET_SPECIFIC>;")
                constraints.append(f"{case_value} -> one of {returns}")
        
        template_parts.append('}')
        
        return AbstractPattern(
            pattern_type='switch',
            template='\n'.join(template_parts),
            variables=['<TARGET>', '<TARGET_SPECIFIC>'],
            constraints=constraints,
            concrete_instances=all_instances
        )
    
    def extract_mapping_rules(
        self,
        patterns: List[AbstractPattern]
    ) -> Dict[str, str]:
        """
        Extract case -> return mapping rules from patterns.
        
        Returns:
            Dictionary of fixup_kind -> abstract_return_pattern
        """
        rules = {}
        
        for p in patterns:
            if p.pattern_type == 'switch':
                for constraint in p.constraints:
                    if ' -> ' in constraint:
                        parts = constraint.split(' -> ')
                        if len(parts) == 2:
                            rules[parts[0]] = parts[1]
        
        return rules
