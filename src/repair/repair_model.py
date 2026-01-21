"""
Neural Repair Model for VEGA-Verified.

Provides neural-guided code repair using various strategies:
1. Template-based repair (rule-based patterns)
2. LLM-based repair (API integration)
3. Transformer-based repair (local model)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
from abc import ABC, abstractmethod
import re
import json
from pathlib import Path


class RepairStrategy(Enum):
    """Available repair strategies."""
    TEMPLATE = "template"       # Rule-based template matching
    LLM_API = "llm_api"         # External LLM API (OpenAI, etc.)
    LOCAL_MODEL = "local_model"  # Local transformer model
    HYBRID = "hybrid"           # Combination of strategies


@dataclass
class RepairCandidate:
    """A candidate repair."""
    code: str
    confidence: float
    strategy: RepairStrategy
    explanation: str = ""
    changes: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code[:500],
            "confidence": self.confidence,
            "strategy": self.strategy.value,
            "explanation": self.explanation,
            "changes": self.changes,
        }


class RepairModelBase(ABC):
    """Base class for repair models."""
    
    @abstractmethod
    def repair(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> List[RepairCandidate]:
        """
        Generate repair candidates.
        
        Args:
            code: Original buggy code
            counterexample: Counterexample from verification
            fault_location: Suspected fault location
            specification: Function specification
        
        Returns:
            List of repair candidates sorted by confidence
        """
        pass


class TemplateRepairModel(RepairModelBase):
    """
    Template-based repair model.
    
    Uses predefined patterns for common compiler backend bugs:
    1. Missing switch cases
    2. Wrong return values
    3. Missing null checks
    4. Off-by-one errors
    5. Incorrect enum mappings
    """
    
    def __init__(self):
        # Repair templates for common patterns
        self.templates: Dict[str, Callable] = {
            "missing_case": self._repair_missing_case,
            "wrong_return": self._repair_wrong_return,
            "missing_null_check": self._repair_missing_null_check,
            "off_by_one": self._repair_off_by_one,
            "wrong_enum": self._repair_wrong_enum,
            "missing_break": self._repair_missing_break,
        }
    
    def repair(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> List[RepairCandidate]:
        """Generate template-based repairs."""
        candidates = []
        
        # Detect bug pattern
        pattern = self._detect_pattern(code, counterexample, fault_location)
        
        if pattern in self.templates:
            repair_func = self.templates[pattern]
            repaired = repair_func(code, counterexample, fault_location, specification)
            
            if repaired:
                candidates.append(RepairCandidate(
                    code=repaired,
                    confidence=0.7,
                    strategy=RepairStrategy.TEMPLATE,
                    explanation=f"Applied {pattern} repair template",
                    changes=[{"pattern": pattern}]
                ))
        
        # Try all templates if specific pattern not found
        if not candidates:
            for pattern_name, repair_func in self.templates.items():
                try:
                    repaired = repair_func(code, counterexample, fault_location, specification)
                    if repaired and repaired != code:
                        candidates.append(RepairCandidate(
                            code=repaired,
                            confidence=0.5,
                            strategy=RepairStrategy.TEMPLATE,
                            explanation=f"Attempted {pattern_name} repair",
                            changes=[{"pattern": pattern_name}]
                        ))
                except Exception:
                    continue
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        return candidates[:5]  # Return top 5
    
    def _detect_pattern(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any]
    ) -> str:
        """Detect the type of bug pattern."""
        ce = counterexample or {}
        fault = fault_location or {}
        
        # Missing case: counterexample has unhandled enum value
        if 'input_values' in ce:
            inputs = ce['input_values']
            for key, value in inputs.items():
                if 'Kind' in key or 'kind' in key:
                    if str(value) not in code:
                        return "missing_case"
        
        # Wrong return: output doesn't match expected
        if ce.get('actual_output') != ce.get('expected_output'):
            if 'return' in fault.get('statement', ''):
                return "wrong_return"
        
        # Check for missing null check
        if 'nullptr' not in code and 'null' not in code.lower():
            for var in (ce.get('input_values', {}) or {}).keys():
                if '*' in var or '&' in var:
                    return "missing_null_check"
        
        return "unknown"
    
    def _repair_missing_case(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> Optional[str]:
        """Repair missing switch case."""
        ce = counterexample or {}
        inputs = ce.get('input_values', {}) or {}
        
        # Find the missing case value
        missing_case = None
        for key, value in inputs.items():
            if 'Kind' in key or 'kind' in key:
                if str(value) not in code:
                    missing_case = str(value)
                    break
        
        if not missing_case:
            return None
        
        # Find switch statement and add case
        switch_pattern = r'(switch\s*\([^)]+\)\s*\{)'
        match = re.search(switch_pattern, code)
        
        if match:
            # Determine return value based on case pattern
            # e.g., FK_Data_8 -> R_TARGET_64
            return_value = self._infer_return_for_case(missing_case, code)
            
            # Add case before default
            default_pattern = r'(\s*default\s*:)'
            new_case = f"\n    case {missing_case}: return {return_value};"
            
            if re.search(default_pattern, code):
                repaired = re.sub(default_pattern, new_case + r'\1', code)
            else:
                # Add before closing brace
                repaired = code.rstrip('}') + new_case + '\n}'
            
            return repaired
        
        return None
    
    def _infer_return_for_case(self, case_value: str, code: str) -> str:
        """Infer appropriate return value for a case."""
        # Extract pattern from existing cases
        case_return_pattern = r'case\s+(\w+):\s*return\s+(\S+);'
        
        existing_cases = {}
        for match in re.finditer(case_return_pattern, code):
            existing_cases[match.group(1)] = match.group(2)
        
        # Infer based on naming pattern
        if 'Data_8' in case_value or '_64' in case_value:
            # Find similar _64 return
            for ret in existing_cases.values():
                if '_64' in ret:
                    # Adjust target
                    return ret
            return 'ELF::R_TARGET_64'
        
        elif 'Data_4' in case_value or '_32' in case_value:
            for ret in existing_cases.values():
                if '_32' in ret:
                    return ret
            return 'ELF::R_TARGET_32'
        
        elif 'Data_2' in case_value or '_16' in case_value:
            for ret in existing_cases.values():
                if '_16' in ret:
                    return ret
            return 'ELF::R_TARGET_16'
        
        elif 'Data_1' in case_value or '_8' in case_value:
            for ret in existing_cases.values():
                if '_8' in ret:
                    return ret
            return 'ELF::R_TARGET_8'
        
        elif 'NONE' in case_value:
            for ret in existing_cases.values():
                if 'NONE' in ret:
                    return ret
            return 'ELF::R_TARGET_NONE'
        
        return '0'
    
    def _repair_wrong_return(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> Optional[str]:
        """Repair wrong return value."""
        ce = counterexample or {}
        fault = fault_location or {}
        
        expected = ce.get('expected_output')
        actual = ce.get('actual_output')
        stmt = fault.get('statement', '')
        
        if not expected or not stmt:
            return None
        
        # Find and replace wrong return
        if 'return' in stmt:
            # Extract return value
            match = re.search(r'return\s+(\S+);', stmt)
            if match:
                wrong_value = match.group(1)
                repaired = code.replace(
                    f'return {wrong_value};',
                    f'return {expected};',
                    1
                )
                return repaired
        
        return None
    
    def _repair_missing_null_check(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> Optional[str]:
        """Add missing null check."""
        # Find pointer parameters
        param_pattern = r'(\w+)\s*\*\s*(\w+)'
        
        for match in re.finditer(param_pattern, code):
            param_type = match.group(1)
            param_name = match.group(2)
            
            # Check if null check exists
            if f'{param_name} ==' not in code and f'!{param_name}' not in code:
                # Find function body start
                body_pattern = r'\)\s*(const)?\s*\{'
                body_match = re.search(body_pattern, code)
                
                if body_match:
                    insert_pos = body_match.end()
                    null_check = f'\n    if (!{param_name}) return 0;'
                    repaired = code[:insert_pos] + null_check + code[insert_pos:]
                    return repaired
        
        return None
    
    def _repair_off_by_one(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> Optional[str]:
        """Repair off-by-one errors."""
        fault = fault_location or {}
        stmt = fault.get('statement', '')
        
        if not stmt:
            return None
        
        # Common off-by-one patterns
        patterns = [
            (r'<\s*(\w+)', r'<= \1'),      # < to <=
            (r'<=\s*(\w+)', r'< \1'),      # <= to <
            (r'>\s*(\w+)', r'>= \1'),      # > to >=
            (r'>=\s*(\w+)', r'> \1'),      # >= to >
            (r'\+\s*1', ''),               # Remove +1
            (r'-\s*1', ''),                # Remove -1
        ]
        
        for old, new in patterns:
            if re.search(old, stmt):
                fixed_stmt = re.sub(old, new, stmt)
                repaired = code.replace(stmt, fixed_stmt)
                if repaired != code:
                    return repaired
        
        return None
    
    def _repair_wrong_enum(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> Optional[str]:
        """Repair wrong enum value."""
        ce = counterexample or {}
        expected = ce.get('expected_output', '')
        actual = ce.get('actual_output', '')
        
        if not expected or not actual:
            return None
        
        # Simple replacement
        if actual in code:
            return code.replace(actual, expected, 1)
        
        return None
    
    def _repair_missing_break(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> Optional[str]:
        """Repair missing break in switch."""
        # Find cases without break or return
        case_pattern = r'(case\s+\w+:)([^}]+?)(?=case|default|\})'
        
        def add_break(match):
            case_label = match.group(1)
            body = match.group(2)
            
            if 'break;' not in body and 'return' not in body:
                return case_label + body.rstrip() + '\n        break;\n    '
            return match.group(0)
        
        repaired = re.sub(case_pattern, add_break, code, flags=re.DOTALL)
        
        if repaired != code:
            return repaired
        
        return None


class LLMRepairModel(RepairModelBase):
    """
    LLM-based repair model using external API.
    
    Supports OpenAI, Anthropic, and other compatible APIs.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        api_base: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Check if API is available
        self._api_available = bool(api_key)
    
    def repair(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> List[RepairCandidate]:
        """Generate LLM-based repairs."""
        if not self._api_available:
            return []
        
        prompt = self._build_prompt(code, counterexample, fault_location, specification)
        
        try:
            response = self._call_api(prompt)
            candidates = self._parse_response(response)
            return candidates
        except Exception as e:
            return []
    
    def _build_prompt(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM."""
        ce = counterexample or {}
        fault = fault_location or {}
        spec = specification or {}
        
        return f"""You are an expert compiler engineer. Fix the bug in this LLVM backend function.

## Buggy Code
```cpp
{code}
```

## Counterexample (Test Case That Fails)
- Input: {ce.get('input_values', {})}
- Expected Output: {ce.get('expected_output', 'unknown')}
- Actual Output: {ce.get('actual_output', 'unknown')}

## Fault Location
Line {fault.get('line', '?')}: {fault.get('statement', 'unknown')}
Suspiciousness: {fault.get('suspiciousness', 0):.2f}

## Specification
Function: {spec.get('function_name', 'unknown')}
Preconditions: {spec.get('preconditions', [])}
Postconditions: {spec.get('postconditions', [])}

## Instructions
1. Identify the bug based on the counterexample
2. Fix ONLY the buggy part
3. Preserve all correct behavior
4. Return the complete fixed function

## Fixed Code
```cpp
"""
    
    def _call_api(self, prompt: str) -> str:
        """Call LLM API."""
        # Placeholder - would use actual API
        # In production, use openai.ChatCompletion.create() or similar
        return ""
    
    def _parse_response(self, response: str) -> List[RepairCandidate]:
        """Parse LLM response into repair candidates."""
        candidates = []
        
        # Extract code blocks
        code_pattern = r'```cpp\s*(.*?)\s*```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        for i, code in enumerate(matches):
            candidates.append(RepairCandidate(
                code=code.strip(),
                confidence=0.8 - (i * 0.1),  # Decreasing confidence
                strategy=RepairStrategy.LLM_API,
                explanation="LLM-generated repair"
            ))
        
        return candidates


class HybridRepairModel(RepairModelBase):
    """
    Hybrid repair model combining multiple strategies.
    
    Uses template-based repair first, falls back to LLM if needed.
    """
    
    def __init__(
        self,
        template_model: Optional[TemplateRepairModel] = None,
        llm_model: Optional[LLMRepairModel] = None,
        use_llm_fallback: bool = True
    ):
        self.template_model = template_model or TemplateRepairModel()
        self.llm_model = llm_model or LLMRepairModel()
        self.use_llm_fallback = use_llm_fallback
    
    def repair(
        self,
        code: str,
        counterexample: Dict[str, Any],
        fault_location: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> List[RepairCandidate]:
        """Generate repairs using hybrid strategy."""
        candidates = []
        
        # Try template-based repair first
        template_candidates = self.template_model.repair(
            code, counterexample, fault_location, specification
        )
        candidates.extend(template_candidates)
        
        # Use LLM as fallback if template repair didn't produce high-confidence results
        if self.use_llm_fallback:
            best_confidence = max((c.confidence for c in candidates), default=0)
            
            if best_confidence < 0.7:
                llm_candidates = self.llm_model.repair(
                    code, counterexample, fault_location, specification
                )
                candidates.extend(llm_candidates)
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        return candidates[:10]  # Return top 10


# Factory function
def create_repair_model(
    strategy: RepairStrategy = RepairStrategy.HYBRID,
    **kwargs
) -> RepairModelBase:
    """
    Create a repair model with the specified strategy.
    
    Args:
        strategy: Repair strategy to use
        **kwargs: Additional arguments for the model
    
    Returns:
        Configured repair model
    """
    if strategy == RepairStrategy.TEMPLATE:
        return TemplateRepairModel()
    elif strategy == RepairStrategy.LLM_API:
        return LLMRepairModel(**kwargs)
    elif strategy == RepairStrategy.HYBRID:
        return HybridRepairModel(**kwargs)
    else:
        return TemplateRepairModel()
