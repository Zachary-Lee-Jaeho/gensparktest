"""
C++ Parser for LLVM Backend Code.

This module provides robust C++ parsing capabilities for extracting
function definitions, signatures, and patterns from LLVM source code.

Features:
- Function signature extraction
- Switch/case pattern recognition  
- Control flow analysis
- Comment and documentation extraction
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from pathlib import Path


class FunctionType(Enum):
    """Types of functions in LLVM backends."""
    MEMBER = "member"           # Class member function
    STATIC = "static"           # Static function
    VIRTUAL = "virtual"         # Virtual function
    OVERRIDE = "override"       # Override function
    TEMPLATE = "template"       # Template function
    FREE = "free"               # Free function


@dataclass
class FunctionSignature:
    """Parsed C++ function signature."""
    name: str
    return_type: str
    parameters: List[Tuple[str, str]]  # [(type, name), ...]
    class_name: Optional[str] = None
    qualifiers: List[str] = field(default_factory=list)  # const, override, etc.
    is_virtual: bool = False
    is_static: bool = False
    is_inline: bool = False
    template_params: Optional[str] = None
    
    @property
    def full_name(self) -> str:
        """Get fully qualified name."""
        if self.class_name:
            return f"{self.class_name}::{self.name}"
        return self.name
    
    @property
    def param_types(self) -> List[str]:
        """Get parameter types only."""
        return [p[0] for p in self.parameters]
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "full_name": self.full_name,
            "return_type": self.return_type,
            "parameters": self.parameters,
            "class_name": self.class_name,
            "qualifiers": self.qualifiers,
            "is_virtual": self.is_virtual,
            "is_static": self.is_static,
        }


@dataclass 
class SwitchCasePattern:
    """Represents a switch/case pattern found in code."""
    switch_variable: str
    cases: List[Tuple[str, str]]  # [(case_value, return_value), ...]
    default_value: Optional[str] = None
    has_ternary_ispcrel: bool = False
    location: Tuple[int, int] = (0, 0)  # (start_line, end_line)
    
    def to_dict(self) -> Dict:
        return {
            "switch_variable": self.switch_variable,
            "cases": self.cases,
            "default_value": self.default_value,
            "has_ternary_ispcrel": self.has_ternary_ispcrel,
            "num_cases": len(self.cases),
        }


@dataclass
class ParsedFunction:
    """Complete parsed function with body and metadata."""
    signature: FunctionSignature
    body: str
    raw_code: str
    file_path: str
    start_line: int
    end_line: int
    switch_patterns: List[SwitchCasePattern] = field(default_factory=list)
    called_functions: Set[str] = field(default_factory=set)
    comments: List[str] = field(default_factory=list)
    
    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1
    
    def to_dict(self) -> Dict:
        return {
            "signature": self.signature.to_dict(),
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "line_count": self.line_count,
            "switch_patterns": [s.to_dict() for s in self.switch_patterns],
            "called_functions": list(self.called_functions),
            "has_switch": len(self.switch_patterns) > 0,
        }


class CppParser:
    """
    C++ Parser optimized for LLVM backend code.
    
    Uses regex-based parsing which is fast and sufficient for 
    well-structured LLVM code. For complex cases, can integrate
    tree-sitter as a fallback.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for parsing."""
        
        # Function definition pattern (handles multi-line)
        # Matches: ReturnType ClassName::FunctionName(params) qualifiers {
        self.func_def_pattern = re.compile(
            r'''
            # Optional template
            (?:template\s*<[^>]*>\s*)?
            # Return type (handles complex types)
            ([\w\s\*&:<>,]+?)
            \s+
            # Class name (optional) and function name
            ((?:\w+::)*)(\w+)
            \s*
            # Parameters
            \(([^)]*)\)
            # Qualifiers (const, override, etc.)
            \s*((?:const|override|final|noexcept|\[\[.*?\]\]|\s)*)
            \s*
            # Function body start
            \{
            ''',
            re.VERBOSE | re.MULTILINE
        )
        
        # Switch statement pattern
        self.switch_pattern = re.compile(
            r'switch\s*\(\s*(\w+(?:\.\w+|\->w+)*(?:\(\))?)\s*\)\s*\{',
            re.MULTILINE
        )
        
        # Case pattern with return
        self.case_return_pattern = re.compile(
            r'case\s+([\w:]+)\s*:\s*(?:return\s+)?([\w:]+(?:\s*\?\s*[\w:]+\s*:\s*[\w:]+)?)\s*;',
            re.MULTILINE
        )
        
        # Case with ternary (IsPCRel pattern)
        self.case_ternary_pattern = re.compile(
            r'case\s+([\w:]+)\s*:\s*return\s+(\w+)\s*\?\s*([\w:]+)\s*:\s*([\w:]+)\s*;',
            re.MULTILINE
        )
        
        # Default case pattern
        self.default_pattern = re.compile(
            r'default\s*:\s*(?:return\s+)?([\w:]+)\s*;',
            re.MULTILINE
        )
        
        # Function call pattern
        self.func_call_pattern = re.compile(
            r'\b(\w+)\s*\(',
            re.MULTILINE
        )
        
        # Class definition pattern
        self.class_pattern = re.compile(
            r'class\s+(\w+)\s*(?::\s*(?:public|private|protected)\s+[\w<>:,\s]+)?\s*\{',
            re.MULTILINE
        )
        
        # Comment patterns
        self.single_comment = re.compile(r'//.*$', re.MULTILINE)
        self.multi_comment = re.compile(r'/\*.*?\*/', re.DOTALL)
    
    def parse_file(self, file_path: Path) -> List[ParsedFunction]:
        """
        Parse a C++ file and extract all function definitions.
        
        Args:
            file_path: Path to the C++ source file
            
        Returns:
            List of ParsedFunction objects
        """
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            if self.verbose:
                print(f"Error reading {file_path}: {e}")
            return []
        
        return self.parse_content(content, str(file_path))
    
    def parse_content(self, content: str, file_path: str = "<string>") -> List[ParsedFunction]:
        """
        Parse C++ content and extract functions.
        
        Args:
            content: C++ source code
            file_path: Source file path for metadata
            
        Returns:
            List of ParsedFunction objects
        """
        functions = []
        lines = content.split('\n')
        
        # Find all function definitions
        for match in self.func_def_pattern.finditer(content):
            try:
                func = self._parse_function_match(match, content, lines, file_path)
                if func:
                    functions.append(func)
            except Exception as e:
                if self.verbose:
                    print(f"Error parsing function: {e}")
                continue
        
        return functions
    
    def _parse_function_match(
        self, 
        match: re.Match, 
        content: str, 
        lines: List[str],
        file_path: str
    ) -> Optional[ParsedFunction]:
        """Parse a single function from regex match."""
        
        return_type = match.group(1).strip()
        class_prefix = match.group(2).rstrip('::') if match.group(2) else None
        func_name = match.group(3)
        params_str = match.group(4)
        qualifiers_str = match.group(5).strip()
        
        # Skip if this looks like a control structure
        if func_name in ('if', 'while', 'for', 'switch', 'catch'):
            return None
        
        # Parse parameters
        parameters = self._parse_parameters(params_str)
        
        # Parse qualifiers
        qualifiers = [q.strip() for q in qualifiers_str.split() if q.strip()]
        
        # Create signature
        signature = FunctionSignature(
            name=func_name,
            return_type=return_type,
            parameters=parameters,
            class_name=class_prefix,
            qualifiers=qualifiers,
            is_virtual='virtual' in return_type.lower(),
            is_static='static' in return_type.lower(),
            is_inline='inline' in return_type.lower(),
        )
        
        # Extract function body
        start_pos = match.end() - 1  # Position of opening brace
        body, end_pos = self._extract_brace_content(content, start_pos)
        
        if body is None:
            return None
        
        # Calculate line numbers
        start_line = content[:match.start()].count('\n') + 1
        end_line = content[:end_pos].count('\n') + 1
        
        # Get raw code
        raw_code = content[match.start():end_pos + 1]
        
        # Analyze function body
        switch_patterns = self._extract_switch_patterns(body)
        called_functions = self._extract_called_functions(body)
        
        # Extract leading comments
        comments = self._extract_comments(content, match.start())
        
        return ParsedFunction(
            signature=signature,
            body=body,
            raw_code=raw_code,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            switch_patterns=switch_patterns,
            called_functions=called_functions,
            comments=comments,
        )
    
    def _parse_parameters(self, params_str: str) -> List[Tuple[str, str]]:
        """Parse function parameters."""
        params = []
        if not params_str.strip():
            return params
        
        # Split by comma, handling template commas
        depth = 0
        current = ""
        for char in params_str:
            if char in '<(':
                depth += 1
            elif char in '>)':
                depth -= 1
            elif char == ',' and depth == 0:
                if current.strip():
                    params.append(self._parse_single_param(current.strip()))
                current = ""
                continue
            current += char
        
        if current.strip():
            params.append(self._parse_single_param(current.strip()))
        
        return params
    
    def _parse_single_param(self, param: str) -> Tuple[str, str]:
        """Parse a single parameter into (type, name)."""
        # Handle default values
        if '=' in param:
            param = param.split('=')[0].strip()
        
        # Split type and name
        parts = param.rsplit(None, 1)
        if len(parts) == 2:
            type_part, name_part = parts
            # Handle pointers/references attached to name
            while name_part and name_part[0] in '*&':
                type_part += name_part[0]
                name_part = name_part[1:]
            return (type_part.strip(), name_part.strip())
        else:
            return (param, "")
    
    def _extract_brace_content(self, content: str, start: int) -> Tuple[Optional[str], int]:
        """Extract content between matching braces."""
        if start >= len(content) or content[start] != '{':
            return None, start
        
        depth = 1
        pos = start + 1
        
        while pos < len(content) and depth > 0:
            char = content[pos]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
            elif char == '"':
                # Skip string literals
                pos += 1
                while pos < len(content) and content[pos] != '"':
                    if content[pos] == '\\':
                        pos += 1
                    pos += 1
            elif char == "'":
                # Skip char literals
                pos += 1
                while pos < len(content) and content[pos] != "'":
                    if content[pos] == '\\':
                        pos += 1
                    pos += 1
            elif content[pos:pos+2] == '//':
                # Skip single-line comments
                while pos < len(content) and content[pos] != '\n':
                    pos += 1
            elif content[pos:pos+2] == '/*':
                # Skip multi-line comments
                pos += 2
                while pos < len(content) - 1 and content[pos:pos+2] != '*/':
                    pos += 1
                pos += 1
            pos += 1
        
        if depth != 0:
            return None, start
        
        return content[start+1:pos-1], pos - 1
    
    def _extract_switch_patterns(self, body: str) -> List[SwitchCasePattern]:
        """Extract switch/case patterns from function body."""
        patterns = []
        
        for switch_match in self.switch_pattern.finditer(body):
            switch_var = switch_match.group(1)
            switch_start = switch_match.end() - 1
            
            # Extract switch body
            switch_body, switch_end = self._extract_brace_content(body, switch_start)
            if switch_body is None:
                continue
            
            cases = []
            has_ternary = False
            
            # Find ternary cases (IsPCRel pattern)
            for case_match in self.case_ternary_pattern.finditer(switch_body):
                case_val = case_match.group(1)
                condition = case_match.group(2)
                true_val = case_match.group(3)
                false_val = case_match.group(4)
                
                if condition == "IsPCRel":
                    has_ternary = True
                    cases.append((case_val, f"{condition} ? {true_val} : {false_val}"))
                else:
                    cases.append((case_val, true_val))
            
            # Find simple cases
            for case_match in self.case_return_pattern.finditer(switch_body):
                case_val = case_match.group(1)
                ret_val = case_match.group(2)
                
                # Skip if already found as ternary
                if any(c[0] == case_val for c in cases):
                    continue
                    
                cases.append((case_val, ret_val))
            
            # Find default
            default_match = self.default_pattern.search(switch_body)
            default_val = default_match.group(1) if default_match else None
            
            if cases:
                patterns.append(SwitchCasePattern(
                    switch_variable=switch_var,
                    cases=cases,
                    default_value=default_val,
                    has_ternary_ispcrel=has_ternary,
                ))
        
        return patterns
    
    def _extract_called_functions(self, body: str) -> Set[str]:
        """Extract function calls from body."""
        # Remove comments first
        clean_body = self.single_comment.sub('', body)
        clean_body = self.multi_comment.sub('', clean_body)
        
        calls = set()
        for match in self.func_call_pattern.finditer(clean_body):
            func_name = match.group(1)
            # Filter out keywords and common patterns
            if func_name not in ('if', 'while', 'for', 'switch', 'return', 'sizeof', 'typeof', 'alignof'):
                calls.add(func_name)
        
        return calls
    
    def _extract_comments(self, content: str, func_start: int) -> List[str]:
        """Extract comments preceding a function."""
        comments = []
        
        # Look backwards for comments
        pos = func_start - 1
        while pos > 0 and content[pos] in ' \t\n':
            pos -= 1
        
        # Check for multi-line comment
        if pos > 1 and content[pos-1:pos+1] == '*/':
            end = pos + 1
            start = content.rfind('/*', 0, pos)
            if start != -1:
                comments.append(content[start:end])
        
        # Check for single-line comments
        lines = content[:func_start].split('\n')
        for line in reversed(lines[-5:]):
            stripped = line.strip()
            if stripped.startswith('//'):
                comments.insert(0, stripped)
            elif stripped and not stripped.startswith('//'):
                break
        
        return comments


# Quick test
if __name__ == "__main__":
    test_code = '''
    /// Get the relocation type for a fixup.
    unsigned RISCVELFObjectWriter::getRelocType(MCContext &Ctx,
                                                 const MCValue &Target,
                                                 const MCFixup &Fixup,
                                                 bool IsPCRel) const {
        unsigned Kind = Fixup.getTargetKind();
        
        switch (Kind) {
        case FK_Data_4:
            return IsPCRel ? ELF::R_RISCV_32_PCREL : ELF::R_RISCV_32;
        case FK_Data_8:
            return IsPCRel ? ELF::R_RISCV_64_PCREL : ELF::R_RISCV_64;
        default:
            return ELF::R_RISCV_NONE;
        }
    }
    
    void RISCVMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                                SmallVectorImpl<char> &CB,
                                                SmallVectorImpl<MCFixup> &Fixups,
                                                const MCSubtargetInfo &STI) const {
        uint32_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
        support::endian::write<uint32_t>(CB, Bits, support::little);
    }
    '''
    
    parser = CppParser(verbose=True)
    functions = parser.parse_content(test_code, "test.cpp")
    
    print(f"\nFound {len(functions)} functions:")
    for func in functions:
        print(f"\n  {func.signature.full_name}")
        print(f"    Return: {func.signature.return_type}")
        print(f"    Params: {func.signature.parameters}")
        print(f"    Lines: {func.start_line}-{func.end_line}")
        if func.switch_patterns:
            for sp in func.switch_patterns:
                print(f"    Switch on '{sp.switch_variable}': {len(sp.cases)} cases")
                print(f"      IsPCRel ternary: {sp.has_ternary_ispcrel}")
