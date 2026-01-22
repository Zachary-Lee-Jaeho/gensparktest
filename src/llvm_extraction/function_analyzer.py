"""
LLVM Function Analyzer for VEGA-Verified.

This module provides advanced analysis of LLVM backend functions:
1. Call Graph construction
2. Dependency analysis
3. Function-Test mapping
4. Instruction coverage tracking

Key features:
- Static call graph analysis from source code
- Cross-module dependency tracking
- Test coverage mapping
- Instruction usage analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import re
import json
from pathlib import Path
from collections import defaultdict


class FunctionType(Enum):
    """Types of backend functions."""
    INSTRUCTION_SELECTION = "isel"
    CODE_EMISSION = "emit"
    REGISTER_ALLOCATION = "regalloc"
    FRAME_LOWERING = "frame"
    RELOCATION = "reloc"
    SCHEDULING = "sched"
    UTILITY = "util"
    UNKNOWN = "unknown"


@dataclass
class FunctionSignature:
    """Parsed function signature."""
    name: str
    return_type: str
    parameters: List[Tuple[str, str]]  # (type, name)
    is_const: bool = False
    is_virtual: bool = False
    is_override: bool = False
    is_static: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'return_type': self.return_type,
            'parameters': [{'type': t, 'name': n} for t, n in self.parameters],
            'is_const': self.is_const,
            'is_virtual': self.is_virtual,
            'is_override': self.is_override,
            'is_static': self.is_static,
        }


@dataclass
class CallSite:
    """A call to another function."""
    callee: str
    line_number: int
    arguments: List[str] = field(default_factory=list)
    is_member_call: bool = False
    caller_class: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'callee': self.callee,
            'line': self.line_number,
            'arguments': self.arguments,
            'is_member': self.is_member_call,
        }


@dataclass
class FunctionNode:
    """A node in the call graph."""
    name: str
    backend: str
    module: str
    signature: Optional[FunctionSignature] = None
    function_type: FunctionType = FunctionType.UNKNOWN
    calls: List[CallSite] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    instructions_used: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    test_files: List[str] = field(default_factory=list)
    source_code: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'backend': self.backend,
            'module': self.module,
            'type': self.function_type.value,
            'signature': self.signature.to_dict() if self.signature else None,
            'calls': [c.to_dict() for c in self.calls],
            'called_by': self.called_by,
            'instructions_used': list(self.instructions_used),
            'dependencies': list(self.dependencies),
            'test_files': self.test_files,
            'stats': {
                'call_count': len(self.calls),
                'caller_count': len(self.called_by),
                'instruction_count': len(self.instructions_used),
                'dependency_count': len(self.dependencies),
            }
        }


class CallGraphBuilder:
    """
    Builds a call graph from LLVM backend source code.
    
    Analyzes:
    - Direct function calls
    - Member function calls
    - Virtual dispatch
    - Cross-file dependencies
    """
    
    # Pattern to match function calls
    CALL_PATTERN = re.compile(
        r'(?:(?:(\w+)(?:->|\.))|(?:::))?(\w+)\s*\(([^)]*)\)',
        re.MULTILINE
    )
    
    # Pattern to match function definitions
    FUNCTION_DEF_PATTERN = re.compile(
        r'(?:(static|virtual)\s+)?'
        r'(\w+(?:\s*[*&])?(?:\s*const)?)\s+'
        r'(\w+)::(\w+)\s*\(([^)]*)\)'
        r'(?:\s*(const))?\s*(?:(override))?\s*\{',
        re.MULTILINE
    )
    
    # Pattern for LLVM-specific calls
    LLVM_CALL_PATTERNS = [
        re.compile(r'BuildMI\s*\([^,]+,\s*[^,]+,\s*(\w+)'),  # BuildMI for instruction
        re.compile(r'get(\w+)Opcode\s*\(\)'),  # getXXXOpcode calls
        re.compile(r'emitTo(\w+)\s*\('),  # emitToXXX calls
        re.compile(r'lower(\w+)\s*\('),  # lowerXXX calls
        re.compile(r'select(\w+)\s*\('),  # selectXXX calls
    ]
    
    # LLVM instruction patterns in code
    LLVM_INST_PATTERN = re.compile(
        r'\b(RISCV|ARM|AArch64|X86|Mips)::(\w+)\b'
    )
    
    def __init__(self):
        self.nodes: Dict[str, FunctionNode] = {}
        self.edges: List[Tuple[str, str]] = []
    
    def add_function(self, name: str, backend: str, module: str, source_code: str) -> FunctionNode:
        """Add a function to the call graph."""
        if name in self.nodes:
            return self.nodes[name]
        
        node = FunctionNode(
            name=name,
            backend=backend,
            module=module,
            source_code=source_code,
        )
        
        # Detect function type
        node.function_type = self._detect_function_type(name, source_code)
        
        # Parse signature
        node.signature = self._parse_signature(name, source_code)
        
        # Extract calls
        node.calls = self._extract_calls(source_code)
        
        # Extract instructions used
        node.instructions_used = self._extract_instructions(source_code, backend)
        
        # Extract dependencies
        node.dependencies = self._extract_dependencies(source_code)
        
        self.nodes[name] = node
        return node
    
    def _detect_function_type(self, name: str, source_code: str) -> FunctionType:
        """Detect the type of function based on name and content."""
        name_lower = name.lower()
        
        if any(kw in name_lower for kw in ['select', 'isel', 'lower']):
            return FunctionType.INSTRUCTION_SELECTION
        elif any(kw in name_lower for kw in ['emit', 'encode', 'write']):
            return FunctionType.CODE_EMISSION
        elif any(kw in name_lower for kw in ['reloc', 'fixup', 'getreloc']):
            return FunctionType.RELOCATION
        elif any(kw in name_lower for kw in ['regalloc', 'spill', 'reload', 'regclass']):
            return FunctionType.REGISTER_ALLOCATION
        elif any(kw in name_lower for kw in ['frame', 'stack', 'prologue', 'epilogue']):
            return FunctionType.FRAME_LOWERING
        elif any(kw in name_lower for kw in ['sched', 'latency', 'hazard']):
            return FunctionType.SCHEDULING
        
        return FunctionType.UTILITY
    
    def _parse_signature(self, name: str, source_code: str) -> Optional[FunctionSignature]:
        """Parse function signature from source code."""
        # Try to find the function definition
        pattern = re.compile(
            rf'(?:(static|virtual)\s+)?'
            rf'(\w+(?:\s*[*&])?(?:\s*const)?)\s+'
            rf'(?:\w+::)?{re.escape(name)}\s*\(([^)]*)\)'
            rf'(?:\s*(const))?\s*(?:(override))?',
            re.MULTILINE
        )
        
        match = pattern.search(source_code)
        if not match:
            return None
        
        modifier = match.group(1)
        return_type = match.group(2)
        params_str = match.group(3)
        is_const = match.group(4) is not None
        is_override = match.group(5) is not None
        
        # Parse parameters
        parameters = []
        if params_str.strip():
            for param in params_str.split(','):
                param = param.strip()
                if param:
                    # Split type and name
                    parts = param.rsplit(None, 1)
                    if len(parts) == 2:
                        parameters.append((parts[0], parts[1]))
                    else:
                        parameters.append((parts[0], ''))
        
        return FunctionSignature(
            name=name,
            return_type=return_type,
            parameters=parameters,
            is_const=is_const,
            is_virtual=modifier == 'virtual',
            is_static=modifier == 'static',
            is_override=is_override,
        )
    
    def _extract_calls(self, source_code: str) -> List[CallSite]:
        """Extract function calls from source code."""
        calls = []
        lines = source_code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith('//'):
                continue
            
            # Find function calls
            for match in self.CALL_PATTERN.finditer(line):
                obj = match.group(1)  # Object if member call
                func_name = match.group(2)
                args_str = match.group(3)
                
                # Skip keywords and common patterns
                if func_name in ['if', 'while', 'for', 'switch', 'return', 'sizeof', 'assert']:
                    continue
                
                # Parse arguments
                args = [a.strip() for a in args_str.split(',') if a.strip()]
                
                calls.append(CallSite(
                    callee=func_name,
                    line_number=line_num,
                    arguments=args,
                    is_member_call=obj is not None,
                    caller_class=obj,
                ))
        
        return calls
    
    def _extract_instructions(self, source_code: str, backend: str) -> Set[str]:
        """Extract LLVM instructions used in the function."""
        instructions = set()
        
        for match in self.LLVM_INST_PATTERN.finditer(source_code):
            inst_backend = match.group(1)
            inst_name = match.group(2)
            
            # Only include instructions from the same backend
            if inst_backend == backend or inst_backend in backend:
                instructions.add(inst_name)
        
        return instructions
    
    def _extract_dependencies(self, source_code: str) -> Set[str]:
        """Extract file/module dependencies."""
        dependencies = set()
        
        # Find includes
        include_pattern = re.compile(r'#include\s*[<"]([^>"]+)[>"]')
        for match in include_pattern.finditer(source_code):
            dependencies.add(match.group(1))
        
        # Find using declarations
        using_pattern = re.compile(r'using\s+(\w+)(?:::(\w+))?')
        for match in using_pattern.finditer(source_code):
            namespace = match.group(1)
            if namespace not in ['namespace', 'std']:
                dependencies.add(namespace)
        
        return dependencies
    
    def build_edges(self):
        """Build edges between function nodes."""
        self.edges = []
        
        for name, node in self.nodes.items():
            for call in node.calls:
                if call.callee in self.nodes:
                    self.edges.append((name, call.callee))
                    self.nodes[call.callee].called_by.append(name)
    
    def get_call_graph(self) -> Dict[str, Any]:
        """Get the complete call graph."""
        return {
            'nodes': {name: node.to_dict() for name, node in self.nodes.items()},
            'edges': self.edges,
            'stats': {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'function_types': self._count_function_types(),
            }
        }
    
    def _count_function_types(self) -> Dict[str, int]:
        """Count functions by type."""
        counts = defaultdict(int)
        for node in self.nodes.values():
            counts[node.function_type.value] += 1
        return dict(counts)
    
    def get_reachable_from(self, func_name: str) -> Set[str]:
        """Get all functions reachable from a given function."""
        if func_name not in self.nodes:
            return set()
        
        visited = set()
        stack = [func_name]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            if current in self.nodes:
                for call in self.nodes[current].calls:
                    if call.callee in self.nodes:
                        stack.append(call.callee)
        
        return visited
    
    def get_callers_of(self, func_name: str) -> Set[str]:
        """Get all functions that call a given function."""
        if func_name not in self.nodes:
            return set()
        
        return set(self.nodes[func_name].called_by)


class FunctionTestMapper:
    """
    Maps functions to their test cases.
    
    Uses multiple strategies:
    1. Name matching (function name in test file name)
    2. Instruction matching (instructions used in function appear in test)
    3. Module matching (test directory matches module)
    4. Content matching (function name appears in test content)
    """
    
    def __init__(self, call_graph: CallGraphBuilder):
        self.call_graph = call_graph
        self.mappings: Dict[str, List[str]] = {}
    
    def map_tests(self, test_cases: List[Any]) -> Dict[str, List[str]]:
        """Map functions to test cases."""
        from .test_parser import LLVMTestCase
        
        for func_name, node in self.call_graph.nodes.items():
            matched_tests = []
            
            for test_case in test_cases:
                if not isinstance(test_case, LLVMTestCase):
                    continue
                
                if self._is_test_relevant(node, test_case):
                    matched_tests.append(test_case.file_path)
            
            if matched_tests:
                self.mappings[func_name] = matched_tests
                node.test_files = matched_tests
        
        return self.mappings
    
    def _is_test_relevant(self, func_node: FunctionNode, test_case: Any) -> bool:
        """Check if a test case is relevant to a function."""
        # Strategy 1: Name matching
        func_name_lower = func_node.name.lower()
        if func_name_lower in test_case.file_path.lower():
            return True
        
        # Strategy 2: Instruction matching
        if func_node.instructions_used:
            common_insts = func_node.instructions_used & test_case.instructions_tested
            if len(common_insts) >= 2:  # At least 2 common instructions
                return True
        
        # Strategy 3: Function type matching
        if func_node.function_type == FunctionType.RELOCATION:
            if 'reloc' in test_case.file_path.lower():
                return True
        elif func_node.function_type == FunctionType.CODE_EMISSION:
            if 'mc/' in test_case.file_path.lower():
                return True
        elif func_node.function_type == FunctionType.INSTRUCTION_SELECTION:
            if 'codegen' in test_case.file_path.lower():
                return True
        
        return False
    
    def get_coverage_stats(self) -> Dict[str, Any]:
        """Get test coverage statistics."""
        total_functions = len(self.call_graph.nodes)
        functions_with_tests = len(self.mappings)
        
        # Count by function type
        type_coverage = defaultdict(lambda: {'total': 0, 'tested': 0})
        
        for func_name, node in self.call_graph.nodes.items():
            type_name = node.function_type.value
            type_coverage[type_name]['total'] += 1
            if func_name in self.mappings:
                type_coverage[type_name]['tested'] += 1
        
        return {
            'total_functions': total_functions,
            'functions_with_tests': functions_with_tests,
            'coverage_percentage': (functions_with_tests / total_functions * 100) if total_functions > 0 else 0,
            'by_type': {
                t: {
                    'total': d['total'],
                    'tested': d['tested'],
                    'percentage': (d['tested'] / d['total'] * 100) if d['total'] > 0 else 0,
                }
                for t, d in type_coverage.items()
            }
        }


class FunctionDatabaseEnhancer:
    """
    Enhances the function database with call graph and test mapping information.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path('data/llvm_functions_multi.json')
        self.enhanced_db_path = Path('data/llvm_functions_enhanced.json')
        self.data: Dict[str, Any] = {}
        self.call_graph = CallGraphBuilder()
    
    def load(self) -> bool:
        """Load the function database."""
        if self.db_path.exists():
            with open(self.db_path) as f:
                self.data = json.load(f)
            return True
        return False
    
    def enhance(self, backend: str) -> Dict[str, Any]:
        """Enhance functions for a backend with call graph analysis."""
        enhanced = {
            'backend': backend,
            'functions': {},
            'call_graph': None,
            'stats': {},
        }
        
        # Get functions from the correct location in the database
        # The database has a 'functions' key that contains all functions
        functions_data = self.data.get('functions', self.data)
        
        # Build call graph from function source code
        for func_id, func_data in functions_data.items():
            if not isinstance(func_data, dict):
                continue
            
            # Check if this function belongs to the requested backend
            func_backend = func_data.get('backend', '')
            if func_backend != backend:
                continue
            
            source_code = func_data.get('source_code', '') or func_data.get('body', '')
            module = func_data.get('module', 'unknown')
            
            # Use the actual function name if available, otherwise use the ID
            func_name = func_data.get('name', func_id)
            
            if source_code:
                node = self.call_graph.add_function(
                    name=func_name,
                    backend=backend,
                    module=module,
                    source_code=source_code,
                )
                enhanced['functions'][func_name] = {
                    'id': func_id,
                    **func_data,
                    'call_graph_node': node.to_dict(),
                }
        
        # Build edges
        self.call_graph.build_edges()
        
        # Get call graph
        enhanced['call_graph'] = self.call_graph.get_call_graph()
        
        # Stats
        enhanced['stats'] = {
            'total_functions': len(enhanced['functions']),
            'functions_with_calls': sum(1 for n in self.call_graph.nodes.values() if n.calls),
            'total_edges': len(self.call_graph.edges),
            'function_types': self.call_graph._count_function_types(),
        }
        
        return enhanced
    
    def save_enhanced(self, enhanced_data: Dict[str, Any], backend: str):
        """Save enhanced database."""
        output_path = Path(f'data/llvm_functions_{backend.lower()}_enhanced.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        return output_path


# Demo / Test
if __name__ == '__main__':
    print("=" * 70)
    print("LLVM Function Analyzer Demo")
    print("=" * 70)
    
    # Load function database
    enhancer = FunctionDatabaseEnhancer()
    
    if enhancer.load():
        print(f"\n📂 Loaded function database")
        
        # Enhance RISCV functions
        print("\n🔍 Analyzing RISCV functions...")
        enhanced = enhancer.enhance('RISCV')
        
        print(f"\n📊 Analysis Results:")
        print(f"   Total functions: {enhanced['stats']['total_functions']}")
        print(f"   Functions with calls: {enhanced['stats']['functions_with_calls']}")
        print(f"   Total call edges: {enhanced['stats']['total_edges']}")
        print(f"   Function types: {enhanced['stats']['function_types']}")
        
        # Show sample call graph
        call_graph = enhanced['call_graph']
        print(f"\n📈 Call Graph Nodes: {call_graph['stats']['total_nodes']}")
        
        # Show sample functions with their calls
        print("\n📌 Sample Functions with Calls:")
        count = 0
        for name, node_dict in call_graph['nodes'].items():
            if node_dict['stats']['call_count'] > 0 and count < 5:
                print(f"\n   {name} ({node_dict['type']}):")
                print(f"     Calls: {node_dict['stats']['call_count']}")
                print(f"     Called by: {node_dict['stats']['caller_count']}")
                print(f"     Instructions: {node_dict['stats']['instruction_count']}")
                
                # Show some calls
                for call in node_dict['calls'][:3]:
                    print(f"       -> {call['callee']}")
                
                count += 1
        
        # Test with test parser
        print("\n🧪 Testing Function-Test Mapping...")
        try:
            from test_parser import LLVMTestFetcher, LLVMTestCase
            
            fetcher = LLVMTestFetcher()
            test_cases = fetcher.fetch_backend_tests('RISCV', max_files=5)
            
            if test_cases:
                mapper = FunctionTestMapper(enhancer.call_graph)
                mappings = mapper.map_tests(test_cases)
                
                stats = mapper.get_coverage_stats()
                print(f"\n📊 Test Coverage:")
                print(f"   Total functions: {stats['total_functions']}")
                print(f"   Functions with tests: {stats['functions_with_tests']}")
                print(f"   Coverage: {stats['coverage_percentage']:.1f}%")
                
                print(f"\n   By function type:")
                for type_name, type_stats in stats['by_type'].items():
                    print(f"     {type_name}: {type_stats['tested']}/{type_stats['total']} ({type_stats['percentage']:.0f}%)")
                
                # Show some mappings
                print(f"\n📌 Sample Mappings:")
                for func_name, test_files in list(mappings.items())[:5]:
                    print(f"   {func_name}:")
                    for tf in test_files[:2]:
                        print(f"     -> {tf}")
        except ImportError as e:
            print(f"   (Skipping test mapping: {e})")
        
        # Save enhanced database
        output_path = enhancer.save_enhanced(enhanced, 'RISCV')
        print(f"\n💾 Saved enhanced database to: {output_path}")
    else:
        print("❌ Function database not found")
    
    print("\n" + "=" * 70)
    print("✅ LLVM Function Analyzer Demo Complete")
    print("=" * 70)
