"""
LLVM Test Infrastructure Parser for VEGA-Verified.

This module parses LLVM's lit-based test files to:
1. Extract test cases and their RUN lines
2. Parse FileCheck directives and patterns
3. Map tests to functions being tested
4. Analyze test coverage

LLVM Test Structure:
- test/CodeGen/{Target}/*.ll - CodeGen tests (LLVM IR -> Assembly)
- test/MC/{Target}/*.s - Assembly tests (Assembly -> Binary)
- test/Transforms/*.ll - IR transformation tests

Key directives:
- RUN: - Command to execute
- CHECK: - Pattern to match in output
- CHECK-LABEL: - Named pattern anchor
- CHECK-NEXT: - Pattern on next line
- CHECK-NOT: - Pattern must not appear
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import re
import json
from pathlib import Path
import urllib.request
import urllib.error


class CheckDirective(Enum):
    """FileCheck directive types."""
    CHECK = "CHECK"
    CHECK_LABEL = "CHECK-LABEL"
    CHECK_NEXT = "CHECK-NEXT"
    CHECK_NOT = "CHECK-NOT"
    CHECK_SAME = "CHECK-SAME"
    CHECK_DAG = "CHECK-DAG"
    CHECK_COUNT = "CHECK-COUNT"


@dataclass
class RunLine:
    """A RUN line from a test file."""
    command: str
    tool: str  # llc, opt, clang, llvm-mc, etc.
    options: List[str] = field(default_factory=list)
    input_file: str = "%s"
    output_pipe: Optional[str] = None
    check_prefix: str = "CHECK"
    triple: Optional[str] = None
    cpu: Optional[str] = None
    features: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'command': self.command,
            'tool': self.tool,
            'options': self.options,
            'triple': self.triple,
            'cpu': self.cpu,
            'features': self.features,
            'check_prefix': self.check_prefix,
        }


@dataclass
class CheckPattern:
    """A FileCheck pattern."""
    directive: CheckDirective
    prefix: str
    pattern: str
    line_number: int
    original_line: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'directive': self.directive.value,
            'prefix': self.prefix,
            'pattern': self.pattern,
            'line': self.line_number,
        }


@dataclass
class TestFunction:
    """A function definition in a test file."""
    name: str
    return_type: str
    parameters: List[str]
    body: str
    line_number: int
    check_patterns: List[CheckPattern] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'return_type': self.return_type,
            'parameters': self.parameters,
            'line': self.line_number,
            'checks': [c.to_dict() for c in self.check_patterns],
        }


@dataclass
class LLVMTestCase:
    """A parsed LLVM test case."""
    file_path: str
    description: str
    run_lines: List[RunLine] = field(default_factory=list)
    check_patterns: List[CheckPattern] = field(default_factory=list)
    functions: List[TestFunction] = field(default_factory=list)
    backend: Optional[str] = None
    test_type: str = "unknown"  # codegen, mc, transforms, etc.
    instructions_tested: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file': self.file_path,
            'description': self.description,
            'backend': self.backend,
            'type': self.test_type,
            'run_lines': [r.to_dict() for r in self.run_lines],
            'functions': [f.to_dict() for f in self.functions],
            'instructions_tested': list(self.instructions_tested),
            'stats': {
                'run_lines': len(self.run_lines),
                'functions': len(self.functions),
                'check_patterns': len(self.check_patterns),
            }
        }


class LLVMTestParser:
    """
    Parser for LLVM lit-based test files.
    
    Extracts:
    - RUN lines with tool and options
    - CHECK patterns for FileCheck
    - Function definitions and their tests
    - Instructions being tested
    """
    
    # Pattern to match RUN lines
    RUN_PATTERN = re.compile(
        r'^;\s*RUN:\s*(.+?)(?:\s*\\)?$',
        re.MULTILINE
    )
    
    # Pattern to match multi-line RUN continuations
    RUN_CONTINUATION = re.compile(
        r'^;\s*RUN:\s*(.+?)(?:\s*\\)?$',
        re.MULTILINE
    )
    
    # Pattern to match CHECK directives
    CHECK_PATTERN = re.compile(
        r'^;\s*(CHECK(?:-[A-Z]+)?(?:-[A-Z0-9]+)?(?:\{[^}]+\})?)\s*:\s*(.*)$',
        re.MULTILINE
    )
    
    # Pattern to match LLVM IR function definitions
    IR_FUNCTION_PATTERN = re.compile(
        r'^define\s+(\S+)\s+@(\w+)\s*\(([^)]*)\)[^{]*\{',
        re.MULTILINE
    )
    
    # Pattern to match assembly labels (function names in CHECK output)
    ASM_LABEL_PATTERN = re.compile(
        r'([a-z_][a-z0-9_]*):',
        re.IGNORECASE
    )
    
    # RISCV instruction pattern
    RISCV_INST_PATTERN = re.compile(
        r'\b(add|sub|mul|div|rem|and|or|xor|sll|srl|sra|slt|sltu|'
        r'lb|lh|lw|ld|sb|sh|sw|sd|'
        r'beq|bne|blt|bge|bltu|bgeu|jal|jalr|'
        r'lui|auipc|'
        r'addi|slti|sltiu|xori|ori|andi|slli|srli|srai|'
        r'addiw|slliw|srliw|sraiw|addw|subw|sllw|srlw|sraw|'
        r'fadd|fsub|fmul|fdiv|fsqrt|fmin|fmax|'
        r'flw|fsw|fld|fsd|'
        r'fcvt|fmv|feq|flt|fle|fclass|fsgnj|fsgnjn|fsgnjx|'
        r'lr|sc|amoswap|amoadd|amoxor|amoand|amoor|amomin|amomax|amominu|amomaxu|'
        r'csrr|csrw|csrs|csrc|csrrw|csrrs|csrrc|'
        r'ecall|ebreak|fence|sfence|mret|sret|wfi)\b',
        re.IGNORECASE
    )
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def parse(self, content: str, file_path: str = "") -> LLVMTestCase:
        """Parse a test file content."""
        test_case = LLVMTestCase(
            file_path=file_path,
            description=self._extract_description(content)
        )
        
        # Determine test type and backend
        test_case.test_type = self._detect_test_type(file_path)
        test_case.backend = self._detect_backend(file_path, content)
        
        # Parse RUN lines
        test_case.run_lines = self._parse_run_lines(content)
        
        # Parse CHECK patterns
        test_case.check_patterns = self._parse_check_patterns(content)
        
        # Parse functions
        test_case.functions = self._parse_functions(content, test_case.check_patterns)
        
        # Extract tested instructions
        test_case.instructions_tested = self._extract_instructions(content)
        
        return test_case
    
    def _extract_description(self, content: str) -> str:
        """Extract test description from comments."""
        # Look for NOTE or description comments
        note_match = re.search(r'^;\s*NOTE:\s*(.+)$', content, re.MULTILINE)
        if note_match:
            return note_match.group(1).strip()
        
        # Look for any leading comment
        comment_match = re.search(r'^;\s*([^;R].+)$', content, re.MULTILINE)
        if comment_match:
            return comment_match.group(1).strip()
        
        return ""
    
    def _detect_test_type(self, file_path: str) -> str:
        """Detect test type from file path."""
        if 'CodeGen' in file_path:
            return 'codegen'
        elif 'MC' in file_path:
            return 'mc'
        elif 'Transforms' in file_path:
            return 'transforms'
        elif 'Analysis' in file_path:
            return 'analysis'
        return 'unknown'
    
    def _detect_backend(self, file_path: str, content: str) -> Optional[str]:
        """Detect backend from file path or content."""
        # From file path
        backends = ['RISCV', 'ARM', 'AArch64', 'X86', 'Mips', 'PowerPC', 'SystemZ']
        for backend in backends:
            if backend in file_path or backend.lower() in file_path:
                return backend
        
        # From triple in RUN lines
        triple_match = re.search(r'-mtriple=(\w+)', content)
        if triple_match:
            triple = triple_match.group(1)
            if 'riscv' in triple.lower():
                return 'RISCV'
            elif 'arm' in triple.lower():
                return 'ARM'
            elif 'aarch64' in triple.lower():
                return 'AArch64'
            elif 'x86' in triple.lower():
                return 'X86'
        
        return None
    
    def _parse_run_lines(self, content: str) -> List[RunLine]:
        """Parse RUN lines from test content."""
        run_lines = []
        lines = content.split('\n')
        current_run = []
        
        for line in lines:
            # Check for RUN line
            run_match = re.match(r'^;\s*RUN:\s*(.+?)(\s*\\)?$', line)
            if run_match:
                cmd_part = run_match.group(1)
                is_continuation = run_match.group(2) is not None
                
                current_run.append(cmd_part)
                
                if not is_continuation and current_run:
                    full_cmd = ' '.join(current_run)
                    run_line = self._parse_single_run_line(full_cmd)
                    if run_line:
                        run_lines.append(run_line)
                    current_run = []
            elif current_run and line.strip().startswith(';'):
                # Continuation line
                cont_match = re.match(r'^;\s*(.+?)(\s*\\)?$', line)
                if cont_match:
                    current_run.append(cont_match.group(1))
                    if not cont_match.group(2):
                        full_cmd = ' '.join(current_run)
                        run_line = self._parse_single_run_line(full_cmd)
                        if run_line:
                            run_lines.append(run_line)
                        current_run = []
        
        return run_lines
    
    def _parse_single_run_line(self, command: str) -> Optional[RunLine]:
        """Parse a single RUN command."""
        # Determine the tool
        tools = ['llc', 'opt', 'clang', 'llvm-mc', 'not', 'FileCheck']
        tool = 'unknown'
        
        for t in tools:
            if t in command:
                tool = t
                break
        
        # Extract triple
        triple_match = re.search(r'-mtriple=([^\s]+)', command)
        triple = triple_match.group(1) if triple_match else None
        
        # Extract CPU
        cpu_match = re.search(r'-mcpu=([^\s]+)', command)
        cpu = cpu_match.group(1) if cpu_match else None
        
        # Extract features
        features = []
        for feat_match in re.finditer(r'-mattr=\+([^\s,]+)', command):
            features.append(feat_match.group(1))
        
        # Extract check prefix
        prefix_match = re.search(r'-check-prefix(?:es)?=([^\s]+)', command)
        check_prefix = prefix_match.group(1) if prefix_match else "CHECK"
        
        # Extract options
        options = re.findall(r'-([a-zA-Z-]+)(?:=([^\s]+))?', command)
        
        return RunLine(
            command=command,
            tool=tool,
            options=[f"-{opt[0]}={opt[1]}" if opt[1] else f"-{opt[0]}" for opt in options],
            triple=triple,
            cpu=cpu,
            features=features,
            check_prefix=check_prefix,
        )
    
    def _parse_check_patterns(self, content: str) -> List[CheckPattern]:
        """Parse CHECK patterns from test content."""
        patterns = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Match CHECK patterns
            check_match = re.match(r'^;\s*(\w+(?:-\w+)?)\s*:\s*(.*)$', line)
            if check_match:
                prefix_directive = check_match.group(1)
                pattern_text = check_match.group(2)
                
                # Determine directive type
                directive = CheckDirective.CHECK
                prefix = prefix_directive
                
                if '-LABEL' in prefix_directive:
                    directive = CheckDirective.CHECK_LABEL
                    prefix = prefix_directive.replace('-LABEL', '')
                elif '-NEXT' in prefix_directive:
                    directive = CheckDirective.CHECK_NEXT
                    prefix = prefix_directive.replace('-NEXT', '')
                elif '-NOT' in prefix_directive:
                    directive = CheckDirective.CHECK_NOT
                    prefix = prefix_directive.replace('-NOT', '')
                elif '-SAME' in prefix_directive:
                    directive = CheckDirective.CHECK_SAME
                    prefix = prefix_directive.replace('-SAME', '')
                elif '-DAG' in prefix_directive:
                    directive = CheckDirective.CHECK_DAG
                    prefix = prefix_directive.replace('-DAG', '')
                
                patterns.append(CheckPattern(
                    directive=directive,
                    prefix=prefix,
                    pattern=pattern_text,
                    line_number=line_num,
                    original_line=line,
                ))
        
        return patterns
    
    def _parse_functions(self, content: str, check_patterns: List[CheckPattern]) -> List[TestFunction]:
        """Parse function definitions from IR content."""
        functions = []
        
        for match in self.IR_FUNCTION_PATTERN.finditer(content):
            ret_type = match.group(1)
            name = match.group(2)
            params = match.group(3)
            
            # Find function body
            start = match.end() - 1  # Include opening brace
            brace_count = 1
            end = start + 1
            
            while end < len(content) and brace_count > 0:
                if content[end] == '{':
                    brace_count += 1
                elif content[end] == '}':
                    brace_count -= 1
                end += 1
            
            body = content[start:end]
            line_number = content[:match.start()].count('\n') + 1
            
            # Find CHECK patterns for this function
            func_checks = [p for p in check_patterns if name in p.pattern]
            
            functions.append(TestFunction(
                name=name,
                return_type=ret_type,
                parameters=[p.strip() for p in params.split(',') if p.strip()],
                body=body,
                line_number=line_number,
                check_patterns=func_checks,
            ))
        
        return functions
    
    def _extract_instructions(self, content: str) -> Set[str]:
        """Extract RISCV instructions being tested."""
        instructions = set()
        
        for match in self.RISCV_INST_PATTERN.finditer(content):
            instructions.add(match.group(1).lower())
        
        return instructions


class LLVMTestFetcher:
    """Fetches LLVM test files from GitHub."""
    
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/llvm/llvm-project/release/18.x"
    
    # Key test directories per backend
    TEST_PATHS = {
        'RISCV': [
            'llvm/test/CodeGen/RISCV',
            'llvm/test/MC/RISCV',
        ],
        'ARM': [
            'llvm/test/CodeGen/ARM',
            'llvm/test/MC/ARM',
        ],
        'AArch64': [
            'llvm/test/CodeGen/AArch64',
            'llvm/test/MC/AArch64',
        ],
        'X86': [
            'llvm/test/CodeGen/X86',
            'llvm/test/MC/X86',
        ],
    }
    
    # Known test files for each backend (sampling)
    KNOWN_TEST_FILES = {
        'RISCV': [
            'double-arith.ll',
            'float-arith.ll',
            'alu32.ll',
            'alu64.ll',
            'branch.ll',
            'calls.ll',
            'mem.ll',
            'atomic.ll',
            'compress.ll',
            'zbb.ll',
            'rvv/vsetvli.ll',
        ],
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'vega-verified' / 'tests'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.parser = LLVMTestParser()
    
    def fetch_test_file(self, backend: str, test_path: str, filename: str) -> Optional[str]:
        """Fetch a single test file."""
        # Check cache
        cache_path = self.cache_dir / backend / test_path.replace('/', '_') / filename
        if cache_path.exists():
            return cache_path.read_text()
        
        # Fetch from GitHub
        url = f"{self.GITHUB_RAW_URL}/{test_path}/{filename}"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode('utf-8')
                
                # Cache
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(content)
                
                return content
        except Exception as e:
            if self.parser.verbose:
                print(f"Error fetching {url}: {e}")
            return None
    
    def fetch_backend_tests(self, backend: str, max_files: int = 10) -> List[LLVMTestCase]:
        """Fetch and parse test files for a backend."""
        test_cases = []
        files_fetched = 0
        
        # Try known test files first
        known_files = self.KNOWN_TEST_FILES.get(backend, [])
        test_paths = self.TEST_PATHS.get(backend, [])
        
        for test_path in test_paths:
            if files_fetched >= max_files:
                break
            
            for filename in known_files:
                if files_fetched >= max_files:
                    break
                
                # Handle subdirectory files
                if '/' in filename:
                    subdir, fname = filename.rsplit('/', 1)
                    full_path = f"{test_path}/{subdir}"
                    filename = fname
                else:
                    full_path = test_path
                
                content = self.fetch_test_file(backend, full_path, filename)
                if content:
                    test_case = self.parser.parse(content, f"{full_path}/{filename}")
                    test_cases.append(test_case)
                    files_fetched += 1
        
        return test_cases


@dataclass
class FunctionTestMapping:
    """Mapping between a backend function and its tests."""
    function_name: str
    backend: str
    test_files: List[str] = field(default_factory=list)
    instructions_covered: Set[str] = field(default_factory=set)
    coverage_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'function': self.function_name,
            'backend': self.backend,
            'test_files': self.test_files,
            'instructions': list(self.instructions_covered),
            'coverage': self.coverage_score,
        }


class TestCoverageAnalyzer:
    """Analyzes test coverage for backend functions."""
    
    def __init__(self, function_db_path: Optional[Path] = None):
        self.function_db_path = function_db_path or Path('data/llvm_functions_multi.json')
        self.function_db: Dict[str, Any] = {}
        self.test_cases: List[LLVMTestCase] = []
        self.mappings: Dict[str, FunctionTestMapping] = {}
    
    def load_function_db(self) -> bool:
        """Load function database."""
        if self.function_db_path.exists():
            with open(self.function_db_path) as f:
                self.function_db = json.load(f)
            return True
        return False
    
    def add_test_cases(self, test_cases: List[LLVMTestCase]):
        """Add test cases for analysis."""
        self.test_cases.extend(test_cases)
    
    def analyze_coverage(self, backend: str) -> Dict[str, Any]:
        """Analyze test coverage for a backend."""
        # Get backend functions
        backend_functions = self._get_backend_functions(backend)
        
        # Get all instructions tested
        all_tested_instructions: Set[str] = set()
        for tc in self.test_cases:
            if tc.backend == backend:
                all_tested_instructions.update(tc.instructions_tested)
        
        # Analyze each function
        results = {
            'backend': backend,
            'total_functions': len(backend_functions),
            'functions_with_tests': 0,
            'total_instructions_tested': len(all_tested_instructions),
            'instructions_tested': list(all_tested_instructions),
            'function_coverage': {},
            'untested_functions': [],
        }
        
        for func_name, func_data in backend_functions.items():
            # Find tests that might cover this function
            related_tests = self._find_related_tests(func_name, backend)
            
            if related_tests:
                results['functions_with_tests'] += 1
                results['function_coverage'][func_name] = {
                    'tests': [t.file_path for t in related_tests],
                    'instructions': list(set.union(*[t.instructions_tested for t in related_tests]) if related_tests else set()),
                }
            else:
                results['untested_functions'].append(func_name)
        
        # Calculate coverage percentage
        if results['total_functions'] > 0:
            results['coverage_percentage'] = (results['functions_with_tests'] / results['total_functions']) * 100
        else:
            results['coverage_percentage'] = 0.0
        
        return results
    
    def _get_backend_functions(self, backend: str) -> Dict[str, Any]:
        """Get functions for a specific backend."""
        functions = {}
        
        for func_name, func_data in self.function_db.items():
            if isinstance(func_data, dict) and func_data.get('backend') == backend:
                functions[func_name] = func_data
        
        return functions
    
    def _find_related_tests(self, func_name: str, backend: str) -> List[LLVMTestCase]:
        """Find tests related to a function."""
        related = []
        
        # Simple heuristic: function name appears in test or related instructions
        func_name_lower = func_name.lower()
        
        for tc in self.test_cases:
            if tc.backend != backend:
                continue
            
            # Check if function name appears in test
            if func_name_lower in tc.file_path.lower():
                related.append(tc)
                continue
            
            # Check instructions
            # This is a simplified heuristic - real implementation would need more sophisticated analysis
            if any(inst in func_name_lower for inst in tc.instructions_tested):
                related.append(tc)
        
        return related
    
    def generate_report(self, backend: str) -> str:
        """Generate a coverage report."""
        analysis = self.analyze_coverage(backend)
        
        report = [
            f"=" * 70,
            f"Test Coverage Report: {backend}",
            f"=" * 70,
            f"",
            f"Summary:",
            f"  Total Functions: {analysis['total_functions']}",
            f"  Functions with Tests: {analysis['functions_with_tests']}",
            f"  Coverage: {analysis['coverage_percentage']:.1f}%",
            f"  Instructions Tested: {analysis['total_instructions_tested']}",
            f"",
        ]
        
        if analysis['instructions_tested']:
            report.append("Tested Instructions:")
            for inst in sorted(analysis['instructions_tested'])[:20]:
                report.append(f"  - {inst}")
            if len(analysis['instructions_tested']) > 20:
                report.append(f"  ... and {len(analysis['instructions_tested']) - 20} more")
        
        report.append("")
        
        if analysis['untested_functions']:
            report.append(f"Untested Functions ({len(analysis['untested_functions'])}):")
            for func in analysis['untested_functions'][:10]:
                report.append(f"  - {func}")
            if len(analysis['untested_functions']) > 10:
                report.append(f"  ... and {len(analysis['untested_functions']) - 10} more")
        
        return "\n".join(report)


class TestDatabase:
    """Database for storing parsed test information."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path('data/test_database.json')
        self.data: Dict[str, Any] = {
            'version': '1.0',
            'backends': {},
            'stats': {},
        }
    
    def add_test_cases(self, backend: str, test_cases: List[LLVMTestCase]):
        """Add test cases for a backend."""
        if backend not in self.data['backends']:
            self.data['backends'][backend] = {
                'tests': [],
                'stats': {},
            }
        
        for tc in test_cases:
            self.data['backends'][backend]['tests'].append(tc.to_dict())
        
        # Update stats
        all_tests = self.data['backends'][backend]['tests']
        all_instructions = set()
        for t in all_tests:
            all_instructions.update(t.get('instructions_tested', []))
        
        self.data['backends'][backend]['stats'] = {
            'total_tests': len(all_tests),
            'total_functions': sum(t['stats']['functions'] for t in all_tests),
            'total_instructions': len(all_instructions),
            'instructions': list(all_instructions),
        }
    
    def save(self):
        """Save database to file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def load(self) -> bool:
        """Load database from file."""
        if self.db_path.exists():
            with open(self.db_path) as f:
                self.data = json.load(f)
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'backends': list(self.data.get('backends', {}).keys()),
            'per_backend': {
                backend: data.get('stats', {})
                for backend, data in self.data.get('backends', {}).items()
            }
        }


# Demo / Test
if __name__ == '__main__':
    print("=" * 70)
    print("LLVM Test Parser Demo")
    print("=" * 70)
    
    # Fetch and parse RISCV tests
    fetcher = LLVMTestFetcher()
    print("\n📥 Fetching RISCV test files...")
    test_cases = fetcher.fetch_backend_tests('RISCV', max_files=5)
    
    print(f"\n📊 Parsed {len(test_cases)} test files:")
    for tc in test_cases:
        print(f"\n  📄 {tc.file_path}")
        print(f"     Type: {tc.test_type}")
        print(f"     RUN lines: {len(tc.run_lines)}")
        print(f"     Functions: {len(tc.functions)}")
        print(f"     Instructions: {len(tc.instructions_tested)}")
        
        if tc.run_lines:
            print(f"     Sample RUN: {tc.run_lines[0].tool} (triple={tc.run_lines[0].triple})")
        
        if tc.instructions_tested:
            sample_insts = list(tc.instructions_tested)[:5]
            print(f"     Sample instructions: {', '.join(sample_insts)}")
    
    # Save to database
    print("\n💾 Saving to database...")
    db = TestDatabase()
    db.add_test_cases('RISCV', test_cases)
    db.save()
    
    print(f"\n📊 Database stats: {db.get_stats()}")
    
    # Coverage analysis
    print("\n📈 Coverage Analysis:")
    analyzer = TestCoverageAnalyzer()
    if analyzer.load_function_db():
        analyzer.add_test_cases(test_cases)
        report = analyzer.generate_report('RISCV')
        print(report)
    else:
        print("  (Function database not found, skipping coverage analysis)")
    
    print("\n" + "=" * 70)
    print("✅ LLVM Test Parser Demo Complete")
    print("=" * 70)
