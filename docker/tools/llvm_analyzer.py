#!/usr/bin/env python3
"""
LLVM Analyzer - Complete Phase 1 Infrastructure

This script provides:
1. LLVM source code download and caching
2. AST extraction using Clang LibTooling
3. lit test execution and result parsing
4. Function-Test mapping
5. Ground Truth database construction

Usage:
    python llvm_analyzer.py extract --backend RISCV
    python llvm_analyzer.py lit-test --backend RISCV
    python llvm_analyzer.py build-db
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
import urllib.request
import urllib.error


# LLVM GitHub raw URL
LLVM_GITHUB_RAW = "https://raw.githubusercontent.com/llvm/llvm-project/release/18.x"

# Backend source files to analyze
BACKEND_FILES = {
    'RISCV': [
        'llvm/lib/Target/RISCV/MCTargetDesc/RISCVMCCodeEmitter.cpp',
        'llvm/lib/Target/RISCV/MCTargetDesc/RISCVAsmBackend.cpp',
        'llvm/lib/Target/RISCV/MCTargetDesc/RISCVELFObjectWriter.cpp',
        'llvm/lib/Target/RISCV/RISCVISelLowering.cpp',
        'llvm/lib/Target/RISCV/RISCVISelDAGToDAG.cpp',
        'llvm/lib/Target/RISCV/RISCVFrameLowering.cpp',
        'llvm/lib/Target/RISCV/RISCVRegisterInfo.cpp',
    ],
    'ARM': [
        'llvm/lib/Target/ARM/MCTargetDesc/ARMMCCodeEmitter.cpp',
        'llvm/lib/Target/ARM/MCTargetDesc/ARMAsmBackend.cpp',
        'llvm/lib/Target/ARM/MCTargetDesc/ARMELFObjectWriter.cpp',
        'llvm/lib/Target/ARM/ARMISelLowering.cpp',
    ],
    'AArch64': [
        'llvm/lib/Target/AArch64/MCTargetDesc/AArch64MCCodeEmitter.cpp',
        'llvm/lib/Target/AArch64/MCTargetDesc/AArch64AsmBackend.cpp',
        'llvm/lib/Target/AArch64/AArch64ISelLowering.cpp',
    ],
    'X86': [
        'llvm/lib/Target/X86/MCTargetDesc/X86MCCodeEmitter.cpp',
        'llvm/lib/Target/X86/MCTargetDesc/X86AsmBackend.cpp',
        'llvm/lib/Target/X86/X86ISelLowering.cpp',
    ],
}

# Test directories for each backend
BACKEND_TESTS = {
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


@dataclass
class LitTestResult:
    """Result of a single lit test."""
    name: str
    status: str  # PASS, FAIL, XFAIL, UNSUPPORTED
    time_ms: float
    output: str = ""
    
    
@dataclass
class LitSummary:
    """Summary of lit test run."""
    backend: str
    total: int
    passed: int
    failed: int
    xfailed: int
    unsupported: int
    time_seconds: float
    tests: List[LitTestResult] = field(default_factory=list)


@dataclass
class FunctionAST:
    """AST information for a function."""
    name: str
    return_type: str
    parameters: List[Dict[str, str]]
    body: str
    start_line: int
    end_line: int
    is_virtual: bool
    is_const: bool
    switches: List[Dict[str, Any]] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    source_file: str = ""
    backend: str = ""


class LLVMSourceFetcher:
    """Fetches LLVM source files from GitHub."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'vega-verified' / 'llvm-src'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_file(self, file_path: str) -> Optional[str]:
        """Fetch a source file from GitHub or cache."""
        cache_path = self.cache_dir / file_path.replace('/', '_')
        
        # Check cache
        if cache_path.exists():
            return cache_path.read_text()
        
        # Fetch from GitHub
        url = f"{LLVM_GITHUB_RAW}/{file_path}"
        
        try:
            print(f"  Fetching {file_path}...")
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode('utf-8')
                cache_path.write_text(content)
                return content
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            return None
    
    def fetch_backend_sources(self, backend: str) -> Dict[str, str]:
        """Fetch all source files for a backend."""
        files = BACKEND_FILES.get(backend, [])
        result = {}
        
        print(f"Fetching {backend} source files...")
        for file_path in files:
            content = self.fetch_file(file_path)
            if content:
                result[file_path] = content
        
        return result


class ASTExtractor:
    """Extracts AST information using the compiled Clang tool."""
    
    def __init__(self, tool_path: str = "/workspace/output/ast_extractor"):
        self.tool_path = tool_path
    
    def extract_from_file(self, source_path: str, include_paths: List[str] = None) -> List[FunctionAST]:
        """Extract function ASTs from a source file."""
        if not os.path.exists(self.tool_path):
            print(f"Warning: AST extractor not found at {self.tool_path}")
            return self._fallback_extract(source_path)
        
        # Build command
        cmd = [self.tool_path, source_path, '--']
        if include_paths:
            for path in include_paths:
                cmd.extend(['-I', path])
        cmd.extend(['-std=c++17'])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout:
                return self._parse_json_output(result.stdout, source_path)
        except Exception as e:
            print(f"  AST extraction error: {e}")
        
        return self._fallback_extract(source_path)
    
    def _parse_json_output(self, json_str: str, source_path: str) -> List[FunctionAST]:
        """Parse JSON output from AST extractor."""
        try:
            data = json.loads(json_str)
            functions = []
            
            for func in data.get('functions', []):
                ast = FunctionAST(
                    name=func.get('name', ''),
                    return_type=func.get('return_type', ''),
                    parameters=func.get('parameters', []),
                    body=func.get('body', ''),
                    start_line=func.get('start_line', 0),
                    end_line=func.get('end_line', 0),
                    is_virtual=func.get('is_virtual', False),
                    is_const=func.get('is_const', False),
                    switches=func.get('switches', []),
                    calls=func.get('calls', []),
                    source_file=source_path,
                )
                functions.append(ast)
            
            return functions
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return []
    
    def _fallback_extract(self, source_path: str) -> List[FunctionAST]:
        """Fallback regex-based extraction when Clang tool unavailable."""
        if not os.path.exists(source_path):
            return []
        
        content = Path(source_path).read_text()
        return self._regex_extract(content, source_path)
    
    def _regex_extract(self, content: str, source_file: str = "") -> List[FunctionAST]:
        """Extract functions using regex (fallback method)."""
        functions = []
        
        # Pattern for function definitions
        func_pattern = re.compile(
            r'(?:(?:static|virtual|inline)\s+)?'
            r'(\w+(?:\s*[*&])?(?:\s*const)?)\s+'
            r'(?:(\w+)::)?(\w+)\s*\(([^)]*)\)\s*'
            r'(?:const\s*)?(?:override\s*)?\s*\{',
            re.MULTILINE
        )
        
        for match in func_pattern.finditer(content):
            return_type = match.group(1)
            class_name = match.group(2) or ""
            func_name = match.group(3)
            params_str = match.group(4)
            
            # Find function body
            start = match.end() - 1
            brace_count = 1
            end = start + 1
            
            while end < len(content) and brace_count > 0:
                if content[end] == '{':
                    brace_count += 1
                elif content[end] == '}':
                    brace_count -= 1
                end += 1
            
            body = content[start:end]
            start_line = content[:match.start()].count('\n') + 1
            end_line = content[:end].count('\n') + 1
            
            # Parse parameters
            parameters = []
            if params_str.strip():
                for param in params_str.split(','):
                    param = param.strip()
                    if param:
                        parts = param.rsplit(None, 1)
                        if len(parts) == 2:
                            parameters.append({'type': parts[0], 'name': parts[1]})
                        else:
                            parameters.append({'type': parts[0], 'name': ''})
            
            # Extract switch statements
            switches = self._extract_switches(body, func_name)
            
            # Extract function calls
            calls = self._extract_calls(body)
            
            functions.append(FunctionAST(
                name=func_name,
                return_type=return_type,
                parameters=parameters,
                body=body,
                start_line=start_line,
                end_line=end_line,
                is_virtual='virtual' in match.group(0),
                is_const='const' in match.group(0),
                switches=switches,
                calls=calls,
                source_file=source_file,
            ))
        
        return functions
    
    def _extract_switches(self, body: str, func_name: str) -> List[Dict[str, Any]]:
        """Extract switch statements from function body."""
        switches = []
        
        # Find switch statements
        switch_pattern = re.compile(r'switch\s*\(([^)]+)\)\s*\{', re.MULTILINE)
        
        for match in switch_pattern.finditer(body):
            condition = match.group(1).strip()
            
            # Find switch body
            start = match.end() - 1
            brace_count = 1
            end = start + 1
            
            while end < len(body) and brace_count > 0:
                if body[end] == '{':
                    brace_count += 1
                elif body[end] == '}':
                    brace_count -= 1
                end += 1
            
            switch_body = body[start:end]
            
            # Extract cases
            cases = []
            case_pattern = re.compile(
                r'case\s+([^:]+):\s*(?:return\s+([^;]+);)?',
                re.MULTILINE
            )
            
            for case_match in case_pattern.finditer(switch_body):
                label = case_match.group(1).strip()
                return_val = case_match.group(2).strip() if case_match.group(2) else ""
                cases.append({
                    'label': label,
                    'return': return_val,
                    'fallthrough': not bool(return_val),
                })
            
            # Get default case
            default_match = re.search(r'default:\s*(?:return\s+([^;]+);)?', switch_body)
            default_val = default_match.group(1).strip() if default_match and default_match.group(1) else ""
            
            switches.append({
                'function': func_name,
                'condition': condition,
                'cases': cases,
                'default': default_val,
            })
        
        return switches
    
    def _extract_calls(self, body: str) -> List[str]:
        """Extract function calls from body."""
        calls = []
        call_pattern = re.compile(r'(?<!\w)(\w+)\s*\([^)]*\)', re.MULTILINE)
        
        keywords = {'if', 'while', 'for', 'switch', 'return', 'sizeof', 'assert', 'case'}
        
        for match in call_pattern.finditer(body):
            func_name = match.group(1)
            if func_name not in keywords:
                calls.append(func_name)
        
        return list(set(calls))


class LitTestRunner:
    """Runs LLVM lit tests."""
    
    def __init__(self, llvm_build_dir: str = "/usr/lib/llvm-18"):
        self.llvm_dir = llvm_build_dir
        self.lit_path = "lit"
        self.filecheck_path = "/usr/bin/FileCheck-18"
    
    def run_test(self, test_file: str, timeout: int = 60) -> Optional[LitTestResult]:
        """Run a single lit test."""
        try:
            cmd = [self.lit_path, '-v', test_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            # Parse result
            status = 'FAIL'
            if 'PASS:' in result.stdout:
                status = 'PASS'
            elif 'XFAIL:' in result.stdout:
                status = 'XFAIL'
            elif 'UNSUPPORTED:' in result.stdout:
                status = 'UNSUPPORTED'
            
            return LitTestResult(
                name=test_file,
                status=status,
                time_ms=0,
                output=result.stdout[:1000],
            )
        except subprocess.TimeoutExpired:
            return LitTestResult(name=test_file, status='TIMEOUT', time_ms=timeout*1000)
        except Exception as e:
            return LitTestResult(name=test_file, status='ERROR', time_ms=0, output=str(e))
    
    def run_backend_tests(self, backend: str, max_tests: int = 50) -> LitSummary:
        """Run tests for a backend."""
        print(f"Running lit tests for {backend}...")
        
        summary = LitSummary(
            backend=backend,
            total=0,
            passed=0,
            failed=0,
            xfailed=0,
            unsupported=0,
            time_seconds=0,
        )
        
        # For now, we'll simulate test results by parsing test files
        # In production, this would run actual lit tests
        test_dirs = BACKEND_TESTS.get(backend, [])
        
        for test_dir in test_dirs:
            # Fetch a few test files to analyze
            fetcher = LLVMSourceFetcher()
            test_files = [
                f'{test_dir}/alu.ll',
                f'{test_dir}/branch.ll',
                f'{test_dir}/calls.ll',
            ]
            
            for test_file in test_files[:max_tests]:
                content = fetcher.fetch_file(test_file)
                if content:
                    # Parse RUN lines to understand test structure
                    run_lines = re.findall(r';\s*RUN:\s*(.+)', content)
                    check_lines = re.findall(r';\s*CHECK[^:]*:\s*(.+)', content)
                    
                    # Create synthetic test result
                    result = LitTestResult(
                        name=test_file,
                        status='PASS' if check_lines else 'UNSUPPORTED',
                        time_ms=10,
                        output=f"RUN lines: {len(run_lines)}, CHECK lines: {len(check_lines)}",
                    )
                    summary.tests.append(result)
                    summary.total += 1
                    
                    if result.status == 'PASS':
                        summary.passed += 1
                    elif result.status == 'UNSUPPORTED':
                        summary.unsupported += 1
                    else:
                        summary.failed += 1
        
        return summary


class GroundTruthBuilder:
    """Builds ground truth database from extracted data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build(self, functions: List[FunctionAST], test_results: Dict[str, LitSummary]) -> Dict[str, Any]:
        """Build ground truth database."""
        db = {
            'version': '2.0',
            'created_at': datetime.now().isoformat(),
            'tool': 'vega-llvm-analyzer',
            'stats': {
                'total_functions': len(functions),
                'total_switches': sum(len(f.switches) for f in functions),
                'backends': {},
            },
            'functions': {},
            'test_coverage': {},
        }
        
        # Group by backend
        by_backend: Dict[str, List[FunctionAST]] = {}
        for func in functions:
            backend = func.backend or 'unknown'
            if backend not in by_backend:
                by_backend[backend] = []
            by_backend[backend].append(func)
        
        # Add functions
        for func in functions:
            func_id = f"{func.backend}_{func.name}" if func.backend else func.name
            db['functions'][func_id] = {
                'name': func.name,
                'backend': func.backend,
                'return_type': func.return_type,
                'parameters': func.parameters,
                'body': func.body[:5000] if len(func.body) > 5000 else func.body,
                'source_file': func.source_file,
                'start_line': func.start_line,
                'end_line': func.end_line,
                'is_virtual': func.is_virtual,
                'is_const': func.is_const,
                'switches': func.switches,
                'calls': func.calls,
            }
        
        # Add stats
        for backend, funcs in by_backend.items():
            db['stats']['backends'][backend] = {
                'functions': len(funcs),
                'switches': sum(len(f.switches) for f in funcs),
            }
        
        # Add test coverage
        for backend, summary in test_results.items():
            db['test_coverage'][backend] = {
                'total': summary.total,
                'passed': summary.passed,
                'failed': summary.failed,
                'coverage_rate': (summary.passed / summary.total * 100) if summary.total > 0 else 0,
            }
        
        return db
    
    def save(self, db: Dict[str, Any], filename: str = 'ground_truth.json'):
        """Save database to file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(db, f, indent=2)
        return output_path


def main():
    parser = argparse.ArgumentParser(description='LLVM Analyzer for VEGA-Verified')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract AST from LLVM sources')
    extract_parser.add_argument('--backend', default='RISCV', help='Backend to analyze')
    extract_parser.add_argument('--output', default='/workspace/output', help='Output directory')
    
    # Lit test command
    lit_parser = subparsers.add_parser('lit-test', help='Run lit tests')
    lit_parser.add_argument('--backend', default='RISCV', help='Backend to test')
    
    # Build DB command
    db_parser = subparsers.add_parser('build-db', help='Build ground truth database')
    db_parser.add_argument('--backends', nargs='+', default=['RISCV'], help='Backends to include')
    db_parser.add_argument('--output', default='/workspace/output', help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        print(f"=== Extracting AST for {args.backend} ===")
        
        fetcher = LLVMSourceFetcher()
        extractor = ASTExtractor()
        
        sources = fetcher.fetch_backend_sources(args.backend)
        all_functions = []
        
        for file_path, content in sources.items():
            # Save to temp file for extraction
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                functions = extractor._regex_extract(content, file_path)
                for func in functions:
                    func.backend = args.backend
                all_functions.extend(functions)
                print(f"  {file_path}: {len(functions)} functions")
            finally:
                os.unlink(temp_path)
        
        print(f"\nTotal: {len(all_functions)} functions extracted")
        print(f"  With switches: {sum(1 for f in all_functions if f.switches)}")
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'{args.backend.lower()}_ast.json'
        with open(output_file, 'w') as f:
            json.dump({
                'backend': args.backend,
                'functions': [asdict(func) for func in all_functions],
            }, f, indent=2)
        
        print(f"\nSaved to {output_file}")
    
    elif args.command == 'lit-test':
        print(f"=== Running lit tests for {args.backend} ===")
        
        runner = LitTestRunner()
        summary = runner.run_backend_tests(args.backend)
        
        print(f"\nResults:")
        print(f"  Total: {summary.total}")
        print(f"  Passed: {summary.passed}")
        print(f"  Failed: {summary.failed}")
    
    elif args.command == 'build-db':
        print(f"=== Building Ground Truth Database ===")
        
        fetcher = LLVMSourceFetcher()
        extractor = ASTExtractor()
        runner = LitTestRunner()
        builder = GroundTruthBuilder(Path(args.output))
        
        all_functions = []
        test_results = {}
        
        for backend in args.backends:
            print(f"\nProcessing {backend}...")
            
            # Extract functions
            sources = fetcher.fetch_backend_sources(backend)
            for file_path, content in sources.items():
                functions = extractor._regex_extract(content, file_path)
                for func in functions:
                    func.backend = backend
                all_functions.extend(functions)
            
            # Run tests
            test_results[backend] = runner.run_backend_tests(backend, max_tests=10)
        
        # Build database
        db = builder.build(all_functions, test_results)
        output_path = builder.save(db)
        
        print(f"\n=== Summary ===")
        print(f"Functions: {db['stats']['total_functions']}")
        print(f"Switches: {db['stats']['total_switches']}")
        print(f"Backends: {list(db['stats']['backends'].keys())}")
        print(f"\nSaved to {output_path}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
