"""
Benchmark Adapter for LLVM Extracted Functions

This module bridges the extracted LLVM functions with the existing
VEGA-Verified benchmark system, replacing mock data with real functions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .database import FunctionDatabase


@dataclass
class RealLLVMBenchmark:
    """A benchmark using real LLVM function code."""
    backend: str
    module: str
    function_name: str
    code: str
    return_type: str
    parameters: List[str]
    has_switch: bool
    switch_cases: List[Dict]
    line_count: int
    is_interface: bool
    
    # Expected metrics from VEGA paper (if available)
    vega_function_accuracy: Optional[float] = None
    vega_statement_accuracy: Optional[float] = None


class LLVMBenchmarkAdapter:
    """
    Adapts extracted LLVM functions to the VEGA-Verified benchmark format.
    
    This replaces the mock/synthetic benchmarks with real LLVM code,
    enabling meaningful verification testing.
    """
    
    # VEGA paper accuracy metrics
    VEGA_METRICS = {
        'RISCV': {'function_accuracy': 0.715, 'statement_accuracy': 0.550},
        'RI5CY': {'function_accuracy': 0.732, 'statement_accuracy': 0.541},
        'xCORE': {'function_accuracy': 0.622, 'statement_accuracy': 0.463},
        'ARM': {'function_accuracy': 0.70, 'statement_accuracy': 0.52},  # Estimated
        'AArch64': {'function_accuracy': 0.71, 'statement_accuracy': 0.53},  # Estimated
        'X86': {'function_accuracy': 0.68, 'statement_accuracy': 0.50},  # Estimated
        'Mips': {'function_accuracy': 0.69, 'statement_accuracy': 0.51},  # Estimated
    }
    
    # Module priority for benchmark creation
    MODULE_PRIORITY = [
        'MCCodeEmitter',
        'ELFObjectWriter', 
        'ISelDAGToDAG',
        'AsmPrinter',
        'ISelLowering',
    ]
    
    def __init__(self, database_path: Optional[str] = None):
        """Initialize adapter with function database."""
        if database_path is None:
            # Default path
            database_path = str(
                Path(__file__).parent.parent.parent / 'data' / 'llvm_functions_multi.json'
            )
        
        self.db_path = Path(database_path)
        self.database: Optional[FunctionDatabase] = None
        self._load_database()
    
    def _load_database(self):
        """Load the function database."""
        if self.db_path.exists():
            self.database = FunctionDatabase.load(str(self.db_path))
            print(f"Loaded {self.database.get_statistics()['total_functions']} functions")
        else:
            print(f"Warning: Database not found at {self.db_path}")
            self.database = FunctionDatabase()
    
    def get_backend_benchmarks(self, backend: str, 
                                max_functions: int = 50) -> List[RealLLVMBenchmark]:
        """
        Get benchmark functions for a specific backend.
        
        Prioritizes:
        1. Interface functions (encodeInstruction, getRelocType, etc.)
        2. Functions with switch statements (complex logic)
        3. Functions from important modules
        """
        if self.database is None:
            return []
        
        functions = self.database.get_backend_functions(backend)
        if not functions:
            return []
        
        # Sort by priority
        def priority_score(func: Dict) -> int:
            score = 0
            if func.get('is_interface', False):
                score += 1000
            if func.get('has_switch', False):
                score += 500
            
            # Module priority
            module = func.get('module', '')
            if module in self.MODULE_PRIORITY:
                score += (len(self.MODULE_PRIORITY) - self.MODULE_PRIORITY.index(module)) * 100
            
            # Prefer medium-sized functions (not too simple, not too complex)
            lines = func.get('line_count', 0)
            if 20 <= lines <= 200:
                score += 50
            
            return score
        
        sorted_funcs = sorted(functions, key=priority_score, reverse=True)
        selected = sorted_funcs[:max_functions]
        
        # Get VEGA metrics for this backend
        metrics = self.VEGA_METRICS.get(backend, {})
        
        benchmarks = []
        for func in selected:
            benchmark = RealLLVMBenchmark(
                backend=backend,
                module=func.get('module', 'Unknown'),
                function_name=func['name'],
                code=func.get('code', ''),
                return_type=func.get('return_type', 'void'),
                parameters=func.get('parameters', []),
                has_switch=func.get('has_switch', False),
                switch_cases=func.get('switch_cases', []),
                line_count=func.get('line_count', 0),
                is_interface=func.get('is_interface', False),
                vega_function_accuracy=metrics.get('function_accuracy'),
                vega_statement_accuracy=metrics.get('statement_accuracy'),
            )
            benchmarks.append(benchmark)
        
        return benchmarks
    
    def get_vega_paper_benchmarks(self) -> Dict[str, List[RealLLVMBenchmark]]:
        """
        Get benchmarks matching the VEGA paper evaluation targets.
        
        Returns functions for RISCV (representing RI5CY as well) since
        RI5CY is a RISC-V variant.
        """
        return {
            'RISCV': self.get_backend_benchmarks('RISCV', max_functions=100),
            'ARM': self.get_backend_benchmarks('ARM', max_functions=50),
            'AArch64': self.get_backend_benchmarks('AArch64', max_functions=50),
        }
    
    def get_switch_function_benchmarks(self, 
                                        max_per_backend: int = 20) -> Dict[str, List[RealLLVMBenchmark]]:
        """
        Get benchmarks focusing on functions with switch statements.
        
        These are particularly interesting for verification as they
        have complex control flow similar to getRelocType.
        """
        if self.database is None:
            return {}
        
        result = {}
        stats = self.database.get_statistics()
        
        for backend in stats.get('backends', []):
            functions = self.database.get_backend_functions(backend)
            switch_funcs = [f for f in functions if f.get('has_switch', False)]
            
            # Sort by number of cases (complexity)
            def switch_complexity(func):
                cases = func.get('switch_cases', [])
                if not cases:
                    return 0
                return sum(len(c.get('cases', [])) for c in cases)
            
            sorted_funcs = sorted(switch_funcs, key=switch_complexity, reverse=True)
            selected = sorted_funcs[:max_per_backend]
            
            metrics = self.VEGA_METRICS.get(backend, {})
            
            benchmarks = []
            for func in selected:
                benchmark = RealLLVMBenchmark(
                    backend=backend,
                    module=func.get('module', 'Unknown'),
                    function_name=func['name'],
                    code=func.get('code', ''),
                    return_type=func.get('return_type', 'void'),
                    parameters=func.get('parameters', []),
                    has_switch=True,
                    switch_cases=func.get('switch_cases', []),
                    line_count=func.get('line_count', 0),
                    is_interface=func.get('is_interface', False),
                    vega_function_accuracy=metrics.get('function_accuracy'),
                    vega_statement_accuracy=metrics.get('statement_accuracy'),
                )
                benchmarks.append(benchmark)
            
            if benchmarks:
                result[backend] = benchmarks
        
        return result
    
    def get_interface_function_benchmarks(self) -> Dict[str, Dict[str, RealLLVMBenchmark]]:
        """
        Get key interface functions for each backend.
        
        These are the most important functions for compiler backend
        functionality (encodeInstruction, getRelocType, etc.)
        """
        if self.database is None:
            return {}
        
        result = {}
        stats = self.database.get_statistics()
        
        for backend in stats.get('backends', []):
            functions = self.database.get_backend_functions(backend)
            interface_funcs = [f for f in functions if f.get('is_interface', False)]
            
            metrics = self.VEGA_METRICS.get(backend, {})
            backend_interfaces = {}
            
            for func in interface_funcs:
                benchmark = RealLLVMBenchmark(
                    backend=backend,
                    module=func.get('module', 'Unknown'),
                    function_name=func['name'],
                    code=func.get('code', ''),
                    return_type=func.get('return_type', 'void'),
                    parameters=func.get('parameters', []),
                    has_switch=func.get('has_switch', False),
                    switch_cases=func.get('switch_cases', []),
                    line_count=func.get('line_count', 0),
                    is_interface=True,
                    vega_function_accuracy=metrics.get('function_accuracy'),
                    vega_statement_accuracy=metrics.get('statement_accuracy'),
                )
                backend_interfaces[func['name']] = benchmark
            
            if backend_interfaces:
                result[backend] = backend_interfaces
        
        return result
    
    def get_statistics_comparison(self) -> Dict[str, Any]:
        """
        Get statistics comparing our extraction with VEGA paper.
        """
        if self.database is None:
            return {}
        
        stats = self.database.get_statistics()
        
        vega_paper = {
            'total_functions': 1454,
            'backends': 3,  # RISC-V, RI5CY, xCORE
            'modules_per_backend': 7,
        }
        
        comparison = {
            'vega_paper': vega_paper,
            'our_extraction': {
                'total_functions': stats['total_functions'],
                'backends': len(stats['backends']),
                'backend_list': stats['backends'],
                'interface_functions': stats.get('interface_functions', 0),
                'functions_with_switch': stats.get('functions_with_switch', 0),
            },
            'coverage': {
                'function_coverage': stats['total_functions'] / vega_paper['total_functions'] * 100,
                'exceeds_vega': stats['total_functions'] >= vega_paper['total_functions'],
            }
        }
        
        return comparison
    
    def export_for_verification(self, output_path: str, 
                                 max_per_backend: int = 50) -> Dict[str, Any]:
        """
        Export benchmarks in a format ready for the verification pipeline.
        
        Creates a JSON file with all necessary information for running
        VEGA-Verified verification and repair.
        """
        export_data = {
            'metadata': {
                'source': 'LLVM 18.1.0',
                'extraction_method': 'github_api_fetch + cpp_parser',
                'total_functions': 0,
            },
            'backends': {}
        }
        
        if self.database is None:
            return export_data
        
        stats = self.database.get_statistics()
        export_data['metadata']['total_functions'] = stats['total_functions']
        
        for backend in stats.get('backends', []):
            benchmarks = self.get_backend_benchmarks(backend, max_per_backend)
            
            export_data['backends'][backend] = {
                'metrics': self.VEGA_METRICS.get(backend, {}),
                'functions': [
                    {
                        'name': b.function_name,
                        'module': b.module,
                        'code': b.code,
                        'return_type': b.return_type,
                        'parameters': b.parameters,
                        'has_switch': b.has_switch,
                        'switch_cases': b.switch_cases,
                        'line_count': b.line_count,
                        'is_interface': b.is_interface,
                    }
                    for b in benchmarks
                ]
            }
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported {sum(len(b['functions']) for b in export_data['backends'].values())} functions to {output_path}")
        
        return export_data


def create_vega_verified_benchmarks() -> Dict[str, Any]:
    """
    Create the main VEGA-Verified benchmark suite from real LLVM code.
    
    This is the entry point for integrating real LLVM functions
    into the verification pipeline.
    """
    adapter = LLVMBenchmarkAdapter()
    
    return {
        'vega_paper_targets': adapter.get_vega_paper_benchmarks(),
        'switch_functions': adapter.get_switch_function_benchmarks(),
        'interface_functions': adapter.get_interface_function_benchmarks(),
        'statistics': adapter.get_statistics_comparison(),
    }


if __name__ == '__main__':
    # Demo usage
    print("=" * 60)
    print("VEGA-Verified Benchmark Adapter Demo")
    print("=" * 60)
    
    adapter = LLVMBenchmarkAdapter()
    
    # Get statistics comparison
    comparison = adapter.get_statistics_comparison()
    
    print("\n=== Statistics Comparison ===")
    print(f"VEGA Paper: {comparison['vega_paper']['total_functions']} functions")
    print(f"Our Extraction: {comparison['our_extraction']['total_functions']} functions")
    print(f"Coverage: {comparison['coverage']['function_coverage']:.1f}%")
    print(f"Exceeds VEGA: {'Yes' if comparison['coverage']['exceeds_vega'] else 'No'}")
    
    # Get interface functions
    interfaces = adapter.get_interface_function_benchmarks()
    print(f"\n=== Interface Functions ===")
    for backend, funcs in interfaces.items():
        print(f"{backend}: {len(funcs)} interface functions")
        for name in list(funcs.keys())[:3]:
            print(f"  - {name}")
    
    # Get switch functions
    switch_funcs = adapter.get_switch_function_benchmarks()
    print(f"\n=== Switch Statement Functions ===")
    for backend, funcs in switch_funcs.items():
        print(f"{backend}: {len(funcs)} functions with switch")
        if funcs:
            top = funcs[0]
            cases = sum(len(c.get('cases', [])) for c in top.switch_cases)
            print(f"  Most complex: {top.function_name} ({cases} cases)")
    
    # Export for verification
    print("\n=== Exporting Benchmarks ===")
    adapter.export_for_verification(
        'data/verification_benchmarks.json',
        max_per_backend=30
    )
