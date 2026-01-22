"""
Tests for LLVM Extraction Pipeline

Tests the functionality of the LLVM source code extraction system.
Focuses on integration tests with the extracted LLVM database.
"""

import pytest
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestLLVMExtractionModules:
    """Tests for LLVM extraction module definitions."""
    
    def test_backend_modules_defined(self):
        """Test that backend modules are correctly defined."""
        from src.llvm_extraction.extractor import BACKEND_MODULES
        
        # Key modules should exist
        assert 'MCCodeEmitter' in BACKEND_MODULES
        assert 'ELFObjectWriter' in BACKEND_MODULES
        assert 'AsmPrinter' in BACKEND_MODULES
        assert 'ISelDAGToDAG' in BACKEND_MODULES
        assert 'ISelLowering' in BACKEND_MODULES
    
    def test_known_backends_defined(self):
        """Test that known backends list is defined."""
        from src.llvm_extraction.extractor import KNOWN_BACKENDS
        
        # Should have key targets
        assert 'RISCV' in KNOWN_BACKENDS
        assert 'ARM' in KNOWN_BACKENDS
        assert 'AArch64' in KNOWN_BACKENDS
        assert 'X86' in KNOWN_BACKENDS
        assert 'Mips' in KNOWN_BACKENDS
    
    def test_interface_functions_defined(self):
        """Test interface functions are defined for each module."""
        from src.llvm_extraction.extractor import BACKEND_MODULES
        
        # Key interface functions should be present
        assert 'encodeInstruction' in BACKEND_MODULES['MCCodeEmitter']['key_functions']
        assert 'getRelocType' in BACKEND_MODULES['ELFObjectWriter']['key_functions']
        assert 'emitInstruction' in BACKEND_MODULES['AsmPrinter']['key_functions']
        assert 'Select' in BACKEND_MODULES['ISelDAGToDAG']['key_functions']
        assert 'LowerOperation' in BACKEND_MODULES['ISelLowering']['key_functions']


class TestExtractedDatabase:
    """Integration tests with the extracted LLVM database."""
    
    @pytest.fixture
    def db_data(self):
        """Load the extracted LLVM database."""
        db_path = Path(__file__).parent.parent / 'data' / 'llvm_functions_multi.json'
        if not db_path.exists():
            pytest.skip("LLVM database not available - run extraction first")
        
        with open(db_path) as f:
            return json.load(f)
    
    def test_database_has_functions(self, db_data):
        """Test that the database contains functions."""
        assert 'functions' in db_data
        assert 'stats' in db_data
        
        total = db_data['stats']['total_functions']
        print(f"\nTotal functions in database: {total}")
        assert total > 0
    
    def test_database_has_multiple_backends(self, db_data):
        """Test that multiple backends are present."""
        backends = set()
        
        # Functions is a dict with function IDs as keys
        for func_id, func in db_data['functions'].items():
            backends.add(func['backend'])
        
        print(f"\nBackends: {backends}")
        assert len(backends) >= 3, f"Expected at least 3 backends, got {len(backends)}"
    
    def test_database_has_riscv(self, db_data):
        """Test that RISCV backend has functions."""
        riscv_count = sum(1 for func in db_data['functions'].values() 
                        if func.get('backend') == 'RISCV')
        
        print(f"\nRISCV functions: {riscv_count}")
        assert riscv_count > 0
    
    def test_functions_have_required_fields(self, db_data):
        """Test that functions have required fields."""
        required_fields = ['name', 'backend', 'module']
        
        # Check first 10 functions
        for func_id, func in list(db_data['functions'].items())[:10]:
            for field in required_fields:
                assert field in func, f"Function {func_id} missing {field}"
    
    def test_has_switch_functions(self, db_data):
        """Test that some functions have switch statements."""
        switch_count = sum(1 for func in db_data['functions'].values() 
                         if func.get('has_switch'))
        
        print(f"\nFunctions with switch statements: {switch_count}")
        assert switch_count > 0, "Expected some functions with switch statements"


class TestVEGAPaperComparison:
    """Compare extraction results with VEGA paper numbers."""
    
    @pytest.fixture
    def db_data(self):
        """Load the extracted LLVM database."""
        db_path = Path(__file__).parent.parent / 'data' / 'llvm_functions_multi.json'
        if not db_path.exists():
            pytest.skip("LLVM database not available")
        
        with open(db_path) as f:
            return json.load(f)
    
    def test_exceeds_vega_function_count(self, db_data):
        """Test that we extracted at least as many functions as VEGA."""
        # VEGA paper claims 1,454 functions across 3 targets
        vega_total = 1454
        
        total = db_data['stats']['total_functions']
        
        print(f"\n=== VEGA Paper Comparison ===")
        print(f"VEGA paper total: {vega_total}")
        print(f"Our extraction: {total}")
        print(f"Coverage: {total / vega_total * 100:.1f}%")
        
        # We should have at least as many (since we have 5 backends vs their 3)
        assert total >= vega_total, f"Expected at least {vega_total} functions, got {total}"
    
    def test_has_vega_target_backends(self, db_data):
        """Test that we have the VEGA paper target backends."""
        backends = set(func['backend'] for func in db_data['functions'].values())
        
        print(f"\nBackends: {backends}")
        
        # Must have RISCV
        assert 'RISCV' in backends, "Missing RISCV backend"
    
    def test_module_coverage(self, db_data):
        """Test that we cover the VEGA paper modules."""
        # VEGA paper lists 7 function modules (our names may differ slightly)
        vega_modules = {
            'MCCodeEmitter', 'ELFObjectWriter', 'ISelDAGToDAG',
            'AsmPrinter', 'RegisterInfo', 'InstrInfo', 'ISelLowering'
        }
        
        found_modules = set(func['module'] for func in db_data['functions'].values())
        
        print(f"\nFound modules: {found_modules}")
        print(f"VEGA modules: {vega_modules}")
        
        # We should have at least some of the key modules
        overlap = found_modules & vega_modules
        assert len(overlap) >= 3, f"Expected at least 3 VEGA modules, found {overlap}"


class TestDatabaseStatistics:
    """Test database statistics and metrics."""
    
    @pytest.fixture
    def db_data(self):
        """Load the extracted LLVM database."""
        db_path = Path(__file__).parent.parent / 'data' / 'llvm_functions_multi.json'
        if not db_path.exists():
            pytest.skip("LLVM database not available")
        
        with open(db_path) as f:
            return json.load(f)
    
    def test_print_statistics(self, db_data):
        """Print detailed statistics for the database."""
        print("\n" + "="*60)
        print("LLVM Extraction Database Statistics")
        print("="*60)
        
        # Overall stats
        print(f"\nTotal Functions: {db_data['stats']['total_functions']}")
        print(f"Total Backends: {db_data['stats']['total_backends']}")
        
        # Per-backend breakdown
        backend_counts = {}
        module_counts = {}
        switch_counts = {}
        interface_counts = {}
        
        for func in db_data['functions'].values():
            backend = func['backend']
            module = func['module']
            
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            module_counts[module] = module_counts.get(module, 0) + 1
            
            if func.get('has_switch'):
                switch_counts[backend] = switch_counts.get(backend, 0) + 1
            
            if func.get('is_interface'):
                interface_counts[backend] = interface_counts.get(backend, 0) + 1
        
        print("\nFunctions per Backend:")
        for backend, count in sorted(backend_counts.items(), key=lambda x: -x[1]):
            switch = switch_counts.get(backend, 0)
            interface = interface_counts.get(backend, 0)
            print(f"  {backend}: {count} (switch: {switch}, interface: {interface})")
        
        print("\nFunctions per Module:")
        for module, count in sorted(module_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {module}: {count}")
        
        # Pass the test (this is just for printing stats)
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
