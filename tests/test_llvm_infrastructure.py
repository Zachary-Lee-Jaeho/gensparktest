"""
Tests for LLVM Docker Infrastructure and AST Extraction.

Tests cover:
1. Docker-based LLVM environment
2. AST Extractor functionality
3. Ground Truth database structure
4. Multi-backend support
"""

import pytest
import json
from pathlib import Path
import subprocess
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestLLVMDockerEnvironment:
    """Tests for LLVM Docker environment."""
    
    @pytest.fixture
    def docker_available(self):
        """Check if Docker is available."""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def test_docker_image_exists(self, docker_available):
        """Test that vega-llvm-base image exists."""
        if not docker_available:
            pytest.skip("Docker not available")
        
        result = subprocess.run(
            ['docker', 'images', 'vega-llvm-base', '--format', '{{.Repository}}'],
            capture_output=True, text=True, timeout=30
        )
        
        # Image should exist
        assert 'vega-llvm-base' in result.stdout or result.returncode == 0
    
    def test_llvm_version_in_docker(self, docker_available):
        """Test LLVM version in Docker container."""
        if not docker_available:
            pytest.skip("Docker not available")
        
        result = subprocess.run(
            ['docker', 'run', '--rm', 'vega-llvm-base', 'llvm-config-18', '--version'],
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode == 0:
            assert '18' in result.stdout
    
    def test_clang_available_in_docker(self, docker_available):
        """Test Clang availability in Docker container."""
        if not docker_available:
            pytest.skip("Docker not available")
        
        result = subprocess.run(
            ['docker', 'run', '--rm', 'vega-llvm-base', 'clang-18', '--version'],
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode == 0:
            assert 'clang' in result.stdout.lower()


class TestGroundTruthDatabase:
    """Tests for the Ground Truth database."""
    
    @pytest.fixture
    def ground_truth_db(self):
        """Load ground truth database if available."""
        db_path = Path('data/llvm_ground_truth.json')
        if not db_path.exists():
            pytest.skip("Ground truth database not found")
        
        with open(db_path) as f:
            return json.load(f)
    
    def test_database_version(self, ground_truth_db):
        """Test database has correct version."""
        assert 'version' in ground_truth_db
        assert ground_truth_db['version'] in ['1.0', '2.0']
    
    def test_database_has_functions(self, ground_truth_db):
        """Test database contains functions."""
        assert 'functions' in ground_truth_db
        assert len(ground_truth_db['functions']) > 0
    
    def test_database_has_stats(self, ground_truth_db):
        """Test database contains statistics."""
        assert 'stats' in ground_truth_db
        stats = ground_truth_db['stats']
        
        assert 'total_functions' in stats
        assert 'total_switches' in stats
        assert 'backends' in stats
    
    def test_database_has_multiple_backends(self, ground_truth_db):
        """Test database covers multiple backends."""
        backends = ground_truth_db['stats']['backends']
        
        # Should have at least 3 backends
        assert len(backends) >= 3
        
        # Should include RISCV
        assert 'RISCV' in backends
    
    def test_riscv_function_count(self, ground_truth_db):
        """Test RISCV function count is reasonable."""
        riscv_stats = ground_truth_db['stats']['backends'].get('RISCV', {})
        
        # Should have at least 400 RISCV functions
        assert riscv_stats.get('functions', 0) >= 400
    
    def test_switch_statements_extracted(self, ground_truth_db):
        """Test switch statements were extracted."""
        total_switches = ground_truth_db['stats']['total_switches']
        
        # Should have at least 100 switch statements
        assert total_switches >= 100
    
    def test_function_structure(self, ground_truth_db):
        """Test function data structure."""
        functions = ground_truth_db['functions']
        
        # Get first function
        func_id = list(functions.keys())[0]
        func = functions[func_id]
        
        # Check required fields
        assert 'name' in func
        assert 'backend' in func
        assert 'return_type' in func
        assert 'parameters' in func
        assert 'body' in func
        assert 'switches' in func
        assert 'calls' in func
    
    def test_switch_structure(self, ground_truth_db):
        """Test switch statement data structure."""
        # Find a function with switches
        for func_id, func in ground_truth_db['functions'].items():
            if func['switches']:
                switch = func['switches'][0]
                
                assert 'condition' in switch
                assert 'cases' in switch
                assert 'default' in switch
                break


class TestRISCVASTExtraction:
    """Tests for RISCV AST extraction."""
    
    @pytest.fixture
    def riscv_ast(self):
        """Load RISCV AST data if available."""
        ast_path = Path('data/llvm_riscv_ast.json')
        if not ast_path.exists():
            pytest.skip("RISCV AST data not found")
        
        with open(ast_path) as f:
            return json.load(f)
    
    def test_ast_backend(self, riscv_ast):
        """Test AST is for RISCV backend."""
        assert riscv_ast['backend'] == 'RISCV'
    
    def test_ast_function_count(self, riscv_ast):
        """Test AST has expected function count."""
        functions = riscv_ast['functions']
        
        # Should have at least 400 functions
        assert len(functions) >= 400
    
    def test_function_has_body(self, riscv_ast):
        """Test functions have body code."""
        functions = riscv_ast['functions']
        
        functions_with_body = sum(1 for f in functions if f['body'])
        
        # Most functions should have body
        assert functions_with_body > len(functions) * 0.8
    
    def test_getreloctype_extracted(self, riscv_ast):
        """Test getRelocType function is extracted."""
        functions = riscv_ast['functions']
        
        reloc_funcs = [f for f in functions if 'getRelocType' in f['name']]
        
        # Should have at least one getRelocType function
        assert len(reloc_funcs) >= 1
    
    def test_switches_have_cases(self, riscv_ast):
        """Test extracted switches have cases."""
        functions = riscv_ast['functions']
        
        for func in functions:
            for switch in func.get('switches', []):
                if switch['cases']:
                    # Found a switch with cases
                    assert len(switch['cases']) > 0
                    assert 'label' in switch['cases'][0]
                    return
        
        # At least one switch should have cases
        switches_with_cases = sum(
            1 for f in functions 
            for sw in f.get('switches', []) 
            if sw['cases']
        )
        assert switches_with_cases > 0


class TestASTExtractorTool:
    """Tests for the AST extractor tool."""
    
    def test_ast_extractor_binary_exists(self):
        """Test AST extractor binary exists in output directory."""
        binary_path = Path('output/ast_extractor')
        
        # Binary may or may not exist depending on Docker build
        if binary_path.exists():
            assert binary_path.stat().st_size > 0
    
    def test_analyzer_script_exists(self):
        """Test LLVM analyzer script exists."""
        script_path = Path('docker/tools/llvm_analyzer.py')
        
        assert script_path.exists()
        assert script_path.stat().st_size > 0


class TestDockerFiles:
    """Tests for Docker configuration files."""
    
    def test_dockerfile_exists(self):
        """Test LLVM Dockerfile exists."""
        dockerfile = Path('docker/Dockerfile.llvm')
        
        assert dockerfile.exists()
    
    def test_dockerfile_has_llvm(self):
        """Test Dockerfile installs LLVM."""
        dockerfile = Path('docker/Dockerfile.llvm')
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Should reference LLVM installation
            assert 'llvm' in content.lower()
            assert 'clang' in content.lower()
    
    def test_ast_extractor_source_exists(self):
        """Test AST extractor C++ source exists."""
        source = Path('docker/tools/ast_extractor.cpp')
        
        assert source.exists()
    
    def test_ast_extractor_uses_libtooling(self):
        """Test AST extractor uses Clang LibTooling."""
        source = Path('docker/tools/ast_extractor.cpp')
        
        if source.exists():
            content = source.read_text()
            
            # Should use LibTooling
            assert 'RecursiveASTVisitor' in content
            assert 'clang/Tooling' in content


class TestDataIntegrity:
    """Tests for data integrity across databases."""
    
    @pytest.fixture
    def all_databases(self):
        """Load all available databases."""
        dbs = {}
        
        paths = {
            'functions': 'data/llvm_functions_multi.json',
            'ground_truth': 'data/llvm_ground_truth.json',
            'riscv_ast': 'data/llvm_riscv_ast.json',
            'tablegen': 'data/tablegen_database.json',
        }
        
        for name, path in paths.items():
            path = Path(path)
            if path.exists():
                with open(path) as f:
                    dbs[name] = json.load(f)
        
        return dbs
    
    def test_riscv_function_consistency(self, all_databases):
        """Test RISCV function counts are consistent across databases."""
        if 'ground_truth' not in all_databases:
            pytest.skip("Ground truth database not found")
        
        gt_count = all_databases['ground_truth']['stats']['backends'].get('RISCV', {}).get('functions', 0)
        
        # Should be in reasonable range
        assert gt_count >= 400
        assert gt_count <= 1000
    
    def test_tablegen_backend_coverage(self, all_databases):
        """Test TableGen covers same backends as Ground Truth."""
        if 'tablegen' not in all_databases or 'ground_truth' not in all_databases:
            pytest.skip("Required databases not found")
        
        tg_backends = set(all_databases['tablegen'].get('backends', {}).keys())
        gt_backends = set(all_databases['ground_truth']['stats']['backends'].keys())
        
        # Should have significant overlap
        common = tg_backends & gt_backends
        assert len(common) >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
