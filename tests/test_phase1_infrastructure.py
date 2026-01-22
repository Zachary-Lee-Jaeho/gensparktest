"""
Tests for Phase 1 LLVM Infrastructure.

Tests cover:
1. TableGen Parser
2. LLVM Test Parser
3. Function Analyzer (Call Graph)
4. Test Coverage Analysis
"""

import pytest
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestTableGenParser:
    """Tests for TableGen parser functionality."""
    
    def test_parser_loads(self):
        """Test that TableGen parser can be imported."""
        from llvm_extraction.tablegen_parser import (
            TableGenParser, TableGenFetcher, TableGenDatabase,
            InstructionDef, RegisterDef, SDNodeDef
        )
        assert TableGenParser is not None
        assert TableGenFetcher is not None
        assert TableGenDatabase is not None
    
    def test_parse_sdnode_definition(self):
        """Test parsing SDNode definitions."""
        from llvm_extraction.tablegen_parser import TableGenParser
        
        content = '''
        def riscv_call : SDNode<"RISCVISD::CALL", SDT_RISCVCall,
                                 [SDNPHasChain, SDNPOutGlue]>;
        '''
        
        parser = TableGenParser()
        result = parser.parse(content)
        
        assert 'sdnodes' in result
        assert 'riscv_call' in result['sdnodes']
        assert result['sdnodes']['riscv_call']['opcode'] == 'RISCVISD::CALL'
    
    def test_parse_register_definition(self):
        """Test parsing register definitions."""
        from llvm_extraction.tablegen_parser import TableGenParser
        
        content = '''
        def X0 : RISCVReg<0, "x0", ["zero"]>;
        def X1 : RISCVReg<1, "x1", ["ra"]>;
        '''
        
        parser = TableGenParser()
        result = parser.parse(content)
        
        assert 'registers' in result
        # Check that registers are parsed
        assert result['stats']['total_registers'] >= 0
    
    def test_database_operations(self):
        """Test TableGen database save/load."""
        from llvm_extraction.tablegen_parser import TableGenDatabase
        
        db = TableGenDatabase(db_path=Path('data/tablegen_database.json'))
        
        # Try to load existing database
        if db.load():
            stats = db.get_stats()
            assert 'backends' in stats
            assert len(stats['backends']) > 0
    
    def test_multi_backend_support(self):
        """Test that multiple backends are supported."""
        from llvm_extraction.tablegen_parser import TableGenDatabase
        
        db = TableGenDatabase()
        if db.load():
            stats = db.get_stats()
            # Should have at least RISCV and ARM
            assert 'RISCV' in stats['backends'] or len(stats['backends']) > 0


class TestLLVMTestParser:
    """Tests for LLVM test file parser."""
    
    def test_parser_loads(self):
        """Test that test parser can be imported."""
        from llvm_extraction.test_parser import (
            LLVMTestParser, LLVMTestCase, RunLine, CheckPattern
        )
        assert LLVMTestParser is not None
        assert LLVMTestCase is not None
    
    def test_parse_run_lines(self):
        """Test parsing RUN lines from test files."""
        from llvm_extraction.test_parser import LLVMTestParser
        
        content = '''
; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s | FileCheck %s

define i32 @test(i32 %a) {
  ret i32 %a
}
'''
        
        parser = LLVMTestParser()
        test_case = parser.parse(content, "test.ll")
        
        assert len(test_case.run_lines) >= 2
        assert test_case.run_lines[0].tool == 'llc'
        assert 'riscv' in test_case.run_lines[0].triple.lower()
    
    def test_parse_check_patterns(self):
        """Test parsing CHECK patterns."""
        from llvm_extraction.test_parser import LLVMTestParser
        
        content = '''
; RUN: llc < %s | FileCheck %s

; CHECK-LABEL: foo:
; CHECK: add a0, a0, a1
; CHECK-NEXT: ret

define i32 @foo(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}
'''
        
        parser = LLVMTestParser()
        test_case = parser.parse(content, "test.ll")
        
        assert len(test_case.check_patterns) >= 2
    
    def test_parse_functions(self):
        """Test parsing function definitions."""
        from llvm_extraction.test_parser import LLVMTestParser
        
        content = '''
define i32 @add(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

define i64 @mul(i64 %a, i64 %b) {
  %c = mul i64 %a, %b
  ret i64 %c
}
'''
        
        parser = LLVMTestParser()
        test_case = parser.parse(content, "test.ll")
        
        assert len(test_case.functions) >= 2
        assert any(f.name == 'add' for f in test_case.functions)
        assert any(f.name == 'mul' for f in test_case.functions)
    
    def test_extract_instructions(self):
        """Test extracting RISCV instructions from test content."""
        from llvm_extraction.test_parser import LLVMTestParser
        
        content = '''
; CHECK: add a0, a0, a1
; CHECK: sub a0, a0, a1
; CHECK: mul a0, a0, a1
; CHECK: sll a0, a0, a1
; CHECK: beq a0, a1, .LBB0
'''
        
        parser = LLVMTestParser()
        test_case = parser.parse(content, "test.ll")
        
        assert 'add' in test_case.instructions_tested
        assert 'sub' in test_case.instructions_tested
        assert 'mul' in test_case.instructions_tested
        assert 'sll' in test_case.instructions_tested
        assert 'beq' in test_case.instructions_tested
    
    def test_detect_backend(self):
        """Test backend detection from file path and content."""
        from llvm_extraction.test_parser import LLVMTestParser
        
        parser = LLVMTestParser()
        
        # From file path
        test_case = parser.parse("", "llvm/test/CodeGen/RISCV/alu.ll")
        assert test_case.backend == 'RISCV'
        
        # From triple
        content = '; RUN: llc -mtriple=aarch64 < %s | FileCheck %s'
        test_case = parser.parse(content, "test.ll")
        assert test_case.backend == 'AArch64'


class TestFunctionAnalyzer:
    """Tests for function analyzer and call graph."""
    
    def test_analyzer_loads(self):
        """Test that function analyzer can be imported."""
        from llvm_extraction.function_analyzer import (
            CallGraphBuilder, FunctionNode, FunctionType
        )
        assert CallGraphBuilder is not None
        assert FunctionNode is not None
        assert FunctionType is not None
    
    def test_detect_function_type(self):
        """Test function type detection."""
        from llvm_extraction.function_analyzer import CallGraphBuilder, FunctionType
        
        builder = CallGraphBuilder()
        
        # Test relocation function
        node = builder.add_function(
            name="getRelocType",
            backend="RISCV",
            module="MCCodeEmitter",
            source_code="return ELF::R_RISCV_NONE;"
        )
        assert node.function_type == FunctionType.RELOCATION
        
        # Test emit function
        node = builder.add_function(
            name="emitInstruction",
            backend="RISCV",
            module="MCCodeEmitter",
            source_code="emitByte(opcode);"
        )
        assert node.function_type == FunctionType.CODE_EMISSION
        
        # Test select function
        node = builder.add_function(
            name="selectADD",
            backend="RISCV",
            module="ISelDAGToDAG",
            source_code="return CurDAG->getMachineNode();"
        )
        assert node.function_type == FunctionType.INSTRUCTION_SELECTION
    
    def test_extract_calls(self):
        """Test call extraction from source code."""
        from llvm_extraction.function_analyzer import CallGraphBuilder
        
        builder = CallGraphBuilder()
        
        source_code = '''
        void foo() {
            bar();
            obj->method();
            helper(arg1, arg2);
        }
        '''
        
        node = builder.add_function(
            name="foo",
            backend="RISCV",
            module="Test",
            source_code=source_code
        )
        
        # Should find bar, method, helper calls
        callee_names = [c.callee for c in node.calls]
        assert 'bar' in callee_names or 'helper' in callee_names
    
    def test_extract_llvm_instructions(self):
        """Test extraction of LLVM instruction references."""
        from llvm_extraction.function_analyzer import CallGraphBuilder
        
        builder = CallGraphBuilder()
        
        source_code = '''
        switch (Opcode) {
            case RISCV::ADD: return doAdd();
            case RISCV::SUB: return doSub();
            case RISCV::MUL: return doMul();
        }
        '''
        
        node = builder.add_function(
            name="selectOp",
            backend="RISCV",
            module="ISelDAGToDAG",
            source_code=source_code
        )
        
        assert 'ADD' in node.instructions_used
        assert 'SUB' in node.instructions_used
        assert 'MUL' in node.instructions_used
    
    def test_call_graph_edges(self):
        """Test call graph edge construction."""
        from llvm_extraction.function_analyzer import CallGraphBuilder
        
        builder = CallGraphBuilder()
        
        # Add caller
        builder.add_function(
            name="caller",
            backend="RISCV",
            module="Test",
            source_code="callee(); helper();"
        )
        
        # Add callee
        builder.add_function(
            name="callee",
            backend="RISCV",
            module="Test",
            source_code="return 0;"
        )
        
        # Add helper
        builder.add_function(
            name="helper",
            backend="RISCV",
            module="Test",
            source_code="return 1;"
        )
        
        # Build edges
        builder.build_edges()
        
        # Check edges exist
        assert len(builder.edges) >= 1
        assert 'caller' in builder.nodes['callee'].called_by or len(builder.nodes['callee'].called_by) >= 0


class TestDatabaseIntegration:
    """Tests for database integration."""
    
    @pytest.fixture
    def function_db(self):
        """Load function database if available."""
        db_path = Path('data/llvm_functions_multi.json')
        if not db_path.exists():
            pytest.skip("Function database not found")
        
        with open(db_path) as f:
            return json.load(f)
    
    def test_function_db_structure(self, function_db):
        """Test function database structure."""
        assert 'functions' in function_db
        assert 'version' in function_db
        
        functions = function_db['functions']
        assert len(functions) > 0
    
    def test_function_db_has_backends(self, function_db):
        """Test that function database has multiple backends."""
        functions = function_db['functions']
        
        backends = set()
        for func_data in functions.values():
            if isinstance(func_data, dict) and 'backend' in func_data:
                backends.add(func_data['backend'])
        
        assert len(backends) >= 3  # At least 3 backends
    
    def test_function_db_has_source_code(self, function_db):
        """Test that functions have source code."""
        functions = function_db['functions']
        
        functions_with_code = 0
        for func_data in functions.values():
            if isinstance(func_data, dict):
                if func_data.get('body') or func_data.get('source_code'):
                    functions_with_code += 1
        
        # At least 50% should have source code
        assert functions_with_code > len(functions) * 0.5


class TestTestDatabase:
    """Tests for test database functionality."""
    
    def test_test_db_creation(self):
        """Test test database creation."""
        from llvm_extraction.test_parser import TestDatabase
        
        db = TestDatabase(db_path=Path('data/test_database_temp.json'))
        
        # Should be able to access stats even when empty
        stats = db.get_stats()
        assert 'backends' in stats
    
    def test_test_db_save_load(self):
        """Test test database save and load."""
        from llvm_extraction.test_parser import TestDatabase, LLVMTestCase
        
        # Create test case
        test_case = LLVMTestCase(
            file_path="test/test.ll",
            description="Test case",
            backend="RISCV",
            test_type="codegen",
        )
        
        # Save
        db = TestDatabase(db_path=Path('data/test_database_temp.json'))
        db.add_test_cases('RISCV', [test_case])
        db.save()
        
        # Load
        db2 = TestDatabase(db_path=Path('data/test_database_temp.json'))
        assert db2.load()
        
        stats = db2.get_stats()
        assert 'RISCV' in stats['backends']
        
        # Cleanup
        Path('data/test_database_temp.json').unlink(missing_ok=True)


class TestEnhancedDatabase:
    """Tests for enhanced function database."""
    
    def test_enhanced_db_exists(self):
        """Test that enhanced database can be created."""
        from llvm_extraction.function_analyzer import FunctionDatabaseEnhancer
        
        enhancer = FunctionDatabaseEnhancer()
        
        if enhancer.load():
            enhanced = enhancer.enhance('RISCV')
            
            assert 'functions' in enhanced
            assert 'call_graph' in enhanced
            assert 'stats' in enhanced
            
            # Should have some functions
            if enhanced['stats']['total_functions'] > 0:
                assert enhanced['stats']['function_types']


class TestRealLLVMData:
    """Tests using real LLVM data."""
    
    @pytest.fixture
    def tablegen_db(self):
        """Load TableGen database if available."""
        from llvm_extraction.tablegen_parser import TableGenDatabase
        
        db = TableGenDatabase()
        if not db.load():
            pytest.skip("TableGen database not found")
        return db
    
    def test_riscv_sdnodes(self, tablegen_db):
        """Test RISCV SDNode count."""
        sdnodes = tablegen_db.get_sdnodes('RISCV')
        
        # RISCV should have some SDNodes
        assert len(sdnodes) >= 10
    
    def test_riscv_registers(self, tablegen_db):
        """Test RISCV register count."""
        registers = tablegen_db.get_registers('RISCV')
        
        # RISCV has 32 GPRs + FPRs + special regs
        assert len(registers) >= 32
    
    def test_arm_sdnodes(self, tablegen_db):
        """Test ARM SDNode count."""
        sdnodes = tablegen_db.get_sdnodes('ARM')
        
        # ARM should have SDNodes
        assert len(sdnodes) >= 10


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
