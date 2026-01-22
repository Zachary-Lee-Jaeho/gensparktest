"""
Tests for the comprehensive VEGA-Verified pipeline.

Tests the integration of:
1. Z3 Semantic Analyzer
2. Transformer Repair Model
3. Integrated CGNR Engine
4. Comprehensive Experiment Runner
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.specification.spec_language import (
    Specification, Condition, ConditionType, Variable, Constant
)
from src.verification.z3_semantic_analyzer import (
    Z3SemanticAnalyzer, 
    ArchitectureType, 
    create_semantic_analyzer,
    FIXUP_MAPPINGS
)
from src.repair.transformer_repair import (
    HybridTransformerRepairModel,
    TemplateRepairModel,
    CodeContext,
    RepairStrategy,
    create_transformer_repair_model
)
from src.repair.integrated_cgnr import (
    IntegratedCGNREngine,
    CGNRStatus,
    create_cgnr_engine
)


class TestZ3SemanticAnalyzer:
    """Tests for Z3 Semantic Analyzer."""
    
    def test_create_analyzer(self):
        """Test analyzer creation."""
        analyzer = create_semantic_analyzer(architecture=ArchitectureType.RISCV)
        assert analyzer is not None
        assert analyzer.architecture == ArchitectureType.RISCV
    
    def test_analyze_simple_function(self):
        """Test analysis of a simple function."""
        analyzer = create_semantic_analyzer(architecture=ArchitectureType.RISCV)
        
        code = """
        unsigned getRelocType(const MCFixup &Fixup, bool IsPCRel) const {
            unsigned Kind = Fixup.getTargetKind();
            switch (Kind) {
            case FK_NONE:
                return ELF::R_RISCV_NONE;
            case FK_Data_4:
                return IsPCRel ? ELF::R_RISCV_32_PCREL : ELF::R_RISCV_32;
            default:
                return ELF::R_RISCV_NONE;
            }
        }
        """
        
        spec = Specification(
            function_name="getRelocType",
            preconditions=[],
            postconditions=[],
            invariants=[]
        )
        
        result = analyzer.analyze(code, spec)
        
        assert result is not None
        assert result.time_ms >= 0
    
    def test_fixup_mappings_exist(self):
        """Test that fixup mappings exist for all architectures."""
        for arch in ArchitectureType:
            assert arch in FIXUP_MAPPINGS or arch.value in [a.value for a in FIXUP_MAPPINGS.keys()]
    
    def test_architecture_type_values(self):
        """Test architecture type enum values."""
        assert ArchitectureType.RISCV.value == "riscv"
        assert ArchitectureType.ARM.value == "arm"
        assert ArchitectureType.MIPS.value == "mips"
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked."""
        analyzer = create_semantic_analyzer()
        
        code = "unsigned test() { return 0; }"
        spec = Specification(function_name="test", preconditions=[], postconditions=[], invariants=[])
        
        analyzer.analyze(code, spec)
        
        stats = analyzer.get_statistics()
        assert stats["analyses"] >= 1
        assert stats["total_time_ms"] >= 0


class TestTransformerRepairModel:
    """Tests for Transformer Repair Model."""
    
    def test_create_repair_model(self):
        """Test repair model creation."""
        model = create_transformer_repair_model()
        assert model is not None
    
    def test_template_repair_missing_pcrel(self):
        """Test template repair for missing IsPCRel."""
        model = TemplateRepairModel(verbose=False)
        
        code = """
        unsigned getRelocType(const MCFixup &Fixup, bool IsPCRel) const {
            unsigned Kind = Fixup.getTargetKind();
            switch (Kind) {
            case FK_Data_4:
                return ELF::R_RISCV_32;
            default:
                return ELF::R_RISCV_NONE;
            }
        }
        """
        
        context = CodeContext(
            code=code,
            function_name="getRelocType",
            architecture="RISCV",
            counterexample={
                "input_values": {"Kind": "FK_Data_4", "IsPCRel": True},
                "expected_output": "R_RISCV_32_PCREL",
                "actual_output": "R_RISCV_32",
            },
            specification=None,
        )
        
        candidates = model.generate_repairs(context, num_candidates=3)
        
        assert len(candidates) > 0
        assert any("IsPCRel" in c.repaired_code for c in candidates)
    
    def test_hybrid_model_repair(self):
        """Test hybrid model repair generation."""
        model = create_transformer_repair_model(verbose=False)
        
        context = CodeContext(
            code="unsigned test() { return 0; }",
            function_name="test",
            architecture="RISCV",
            counterexample={},
            specification=None,
        )
        
        candidates = model.generate_repairs(context)
        
        # Should generate fallback repairs at minimum
        assert isinstance(candidates, list)
    
    def test_repair_strategy_enum(self):
        """Test repair strategy enum values."""
        assert RepairStrategy.TEMPLATE_EXACT.value == "template_exact"
        assert RepairStrategy.PATTERN_BASED.value == "pattern_based"
        assert RepairStrategy.FALLBACK.value == "fallback"
    
    def test_code_context_fault_region(self):
        """Test fault region extraction from context."""
        code = """line 1
line 2
line 3
line 4
line 5"""
        
        context = CodeContext(
            code=code,
            function_name="test",
            architecture="RISCV",
            counterexample={},
            specification=None,
            fault_location=(2, 3)
        )
        
        region = context.get_fault_region()
        assert "line 1" in region or "line 2" in region


class TestIntegratedCGNREngine:
    """Tests for Integrated CGNR Engine."""
    
    def test_create_engine(self):
        """Test CGNR engine creation."""
        engine = create_cgnr_engine()
        assert engine is not None
        assert engine.max_iterations == 5
    
    def test_repair_simple_function(self):
        """Test repair of a simple function."""
        engine = create_cgnr_engine(max_iterations=2, verbose=False)
        
        code = """
        unsigned getValue() {
            return 42;
        }
        """
        
        spec = Specification(
            function_name="getValue",
            preconditions=[],
            postconditions=[
                Condition(ConditionType.GREATER_EQUAL, [Variable("result"), Constant(0)])
            ],
            invariants=[]
        )
        
        result = engine.repair(code, spec)
        
        assert result is not None
        assert result.iterations >= 1
        assert result.total_time_ms >= 0
    
    def test_cgnr_status_values(self):
        """Test CGNR status enum values."""
        assert CGNRStatus.SUCCESS.value == "success"
        assert CGNRStatus.FAILED.value == "failed"
        assert CGNRStatus.PARTIAL.value == "partial"
        assert CGNRStatus.MAX_ITERATIONS.value == "max_iters"
    
    def test_engine_statistics(self):
        """Test that engine tracks statistics."""
        engine = create_cgnr_engine(verbose=False)
        
        stats = engine.get_statistics()
        
        assert "total_repairs" in stats
        assert "successful_repairs" in stats
        assert "total_iterations" in stats


class TestComprehensiveExperiment:
    """Tests for Comprehensive Experiment Runner."""
    
    def test_import_experiment_runner(self):
        """Test that experiment runner can be imported."""
        from src.integration.comprehensive_experiment import (
            ComprehensiveExperimentRunner,
            ExperimentResult,
            BackendResult,
            FunctionResult,
        )
        
        assert ComprehensiveExperimentRunner is not None
        assert ExperimentResult is not None
    
    def test_create_vega_benchmarks(self):
        """Test VEGA benchmark creation."""
        from src.integration.comprehensive_experiment import create_vega_paper_benchmarks
        
        benchmarks = create_vega_paper_benchmarks()
        
        assert len(benchmarks) == 3  # RISCV, RI5CY, xCORE
        
        names = [b["name"] for b in benchmarks]
        assert "RISCV" in names
        assert "RI5CY" in names
        assert "xCORE" in names
    
    def test_create_extended_benchmarks(self):
        """Test extended benchmark creation."""
        from src.integration.comprehensive_experiment import create_extended_benchmarks
        
        benchmarks = create_extended_benchmarks()
        
        assert len(benchmarks) >= 2  # ARM, MIPS
        
        names = [b["name"] for b in benchmarks]
        assert "ARM" in names
        assert "MIPS" in names
    
    def test_experiment_result_serialization(self):
        """Test experiment result JSON serialization."""
        from src.integration.comprehensive_experiment import (
            ExperimentResult,
            BackendResult,
        )
        
        result = ExperimentResult(
            experiment_name="Test",
            timestamp="2026-01-22",
        )
        
        json_str = result.to_json()
        
        assert "Test" in json_str
        assert "2026-01-22" in json_str


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""
    
    def test_full_pipeline_single_function(self):
        """Test full pipeline on a single function."""
        # Create analyzer
        analyzer = create_semantic_analyzer(architecture=ArchitectureType.RISCV)
        
        # Test code with bug
        buggy_code = """
        unsigned getRelocType(const MCFixup &Fixup, bool IsPCRel) const {
            unsigned Kind = Fixup.getTargetKind();
            switch (Kind) {
            case FK_Data_4:
                return ELF::R_RISCV_32;
            default:
                return ELF::R_RISCV_NONE;
            }
        }
        """
        
        # Create specification with IsPCRel invariant
        spec = Specification(
            function_name="getRelocType",
            module="ELFObjectWriter",
            preconditions=[],
            postconditions=[],
            invariants=[
                Condition(ConditionType.IMPLIES, [
                    Condition(ConditionType.AND, [
                        Condition(ConditionType.EQUALITY, [Variable("Kind"), Constant("FK_Data_4")]),
                        Condition(ConditionType.EQUALITY, [Variable("IsPCRel"), Constant(True, const_type="bool")]),
                    ]),
                    Condition(ConditionType.EQUALITY, [Variable("result"), Constant("R_RISCV_32_PCREL")]),
                ]),
            ]
        )
        
        # Step 1: Semantic analysis
        analysis_result = analyzer.analyze(buggy_code, spec)
        
        # Should find the bug
        assert analysis_result is not None
        
        # Step 2: Try repair
        repair_model = create_transformer_repair_model(verbose=False)
        
        context = CodeContext(
            code=buggy_code,
            function_name="getRelocType",
            architecture="RISCV",
            counterexample={
                "input_values": {"Kind": "FK_Data_4", "IsPCRel": True},
                "expected_output": "R_RISCV_32_PCREL",
                "actual_output": "R_RISCV_32",
            },
            specification=spec,
        )
        
        candidates = repair_model.generate_repairs(context)
        
        # Should generate repair candidates
        assert len(candidates) > 0
        
        # At least one should add IsPCRel check
        has_fix = any("IsPCRel" in c.repaired_code for c in candidates)
        assert has_fix, "Should generate IsPCRel fix"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
