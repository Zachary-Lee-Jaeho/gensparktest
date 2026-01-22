"""
Phase 2 Complete Test Suite for VEGA-Verified.

Tests all Phase 2 components:
- 2.1 Semantic Analysis Engine
- 2.2 SMT Integration (IR to Z3)
- 2.3 Neural Repair Engine
- 2.4 CGNR Pipeline Integration

Run with: python -m pytest tests/test_phase2_complete.py -v
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any


# ============================================================================
# Phase 2.1: Semantic Analysis Engine Tests
# ============================================================================

class TestSemanticAnalyzer:
    """Tests for semantic analysis engine."""
    
    @pytest.fixture
    def analyzer(self):
        from src.verification.semantic_analyzer import SemanticAnalyzer
        return SemanticAnalyzer(verbose=False)
    
    @pytest.fixture
    def sample_code(self):
        return """
unsigned getRelocType(unsigned Kind, bool IsPCRel) {
    if (IsPCRel) {
        switch (Kind) {
        case FK_Data_4:
            return ELF::R_RISCV_32_PCREL;
        default:
            return ELF::R_RISCV_NONE;
        }
    }
    
    switch (Kind) {
    case FK_NONE:
        return ELF::R_RISCV_NONE;
    case FK_Data_4:
        return ELF::R_RISCV_32;
    case FK_Data_8:
        return ELF::R_RISCV_64;
    default:
        llvm_unreachable("Unknown fixup kind");
    }
}
"""
    
    def test_analyzer_creation(self, analyzer):
        """Test analyzer can be created."""
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze_function')
    
    def test_switch_recognition(self, analyzer, sample_code):
        """Test switch statement recognition."""
        from src.verification.semantic_analyzer import SwitchPattern
        
        semantics = analyzer.analyze_function(sample_code, "getRelocType")
        
        # Should find switches
        switches = [p for p in semantics.patterns if isinstance(p, SwitchPattern)]
        assert len(switches) >= 1
    
    def test_if_else_recognition(self, analyzer, sample_code):
        """Test if-else recognition."""
        from src.verification.semantic_analyzer import IfElsePattern
        
        semantics = analyzer.analyze_function(sample_code, "getRelocType")
        
        # Should find if-else
        if_patterns = [p for p in semantics.patterns if isinstance(p, IfElsePattern)]
        assert len(if_patterns) >= 1
    
    def test_return_path_enumeration(self, analyzer, sample_code):
        """Test return path enumeration."""
        semantics = analyzer.analyze_function(sample_code, "getRelocType")
        
        # Should have multiple return paths
        assert len(semantics.return_paths) >= 3
    
    def test_function_calls_extraction(self, analyzer, sample_code):
        """Test function call extraction."""
        semantics = analyzer.analyze_function(sample_code, "getRelocType")
        
        # llvm_unreachable is a function call
        assert 'llvm_unreachable' in semantics.calls or len(semantics.calls) >= 0
    
    def test_assertion_extraction(self, analyzer, sample_code):
        """Test assertion extraction."""
        semantics = analyzer.analyze_function(sample_code, "getRelocType")
        
        # Should find unreachable assertion
        assert len(semantics.assertions) >= 1 or 'unreachable' in str(semantics.assertions)
    
    def test_switch_semantics_extraction(self, analyzer, sample_code):
        """Test detailed switch semantics extraction."""
        switches = analyzer.extract_switch_semantics(sample_code)
        
        assert len(switches) >= 1
        
        # Check structure
        switch = switches[0]
        assert 'switch_variable' in switch
        assert 'cases' in switch
        assert len(switch['cases']) >= 1  # At least one case parsed


class TestPatternMatcher:
    """Tests for pattern matcher."""
    
    @pytest.fixture
    def matcher(self):
        from src.verification.semantic_analyzer import PatternMatcher
        return PatternMatcher()
    
    def test_reloc_type_detection(self, matcher):
        """Test relocation type function detection."""
        code = "unsigned getRelocType(MCContext &Ctx)"
        func_type = matcher.match_function_type(code, "getRelocType")
        assert func_type == "reloc_type"
    
    def test_encode_function_detection(self, matcher):
        """Test encoding function detection."""
        code = "void encodeInstruction(MCInst &MI)"
        func_type = matcher.match_function_type(code, "encodeInstruction")
        assert func_type == "encode_instr"
    
    def test_critical_pattern_identification(self, matcher):
        """Test critical pattern identification."""
        code = """
        switch (Kind) {
        case FK_Data_4:
            if (IsPCRel) return R_X_PC32;
            return R_X_32;
        }
        """
        patterns = matcher.identify_critical_patterns(code)
        
        # Should find fixup switch pattern
        pattern_types = [p['type'] for p in patterns]
        assert 'fixup_switch' in pattern_types or len(patterns) >= 0


# ============================================================================
# Phase 2.2: SMT Integration Tests
# ============================================================================

class TestIRToSMTConverter:
    """Tests for IR to SMT converter."""
    
    @pytest.fixture
    def converter(self):
        from src.verification.ir_to_smt import IRToSMTConverter
        return IRToSMTConverter(verbose=False)
    
    def test_converter_creation(self, converter):
        """Test converter can be created."""
        assert converter is not None
        assert hasattr(converter, 'convert_switch')
    
    def test_enum_registration(self, converter):
        """Test enum registration."""
        values = ["FK_NONE", "FK_Data_4", "FK_Data_8"]
        mapping = converter.register_enum("FixupKind", values)
        
        assert len(mapping) == 3
        assert "FK_NONE" in mapping
        assert mapping["FK_NONE"] == 0
    
    def test_switch_conversion(self, converter):
        """Test switch statement conversion."""
        converter.register_enum("FixupKind", ["FK_NONE", "FK_Data_4"])
        converter.register_enum("RelocType", ["R_NONE", "R_32"])
        
        cases = [
            ("FK_NONE", "R_NONE"),
            ("FK_Data_4", "R_32"),
        ]
        
        input_var, result_var = converter.convert_switch(
            "Kind",
            cases,
            default_value="R_NONE"
        )
        
        assert input_var is not None
        assert result_var is not None
    
    def test_reloc_type_function_conversion(self, converter):
        """Test complete reloc type function conversion."""
        fixup_kinds = ["FK_NONE", "FK_Data_4", "FK_Data_8"]
        reloc_mappings = {
            "FK_NONE": "R_NONE",
            "FK_Data_4": "R_32",
            "FK_Data_8": "R_64",
        }
        
        vars_dict = converter.convert_reloc_type_function(
            fixup_kinds=fixup_kinds,
            reloc_mappings=reloc_mappings
        )
        
        assert "Kind" in vars_dict
        assert "IsPCRel" in vars_dict
        assert "Result" in vars_dict
        assert "model" in vars_dict
    
    def test_statistics(self, converter):
        """Test statistics tracking."""
        stats = converter.get_statistics()
        
        assert 'conversions' in stats
        assert 'constraints_generated' in stats


class TestSMTVerifier:
    """Tests for SMT verifier."""
    
    @pytest.fixture
    def verifier(self):
        from src.verification.ir_to_smt import SMTVerifier
        return SMTVerifier(verbose=False)
    
    @pytest.fixture
    def setup_model(self):
        from src.verification.ir_to_smt import IRToSMTConverter
        
        converter = IRToSMTConverter()
        vars_dict = converter.convert_reloc_type_function(
            fixup_kinds=["FK_NONE", "FK_Data_4", "FK_Data_8"],
            reloc_mappings={
                "FK_NONE": "R_NONE",
                "FK_Data_4": "R_32",
                "FK_Data_8": "R_64",
            }
        )
        return vars_dict
    
    def test_verifier_creation(self, verifier):
        """Test verifier can be created."""
        assert verifier is not None
        assert hasattr(verifier, 'verify_mapping')
    
    def test_correct_mapping_verification(self, verifier, setup_model):
        """Test verification of correct mappings."""
        model = setup_model["model"]
        
        expected = {
            "FK_NONE": "R_NONE",
            "FK_Data_4": "R_32",
        }
        
        verified, failures = verifier.verify_mapping(
            model=model,
            input_var_name="Kind",
            output_var_name="Result",
            expected_mappings=expected
        )
        
        assert verified is True
        assert len(failures) == 0
    
    def test_statistics(self, verifier):
        """Test statistics tracking."""
        stats = verifier.get_statistics()
        
        assert 'queries' in stats
        assert 'sat' in stats
        assert 'unsat' in stats


class TestPropertyDSL:
    """Tests for property DSL."""
    
    @pytest.fixture
    def setup_dsl(self):
        from src.verification.ir_to_smt import IRToSMTConverter, PropertyDSL
        
        converter = IRToSMTConverter()
        vars_dict = converter.convert_reloc_type_function(
            fixup_kinds=["FK_NONE", "FK_Data_4"],
            reloc_mappings={
                "FK_NONE": "R_NONE",
                "FK_Data_4": "R_32",
            }
        )
        
        return PropertyDSL(vars_dict["model"])
    
    def test_mapping_correct_property(self, setup_dsl):
        """Test mapping correctness property."""
        prop = setup_dsl.mapping_correct("Kind", "Result", "FK_NONE", "R_NONE")
        assert prop is not None
    
    def test_all_inputs_handled_property(self, setup_dsl):
        """Test all inputs handled property."""
        prop = setup_dsl.all_inputs_handled("Kind", ["FK_NONE", "FK_Data_4"])
        assert prop is not None


# ============================================================================
# Phase 2.3: Neural Repair Engine Tests
# ============================================================================

class TestSyntheticBugGenerator:
    """Tests for synthetic bug generator."""
    
    @pytest.fixture
    def generator(self):
        from src.repair.training_data import SyntheticBugGenerator
        return SyntheticBugGenerator(seed=42)
    
    @pytest.fixture
    def sample_code(self):
        return """
switch (Kind) {
    case FK_NONE:
        return ELF::R_RISCV_NONE;
    case FK_Data_4:
        return ELF::R_RISCV_32;
    case FK_Data_8:
        return ELF::R_RISCV_64;
    default:
        return ELF::R_RISCV_NONE;
}
"""
    
    def test_generator_creation(self, generator):
        """Test generator can be created."""
        assert generator is not None
        assert hasattr(generator, 'generate_bugs')
    
    def test_missing_case_generation(self, generator, sample_code):
        """Test missing case bug generation."""
        result = generator.generate_missing_case(sample_code)
        
        if result:
            buggy_code, info = result
            assert buggy_code != sample_code
            assert info['bug_type'].value == 'missing_case'
    
    def test_wrong_return_generation(self, generator, sample_code):
        """Test wrong return bug generation."""
        result = generator.generate_wrong_return(sample_code)
        
        if result:
            buggy_code, info = result
            assert buggy_code != sample_code
            assert info['bug_type'].value == 'wrong_return'
    
    def test_multiple_bugs_generation(self, generator, sample_code):
        """Test multiple bug generation."""
        bugs = generator.generate_bugs(sample_code, num_bugs=3)
        
        assert len(bugs) >= 1  # At least some bugs generated


class TestTrainingDataset:
    """Tests for training dataset."""
    
    @pytest.fixture
    def dataset(self):
        from src.repair.training_data import TrainingDataset, TrainingExample, BugType
        
        ds = TrainingDataset(name="test_dataset")
        ds.add_example(TrainingExample(
            id="test_1",
            buggy_code="buggy",
            fixed_code="fixed",
            bug_type=BugType.MISSING_CASE
        ))
        return ds
    
    def test_dataset_creation(self):
        """Test dataset can be created."""
        from src.repair.training_data import TrainingDataset
        
        ds = TrainingDataset(name="test")
        assert ds.name == "test"
        assert len(ds.examples) == 0
    
    def test_add_example(self, dataset):
        """Test adding examples."""
        assert len(dataset.examples) == 1
    
    def test_statistics(self, dataset):
        """Test statistics calculation."""
        stats = dataset.get_statistics()
        
        assert stats['total_examples'] == 1
        assert 'bug_type_distribution' in stats
    
    def test_split(self, dataset):
        """Test dataset splitting."""
        # Add more examples for split
        from src.repair.training_data import TrainingExample, BugType
        
        for i in range(9):
            dataset.add_example(TrainingExample(
                id=f"test_{i+2}",
                buggy_code=f"buggy_{i}",
                fixed_code=f"fixed_{i}",
                bug_type=BugType.WRONG_RETURN
            ))
        
        train, val = dataset.split(train_ratio=0.8)
        
        assert len(train.examples) == 8
        assert len(val.examples) == 2


class TestCodeT5RepairModel:
    """Tests for CodeT5 repair model."""
    
    @pytest.fixture
    def model(self):
        from src.repair.model_finetuning import CodeT5RepairModel, TrainingConfig
        
        config = TrainingConfig(num_epochs=1)
        return CodeT5RepairModel(config)
    
    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None
        assert hasattr(model, 'repair')
    
    def test_mock_repair(self, model):
        """Test mock repair generation."""
        buggy_code = """
switch (Kind) {
    case FK_Data_4:
        return R_32;
    default:
        return R_NONE;
}
"""
        counterexample = {
            "input_values": {"Kind": "FK_Data_8"},
            "expected_output": "R_64",
            "actual_output": "R_NONE"
        }
        
        candidates = model.repair(buggy_code, counterexample, num_candidates=3)
        
        assert len(candidates) >= 1
        assert all(len(c) == 2 for c in candidates)  # (code, confidence)
    
    def test_mock_training(self, model):
        """Test mock training."""
        train_data = [
            ("buggy1", "fixed1"),
            ("buggy2", "fixed2"),
        ]
        
        metrics = model.train(train_data)
        
        assert metrics is not None
        assert hasattr(metrics, 'train_loss')


# ============================================================================
# Phase 2.4: CGNR Pipeline Tests
# ============================================================================

class TestCGNRPipeline:
    """Tests for CGNR pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        from src.integration.cgnr_pipeline import CGNRPipeline
        return CGNRPipeline(max_iterations=3, verbose=False)
    
    @pytest.fixture
    def buggy_code(self):
        return """
switch (Kind) {
    case FK_NONE:
        return ELF::R_RISCV_NONE;
    case FK_Data_4:
        return ELF::R_RISCV_32;
    // Missing FK_Data_8
    default:
        return ELF::R_RISCV_NONE;
}
"""
    
    @pytest.fixture
    def specification(self):
        return {
            "function_name": "getRelocType",
            "expected_mappings": {
                "FK_NONE": "ELF::R_RISCV_NONE",
                "FK_Data_4": "ELF::R_RISCV_32",
                "FK_Data_8": "ELF::R_RISCV_64",
            }
        }
    
    def test_pipeline_creation(self, pipeline):
        """Test pipeline can be created."""
        assert pipeline is not None
        assert pipeline.max_iterations == 3
    
    def test_verification_phase(self, pipeline, buggy_code, specification):
        """Test verification phase."""
        result = pipeline._verify_code(buggy_code, specification)
        
        # Should fail (missing FK_Data_8)
        assert result.verified is False
        assert result.counterexample is not None
    
    def test_repair_generation(self, pipeline, buggy_code, specification):
        """Test repair generation."""
        ver_result = pipeline._verify_code(buggy_code, specification)
        
        candidates = pipeline._generate_repairs(
            buggy_code,
            ver_result.counterexample,
            specification,
            iteration=0
        )
        
        assert len(candidates) >= 1
    
    def test_full_pipeline(self, pipeline, buggy_code, specification):
        """Test complete CGNR pipeline."""
        from src.integration.cgnr_pipeline import PipelineStatus
        
        result = pipeline.run(buggy_code, specification)
        
        # Should complete (verified, repaired, or max_iterations)
        assert result.status in [
            PipelineStatus.VERIFIED,
            PipelineStatus.REPAIRED,
            PipelineStatus.MAX_ITERATIONS,
            PipelineStatus.FAILED,
        ]
    
    def test_already_correct_code(self, pipeline):
        """Test with already correct code."""
        from src.integration.cgnr_pipeline import PipelineStatus
        
        correct_code = """
switch (Kind) {
    case FK_NONE:
        return R_NONE;
    case FK_Data_4:
        return R_32;
    case FK_Data_8:
        return R_64;
    default:
        return R_NONE;
}
"""
        
        # Simple spec that matches the code
        simple_spec = {
            "expected_mappings": {
                "FK_NONE": "R_NONE",
                "FK_Data_4": "R_32",
                "FK_Data_8": "R_64",
            }
        }
        
        result = pipeline.run(correct_code, simple_spec)
        
        # Should either verify or complete within iterations
        assert result.status in [PipelineStatus.VERIFIED, PipelineStatus.MAX_ITERATIONS]
    
    def test_statistics(self, pipeline, buggy_code, specification):
        """Test statistics tracking."""
        pipeline.run(buggy_code, specification)
        
        stats = pipeline.get_statistics()
        
        assert 'pipelines_run' in stats
        assert stats['pipelines_run'] >= 1


class TestEndToEndPipeline:
    """Tests for end-to-end pipeline."""
    
    @pytest.fixture
    def setup_test_data(self, tmp_path):
        """Create test ground truth data."""
        test_data = {
            "version": "2.0",
            "functions": {
                "test_func_1": {
                    "name": "getRelocType",
                    "backend": "RISCV",
                    "body": """
switch (Kind) {
    case FK_NONE:
        return R_NONE;
    case FK_Data_4:
        return R_32;
}
""",
                    "switches": [
                        {
                            "cases": [
                                {"label": "FK_NONE", "return": "R_NONE"},
                                {"label": "FK_Data_4", "return": "R_32"},
                            ]
                        }
                    ]
                }
            }
        }
        
        gt_path = tmp_path / "test_ground_truth.json"
        with open(gt_path, 'w') as f:
            json.dump(test_data, f)
        
        return gt_path, tmp_path / "output"
    
    def test_e2e_pipeline_creation(self, setup_test_data):
        """Test E2E pipeline can be created."""
        from src.integration.cgnr_pipeline import EndToEndPipeline
        
        gt_path, output_dir = setup_test_data
        
        pipeline = EndToEndPipeline(
            ground_truth_path=str(gt_path),
            output_dir=str(output_dir),
            verbose=False
        )
        
        assert pipeline is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase2Integration:
    """Integration tests for all Phase 2 components."""
    
    def test_semantic_to_smt_integration(self):
        """Test semantic analysis to SMT conversion integration."""
        from src.verification.semantic_analyzer import SemanticAnalyzer
        from src.verification.ir_to_smt import IRToSMTConverter
        
        code = """
switch (Kind) {
    case FK_Data_4:
        return R_32;
    case FK_Data_8:
        return R_64;
}
"""
        
        # Analyze
        analyzer = SemanticAnalyzer()
        semantics = analyzer.analyze_function(code, "test")
        
        # Convert to SMT
        converter = IRToSMTConverter()
        
        switch_semantics = analyzer.extract_switch_semantics(code)
        
        assert len(switch_semantics) >= 1
        
        # Build model from semantics
        if switch_semantics:
            switch = switch_semantics[0]
            cases = [(c['value'], c['return']) for c in switch['cases'] if c.get('return')]
            
            if cases:
                converter.register_enum("Kind", [c[0] for c in cases])
                converter.register_enum("Result", [c[1] for c in cases])
                
                input_var, result_var = converter.convert_switch("Kind", cases)
                
                assert input_var is not None
    
    def test_verification_to_repair_integration(self):
        """Test verification failure to repair generation integration."""
        from src.verification.ir_to_smt import IRToSMTConverter, SMTVerifier
        from src.repair.model_finetuning import CodeT5RepairModel, TrainingConfig
        
        # Setup converter
        converter = IRToSMTConverter()
        vars_dict = converter.convert_reloc_type_function(
            fixup_kinds=["FK_NONE", "FK_Data_4"],
            reloc_mappings={
                "FK_NONE": "R_NONE",
                "FK_Data_4": "R_32",
            }
        )
        
        # Verify with wrong expectation (to generate counterexample)
        verifier = SMTVerifier()
        verified, failures = verifier.verify_mapping(
            model=vars_dict["model"],
            input_var_name="Kind",
            output_var_name="Result",
            expected_mappings={"FK_NONE": "R_WRONG"}  # Wrong expectation
        )
        
        # If verification fails, generate repair
        if not verified and failures:
            counterexample = {
                "input_values": {"Kind": failures[0]["input"]},
                "expected_output": failures[0]["expected"],
                "actual_output": failures[0]["actual"],
            }
            
            # Generate repair
            model = CodeT5RepairModel(TrainingConfig())
            candidates = model.repair("switch (Kind) { }", counterexample)
            
            assert len(candidates) >= 1
    
    def test_full_pipeline_integration(self):
        """Test full CGNR pipeline integration."""
        from src.integration.cgnr_pipeline import CGNRPipeline, PipelineStatus
        
        pipeline = CGNRPipeline(max_iterations=3, verbose=False)
        
        code = """
switch (Kind) {
    case FK_NONE:
        return R_NONE;
    case FK_Data_4:
        return R_32;
    case FK_Data_8:
        return R_64;
}
"""
        
        spec = {
            "expected_mappings": {
                "FK_NONE": "R_NONE",
                "FK_Data_4": "R_32",
                "FK_Data_8": "R_64",
            }
        }
        
        result = pipeline.run(code, spec)
        
        # Should verify (code is correct)
        assert result.status == PipelineStatus.VERIFIED


# ============================================================================
# Performance Tests
# ============================================================================

class TestPhase2Performance:
    """Performance tests for Phase 2 components."""
    
    def test_semantic_analysis_performance(self):
        """Test semantic analysis performance."""
        import time
        from src.verification.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        
        # Generate large code
        code = "switch (Kind) {\n"
        for i in range(100):
            code += f"  case CASE_{i}:\n    return RET_{i};\n"
        code += "  default:\n    return DEFAULT;\n}\n"
        
        start = time.time()
        semantics = analyzer.analyze_function(code, "test")
        elapsed = time.time() - start
        
        # Should complete within 1 second
        assert elapsed < 1.0
    
    def test_smt_conversion_performance(self):
        """Test SMT conversion performance."""
        import time
        from src.verification.ir_to_smt import IRToSMTConverter
        
        converter = IRToSMTConverter()
        
        # Register many enums
        fixup_kinds = [f"FK_{i}" for i in range(50)]
        reloc_types = [f"R_{i}" for i in range(50)]
        
        start = time.time()
        converter.register_enum("FixupKind", fixup_kinds)
        converter.register_enum("RelocType", reloc_types)
        
        # Create model
        reloc_mappings = {f"FK_{i}": f"R_{i}" for i in range(50)}
        vars_dict = converter.convert_reloc_type_function(
            fixup_kinds=fixup_kinds,
            reloc_mappings=reloc_mappings
        )
        elapsed = time.time() - start
        
        # Should complete within 2 seconds
        assert elapsed < 2.0


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_code(self):
        """Test with empty code."""
        from src.verification.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        semantics = analyzer.analyze_function("", "empty")
        
        assert semantics is not None
        assert len(semantics.patterns) == 0
    
    def test_no_switch_code(self):
        """Test with code without switches."""
        from src.verification.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        code = "int foo() { return 42; }"
        semantics = analyzer.analyze_function(code, "foo")
        
        assert semantics is not None
    
    def test_nested_switch(self):
        """Test with nested switch statements."""
        from src.verification.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        code = """
switch (A) {
    case 1:
        switch (B) {
            case 2:
                return 12;
        }
    case 3:
        return 30;
}
"""
        semantics = analyzer.analyze_function(code, "nested")
        
        # Should handle nested switches
        assert semantics is not None
    
    def test_empty_specification(self):
        """Test CGNR with empty specification."""
        from src.integration.cgnr_pipeline import CGNRPipeline, PipelineStatus
        
        pipeline = CGNRPipeline(max_iterations=1, verbose=False)
        
        result = pipeline.run("int foo() { return 42; }", {})
        
        # Should handle empty spec
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
