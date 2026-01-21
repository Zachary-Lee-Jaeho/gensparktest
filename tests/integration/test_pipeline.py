"""
Integration tests for VEGA-Verified Pipeline.

Tests the complete verification workflow including:
- Specification inference
- Verification condition generation
- SMT-based verification
- CGNR repair
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.integration import (
    VEGAVerifiedPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    create_pipeline,
)
from src.specification import Specification, ConditionType
from src.specification.spec_language import Condition, Variable, Constant


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.enable_spec_inference is True
        assert config.verification_timeout_ms == 30000
        assert config.enable_repair is True
        assert config.max_repair_iterations == 5
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "enable_spec_inference": False,
            "verification_timeout_ms": 60000,
            "enable_repair": False,
        }
        
        config = PipelineConfig.from_dict(data)
        
        assert config.enable_spec_inference is False
        assert config.verification_timeout_ms == 60000
        assert config.enable_repair is False
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = PipelineConfig()
        data = config.to_dict()
        
        assert "enable_spec_inference" in data
        assert "verification_timeout_ms" in data
        assert "enable_repair" in data


class TestPipelineResult:
    """Tests for PipelineResult."""
    
    def test_empty_result(self):
        """Test empty result initialization."""
        result = PipelineResult(function_name="test_func")
        
        assert result.function_name == "test_func"
        assert result.final_status == "unknown"
        assert len(result.stages) == 0
    
    def test_verified_status(self):
        """Test verified status check."""
        result = PipelineResult(function_name="test_func")
        result.final_status = "verified"
        
        assert result.is_verified() is True
        assert result.is_repaired() is False
    
    def test_repaired_status(self):
        """Test repaired status check."""
        result = PipelineResult(function_name="test_func")
        result.final_status = "repaired"
        
        assert result.is_verified() is False
        assert result.is_repaired() is True
    
    def test_to_dict(self):
        """Test result serialization."""
        result = PipelineResult(function_name="test_func")
        result.final_status = "verified"
        result.total_time_ms = 100.5
        
        data = result.to_dict()
        
        assert data["function_name"] == "test_func"
        assert data["final_status"] == "verified"
        assert data["total_time_ms"] == 100.5
        assert data["is_verified"] is True
    
    def test_summary(self):
        """Test human-readable summary."""
        result = PipelineResult(function_name="test_func")
        result.final_status = "verified"
        
        summary = result.summary()
        
        assert "test_func" in summary
        assert "verified" in summary


class TestPipelineCreation:
    """Tests for pipeline creation."""
    
    def test_default_pipeline(self):
        """Test creating pipeline with defaults."""
        pipeline = VEGAVerifiedPipeline()
        
        assert pipeline.config is not None
        assert pipeline.verifier is not None
        assert pipeline.vcgen is not None
    
    def test_custom_config_pipeline(self):
        """Test creating pipeline with custom config."""
        config = PipelineConfig(
            verification_timeout_ms=60000,
            enable_repair=False,
        )
        
        pipeline = VEGAVerifiedPipeline(config)
        
        assert pipeline.config.verification_timeout_ms == 60000
        assert pipeline.cgnr is None  # Repair disabled
    
    def test_create_pipeline_factory(self):
        """Test factory function."""
        pipeline = create_pipeline(verbose=True)
        
        assert pipeline.config.verbose is True


class TestPipelineVerification:
    """Tests for pipeline verification functionality."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline for testing."""
        config = PipelineConfig(
            verbose=False,
            enable_repair=True,
        )
        return VEGAVerifiedPipeline(config)
    
    @pytest.fixture
    def simple_code(self):
        """Simple C++ function for testing."""
        return '''
        unsigned getRelocType(const MCFixup &Fixup) {
            switch (Fixup.getTargetKind()) {
                case FK_NONE:
                    return ELF::R_TARGET_NONE;
                case FK_Data_4:
                    return ELF::R_TARGET_32;
                case FK_Data_8:
                    return ELF::R_TARGET_64;
                default:
                    return ELF::R_TARGET_NONE;
            }
        }
        '''
    
    @pytest.fixture
    def buggy_code(self):
        """Buggy C++ function for testing repair."""
        return '''
        unsigned getRelocType(const MCFixup &Fixup) {
            switch (Fixup.getTargetKind()) {
                case FK_NONE:
                    return ELF::R_TARGET_NONE;
                case FK_Data_4:
                    return ELF::R_TARGET_64;  // Bug: wrong size
                default:
                    return ELF::R_TARGET_NONE;
            }
        }
        '''
    
    def test_verify_simple_function(self, pipeline, simple_code):
        """Test verifying a simple function."""
        result = pipeline.verify_function(
            code=simple_code,
            function_name="getRelocType"
        )
        
        assert result.function_name == "getRelocType"
        assert len(result.stages) > 0
        assert result.total_time_ms > 0
    
    def test_pipeline_stages(self, pipeline, simple_code):
        """Test that pipeline executes expected stages."""
        result = pipeline.verify_function(
            code=simple_code,
            function_name="getRelocType"
        )
        
        stage_names = [s.stage.value for s in result.stages]
        
        # At minimum, should have parse stage
        assert "parse" in stage_names
    
    def test_with_specification(self, pipeline, simple_code):
        """Test verification with provided specification."""
        spec = Specification(
            function_name="getRelocType",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("Fixup"))
            ],
            postconditions=[
                Condition(ConditionType.GREATER_EQUAL, Variable("result"), Constant(0))
            ]
        )
        
        result = pipeline.verify_function(
            code=simple_code,
            function_name="getRelocType",
            specification=spec
        )
        
        assert result.specification is not None
        assert result.specification.function_name == "getRelocType"
    
    def test_statistics(self, pipeline, simple_code):
        """Test pipeline statistics tracking."""
        # Reset stats
        pipeline.reset_statistics()
        
        # Run verification
        pipeline.verify_function(
            code=simple_code,
            function_name="getRelocType"
        )
        
        stats = pipeline.get_statistics()
        
        assert stats["total_runs"] == 1
        assert stats["total_time_ms"] > 0


class TestPipelineBatch:
    """Tests for batch verification."""
    
    @pytest.fixture
    def pipeline(self):
        config = PipelineConfig(verbose=False)
        return VEGAVerifiedPipeline(config)
    
    def test_verify_batch(self, pipeline):
        """Test batch verification."""
        functions = [
            ("func1", "unsigned func1() { return 0; }"),
            ("func2", "unsigned func2() { return 1; }"),
        ]
        
        results = pipeline.verify_batch(functions)
        
        assert "func1" in results
        assert "func2" in results
        assert results["func1"].function_name == "func1"
        assert results["func2"].function_name == "func2"


class TestIntegrationScenarios:
    """End-to-end integration scenarios."""
    
    @pytest.fixture
    def pipeline(self):
        config = PipelineConfig(
            enable_spec_inference=True,
            enable_repair=True,
            verbose=False,
        )
        return VEGAVerifiedPipeline(config)
    
    def test_full_pipeline_flow(self, pipeline):
        """Test complete pipeline from parsing to verification."""
        code = '''
        unsigned computeSize(int kind) {
            if (kind < 0) return 0;
            return kind * 4;
        }
        '''
        
        result = pipeline.verify_function(
            code=code,
            function_name="computeSize"
        )
        
        # Pipeline should complete without errors
        assert result.final_status != "parse_error"
        assert result.total_time_ms > 0
    
    def test_spec_inference_with_references(self, pipeline):
        """Test specification inference with reference implementations."""
        target_code = '''
        unsigned getRelocType(int kind) {
            switch (kind) {
                case 0: return 0;
                case 4: return 32;
                default: return 0;
            }
        }
        '''
        
        references = [
            ("ARM", '''
            unsigned getRelocType(int kind) {
                switch (kind) {
                    case 0: return ARM_NONE;
                    case 4: return ARM_32;
                    default: return ARM_NONE;
                }
            }
            '''),
            ("MIPS", '''
            unsigned getRelocType(int kind) {
                switch (kind) {
                    case 0: return MIPS_NONE;
                    case 4: return MIPS_32;
                    default: return MIPS_NONE;
                }
            }
            '''),
        ]
        
        result = pipeline.verify_function(
            code=target_code,
            function_name="getRelocType",
            references=references
        )
        
        # Should have specification inferred
        if result.specification:
            assert result.specification.function_name == "getRelocType"


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
