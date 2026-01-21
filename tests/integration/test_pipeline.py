"""
Integration tests for the full VEGA-Verified pipeline.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.integration.vega_adapter import VEGAAdapter, VEGAMode
from src.integration.llvm_adapter import LLVMAdapter
from src.integration.experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentMode,
)
from src.specification.spec_language import Specification, Condition, Variable, Constant
from src.verification.verifier import Verifier
from src.repair.cgnr import CGNREngine


class TestVEGAAdapter:
    """Tests for VEGA adapter."""
    
    def test_adapter_creation(self):
        """Test adapter creation."""
        adapter = VEGAAdapter(mode=VEGAMode.SIMULATION)
        assert adapter is not None
        assert adapter.mode == VEGAMode.SIMULATION
    
    def test_generation(self):
        """Test code generation."""
        adapter = VEGAAdapter(
            mode=VEGAMode.SIMULATION,
            target="RISCV"
        )
        
        result = adapter.generate("getRelocType", "ELFObjectWriter")
        
        assert result.function_name == "getRelocType"
        assert result.generated_code is not None
        assert len(result.generated_code) > 0
    
    def test_batch_generation(self):
        """Test batch code generation."""
        adapter = VEGAAdapter(mode=VEGAMode.SIMULATION)
        
        functions = [
            ("getRelocType", "ELFObjectWriter"),
            ("encodeInstruction", "MCCodeEmitter"),
        ]
        
        results = adapter.batch_generate(functions)
        
        assert len(results) == 2
        assert all(r.generated_code for r in results)
    
    def test_statistics(self):
        """Test generation statistics."""
        adapter = VEGAAdapter(mode=VEGAMode.SIMULATION)
        
        # Generate some code
        adapter.generate("getRelocType", "ELFObjectWriter")
        adapter.generate("encodeInstruction", "MCCodeEmitter")
        
        stats = adapter.get_statistics()
        
        assert stats["generations"] == 2
    
    def test_target_change(self):
        """Test changing target architecture."""
        adapter = VEGAAdapter(mode=VEGAMode.SIMULATION, target="RISCV")
        
        result1 = adapter.generate("getRelocType", "ELFObjectWriter")
        assert "RISCV" in result1.generated_code
        
        adapter.set_target("ARM")
        result2 = adapter.generate("getRelocType", "ELFObjectWriter")
        assert "ARM" in result2.generated_code


class TestLLVMAdapter:
    """Tests for LLVM adapter."""
    
    def test_adapter_creation(self):
        """Test adapter creation."""
        adapter = LLVMAdapter()
        assert adapter is not None
    
    def test_get_backend_info(self):
        """Test getting backend information."""
        adapter = LLVMAdapter()
        
        info = adapter.get_backend_info("RISCV")
        
        assert info.name == "RISCV"
        assert "MCCodeEmitter" in info.modules
        assert info.total_functions > 0
    
    def test_supported_targets(self):
        """Test getting supported targets."""
        adapter = LLVMAdapter()
        
        targets = adapter.get_supported_targets()
        
        assert "RISCV" in targets
        assert "ARM" in targets
    
    def test_reference_backends(self):
        """Test getting reference backends."""
        adapter = LLVMAdapter()
        
        refs = adapter.get_reference_backends("RISCV")
        
        assert "RISCV" not in refs
        assert len(refs) >= 1


class TestFullPipeline:
    """Tests for the complete pipeline."""
    
    def test_vega_only_pipeline(self):
        """Test VEGA-only mode."""
        config = ExperimentConfig(
            name="test_vega_only",
            mode=ExperimentMode.VEGA_ONLY,
            target="RISCV",
            output_dir="/tmp/vega_test",
            modules=["MCCodeEmitter"],
        )
        
        runner = ExperimentRunner(config, verbose=False)
        result = runner.run()
        
        assert result.vega_results
        assert "vega" in result.statistics
    
    def test_vega_verified_pipeline(self):
        """Test VEGA-Verified mode."""
        config = ExperimentConfig(
            name="test_vega_verified",
            mode=ExperimentMode.VEGA_VERIFIED,
            target="RISCV",
            output_dir="/tmp/vega_test",
            modules=["ELFObjectWriter"],
            enable_verification=True,
            enable_repair=True,
        )
        
        runner = ExperimentRunner(config, verbose=False)
        result = runner.run()
        
        assert result.vega_verified_results
        assert "vega_verified" in result.statistics
    
    def test_comparison_pipeline(self):
        """Test comparison mode."""
        config = ExperimentConfig(
            name="test_comparison",
            mode=ExperimentMode.COMPARISON,
            target="RISCV",
            output_dir="/tmp/vega_test",
            modules=["ELFObjectWriter"],
            seed=42,  # Reproducible results
        )
        
        runner = ExperimentRunner(config, verbose=False)
        result = runner.run()
        
        assert result.vega_results
        assert result.vega_verified_results
        assert "comparison" in result.statistics


class TestVerificationPipeline:
    """Tests for verification components in pipeline."""
    
    def test_verification_integration(self):
        """Test verifier integration."""
        verifier = Verifier(timeout_ms=5000)
        
        code = """
        unsigned getRelocType(int kind) {
            switch (kind) {
            case 0: return 0;
            default: return 1;
            }
        }
        """
        
        spec = Specification(
            function_name="getRelocType",
            module="Test",
            postconditions=[
                Condition.ge(Variable("result"), Constant(0))
            ]
        )
        
        result = verifier.verify(code, spec)
        
        assert result is not None
        assert result.time_ms > 0
    
    def test_cgnr_integration(self):
        """Test CGNR repair integration."""
        verifier = Verifier(timeout_ms=5000)
        cgnr = CGNREngine(verifier=verifier, max_iterations=3, verbose=False)
        
        code = """
        int buggy(int x) {
            return x;  // Should return abs(x)
        }
        """
        
        spec = Specification(
            function_name="buggy",
            module="Test",
            postconditions=[
                Condition.ge(Variable("result"), Constant(0))
            ]
        )
        
        result = cgnr.repair(code, spec)
        
        assert result is not None
        assert result.iterations > 0


class TestExperimentConfig:
    """Tests for experiment configuration."""
    
    def test_config_creation(self):
        """Test config creation."""
        config = ExperimentConfig(
            name="test",
            mode=ExperimentMode.COMPARISON,
            target="RISCV"
        )
        
        assert config.name == "test"
        assert config.mode == ExperimentMode.COMPARISON
    
    def test_config_serialization(self):
        """Test config to/from dict."""
        config = ExperimentConfig(
            name="test",
            mode=ExperimentMode.VEGA_VERIFIED,
            target="ARM",
            enable_repair=False,
        )
        
        data = config.to_dict()
        loaded = ExperimentConfig.from_dict(data)
        
        assert loaded.name == config.name
        assert loaded.mode == config.mode
        assert loaded.target == config.target
        assert loaded.enable_repair == config.enable_repair


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
