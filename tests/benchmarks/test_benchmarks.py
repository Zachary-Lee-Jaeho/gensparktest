"""
Unit tests for VEGA-Verified benchmark suite.

Tests:
1. Processor backend benchmark creation
2. VEGA paper targets (RI5CY, xCORE)
3. Comparison framework
4. Metrics calculation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest


class TestProcessorBackends(unittest.TestCase):
    """Test processor backend benchmarks."""
    
    def test_get_all_benchmarks(self):
        """Test getting all available benchmarks."""
        from tests.benchmarks.processor_backends import get_all_benchmarks
        
        benchmarks = get_all_benchmarks()
        
        self.assertIn("RISCV", benchmarks)
        self.assertIn("ARM", benchmarks)
        self.assertIn("AArch64", benchmarks)
        self.assertIn("MIPS", benchmarks)
        self.assertIn("X86_64", benchmarks)
        self.assertIn("PowerPC", benchmarks)
    
    def test_riscv_benchmark_structure(self):
        """Test RISC-V benchmark structure."""
        from tests.benchmarks.processor_backends import create_riscv_benchmark
        
        benchmark = create_riscv_benchmark()
        
        self.assertEqual(benchmark.name, "RISCV")
        self.assertEqual(benchmark.triple, "riscv64-unknown-linux-gnu")
        self.assertIn("MCCodeEmitter", benchmark.modules)
        self.assertIn("ELFObjectWriter", benchmark.modules)
        
        # Check expected metrics
        self.assertIn("vega_function_accuracy", benchmark.expected_metrics)
        self.assertAlmostEqual(
            benchmark.expected_metrics["vega_function_accuracy"], 
            0.715, 
            places=3
        )
    
    def test_benchmark_to_backend(self):
        """Test converting benchmark to Backend object."""
        from tests.benchmarks.processor_backends import create_riscv_benchmark
        
        benchmark = create_riscv_benchmark()
        backend = benchmark.to_backend()
        
        self.assertEqual(backend.name, "RISCV")
        self.assertEqual(backend.target_triple, "riscv64-unknown-linux-gnu")
        self.assertGreater(len(backend.modules), 0)


class TestVEGAPaperTargets(unittest.TestCase):
    """Test VEGA paper targets (RI5CY, xCORE)."""
    
    def test_ri5cy_benchmark(self):
        """Test RI5CY benchmark creation."""
        from tests.benchmarks.vega_paper_targets import create_ri5cy_benchmark
        
        benchmark = create_ri5cy_benchmark()
        
        self.assertEqual(benchmark.name, "RI5CY")
        self.assertIn("pulp", benchmark.triple.lower())
        
        # Check modules
        self.assertIn("MCCodeEmitter", benchmark.modules)
        self.assertIn("ISelDAGToDAG", benchmark.modules)
        
        # Check PULP-specific functions
        mc_emitter = benchmark.modules["MCCodeEmitter"]
        self.assertIn("getPULPBinaryCode", mc_emitter.functions)
        self.assertIn("encodeLpSetupImm", mc_emitter.functions)
        
        # Check expected metrics from VEGA paper
        self.assertAlmostEqual(
            benchmark.expected_metrics["vega_function_accuracy"],
            0.732,
            places=3
        )
    
    def test_xcore_benchmark(self):
        """Test xCORE benchmark creation."""
        from tests.benchmarks.vega_paper_targets import create_xcore_benchmark
        
        benchmark = create_xcore_benchmark()
        
        self.assertEqual(benchmark.name, "xCORE")
        self.assertIn("xcore", benchmark.triple.lower())
        
        # Check modules
        self.assertIn("MCCodeEmitter", benchmark.modules)
        self.assertIn("ISelDAGToDAG", benchmark.modules)
        
        # Check xCORE-specific functions
        mc_emitter = benchmark.modules["MCCodeEmitter"]
        self.assertIn("encodeChannelOp", mc_emitter.functions)
        
        # Check expected metrics from VEGA paper
        self.assertAlmostEqual(
            benchmark.expected_metrics["vega_function_accuracy"],
            0.622,
            places=3
        )
    
    def test_get_vega_paper_targets(self):
        """Test getting all VEGA paper targets."""
        from tests.benchmarks.vega_paper_targets import get_vega_paper_targets
        
        targets = get_vega_paper_targets()
        
        self.assertIn("RISCV", targets)
        self.assertIn("RI5CY", targets)
        self.assertIn("xCORE", targets)
        self.assertEqual(len(targets), 3)
    
    def test_vega_paper_metrics(self):
        """Test VEGA paper metrics retrieval."""
        from tests.benchmarks.vega_paper_targets import get_vega_paper_metrics
        
        metrics = get_vega_paper_metrics()
        
        # Check RISC-V metrics from paper
        self.assertIn("RISCV", metrics)
        self.assertAlmostEqual(
            metrics["RISCV"]["vega_function_accuracy"],
            0.715,
            places=3
        )
        self.assertAlmostEqual(
            metrics["RISCV"]["vega_statement_accuracy"],
            0.55,
            places=2
        )
        
        # Check all three targets have fork_flow_accuracy
        for target in ["RISCV", "RI5CY", "xCORE"]:
            self.assertIn("fork_flow_accuracy", metrics[target])


class TestComparisonFramework(unittest.TestCase):
    """Test comparison framework."""
    
    def test_function_metrics(self):
        """Test FunctionMetrics dataclass."""
        from tests.benchmarks.comparison_framework import FunctionMetrics
        
        metrics = FunctionMetrics(
            function_name="testFunc",
            module_name="TestModule"
        )
        
        self.assertEqual(metrics.function_name, "testFunc")
        self.assertFalse(metrics.vega_generated)
        self.assertFalse(metrics.verification_passed)
        
        # Test to_dict
        d = metrics.to_dict()
        self.assertIn("function_name", d)
        self.assertIn("module_name", d)
    
    def test_module_metrics_aggregation(self):
        """Test ModuleMetrics aggregation."""
        from tests.benchmarks.comparison_framework import (
            FunctionMetrics, ModuleMetrics
        )
        
        module_metrics = ModuleMetrics(module_name="TestModule")
        
        # Add some functions
        f1 = FunctionMetrics("func1", "TestModule")
        f1.vega_syntactically_correct = True
        f1.verification_passed = True
        
        f2 = FunctionMetrics("func2", "TestModule")
        f2.vega_syntactically_correct = True
        f2.repair_attempted = True
        f2.repair_succeeded = True
        
        f3 = FunctionMetrics("func3", "TestModule")
        f3.vega_syntactically_correct = False
        
        module_metrics.functions = [f1, f2, f3]
        
        # Test properties
        self.assertEqual(module_metrics.total_functions, 3)
        self.assertAlmostEqual(module_metrics.vega_accuracy, 2/3, places=3)
        self.assertAlmostEqual(module_metrics.verified_accuracy, 2/3, places=3)
        self.assertAlmostEqual(module_metrics.repair_success_rate, 1.0, places=3)
    
    def test_backend_metrics(self):
        """Test BackendMetrics."""
        from tests.benchmarks.comparison_framework import (
            FunctionMetrics, ModuleMetrics, BackendMetrics
        )
        
        backend = BackendMetrics(
            backend_name="TestBackend",
            target_triple="test-triple"
        )
        
        module = ModuleMetrics(module_name="TestModule")
        f1 = FunctionMetrics("func1", "TestModule")
        f1.vega_syntactically_correct = True
        f1.verification_passed = True
        f1.specification_inferred = True
        module.functions = [f1]
        
        backend.modules = [module]
        
        self.assertEqual(backend.total_functions, 1)
        self.assertAlmostEqual(backend.vega_function_accuracy, 1.0, places=3)
        self.assertAlmostEqual(backend.vega_verified_accuracy, 1.0, places=3)
        self.assertAlmostEqual(backend.specification_coverage, 1.0, places=3)
    
    def test_comparison_framework_initialization(self):
        """Test ComparisonFramework initialization."""
        from tests.benchmarks.comparison_framework import ComparisonFramework
        
        framework = ComparisonFramework(verbose=False)
        
        self.assertIsNotNone(framework.verifier)
        self.assertIsNotNone(framework.cgnr_engine)
        self.assertIsNotNone(framework.function_verifier)
        self.assertIsNotNone(framework.module_verifier)
        self.assertIsNotNone(framework.backend_verifier)
    
    def test_evaluate_function(self):
        """Test single function evaluation."""
        from tests.benchmarks.comparison_framework import (
            ComparisonFramework, EvaluationMode
        )
        from src.specification.spec_language import Specification
        
        framework = ComparisonFramework(verbose=False)
        
        code = "unsigned testFunc() { return 42; }"
        spec = Specification(function_name="testFunc")
        
        metrics = framework.evaluate_function(
            code=code,
            spec=spec,
            function_name="testFunc",
            module_name="TestModule",
            mode=EvaluationMode.VEGA_VERIFIED
        )
        
        self.assertEqual(metrics.function_name, "testFunc")
        self.assertTrue(metrics.vega_generated)
        self.assertTrue(metrics.specification_inferred)
        self.assertTrue(metrics.verification_attempted)


class TestProcessorFamilies(unittest.TestCase):
    """Test processor family enumeration."""
    
    def test_processor_family_enum(self):
        """Test ProcessorFamily enum."""
        from tests.benchmarks.processor_backends import ProcessorFamily
        
        self.assertEqual(ProcessorFamily.RISCV.value, "riscv")
        self.assertEqual(ProcessorFamily.ARM.value, "arm")
        self.assertEqual(ProcessorFamily.AARCH64.value, "aarch64")
        self.assertEqual(ProcessorFamily.MIPS.value, "mips")
        self.assertEqual(ProcessorFamily.X86_64.value, "x86_64")
    
    def test_benchmark_family_assignment(self):
        """Test that benchmarks have correct family."""
        from tests.benchmarks.processor_backends import (
            ProcessorFamily,
            create_riscv_benchmark,
            create_arm_benchmark,
            create_mips_benchmark,
        )
        
        self.assertEqual(create_riscv_benchmark().family, ProcessorFamily.RISCV)
        self.assertEqual(create_arm_benchmark().family, ProcessorFamily.ARM)
        self.assertEqual(create_mips_benchmark().family, ProcessorFamily.MIPS)


class TestSpecificationIntegration(unittest.TestCase):
    """Test specification integration in benchmarks."""
    
    def test_function_has_specification(self):
        """Test that benchmark functions have specifications."""
        from tests.benchmarks.processor_backends import create_riscv_benchmark
        
        benchmark = create_riscv_benchmark()
        mc_emitter = benchmark.modules["MCCodeEmitter"]
        
        # Check that functions have specifications
        for func in mc_emitter.functions.values():
            if func.is_interface:
                self.assertIsNotNone(
                    func.specification,
                    f"Function {func.name} missing specification"
                )
    
    def test_interface_contract_creation(self):
        """Test interface contract creation."""
        from tests.benchmarks.processor_backends import create_riscv_benchmark
        
        benchmark = create_riscv_benchmark()
        mc_emitter = benchmark.modules["MCCodeEmitter"]
        
        self.assertIsNotNone(mc_emitter.interface_contract)
        self.assertTrue(len(mc_emitter.interface_contract.assumptions) > 0)
        self.assertTrue(len(mc_emitter.interface_contract.guarantees) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
