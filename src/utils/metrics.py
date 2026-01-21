"""
Metrics collection and comparison for VEGA vs VEGA-Verified experiments.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum


class MetricType(Enum):
    """Types of metrics collected."""
    ACCURACY = "accuracy"
    TIME = "time"
    COUNT = "count"
    RATE = "rate"
    COMPARISON = "comparison"


@dataclass
class FunctionMetrics:
    """Metrics for a single function."""
    function_name: str
    module: str
    
    # Generation metrics
    generation_time_ms: float = 0.0
    generated_successfully: bool = False
    
    # Verification metrics (VEGA-Verified only)
    verification_time_ms: float = 0.0
    verified: bool = False
    verification_status: str = "not_run"  # not_run, verified, failed, timeout
    counterexample_found: bool = False
    
    # Repair metrics (VEGA-Verified only)
    repair_iterations: int = 0
    repair_time_ms: float = 0.0
    repaired_successfully: bool = False
    
    # Confidence scores (VEGA)
    vega_confidence: float = 0.0
    
    # Ground truth comparison
    matches_ground_truth: bool = False
    edit_distance_to_ground_truth: int = -1


@dataclass
class ModuleMetrics:
    """Aggregated metrics for a module."""
    module_name: str
    
    total_functions: int = 0
    generated_functions: int = 0
    verified_functions: int = 0
    repaired_functions: int = 0
    
    total_generation_time_ms: float = 0.0
    total_verification_time_ms: float = 0.0
    total_repair_time_ms: float = 0.0
    
    function_metrics: List[FunctionMetrics] = field(default_factory=list)
    
    @property
    def generation_accuracy(self) -> float:
        """Percentage of functions generated successfully."""
        if self.total_functions == 0:
            return 0.0
        return self.generated_functions / self.total_functions
    
    @property
    def verification_rate(self) -> float:
        """Percentage of functions that passed verification."""
        if self.total_functions == 0:
            return 0.0
        return self.verified_functions / self.total_functions
    
    @property
    def repair_success_rate(self) -> float:
        """Percentage of failed verifications that were repaired."""
        failed = self.generated_functions - self.verified_functions
        if failed == 0:
            return 1.0  # No failures to repair
        return self.repaired_functions / failed


@dataclass
class ExperimentMetrics:
    """Complete metrics for an experiment run."""
    experiment_id: str
    mode: str  # vega | vega-verified | verify-only
    target: str
    
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    
    # Summary metrics
    total_functions: int = 0
    total_modules: int = 0
    
    # Per-module metrics
    module_metrics: Dict[str, ModuleMetrics] = field(default_factory=dict)
    
    # Aggregated metrics
    overall_generation_accuracy: float = 0.0
    overall_verification_rate: float = 0.0
    overall_repair_success_rate: float = 0.0
    
    # Time metrics
    total_generation_time_ms: float = 0.0
    total_verification_time_ms: float = 0.0
    total_repair_time_ms: float = 0.0
    
    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from module metrics."""
        if not self.module_metrics:
            return
        
        total_generated = 0
        total_verified = 0
        total_repaired = 0
        total_funcs = 0
        
        for module in self.module_metrics.values():
            total_funcs += module.total_functions
            total_generated += module.generated_functions
            total_verified += module.verified_functions
            total_repaired += module.repaired_functions
            
            self.total_generation_time_ms += module.total_generation_time_ms
            self.total_verification_time_ms += module.total_verification_time_ms
            self.total_repair_time_ms += module.total_repair_time_ms
        
        self.total_functions = total_funcs
        self.total_modules = len(self.module_metrics)
        
        if total_funcs > 0:
            self.overall_generation_accuracy = total_generated / total_funcs
            self.overall_verification_rate = total_verified / total_funcs
        
        failed = total_generated - total_verified
        if failed > 0:
            self.overall_repair_success_rate = total_repaired / failed
        elif total_generated > 0:
            self.overall_repair_success_rate = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "experiment_id": self.experiment_id,
            "mode": self.mode,
            "target": self.target,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "summary": {
                "total_functions": self.total_functions,
                "total_modules": self.total_modules,
                "overall_generation_accuracy": self.overall_generation_accuracy,
                "overall_verification_rate": self.overall_verification_rate,
                "overall_repair_success_rate": self.overall_repair_success_rate,
            },
            "time_metrics": {
                "total_generation_time_ms": self.total_generation_time_ms,
                "total_verification_time_ms": self.total_verification_time_ms,
                "total_repair_time_ms": self.total_repair_time_ms,
            },
            "modules": {
                name: {
                    "total_functions": m.total_functions,
                    "generated_functions": m.generated_functions,
                    "verified_functions": m.verified_functions,
                    "repaired_functions": m.repaired_functions,
                    "generation_accuracy": m.generation_accuracy,
                    "verification_rate": m.verification_rate,
                }
                for name, m in self.module_metrics.items()
            }
        }
        return result


@dataclass
class ComparisonResult:
    """Result of comparing two experiment runs."""
    baseline_mode: str
    compared_mode: str
    target: str
    
    # Accuracy improvements
    accuracy_improvement: float = 0.0  # percentage points
    verification_rate_improvement: float = 0.0
    
    # Time overhead
    time_overhead_percent: float = 0.0
    
    # Per-module comparison
    module_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Detailed improvements
    functions_improved: int = 0
    functions_degraded: int = 0
    functions_unchanged: int = 0


class MetricsCollector:
    """
    Collects and manages metrics for VEGA vs VEGA-Verified comparison.
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments: Dict[str, ExperimentMetrics] = {}
        self.current_experiment: Optional[ExperimentMetrics] = None
        
    def start_experiment(self, experiment_id: str, mode: str, target: str) -> ExperimentMetrics:
        """Start a new experiment and return its metrics object."""
        metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            mode=mode,
            target=target,
            start_time=datetime.utcnow().isoformat()
        )
        self.experiments[experiment_id] = metrics
        self.current_experiment = metrics
        return metrics
    
    def end_experiment(self, experiment_id: Optional[str] = None) -> ExperimentMetrics:
        """End an experiment and compute final metrics."""
        if experiment_id:
            metrics = self.experiments[experiment_id]
        else:
            metrics = self.current_experiment
        
        if metrics is None:
            raise ValueError("No active experiment")
        
        metrics.end_time = datetime.utcnow().isoformat()
        
        # Parse times and compute duration
        start = datetime.fromisoformat(metrics.start_time)
        end = datetime.fromisoformat(metrics.end_time)
        metrics.duration_seconds = (end - start).total_seconds()
        
        metrics.compute_aggregates()
        return metrics
    
    def add_function_metrics(
        self,
        function_name: str,
        module_name: str,
        metrics: FunctionMetrics,
        experiment_id: Optional[str] = None
    ) -> None:
        """Add metrics for a function."""
        exp = self.experiments.get(experiment_id) or self.current_experiment
        if exp is None:
            raise ValueError("No active experiment")
        
        # Get or create module metrics
        if module_name not in exp.module_metrics:
            exp.module_metrics[module_name] = ModuleMetrics(module_name=module_name)
        
        module = exp.module_metrics[module_name]
        module.function_metrics.append(metrics)
        module.total_functions += 1
        
        # Update aggregates
        if metrics.generated_successfully:
            module.generated_functions += 1
        if metrics.verified:
            module.verified_functions += 1
        if metrics.repaired_successfully:
            module.repaired_functions += 1
        
        module.total_generation_time_ms += metrics.generation_time_ms
        module.total_verification_time_ms += metrics.verification_time_ms
        module.total_repair_time_ms += metrics.repair_time_ms
    
    def compare_experiments(
        self,
        baseline_id: str,
        compared_id: str
    ) -> ComparisonResult:
        """Compare two experiments."""
        baseline = self.experiments[baseline_id]
        compared = self.experiments[compared_id]
        
        result = ComparisonResult(
            baseline_mode=baseline.mode,
            compared_mode=compared.mode,
            target=baseline.target
        )
        
        # Compute improvements
        result.accuracy_improvement = (
            compared.overall_generation_accuracy - baseline.overall_generation_accuracy
        ) * 100  # Convert to percentage points
        
        result.verification_rate_improvement = (
            compared.overall_verification_rate - baseline.overall_verification_rate
        ) * 100
        
        # Time overhead
        baseline_time = baseline.total_generation_time_ms
        compared_time = (
            compared.total_generation_time_ms +
            compared.total_verification_time_ms +
            compared.total_repair_time_ms
        )
        
        if baseline_time > 0:
            result.time_overhead_percent = (
                (compared_time - baseline_time) / baseline_time
            ) * 100
        
        # Per-module comparison
        all_modules = set(baseline.module_metrics.keys()) | set(compared.module_metrics.keys())
        for module_name in all_modules:
            base_module = baseline.module_metrics.get(module_name)
            comp_module = compared.module_metrics.get(module_name)
            
            if base_module and comp_module:
                result.module_comparisons[module_name] = {
                    "accuracy_improvement": (
                        comp_module.generation_accuracy - base_module.generation_accuracy
                    ) * 100,
                    "verification_rate": comp_module.verification_rate * 100,
                }
        
        return result
    
    def save_metrics(self, experiment_id: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Save metrics to file."""
        exp = self.experiments.get(experiment_id) or self.current_experiment
        if exp is None:
            raise ValueError("No experiment to save")
        
        if filename is None:
            filename = f"{exp.experiment_id}_{exp.mode}_metrics.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(exp.to_dict(), f, indent=2)
        
        return str(filepath)
    
    def load_metrics(self, filepath: str) -> ExperimentMetrics:
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metrics = ExperimentMetrics(
            experiment_id=data["experiment_id"],
            mode=data["mode"],
            target=data["target"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            duration_seconds=data["duration_seconds"]
        )
        
        # Load summary
        summary = data.get("summary", {})
        metrics.total_functions = summary.get("total_functions", 0)
        metrics.total_modules = summary.get("total_modules", 0)
        metrics.overall_generation_accuracy = summary.get("overall_generation_accuracy", 0.0)
        metrics.overall_verification_rate = summary.get("overall_verification_rate", 0.0)
        metrics.overall_repair_success_rate = summary.get("overall_repair_success_rate", 0.0)
        
        # Load time metrics
        time_metrics = data.get("time_metrics", {})
        metrics.total_generation_time_ms = time_metrics.get("total_generation_time_ms", 0.0)
        metrics.total_verification_time_ms = time_metrics.get("total_verification_time_ms", 0.0)
        metrics.total_repair_time_ms = time_metrics.get("total_repair_time_ms", 0.0)
        
        self.experiments[metrics.experiment_id] = metrics
        return metrics
    
    def generate_comparison_report(
        self,
        baseline_id: str,
        compared_id: str
    ) -> str:
        """Generate a markdown comparison report."""
        comparison = self.compare_experiments(baseline_id, compared_id)
        baseline = self.experiments[baseline_id]
        compared = self.experiments[compared_id]
        
        report = f"""# VEGA vs VEGA-Verified Comparison Report

## Overview
- **Target**: {comparison.target}
- **Baseline Mode**: {comparison.baseline_mode}
- **Compared Mode**: {comparison.compared_mode}

## Summary

| Metric | {comparison.baseline_mode} | {comparison.compared_mode} | Improvement |
|--------|------|------|-------------|
| Generation Accuracy | {baseline.overall_generation_accuracy*100:.1f}% | {compared.overall_generation_accuracy*100:.1f}% | {comparison.accuracy_improvement:+.1f}pp |
| Verification Rate | N/A | {compared.overall_verification_rate*100:.1f}% | - |
| Repair Success Rate | N/A | {compared.overall_repair_success_rate*100:.1f}% | - |

## Time Analysis

| Metric | {comparison.baseline_mode} | {comparison.compared_mode} |
|--------|------|------|
| Generation Time | {baseline.total_generation_time_ms:.0f}ms | {compared.total_generation_time_ms:.0f}ms |
| Verification Time | N/A | {compared.total_verification_time_ms:.0f}ms |
| Repair Time | N/A | {compared.total_repair_time_ms:.0f}ms |
| **Total Time** | {baseline.total_generation_time_ms:.0f}ms | {compared.total_generation_time_ms + compared.total_verification_time_ms + compared.total_repair_time_ms:.0f}ms |
| **Overhead** | - | {comparison.time_overhead_percent:+.1f}% |

## Per-Module Results

| Module | {comparison.baseline_mode} Accuracy | {comparison.compared_mode} Accuracy | Improvement |
|--------|------|------|-------------|
"""
        
        for module_name, comp in comparison.module_comparisons.items():
            base_acc = baseline.module_metrics.get(module_name)
            comp_acc = compared.module_metrics.get(module_name)
            
            base_val = base_acc.generation_accuracy * 100 if base_acc else 0
            comp_val = comp_acc.generation_accuracy * 100 if comp_acc else 0
            
            report += f"| {module_name} | {base_val:.1f}% | {comp_val:.1f}% | {comp['accuracy_improvement']:+.1f}pp |\n"
        
        report += "\n---\n*Generated by VEGA-Verified Metrics System*\n"
        
        return report
