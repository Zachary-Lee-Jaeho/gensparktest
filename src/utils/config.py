"""
Configuration management for VEGA-Verified.
Supports YAML configuration files with mode-specific settings.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class SpecificationConfig:
    """Configuration for specification inference."""
    symbolic_execution_max_depth: int = 100
    symbolic_execution_timeout_ms: int = 5000
    pattern_min_similarity: float = 0.7
    include_null_checks: bool = True
    include_bounds_checks: bool = True


@dataclass
class VerificationConfig:
    """Configuration for formal verification."""
    solver: str = "z3"
    timeout_ms: int = 30000
    bmc_bound: int = 10
    incremental: bool = True


@dataclass
class RepairConfig:
    """Configuration for CGNR repair."""
    max_iterations: int = 5
    model_path: str = "models/repair_model"
    beam_size: int = 5
    temperature: float = 0.7
    use_repair: bool = True


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical verification."""
    levels: List[str] = field(default_factory=lambda: ["function", "module", "backend"])
    parallel_verification: bool = True
    max_workers: int = 4


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    output_dir: str = "results"
    save_intermediate: bool = True
    compare_modes: List[str] = field(default_factory=lambda: ["vega", "vega-verified"])
    metrics_file: str = "metrics.json"


@dataclass
class Config:
    """Main configuration class for VEGA-Verified."""
    
    # Execution mode
    mode: str = "vega-verified"  # vega | vega-verified | verify-only
    
    # Target configuration
    target: str = "riscv"
    references: List[str] = field(default_factory=lambda: ["ARM", "MIPS", "X86"])
    
    # Sub-configurations
    specification: SpecificationConfig = field(default_factory=SpecificationConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def is_vega_only(self) -> bool:
        """Check if running in VEGA-only mode."""
        return self.mode == "vega"
    
    def is_verified(self) -> bool:
        """Check if verification is enabled."""
        return self.mode in ("vega-verified", "verify-only")
    
    def is_repair_enabled(self) -> bool:
        """Check if CGNR repair is enabled."""
        return self.mode == "vega-verified" and self.repair.use_repair
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode,
            "target": self.target,
            "references": self.references,
            "specification": {
                "symbolic_execution_max_depth": self.specification.symbolic_execution_max_depth,
                "symbolic_execution_timeout_ms": self.specification.symbolic_execution_timeout_ms,
                "pattern_min_similarity": self.specification.pattern_min_similarity,
                "include_null_checks": self.specification.include_null_checks,
                "include_bounds_checks": self.specification.include_bounds_checks,
            },
            "verification": {
                "solver": self.verification.solver,
                "timeout_ms": self.verification.timeout_ms,
                "bmc_bound": self.verification.bmc_bound,
                "incremental": self.verification.incremental,
            },
            "repair": {
                "max_iterations": self.repair.max_iterations,
                "model_path": self.repair.model_path,
                "beam_size": self.repair.beam_size,
                "temperature": self.repair.temperature,
                "use_repair": self.repair.use_repair,
            },
            "hierarchical": {
                "levels": self.hierarchical.levels,
                "parallel_verification": self.hierarchical.parallel_verification,
                "max_workers": self.hierarchical.max_workers,
            },
            "experiment": {
                "output_dir": self.experiment.output_dir,
                "save_intermediate": self.experiment.save_intermediate,
                "compare_modes": self.experiment.compare_modes,
                "metrics_file": self.experiment.metrics_file,
            },
            "log_level": self.log_level,
            "log_file": self.log_file,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config = cls()
        
        # Top-level settings
        config.mode = data.get("mode", config.mode)
        config.target = data.get("target", config.target)
        config.references = data.get("references", config.references)
        config.log_level = data.get("log_level", config.log_level)
        config.log_file = data.get("log_file", config.log_file)
        
        # Sub-configurations
        if "specification" in data:
            spec_data = data["specification"]
            config.specification = SpecificationConfig(
                symbolic_execution_max_depth=spec_data.get("symbolic_execution_max_depth", 100),
                symbolic_execution_timeout_ms=spec_data.get("symbolic_execution_timeout_ms", 5000),
                pattern_min_similarity=spec_data.get("pattern_min_similarity", 0.7),
                include_null_checks=spec_data.get("include_null_checks", True),
                include_bounds_checks=spec_data.get("include_bounds_checks", True),
            )
        
        if "verification" in data:
            ver_data = data["verification"]
            config.verification = VerificationConfig(
                solver=ver_data.get("solver", "z3"),
                timeout_ms=ver_data.get("timeout_ms", 30000),
                bmc_bound=ver_data.get("bmc_bound", 10),
                incremental=ver_data.get("incremental", True),
            )
        
        if "repair" in data:
            rep_data = data["repair"]
            config.repair = RepairConfig(
                max_iterations=rep_data.get("max_iterations", 5),
                model_path=rep_data.get("model_path", "models/repair_model"),
                beam_size=rep_data.get("beam_size", 5),
                temperature=rep_data.get("temperature", 0.7),
                use_repair=rep_data.get("use_repair", True),
            )
        
        if "hierarchical" in data:
            hier_data = data["hierarchical"]
            config.hierarchical = HierarchicalConfig(
                levels=hier_data.get("levels", ["function", "module", "backend"]),
                parallel_verification=hier_data.get("parallel_verification", True),
                max_workers=hier_data.get("max_workers", 4),
            )
        
        if "experiment" in data:
            exp_data = data["experiment"]
            config.experiment = ExperimentConfig(
                output_dir=exp_data.get("output_dir", "results"),
                save_intermediate=exp_data.get("save_intermediate", True),
                compare_modes=exp_data.get("compare_modes", ["vega", "vega-verified"]),
                metrics_file=exp_data.get("metrics_file", "metrics.json"),
            )
        
        return config


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    return Config.from_dict(data)


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
