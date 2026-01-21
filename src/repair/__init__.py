"""
CGNR (Counterexample-Guided Neural Repair) module for VEGA-Verified.

Phase 3 components:
- CGNR main loop
- Fault localization
- Neural repair models (template, LLM, transformer, hybrid)
"""

from .cgnr import CGNREngine, RepairContext, RepairResult, RepairStatus, RepairAttempt
from .fault_loc import FaultLocalizer, FaultLocation
from .repair_model import (
    RepairModelBase,
    TemplateRepairModel,
    LLMRepairModel,
    RuleBasedRepairModel,
    HybridRepairModel as LegacyHybridRepairModel,
    RepairCandidate as LegacyRepairCandidate,
    RepairStrategy,
    create_repair_model as create_legacy_repair_model,
)
from .neural_model import (
    BaseRepairModel,
    TransformerRepairModel,
    APIRepairModel,
    HybridRepairModel,
    RepairCandidate,
    ModelConfig,
    ModelBackend,
    create_repair_model,
)

__all__ = [
    # CGNR Engine
    "CGNREngine",
    "RepairContext",
    "RepairResult",
    "RepairStatus",
    "RepairAttempt",
    # Fault localization
    "FaultLocalizer",
    "FaultLocation",
    # Legacy repair models
    "RepairModelBase",
    "TemplateRepairModel",
    "LLMRepairModel",
    "RuleBasedRepairModel",
    "RepairStrategy",
    # New neural repair models
    "BaseRepairModel",
    "TransformerRepairModel",
    "APIRepairModel",
    "HybridRepairModel",
    "RepairCandidate",
    "ModelConfig",
    "ModelBackend",
    "create_repair_model",
]
