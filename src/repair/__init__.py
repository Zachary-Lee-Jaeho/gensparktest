"""
CGNR (Counterexample-Guided Neural Repair) module for VEGA-Verified.

Phase 3 components:
- CGNR main loop
- Fault localization
- Neural repair models (template, LLM, hybrid)
"""

from .cgnr import CGNREngine, RepairContext, RepairResult, RepairStatus, RepairAttempt
from .fault_loc import FaultLocalizer, FaultLocation
from .repair_model import (
    RepairModelBase,
    TemplateRepairModel,
    LLMRepairModel,
    HybridRepairModel,
    RepairCandidate,
    RepairStrategy,
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
    # Repair models
    "RepairModelBase",
    "TemplateRepairModel",
    "LLMRepairModel",
    "HybridRepairModel",
    "RepairCandidate",
    "RepairStrategy",
    "create_repair_model",
]
