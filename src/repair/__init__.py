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
from .neural_repair import (
    NeuralRepairModel,
    HybridRepairModel as NeuralHybridRepairModel,
    RepairCandidate as NeuralRepairCandidate,
    RepairContext as NeuralRepairContext,
    RepairPattern,
    create_neural_repair_model,
)
from .transformer_repair import (
    HybridTransformerRepairModel,
    TemplateRepairModel as TransformerTemplateModel,
    PatternRepairModel,
    CodeContext,
    RepairStrategy as TransformerRepairStrategy,
    create_transformer_repair_model,
)
from .integrated_cgnr import (
    IntegratedCGNREngine,
    CGNRResult,
    CGNRStatus,
    CGNRAttempt,
    create_cgnr_engine,
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
    # Neural Repair (from neural_repair.py)
    "NeuralRepairModel",
    "NeuralHybridRepairModel",
    "NeuralRepairCandidate",
    "NeuralRepairContext",
    "RepairPattern",
    "create_neural_repair_model",
    # Transformer Repair (from transformer_repair.py)
    "HybridTransformerRepairModel",
    "TransformerTemplateModel",
    "PatternRepairModel",
    "CodeContext",
    "TransformerRepairStrategy",
    "create_transformer_repair_model",
    # Integrated CGNR
    "IntegratedCGNREngine",
    "CGNRResult",
    "CGNRStatus",
    "CGNRAttempt",
    "create_cgnr_engine",
]
