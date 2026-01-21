"""
VEGA-Verified: Semantically Verified Neural Compiler Backend Generation

This package provides formal verification capabilities on top of VEGA's
neural code generation for compiler backends.

Modes:
    - vega: Original VEGA neural generation only
    - vega-verified: VEGA + formal verification + CGNR repair
    - verify-only: Verification without neural generation
"""

__version__ = "0.1.0"
__author__ = "VEGA-Verified Team"

from enum import Enum

class ExecutionMode(Enum):
    """Execution modes for comparison experiments."""
    VEGA = "vega"                    # Original VEGA only
    VEGA_VERIFIED = "vega-verified"  # VEGA + Verification + Repair
    VERIFY_ONLY = "verify-only"      # Verification only (no generation)
    
    @classmethod
    def from_string(cls, mode: str) -> "ExecutionMode":
        """Parse execution mode from string."""
        mode_map = {
            "vega": cls.VEGA,
            "vega-verified": cls.VEGA_VERIFIED,
            "verify-only": cls.VERIFY_ONLY,
        }
        if mode.lower() not in mode_map:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {list(mode_map.keys())}")
        return mode_map[mode.lower()]
