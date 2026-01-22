"""
Function Database for VEGA-Verified.

This module provides persistent storage for extracted LLVM functions
with full metadata, enabling queries and analysis.

Storage Format:
- JSON for portability and inspection
- SQLite for complex queries (optional)
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Iterator
from pathlib import Path
from datetime import datetime
from enum import Enum
import hashlib


class ModuleType(Enum):
    """LLVM Backend Module Types."""
    MC_CODE_EMITTER = "MCCodeEmitter"
    ASM_PRINTER = "AsmPrinter"
    ELF_OBJECT_WRITER = "ELFObjectWriter"
    ISEL_DAG_TO_DAG = "ISelDAGToDAG"
    REGISTER_INFO = "RegisterInfo"
    INSTR_INFO = "InstrInfo"
    TARGET_LOWERING = "TargetLowering"
    ASM_PARSER = "AsmParser"
    DISASSEMBLER = "Disassembler"
    SUBTARGET = "Subtarget"
    OTHER = "Other"
    
    @classmethod
    def from_filename(cls, filename: str) -> "ModuleType":
        """Infer module type from filename."""
        filename = filename.lower()
        mappings = {
            "mccodeemitter": cls.MC_CODE_EMITTER,
            "asmprinter": cls.ASM_PRINTER,
            "elfobjectwriter": cls.ELF_OBJECT_WRITER,
            "iseldagtodag": cls.ISEL_DAG_TO_DAG,
            "registerinfo": cls.REGISTER_INFO,
            "instrinfo": cls.INSTR_INFO,
            "isellowering": cls.TARGET_LOWERING,
            "targetlowering": cls.TARGET_LOWERING,
            "asmparser": cls.ASM_PARSER,
            "disassembler": cls.DISASSEMBLER,
            "subtarget": cls.SUBTARGET,
        }
        for key, value in mappings.items():
            if key in filename:
                return value
        return cls.OTHER


@dataclass
class FunctionRecord:
    """
    Complete record of an extracted function.
    """
    # Identity
    id: str                         # Unique ID (hash of content)
    name: str                       # Function name
    full_name: str                  # Class::Function
    
    # Location
    backend: str                    # e.g., "RISCV", "ARM"
    module: str                     # e.g., "MCCodeEmitter"
    file_path: str                  # Relative path in LLVM
    start_line: int
    end_line: int
    
    # Signature
    return_type: str
    parameters: List[Dict[str, str]]  # [{"type": ..., "name": ...}, ...]
    qualifiers: List[str]
    
    # Content
    body: str                       # Function body only
    raw_code: str                   # Complete function code
    
    # Analysis
    line_count: int
    has_switch: bool
    switch_patterns: List[Dict]     # Extracted switch patterns
    called_functions: List[str]
    
    # Metadata
    llvm_version: str
    extracted_at: str
    
    # Classification
    is_interface: bool = False      # Key interface function
    is_target_specific: bool = True # Contains target-specific code
    complexity_score: float = 0.0   # Estimated complexity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionRecord":
        """Create from dictionary."""
        return cls(**data)
    
    @staticmethod
    def compute_id(raw_code: str) -> str:
        """Compute unique ID from code content."""
        return hashlib.sha256(raw_code.encode()).hexdigest()[:16]


@dataclass
class ModuleRecord:
    """
    Record of an LLVM backend module.
    """
    name: str
    module_type: str
    backend: str
    files: List[str]
    function_count: int
    function_ids: List[str]
    
    # Statistics
    total_lines: int = 0
    avg_function_lines: float = 0.0
    switch_function_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleRecord":
        return cls(**data)


@dataclass
class BackendRecord:
    """
    Record of a complete LLVM backend.
    """
    name: str                       # e.g., "RISCV"
    target_triple: str              # e.g., "riscv64-unknown-linux-gnu"
    llvm_version: str
    
    # Structure
    modules: List[str]              # Module names
    total_functions: int
    total_files: int
    
    # Statistics
    total_lines: int = 0
    interface_functions: int = 0
    switch_functions: int = 0
    
    # Extraction metadata
    extracted_at: str = ""
    extraction_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackendRecord":
        return cls(**data)


class FunctionDatabase:
    """
    Database for storing and querying extracted LLVM functions.
    
    Supports:
    - JSON-based storage (portable, human-readable)
    - SQLite backend (for complex queries)
    - In-memory caching
    """
    
    def __init__(self, db_path: Optional[Path] = None, use_sqlite: bool = False):
        """
        Initialize the database.
        
        Args:
            db_path: Path to database file (JSON or SQLite)
            use_sqlite: Use SQLite instead of JSON
        """
        self.db_path = db_path
        self.use_sqlite = use_sqlite
        
        # In-memory storage
        self.functions: Dict[str, FunctionRecord] = {}
        self.modules: Dict[str, ModuleRecord] = {}
        self.backends: Dict[str, BackendRecord] = {}
        
        # Indices for fast lookup
        self._by_name: Dict[str, List[str]] = {}
        self._by_backend: Dict[str, List[str]] = {}
        self._by_module: Dict[str, List[str]] = {}
        
        # Statistics
        self.stats = {
            "total_functions": 0,
            "total_modules": 0,
            "total_backends": 0,
            "last_updated": None,
        }
        
        # Load existing data
        if db_path and db_path.exists():
            self.load()
    
    def add_function(self, func: FunctionRecord) -> str:
        """
        Add a function to the database.
        
        Returns:
            Function ID
        """
        # Store function
        self.functions[func.id] = func
        
        # Update indices
        if func.name not in self._by_name:
            self._by_name[func.name] = []
        self._by_name[func.name].append(func.id)
        
        if func.backend not in self._by_backend:
            self._by_backend[func.backend] = []
        self._by_backend[func.backend].append(func.id)
        
        module_key = f"{func.backend}::{func.module}"
        if module_key not in self._by_module:
            self._by_module[module_key] = []
        self._by_module[module_key].append(func.id)
        
        self.stats["total_functions"] = len(self.functions)
        self.stats["last_updated"] = datetime.now().isoformat()
        
        return func.id
    
    def add_module(self, module: ModuleRecord) -> None:
        """Add a module record."""
        key = f"{module.backend}::{module.name}"
        self.modules[key] = module
        self.stats["total_modules"] = len(self.modules)
    
    def add_backend(self, backend: BackendRecord) -> None:
        """Add a backend record."""
        self.backends[backend.name] = backend
        self.stats["total_backends"] = len(self.backends)
    
    def get_function(self, func_id: str) -> Optional[FunctionRecord]:
        """Get function by ID."""
        return self.functions.get(func_id)
    
    def get_functions_by_name(self, name: str) -> List[FunctionRecord]:
        """Get all functions with a given name."""
        ids = self._by_name.get(name, [])
        return [self.functions[id] for id in ids if id in self.functions]
    
    def get_functions_by_backend(self, backend: str) -> List[FunctionRecord]:
        """Get all functions for a backend."""
        ids = self._by_backend.get(backend, [])
        return [self.functions[id] for id in ids if id in self.functions]
    
    def get_functions_by_module(self, backend: str, module: str) -> List[FunctionRecord]:
        """Get all functions for a specific module."""
        key = f"{backend}::{module}"
        ids = self._by_module.get(key, [])
        return [self.functions[id] for id in ids if id in self.functions]
    
    def get_interface_functions(self, backend: str) -> List[FunctionRecord]:
        """Get key interface functions for a backend."""
        funcs = self.get_functions_by_backend(backend)
        return [f for f in funcs if f.is_interface]
    
    def get_switch_functions(self, backend: str) -> List[FunctionRecord]:
        """Get functions with switch statements."""
        funcs = self.get_functions_by_backend(backend)
        return [f for f in funcs if f.has_switch]
    
    def search(
        self,
        name_pattern: Optional[str] = None,
        backend: Optional[str] = None,
        module: Optional[str] = None,
        has_switch: Optional[bool] = None,
        min_lines: Optional[int] = None,
        max_lines: Optional[int] = None,
    ) -> List[FunctionRecord]:
        """
        Search functions with filters.
        """
        results = list(self.functions.values())
        
        if name_pattern:
            import re
            pattern = re.compile(name_pattern, re.IGNORECASE)
            results = [f for f in results if pattern.search(f.name)]
        
        if backend:
            results = [f for f in results if f.backend == backend]
        
        if module:
            results = [f for f in results if f.module == module]
        
        if has_switch is not None:
            results = [f for f in results if f.has_switch == has_switch]
        
        if min_lines is not None:
            results = [f for f in results if f.line_count >= min_lines]
        
        if max_lines is not None:
            results = [f for f in results if f.line_count <= max_lines]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = dict(self.stats)
        
        # Per-backend stats
        backend_stats = {}
        for backend in self.backends:
            funcs = self.get_functions_by_backend(backend)
            backend_stats[backend] = {
                "total_functions": len(funcs),
                "switch_functions": sum(1 for f in funcs if f.has_switch),
                "interface_functions": sum(1 for f in funcs if f.is_interface),
                "total_lines": sum(f.line_count for f in funcs),
                "avg_lines": sum(f.line_count for f in funcs) / len(funcs) if funcs else 0,
            }
        stats["backends"] = backend_stats
        
        # Module type distribution
        module_dist = {}
        for func in self.functions.values():
            if func.module not in module_dist:
                module_dist[func.module] = 0
            module_dist[func.module] += 1
        stats["module_distribution"] = module_dist
        
        return stats
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save database to file."""
        path = path or self.db_path
        if not path:
            raise ValueError("No database path specified")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "stats": self.stats,
            "functions": {id: f.to_dict() for id, f in self.functions.items()},
            "modules": {k: m.to_dict() for k, m in self.modules.items()},
            "backends": {k: b.to_dict() for k, b in self.backends.items()},
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved database to {path} ({len(self.functions)} functions)")
    
    def load(self, path: Optional[Path] = None) -> None:
        """Load database from file."""
        path = path or self.db_path
        if not path or not Path(path).exists():
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Load functions
        for id, func_data in data.get("functions", {}).items():
            func = FunctionRecord.from_dict(func_data)
            self.functions[id] = func
            
            # Rebuild indices
            if func.name not in self._by_name:
                self._by_name[func.name] = []
            self._by_name[func.name].append(id)
            
            if func.backend not in self._by_backend:
                self._by_backend[func.backend] = []
            self._by_backend[func.backend].append(id)
            
            module_key = f"{func.backend}::{func.module}"
            if module_key not in self._by_module:
                self._by_module[module_key] = []
            self._by_module[module_key].append(id)
        
        # Load modules
        for k, m_data in data.get("modules", {}).items():
            self.modules[k] = ModuleRecord.from_dict(m_data)
        
        # Load backends
        for k, b_data in data.get("backends", {}).items():
            self.backends[k] = BackendRecord.from_dict(b_data)
        
        self.stats = data.get("stats", self.stats)
        self.stats["total_functions"] = len(self.functions)
        self.stats["total_modules"] = len(self.modules)
        self.stats["total_backends"] = len(self.backends)
        
        print(f"Loaded database from {path} ({len(self.functions)} functions)")
    
    def export_for_vega_verified(self, backend: str) -> Dict[str, Any]:
        """
        Export functions in format suitable for VEGA-Verified evaluation.
        
        Returns dictionary with:
        - functions: List of function data
        - modules: Module structure
        - metadata: Backend information
        """
        funcs = self.get_functions_by_backend(backend)
        
        # Group by module
        by_module = {}
        for func in funcs:
            if func.module not in by_module:
                by_module[func.module] = []
            by_module[func.module].append({
                "name": func.name,
                "full_name": func.full_name,
                "code": func.raw_code,
                "has_switch": func.has_switch,
                "switch_patterns": func.switch_patterns,
                "is_interface": func.is_interface,
                "line_count": func.line_count,
            })
        
        return {
            "backend": backend,
            "llvm_version": self.backends.get(backend, BackendRecord(
                name=backend, target_triple="", llvm_version="unknown",
                modules=[], total_functions=0, total_files=0
            )).llvm_version,
            "total_functions": len(funcs),
            "modules": by_module,
            "statistics": {
                "total_lines": sum(f.line_count for f in funcs),
                "switch_functions": sum(1 for f in funcs if f.has_switch),
                "interface_functions": sum(1 for f in funcs if f.is_interface),
            }
        }
    
    def __len__(self) -> int:
        return len(self.functions)
    
    def __iter__(self) -> Iterator[FunctionRecord]:
        return iter(self.functions.values())


# Quick test
if __name__ == "__main__":
    from datetime import datetime
    
    db = FunctionDatabase()
    
    # Add test function
    func = FunctionRecord(
        id="test123",
        name="getRelocType",
        full_name="RISCVELFObjectWriter::getRelocType",
        backend="RISCV",
        module="ELFObjectWriter",
        file_path="lib/Target/RISCV/MCTargetDesc/RISCVELFObjectWriter.cpp",
        start_line=50,
        end_line=80,
        return_type="unsigned",
        parameters=[
            {"type": "MCContext &", "name": "Ctx"},
            {"type": "const MCValue &", "name": "Target"},
        ],
        qualifiers=["const"],
        body="{ switch(Kind) { ... } }",
        raw_code="unsigned getRelocType(...) { ... }",
        line_count=30,
        has_switch=True,
        switch_patterns=[{"switch_variable": "Kind", "cases": []}],
        called_functions=["Fixup.getTargetKind"],
        llvm_version="18.1.0",
        extracted_at=datetime.now().isoformat(),
        is_interface=True,
    )
    
    db.add_function(func)
    
    print(f"Database has {len(db)} functions")
    print(f"RISCV functions: {len(db.get_functions_by_backend('RISCV'))}")
    print(f"Switch functions: {len(db.search(has_switch=True))}")
    
    stats = db.get_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2)}")
