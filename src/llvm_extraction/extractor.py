"""
LLVM Source Code Extractor for VEGA-Verified.

This module provides tools to download LLVM source code and extract
backend functions without requiring a full LLVM build.

Features:
- Fetch LLVM source from GitHub (shallow clone)
- Extract functions from specific backends
- Analyze module structure
- Build function database with metadata
"""

import os
import re
import subprocess
import shutil
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .parser import CppParser, ParsedFunction, FunctionSignature
from .database import (
    FunctionDatabase, FunctionRecord, ModuleRecord, BackendRecord, ModuleType
)


# LLVM Backend structure
BACKEND_MODULES = {
    "MCCodeEmitter": {
        "patterns": ["MCCodeEmitter.cpp", "MCCodeEmitter.h"],
        "key_functions": ["encodeInstruction", "getBinaryCodeForInstr", "getMachineOpValue"],
    },
    "ELFObjectWriter": {
        "patterns": ["ELFObjectWriter.cpp", "ELFObjectWriter.h", "ObjectWriter.cpp"],
        "key_functions": ["getRelocType", "needsRelocateWithSymbol", "getEFlags"],
    },
    "AsmPrinter": {
        "patterns": ["AsmPrinter.cpp", "AsmPrinter.h", "InstPrinter.cpp"],
        "key_functions": ["emitInstruction", "PrintInst", "printOperand"],
    },
    "ISelDAGToDAG": {
        "patterns": ["ISelDAGToDAG.cpp", "ISelDAGToDAG.h"],
        "key_functions": ["Select", "SelectAddr", "SelectInlineAsmMemoryOperand"],
    },
    "ISelLowering": {
        "patterns": ["ISelLowering.cpp", "ISelLowering.h"],
        "key_functions": ["LowerOperation", "LowerCall", "LowerReturn", "LowerFormalArguments"],
    },
    "RegisterInfo": {
        "patterns": ["RegisterInfo.cpp", "RegisterInfo.h"],
        "key_functions": ["getCalleeSavedRegs", "getReservedRegs", "eliminateFrameIndex"],
    },
    "InstrInfo": {
        "patterns": ["InstrInfo.cpp", "InstrInfo.h"],
        "key_functions": ["copyPhysReg", "storeRegToStackSlot", "loadRegFromStackSlot"],
    },
    "Subtarget": {
        "patterns": ["Subtarget.cpp", "Subtarget.h"],
        "key_functions": ["initializeSubtargetDependencies", "getCallLowering"],
    },
    "AsmParser": {
        "patterns": ["AsmParser.cpp", "AsmParser.h"],
        "key_functions": ["ParseInstruction", "ParseRegister", "ParseDirective"],
    },
    "Disassembler": {
        "patterns": ["Disassembler.cpp", "Disassembler.h"],
        "key_functions": ["getInstruction", "decodeInstruction"],
    },
}

# Known LLVM backends
KNOWN_BACKENDS = [
    "RISCV", "ARM", "AArch64", "X86", "Mips", "PowerPC", "AMDGPU",
    "Hexagon", "Lanai", "MSP430", "NVPTX", "Sparc", "SystemZ",
    "WebAssembly", "XCore", "AVR", "BPF", "LoongArch", "VE", "CSKY",
]


class LLVMSourceFetcher:
    """
    Fetches LLVM source code from GitHub.
    """
    
    GITHUB_URL = "https://github.com/llvm/llvm-project.git"
    
    def __init__(self, cache_dir: Optional[Path] = None, verbose: bool = True):
        """
        Initialize the fetcher.
        
        Args:
            cache_dir: Directory to store LLVM source
            verbose: Print progress messages
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "vega-verified" / "llvm"
        self.verbose = verbose
        self.llvm_root: Optional[Path] = None
    
    def fetch(
        self, 
        version: str = "18.1.0",
        shallow: bool = True,
        backends_only: bool = True
    ) -> Path:
        """
        Fetch LLVM source code.
        
        Args:
            version: LLVM version tag (e.g., "18.1.0")
            shallow: Use shallow clone (faster, less disk space)
            backends_only: Only fetch backend-related directories
            
        Returns:
            Path to LLVM source root
        """
        tag = f"llvmorg-{version}"
        dest = self.cache_dir / f"llvm-{version}"
        
        if dest.exists() and (dest / "llvm").exists():
            if self.verbose:
                print(f"Using cached LLVM source at {dest}")
            self.llvm_root = dest
            return dest
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"Fetching LLVM {version} from GitHub...")
        
        try:
            if shallow:
                # Shallow clone with sparse checkout for faster download
                cmd = [
                    "git", "clone",
                    "--depth", "1",
                    "--filter=blob:none",
                    "--sparse",
                    "--branch", tag,
                    self.GITHUB_URL,
                    str(dest)
                ]
                subprocess.run(cmd, check=True, capture_output=not self.verbose)
                
                # Set up sparse checkout
                sparse_paths = ["llvm/lib/Target", "llvm/include/llvm/MC"]
                os.chdir(dest)
                subprocess.run(["git", "sparse-checkout", "set"] + sparse_paths, 
                             check=True, capture_output=not self.verbose)
                
            else:
                # Full clone (slow, large)
                cmd = [
                    "git", "clone",
                    "--depth", "1",
                    "--branch", tag,
                    self.GITHUB_URL,
                    str(dest)
                ]
                subprocess.run(cmd, check=True, capture_output=not self.verbose)
            
            if self.verbose:
                print(f"LLVM source fetched to {dest}")
            
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"Git clone failed: {e}")
            # Try alternative: download archive
            return self._fetch_archive(version, dest)
        
        self.llvm_root = dest
        return dest
    
    def _fetch_archive(self, version: str, dest: Path) -> Path:
        """Fallback: download source archive."""
        import urllib.request
        import tarfile
        
        url = f"https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-{version}.tar.gz"
        archive_path = self.cache_dir / f"llvm-{version}.tar.gz"
        
        if self.verbose:
            print(f"Downloading archive from {url}...")
        
        urllib.request.urlretrieve(url, archive_path)
        
        if self.verbose:
            print(f"Extracting archive...")
        
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(self.cache_dir)
        
        # Rename extracted directory
        extracted = self.cache_dir / f"llvm-project-llvmorg-{version}"
        if extracted.exists():
            shutil.move(str(extracted), str(dest))
        
        # Clean up archive
        archive_path.unlink()
        
        self.llvm_root = dest
        return dest
    
    def get_backend_path(self, backend: str) -> Optional[Path]:
        """Get path to a specific backend."""
        if not self.llvm_root:
            return None
        
        target_path = self.llvm_root / "llvm" / "lib" / "Target" / backend
        if target_path.exists():
            return target_path
        
        return None
    
    def list_backends(self) -> List[str]:
        """List available backends."""
        if not self.llvm_root:
            return []
        
        target_dir = self.llvm_root / "llvm" / "lib" / "Target"
        if not target_dir.exists():
            return []
        
        backends = []
        for item in target_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                backends.append(item.name)
        
        return sorted(backends)


class CppFunctionExtractor:
    """
    Extracts C++ functions from source files.
    """
    
    def __init__(self, verbose: bool = False):
        self.parser = CppParser(verbose=verbose)
        self.verbose = verbose
    
    def extract_from_file(self, file_path: Path) -> List[ParsedFunction]:
        """Extract functions from a single file."""
        return self.parser.parse_file(file_path)
    
    def extract_from_directory(
        self, 
        dir_path: Path,
        patterns: Optional[List[str]] = None,
        recursive: bool = True
    ) -> Dict[str, List[ParsedFunction]]:
        """
        Extract functions from all C++ files in a directory.
        
        Returns:
            Dict mapping file paths to list of functions
        """
        results = {}
        
        if patterns:
            # Only match specific patterns
            for pattern in patterns:
                if recursive:
                    files = list(dir_path.rglob(f"*{pattern}*"))
                else:
                    files = list(dir_path.glob(f"*{pattern}*"))
                
                for file in files:
                    if file.suffix in ('.cpp', '.h', '.c'):
                        funcs = self.extract_from_file(file)
                        if funcs:
                            results[str(file)] = funcs
        else:
            # All C++ files
            glob_pattern = "**/*.cpp" if recursive else "*.cpp"
            for file in dir_path.glob(glob_pattern):
                funcs = self.extract_from_file(file)
                if funcs:
                    results[str(file)] = funcs
            
            # Also headers
            glob_pattern = "**/*.h" if recursive else "*.h"
            for file in dir_path.glob(glob_pattern):
                funcs = self.extract_from_file(file)
                if funcs:
                    results[str(file)] = funcs
        
        return results


class BackendAnalyzer:
    """
    Analyzes LLVM backend structure and identifies key functions.
    """
    
    def __init__(self, verbose: bool = False):
        self.extractor = CppFunctionExtractor(verbose=verbose)
        self.verbose = verbose
    
    def analyze_backend(
        self, 
        backend_path: Path,
        backend_name: str
    ) -> Tuple[Dict[str, List[ParsedFunction]], Dict[str, Any]]:
        """
        Analyze a backend directory.
        
        Returns:
            Tuple of (functions_by_module, metadata)
        """
        if self.verbose:
            print(f"Analyzing {backend_name} backend at {backend_path}")
        
        functions_by_module = {}
        metadata = {
            "backend": backend_name,
            "path": str(backend_path),
            "modules_found": [],
            "total_functions": 0,
            "files_processed": 0,
        }
        
        # Find MCTargetDesc subdirectory (contains key modules)
        mctargetdesc = backend_path / "MCTargetDesc"
        
        # Process each known module
        for module_name, module_info in BACKEND_MODULES.items():
            module_funcs = []
            
            # Search in main directory and MCTargetDesc
            search_dirs = [backend_path]
            if mctargetdesc.exists():
                search_dirs.append(mctargetdesc)
            
            for search_dir in search_dirs:
                for pattern in module_info["patterns"]:
                    # Find matching files
                    for file in search_dir.glob(f"*{pattern}*"):
                        if file.suffix in ('.cpp', '.h'):
                            funcs = self.extractor.extract_from_file(file)
                            
                            for func in funcs:
                                # Mark interface functions
                                func.is_interface = func.signature.name in module_info["key_functions"]
                                module_funcs.append(func)
                            
                            metadata["files_processed"] += 1
            
            if module_funcs:
                functions_by_module[module_name] = module_funcs
                metadata["modules_found"].append(module_name)
                metadata["total_functions"] += len(module_funcs)
        
        # Also extract from any remaining files
        other_funcs = []
        all_processed_files = set()
        
        for module_funcs in functions_by_module.values():
            for func in module_funcs:
                all_processed_files.add(func.file_path)
        
        for file in backend_path.rglob("*.cpp"):
            if str(file) not in all_processed_files:
                funcs = self.extractor.extract_from_file(file)
                other_funcs.extend(funcs)
                metadata["files_processed"] += 1
        
        if other_funcs:
            functions_by_module["Other"] = other_funcs
            metadata["total_functions"] += len(other_funcs)
        
        return functions_by_module, metadata


class LLVMExtractor:
    """
    Main class for extracting LLVM backend code.
    
    Combines fetching, parsing, and database storage.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
        verbose: bool = True
    ):
        """
        Initialize the extractor.
        
        Args:
            cache_dir: Directory to cache LLVM source
            db_path: Path to function database
            verbose: Print progress
        """
        self.fetcher = LLVMSourceFetcher(cache_dir, verbose)
        self.analyzer = BackendAnalyzer(verbose)
        self.verbose = verbose
        
        self.db_path = db_path or Path("data/function_database.json")
        self.db = FunctionDatabase(self.db_path)
        
        self.llvm_version = ""
    
    def fetch_llvm_source(self, version: str = "18.1.0") -> Path:
        """Fetch LLVM source code."""
        self.llvm_version = version
        return self.fetcher.fetch(version)
    
    def extract_backend(
        self,
        backend: str,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Extract all functions from a backend.
        
        Args:
            backend: Backend name (e.g., "RISCV")
            save: Save to database
            
        Returns:
            Extraction results with statistics
        """
        start_time = time.time()
        
        backend_path = self.fetcher.get_backend_path(backend)
        if not backend_path:
            raise ValueError(f"Backend {backend} not found. Did you fetch LLVM source?")
        
        if self.verbose:
            print(f"\nExtracting {backend} backend...")
        
        # Analyze backend
        functions_by_module, metadata = self.analyzer.analyze_backend(backend_path, backend)
        
        # Convert to database records
        all_function_ids = []
        module_records = []
        
        for module_name, parsed_funcs in functions_by_module.items():
            module_func_ids = []
            
            for parsed in parsed_funcs:
                # Create function record
                func_record = self._create_function_record(
                    parsed, backend, module_name
                )
                
                # Add to database
                func_id = self.db.add_function(func_record)
                module_func_ids.append(func_id)
                all_function_ids.append(func_id)
            
            # Create module record
            module_record = ModuleRecord(
                name=module_name,
                module_type=ModuleType.from_filename(module_name).value,
                backend=backend,
                files=list(set(f.file_path for f in parsed_funcs)),
                function_count=len(parsed_funcs),
                function_ids=module_func_ids,
                total_lines=sum(f.line_count for f in parsed_funcs),
                avg_function_lines=sum(f.line_count for f in parsed_funcs) / len(parsed_funcs) if parsed_funcs else 0,
                switch_function_count=sum(1 for f in parsed_funcs if f.switch_patterns),
            )
            self.db.add_module(module_record)
            module_records.append(module_record)
        
        # Create backend record
        extraction_time = time.time() - start_time
        
        backend_record = BackendRecord(
            name=backend,
            target_triple=self._get_target_triple(backend),
            llvm_version=self.llvm_version,
            modules=[m.name for m in module_records],
            total_functions=len(all_function_ids),
            total_files=metadata["files_processed"],
            total_lines=sum(m.total_lines for m in module_records),
            interface_functions=sum(1 for fid in all_function_ids 
                                   if self.db.get_function(fid).is_interface),
            switch_functions=sum(m.switch_function_count for m in module_records),
            extracted_at=datetime.now().isoformat(),
            extraction_time_sec=extraction_time,
        )
        self.db.add_backend(backend_record)
        
        # Save if requested
        if save:
            self.db.save()
        
        # Print summary
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Extraction complete for {backend}")
            print(f"{'='*60}")
            print(f"  Total functions: {len(all_function_ids)}")
            print(f"  Total files: {metadata['files_processed']}")
            print(f"  Modules found: {len(module_records)}")
            print(f"  Time: {extraction_time:.2f}s")
            print(f"\n  By module:")
            for m in module_records:
                print(f"    {m.name}: {m.function_count} functions ({m.switch_function_count} with switch)")
        
        return {
            "backend": backend,
            "total_functions": len(all_function_ids),
            "modules": {m.name: m.function_count for m in module_records},
            "extraction_time_sec": extraction_time,
            "database_path": str(self.db_path),
        }
    
    def extract_all_backends(
        self,
        backends: Optional[List[str]] = None,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Extract multiple backends.
        
        Args:
            backends: List of backends to extract (None = all)
            parallel: Use parallel extraction
            
        Returns:
            Summary of all extractions
        """
        if backends is None:
            backends = self.fetcher.list_backends()
        
        if self.verbose:
            print(f"Extracting {len(backends)} backends: {backends}")
        
        results = {}
        
        for backend in backends:
            try:
                result = self.extract_backend(backend, save=False)
                results[backend] = result
            except Exception as e:
                if self.verbose:
                    print(f"Error extracting {backend}: {e}")
                results[backend] = {"error": str(e)}
        
        # Save once at the end
        self.db.save()
        
        return results
    
    def _create_function_record(
        self,
        parsed: ParsedFunction,
        backend: str,
        module: str
    ) -> FunctionRecord:
        """Create a FunctionRecord from ParsedFunction."""
        
        # Determine if interface function
        module_info = BACKEND_MODULES.get(module, {})
        key_funcs = module_info.get("key_functions", [])
        is_interface = parsed.signature.name in key_funcs
        
        # Calculate complexity score
        complexity = self._calculate_complexity(parsed)
        
        return FunctionRecord(
            id=FunctionRecord.compute_id(parsed.raw_code),
            name=parsed.signature.name,
            full_name=parsed.signature.full_name,
            backend=backend,
            module=module,
            file_path=parsed.file_path,
            start_line=parsed.start_line,
            end_line=parsed.end_line,
            return_type=parsed.signature.return_type,
            parameters=[{"type": p[0], "name": p[1]} for p in parsed.signature.parameters],
            qualifiers=parsed.signature.qualifiers,
            body=parsed.body,
            raw_code=parsed.raw_code,
            line_count=parsed.line_count,
            has_switch=len(parsed.switch_patterns) > 0,
            switch_patterns=[sp.to_dict() for sp in parsed.switch_patterns],
            called_functions=list(parsed.called_functions),
            llvm_version=self.llvm_version,
            extracted_at=datetime.now().isoformat(),
            is_interface=is_interface,
            is_target_specific=True,
            complexity_score=complexity,
        )
    
    def _calculate_complexity(self, parsed: ParsedFunction) -> float:
        """Calculate a complexity score for a function."""
        score = 0.0
        
        # Base on line count
        score += min(parsed.line_count / 10, 5.0)
        
        # Switch statements
        score += len(parsed.switch_patterns) * 2
        
        # Total switch cases
        for sp in parsed.switch_patterns:
            score += len(sp.cases) * 0.1
        
        # Called functions
        score += len(parsed.called_functions) * 0.2
        
        # IsPCRel patterns
        if any(sp.has_ternary_ispcrel for sp in parsed.switch_patterns):
            score += 1.0
        
        return round(score, 2)
    
    def _get_target_triple(self, backend: str) -> str:
        """Get target triple for a backend."""
        triples = {
            "RISCV": "riscv64-unknown-linux-gnu",
            "ARM": "arm-none-eabi",
            "AArch64": "aarch64-unknown-linux-gnu",
            "X86": "x86_64-unknown-linux-gnu",
            "Mips": "mips64-unknown-linux-gnu",
            "PowerPC": "powerpc64-unknown-linux-gnu",
            "AMDGPU": "amdgcn-amd-amdhsa",
            "NVPTX": "nvptx64-nvidia-cuda",
            "WebAssembly": "wasm32-unknown-unknown",
            "XCore": "xcore-unknown-unknown",
            "Hexagon": "hexagon-unknown-linux-musl",
        }
        return triples.get(backend, f"{backend.lower()}-unknown-unknown")
    
    def get_database(self) -> FunctionDatabase:
        """Get the function database."""
        return self.db


# Quick test / CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract LLVM backend functions")
    parser.add_argument("--version", default="18.1.0", help="LLVM version")
    parser.add_argument("--backend", default="RISCV", help="Backend to extract")
    parser.add_argument("--output", default="data/function_database.json", help="Output database path")
    parser.add_argument("--all", action="store_true", help="Extract all backends")
    
    args = parser.parse_args()
    
    extractor = LLVMExtractor(
        db_path=Path(args.output),
        verbose=True
    )
    
    # Fetch LLVM source
    extractor.fetch_llvm_source(args.version)
    
    # Extract
    if args.all:
        results = extractor.extract_all_backends()
    else:
        results = extractor.extract_backend(args.backend)
    
    print(f"\nResults: {results}")
