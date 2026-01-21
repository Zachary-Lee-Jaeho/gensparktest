"""
LLVM Adapter for VEGA-Verified.

Provides integration with LLVM backend infrastructure,
including TableGen parsing and backend information extraction.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import json
import re


@dataclass
class RegisterInfo:
    """Information about a register."""
    name: str
    encoding: int
    aliases: List[str] = field(default_factory=list)
    subregisters: List[str] = field(default_factory=list)
    superregisters: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "encoding": self.encoding,
            "aliases": self.aliases,
            "subregisters": self.subregisters,
            "superregisters": self.superregisters,
        }


@dataclass
class InstructionInfo:
    """Information about an instruction."""
    name: str
    opcode: int
    mnemonic: str
    operands: List[Dict[str, str]] = field(default_factory=list)
    encoding: str = ""
    is_branch: bool = False
    is_call: bool = False
    is_return: bool = False
    is_load: bool = False
    is_store: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "opcode": self.opcode,
            "mnemonic": self.mnemonic,
            "operands": self.operands,
            "encoding": self.encoding,
            "is_branch": self.is_branch,
            "is_call": self.is_call,
            "is_return": self.is_return,
            "is_load": self.is_load,
            "is_store": self.is_store,
        }


@dataclass
class ModuleInfo:
    """Information about a backend module."""
    name: str
    description: str
    functions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "functions": self.functions,
            "dependencies": self.dependencies,
            "source_files": self.source_files,
        }


@dataclass
class BackendInfo:
    """Complete information about an LLVM backend."""
    name: str
    target_triple: str
    description: str = ""
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)
    registers: Dict[str, RegisterInfo] = field(default_factory=dict)
    instructions: Dict[str, InstructionInfo] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_functions(self) -> int:
        return sum(len(m.functions) for m in self.modules.values())
    
    @property
    def total_instructions(self) -> int:
        return len(self.instructions)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "target_triple": self.target_triple,
            "description": self.description,
            "total_functions": self.total_functions,
            "total_instructions": self.total_instructions,
            "modules": {k: v.to_dict() for k, v in self.modules.items()},
            "registers": {k: v.to_dict() for k, v in self.registers.items()},
            "metadata": self.metadata,
        }
    
    def save(self, path: str) -> None:
        """Save backend info to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BackendInfo':
        """Load backend info from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        info = cls(
            name=data["name"],
            target_triple=data["target_triple"],
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )
        
        for name, mod_data in data.get("modules", {}).items():
            info.modules[name] = ModuleInfo(
                name=mod_data["name"],
                description=mod_data["description"],
                functions=mod_data.get("functions", []),
                dependencies=mod_data.get("dependencies", []),
                source_files=mod_data.get("source_files", []),
            )
        
        for name, reg_data in data.get("registers", {}).items():
            info.registers[name] = RegisterInfo(
                name=reg_data["name"],
                encoding=reg_data["encoding"],
                aliases=reg_data.get("aliases", []),
            )
        
        return info


class LLVMAdapter:
    """
    Adapter for integrating with LLVM backend infrastructure.
    
    Provides:
    1. Backend information extraction
    2. TableGen parsing
    3. Reference code loading
    """
    
    # Standard LLVM backend modules
    STANDARD_MODULES = {
        "MCCodeEmitter": ModuleInfo(
            name="MCCodeEmitter",
            description="Machine code emission",
            functions=[
                "encodeInstruction",
                "getMachineOpValue",
                "getBinaryCodeForInstr",
                "getInstSizeInBytes",
            ],
            dependencies=[],
        ),
        "AsmPrinter": ModuleInfo(
            name="AsmPrinter",
            description="Assembly printing",
            functions=[
                "printOperand",
                "printInstruction",
                "printRegName",
                "printMemOperand",
            ],
            dependencies=["MCCodeEmitter"],
        ),
        "InstPrinter": ModuleInfo(
            name="InstPrinter",
            description="Instruction printing",
            functions=[
                "printInst",
                "printOperand",
                "getRegisterName",
                "getMnemonic",
            ],
            dependencies=[],
        ),
        "ELFObjectWriter": ModuleInfo(
            name="ELFObjectWriter",
            description="ELF object file writing",
            functions=[
                "getRelocType",
                "writeObject",
                "recordRelocation",
                "fixupNeedsRelaxation",
            ],
            dependencies=["MCCodeEmitter", "AsmPrinter"],
        ),
        "AsmParser": ModuleInfo(
            name="AsmParser",
            description="Assembly parsing",
            functions=[
                "parseInstruction",
                "parseOperand",
                "parseRegister",
                "parseImmediate",
            ],
            dependencies=["InstPrinter"],
        ),
        "ISelDAGToDAG": ModuleInfo(
            name="ISelDAGToDAG",
            description="DAG instruction selection",
            functions=[
                "Select",
                "SelectAddr",
                "SelectAddrFI",
            ],
            dependencies=[],
        ),
        "RegisterInfo": ModuleInfo(
            name="RegisterInfo",
            description="Register information",
            functions=[
                "getCalleeSavedRegs",
                "getReservedRegs",
                "eliminateFrameIndex",
            ],
            dependencies=[],
        ),
    }
    
    # Known targets and their info
    KNOWN_TARGETS = {
        "RISCV": {
            "triple": "riscv64-unknown-linux-gnu",
            "description": "RISC-V 64-bit backend",
        },
        "ARM": {
            "triple": "arm-unknown-linux-gnueabi",
            "description": "ARM 32-bit backend",
        },
        "AArch64": {
            "triple": "aarch64-unknown-linux-gnu",
            "description": "ARM 64-bit backend",
        },
        "X86": {
            "triple": "x86_64-unknown-linux-gnu",
            "description": "x86-64 backend",
        },
        "MIPS": {
            "triple": "mips64-unknown-linux-gnu",
            "description": "MIPS 64-bit backend",
        },
    }
    
    def __init__(
        self,
        llvm_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize LLVM adapter.
        
        Args:
            llvm_path: Path to LLVM source/build directory
            cache_dir: Directory for caching parsed information
            verbose: Enable verbose output
        """
        self.llvm_path = Path(llvm_path) if llvm_path else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.verbose = verbose
        
        # Cache for loaded backends
        self._backend_cache: Dict[str, BackendInfo] = {}
    
    def get_backend_info(self, target: str) -> BackendInfo:
        """
        Get information about a backend.
        
        Args:
            target: Target name (e.g., "RISCV", "ARM")
        
        Returns:
            BackendInfo for the target
        """
        if target in self._backend_cache:
            return self._backend_cache[target]
        
        # Check cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{target.lower()}_backend.json"
            if cache_file.exists():
                info = BackendInfo.load(str(cache_file))
                self._backend_cache[target] = info
                return info
        
        # Build backend info
        target_info = self.KNOWN_TARGETS.get(target, {
            "triple": f"{target.lower()}-unknown-unknown",
            "description": f"{target} backend",
        })
        
        info = BackendInfo(
            name=target,
            target_triple=target_info["triple"],
            description=target_info["description"],
        )
        
        # Add standard modules
        for mod_name, mod_info in self.STANDARD_MODULES.items():
            info.modules[mod_name] = ModuleInfo(
                name=mod_info.name,
                description=mod_info.description,
                functions=list(mod_info.functions),
                dependencies=list(mod_info.dependencies),
                source_files=[f"{target}{mod_name}.cpp"],
            )
        
        # If we have LLVM path, try to extract more info
        if self.llvm_path:
            self._extract_llvm_info(info)
        
        # Cache the result
        self._backend_cache[target] = info
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            info.save(str(self.cache_dir / f"{target.lower()}_backend.json"))
        
        return info
    
    def _extract_llvm_info(self, info: BackendInfo) -> None:
        """Extract additional info from LLVM sources."""
        target = info.name
        target_dir = self.llvm_path / "llvm" / "lib" / "Target" / target
        
        if not target_dir.exists():
            if self.verbose:
                print(f"LLVM target directory not found: {target_dir}")
            return
        
        # Find source files
        cpp_files = list(target_dir.glob("*.cpp"))
        
        for cpp_file in cpp_files:
            # Map file to module
            for mod_name in info.modules:
                if mod_name in cpp_file.name:
                    info.modules[mod_name].source_files.append(str(cpp_file))
                    
                    # Extract function names from file
                    self._extract_functions_from_file(
                        cpp_file, info.modules[mod_name]
                    )
        
        # Parse TableGen files if available
        td_files = list(target_dir.glob("*.td"))
        for td_file in td_files:
            self._parse_tablegen(td_file, info)
    
    def _extract_functions_from_file(
        self,
        cpp_file: Path,
        module_info: ModuleInfo
    ) -> None:
        """Extract function definitions from C++ file."""
        try:
            content = cpp_file.read_text()
            
            # Simple regex to find function definitions
            func_pattern = r'(?:void|bool|unsigned|int|[A-Z]\w+)\s+(?:\w+::)?(\w+)\s*\([^)]*\)'
            
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                if func_name not in module_info.functions:
                    module_info.functions.append(func_name)
        
        except Exception as e:
            if self.verbose:
                print(f"Error parsing {cpp_file}: {e}")
    
    def _parse_tablegen(self, td_file: Path, info: BackendInfo) -> None:
        """Parse TableGen file for instruction/register info."""
        try:
            content = td_file.read_text()
            
            # Extract register definitions
            reg_pattern = r'def\s+(\w+)\s*:\s*(?:R\w+Register|GPRRegister).*?'
            for match in re.finditer(reg_pattern, content):
                reg_name = match.group(1)
                if reg_name not in info.registers:
                    info.registers[reg_name] = RegisterInfo(
                        name=reg_name,
                        encoding=len(info.registers),
                    )
            
            # Extract instruction definitions
            inst_pattern = r'def\s+(\w+)\s*:\s*\w*Inst.*?'
            for match in re.finditer(inst_pattern, content):
                inst_name = match.group(1)
                if inst_name not in info.instructions:
                    info.instructions[inst_name] = InstructionInfo(
                        name=inst_name,
                        opcode=len(info.instructions),
                        mnemonic=inst_name.lower(),
                    )
        
        except Exception as e:
            if self.verbose:
                print(f"Error parsing {td_file}: {e}")
    
    def load_reference_code(
        self,
        function_name: str,
        target: str,
        module_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Load reference code for a function from LLVM sources.
        
        Args:
            function_name: Name of function
            target: Target backend
            module_name: Optional module name to narrow search
        
        Returns:
            Function source code if found, None otherwise
        """
        if not self.llvm_path:
            return None
        
        target_dir = self.llvm_path / "llvm" / "lib" / "Target" / target
        
        if not target_dir.exists():
            return None
        
        # Search for function in source files
        search_pattern = f"{module_name}" if module_name else "*"
        cpp_files = list(target_dir.glob(f"{search_pattern}*.cpp"))
        
        for cpp_file in cpp_files:
            code = self._extract_function_code(cpp_file, function_name)
            if code:
                return code
        
        return None
    
    def _extract_function_code(
        self,
        cpp_file: Path,
        function_name: str
    ) -> Optional[str]:
        """Extract function code from C++ file."""
        try:
            content = cpp_file.read_text()
            
            # Find function definition
            # Pattern matches: return_type ClassName::FunctionName(params) {
            pattern = rf'(?:^|\n)([^\n]*?{function_name}\s*\([^)]*\)[^{{]*\{{)'
            
            match = re.search(pattern, content)
            if not match:
                return None
            
            start = match.start()
            
            # Find matching closing brace
            pos = match.end()
            brace_count = 1
            
            while pos < len(content) and brace_count > 0:
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            return content[start:pos].strip()
        
        except Exception:
            return None
    
    def get_reference_backends(self, target: str) -> List[str]:
        """
        Get list of reference backends for a target.
        
        For VEGA-style training, we use other backends as references.
        
        Args:
            target: Target to find references for
        
        Returns:
            List of reference backend names
        """
        all_targets = list(self.KNOWN_TARGETS.keys())
        return [t for t in all_targets if t != target]
    
    def get_supported_targets(self) -> List[str]:
        """Get list of supported target backends."""
        return list(self.KNOWN_TARGETS.keys())


# Module-level function
def create_llvm_adapter(
    llvm_path: Optional[str] = None,
    cache_dir: str = "llvm_cache",
    **kwargs
) -> LLVMAdapter:
    """
    Create an LLVM adapter.
    
    Args:
        llvm_path: Path to LLVM sources
        cache_dir: Cache directory
        **kwargs: Additional arguments
    
    Returns:
        Configured LLVMAdapter instance
    """
    return LLVMAdapter(llvm_path=llvm_path, cache_dir=cache_dir, **kwargs)
