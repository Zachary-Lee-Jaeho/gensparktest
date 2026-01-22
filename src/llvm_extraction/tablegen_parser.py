"""
LLVM TableGen Parser for VEGA-Verified.

TableGen is LLVM's domain-specific language for defining:
- Instructions (InstrInfo.td)
- Registers (RegisterInfo.td)
- Scheduling models (Schedule.td)
- Calling conventions
- Patterns for instruction selection

This parser extracts structured information from TableGen files
to build a comprehensive understanding of target architectures.

Key extractions:
1. Instruction definitions (opcode, operands, encoding)
2. Register definitions (names, aliases, classes)
3. Pattern definitions (DAG patterns for ISel)
4. SDNode definitions (custom DAG nodes)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import re
import json
from pathlib import Path


class TableGenRecordType(Enum):
    """Types of TableGen records."""
    DEF = "def"
    CLASS = "class"
    MULTICLASS = "multiclass"
    DEFM = "defm"
    LET = "let"
    FOREACH = "foreach"
    INCLUDE = "include"


@dataclass
class TableGenField:
    """A field in a TableGen record."""
    name: str
    value: Any
    field_type: Optional[str] = None
    

@dataclass
class TableGenRecord:
    """A TableGen record (def, class, etc.)."""
    record_type: TableGenRecordType
    name: str
    parent_classes: List[str] = field(default_factory=list)
    fields: Dict[str, TableGenField] = field(default_factory=dict)
    body: str = ""
    line_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.record_type.value,
            'name': self.name,
            'parent_classes': self.parent_classes,
            'fields': {k: {'name': v.name, 'value': str(v.value), 'type': v.field_type} 
                      for k, v in self.fields.items()},
        }


@dataclass
class InstructionDef:
    """Parsed instruction definition."""
    name: str
    opcode: Optional[str] = None
    operands: List[str] = field(default_factory=list)
    assembly_string: Optional[str] = None
    encoding: Optional[str] = None
    pattern: Optional[str] = None
    predicates: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'opcode': self.opcode,
            'operands': self.operands,
            'assembly': self.assembly_string,
            'encoding': self.encoding,
            'pattern': self.pattern,
            'predicates': self.predicates,
            'flags': self.flags,
        }


@dataclass
class RegisterDef:
    """Parsed register definition."""
    name: str
    encoding: Optional[int] = None
    alt_names: List[str] = field(default_factory=list)
    sub_regs: List[str] = field(default_factory=list)
    reg_class: Optional[str] = None
    namespace: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'encoding': self.encoding,
            'alt_names': self.alt_names,
            'sub_regs': self.sub_regs,
            'reg_class': self.reg_class,
            'namespace': self.namespace,
        }


@dataclass
class SDNodeDef:
    """Parsed SDNode (SelectionDAG node) definition."""
    name: str
    opcode: str
    type_profile: Optional[str] = None
    properties: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'opcode': self.opcode,
            'type_profile': self.type_profile,
            'properties': self.properties,
        }


@dataclass  
class PatternDef:
    """Parsed pattern definition for instruction selection."""
    name: str
    source_pattern: str
    dest_pattern: str
    predicates: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'source': self.source_pattern,
            'dest': self.dest_pattern,
            'predicates': self.predicates,
        }


class TableGenParser:
    """
    Parser for LLVM TableGen files.
    
    Handles:
    - def/class/multiclass declarations
    - let bindings
    - include directives
    - Nested structures
    """
    
    # Patterns for different constructs
    DEF_PATTERN = re.compile(
        r'^def\s+(\w+)\s*(?::\s*([^{]+))?\s*\{([^}]*)\}',
        re.MULTILINE | re.DOTALL
    )
    
    CLASS_PATTERN = re.compile(
        r'^class\s+(\w+)\s*(?:<([^>]*)>)?\s*(?::\s*([^{]+))?\s*\{([^}]*)\}',
        re.MULTILINE | re.DOTALL
    )
    
    LET_PATTERN = re.compile(
        r'let\s+(\w+)\s*=\s*([^;]+);',
        re.MULTILINE
    )
    
    SDNODE_PATTERN = re.compile(
        r'def\s+(\w+)\s*:\s*SDNode<"([^"]+)",\s*(\w+)(?:,\s*\[([^\]]*)\])?\s*>',
        re.MULTILINE
    )
    
    REGISTER_PATTERN = re.compile(
        r'def\s+(\w+)\s*:\s*(\w*Reg\w*)<([^>]+)>',
        re.MULTILINE
    )
    
    INSTRUCTION_PATTERN = re.compile(
        r'def\s+(\w+)\s*:\s*(\w*Inst\w*)<([^>]*)>(?:\s*,\s*Sched<([^>]*)>)?',
        re.MULTILINE
    )
    
    INCLUDE_PATTERN = re.compile(
        r'include\s+"([^"]+)"',
        re.MULTILINE
    )
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.records: Dict[str, TableGenRecord] = {}
        self.instructions: Dict[str, InstructionDef] = {}
        self.registers: Dict[str, RegisterDef] = {}
        self.sdnodes: Dict[str, SDNodeDef] = {}
        self.patterns: List[PatternDef] = []
        self.includes: List[str] = []
        self.current_namespace: str = ""
    
    def parse(self, content: str, filename: str = "") -> Dict[str, Any]:
        """Parse TableGen content."""
        # Track current namespace
        namespace_match = re.search(r'let\s+Namespace\s*=\s*"(\w+)"', content)
        if namespace_match:
            self.current_namespace = namespace_match.group(1)
        
        # Parse includes
        self._parse_includes(content)
        
        # Parse SDNodes
        self._parse_sdnodes(content)
        
        # Parse registers
        self._parse_registers(content)
        
        # Parse instructions
        self._parse_instructions(content)
        
        # Parse general defs
        self._parse_defs(content)
        
        return self.get_results()
    
    def _parse_includes(self, content: str):
        """Parse include directives."""
        for match in self.INCLUDE_PATTERN.finditer(content):
            self.includes.append(match.group(1))
    
    def _parse_sdnodes(self, content: str):
        """Parse SDNode definitions."""
        for match in self.SDNODE_PATTERN.finditer(content):
            name = match.group(1)
            opcode = match.group(2)
            type_profile = match.group(3)
            props_str = match.group(4) or ""
            
            # Parse properties
            properties = [p.strip() for p in props_str.split(',') if p.strip()]
            
            self.sdnodes[name] = SDNodeDef(
                name=name,
                opcode=opcode,
                type_profile=type_profile,
                properties=properties
            )
    
    def _parse_registers(self, content: str):
        """Parse register definitions."""
        for match in self.REGISTER_PATTERN.finditer(content):
            name = match.group(1)
            reg_class = match.group(2)
            args = match.group(3)
            
            # Parse arguments
            encoding = None
            alt_names = []
            
            # Try to extract encoding (first numeric arg)
            enc_match = re.search(r'(\d+)', args)
            if enc_match:
                encoding = int(enc_match.group(1))
            
            # Try to extract alt names
            alt_match = re.search(r'"([^"]+)"', args)
            if alt_match:
                alt_names = [alt_match.group(1)]
            
            self.registers[name] = RegisterDef(
                name=name,
                encoding=encoding,
                alt_names=alt_names,
                reg_class=reg_class,
                namespace=self.current_namespace
            )
    
    def _parse_instructions(self, content: str):
        """Parse instruction definitions."""
        # Pattern for instruction definitions
        inst_pattern = re.compile(
            r'def\s+(\w+)\s*:\s*([A-Z]\w*)\s*<([^>]*)>',
            re.MULTILINE
        )
        
        for match in inst_pattern.finditer(content):
            name = match.group(1)
            inst_class = match.group(2)
            args = match.group(3)
            
            # Skip if not an instruction
            if 'Inst' not in inst_class and 'Pseudo' not in inst_class:
                continue
            
            # Parse operands from args
            operands = []
            ops_match = re.search(r'\(([^)]+)\)', args)
            if ops_match:
                operands = [o.strip() for o in ops_match.group(1).split(',')]
            
            # Parse assembly string
            asm_match = re.search(r'"([^"]*)"', args)
            asm_string = asm_match.group(1) if asm_match else None
            
            self.instructions[name] = InstructionDef(
                name=name,
                operands=operands,
                assembly_string=asm_string,
            )
    
    def _parse_defs(self, content: str):
        """Parse general def statements."""
        # Simple def pattern
        simple_def = re.compile(
            r'def\s+(\w+)\s*:\s*([^\{;]+)(?:\{([^}]*)\})?',
            re.MULTILINE | re.DOTALL
        )
        
        for match in simple_def.finditer(content):
            name = match.group(1)
            parents = match.group(2).strip()
            body = match.group(3) or ""
            
            # Skip already parsed
            if name in self.sdnodes or name in self.registers or name in self.instructions:
                continue
            
            # Parse parent classes
            parent_list = [p.strip() for p in parents.split(',') if p.strip()]
            parent_list = [re.sub(r'<[^>]*>', '', p) for p in parent_list]  # Remove template args
            
            # Parse fields from body
            fields = {}
            for let_match in self.LET_PATTERN.finditer(body):
                field_name = let_match.group(1)
                field_value = let_match.group(2).strip()
                fields[field_name] = TableGenField(
                    name=field_name,
                    value=field_value
                )
            
            self.records[name] = TableGenRecord(
                record_type=TableGenRecordType.DEF,
                name=name,
                parent_classes=parent_list,
                fields=fields,
                body=body
            )
    
    def get_results(self) -> Dict[str, Any]:
        """Get all parsed results."""
        return {
            'namespace': self.current_namespace,
            'includes': self.includes,
            'sdnodes': {k: v.to_dict() for k, v in self.sdnodes.items()},
            'registers': {k: v.to_dict() for k, v in self.registers.items()},
            'instructions': {k: v.to_dict() for k, v in self.instructions.items()},
            'records': {k: v.to_dict() for k, v in self.records.items()},
            'stats': {
                'total_sdnodes': len(self.sdnodes),
                'total_registers': len(self.registers),
                'total_instructions': len(self.instructions),
                'total_records': len(self.records),
            }
        }


class TableGenFetcher:
    """Fetches and parses TableGen files from LLVM GitHub."""
    
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/llvm/llvm-project/release/18.x/llvm/lib/Target"
    
    # Key TableGen files per backend
    TABLEGEN_FILES = {
        'InstrInfo': '{backend}InstrInfo.td',
        'RegisterInfo': '{backend}RegisterInfo.td',
        'InstrFormats': '{backend}InstrFormats.td',
        'CallingConv': '{backend}CallingConv.td',
        'Schedule': '{backend}Schedule.td',
    }
    
    BACKENDS = ['RISCV', 'ARM', 'AArch64', 'X86', 'Mips']
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'vega-verified' / 'tablegen'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.parser = TableGenParser()
    
    def fetch_file(self, backend: str, file_type: str) -> Optional[str]:
        """Fetch a TableGen file from GitHub or cache."""
        import urllib.request
        import urllib.error
        
        filename = self.TABLEGEN_FILES.get(file_type, '').format(backend=backend)
        if not filename:
            return None
        
        # Check cache
        cache_path = self.cache_dir / backend / filename
        if cache_path.exists():
            return cache_path.read_text()
        
        # Fetch from GitHub
        url = f"{self.GITHUB_RAW_URL}/{backend}/{filename}"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode('utf-8')
                
                # Cache
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(content)
                
                return content
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def fetch_backend(self, backend: str) -> Dict[str, Any]:
        """Fetch and parse all TableGen files for a backend."""
        results = {
            'backend': backend,
            'files': {},
            'combined': {
                'sdnodes': {},
                'registers': {},
                'instructions': {},
            }
        }
        
        for file_type in self.TABLEGEN_FILES:
            content = self.fetch_file(backend, file_type)
            if content:
                parser = TableGenParser()
                parsed = parser.parse(content, f"{backend}/{file_type}")
                results['files'][file_type] = parsed
                
                # Combine results
                results['combined']['sdnodes'].update(parsed.get('sdnodes', {}))
                results['combined']['registers'].update(parsed.get('registers', {}))
                results['combined']['instructions'].update(parsed.get('instructions', {}))
        
        return results
    
    def fetch_all_backends(self) -> Dict[str, Any]:
        """Fetch TableGen data for all backends."""
        all_results = {}
        
        for backend in self.BACKENDS:
            print(f"Fetching TableGen for {backend}...")
            all_results[backend] = self.fetch_backend(backend)
        
        return all_results


class TableGenDatabase:
    """Database for storing parsed TableGen information."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path('data/tablegen_database.json')
        self.data: Dict[str, Any] = {
            'version': '1.0',
            'backends': {}
        }
    
    def add_backend(self, backend: str, data: Dict[str, Any]):
        """Add parsed TableGen data for a backend."""
        self.data['backends'][backend] = data
    
    def save(self):
        """Save database to file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def load(self) -> bool:
        """Load database from file."""
        if self.db_path.exists():
            with open(self.db_path) as f:
                self.data = json.load(f)
            return True
        return False
    
    def get_instructions(self, backend: str) -> Dict[str, Any]:
        """Get all instructions for a backend."""
        return self.data.get('backends', {}).get(backend, {}).get('combined', {}).get('instructions', {})
    
    def get_registers(self, backend: str) -> Dict[str, Any]:
        """Get all registers for a backend."""
        return self.data.get('backends', {}).get(backend, {}).get('combined', {}).get('registers', {})
    
    def get_sdnodes(self, backend: str) -> Dict[str, Any]:
        """Get all SDNodes for a backend."""
        return self.data.get('backends', {}).get(backend, {}).get('combined', {}).get('sdnodes', {})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        stats = {
            'backends': list(self.data.get('backends', {}).keys()),
            'per_backend': {}
        }
        
        for backend, data in self.data.get('backends', {}).items():
            combined = data.get('combined', {})
            stats['per_backend'][backend] = {
                'sdnodes': len(combined.get('sdnodes', {})),
                'registers': len(combined.get('registers', {})),
                'instructions': len(combined.get('instructions', {})),
            }
        
        return stats


# Demo / Test
if __name__ == '__main__':
    print("=" * 70)
    print("TableGen Parser Demo")
    print("=" * 70)
    
    # Test with fetcher
    fetcher = TableGenFetcher()
    
    # Fetch RISCV
    print("\n📥 Fetching RISCV TableGen files...")
    riscv_data = fetcher.fetch_backend('RISCV')
    
    print(f"\n📊 RISCV Results:")
    print(f"  Files parsed: {list(riscv_data['files'].keys())}")
    print(f"  SDNodes: {len(riscv_data['combined']['sdnodes'])}")
    print(f"  Registers: {len(riscv_data['combined']['registers'])}")
    print(f"  Instructions: {len(riscv_data['combined']['instructions'])}")
    
    # Show some SDNodes
    print(f"\n📌 Sample SDNodes:")
    for i, (name, node) in enumerate(list(riscv_data['combined']['sdnodes'].items())[:5]):
        print(f"  {name}: {node['opcode']}")
    
    # Show some registers
    print(f"\n📌 Sample Registers:")
    for i, (name, reg) in enumerate(list(riscv_data['combined']['registers'].items())[:5]):
        print(f"  {name}: encoding={reg['encoding']}, class={reg['reg_class']}")
    
    # Save to database
    print("\n💾 Saving to database...")
    db = TableGenDatabase()
    db.add_backend('RISCV', riscv_data)
    db.save()
    
    print(f"\n✅ Saved to {db.db_path}")
    print(f"   Stats: {db.get_stats()}")
    
    print("\n" + "=" * 70)
    print("✅ TableGen Parser Demo Complete")
    print("=" * 70)
