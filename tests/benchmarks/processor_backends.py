"""
Comprehensive Processor Backend Benchmarks for VEGA-Verified.

This module provides realistic test cases for multiple processor architectures:
1. RISC-V (from VEGA paper - primary target)
2. ARM/AArch64 (widely used mobile/server)
3. MIPS (classic RISC architecture)
4. x86-64 (dominant desktop/server)
5. PowerPC (enterprise/embedded)
6. SPARC (legacy enterprise)

Each backend includes:
- MCCodeEmitter functions (instruction encoding)
- AsmPrinter functions (assembly output)
- ELFObjectWriter functions (relocation handling)
- ISelDAGToDAG functions (instruction selection)

These benchmarks test VEGA-Verified's ability to:
- Infer specifications from diverse architectures
- Verify semantic correctness across different ISAs
- Repair common backend bugs automatically
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.specification.spec_language import (
    Specification, Condition, ConditionType, Variable, Constant
)
from src.hierarchical.module_verify import Module, ModuleFunction
from src.hierarchical.backend_verify import Backend
from src.hierarchical.interface_contract import (
    InterfaceContract, ContractType, Assumption, Guarantee
)


class ProcessorFamily(Enum):
    """Processor architecture families."""
    RISCV = "riscv"
    ARM = "arm"
    AARCH64 = "aarch64"
    MIPS = "mips"
    X86_64 = "x86_64"
    POWERPC = "powerpc"
    SPARC = "sparc"


@dataclass
class ProcessorBackendBenchmark:
    """Benchmark for a processor backend."""
    name: str
    family: ProcessorFamily
    triple: str
    description: str
    modules: Dict[str, Module] = field(default_factory=dict)
    expected_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_backend(self) -> Backend:
        """Convert to Backend object."""
        backend = Backend(name=self.name, target_triple=self.triple)
        for module in self.modules.values():
            backend.add_module(module)
        return backend


# =============================================================================
# RISC-V Backend (Primary VEGA target)
# =============================================================================

def create_riscv_benchmark() -> ProcessorBackendBenchmark:
    """
    Create RISC-V backend benchmark.
    
    RISC-V is the primary target from the VEGA paper with:
    - 71.5% function-level accuracy
    - 55% statement-level accuracy
    - Key modules: MCCodeEmitter, AsmPrinter, ELFObjectWriter
    """
    benchmark = ProcessorBackendBenchmark(
        name="RISCV",
        family=ProcessorFamily.RISCV,
        triple="riscv64-unknown-linux-gnu",
        description="RISC-V 64-bit backend (VEGA paper primary target)",
        expected_metrics={
            "vega_function_accuracy": 0.715,
            "vega_statement_accuracy": 0.55,
            "target_function_accuracy": 0.85,
            "target_verified_rate": 0.80,
        }
    )
    
    # MCCodeEmitter Module
    mc_emitter = Module(name="MCCodeEmitter")
    
    # encodeInstruction - Main encoding function
    mc_emitter.add_function(ModuleFunction(
        name="encodeInstruction",
        code="""
void RISCVMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                           SmallVectorImpl<char> &CB,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
    const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
    unsigned Size = Desc.getSize();
    
    // Handle compressed instructions (RVC)
    if (Size == 2) {
        uint16_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
        support::endian::write<uint16_t>(CB, Bits, support::little);
    } else {
        uint32_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
        support::endian::write<uint32_t>(CB, Bits, support::little);
    }
}
        """,
        specification=Specification(
            function_name="encodeInstruction",
            module="MCCodeEmitter",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MI")),
                Condition(ConditionType.IS_VALID, Variable("STI")),
            ],
            postconditions=[
                Condition(ConditionType.GREATER_EQUAL, Variable("encoded_size"), Constant(2)),
                Condition(ConditionType.LESS_EQUAL, Variable("encoded_size"), Constant(4)),
            ],
            invariants=[
                Condition(ConditionType.IMPLIES,
                    Condition(ConditionType.EQUALITY, Variable("is_compressed"), Constant(True)),
                    Condition(ConditionType.EQUALITY, Variable("encoded_size"), Constant(2))
                ),
            ]
        ),
        is_interface=True
    ))
    
    # getMachineOpValue - Operand encoding
    mc_emitter.add_function(ModuleFunction(
        name="getMachineOpValue",
        code="""
unsigned RISCVMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                                const MCOperand &MO,
                                                SmallVectorImpl<MCFixup> &Fixups,
                                                const MCSubtargetInfo &STI) const {
    if (MO.isReg())
        return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());
    if (MO.isImm())
        return static_cast<unsigned>(MO.getImm());
    
    llvm_unreachable("Unhandled expression!");
    return 0;
}
        """,
        specification=Specification(
            function_name="getMachineOpValue",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MO")),
            ],
            postconditions=[
                Condition(ConditionType.GREATER_EQUAL, Variable("result"), Constant(0)),
            ]
        ),
        is_interface=True
    ))
    
    # getImmOpValue - Immediate encoding
    mc_emitter.add_function(ModuleFunction(
        name="getImmOpValue",
        code="""
unsigned RISCVMCCodeEmitter::getImmOpValue(const MCInst &MI, unsigned OpNo,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
    const MCOperand &MO = MI.getOperand(OpNo);
    
    MCFixupKind Kind = MCFixupKind(RISCV::fixup_riscv_invalid);
    if (MO.isImm())
        return MO.getImm();
    
    assert(MO.isExpr() && "getImmOpValue expects only expressions or immediates");
    const MCExpr *Expr = MO.getExpr();
    
    // Handle RISCV-specific fixups
    if (const RISCVMCExpr *RVExpr = dyn_cast<RISCVMCExpr>(Expr)) {
        switch (RVExpr->getKind()) {
        case RISCVMCExpr::VK_RISCV_LO:
            Kind = MCFixupKind(RISCV::fixup_riscv_lo12_i);
            break;
        case RISCVMCExpr::VK_RISCV_HI:
            Kind = MCFixupKind(RISCV::fixup_riscv_hi20);
            break;
        default:
            break;
        }
    }
    
    Fixups.push_back(MCFixup::create(0, Expr, Kind, MI.getLoc()));
    return 0;
}
        """,
        specification=Specification(
            function_name="getImmOpValue",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MI")),
                Condition(ConditionType.LESS_THAN, Variable("OpNo"), Variable("MI.getNumOperands()")),
            ]
        ),
        is_interface=True
    ))
    
    # Create contract for MCCodeEmitter
    mc_contract = InterfaceContract(
        name="RISCV_MCCodeEmitter_IFC",
        module_name="MCCodeEmitter",
        contract_type=ContractType.MODULE
    )
    mc_contract.add_assumption(Assumption(
        "valid_instruction",
        "Input instruction is a valid RISC-V instruction",
        "(and (>= opcode 0) (< opcode 2048))"
    ))
    mc_contract.add_guarantee(Guarantee(
        "correct_encoding",
        "Encoded bytes correctly represent the instruction",
        "(= (decode (encode MI)) MI)"
    ))
    mc_emitter.interface_contract = mc_contract
    
    benchmark.modules["MCCodeEmitter"] = mc_emitter
    
    # ELFObjectWriter Module
    elf_writer = Module(name="ELFObjectWriter")
    
    # getRelocType - Main relocation mapping
    elf_writer.add_function(ModuleFunction(
        name="getRelocType",
        code="""
unsigned RISCVELFObjectWriter::getRelocType(MCContext &Ctx,
                                             const MCValue &Target,
                                             const MCFixup &Fixup,
                                             bool IsPCRel) const {
    const MCExpr *Expr = Fixup.getValue();
    unsigned Kind = Fixup.getTargetKind();
    
    if (IsPCRel) {
        switch (Kind) {
        default:
            Ctx.reportError(Fixup.getLoc(), "unsupported relocation type");
            return ELF::R_RISCV_NONE;
        case FK_Data_4:
        case FK_PCRel_4:
            return ELF::R_RISCV_32_PCREL;
        case RISCV::fixup_riscv_pcrel_hi20:
            return ELF::R_RISCV_PCREL_HI20;
        case RISCV::fixup_riscv_pcrel_lo12_i:
            return ELF::R_RISCV_PCREL_LO12_I;
        case RISCV::fixup_riscv_pcrel_lo12_s:
            return ELF::R_RISCV_PCREL_LO12_S;
        }
    }
    
    switch (Kind) {
    default:
        Ctx.reportError(Fixup.getLoc(), "unsupported relocation type");
        return ELF::R_RISCV_NONE;
    case FK_NONE:
        return ELF::R_RISCV_NONE;
    case FK_Data_1:
        Ctx.reportError(Fixup.getLoc(), "1-byte data relocations not supported");
        return ELF::R_RISCV_NONE;
    case FK_Data_2:
        Ctx.reportError(Fixup.getLoc(), "2-byte data relocations not supported");
        return ELF::R_RISCV_NONE;
    case FK_Data_4:
        return ELF::R_RISCV_32;
    case FK_Data_8:
        return ELF::R_RISCV_64;
    case RISCV::fixup_riscv_hi20:
        return ELF::R_RISCV_HI20;
    case RISCV::fixup_riscv_lo12_i:
        return ELF::R_RISCV_LO12_I;
    case RISCV::fixup_riscv_lo12_s:
        return ELF::R_RISCV_LO12_S;
    case RISCV::fixup_riscv_jal:
        return ELF::R_RISCV_JAL;
    case RISCV::fixup_riscv_branch:
        return ELF::R_RISCV_BRANCH;
    }
}
        """,
        specification=Specification(
            function_name="getRelocType",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("Fixup")),
            ],
            postconditions=[
                Condition(ConditionType.GREATER_EQUAL, Variable("result"), Constant(0)),
            ],
            invariants=[
                # FK_NONE -> R_RISCV_NONE
                Condition(ConditionType.IMPLIES,
                    Condition(ConditionType.EQUALITY, Variable("Kind"), Constant("FK_NONE")),
                    Condition(ConditionType.EQUALITY, Variable("result"), Constant("R_RISCV_NONE"))
                ),
                # FK_Data_4 && !IsPCRel -> R_RISCV_32
                Condition(ConditionType.IMPLIES,
                    Condition(ConditionType.AND,
                        Condition(ConditionType.EQUALITY, Variable("Kind"), Constant("FK_Data_4")),
                        Condition(ConditionType.NOT, Condition(ConditionType.EQUALITY, Variable("IsPCRel"), Constant(True)))
                    ),
                    Condition(ConditionType.EQUALITY, Variable("result"), Constant("R_RISCV_32"))
                ),
            ]
        ),
        is_interface=True
    ))
    
    # needsRelocateWithSymbol
    elf_writer.add_function(ModuleFunction(
        name="needsRelocateWithSymbol",
        code="""
bool RISCVELFObjectWriter::needsRelocateWithSymbol(const MCSymbol &Sym,
                                                    unsigned Type) const {
    switch (Type) {
    default:
        return false;
    case ELF::R_RISCV_GOT_HI20:
    case ELF::R_RISCV_TLS_GOT_HI20:
    case ELF::R_RISCV_TLS_GD_HI20:
        return true;
    }
}
        """,
        specification=Specification(
            function_name="needsRelocateWithSymbol",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("Sym")),
            ]
        ),
        is_interface=True
    ))
    
    elf_contract = InterfaceContract(
        name="RISCV_ELFObjectWriter_IFC",
        module_name="ELFObjectWriter",
        contract_type=ContractType.MODULE
    )
    elf_contract.add_assumption(Assumption(
        "valid_fixup",
        "Fixup kind is valid for RISC-V",
        "(member Kind RISCV_FIXUP_KINDS)"
    ))
    elf_contract.add_guarantee(Guarantee(
        "valid_reloc",
        "Result is a valid RISC-V relocation type",
        "(member result RISCV_RELOC_TYPES)"
    ))
    elf_writer.interface_contract = elf_contract
    
    benchmark.modules["ELFObjectWriter"] = elf_writer
    
    # AsmPrinter Module
    asm_printer = Module(name="AsmPrinter")
    
    asm_printer.add_function(ModuleFunction(
        name="emitInstruction",
        code="""
void RISCVAsmPrinter::emitInstruction(const MachineInstr *MI) {
    MCInst TmpInst;
    if (!lowerRISCVMachineInstrToMCInst(MI, TmpInst, *this))
        EmitToStreamer(*OutStreamer, TmpInst);
}
        """,
        specification=Specification(
            function_name="emitInstruction",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MI")),
            ]
        ),
        is_interface=True
    ))
    
    benchmark.modules["AsmPrinter"] = asm_printer
    
    return benchmark


# =============================================================================
# ARM/AArch64 Backend
# =============================================================================

def create_arm_benchmark() -> ProcessorBackendBenchmark:
    """
    Create ARM (32-bit) backend benchmark.
    
    ARM is one of the most widely used architectures with:
    - Complex instruction encoding (Thumb, ARM, Thumb-2)
    - Conditional execution
    - NEON SIMD support
    """
    benchmark = ProcessorBackendBenchmark(
        name="ARM",
        family=ProcessorFamily.ARM,
        triple="arm-none-eabi",
        description="ARM 32-bit backend (Thumb/ARM/Thumb-2)",
        expected_metrics={
            "target_function_accuracy": 0.82,
            "target_verified_rate": 0.75,
        }
    )
    
    # MCCodeEmitter Module
    mc_emitter = Module(name="MCCodeEmitter")
    
    mc_emitter.add_function(ModuleFunction(
        name="encodeInstruction",
        code="""
void ARMMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                          SmallVectorImpl<char> &CB,
                                          SmallVectorImpl<MCFixup> &Fixups,
                                          const MCSubtargetInfo &STI) const {
    unsigned Opcode = MI.getOpcode();
    const MCInstrDesc &Desc = MCII.get(Opcode);
    uint64_t TSFlags = Desc.TSFlags;
    
    // Determine instruction format
    if (TSFlags & ARMII::ThumbFrm) {
        // Thumb mode encoding
        uint32_t Binary = getBinaryCodeForInstr(MI, Fixups, STI);
        if (Desc.getSize() == 2) {
            support::endian::write<uint16_t>(CB, Binary, IsLittleEndian ? 
                support::little : support::big);
        } else {
            // Thumb-2: 32-bit encoding, high halfword first
            support::endian::write<uint16_t>(CB, Binary >> 16, 
                IsLittleEndian ? support::little : support::big);
            support::endian::write<uint16_t>(CB, Binary & 0xFFFF,
                IsLittleEndian ? support::little : support::big);
        }
    } else {
        // ARM mode encoding
        uint32_t Binary = getBinaryCodeForInstr(MI, Fixups, STI);
        support::endian::write<uint32_t>(CB, Binary, 
            IsLittleEndian ? support::little : support::big);
    }
}
        """,
        specification=Specification(
            function_name="encodeInstruction",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MI")),
            ],
            postconditions=[
                Condition(ConditionType.GREATER_EQUAL, Variable("encoded_size"), Constant(2)),
                Condition(ConditionType.LESS_EQUAL, Variable("encoded_size"), Constant(4)),
            ],
            invariants=[
                Condition(ConditionType.IMPLIES,
                    Condition(ConditionType.EQUALITY, Variable("is_thumb"), Constant(True)),
                    Condition(ConditionType.OR,
                        Condition(ConditionType.EQUALITY, Variable("encoded_size"), Constant(2)),
                        Condition(ConditionType.EQUALITY, Variable("encoded_size"), Constant(4))
                    )
                ),
            ]
        ),
        is_interface=True
    ))
    
    mc_emitter.add_function(ModuleFunction(
        name="getCondCode",
        code="""
unsigned ARMMCCodeEmitter::getCondCode(ARMCC::CondCodes CC) const {
    switch (CC) {
    case ARMCC::EQ: return 0;
    case ARMCC::NE: return 1;
    case ARMCC::HS: return 2;
    case ARMCC::LO: return 3;
    case ARMCC::MI: return 4;
    case ARMCC::PL: return 5;
    case ARMCC::VS: return 6;
    case ARMCC::VC: return 7;
    case ARMCC::HI: return 8;
    case ARMCC::LS: return 9;
    case ARMCC::GE: return 10;
    case ARMCC::LT: return 11;
    case ARMCC::GT: return 12;
    case ARMCC::LE: return 13;
    case ARMCC::AL: return 14;
    default:
        llvm_unreachable("Invalid condition code!");
    }
}
        """,
        specification=Specification(
            function_name="getCondCode",
            postconditions=[
                Condition(ConditionType.GREATER_EQUAL, Variable("result"), Constant(0)),
                Condition(ConditionType.LESS_EQUAL, Variable("result"), Constant(14)),
            ]
        ),
        is_interface=True
    ))
    
    benchmark.modules["MCCodeEmitter"] = mc_emitter
    
    # ELFObjectWriter
    elf_writer = Module(name="ELFObjectWriter")
    
    elf_writer.add_function(ModuleFunction(
        name="getRelocType",
        code="""
unsigned ARMELFObjectWriter::getRelocType(MCContext &Ctx,
                                           const MCValue &Target,
                                           const MCFixup &Fixup,
                                           bool IsPCRel) const {
    unsigned Kind = Fixup.getTargetKind();
    
    if (IsPCRel) {
        switch (Kind) {
        default:
            return ELF::R_ARM_REL32;
        case ARM::fixup_arm_thumb_bl:
        case ARM::fixup_arm_thumb_blx:
            return ELF::R_ARM_THM_CALL;
        case ARM::fixup_arm_uncondbl:
        case ARM::fixup_arm_condbl:
            return ELF::R_ARM_CALL;
        case ARM::fixup_arm_condbranch:
        case ARM::fixup_arm_uncondbranch:
            return ELF::R_ARM_JUMP24;
        }
    }
    
    switch (Kind) {
    default:
        return ELF::R_ARM_NONE;
    case FK_Data_1:
        return ELF::R_ARM_ABS8;
    case FK_Data_2:
        return ELF::R_ARM_ABS16;
    case FK_Data_4:
        return ELF::R_ARM_ABS32;
    case ARM::fixup_arm_movt_hi16:
        return ELF::R_ARM_MOVT_ABS;
    case ARM::fixup_arm_movw_lo16:
        return ELF::R_ARM_MOVW_ABS_NC;
    }
}
        """,
        specification=Specification(
            function_name="getRelocType",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("Fixup")),
            ]
        ),
        is_interface=True
    ))
    
    benchmark.modules["ELFObjectWriter"] = elf_writer
    
    return benchmark


def create_aarch64_benchmark() -> ProcessorBackendBenchmark:
    """
    Create AArch64 (ARM 64-bit) backend benchmark.
    
    AArch64 features:
    - Fixed 32-bit instruction encoding
    - 31 general-purpose registers
    - Advanced SIMD (NEON) and SVE
    """
    benchmark = ProcessorBackendBenchmark(
        name="AArch64",
        family=ProcessorFamily.AARCH64,
        triple="aarch64-linux-gnu",
        description="ARM 64-bit backend (AArch64)",
        expected_metrics={
            "target_function_accuracy": 0.85,
            "target_verified_rate": 0.80,
        }
    )
    
    # MCCodeEmitter
    mc_emitter = Module(name="MCCodeEmitter")
    
    mc_emitter.add_function(ModuleFunction(
        name="encodeInstruction",
        code="""
void AArch64MCCodeEmitter::encodeInstruction(const MCInst &MI,
                                              SmallVectorImpl<char> &CB,
                                              SmallVectorImpl<MCFixup> &Fixups,
                                              const MCSubtargetInfo &STI) const {
    // AArch64 instructions are always 32 bits
    uint32_t Binary = getBinaryCodeForInstr(MI, Fixups, STI);
    support::endian::write<uint32_t>(CB, Binary, support::little);
}
        """,
        specification=Specification(
            function_name="encodeInstruction",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MI")),
            ],
            postconditions=[
                Condition(ConditionType.EQUALITY, Variable("encoded_size"), Constant(4)),
            ]
        ),
        is_interface=True
    ))
    
    mc_emitter.add_function(ModuleFunction(
        name="getAddSubImmOpValue",
        code="""
uint32_t AArch64MCCodeEmitter::getAddSubImmOpValue(const MCInst &MI,
                                                    unsigned OpIdx,
                                                    SmallVectorImpl<MCFixup> &Fixups,
                                                    const MCSubtargetInfo &STI) const {
    const MCOperand &MO = MI.getOperand(OpIdx);
    if (MO.isImm()) {
        unsigned Val = MO.getImm();
        // Check for shifted immediate
        if (Val & 0xfff000) {
            // 12-bit immediate, shifted by 12
            return ((Val >> 12) & 0xfff) | (1 << 12);
        }
        return Val & 0xfff;
    }
    return 0;
}
        """,
        specification=Specification(
            function_name="getAddSubImmOpValue",
            postconditions=[
                Condition(ConditionType.LESS_THAN, Variable("result"), Constant(0x2000)),
            ]
        ),
        is_interface=True
    ))
    
    benchmark.modules["MCCodeEmitter"] = mc_emitter
    
    # ELFObjectWriter
    elf_writer = Module(name="ELFObjectWriter")
    
    elf_writer.add_function(ModuleFunction(
        name="getRelocType",
        code="""
unsigned AArch64ELFObjectWriter::getRelocType(MCContext &Ctx,
                                               const MCValue &Target,
                                               const MCFixup &Fixup,
                                               bool IsPCRel) const {
    unsigned Kind = Fixup.getTargetKind();
    
    if (IsPCRel) {
        switch (Kind) {
        case FK_Data_2:
            return ELF::R_AARCH64_PREL16;
        case FK_Data_4:
            return ELF::R_AARCH64_PREL32;
        case FK_Data_8:
            return ELF::R_AARCH64_PREL64;
        case AArch64::fixup_aarch64_pcrel_branch26:
            return ELF::R_AARCH64_JUMP26;
        case AArch64::fixup_aarch64_pcrel_call26:
            return ELF::R_AARCH64_CALL26;
        case AArch64::fixup_aarch64_pcrel_adrp_imm21:
            return ELF::R_AARCH64_ADR_PREL_PG_HI21;
        default:
            return ELF::R_AARCH64_NONE;
        }
    }
    
    switch (Kind) {
    case FK_NONE:
        return ELF::R_AARCH64_NONE;
    case FK_Data_2:
        return ELF::R_AARCH64_ABS16;
    case FK_Data_4:
        return ELF::R_AARCH64_ABS32;
    case FK_Data_8:
        return ELF::R_AARCH64_ABS64;
    case AArch64::fixup_aarch64_add_imm12:
        return ELF::R_AARCH64_ADD_ABS_LO12_NC;
    case AArch64::fixup_aarch64_ldst_imm12_scale1:
        return ELF::R_AARCH64_LDST8_ABS_LO12_NC;
    default:
        return ELF::R_AARCH64_NONE;
    }
}
        """,
        specification=Specification(function_name="getRelocType"),
        is_interface=True
    ))
    
    benchmark.modules["ELFObjectWriter"] = elf_writer
    
    return benchmark


# =============================================================================
# MIPS Backend
# =============================================================================

def create_mips_benchmark() -> ProcessorBackendBenchmark:
    """
    Create MIPS backend benchmark.
    
    MIPS is a classic RISC architecture with:
    - Fixed 32-bit instruction encoding (MIPS32)
    - Delay slots
    - Multiple ABI variants
    """
    benchmark = ProcessorBackendBenchmark(
        name="MIPS",
        family=ProcessorFamily.MIPS,
        triple="mips-linux-gnu",
        description="MIPS 32-bit backend",
        expected_metrics={
            "target_function_accuracy": 0.80,
            "target_verified_rate": 0.75,
        }
    )
    
    # MCCodeEmitter
    mc_emitter = Module(name="MCCodeEmitter")
    
    mc_emitter.add_function(ModuleFunction(
        name="encodeInstruction",
        code="""
void MipsMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                           SmallVectorImpl<char> &CB,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
    uint32_t Binary = getBinaryCodeForInstr(MI, Fixups, STI);
    
    // Handle microMIPS encoding
    if (isMicroMips(STI)) {
        unsigned Size = MCII.get(MI.getOpcode()).getSize();
        if (Size == 2) {
            support::endian::write<uint16_t>(CB, Binary, 
                IsLittleEndian ? support::little : support::big);
        } else {
            // 32-bit microMIPS: high half first
            support::endian::write<uint16_t>(CB, (Binary >> 16) & 0xFFFF,
                IsLittleEndian ? support::little : support::big);
            support::endian::write<uint16_t>(CB, Binary & 0xFFFF,
                IsLittleEndian ? support::little : support::big);
        }
    } else {
        // Standard MIPS32/MIPS64
        support::endian::write<uint32_t>(CB, Binary,
            IsLittleEndian ? support::little : support::big);
    }
}
        """,
        specification=Specification(
            function_name="encodeInstruction",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MI")),
            ]
        ),
        is_interface=True
    ))
    
    mc_emitter.add_function(ModuleFunction(
        name="getBranchTargetOpValue",
        code="""
unsigned MipsMCCodeEmitter::getBranchTargetOpValue(const MCInst &MI,
                                                    unsigned OpNo,
                                                    SmallVectorImpl<MCFixup> &Fixups,
                                                    const MCSubtargetInfo &STI) const {
    const MCOperand &MO = MI.getOperand(OpNo);
    
    if (MO.isImm())
        return MO.getImm() >> 2;  // Word-aligned offset
    
    // Create appropriate fixup
    Fixups.push_back(MCFixup::create(0, MO.getExpr(),
        MCFixupKind(Mips::fixup_Mips_PC16)));
    return 0;
}
        """,
        specification=Specification(function_name="getBranchTargetOpValue"),
        is_interface=True
    ))
    
    benchmark.modules["MCCodeEmitter"] = mc_emitter
    
    # ELFObjectWriter
    elf_writer = Module(name="ELFObjectWriter")
    
    elf_writer.add_function(ModuleFunction(
        name="getRelocType",
        code="""
unsigned MipsELFObjectWriter::getRelocType(MCContext &Ctx,
                                            const MCValue &Target,
                                            const MCFixup &Fixup,
                                            bool IsPCRel) const {
    unsigned Kind = Fixup.getTargetKind();
    
    switch (Kind) {
    case FK_NONE:
        return ELF::R_MIPS_NONE;
    case FK_Data_4:
        return IsPCRel ? ELF::R_MIPS_PC32 : ELF::R_MIPS_32;
    case FK_Data_8:
        return ELF::R_MIPS_64;
    case Mips::fixup_Mips_HI16:
        return ELF::R_MIPS_HI16;
    case Mips::fixup_Mips_LO16:
        return ELF::R_MIPS_LO16;
    case Mips::fixup_Mips_PC16:
        return ELF::R_MIPS_PC16;
    case Mips::fixup_Mips_26:
        return ELF::R_MIPS_26;
    case Mips::fixup_Mips_GOT:
        return ELF::R_MIPS_GOT16;
    case Mips::fixup_Mips_CALL16:
        return ELF::R_MIPS_CALL16;
    default:
        return ELF::R_MIPS_NONE;
    }
}
        """,
        specification=Specification(function_name="getRelocType"),
        is_interface=True
    ))
    
    benchmark.modules["ELFObjectWriter"] = elf_writer
    
    return benchmark


# =============================================================================
# x86-64 Backend
# =============================================================================

def create_x86_64_benchmark() -> ProcessorBackendBenchmark:
    """
    Create x86-64 backend benchmark.
    
    x86-64 features:
    - Variable-length instruction encoding (1-15 bytes)
    - Complex addressing modes
    - Legacy prefixes and VEX/EVEX encoding
    """
    benchmark = ProcessorBackendBenchmark(
        name="X86_64",
        family=ProcessorFamily.X86_64,
        triple="x86_64-linux-gnu",
        description="x86-64 backend (AMD64/Intel 64)",
        expected_metrics={
            "target_function_accuracy": 0.78,
            "target_verified_rate": 0.70,  # Lower due to encoding complexity
        }
    )
    
    # MCCodeEmitter
    mc_emitter = Module(name="MCCodeEmitter")
    
    mc_emitter.add_function(ModuleFunction(
        name="encodeInstruction",
        code="""
void X86MCCodeEmitter::encodeInstruction(const MCInst &MI,
                                          SmallVectorImpl<char> &CB,
                                          SmallVectorImpl<MCFixup> &Fixups,
                                          const MCSubtargetInfo &STI) const {
    unsigned Opcode = MI.getOpcode();
    const MCInstrDesc &Desc = MCII.get(Opcode);
    uint64_t TSFlags = Desc.TSFlags;
    
    // Handle prefixes
    emitPrefixes(MI, CB, STI);
    
    // Emit REX prefix if needed (64-bit mode)
    unsigned REX = determineREXPrefix(MI, STI);
    if (REX)
        emitByte(0x40 | REX, CB);
    
    // Handle VEX/EVEX encoding
    if (TSFlags & X86II::VEX)
        emitVEXOpcode(TSFlags, CB);
    else if (TSFlags & X86II::EVEX)
        emitEVEXOpcode(TSFlags, CB);
    else
        emitOpcodePrefix(TSFlags, CB);
    
    // Emit opcode
    emitOpcode(TSFlags, CB);
    
    // Emit ModR/M and SIB bytes
    emitModRMByte(MI, TSFlags, CB, Fixups, STI);
    
    // Emit immediate operands
    emitImmediateOperands(MI, CB, Fixups, STI);
}
        """,
        specification=Specification(
            function_name="encodeInstruction",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MI")),
            ],
            postconditions=[
                Condition(ConditionType.GREATER_EQUAL, Variable("encoded_size"), Constant(1)),
                Condition(ConditionType.LESS_EQUAL, Variable("encoded_size"), Constant(15)),
            ]
        ),
        is_interface=True
    ))
    
    mc_emitter.add_function(ModuleFunction(
        name="determineREXPrefix",
        code="""
unsigned X86MCCodeEmitter::determineREXPrefix(const MCInst &MI,
                                               const MCSubtargetInfo &STI) const {
    unsigned REX = 0;
    
    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
        const MCOperand &MO = MI.getOperand(i);
        if (!MO.isReg())
            continue;
        
        unsigned Reg = MO.getReg();
        
        // Check if extended register (R8-R15, XMM8-XMM15, etc.)
        if (X86II::isX86_64ExtendedReg(Reg))
            REX |= 0x1 << getOperandREXPrefix(i);
        
        // Check if 64-bit register
        if (X86II::is64BitReg(Reg))
            REX |= 0x8;  // REX.W
    }
    
    return REX;
}
        """,
        specification=Specification(
            function_name="determineREXPrefix",
            postconditions=[
                Condition(ConditionType.LESS_EQUAL, Variable("result"), Constant(0xF)),
            ]
        ),
        is_interface=True
    ))
    
    benchmark.modules["MCCodeEmitter"] = mc_emitter
    
    # ELFObjectWriter
    elf_writer = Module(name="ELFObjectWriter")
    
    elf_writer.add_function(ModuleFunction(
        name="getRelocType",
        code="""
unsigned X86ELFObjectWriter::getRelocType(MCContext &Ctx,
                                           const MCValue &Target,
                                           const MCFixup &Fixup,
                                           bool IsPCRel) const {
    unsigned Kind = Fixup.getTargetKind();
    
    if (IsPCRel) {
        switch (Kind) {
        default:
            return ELF::R_X86_64_NONE;
        case FK_Data_4:
        case FK_PCRel_4:
            return ELF::R_X86_64_PC32;
        case FK_Data_8:
        case FK_PCRel_8:
            return ELF::R_X86_64_PC64;
        case X86::reloc_riprel_4byte:
            return ELF::R_X86_64_PC32;
        case X86::reloc_riprel_4byte_movq_load:
            return ELF::R_X86_64_REX_GOTPCRELX;
        }
    }
    
    switch (Kind) {
    default:
        return ELF::R_X86_64_NONE;
    case FK_NONE:
        return ELF::R_X86_64_NONE;
    case FK_Data_1:
        return ELF::R_X86_64_8;
    case FK_Data_2:
        return ELF::R_X86_64_16;
    case FK_Data_4:
        return ELF::R_X86_64_32;
    case FK_Data_8:
        return ELF::R_X86_64_64;
    case X86::reloc_signed_4byte:
        return ELF::R_X86_64_32S;
    case X86::reloc_global_offset_table:
        return ELF::R_X86_64_GOTPC32;
    }
}
        """,
        specification=Specification(function_name="getRelocType"),
        is_interface=True
    ))
    
    benchmark.modules["ELFObjectWriter"] = elf_writer
    
    return benchmark


# =============================================================================
# PowerPC Backend
# =============================================================================

def create_powerpc_benchmark() -> ProcessorBackendBenchmark:
    """
    Create PowerPC backend benchmark.
    
    PowerPC features:
    - Fixed 32-bit instruction encoding
    - Big-endian (default) and little-endian modes
    - Condition register
    """
    benchmark = ProcessorBackendBenchmark(
        name="PowerPC",
        family=ProcessorFamily.POWERPC,
        triple="powerpc64-linux-gnu",
        description="PowerPC 64-bit backend",
        expected_metrics={
            "target_function_accuracy": 0.80,
            "target_verified_rate": 0.75,
        }
    )
    
    # MCCodeEmitter
    mc_emitter = Module(name="MCCodeEmitter")
    
    mc_emitter.add_function(ModuleFunction(
        name="encodeInstruction",
        code="""
void PPCMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                          SmallVectorImpl<char> &CB,
                                          SmallVectorImpl<MCFixup> &Fixups,
                                          const MCSubtargetInfo &STI) const {
    uint64_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
    
    // PowerPC instructions are always 32 bits, big-endian
    if (IsLittleEndian) {
        support::endian::write<uint32_t>(CB, Bits, support::little);
    } else {
        support::endian::write<uint32_t>(CB, Bits, support::big);
    }
}
        """,
        specification=Specification(
            function_name="encodeInstruction",
            postconditions=[
                Condition(ConditionType.EQUALITY, Variable("encoded_size"), Constant(4)),
            ]
        ),
        is_interface=True
    ))
    
    benchmark.modules["MCCodeEmitter"] = mc_emitter
    
    # ELFObjectWriter
    elf_writer = Module(name="ELFObjectWriter")
    
    elf_writer.add_function(ModuleFunction(
        name="getRelocType",
        code="""
unsigned PPCELFObjectWriter::getRelocType(MCContext &Ctx,
                                           const MCValue &Target,
                                           const MCFixup &Fixup,
                                           bool IsPCRel) const {
    unsigned Kind = Fixup.getTargetKind();
    
    if (IsPCRel) {
        switch (Kind) {
        default:
            return ELF::R_PPC64_REL32;
        case PPC::fixup_ppc_br24:
            return ELF::R_PPC64_REL24;
        case PPC::fixup_ppc_brcond14:
            return ELF::R_PPC64_REL14;
        }
    }
    
    switch (Kind) {
    default:
        return ELF::R_PPC64_NONE;
    case FK_Data_4:
        return ELF::R_PPC64_ADDR32;
    case FK_Data_8:
        return ELF::R_PPC64_ADDR64;
    case PPC::fixup_ppc_lo16:
        return ELF::R_PPC64_ADDR16_LO;
    case PPC::fixup_ppc_hi16:
        return ELF::R_PPC64_ADDR16_HI;
    case PPC::fixup_ppc_ha16:
        return ELF::R_PPC64_ADDR16_HA;
    }
}
        """,
        specification=Specification(function_name="getRelocType"),
        is_interface=True
    ))
    
    benchmark.modules["ELFObjectWriter"] = elf_writer
    
    return benchmark


# =============================================================================
# Benchmark Registry
# =============================================================================

def get_all_benchmarks() -> Dict[str, ProcessorBackendBenchmark]:
    """Get all available processor benchmarks."""
    return {
        "RISCV": create_riscv_benchmark(),
        "ARM": create_arm_benchmark(),
        "AArch64": create_aarch64_benchmark(),
        "MIPS": create_mips_benchmark(),
        "X86_64": create_x86_64_benchmark(),
        "PowerPC": create_powerpc_benchmark(),
    }


def get_vega_benchmarks() -> Dict[str, ProcessorBackendBenchmark]:
    """Get benchmarks from VEGA paper (RISC-V, RI5CY, xCORE).
    
    Note: RI5CY and xCORE are variants/extensions, represented here
    by the base RISC-V benchmark with different configurations.
    """
    riscv = create_riscv_benchmark()
    
    # RI5CY (PULP RISC-V) - adds custom extensions
    ri5cy = ProcessorBackendBenchmark(
        name="RI5CY",
        family=ProcessorFamily.RISCV,
        triple="riscv32-pulp-linux-gnu",
        description="RI5CY (PULP RISC-V with custom extensions)",
        expected_metrics={
            "vega_function_accuracy": 0.732,  # From VEGA paper
            "target_function_accuracy": 0.87,
            "target_verified_rate": 0.82,
        }
    )
    ri5cy.modules = riscv.modules.copy()  # Same base modules
    
    # xCORE - different architecture but similar structure
    xcore = ProcessorBackendBenchmark(
        name="xCORE",
        family=ProcessorFamily.RISCV,  # Simplified for benchmark
        triple="xcore-unknown-unknown",
        description="XMOS xCORE processor",
        expected_metrics={
            "vega_function_accuracy": 0.622,  # From VEGA paper
            "target_function_accuracy": 0.80,
            "target_verified_rate": 0.75,
        }
    )
    xcore.modules = riscv.modules.copy()
    
    return {
        "RISCV": riscv,
        "RI5CY": ri5cy,
        "xCORE": xcore,
    }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Available Processor Backend Benchmarks:")
    print("=" * 60)
    
    benchmarks = get_all_benchmarks()
    
    for name, benchmark in benchmarks.items():
        print(f"\n{name}:")
        print(f"  Family: {benchmark.family.value}")
        print(f"  Triple: {benchmark.triple}")
        print(f"  Description: {benchmark.description}")
        print(f"  Modules: {list(benchmark.modules.keys())}")
        
        total_funcs = sum(len(m.functions) for m in benchmark.modules.values())
        print(f"  Total Functions: {total_funcs}")
        
        if benchmark.expected_metrics:
            print(f"  Expected Metrics:")
            for metric, value in benchmark.expected_metrics.items():
                print(f"    {metric}: {value:.1%}" if value <= 1 else f"    {metric}: {value}")
    
    print("\n" + "=" * 60)
    print("VEGA Paper Benchmarks:")
    
    vega_benchmarks = get_vega_benchmarks()
    for name, benchmark in vega_benchmarks.items():
        vega_acc = benchmark.expected_metrics.get('vega_function_accuracy', 0)
        print(f"  {name}: VEGA accuracy = {vega_acc:.1%}")
