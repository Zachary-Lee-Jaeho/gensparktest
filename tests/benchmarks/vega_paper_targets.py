"""
VEGA Paper Target Benchmarks: RI5CY and xCORE.

This module implements detailed benchmarks for the processor targets
specifically evaluated in the VEGA paper (CGO 2025):

1. RI5CY (PULP RISC-V):
   - ETH Zurich's PULP platform
   - Custom DSP extensions (hw loops, post-increment, SIMD)
   - VEGA accuracy: 73.2% function-level

2. xCORE (XMOS):
   - Unique event-driven architecture
   - Hardware threading
   - VEGA accuracy: 62.2% function-level

These benchmarks test VEGA-Verified's ability to handle:
- Custom instruction set extensions
- Non-standard architectures
- Architecture-specific optimizations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.specification.spec_language import (
    Specification, Condition, ConditionType, Variable, Constant
)
from src.hierarchical.module_verify import Module, ModuleFunction
from src.hierarchical.backend_verify import Backend
from src.hierarchical.interface_contract import (
    InterfaceContract, ContractType, Assumption, Guarantee
)

from .processor_backends import ProcessorFamily, ProcessorBackendBenchmark


# =============================================================================
# RI5CY (PULP RISC-V) Backend
# =============================================================================

def create_ri5cy_benchmark() -> ProcessorBackendBenchmark:
    """
    Create RI5CY (PULP RISC-V) backend benchmark.
    
    RI5CY is ETH Zurich's RISC-V implementation for the PULP (Parallel Ultra Low Power)
    platform. It extends standard RISC-V with custom DSP-oriented instructions:
    
    - Hardware loops (lp.setup, lp.starti, etc.)
    - Post-increment load/store
    - SIMD operations (pv.add, pv.sub, etc.)
    - Bit manipulation (p.extract, p.insert, etc.)
    
    From VEGA Paper:
    - Function-level accuracy: 73.2%
    - Statement-level accuracy: 54.1%
    """
    benchmark = ProcessorBackendBenchmark(
        name="RI5CY",
        family=ProcessorFamily.RISCV,
        triple="riscv32-pulp-linux-gnu",
        description="RI5CY (PULP RISC-V with DSP extensions)",
        expected_metrics={
            "vega_function_accuracy": 0.732,
            "vega_statement_accuracy": 0.541,
            "target_function_accuracy": 0.88,
            "target_verified_rate": 0.85,
            "repair_success_rate": 0.90,
        }
    )
    
    # ================== MCCodeEmitter Module ==================
    mc_emitter = Module(name="MCCodeEmitter")
    
    # Main instruction encoding (handles PULP extensions)
    mc_emitter.add_function(ModuleFunction(
        name="encodeInstruction",
        code="""
void RI5CYMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                            SmallVectorImpl<char> &CB,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
    const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
    unsigned Size = Desc.getSize();
    
    uint64_t TSFlags = Desc.TSFlags;
    
    // Check for PULP custom instructions
    if (TSFlags & RI5CYII::IsPULP) {
        // PULP instructions use custom encoding format
        uint32_t Bits = getPULPBinaryCode(MI, Fixups, STI);
        
        if (Size == 2) {
            // Compressed PULP instruction
            support::endian::write<uint16_t>(CB, Bits, support::little);
        } else {
            support::endian::write<uint32_t>(CB, Bits, support::little);
        }
    } else {
        // Standard RISC-V encoding
        uint32_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
        
        if (Size == 2) {
            support::endian::write<uint16_t>(CB, Bits, support::little);
        } else {
            support::endian::write<uint32_t>(CB, Bits, support::little);
        }
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
                # PULP instructions map to PULP encoder
                Condition(ConditionType.IMPLIES,
                    Condition(ConditionType.EQUALITY, Variable("is_pulp_instr"), Constant(True)),
                    Condition(ConditionType.EQUALITY, Variable("uses_pulp_encoder"), Constant(True))
                ),
            ]
        ),
        is_interface=True
    ))
    
    # PULP-specific binary encoding
    mc_emitter.add_function(ModuleFunction(
        name="getPULPBinaryCode",
        code="""
uint32_t RI5CYMCCodeEmitter::getPULPBinaryCode(const MCInst &MI,
                                                SmallVectorImpl<MCFixup> &Fixups,
                                                const MCSubtargetInfo &STI) const {
    unsigned Opcode = MI.getOpcode();
    uint32_t Binary = 0;
    
    // PULP custom opcode encoding (uses custom-0/1/2/3 space)
    switch (Opcode) {
    // Hardware loop instructions
    case RI5CY::LP_SETUPI:
        Binary = encodeLpSetupImm(MI);
        break;
    case RI5CY::LP_SETUP:
        Binary = encodeLpSetup(MI);
        break;
    case RI5CY::LP_STARTI:
    case RI5CY::LP_ENDI:
    case RI5CY::LP_COUNTI:
        Binary = encodeLpImm(MI);
        break;
    
    // Post-increment load/store
    case RI5CY::LW_POSTINC:
    case RI5CY::LH_POSTINC:
    case RI5CY::LB_POSTINC:
        Binary = encodePostIncLoad(MI);
        break;
    case RI5CY::SW_POSTINC:
    case RI5CY::SH_POSTINC:
    case RI5CY::SB_POSTINC:
        Binary = encodePostIncStore(MI);
        break;
    
    // SIMD instructions
    case RI5CY::PV_ADD_H:
    case RI5CY::PV_ADD_B:
        Binary = encodePVAdd(MI);
        break;
    case RI5CY::PV_SUB_H:
    case RI5CY::PV_SUB_B:
        Binary = encodePVSub(MI);
        break;
    case RI5CY::PV_DOTSP_H:
    case RI5CY::PV_DOTSP_B:
        Binary = encodePVDotProduct(MI);
        break;
    
    // Bit manipulation
    case RI5CY::P_EXTRACT:
    case RI5CY::P_EXTRACTU:
        Binary = encodePExtract(MI);
        break;
    case RI5CY::P_INSERT:
        Binary = encodePInsert(MI);
        break;
    
    default:
        // Fall back to standard encoding
        Binary = getBinaryCodeForInstr(MI, Fixups, STI);
        break;
    }
    
    return Binary;
}
        """,
        specification=Specification(
            function_name="getPULPBinaryCode",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MI")),
            ],
            postconditions=[
                Condition(ConditionType.GREATER_EQUAL, Variable("result"), Constant(0)),
            ],
            invariants=[
                # Hardware loop instructions use custom-0 opcode space
                Condition(ConditionType.IMPLIES,
                    Condition(ConditionType.EQUALITY, Variable("is_hwloop"), Constant(True)),
                    Condition(ConditionType.EQUALITY, 
                        Condition(ConditionType.AND, Variable("result"), Constant(0x7F)),
                        Constant(0x0B))  # custom-0 opcode
                ),
            ]
        ),
        is_interface=True
    ))
    
    # Hardware loop encoding
    mc_emitter.add_function(ModuleFunction(
        name="encodeLpSetupImm",
        code="""
uint32_t RI5CYMCCodeEmitter::encodeLpSetupImm(const MCInst &MI) const {
    // lp.setupi: Setup hardware loop with immediate count
    // Format: |funct7|rs2|rs1|funct3|rd|opcode|
    //         |imm[6:0]|count[4:0]|rs1|100|loop_idx|0001011|
    
    uint32_t Binary = 0;
    Binary |= 0x0B;  // custom-0 opcode
    
    // rd = loop index (0 or 1)
    unsigned LoopIdx = MI.getOperand(0).getImm();
    Binary |= (LoopIdx & 0x1F) << 7;
    
    // funct3 = 100
    Binary |= 0x4 << 12;
    
    // rs1 = start address register
    unsigned Rs1 = Ctx.getRegisterInfo()->getEncodingValue(MI.getOperand(1).getReg());
    Binary |= (Rs1 & 0x1F) << 15;
    
    // rs2 = loop count (immediate)
    unsigned Count = MI.getOperand(2).getImm();
    Binary |= (Count & 0x1F) << 20;
    
    // imm[6:0] = end offset (immediate)
    unsigned EndOff = MI.getOperand(3).getImm();
    Binary |= (EndOff & 0x7F) << 25;
    
    return Binary;
}
        """,
        specification=Specification(
            function_name="encodeLpSetupImm",
            postconditions=[
                # Result has custom-0 opcode (0x0B) in bits [6:0]
                Condition(ConditionType.EQUALITY,
                    Condition(ConditionType.AND, Variable("result"), Constant(0x7F)),
                    Constant(0x0B)
                ),
            ]
        ),
        is_interface=False
    ))
    
    # Post-increment load encoding
    mc_emitter.add_function(ModuleFunction(
        name="encodePostIncLoad",
        code="""
uint32_t RI5CYMCCodeEmitter::encodePostIncLoad(const MCInst &MI) const {
    // Post-increment load: lw rd, imm(rs1!)
    // Updates rs1 after load: rs1 = rs1 + imm
    
    uint32_t Binary = 0;
    unsigned Opcode = MI.getOpcode();
    
    // Determine funct3 based on width
    unsigned Funct3;
    switch (Opcode) {
    case RI5CY::LW_POSTINC: Funct3 = 0x2; break;  // word
    case RI5CY::LH_POSTINC: Funct3 = 0x1; break;  // halfword
    case RI5CY::LB_POSTINC: Funct3 = 0x0; break;  // byte
    default: llvm_unreachable("Invalid post-inc load");
    }
    
    Binary |= 0x0B;  // custom-0 opcode
    
    // rd = destination register
    unsigned Rd = Ctx.getRegisterInfo()->getEncodingValue(MI.getOperand(0).getReg());
    Binary |= (Rd & 0x1F) << 7;
    
    Binary |= (Funct3 & 0x7) << 12;
    
    // rs1 = base register (also modified)
    unsigned Rs1 = Ctx.getRegisterInfo()->getEncodingValue(MI.getOperand(1).getReg());
    Binary |= (Rs1 & 0x1F) << 15;
    
    // imm = increment value
    int32_t Imm = MI.getOperand(2).getImm();
    Binary |= ((Imm & 0xFFF) << 20);
    
    return Binary;
}
        """,
        specification=Specification(
            function_name="encodePostIncLoad",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MI")),
            ]
        ),
        is_interface=False
    ))
    
    # SIMD packed vector add
    mc_emitter.add_function(ModuleFunction(
        name="encodePVAdd",
        code="""
uint32_t RI5CYMCCodeEmitter::encodePVAdd(const MCInst &MI) const {
    // Packed vector add: pv.add.h rd, rs1, rs2
    // Performs parallel addition on packed halfwords/bytes
    
    uint32_t Binary = 0;
    unsigned Opcode = MI.getOpcode();
    
    Binary |= 0x5B;  // custom-1 opcode for SIMD
    
    // rd
    unsigned Rd = Ctx.getRegisterInfo()->getEncodingValue(MI.getOperand(0).getReg());
    Binary |= (Rd & 0x1F) << 7;
    
    // funct3: 000 for halfword, 001 for byte
    unsigned Funct3 = (Opcode == RI5CY::PV_ADD_H) ? 0x0 : 0x1;
    Binary |= (Funct3 & 0x7) << 12;
    
    // rs1
    unsigned Rs1 = Ctx.getRegisterInfo()->getEncodingValue(MI.getOperand(1).getReg());
    Binary |= (Rs1 & 0x1F) << 15;
    
    // rs2
    unsigned Rs2 = Ctx.getRegisterInfo()->getEncodingValue(MI.getOperand(2).getReg());
    Binary |= (Rs2 & 0x1F) << 20;
    
    // funct7: operation type
    Binary |= 0x00 << 25;  // add operation
    
    return Binary;
}
        """,
        specification=Specification(
            function_name="encodePVAdd",
            postconditions=[
                # Result has custom-1 opcode (0x5B) in bits [6:0]
                Condition(ConditionType.EQUALITY,
                    Condition(ConditionType.AND, Variable("result"), Constant(0x7F)),
                    Constant(0x5B)
                ),
            ]
        ),
        is_interface=False
    ))
    
    # Interface contract for MCCodeEmitter
    mc_contract = InterfaceContract(
        name="RI5CY_MCCodeEmitter_IFC",
        module_name="MCCodeEmitter",
        contract_type=ContractType.MODULE
    )
    mc_contract.add_assumption(Assumption(
        "valid_ri5cy_instruction",
        "Input is a valid RI5CY instruction (standard RISC-V or PULP extension)",
        "(or (member opcode RISCV_OPCODES) (member opcode PULP_OPCODES))"
    ))
    mc_contract.add_assumption(Assumption(
        "valid_pulp_subtarget",
        "Subtarget info indicates PULP feature support",
        "(implies (is_pulp_instr MI) (has_feature STI PULP))"
    ))
    mc_contract.add_guarantee(Guarantee(
        "correct_encoding",
        "Encoded bytes correctly represent the instruction",
        "(= (decode (encode MI)) MI)"
    ))
    mc_contract.add_guarantee(Guarantee(
        "valid_opcode_space",
        "PULP instructions use custom opcode space",
        "(implies (is_pulp_instr MI) (in_custom_space (encode MI)))"
    ))
    mc_emitter.interface_contract = mc_contract
    
    benchmark.modules["MCCodeEmitter"] = mc_emitter
    
    # ================== ELFObjectWriter Module ==================
    elf_writer = Module(name="ELFObjectWriter")
    
    elf_writer.add_function(ModuleFunction(
        name="getRelocType",
        code="""
unsigned RI5CYELFObjectWriter::getRelocType(MCContext &Ctx,
                                             const MCValue &Target,
                                             const MCFixup &Fixup,
                                             bool IsPCRel) const {
    unsigned Kind = Fixup.getTargetKind();
    
    // Handle PULP-specific fixups
    switch (Kind) {
    // PULP hardware loop relocations
    case RI5CY::fixup_ri5cy_loop_start:
        return ELF::R_RISCV_PULP_LOOP_START;
    case RI5CY::fixup_ri5cy_loop_end:
        return ELF::R_RISCV_PULP_LOOP_END;
    
    // Standard RISC-V relocations
    default:
        return RISCVELFObjectWriter::getRelocType(Ctx, Target, Fixup, IsPCRel);
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
            ]
        ),
        is_interface=True
    ))
    
    benchmark.modules["ELFObjectWriter"] = elf_writer
    
    # ================== ISelDAGToDAG Module ==================
    isel = Module(name="ISelDAGToDAG")
    
    isel.add_function(ModuleFunction(
        name="SelectHWLoop",
        code="""
bool RI5CYDAGToDAGISel::SelectHWLoop(SDNode *N) {
    // Select hardware loop pattern
    // Pattern: for (i = 0; i < N; i++) { body }
    // Becomes: lp.setup loop_idx, count, body_end
    
    SDLoc DL(N);
    EVT VT = N->getValueType(0);
    
    // Extract loop bounds
    SDValue Start = N->getOperand(0);
    SDValue End = N->getOperand(1);
    SDValue Step = N->getOperand(2);
    
    // Check if this is a countable loop
    if (!isCountableLoop(Start, End, Step))
        return false;
    
    // Calculate iteration count
    SDValue Count = calculateLoopCount(Start, End, Step);
    
    // Get loop body size for end offset
    unsigned BodySize = getLoopBodySize(N);
    
    // Allocate hardware loop
    unsigned LoopIdx = allocateHWLoop();
    if (LoopIdx > 1)
        return false;  // Only 2 hardware loops available
    
    // Generate lp.setup instruction
    SDValue Setup = CurDAG->getMachineNode(
        RI5CY::LP_SETUP, DL, VT,
        CurDAG->getTargetConstant(LoopIdx, DL, MVT::i32),
        Count,
        CurDAG->getTargetConstant(BodySize, DL, MVT::i32)
    );
    
    ReplaceNode(N, Setup.getNode());
    return true;
}
        """,
        specification=Specification(
            function_name="SelectHWLoop",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("N")),
            ],
            invariants=[
                # Only 2 hardware loops available
                Condition(ConditionType.LESS_EQUAL, Variable("loop_idx"), Constant(1)),
            ]
        ),
        is_interface=True
    ))
    
    isel.add_function(ModuleFunction(
        name="SelectPostIncLoad",
        code="""
bool RI5CYDAGToDAGISel::SelectPostIncLoad(SDNode *N) {
    // Select post-increment load pattern
    // Pattern: *p++  or  *(p += offset)
    
    LoadSDNode *LD = cast<LoadSDNode>(N);
    SDValue Base = LD->getBasePtr();
    SDValue Offset = LD->getOffset();
    
    // Check if base is updated after load
    if (!isPostIncrementPattern(Base, N))
        return false;
    
    SDLoc DL(N);
    EVT MemVT = LD->getMemoryVT();
    
    unsigned Opc;
    if (MemVT == MVT::i32)
        Opc = RI5CY::LW_POSTINC;
    else if (MemVT == MVT::i16)
        Opc = RI5CY::LH_POSTINC;
    else if (MemVT == MVT::i8)
        Opc = RI5CY::LB_POSTINC;
    else
        return false;
    
    // Generate post-increment load
    SDValue Ops[] = {Base, Offset, LD->getChain()};
    SDNode *Result = CurDAG->getMachineNode(Opc, DL, 
        LD->getValueType(0), Base.getValueType(), MVT::Other, Ops);
    
    ReplaceUses(SDValue(N, 0), SDValue(Result, 0));  // Load result
    ReplaceUses(SDValue(N, 1), SDValue(Result, 1));  // Updated base
    ReplaceUses(SDValue(N, 2), SDValue(Result, 2));  // Chain
    
    return true;
}
        """,
        specification=Specification(function_name="SelectPostIncLoad"),
        is_interface=True
    ))
    
    benchmark.modules["ISelDAGToDAG"] = isel
    
    # ================== AsmPrinter Module ==================
    asm_printer = Module(name="AsmPrinter")
    
    asm_printer.add_function(ModuleFunction(
        name="emitInstruction",
        code="""
void RI5CYAsmPrinter::emitInstruction(const MachineInstr *MI) {
    MCInst TmpInst;
    
    // Handle PULP pseudo-instructions
    unsigned Opcode = MI->getOpcode();
    
    switch (Opcode) {
    case RI5CY::PseudoLP_SETUP:
        // Expand pseudo to actual hardware loop setup
        emitHWLoopSetup(MI);
        return;
    
    case RI5CY::PseudoPV_DOTPRODUCT:
        // Expand to sequence of dot product operations
        emitSIMDDotProduct(MI);
        return;
    
    default:
        // Standard lowering
        if (!lowerRI5CYMachineInstrToMCInst(MI, TmpInst, *this))
            EmitToStreamer(*OutStreamer, TmpInst);
    }
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
# xCORE Backend
# =============================================================================

def create_xcore_benchmark() -> ProcessorBackendBenchmark:
    """
    Create xCORE backend benchmark.
    
    xCORE is XMOS's event-driven processor architecture with unique features:
    - Hardware multi-threading (8 threads per tile)
    - Channels for inter-thread communication
    - Event-driven execution model
    - Resource-based I/O
    
    From VEGA Paper:
    - Function-level accuracy: 62.2%
    - Statement-level accuracy: 46.3%
    
    xCORE has the lowest accuracy in VEGA due to its unique architecture
    that differs significantly from typical RISC processors.
    """
    benchmark = ProcessorBackendBenchmark(
        name="xCORE",
        family=ProcessorFamily.RISCV,  # Different family but using RISCV for categorization
        triple="xcore-unknown-unknown",
        description="XMOS xCORE event-driven processor",
        expected_metrics={
            "vega_function_accuracy": 0.622,
            "vega_statement_accuracy": 0.463,
            "target_function_accuracy": 0.82,
            "target_verified_rate": 0.78,
            "repair_success_rate": 0.85,
        }
    )
    
    # ================== MCCodeEmitter Module ==================
    mc_emitter = Module(name="MCCodeEmitter")
    
    mc_emitter.add_function(ModuleFunction(
        name="encodeInstruction",
        code="""
void XCoreMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                            SmallVectorImpl<char> &CB,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
    const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
    
    // xCORE uses 16-bit and 32-bit instruction formats
    unsigned Format = (Desc.TSFlags >> XCoreII::FormatShift) & XCoreII::FormatMask;
    
    switch (Format) {
    case XCoreII::Fmt2R:
    case XCoreII::Fmt2R_ONLY:
        // 16-bit two-register format
        emitShortInstr(MI, CB, Fixups, STI);
        break;
    
    case XCoreII::FmtL2R:
    case XCoreII::FmtL3R:
    case XCoreII::FmtL4R:
    case XCoreII::FmtL5R:
    case XCoreII::FmtL6R:
        // 32-bit long format
        emitLongInstr(MI, CB, Fixups, STI);
        break;
    
    case XCoreII::FmtBranch:
        // Branch instructions (variable length)
        emitBranchInstr(MI, CB, Fixups, STI);
        break;
    
    default:
        llvm_unreachable("Unknown xCORE instruction format");
    }
}
        """,
        specification=Specification(
            function_name="encodeInstruction",
            module="MCCodeEmitter",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MI")),
            ],
            postconditions=[
                # xCORE instructions are 16-bit or 32-bit
                Condition(ConditionType.OR,
                    Condition(ConditionType.EQUALITY, Variable("encoded_size"), Constant(2)),
                    Condition(ConditionType.EQUALITY, Variable("encoded_size"), Constant(4))
                ),
            ]
        ),
        is_interface=True
    ))
    
    mc_emitter.add_function(ModuleFunction(
        name="emitShortInstr",
        code="""
void XCoreMCCodeEmitter::emitShortInstr(const MCInst &MI,
                                         SmallVectorImpl<char> &CB,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
    // 16-bit xCORE instruction format:
    // |op[5:0]|op1[3:0]|op2[3:0]|op3[1:0]|
    
    uint16_t Binary = 0;
    unsigned Opcode = MI.getOpcode();
    
    // Get short opcode encoding
    Binary |= getShortOpcodeEncoding(Opcode);
    
    // Encode operands
    unsigned NumOps = MI.getNumOperands();
    for (unsigned i = 0; i < NumOps; ++i) {
        const MCOperand &MO = MI.getOperand(i);
        
        if (MO.isReg()) {
            unsigned Reg = Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());
            // xCORE uses 4-bit register encoding (r0-r11 + special)
            Binary |= (Reg & 0xF) << getOperandShift(i);
        } else if (MO.isImm()) {
            Binary |= (MO.getImm() & getOperandMask(i)) << getOperandShift(i);
        }
    }
    
    // xCORE is little-endian
    support::endian::write<uint16_t>(CB, Binary, support::little);
}
        """,
        specification=Specification(
            function_name="emitShortInstr",
            postconditions=[
                Condition(ConditionType.EQUALITY, Variable("bytes_written"), Constant(2)),
            ]
        ),
        is_interface=False
    ))
    
    mc_emitter.add_function(ModuleFunction(
        name="emitLongInstr",
        code="""
void XCoreMCCodeEmitter::emitLongInstr(const MCInst &MI,
                                        SmallVectorImpl<char> &CB,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
    // 32-bit xCORE instruction format:
    // |prefix[15:0]|op[5:0]|operands[9:0]|
    
    uint32_t Binary = 0;
    unsigned Opcode = MI.getOpcode();
    
    // Get long opcode prefix and encoding
    Binary |= getLongOpcodePrefix(Opcode) << 16;
    Binary |= getLongOpcodeEncoding(Opcode);
    
    // Encode operands (up to 6 registers in long format)
    unsigned NumOps = MI.getNumOperands();
    for (unsigned i = 0; i < NumOps && i < 6; ++i) {
        const MCOperand &MO = MI.getOperand(i);
        
        if (MO.isReg()) {
            unsigned Reg = Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());
            Binary |= encodeRegOperand(Reg, i, Opcode);
        } else if (MO.isImm()) {
            Binary |= encodeImmOperand(MO.getImm(), i, Opcode);
        } else if (MO.isExpr()) {
            // Create fixup for expressions
            Fixups.push_back(MCFixup::create(0, MO.getExpr(),
                MCFixupKind(XCore::fixup_xcore_32)));
        }
    }
    
    support::endian::write<uint32_t>(CB, Binary, support::little);
}
        """,
        specification=Specification(
            function_name="emitLongInstr",
            postconditions=[
                Condition(ConditionType.EQUALITY, Variable("bytes_written"), Constant(4)),
            ]
        ),
        is_interface=False
    ))
    
    # Channel operations (unique to xCORE)
    mc_emitter.add_function(ModuleFunction(
        name="encodeChannelOp",
        code="""
uint32_t XCoreMCCodeEmitter::encodeChannelOp(const MCInst &MI) const {
    // xCORE channel operations:
    // - CHKCT: Check channel control token
    // - OUTCT: Output control token
    // - OUT: Output data on channel
    // - IN: Input data from channel
    // - SETD: Set channel destination
    
    unsigned Opcode = MI.getOpcode();
    uint32_t Binary = 0;
    
    switch (Opcode) {
    case XCore::CHKCT_RR:
        Binary = 0x8F00;  // CHKCT short format
        break;
    case XCore::OUTCT_RR:
        Binary = 0x8E00;  // OUTCT short format
        break;
    case XCore::OUT_RR:
        Binary = 0xA000;  // OUT short format
        break;
    case XCore::IN_RR:
        Binary = 0xB000;  // IN short format
        break;
    case XCore::SETD_RR:
        Binary = 0x8D00;  // SETD short format
        break;
    default:
        llvm_unreachable("Unknown channel operation");
    }
    
    // Encode channel resource register
    unsigned ChanReg = Ctx.getRegisterInfo()->getEncodingValue(MI.getOperand(0).getReg());
    Binary |= (ChanReg & 0xF) << 4;
    
    // Encode data/control operand
    if (MI.getNumOperands() > 1) {
        const MCOperand &MO = MI.getOperand(1);
        if (MO.isReg()) {
            unsigned Reg = Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());
            Binary |= (Reg & 0xF);
        } else if (MO.isImm()) {
            Binary |= (MO.getImm() & 0xF);
        }
    }
    
    return Binary;
}
        """,
        specification=Specification(
            function_name="encodeChannelOp",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("MI")),
            ],
            postconditions=[
                Condition(ConditionType.GREATER_EQUAL, Variable("result"), Constant(0)),
            ]
        ),
        is_interface=True
    ))
    
    # Interface contract
    mc_contract = InterfaceContract(
        name="XCore_MCCodeEmitter_IFC",
        module_name="MCCodeEmitter",
        contract_type=ContractType.MODULE
    )
    mc_contract.add_assumption(Assumption(
        "valid_xcore_instruction",
        "Input is a valid xCORE instruction",
        "(member opcode XCORE_OPCODES)"
    ))
    mc_contract.add_guarantee(Guarantee(
        "correct_format",
        "Instruction uses correct format (short/long)",
        "(implies (is_short_format MI) (= encoded_size 2))"
    ))
    mc_contract.add_guarantee(Guarantee(
        "channel_semantics",
        "Channel operations preserve channel semantics",
        "(implies (is_channel_op MI) (valid_channel_encoding result))"
    ))
    mc_emitter.interface_contract = mc_contract
    
    benchmark.modules["MCCodeEmitter"] = mc_emitter
    
    # ================== ELFObjectWriter Module ==================
    elf_writer = Module(name="ELFObjectWriter")
    
    elf_writer.add_function(ModuleFunction(
        name="getRelocType",
        code="""
unsigned XCoreELFObjectWriter::getRelocType(MCContext &Ctx,
                                             const MCValue &Target,
                                             const MCFixup &Fixup,
                                             bool IsPCRel) const {
    unsigned Kind = Fixup.getTargetKind();
    
    // xCORE-specific relocations
    switch (Kind) {
    case FK_NONE:
        return ELF::R_XCORE_NONE;
    
    case FK_Data_4:
        return ELF::R_XCORE_32;
    
    case XCore::fixup_xcore_32:
        return IsPCRel ? ELF::R_XCORE_32_PCREL : ELF::R_XCORE_32;
    
    case XCore::fixup_xcore_dp_rel_10:
        return ELF::R_XCORE_DPREL_10;
    
    case XCore::fixup_xcore_cp_rel_10:
        return ELF::R_XCORE_CPREL_10;
    
    case XCore::fixup_xcore_pcrel_10:
        return ELF::R_XCORE_PCREL_10;
    
    case XCore::fixup_xcore_pcrel_20:
        return ELF::R_XCORE_PCREL_20;
    
    default:
        Ctx.reportError(Fixup.getLoc(), "unsupported xCORE relocation type");
        return ELF::R_XCORE_NONE;
    }
}
        """,
        specification=Specification(
            function_name="getRelocType",
            preconditions=[
                Condition(ConditionType.IS_VALID, Variable("Fixup")),
            ],
            invariants=[
                # FK_NONE always maps to R_XCORE_NONE
                Condition(ConditionType.IMPLIES,
                    Condition(ConditionType.EQUALITY, Variable("Kind"), Constant("FK_NONE")),
                    Condition(ConditionType.EQUALITY, Variable("result"), Constant("R_XCORE_NONE"))
                ),
            ]
        ),
        is_interface=True
    ))
    
    benchmark.modules["ELFObjectWriter"] = elf_writer
    
    # ================== ISelDAGToDAG Module ==================
    isel = Module(name="ISelDAGToDAG")
    
    isel.add_function(ModuleFunction(
        name="SelectChannelOp",
        code="""
bool XCoreDAGToDAGISel::SelectChannelOp(SDNode *N) {
    // Select xCORE channel operations
    // Pattern: channel.out(ch, data) or channel.in(ch)
    
    SDLoc DL(N);
    EVT VT = N->getValueType(0);
    
    unsigned ISD_Op = N->getOpcode();
    unsigned Opc;
    
    switch (ISD_Op) {
    case XCoreISD::CHAN_OUT:
        Opc = XCore::OUT_RR;
        break;
    case XCoreISD::CHAN_IN:
        Opc = XCore::IN_RR;
        break;
    case XCoreISD::CHAN_OUTCT:
        Opc = XCore::OUTCT_RR;
        break;
    case XCoreISD::CHAN_CHKCT:
        Opc = XCore::CHKCT_RR;
        break;
    default:
        return false;
    }
    
    // Get channel resource operand
    SDValue ChanRes = N->getOperand(1);
    
    // Get data operand (if applicable)
    SDValue Data;
    if (ISD_Op == XCoreISD::CHAN_OUT || ISD_Op == XCoreISD::CHAN_OUTCT) {
        Data = N->getOperand(2);
    }
    
    SDValue Ops[] = {ChanRes, Data, N->getOperand(0)};  // Chain
    SDNode *Result = CurDAG->getMachineNode(Opc, DL, VT, MVT::Other, Ops);
    
    ReplaceNode(N, Result);
    return true;
}
        """,
        specification=Specification(function_name="SelectChannelOp"),
        is_interface=True
    ))
    
    isel.add_function(ModuleFunction(
        name="SelectThreadOp",
        code="""
bool XCoreDAGToDAGISel::SelectThreadOp(SDNode *N) {
    // Select xCORE thread operations
    // Pattern: thread.start(tid) or thread.sync()
    
    SDLoc DL(N);
    unsigned ISD_Op = N->getOpcode();
    unsigned Opc;
    
    switch (ISD_Op) {
    case XCoreISD::THREAD_START:
        Opc = XCore::SSYNC;  // Start synchronization
        break;
    case XCoreISD::THREAD_JOIN:
        Opc = XCore::MJOIN;  // Master join
        break;
    case XCoreISD::THREAD_FREE:
        Opc = XCore::FREET;  // Free thread
        break;
    default:
        return false;
    }
    
    SDValue ThreadId = N->getOperand(1);
    SDValue Chain = N->getOperand(0);
    
    SDNode *Result = CurDAG->getMachineNode(Opc, DL, MVT::Other, ThreadId, Chain);
    
    ReplaceNode(N, Result);
    return true;
}
        """,
        specification=Specification(function_name="SelectThreadOp"),
        is_interface=True
    ))
    
    benchmark.modules["ISelDAGToDAG"] = isel
    
    # ================== AsmPrinter Module ==================
    asm_printer = Module(name="AsmPrinter")
    
    asm_printer.add_function(ModuleFunction(
        name="emitInstruction",
        code="""
void XCoreAsmPrinter::emitInstruction(const MachineInstr *MI) {
    MCInst TmpInst;
    unsigned Opcode = MI->getOpcode();
    
    // Handle xCORE-specific pseudo instructions
    switch (Opcode) {
    case XCore::PseudoRET:
        // Expand to: retsp 0 or ldw r0, sp[0]; retsp n
        emitReturnSequence(MI);
        return;
    
    case XCore::PseudoCALL:
        // Expand to: bl target (may need multiple instructions for far calls)
        emitCallSequence(MI);
        return;
    
    case XCore::PseudoTAILCALL:
        // Expand to: bu target
        emitTailCallSequence(MI);
        return;
    
    case XCore::PseudoRES:
        // Resource acquisition pseudo
        emitResourceAcquire(MI);
        return;
    
    default:
        // Standard lowering
        if (!lowerXCoreMachineInstrToMCInst(MI, TmpInst, *this))
            EmitToStreamer(*OutStreamer, TmpInst);
    }
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
# Combined VEGA Paper Targets
# =============================================================================

def get_vega_paper_targets() -> Dict[str, ProcessorBackendBenchmark]:
    """
    Get all processor targets evaluated in the VEGA paper.
    
    Returns a dictionary containing:
    - RISC-V: Standard RISC-V backend (from processor_backends.py)
    - RI5CY: PULP RISC-V with DSP extensions
    - xCORE: XMOS event-driven processor
    
    These represent the actual benchmarks used in the VEGA paper
    for evaluating neural compiler backend generation.
    """
    from .processor_backends import create_riscv_benchmark
    
    return {
        "RISCV": create_riscv_benchmark(),
        "RI5CY": create_ri5cy_benchmark(),
        "xCORE": create_xcore_benchmark(),
    }


def get_vega_paper_metrics() -> Dict[str, Dict[str, float]]:
    """
    Get the metrics reported in the VEGA paper for comparison.
    
    These are the actual numbers from the VEGA paper (CGO 2025).
    """
    return {
        "RISCV": {
            "vega_function_accuracy": 0.715,
            "vega_statement_accuracy": 0.550,
            "fork_flow_accuracy": 0.079,
        },
        "RI5CY": {
            "vega_function_accuracy": 0.732,
            "vega_statement_accuracy": 0.541,
            "fork_flow_accuracy": 0.085,
        },
        "xCORE": {
            "vega_function_accuracy": 0.622,
            "vega_statement_accuracy": 0.463,
            "fork_flow_accuracy": 0.062,
        },
    }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("VEGA Paper Target Benchmarks")
    print("=" * 60)
    
    targets = get_vega_paper_targets()
    metrics = get_vega_paper_metrics()
    
    for name, benchmark in targets.items():
        print(f"\n{name}:")
        print(f"  Triple: {benchmark.triple}")
        print(f"  Description: {benchmark.description}")
        print(f"  Modules: {list(benchmark.modules.keys())}")
        
        total_funcs = sum(len(m.functions) for m in benchmark.modules.values())
        print(f"  Total Functions: {total_funcs}")
        
        # Show VEGA paper metrics
        if name in metrics:
            print(f"  VEGA Paper Metrics:")
            for metric, value in metrics[name].items():
                print(f"    {metric}: {value:.1%}")
        
        # Show expected VEGA-Verified metrics
        print(f"  Expected VEGA-Verified Metrics:")
        for metric, value in benchmark.expected_metrics.items():
            print(f"    {metric}: {value:.1%}")
    
    print("\n" + "=" * 60)
    print("Summary: VEGA vs VEGA-Verified (Expected)")
    print("-" * 60)
    print(f"{'Target':<12} {'VEGA Acc':<12} {'VV Target':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for name in targets:
        vega_acc = metrics[name]["vega_function_accuracy"]
        vv_target = targets[name].expected_metrics["target_function_accuracy"]
        improvement = vv_target - vega_acc
        print(f"{name:<12} {vega_acc:>10.1%}   {vv_target:>10.1%}   +{improvement:>9.1%}")
