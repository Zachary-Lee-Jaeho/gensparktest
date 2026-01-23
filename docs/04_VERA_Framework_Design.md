# 04. VERA Framework Design

**Author**: Compiler Backend Autogeneration Research Group  
**Version**: 2.0  
**Date**: 2026-01-23  
**Classification**: Technical Design Document

---

## Executive Summary

**VERA (Verified End-to-end Retargetable Architecture)** 는 ISA 스펙 문서만으로 검증된 컴파일러 백엔드를 자동 생성하는 프레임워크이다. 기존 연구들(VEGA, ACT, Isaria, VADL, VeGen, Hydride)의 장점을 통합하고 한계를 극복하기 위해 설계되었다.

---

## 1. Motivation and Goals

### 1.1 기존 접근법의 핵심 Gap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Current Research Landscape                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  VEGA (CGO 2025)          │  "기존 백엔드 필요, 의미론 이해 없음"       │
│  ─────────────────        │                                             │
│  Input: .td files          │  ────────────────────────────────────────  │
│  Output: Backend code      │                                             │
│  Method: Transformer       │  GAP: 완전히 새로운 ISA에서 시작 불가       │
│                            │                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ACT (arXiv 2025)         │  "텐서 가속기 전용, 범용 CPU 불가"          │
│  ─────────────────        │                                             │
│  Input: Tensor IR ISA      │  ────────────────────────────────────────  │
│  Output: Tensor backend    │                                             │
│  Method: E-sat + CP        │  GAP: 일반 ISA에 적용할 수 없는 형식화     │
│                            │                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Isaria (ASPLOS 2024)     │  "벡터화에 특화, 전체 백엔드 아님"          │
│  ─────────────────        │                                             │
│  Input: Rosette spec       │  ────────────────────────────────────────  │
│  Output: Vectorizer        │                                             │
│  Method: Rule synthesis    │  GAP: 명령어 선택, 레지스터 할당 등 미포함 │
│                            │                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  VADL (arXiv 2024)        │  "명세 작성이 곧 백엔드 구현 수준의 노력"   │
│  ─────────────────        │                                             │
│  Input: VADL spec          │  ────────────────────────────────────────  │
│  Output: Full toolchain    │                                             │
│  Method: Pattern inference │  GAP: 명세가 너무 상세해야 동작            │
│                            │                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  VeGen (ASPLOS 2021)      │  "벡터 명령어에 한정, 직선 코드만"          │
│  ─────────────────        │                                             │
│  Input: Vendor pseudocode  │  ────────────────────────────────────────  │
│  Output: Vectorizer        │                                             │
│  Method: SMT semantics     │  GAP: 제어 흐름, 메모리 연산 처리 부족     │
│                            │                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Hydride (ASPLOS 2022)    │  "Cross-ISA 이식에 특화, 생성이 아님"       │
│  ─────────────────        │                                             │
│  Input: Existing SIMD code │  ────────────────────────────────────────  │
│  Output: Ported code       │                                             │
│  Method: Similarity + IR   │  GAP: 기존 구현 기반 이식, 새 생성 아님    │
│                            │                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 VERA의 목표

```
VERA Goals:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  G1. ISA Spec → Backend                                                │
│      "ISA 명세 문서(PDF/XML)만으로 시작하여 동작하는 백엔드 생성"       │
│                                                                         │
│  G2. Verified Correctness                                               │
│      "생성된 코드가 ISA 의미론을 정확히 구현함을 형식적으로 검증"       │
│                                                                         │
│  G3. Minimal Human Effort                                               │
│      "수동 작업을 최소화하고, 필요한 곳에만 인간 개입 요청"             │
│                                                                         │
│  G4. Incremental Applicability                                          │
│      "부분적 성공도 유용하며, 점진적 개선 가능"                         │
│                                                                         │
│  G5. General ISA Support                                                │
│      "특정 도메인(텐서, DSP)이 아닌 범용 CPU ISA 지원"                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 핵심 통찰: 왜 기존 방법들이 실패하는가?

```
분석:

1. 의미론 Gap (Semantic Gap)
   ───────────────────────────
   - ISA 문서의 자연어 설명과 형식 의미론 사이의 거리
   - 예: "The carry flag is set if..." → BV formula로 자동 변환?
   
   → VERA 해결책: LLM + 형식 검증의 조합

2. 추상화 수준 불일치 (Abstraction Mismatch)
   ─────────────────────────────────────────
   - ISA 레벨 (개별 명령어) vs 백엔드 레벨 (패턴, 규칙)
   - 예: ISA "ADD r1, r2, r3" → SelectionDAG 패턴?
   
   → VERA 해결책: 계층적 변환 + 템플릿 기반 합성

3. 복잡성 폭발 (Complexity Explosion)
   ──────────────────────────────────
   - 전체 백엔드의 검증은 계산적으로 불가능
   - 예: 수천 개 명령어 × 수백 개 패턴 × 다양한 최적화
   
   → VERA 해결책: 모듈화 + 컴포지셔널 검증

4. 피드백 루프 부재 (Missing Feedback Loop)
   ─────────────────────────────────────────
   - 생성 후 검증 실패 시 개선 방법 부재
   - 예: VEGA는 오류 시 수동 수정만 가능
   
   → VERA 해결책: Counterexample-Guided Neural Repair
```

---

## 2. Architecture Overview

### 2.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VERA Architecture                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Layer 0: Input Processing                     │   │
│  │                                                                  │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │   │
│  │  │  PDF     │    │  XML     │    │  HTML    │    │  Manual  │  │   │
│  │  │  Parser  │    │  Parser  │    │  Parser  │    │  Input   │  │   │
│  │  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘  │   │
│  │       └───────────────┴───────────────┴───────────────┘        │   │
│  │                           │                                     │   │
│  │                           v                                     │   │
│  │                  ┌────────────────┐                             │   │
│  │                  │ Raw ISA Corpus │                             │   │
│  │                  └────────────────┘                             │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                          │
│  ┌──────────────────────────v──────────────────────────────────────┐   │
│  │               Layer 1: ISA Understanding                         │   │
│  │                                                                  │   │
│  │  ┌─────────────────┐         ┌─────────────────┐                │   │
│  │  │  LLM Extraction │ ◄─────► │ Formal Modeling │                │   │
│  │  │  (GPT-4/Claude) │         │ (SMT-LIB/Rosette)│               │   │
│  │  └────────┬────────┘         └────────┬────────┘                │   │
│  │           └──────────┬───────────────┘                          │   │
│  │                      v                                          │   │
│  │            ┌──────────────────┐                                 │   │
│  │            │  ISA Formal Model │                                │   │
│  │            │  - Registers      │                                │   │
│  │            │  - Instructions   │                                │   │
│  │            │  - Encodings      │                                │   │
│  │            │  - Semantics      │                                │   │
│  │            └────────┬─────────┘                                 │   │
│  └─────────────────────┬───────────────────────────────────────────┘   │
│                        │                                               │
│  ┌─────────────────────v───────────────────────────────────────────┐   │
│  │            Layer 2: Compositional Synthesis                      │   │
│  │                                                                  │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │              Component Synthesizers                      │    │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │    │   │
│  │  │  │TargetMC │ │Register │ │Instr    │ │ ISelDAG │       │    │   │
│  │  │  │Info     │ │Info     │ │Info     │ │ToDAG    │       │    │   │
│  │  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │    │   │
│  │  │       │           │           │           │             │    │   │
│  │  │  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐       │    │   │
│  │  │  │Frame    │ │AsmParser│ │AsmPrinter│ │MCCode  │       │    │   │
│  │  │  │Lowering │ │         │ │         │ │Emitter │       │    │   │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │                        │                                        │   │
│  │                        v                                        │   │
│  │              ┌──────────────────┐                               │   │
│  │              │  Draft Backend   │                               │   │
│  │              └────────┬─────────┘                               │   │
│  └───────────────────────┬─────────────────────────────────────────┘   │
│                          │                                             │
│  ┌───────────────────────v─────────────────────────────────────────┐   │
│  │           Layer 3: Verification-Guided Refinement                │   │
│  │                                                                  │   │
│  │         ┌──────────────────────────────────────────┐            │   │
│  │         │        Verification Loop                  │            │   │
│  │         │                                          │            │   │
│  │         │  ┌────────┐    ┌────────┐    ┌────────┐ │            │   │
│  │         │  │ VC Gen │ ─► │ SMT    │ ─► │Counter │ │            │   │
│  │         │  │        │    │ Solver │    │Example │ │            │   │
│  │         │  └────────┘    └────────┘    └───┬────┘ │            │   │
│  │         │                                  │      │            │   │
│  │         │                    ┌─────────────┘      │            │   │
│  │         │                    v                    │            │   │
│  │         │         ┌──────────────────┐           │            │   │
│  │         │         │  CGNR (Neural    │           │            │   │
│  │         │         │  Repair Engine)  │           │            │   │
│  │         │         └────────┬─────────┘           │            │   │
│  │         │                  │                      │            │   │
│  │         │                  v                      │            │   │
│  │         │         ┌──────────────────┐           │            │   │
│  │         │         │  Refined Code    │───────────┘            │   │
│  │         │         └──────────────────┘  (iterate)             │   │
│  │         └──────────────────────────────────────────┘            │   │
│  │                                                                  │   │
│  │                        │ Verified                               │   │
│  │                        v                                        │   │
│  │              ┌──────────────────┐                               │   │
│  │              │ Verified Backend │                               │   │
│  │              └──────────────────┘                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 데이터 흐름

```
Data Flow in VERA:

ISA Spec Document (e.g., RISC-V User Manual, 200+ pages)
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  CHUNKING & PARSING                                        │
│  - Document structure analysis                             │
│  - Section identification (Registers, Instructions, etc.) │
│  - Table/figure extraction                                 │
└────────────────────────────────────────────────────────────┘
    │
    ├─► Register Section ──────────────────────┐
    ├─► Instruction Encoding Section ──────────┤
    ├─► Instruction Semantics Section ─────────┼───► ISA Formal Model
    ├─► Exception Handling Section ────────────┤
    └─► ABI/Calling Convention Section ────────┘
                                                      │
                                                      ▼
                              ┌────────────────────────────────┐
                              │  Component Generation Inputs   │
                              │  - RegisterModel               │
                              │  - InstructionModel            │
                              │  - EncodingModel               │
                              │  - SemanticsModel              │
                              │  - ABIModel                    │
                              └────────────────────────────────┘
                                                      │
                    ┌─────────────┬─────────────┬─────┴─────────┐
                    ▼             ▼             ▼               ▼
              RegisterInfo  InstrInfo    ISelDAGToDAG     MCCodeEmitter
                    │             │             │               │
                    └─────────────┴─────────────┴───────────────┘
                                                      │
                                                      ▼
                              ┌────────────────────────────────┐
                              │      Integrated Backend        │
                              │      (Draft Version)           │
                              └────────────────────────────────┘
                                                      │
                                        ┌─────────────┴─────────────┐
                                        │    Verification Suite     │
                                        │    - Unit tests          │
                                        │    - Integration tests   │
                                        │    - Semantic checks     │
                                        └─────────────┬─────────────┘
                                                      │
                                            Pass?─────┼─────Fail?
                                              │               │
                                              ▼               ▼
                                        ┌─────────┐   ┌─────────────┐
                                        │VERIFIED │   │ CGNR Repair │
                                        │ Backend │   │    Loop     │
                                        └─────────┘   └─────────────┘
```

---

## 3. Layer 1: ISA Understanding

### 3.1 대용량 ISA 문서 처리 전략

ISA 명세 문서는 일반적으로 수백~수천 페이지에 달한다. 전체를 한 번에 LLM에 입력하는 것은 불가능하므로, **구조적 분할 처리** 전략을 사용한다.

```python
class ISADocumentProcessor:
    """
    대용량 ISA 명세 문서의 계층적 처리
    """
    
    # 일반적인 ISA 문서 구조
    SECTION_TYPES = {
        'overview': ['introduction', 'overview', 'architecture'],
        'registers': ['register', 'register file', 'general purpose'],
        'instructions': ['instruction set', 'instruction listing', 'opcode'],
        'encodings': ['encoding', 'format', 'binary'],
        'semantics': ['operation', 'description', 'pseudocode'],
        'exceptions': ['exception', 'trap', 'interrupt'],
        'memory': ['memory', 'addressing', 'endian'],
        'abi': ['calling convention', 'abi', 'stack frame']
    }
    
    def __init__(self, llm_client, max_chunk_tokens=8000):
        self.llm = llm_client
        self.max_chunk = max_chunk_tokens
        
    def process_document(self, document_path):
        """
        전체 처리 파이프라인
        
        1. 문서 구조 분석 (목차 기반)
        2. 섹션별 분리
        3. 각 섹션 독립 처리
        4. 크로스 레퍼런스 해결
        5. 통합 모델 생성
        """
        # Step 1: 구조 분석
        structure = self.analyze_structure(document_path)
        
        # Step 2: 섹션 분리
        sections = self.segment_by_sections(document_path, structure)
        
        # Step 3: 섹션별 처리
        processed = {}
        for section_type, content in sections.items():
            if section_type == 'registers':
                processed['registers'] = self.process_register_section(content)
            elif section_type == 'instructions':
                processed['instructions'] = self.process_instruction_section(content)
            elif section_type == 'encodings':
                processed['encodings'] = self.process_encoding_section(content)
            elif section_type == 'semantics':
                processed['semantics'] = self.process_semantics_section(content)
            elif section_type == 'abi':
                processed['abi'] = self.process_abi_section(content)
        
        # Step 4: 크로스 레퍼런스 해결
        self.resolve_cross_references(processed)
        
        # Step 5: 통합 모델 생성
        return self.create_isa_model(processed)
    
    def process_register_section(self, content):
        """
        레지스터 섹션 처리
        
        추출 대상:
        - 레지스터 클래스 (GPR, FPR, VR, SPR)
        - 각 클래스의 레지스터 목록
        - 레지스터 크기 (32-bit, 64-bit, etc.)
        - 특수 레지스터 (PC, SP, FP, LR, etc.)
        - ABI 역할 (caller-saved, callee-saved, etc.)
        """
        prompt = """
        Extract register information from the following ISA documentation section.
        
        For each register class, provide:
        1. Class name and purpose
        2. Register names (e.g., x0-x31 for RISC-V)
        3. Register width in bits
        4. Special roles (stack pointer, frame pointer, etc.)
        5. ABI information (caller/callee saved)
        
        Output in structured JSON format.
        
        Documentation:
        {content}
        """
        
        return self.llm_extract(prompt.format(content=content), 
                               schema=RegisterSchema)
    
    def process_instruction_section(self, content):
        """
        명령어 섹션 처리 (청크 기반)
        
        대형 명령어 테이블은 청크로 분할하여 처리
        """
        chunks = self.chunk_content(content, self.max_chunk)
        
        all_instructions = []
        for chunk in chunks:
            prompt = """
            Extract instruction information from this documentation chunk.
            
            For each instruction, provide:
            1. Mnemonic (e.g., ADD, SUB, LW)
            2. Operand types (reg, imm, mem)
            3. Brief description
            4. Instruction category (arithmetic, load/store, branch, etc.)
            
            Documentation chunk:
            {chunk}
            """
            
            instructions = self.llm_extract(prompt.format(chunk=chunk),
                                           schema=InstructionListSchema)
            all_instructions.extend(instructions)
        
        # 중복 제거 및 병합
        return self.deduplicate_instructions(all_instructions)
```

### 3.2 LLM + Formal Verification 조합

```python
class HybridISAExtractor:
    """
    LLM 추출 + 형식 검증의 조합
    
    핵심 아이디어:
    - LLM은 자연어에서 구조화된 정보 추출에 강함
    - 하지만 정확성 보장이 어려움
    - 형식 검증으로 LLM 출력을 검증/보정
    """
    
    def __init__(self, llm_client, smt_solver):
        self.llm = llm_client
        self.solver = smt_solver
        
    def extract_instruction_semantics(self, instruction_doc):
        """
        명령어 의미론 추출 및 검증
        """
        # Step 1: LLM으로 초기 의미론 추출
        llm_semantics = self.llm_extract_semantics(instruction_doc)
        
        # Step 2: SMT 공식으로 변환
        smt_formula = self.convert_to_smt(llm_semantics)
        
        # Step 3: 기본 일관성 검증
        consistency_checks = [
            self.check_determinism(smt_formula),
            self.check_output_coverage(smt_formula),
            self.check_type_consistency(smt_formula)
        ]
        
        if all(consistency_checks):
            return smt_formula
        else:
            # Step 4: 실패 시 LLM에 피드백 제공하여 재시도
            feedback = self.generate_feedback(consistency_checks)
            return self.retry_with_feedback(instruction_doc, feedback)
    
    def llm_extract_semantics(self, instruction_doc):
        """
        LLM을 사용한 의미론 추출
        """
        prompt = """
        Extract the formal semantics of this instruction.
        
        Express the semantics as:
        1. Preconditions (if any)
        2. State changes (registers, memory, flags)
        3. Postconditions (if any)
        
        Use the following notation:
        - R[n] for register n
        - M[addr] for memory at address
        - PC for program counter
        - SignExt(v, n) for sign extension to n bits
        - ZeroExt(v, n) for zero extension
        
        Instruction documentation:
        {doc}
        
        Output format (JSON):
        {{
            "mnemonic": "...",
            "operands": [...],
            "preconditions": [...],
            "operations": [...],
            "postconditions": [...],
            "flags_affected": [...]
        }}
        """
        
        return self.llm.query(prompt.format(doc=instruction_doc))
    
    def convert_to_smt(self, semantics):
        """
        추출된 의미론을 SMT-LIB 공식으로 변환
        """
        formula = SMTFormula()
        
        # 레지스터 상태 선언
        for reg in semantics.get('registers_used', []):
            formula.declare_bitvector(f"R_{reg}_pre", 64)
            formula.declare_bitvector(f"R_{reg}_post", 64)
        
        # 메모리 상태 선언
        formula.declare_array("M_pre", BitVec(64), BitVec(8))
        formula.declare_array("M_post", BitVec(64), BitVec(8))
        
        # 연산 인코딩
        for op in semantics['operations']:
            formula.add_assertion(self.encode_operation(op))
        
        return formula
    
    def check_determinism(self, formula):
        """
        결정론적 의미론인지 검증
        
        같은 입력 상태에 대해 유일한 출력 상태가 있어야 함
        """
        # ∀ input state, ∃! output state
        formula_copy = formula.copy()
        formula_copy.add_assertion(
            Not(Equals(formula.get_output_state(), 
                      formula_copy.get_output_state()))
        )
        
        # UNSAT이어야 결정론적
        return self.solver.check(formula_copy) == UNSAT
```

### 3.3 ISA Formal Model 구조

```python
@dataclass
class ISAFormalModel:
    """
    추출된 ISA의 형식 모델
    """
    
    # 기본 정보
    name: str
    version: str
    endianness: Literal['little', 'big']
    pointer_size: int  # 32 or 64
    
    # 레지스터 모델
    register_classes: List[RegisterClass]
    special_registers: Dict[str, RegisterInfo]
    
    # 명령어 모델
    instructions: List[InstructionInfo]
    instruction_formats: List[InstructionFormat]
    
    # 인코딩 모델
    encodings: Dict[str, EncodingSpec]
    
    # 의미론 모델
    semantics: Dict[str, SMTFormula]
    
    # ABI 모델
    calling_convention: CallingConvention
    stack_layout: StackLayout
    
    def validate(self):
        """모델 일관성 검증"""
        self._check_register_references()
        self._check_encoding_completeness()
        self._check_semantics_coverage()
        return True
    
    def export_to_tablegen(self):
        """LLVM TableGen 형식으로 내보내기"""
        pass
    
    def export_to_smt(self):
        """SMT-LIB 형식으로 내보내기"""
        pass


@dataclass
class RegisterClass:
    name: str
    registers: List[str]
    width: int
    abi_names: Dict[str, str]  # x0 -> zero, x1 -> ra, etc.
    properties: Dict[str, Any]  # caller_saved, reserved, etc.


@dataclass
class InstructionInfo:
    mnemonic: str
    category: str  # arithmetic, load_store, branch, etc.
    operands: List[OperandInfo]
    format: str  # R-type, I-type, etc.
    encoding_bits: str
    semantics_ref: str  # semantics dict의 키


@dataclass
class EncodingSpec:
    format_name: str
    total_bits: int
    fields: List[EncodingField]
    
    def get_field_positions(self):
        """각 필드의 비트 위치 반환"""
        pass


@dataclass
class CallingConvention:
    argument_registers: List[str]
    return_registers: List[str]
    callee_saved: List[str]
    caller_saved: List[str]
    stack_pointer: str
    frame_pointer: Optional[str]
    link_register: Optional[str]
    stack_alignment: int
```

---

## 4. Layer 2: Compositional Synthesis

### 4.1 컴포넌트 의존성 그래프

```
Component Dependency Graph:

                    ┌─────────────┐
                    │ ISA Model   │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
  ┌────────────┐   ┌────────────┐   ┌────────────┐
  │ TargetInfo │   │RegisterInfo│   │ InstrInfo  │
  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
        │                │                 │
        │                ▼                 │
        │         ┌────────────┐          │
        └────────►│SubtargetInfo│◄─────────┘
                  └─────┬──────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
  ┌────────────┐ ┌────────────┐ ┌────────────┐
  │Frame       │ │ ISelDAG    │ │ AsmParser  │
  │Lowering    │ │ ToDAG      │ │            │
  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
        │              │              │
        │              ▼              │
        │       ┌────────────┐        │
        └──────►│ MCCodeEmit │◄───────┘
                │ ter        │
                └─────┬──────┘
                      │
                      ▼
               ┌────────────┐
               │ AsmPrinter │
               └────────────┘
```

### 4.2 컴포넌트별 합성 전략

```python
class ComponentSynthesizer:
    """
    각 백엔드 컴포넌트의 합성기 베이스 클래스
    """
    
    def __init__(self, isa_model: ISAFormalModel, 
                 template_library: TemplateLibrary,
                 ml_model: Optional[CodeGenerationModel] = None):
        self.isa = isa_model
        self.templates = template_library
        self.ml_model = ml_model
        
    def synthesize(self) -> SynthesisResult:
        """
        컴포넌트 합성의 일반적 흐름
        """
        # 1. ISA 모델에서 관련 정보 추출
        relevant_info = self.extract_relevant_info()
        
        # 2. 템플릿 선택
        template = self.select_template(relevant_info)
        
        # 3. 템플릿 인스턴스화
        draft = self.instantiate_template(template, relevant_info)
        
        # 4. ML 모델로 갭 채우기 (선택적)
        if self.ml_model and draft.has_gaps():
            draft = self.fill_gaps_with_ml(draft)
        
        # 5. 단위 검증
        verification_result = self.verify_component(draft)
        
        return SynthesisResult(
            code=draft,
            verification=verification_result
        )
    
    @abstractmethod
    def extract_relevant_info(self) -> Dict:
        pass
    
    @abstractmethod  
    def select_template(self, info: Dict) -> Template:
        pass


class RegisterInfoSynthesizer(ComponentSynthesizer):
    """
    RegisterInfo 컴포넌트 합성
    """
    
    def extract_relevant_info(self):
        return {
            'register_classes': self.isa.register_classes,
            'special_registers': self.isa.special_registers,
            'calling_convention': self.isa.calling_convention
        }
    
    def select_template(self, info):
        # 레지스터 구조에 따라 적절한 템플릿 선택
        if info['register_classes'][0].width == 64:
            return self.templates.get('RegisterInfo_64bit')
        else:
            return self.templates.get('RegisterInfo_32bit')
    
    def instantiate_template(self, template, info):
        """
        템플릿 인스턴스화
        
        예: RISC-V RegisterInfo
        """
        code = template.clone()
        
        # 레지스터 클래스 정의 생성
        for rc in info['register_classes']:
            code.add_register_class(
                name=rc.name,
                registers=rc.registers,
                width=rc.width,
                alignment=rc.width // 8
            )
        
        # 특수 레지스터 매핑
        for role, reg in info['special_registers'].items():
            code.add_special_register_mapping(role, reg)
        
        # Calling convention 통합
        cc = info['calling_convention']
        code.set_callee_saved(cc.callee_saved)
        code.set_reserved([cc.stack_pointer, cc.frame_pointer])
        
        return code
    
    def verify_component(self, code):
        """
        RegisterInfo 검증
        
        검증 항목:
        1. 모든 레지스터 클래스가 정의됨
        2. 특수 레지스터가 올바르게 매핑됨
        3. Calling convention과 일관성
        """
        checks = []
        
        # Check 1: 레지스터 클래스 완전성
        for rc in self.isa.register_classes:
            defined = code.has_register_class(rc.name)
            checks.append(('register_class_defined', rc.name, defined))
        
        # Check 2: 특수 레지스터
        for role in ['stack_pointer', 'frame_pointer', 'program_counter']:
            if role in self.isa.special_registers:
                correct = code.get_special_register(role) == self.isa.special_registers[role]
                checks.append(('special_register', role, correct))
        
        return VerificationResult(checks)


class ISelDAGToDAGSynthesizer(ComponentSynthesizer):
    """
    Instruction Selection 패턴 합성
    
    이 컴포넌트가 가장 복잡하며, ML 지원이 가장 유용함
    """
    
    def extract_relevant_info(self):
        return {
            'instructions': self.isa.instructions,
            'semantics': self.isa.semantics,
            'encodings': self.isa.encodings
        }
    
    def synthesize_patterns(self):
        """
        SelectionDAG 패턴 합성
        
        각 ISA 명령어에 대해:
        1. 의미론에서 DAG 패턴 추론
        2. 패턴 우선순위 결정
        3. 특수 케이스 처리 (복합 패턴, 의사 명령어 등)
        """
        patterns = []
        
        for instr in self.isa.instructions:
            semantics = self.isa.semantics.get(instr.mnemonic)
            
            if semantics:
                # 의미론에서 패턴 추론
                dag_pattern = self.infer_dag_pattern(instr, semantics)
                
                if dag_pattern:
                    patterns.append(dag_pattern)
                else:
                    # ML 모델에 위임
                    ml_pattern = self.ml_generate_pattern(instr, semantics)
                    patterns.append(ml_pattern)
        
        return patterns
    
    def infer_dag_pattern(self, instr, semantics):
        """
        의미론에서 DAG 패턴 추론
        
        VeGen/Hydride 스타일의 접근법 적용
        """
        # 의미론을 DFG로 변환
        dfg = semantics_to_dfg(semantics)
        
        # DFG를 SelectionDAG 노드로 매핑
        # 표준 LLVM DAG 노드들: add, sub, load, store, etc.
        dag_nodes = self.map_to_dag_nodes(dfg)
        
        # TableGen 패턴 형식으로 변환
        pattern = self.format_as_tablegen(instr, dag_nodes)
        
        return pattern
```

### 4.3 템플릿 라이브러리

```python
class TemplateLibrary:
    """
    재사용 가능한 백엔드 템플릿 라이브러리
    
    기존 LLVM 백엔드 (RISC-V, ARM, x86) 분석에서 추출
    """
    
    # 템플릿 카테고리
    CATEGORIES = [
        'RegisterInfo',
        'InstrInfo', 
        'ISelDAGToDAG',
        'FrameLowering',
        'CallingConv',
        'MCCodeEmitter',
        'AsmParser',
        'AsmPrinter',
        'Disassembler'
    ]
    
    def __init__(self, template_dir):
        self.templates = {}
        self.load_templates(template_dir)
    
    def load_templates(self, template_dir):
        """
        템플릿 로드 및 파라미터화
        
        템플릿 예시 (RegisterInfo):
        ```
        // RegisterInfo.td template
        class ${TARGET}Reg<bits<16> Enc, string n> : Register<n> {
          let HWEncoding = Enc;
          let Namespace = "${TARGET}";
        }
        
        ${REGISTER_DEFINITIONS}
        
        def ${TARGET}GPR : RegisterClass<"${TARGET}", [${GPR_TYPE}], ${ALIGNMENT}, (
          sequence "${REG_PREFIX}%u", 0, ${NUM_GPRS-1}
        )>;
        ```
        """
        pass
    
    def get(self, template_name: str) -> Template:
        return self.templates.get(template_name)
    
    def find_best_match(self, component_type: str, 
                        isa_features: Dict) -> Template:
        """
        ISA 특성에 가장 적합한 템플릿 선택
        """
        candidates = self.templates.get(component_type, [])
        
        scores = []
        for template in candidates:
            score = self.compute_similarity(template.features, isa_features)
            scores.append((template, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores else None
```

---

## 5. Layer 3: Verification-Guided Refinement

### 5.1 CGNR (Counterexample-Guided Neural Repair) 상세

```python
class CGNREngine:
    """
    Counterexample-Guided Neural Repair Engine
    
    핵심 알고리즘:
    1. 생성된 코드에 대해 검증 조건 생성
    2. SMT 솔버로 검증
    3. 반례 발견 시 결함 위치 추정
    4. 신경망 기반 수정 후보 생성
    5. 검증 통과까지 반복
    """
    
    def __init__(self, vc_generator, smt_solver, repair_model, 
                 max_iterations=10):
        self.vc_gen = vc_generator
        self.solver = smt_solver
        self.repair = repair_model
        self.max_iter = max_iterations
        
        # 수정 이력 (중복 방지)
        self.repair_history = []
        
    def refine(self, code: BackendCode, 
               spec: ComponentSpec) -> RefinementResult:
        """
        검증 주도 반복 개선
        """
        current_code = code
        
        for iteration in range(self.max_iter):
            # Step 1: 검증 조건 생성
            vcs = self.vc_gen.generate(current_code, spec)
            
            # Step 2: SMT 검증
            result = self.solver.check_all(vcs)
            
            if result.all_valid():
                return RefinementResult(
                    status='VERIFIED',
                    code=current_code,
                    iterations=iteration + 1
                )
            
            # Step 3: 반례에서 결함 위치 추정
            counterexample = result.get_counterexample()
            fault_loc = self.localize_fault(counterexample, current_code)
            
            # Step 4: 신경망 기반 수정
            repair_context = self.build_repair_context(
                current_code, fault_loc, counterexample, spec
            )
            
            candidates = self.repair.generate_candidates(
                repair_context, 
                num_candidates=5,
                avoid_previous=self.repair_history
            )
            
            # Step 5: 최선의 후보 선택
            best_candidate = self.select_best_candidate(
                candidates, spec
            )
            
            if best_candidate is None:
                return RefinementResult(
                    status='REPAIR_FAILED',
                    code=current_code,
                    iterations=iteration + 1,
                    error=counterexample
                )
            
            # 코드 업데이트
            current_code = self.apply_repair(
                current_code, fault_loc, best_candidate
            )
            self.repair_history.append(best_candidate.signature)
        
        return RefinementResult(
            status='MAX_ITERATIONS',
            code=current_code,
            iterations=self.max_iter
        )
    
    def localize_fault(self, counterexample, code):
        """
        결함 위치 추정
        
        기법:
        1. 반례 입력에 대한 실행 추적
        2. 예상 출력과 실제 출력의 차이점 분석
        3. 관련 코드 영역 식별
        """
        # 반례 실행
        trace = self.symbolic_execute(code, counterexample.inputs)
        
        # 출력 차이 분석
        expected = counterexample.expected_outputs
        actual = trace.outputs
        
        # 차이가 발생한 변수/레지스터 식별
        diff_vars = self.find_output_differences(expected, actual)
        
        # 해당 변수에 영향을 미친 코드 영역 추적
        relevant_lines = self.backward_slice(trace, diff_vars)
        
        return FaultLocation(
            lines=relevant_lines,
            variables=diff_vars,
            trace=trace
        )
    
    def build_repair_context(self, code, fault_loc, 
                            counterexample, spec):
        """
        수정 컨텍스트 구성
        
        신경망 모델에 제공되는 정보:
        1. 결함이 있는 코드 조각
        2. 반례 (입력/출력)
        3. 관련 스펙/제약조건
        4. 주변 컨텍스트
        """
        return RepairContext(
            faulty_code=code.extract_region(fault_loc.lines),
            counterexample={
                'inputs': counterexample.inputs,
                'expected': counterexample.expected_outputs,
                'actual': fault_loc.trace.outputs
            },
            specification=spec.get_relevant_constraints(fault_loc),
            context={
                'before': code.get_lines_before(fault_loc.lines, n=10),
                'after': code.get_lines_after(fault_loc.lines, n=10),
                'dependencies': code.get_dependencies(fault_loc.lines)
            }
        )
    
    def select_best_candidate(self, candidates, spec):
        """
        후보 중 최선 선택
        
        기준:
        1. 부분 검증 통과 (필수)
        2. 코드 품질 메트릭
        3. 스펙 적합도
        """
        valid_candidates = []
        
        for candidate in candidates:
            # 빠른 부분 검증
            partial_vcs = self.vc_gen.generate_partial(candidate, spec)
            if self.solver.quick_check(partial_vcs):
                score = self.compute_candidate_score(candidate, spec)
                valid_candidates.append((candidate, score))
        
        if not valid_candidates:
            return None
        
        # 점수 기준 정렬
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        return valid_candidates[0][0]
```

### 5.2 검증 조건 생성

```python
class VCGenerator:
    """
    검증 조건 (Verification Condition) 생성기
    """
    
    def generate(self, code: BackendCode, 
                spec: ComponentSpec) -> List[VC]:
        """
        컴포넌트 유형에 따른 VC 생성
        """
        if isinstance(spec, RegisterInfoSpec):
            return self.generate_register_vcs(code, spec)
        elif isinstance(spec, InstructionSelectionSpec):
            return self.generate_isel_vcs(code, spec)
        elif isinstance(spec, MCEmitterSpec):
            return self.generate_emitter_vcs(code, spec)
        # ... etc
    
    def generate_isel_vcs(self, code, spec):
        """
        Instruction Selection 검증 조건
        
        각 패턴에 대해:
        - 패턴이 매칭될 때 올바른 명령어가 선택되는가?
        - 선택된 명령어의 의미론이 원본 DAG의 의미론과 같은가?
        """
        vcs = []
        
        for pattern in code.patterns:
            # VC: pattern matches ∧ pattern semantics ⇒ expected semantics
            vc = VC(
                name=f"isel_{pattern.name}",
                formula=Implies(
                    And(
                        pattern.match_condition(),
                        pattern.instruction_semantics()
                    ),
                    spec.expected_semantics(pattern.dag_node)
                )
            )
            vcs.append(vc)
        
        return vcs
    
    def generate_emitter_vcs(self, code, spec):
        """
        MC Code Emitter 검증 조건
        
        각 명령어에 대해:
        - 생성된 바이너리 인코딩이 스펙과 일치하는가?
        """
        vcs = []
        
        for instr in spec.instructions:
            encoding_spec = spec.get_encoding(instr)
            
            # VC: emitted_bits = spec_encoding
            vc = VC(
                name=f"emitter_{instr}",
                formula=Equals(
                    code.emit_function(instr),
                    encoding_spec.to_bitvector()
                )
            )
            vcs.append(vc)
        
        return vcs
```

### 5.3 신경망 기반 수정 모델

```python
class NeuralRepairModel:
    """
    CGNR의 신경망 수정 모델
    
    현재 시스템의 CodeT5 기반 모델을 확장:
    - ISA 컨텍스트 인식
    - 구조화된 출력 생성
    - 반례 정보 활용
    """
    
    def __init__(self, base_model="Salesforce/codet5-large"):
        self.encoder = ISAContextEncoder(base_model)
        self.decoder = StructuredCodeDecoder(base_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model)
        
    def generate_candidates(self, context: RepairContext,
                          num_candidates: int = 5,
                          avoid_previous: List = None) -> List[RepairCandidate]:
        """
        수정 후보 생성
        
        입력 형식:
        [REPAIR] {faulty_code}
        [COUNTER] Input: {inputs} Expected: {expected} Got: {actual}
        [SPEC] {specification}
        [CONTEXT] {surrounding_code}
        
        출력: 수정된 코드
        """
        # 입력 구성
        input_text = self.format_input(context)
        
        # 토큰화
        inputs = self.tokenizer(
            input_text, 
            return_tensors='pt',
            max_length=2048,
            truncation=True
        )
        
        # Beam search로 후보 생성
        outputs = self.decoder.generate(
            self.encoder(inputs),
            num_beams=num_candidates * 2,
            num_return_sequences=num_candidates,
            max_length=512,
            do_sample=False,  # 결정론적 생성
            bad_words_ids=self.get_avoid_tokens(avoid_previous)
        )
        
        # 후보 파싱 및 검증
        candidates = []
        for output in outputs:
            code = self.tokenizer.decode(output, skip_special_tokens=True)
            if self.is_syntactically_valid(code):
                candidates.append(RepairCandidate(
                    code=code,
                    confidence=self.compute_confidence(output)
                ))
        
        return candidates
    
    def format_input(self, context: RepairContext) -> str:
        """
        수정 컨텍스트를 모델 입력 형식으로 변환
        """
        return f"""[REPAIR] {context.faulty_code}
[COUNTER] Input: {context.counterexample['inputs']} 
Expected: {context.counterexample['expected']} 
Got: {context.counterexample['actual']}
[SPEC] {context.specification}
[CONTEXT] Before: {context.context['before']}
After: {context.context['after']}
Dependencies: {context.context['dependencies']}"""
```

---

## 6. Integration and Pipeline

### 6.1 End-to-End Pipeline

```python
class VERAPipeline:
    """
    VERA End-to-End 파이프라인
    """
    
    def __init__(self, config: VERAConfig):
        # Layer 1 components
        self.doc_processor = ISADocumentProcessor(
            config.llm_client,
            config.max_chunk_tokens
        )
        self.isa_extractor = HybridISAExtractor(
            config.llm_client,
            config.smt_solver
        )
        
        # Layer 2 components
        self.template_library = TemplateLibrary(config.template_dir)
        self.synthesizers = self._init_synthesizers(config)
        
        # Layer 3 components
        self.cgnr = CGNREngine(
            VCGenerator(),
            config.smt_solver,
            NeuralRepairModel(config.repair_model_path),
            config.max_repair_iterations
        )
    
    def generate_backend(self, 
                        isa_spec_path: str,
                        output_dir: str) -> GenerationResult:
        """
        전체 백엔드 생성 파이프라인
        """
        result = GenerationResult()
        
        # Phase 1: ISA Understanding
        print("Phase 1: Extracting ISA formal model...")
        isa_model = self.extract_isa_model(isa_spec_path)
        result.isa_model = isa_model
        
        # Phase 2: Compositional Synthesis
        print("Phase 2: Synthesizing backend components...")
        draft_backend = self.synthesize_components(isa_model)
        result.draft_backend = draft_backend
        
        # Phase 3: Verification-Guided Refinement
        print("Phase 3: Verifying and refining...")
        verified_backend = self.verify_and_refine(
            draft_backend, isa_model
        )
        result.verified_backend = verified_backend
        
        # Export
        print("Exporting backend...")
        self.export_backend(verified_backend, output_dir)
        
        return result
    
    def extract_isa_model(self, spec_path) -> ISAFormalModel:
        """
        Phase 1: ISA 문서에서 형식 모델 추출
        """
        # 문서 처리
        processed = self.doc_processor.process_document(spec_path)
        
        # 형식 모델 구축
        model = ISAFormalModel(
            name=processed['metadata']['name'],
            version=processed['metadata']['version'],
            endianness=processed['metadata']['endianness'],
            pointer_size=processed['metadata']['pointer_size'],
            register_classes=processed['registers'],
            special_registers=processed['special_registers'],
            instructions=processed['instructions'],
            instruction_formats=processed['formats'],
            encodings=processed['encodings'],
            semantics=self.isa_extractor.extract_all_semantics(
                processed['instructions']
            ),
            calling_convention=processed['abi'],
            stack_layout=processed['stack']
        )
        
        # 모델 검증
        model.validate()
        
        return model
    
    def synthesize_components(self, 
                             isa_model: ISAFormalModel) -> BackendCode:
        """
        Phase 2: 컴포넌트별 합성
        """
        backend = BackendCode(target=isa_model.name)
        
        # 의존성 순서대로 합성
        synthesis_order = [
            'TargetInfo',
            'RegisterInfo',
            'InstrInfo',
            'SubtargetInfo',
            'FrameLowering',
            'ISelDAGToDAG',
            'CallingConv',
            'AsmParser',
            'AsmPrinter',
            'MCCodeEmitter',
            'Disassembler'
        ]
        
        for component in synthesis_order:
            print(f"  Synthesizing {component}...")
            
            synthesizer = self.synthesizers[component]
            result = synthesizer.synthesize()
            
            backend.add_component(component, result.code)
            
            if not result.verification.passed():
                print(f"    Warning: {component} has verification issues")
                backend.mark_needs_refinement(component)
        
        return backend
    
    def verify_and_refine(self, 
                         backend: BackendCode,
                         isa_model: ISAFormalModel) -> BackendCode:
        """
        Phase 3: 검증 및 개선
        """
        # 각 컴포넌트 개선
        for component in backend.components_needing_refinement():
            print(f"  Refining {component}...")
            
            code = backend.get_component(component)
            spec = self.get_component_spec(component, isa_model)
            
            result = self.cgnr.refine(code, spec)
            
            if result.status == 'VERIFIED':
                print(f"    {component} verified in {result.iterations} iterations")
                backend.update_component(component, result.code)
            else:
                print(f"    {component} refinement failed: {result.status}")
                backend.mark_manual_review(component)
        
        # 통합 검증
        print("  Running integration tests...")
        integration_result = self.run_integration_tests(backend, isa_model)
        
        return backend
```

### 6.2 설정 및 실행

```python
# Example usage
config = VERAConfig(
    # LLM settings
    llm_client=OpenAIClient(model="gpt-4-turbo"),
    max_chunk_tokens=8000,
    
    # SMT settings
    smt_solver=Z3Solver(timeout=30000),
    
    # Template settings
    template_dir="./templates/llvm",
    
    # Repair settings
    repair_model_path="./models/cgnr_repair_v2",
    max_repair_iterations=10,
    
    # Output settings
    output_format="llvm-backend"
)

pipeline = VERAPipeline(config)

# Generate backend for RISC-V
result = pipeline.generate_backend(
    isa_spec_path="./specs/riscv-user-manual.pdf",
    output_dir="./output/riscv-backend"
)

print(f"Generation complete!")
print(f"  Total components: {len(result.verified_backend.components)}")
print(f"  Verified: {result.verified_backend.verified_count()}")
print(f"  Needs manual review: {result.verified_backend.manual_review_count()}")
```

---

## 7. Expected Outcomes and Metrics

### 7.1 목표 성능 지표

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      VERA Expected Performance                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Metric                          │ VEGA (baseline) │ VERA (target)      │
│  ───────────────────────────────│────────────────│────────────────────│
│  Function-level accuracy         │     71.5%      │     85%+           │
│  Statement-level accuracy        │     55.0%      │     90%+           │
│  Semantic correctness           │    Unknown     │    100%*           │
│  End-to-end compilation         │      <8%       │     95%+           │
│  Manual intervention required   │      High      │     Minimal        │
│                                                                         │
│  * Within verified components                                           │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Time Metrics                   │                 │                    │
│  ───────────────────────────────│────────────────│────────────────────│
│  ISA understanding phase        │      N/A       │   2-4 hours        │
│  Component synthesis            │   ~1 hour      │   1-2 hours        │
│  Verification & refinement      │      N/A       │   4-8 hours        │
│  Total (new ISA)               │   Weeks+       │   1-2 days         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 점진적 성공 기준

```
Level 1: Basic Functionality (Month 3)
├── RegisterInfo 자동 생성 및 검증
├── InstrInfo 기본 명령어 패턴 생성
└── MC Emitter 단순 명령어 지원

Level 2: Core Backend (Month 6)
├── ISelDAGToDAG 주요 패턴 생성
├── FrameLowering 기본 ABI 지원
├── End-to-end simple programs 컴파일

Level 3: Complete Backend (Month 12)
├── 전체 ISA 커버리지
├── 최적화 패스 통합
├── 벤치마크 성능 평가
└── 새로운 ISA (가상)로 일반화 검증
```

---

## 8. Conclusion

VERA 프레임워크는 기존 연구들의 한계를 극복하기 위해:

1. **ISA 문서에서 시작**: VEGA와 달리 기존 백엔드 없이 시작 가능
2. **형식 검증 통합**: ACT/Isaria의 형식 기법을 범용 ISA로 확장
3. **자동화된 개선**: CGNR을 통해 검증 실패 시 자동 수정
4. **모듈화된 접근**: 각 컴포넌트 독립 합성 및 검증

이 프레임워크가 성공적으로 구현되면, 새로운 ISA에 대한 컴파일러 백엔드 개발 비용을 수 주에서 수 일로 단축할 수 있을 것으로 기대된다.

---

## References

[1] M. Zhong et al., "VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model," CGO 2025  
[2] M. Jain et al., "ACT: Compiler Backends from Tensor Accelerator ISA Descriptions," arXiv 2025  
[3] S. Thomas, J. Bornholt, "Automatic Generation of Vectorizing Compilers for Customizable DSPs," ASPLOS 2024  
[4] A. Freitag et al., "The Vienna Architecture Description Language," arXiv 2024  
[5] Y. Chen et al., "VeGen: A Vectorizer Generator for SIMD and Beyond," ASPLOS 2021  
[6] M. Willsey et al., "Hydride: Portably Lifting Vector Intrinsics to IR Level," ASPLOS 2022  
[7] P. Cousot, R. Cousot, "Abstract Interpretation: A Unified Lattice Model," POPL 1977  
[8] C.A.R. Hoare, "An Axiomatic Basis for Computer Programming," CACM 1969  
[9] E. Clarke et al., "Counterexample-Guided Abstraction Refinement," CAV 2000  
