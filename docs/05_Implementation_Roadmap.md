# 05. Implementation Roadmap and Future Directions

**Author**: Compiler Backend Autogeneration Research Group  
**Version**: 2.0  
**Date**: 2026-01-23  
**Classification**: Strategic Planning Document

---

## Executive Summary

이 문서는 VERA 프레임워크의 구현 로드맵과 향후 연구 방향을 제시한다. 현재 VEGA-Verified 시스템에서 VERA로의 전환 전략, 단계별 마일스톤, 그리고 학술적 기여 계획을 포함한다.

---

## 1. Current State Assessment

### 1.1 현재 시스템 (VEGA-Verified) 분석

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Current VEGA-Verified System                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Implemented Components:                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ✓ Neural Repair Model (CodeT5-based)                            │   │
│  │   - Seq2Seq 버그 수정 모델                                       │   │
│  │   - 5개 패턴 → 2000개 샘플 확장                                  │   │
│  │   - "fix bug: {code}" 프롬프트 형식                              │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ ✓ Basic Verification Pipeline                                   │   │
│  │   - SMT 기반 검증 조건 생성                                      │   │
│  │   - Counterexample 추출                                         │   │
│  │   - 반복 수정 루프                                               │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ ✓ Training Infrastructure                                       │   │
│  │   - HuggingFace Trainer 통합                                    │   │
│  │   - GPU 지원 학습                                                │   │
│  │   - 체크포인트 저장/로드                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Limitations:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ✗ ISA 스펙에서 직접 생성 불가                                    │   │
│  │ ✗ 버그 수정에 국한 (생성이 아님)                                  │   │
│  │ ✗ 제한된 학습 데이터 패턴                                        │   │
│  │ ✗ Counterexample의 비효율적 활용                                │   │
│  │ ✗ 결정론적 생성 미지원 (do_sample=True)                         │   │
│  │ ✗ 프롬프트-학습 데이터 불일치                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Gap Analysis

```
Required for VERA vs Currently Available:

┌─────────────────────────────────────────────────────────────────────────┐
│  Component                    │ Required │ Available │ Gap Level        │
├─────────────────────────────────────────────────────────────────────────┤
│  ISA Document Parser          │    ✓     │    ✗     │ HIGH            │
│  LLM-based Extraction         │    ✓     │    ✗     │ MEDIUM          │
│  Formal Model Builder         │    ✓     │    △     │ MEDIUM          │
│  Template Library             │    ✓     │    ✗     │ HIGH            │
│  Component Synthesizers       │    ✓     │    ✗     │ HIGH            │
│  SMT-based Verification       │    ✓     │    △     │ LOW             │
│  Neural Repair Model          │    ✓     │    △     │ MEDIUM          │
│  Integration Pipeline         │    ✓     │    ✗     │ HIGH            │
│  Evaluation Framework         │    ✓     │    ✗     │ MEDIUM          │
└─────────────────────────────────────────────────────────────────────────┘

Legend: ✓ = Available, △ = Partial, ✗ = Missing
```

---

## 2. Implementation Phases

### 2.1 Phase 0: Foundation (Month 1-2)

**목표**: 개발 환경 구축 및 기존 시스템 안정화

```
Milestones:

M0.1: Development Environment Setup
├── Python 3.10+ 개발 환경
├── LLVM/Clang 빌드 환경
├── SMT 솔버 (Z3) 통합
├── LLM API 접근 설정
└── CI/CD 파이프라인 구축

M0.2: Current System Stabilization
├── Neural Repair 모델 결정론적 생성 전환
│   - do_sample=False 설정
│   - temperature=1.0 고정
│   - beam_size 조정
├── 프롬프트 형식 통일
│   - 학습/추론 프롬프트 일치화
│   - 구조화된 입력 형식 도입
└── 테스트 커버리지 확보

M0.3: Baseline Evaluation
├── 현재 시스템 성능 측정
│   - 수정 성공률
│   - 검증 통과율
│   - 반복 횟수
├── 벤치마크 테스트셋 구축
└── 메트릭 대시보드 구축
```

**Deliverables**:
1. 안정화된 현재 시스템
2. 기준 성능 리포트
3. 개발 환경 문서

### 2.2 Phase 1: ISA Understanding Layer (Month 3-5)

**목표**: ISA 명세 문서에서 형식 모델 추출

```
Milestones:

M1.1: ISA Document Parser (Month 3)
├── PDF 파서 구현
│   - PyMuPDF 기반 텍스트 추출
│   - 테이블 구조 인식
│   - 그림/다이어그램 처리
├── XML 파서 구현 (있는 경우)
├── 문서 구조 분석기
│   - 목차 기반 섹션 분류
│   - 섹션 유형 자동 식별
└── 청크 기반 분할 시스템

M1.2: LLM-based Information Extraction (Month 4)
├── 레지스터 정보 추출 프롬프트
├── 명령어 정보 추출 프롬프트
├── 인코딩 정보 추출 프롬프트
├── 의미론 추출 프롬프트
├── 크로스 레퍼런스 해결
└── 추출 결과 검증

M1.3: Formal Model Construction (Month 5)
├── ISAFormalModel 데이터 구조
├── SMT-LIB 의미론 인코딩
├── 모델 일관성 검증
└── RISC-V 초기 타겟 테스트
```

**구현 예시**:

```python
# M1.1: Document Parser
class ISADocumentParser:
    def parse(self, pdf_path: str) -> StructuredDocument:
        """
        ISA PDF 문서 파싱
        """
        # 텍스트 추출
        text = self.extract_text(pdf_path)
        
        # 테이블 추출
        tables = self.extract_tables(pdf_path)
        
        # 구조 분석
        sections = self.identify_sections(text)
        
        return StructuredDocument(
            text=text,
            tables=tables,
            sections=sections
        )

# M1.2: LLM Extraction
class LLMExtractor:
    REGISTER_EXTRACTION_PROMPT = """
    Extract register information from the following ISA documentation.
    
    Required fields:
    - register_class: Name of the register class (e.g., GPR, FPR)
    - registers: List of register names
    - width: Register width in bits
    - abi_roles: ABI-defined roles (argument, return, callee_saved, etc.)
    
    Documentation:
    {doc_chunk}
    
    Output as JSON:
    """
    
    def extract_registers(self, doc_chunk: str) -> Dict:
        response = self.llm.query(
            self.REGISTER_EXTRACTION_PROMPT.format(doc_chunk=doc_chunk)
        )
        return json.loads(response)

# M1.3: Formal Model
@dataclass
class ISAFormalModel:
    name: str
    registers: List[RegisterClass]
    instructions: List[InstructionInfo]
    semantics: Dict[str, SMTFormula]
    
    def validate(self) -> bool:
        """모델 일관성 검증"""
        # 모든 명령어가 정의된 레지스터만 참조하는지 확인
        valid_regs = {r.name for rc in self.registers for r in rc.registers}
        
        for instr in self.instructions:
            for operand in instr.operands:
                if operand.type == 'register':
                    if operand.reg_class not in valid_regs:
                        return False
        
        return True
```

**Deliverables**:
1. ISA 문서 파서
2. LLM 추출 프롬프트 라이브러리
3. ISAFormalModel 구현
4. RISC-V 모델 (검증용)

### 2.3 Phase 2: Compositional Synthesis Layer (Month 6-8)

**목표**: ISA 모델에서 백엔드 컴포넌트 합성

```
Milestones:

M2.1: Template Library Construction (Month 6)
├── LLVM 백엔드 분석
│   - RISC-V 백엔드 구조 분석
│   - ARM 백엔드 구조 분석
│   - 공통 패턴 추출
├── 파라미터화된 템플릿 설계
│   - RegisterInfo 템플릿
│   - InstrInfo 템플릿
│   - 기타 컴포넌트 템플릿
└── 템플릿 인스턴스화 엔진

M2.2: Component Synthesizers (Month 7)
├── RegisterInfoSynthesizer
├── InstrInfoSynthesizer
├── ISelDAGToDAGSynthesizer (기본)
├── FrameLoweringSynthesizer
├── AsmPrinter/ParserSynthesizer
└── MCCodeEmitterSynthesizer

M2.3: ML-Assisted Gap Filling (Month 8)
├── ISA-Aware 인코더 설계
├── 구조화된 디코더 설계
├── 학습 데이터 구축
│   - ISA 스펙 → 코드 쌍
│   - 기존 백엔드에서 추출
└── 모델 학습 및 평가
```

**구현 예시**:

```python
# M2.1: Template Library
class RegisterInfoTemplate:
    """RegisterInfo 컴포넌트 템플릿"""
    
    TEMPLATE = """
// ${TARGET}RegisterInfo.td
// Auto-generated by VERA

class ${TARGET}Reg<bits<16> Enc, string n, list<Register> subregs = []>
    : Register<n> {
  let HWEncoding = Enc;
  let Namespace = "${TARGET}";
  let SubRegs = subregs;
}

${REGISTER_DEFINITIONS}

// Register Classes
${REGISTER_CLASSES}

// Calling Convention
def CC_${TARGET} : CallingConv<[
  ${CC_RULES}
]>;
"""
    
    def instantiate(self, isa_model: ISAFormalModel) -> str:
        code = self.TEMPLATE
        
        # 레지스터 정의 생성
        reg_defs = self.generate_register_definitions(isa_model.registers)
        code = code.replace("${REGISTER_DEFINITIONS}", reg_defs)
        
        # 레지스터 클래스 생성
        reg_classes = self.generate_register_classes(isa_model.registers)
        code = code.replace("${REGISTER_CLASSES}", reg_classes)
        
        # Calling Convention 생성
        cc_rules = self.generate_cc_rules(isa_model.calling_convention)
        code = code.replace("${CC_RULES}", cc_rules)
        
        return code

# M2.2: Component Synthesizer
class ISelDAGToDAGSynthesizer:
    """Instruction Selection 패턴 합성기"""
    
    def synthesize(self, isa_model: ISAFormalModel) -> str:
        patterns = []
        
        for instr in isa_model.instructions:
            # 의미론에서 DAG 패턴 추론
            semantics = isa_model.semantics.get(instr.mnemonic)
            
            if semantics:
                pattern = self.infer_pattern(instr, semantics)
                if pattern:
                    patterns.append(pattern)
                else:
                    # ML 모델로 폴백
                    pattern = self.ml_generate_pattern(instr, semantics)
                    patterns.append(pattern)
        
        return self.format_tablegen_patterns(patterns)
    
    def infer_pattern(self, instr, semantics):
        """
        의미론에서 SelectionDAG 패턴 추론
        
        VeGen/Hydride 스타일 접근
        """
        # 의미론 분석
        operation = semantics.get_main_operation()
        
        # 표준 LLVM 노드로 매핑
        if operation.type == 'add':
            return self.create_add_pattern(instr, semantics)
        elif operation.type == 'sub':
            return self.create_sub_pattern(instr, semantics)
        elif operation.type == 'load':
            return self.create_load_pattern(instr, semantics)
        # ... 기타 연산
        
        return None

# M2.3: ML-Assisted Generation
class ISAAwareCodeGenerator:
    """ISA 인식 코드 생성 모델"""
    
    def __init__(self, base_model="Salesforce/codet5-large"):
        self.encoder = ISASpecEncoder(base_model)
        self.decoder = ComponentDecoder(base_model)
    
    def generate(self, isa_spec, component_type, template_hint=None):
        """
        ISA 스펙에서 컴포넌트 코드 생성
        
        Input format:
        [ISA] {isa_spec_features}
        [COMPONENT] {component_type}
        [TEMPLATE] {template_hint}  # optional
        """
        input_text = f"[ISA] {self.format_isa_spec(isa_spec)}\n"
        input_text += f"[COMPONENT] {component_type}\n"
        if template_hint:
            input_text += f"[TEMPLATE] {template_hint}\n"
        
        # 인코딩
        encoded = self.encoder.encode(input_text)
        
        # 디코딩 (결정론적)
        output = self.decoder.generate(
            encoded,
            max_length=1024,
            do_sample=False,
            num_beams=5
        )
        
        return output
```

**Deliverables**:
1. 템플릿 라이브러리 (8+ 컴포넌트)
2. 컴포넌트 합성기 세트
3. ISA-Aware 생성 모델
4. Draft 백엔드 생성 파이프라인

### 2.4 Phase 3: Verification-Guided Refinement (Month 9-11)

**목표**: CGNR 엔진 완성 및 통합

```
Milestones:

M3.1: Enhanced Verification (Month 9)
├── 컴포넌트별 VC 생성기
│   - RegisterInfo VCs
│   - InstrInfo VCs
│   - ISelDAG VCs
│   - MCEmitter VCs
├── Incremental SMT 검증
├── 결함 위치 추정 알고리즘
└── 반례 분석 및 시각화

M3.2: Advanced Neural Repair (Month 10)
├── 컨텍스트 인식 수정 모델
│   - 반례 정보 활용
│   - ISA 제약 통합
│   - 이전 시도 회피
├── 학습 데이터 확장
│   - 검증 실패 케이스 수집
│   - 자동 데이터 증강
└── 모델 재학습 파이프라인

M3.3: CGNR Integration (Month 11)
├── 반복 개선 루프 통합
├── 수렴 보장 메커니즘
├── 수동 개입 인터페이스
└── 개선 진행 모니터링
```

**구현 예시**:

```python
# M3.1: Enhanced Verification
class ComponentVCGenerator:
    """컴포넌트별 검증 조건 생성"""
    
    def generate_isel_vcs(self, patterns, semantics):
        """
        Instruction Selection 검증 조건
        
        각 패턴에 대해:
        pattern.match(dag) ∧ instr.semantics → dag.semantics
        """
        vcs = []
        
        for pattern in patterns:
            instr_name = pattern.get_instruction()
            instr_sem = semantics.get(instr_name)
            dag_sem = pattern.get_dag_semantics()
            
            # VC: 명령어 의미론이 DAG 의미론을 구현하는가?
            vc = VC(
                name=f"isel_{pattern.name}",
                precondition=pattern.get_match_condition(),
                formula=Implies(instr_sem, dag_sem)
            )
            vcs.append(vc)
        
        return vcs
    
    def generate_emitter_vcs(self, emitter_code, encoding_spec):
        """
        MC Emitter 검증 조건
        
        각 명령어에 대해:
        emitter(instr) = spec_encoding(instr)
        """
        vcs = []
        
        for instr, spec in encoding_spec.items():
            emitted = emitter_code.get_emission(instr)
            expected = spec.to_bitvector()
            
            vc = VC(
                name=f"emit_{instr}",
                formula=Equals(emitted, expected)
            )
            vcs.append(vc)
        
        return vcs

# M3.2: Advanced Neural Repair
class AdvancedRepairModel:
    """향상된 신경망 수정 모델"""
    
    def generate_repair(self, context: RepairContext):
        """
        수정 후보 생성
        
        향상된 컨텍스트:
        1. 반례 정보 (입력/기대/실제)
        2. ISA 제약 조건
        3. 이전 실패한 시도들
        4. 유사 성공 사례
        """
        # 반례 분석
        ce_analysis = self.analyze_counterexample(context.counterexample)
        
        # 유사 성공 사례 검색
        similar_cases = self.retrieve_similar_repairs(context)
        
        # 입력 구성
        input_text = f"""[FAULTY] {context.faulty_code}
[COUNTER] {ce_analysis}
[CONSTRAINT] {context.isa_constraints}
[AVOID] {context.failed_attempts}
[SIMILAR] {similar_cases}
[REPAIR]"""
        
        # 생성
        candidates = self.model.generate(
            input_text,
            num_return_sequences=5,
            do_sample=False
        )
        
        # 빠른 필터링
        valid_candidates = []
        for c in candidates:
            if self.quick_validate(c, context):
                valid_candidates.append(c)
        
        return valid_candidates

# M3.3: CGNR Integration
class CGNRPipeline:
    """통합 CGNR 파이프라인"""
    
    def refine_backend(self, backend: BackendCode, 
                       isa_model: ISAFormalModel):
        """
        전체 백엔드 검증 및 개선
        """
        # 컴포넌트별 개선
        for component in backend.components:
            if self.needs_refinement(component):
                self.refine_component(component, isa_model)
        
        # 통합 테스트
        integration_result = self.run_integration_tests(backend)
        
        if not integration_result.passed():
            # 통합 수준 문제 해결
            self.fix_integration_issues(backend, integration_result)
        
        return backend
    
    def refine_component(self, component, isa_model):
        """단일 컴포넌트 개선"""
        max_iterations = 10
        
        for i in range(max_iterations):
            # VC 생성
            vcs = self.vc_gen.generate(component, isa_model)
            
            # SMT 검증
            result = self.smt_solver.check_all(vcs)
            
            if result.all_valid():
                print(f"  {component.name} verified!")
                return
            
            # 반례 추출 및 수정
            ce = result.get_counterexample()
            fault_loc = self.localize(ce, component)
            
            repair = self.repair_model.generate_repair(
                RepairContext(
                    faulty_code=component.get_code(fault_loc),
                    counterexample=ce,
                    isa_constraints=isa_model.get_constraints()
                )
            )
            
            component.apply_repair(fault_loc, repair)
        
        print(f"  {component.name}: max iterations reached")
```

**Deliverables**:
1. 완전한 CGNR 엔진
2. 컴포넌트별 VC 생성기
3. 향상된 수정 모델
4. 통합 검증 파이프라인

### 2.5 Phase 4: Evaluation and Publication (Month 12-15)

**목표**: 평가, 논문 작성, 오픈소스 공개

```
Milestones:

M4.1: Comprehensive Evaluation (Month 12-13)
├── Target ISAs
│   - RISC-V (primary)
│   - ARM subset
│   - x86 subset
│   - Hypothetical ISA (generalization test)
├── Metrics
│   - Generation accuracy (function/statement level)
│   - Verification success rate
│   - End-to-end compilation success
│   - Manual intervention required
│   - Time-to-backend
├── Baselines
│   - VEGA
│   - Manual implementation
│   - VADL (if comparable)
└── Ablation Studies
    - Without LLM extraction
    - Without CGNR
    - Without templates

M4.2: Paper Writing (Month 13-14)
├── Target venues
│   - PLDI 2027 (Primary)
│   - ASPLOS 2027 (Alternative)
│   - CGO 2027 (Alternative)
├── Paper structure
│   - Introduction: Problem & Motivation
│   - Background: Related Work
│   - Approach: VERA Framework
│   - Implementation: Key Techniques
│   - Evaluation: Comprehensive Results
│   - Discussion: Limitations & Future
│   - Conclusion
└── Artifact preparation

M4.3: Open Source Release (Month 15)
├── Code cleanup and documentation
├── Installation guides
├── Tutorial notebooks
├── Pre-trained models
├── Template library
└── Example ISA specs
```

---

## 3. Resource Requirements

### 3.1 Compute Resources

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Compute Resource Requirements                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Development & Training:                                                │
│  ├── GPU: 2x NVIDIA A100 (40GB) or equivalent                          │
│  ├── CPU: 32+ cores                                                    │
│  ├── RAM: 128GB+                                                       │
│  └── Storage: 2TB+ SSD                                                 │
│                                                                         │
│  SMT Solving:                                                          │
│  ├── CPU: 64+ cores (Z3 parallelization)                               │
│  ├── RAM: 256GB+ (large VCs)                                           │
│  └── Estimated solver time: ~1-4 hours per component                   │
│                                                                         │
│  LLM API:                                                              │
│  ├── GPT-4 Turbo: ~$0.01/1K input + $0.03/1K output                   │
│  ├── Estimated monthly cost: $500-1000                                 │
│  └── Alternative: Local LLM (requires additional GPU)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Human Resources

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Team Composition                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Core Team:                                                            │
│  ├── Lead Researcher (1)                                               │
│  │   - Overall direction                                               │
│  │   - Paper writing                                                   │
│  │   - Evaluation design                                               │
│  │                                                                     │
│  ├── Compiler Engineer (1-2)                                           │
│  │   - LLVM backend expertise                                          │
│  │   - Template library construction                                   │
│  │   - ISA analysis                                                    │
│  │                                                                     │
│  ├── ML Engineer (1)                                                   │
│  │   - Neural repair model                                             │
│  │   - ISA-aware generation                                            │
│  │   - Training infrastructure                                         │
│  │                                                                     │
│  └── Formal Methods Expert (1)                                         │
│      - VC generation                                                   │
│      - SMT optimization                                                │
│      - Verification theory                                             │
│                                                                         │
│  Advisory:                                                             │
│  └── Senior Advisor (0.25 FTE)                                         │
│      - Technical guidance                                              │
│      - Paper review                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Risk Assessment

### 4.1 Technical Risks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Technical Risk Assessment                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Risk: LLM extraction quality                                          │
│  Level: MEDIUM                                                         │
│  Impact: ISA 모델 품질 저하 → 생성 품질 저하                            │
│  Mitigation:                                                           │
│  ├── 형식 검증으로 추출 결과 검증                                       │
│  ├── 인간 검토 인터페이스 제공                                          │
│  └── 대안: 구조화된 ISA 입력 형식 허용                                  │
│                                                                         │
│  Risk: SMT solver scalability                                          │
│  Level: HIGH                                                           │
│  Impact: 대규모 백엔드 검증 불가                                        │
│  Mitigation:                                                           │
│  ├── Incremental SMT 적용                                              │
│  ├── 컴포넌트 단위 검증 (분할 정복)                                     │
│  └── Bounded verification 허용                                         │
│                                                                         │
│  Risk: Template generalization                                         │
│  Level: MEDIUM                                                         │
│  Impact: 새로운 ISA에 템플릿 적용 실패                                  │
│  Mitigation:                                                           │
│  ├── 다양한 ISA에서 템플릿 추출                                         │
│  ├── ML 모델로 폴백                                                    │
│  └── 수동 템플릿 확장 허용                                              │
│                                                                         │
│  Risk: Neural repair convergence                                       │
│  Level: MEDIUM                                                         │
│  Impact: CGNR 루프 미수렴                                              │
│  Mitigation:                                                           │
│  ├── 수렴 보장 조건 연구                                                │
│  ├── 최대 반복 제한                                                     │
│  └── 수동 개입 지점 제공                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Schedule Risks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Schedule Risk Assessment                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Risk: ISA document complexity                                         │
│  Probability: HIGH                                                     │
│  Impact: Phase 1 지연                                                  │
│  Contingency:                                                          │
│  ├── RISC-V에 집중 (문서화 우수)                                       │
│  ├── 수동 보조 허용                                                    │
│  └── 범위 축소 (필수 컴포넌트만)                                        │
│                                                                         │
│  Risk: LLVM version updates                                            │
│  Probability: MEDIUM                                                   │
│  Impact: 템플릿/API 변경 대응 필요                                      │
│  Contingency:                                                          │
│  ├── LLVM 버전 고정 (e.g., LLVM 18)                                    │
│  └── 버전 추상화 레이어                                                 │
│                                                                         │
│  Risk: Publication timeline                                            │
│  Probability: MEDIUM                                                   │
│  Impact: 학회 데드라인 놓침                                             │
│  Contingency:                                                          │
│  ├── 다중 학회 타겟팅                                                   │
│  └── Workshop paper 선행 발표                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Academic Contribution Plan

### 5.1 Publication Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Publication Strategy                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Tier 1 Target (Full Paper):                                           │
│  ├── PLDI 2027: "VERA: Verified End-to-end Retargetable Architecture   │
│  │              for Automatic Compiler Backend Generation"              │
│  │   - Full framework presentation                                     │
│  │   - Comprehensive evaluation                                        │
│  │   - Deadline: November 2026                                         │
│  │                                                                     │
│  └── ASPLOS 2027 (Alternative)                                         │
│      - Systems-oriented presentation                                   │
│      - Performance focus                                               │
│                                                                         │
│  Tier 2 Target (Component Papers):                                     │
│  ├── "LLM-based ISA Semantic Extraction"                               │
│  │   - OOPSLA or ICSE                                                  │
│  │   - Focus on extraction accuracy                                    │
│  │                                                                     │
│  └── "Counterexample-Guided Neural Repair for Compiler Backends"       │
│      - CGO or CC                                                       │
│      - Focus on CGNR technique                                         │
│                                                                         │
│  Workshop Papers:                                                      │
│  ├── "Towards Automated Compiler Backend Generation"                   │
│  │   - LLVM Dev Meeting                                                │
│  │   - Progress report                                                 │
│  │                                                                     │
│  └── "Neural-Symbolic Synthesis for Compiler Construction"             │
│      - ML4Code Workshop                                                │
│      - ML techniques focus                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Novel Contributions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Key Novel Contributions                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  C1. ISA-Spec-to-Backend Pipeline                                      │
│      First system to generate verified backends from ISA spec docs     │
│      - Unlike VEGA: doesn't require existing backends                  │
│      - Unlike VADL: doesn't require detailed custom spec               │
│                                                                         │
│  C2. LLM + Formal Verification Integration                             │
│      Novel combination of LLM extraction with formal verification      │
│      - LLM for natural language understanding                          │
│      - Formal methods for correctness guarantees                       │
│                                                                         │
│  C3. Counterexample-Guided Neural Repair (CGNR)                        │
│      Extension of CEGAR to neural code generation                      │
│      - Automated repair from verification failures                     │
│      - Feedback loop between verification and generation               │
│                                                                         │
│  C4. Compositional Backend Synthesis                                   │
│      Modular approach to backend generation                            │
│      - Independent component verification                              │
│      - Assume-guarantee reasoning                                      │
│                                                                         │
│  C5. Template + ML Hybrid Synthesis                                    │
│      Combining structural templates with ML flexibility                │
│      - Templates provide structure                                     │
│      - ML fills gaps                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Related Work Positioning

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Related Work Comparison                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                    ISA Spec as Input?  Verified?  General ISA?         │
│                    ─────────────────  ─────────  ────────────         │
│  VEGA             │       ✗         │    ✗     │     ✓                │
│  ACT              │       ✓         │    ✓     │     ✗ (tensor)       │
│  Isaria           │       △         │    ✓     │     ✗ (DSP)          │
│  VADL             │       ✓*        │    △     │     ✓                │
│  VeGen            │       △         │    ✓     │     ✗ (vector)       │
│  Hydride          │       ✗         │    ✓     │     △ (SIMD)         │
│  ─────────────────┼─────────────────┼──────────┼─────────────         │
│  VERA (Ours)      │       ✓         │    ✓     │     ✓                │
│                                                                         │
│  * VADL requires custom detailed spec, not standard ISA docs           │
│                                                                         │
│  Key differentiator: VERA is the first to combine:                     │
│  1. Standard ISA documentation as input                                │
│  2. Formal verification of generated code                              │
│  3. Support for general-purpose ISAs                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Long-term Vision

### 6.1 Beyond VERA 1.0

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Long-term Research Directions                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  VERA 2.0: Optimization Integration                                    │
│  ├── Target-specific optimization pass generation                      │
│  ├── Peephole optimization synthesis                                   │
│  └── Auto-tuning for performance                                       │
│                                                                         │
│  VERA 3.0: Multi-target Support                                        │
│  ├── Heterogeneous backend generation                                  │
│  ├── Cross-compilation support                                         │
│  └── Runtime target switching                                          │
│                                                                         │
│  VERA 4.0: Self-improving System                                       │
│  ├── Continuous learning from failures                                 │
│  ├── Automatic template evolution                                      │
│  └── Community-contributed improvements                                │
│                                                                         │
│  Research Spinoffs:                                                    │
│  ├── LLM-based program synthesis verification                          │
│  ├── Neural-symbolic compiler construction                             │
│  └── Automatic compiler testing                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Industry Impact

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Potential Industry Applications                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Hardware Development:                                                 │
│  ├── Rapid prototyping of new processors                               │
│  ├── ASIC/FPGA compiler bootstrap                                      │
│  └── Custom accelerator software support                               │
│                                                                         │
│  Semiconductor Companies:                                              │
│  ├── Reduced time-to-market for new chips                              │
│  ├── Lower compiler development costs                                  │
│  └── Faster iteration on ISA design                                    │
│                                                                         │
│  Research Institutions:                                                │
│  ├── Educational tool for compiler courses                             │
│  ├── Research platform for compiler optimization                       │
│  └── Benchmark for ML4Code research                                    │
│                                                                         │
│  Open Source Community:                                                │
│  ├── LLVM ecosystem contribution                                       │
│  ├── Open ISA backend support                                          │
│  └── Compiler accessibility democratization                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Timeline Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      15-Month Implementation Timeline                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Month  1-2  │ Phase 0: Foundation                                     │
│              │ - Environment setup                                     │
│              │ - Current system stabilization                          │
│              │ - Baseline evaluation                                   │
│                                                                         │
│  Month  3-5  │ Phase 1: ISA Understanding                              │
│              │ - Document parser                                       │
│              │ - LLM extraction                                        │
│              │ - Formal model construction                             │
│                                                                         │
│  Month  6-8  │ Phase 2: Compositional Synthesis                        │
│              │ - Template library                                      │
│              │ - Component synthesizers                                │
│              │ - ML-assisted generation                                │
│                                                                         │
│  Month  9-11 │ Phase 3: Verification-Guided Refinement                 │
│              │ - Enhanced verification                                 │
│              │ - Advanced neural repair                                │
│              │ - CGNR integration                                      │
│                                                                         │
│  Month 12-15 │ Phase 4: Evaluation & Publication                       │
│              │ - Comprehensive evaluation                              │
│              │ - Paper writing (PLDI 2027)                             │
│              │ - Open source release                                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  Key Milestones:                                                       │
│  • M3:  ISA model for RISC-V                                           │
│  • M6:  Template library complete                                      │
│  • M9:  Draft backend generation working                               │
│  • M12: Full pipeline operational                                      │
│  • M14: Paper submission                                               │
│  • M15: Open source release                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Conclusion

VERA 프레임워크의 구현은 야심찬 목표이지만, 기존 연구들의 장점을 조합하고 한계를 극복하는 체계적인 접근을 통해 달성 가능하다.

**핵심 성공 요인**:
1. **점진적 개발**: 각 단계에서 유용한 결과물 생성
2. **위험 완화**: 대안 경로와 폴백 메커니즘 준비
3. **학술적 기여**: 명확한 차별화와 기여점
4. **실용성**: 실제 사용 가능한 도구 개발

이 로드맵을 따르면, 15개월 내에 ISA 스펙에서 검증된 컴파일러 백엔드를 자동 생성하는 최초의 실용적 시스템을 구축할 수 있을 것이다.

---

## References

[1] M. Zhong et al., "VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model," CGO 2025  
[2] M. Jain et al., "ACT: Compiler Backends from Tensor Accelerator ISA Descriptions," arXiv 2025  
[3] S. Thomas, J. Bornholt, "Automatic Generation of Vectorizing Compilers for Customizable DSPs," ASPLOS 2024  
[4] A. Freitag et al., "The Vienna Architecture Description Language," arXiv 2024  
[5] Y. Chen et al., "VeGen: A Vectorizer Generator for SIMD and Beyond," ASPLOS 2021  
[6] M. Willsey et al., "Hydride: Portably Lifting Vector Intrinsics to IR Level," ASPLOS 2022  
[7] C. Lattner, V. Adve, "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation," CGO 2004  
[8] L. De Moura, N. Bjørner, "Z3: An Efficient SMT Solver," TACAS 2008  
