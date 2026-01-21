# VEGA 논문 분석 및 개선 제안 보고서

**작성자**: AI Compiler Backend Autogeneration Researcher  
**일자**: 2026-01-21  
**대상 논문**: VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model (CGO 2025)

---

## 1. 논문 개요

### 1.1 VEGA의 핵심 아이디어

VEGA는 사전 훈련된 트랜스포머 모델(UniXcoder)을 사용하여 LLVM 컴파일러 백엔드를 자동으로 생성하는 AI 기반 시스템입니다.

**핵심 접근법:**
1. **Function Template Abstraction**: 기존 백엔드(ARM, MIPS 등)에서 함수들을 수집하여 공통 구조를 추상화
2. **Feature Vector Extraction**: 각 statement를 target-independent(TI)와 target-specific(TS) 특성으로 구분
3. **Code Generation**: UniXcoder를 fine-tuning하여 새 타겟에 대한 코드 생성
4. **Confidence Scoring**: 생성된 코드의 신뢰도 점수 제공

### 1.2 실험 결과

| Target | Function-level Accuracy | Statement-level Accuracy |
|--------|------------------------|-------------------------|
| RISC-V | 71.5% | 55.0% |
| RI5CY | 73.2% | 58.5% |
| xCORE | 62.2% | - |

**ForkFlow 대비**: 기존 fork-flow 방식(<8% 정확도) 대비 크게 향상

---

## 2. 관련 연구 (Related Work)

### 2.1 전통적 접근법

| 연구 | 방법론 | 한계 |
|------|--------|------|
| **LISA ADL** | Architecture Description Language 기반 | 수동 ADL 작성 필요, 표현력 제한 |
| **VADL/OpenVADL** | Vienna ADL 기반 LLVM 백엔드 생성 | 새로운 DSL 학습 필요 |
| **TableGen** | LLVM의 선언적 테이블 생성기 | 저수준, 많은 수동 작업 필요 |

### 2.2 프로그램 합성 기반 접근법

| 연구 | 방법론 | 특징 |
|------|--------|------|
| **Hydride (ASPLOS 2024)** | Synthesis-based instruction selection | SMT solver 활용, 벡터 명령어 특화 |
| **Diospyros (ASPLOS 2020)** | Program synthesis for DSP | 고정 크기 벡터만 지원 |
| **VeGen** | Vectorizer generation | SIMD 명령어 특화 |

### 2.3 형식 검증 기반 접근법

| 연구 | 방법론 | 특징 |
|------|--------|------|
| **3LA** | ILA(Instruction-Level Abstraction) | Formal SW/HW interface |
| **ACT (Oct 2025)** | Equality saturation + Constraint programming | Tensor accelerator 특화, 형식적 soundness/completeness 보장 |

### 2.4 기계학습 기반 접근법

| 연구 | 방법론 | 한계 |
|------|--------|------|
| **VEGA** | Pre-trained Transformer (UniXcoder) | 의미론적 검증 부재 |
| **Neural Code Generation** | Various LLMs | 컴파일러 특화 부족 |

---

## 3. VEGA의 약점 및 한계점 분석

### 3.1 Critical Limitations (심각한 한계)

#### 3.1.1 No Formal Semantic Verification (형식적 의미론 검증 부재)
```
문제점:
- 생성된 코드가 syntactically correct하더라도 semantically incorrect할 수 있음
- Regression test 통과가 correctness를 보장하지 않음
- 희귀한 edge case에서의 버그를 놓칠 수 있음

영향:
- Mission-critical 시스템에 사용 불가
- 보안 취약점 발생 가능성
```

#### 3.1.2 Limited Function Module Coverage (제한된 함수 모듈 커버리지)
```
커버되는 7개 모듈:
1. AsmPrinter
2. ISelDAGToDAG
3. MCCodeEmitter
4. AsmParser
5. Disassembler
6. RegisterInfo
7. InstrInfo

커버되지 않는 부분 (~40%):
- Instruction Scheduling
- Register Allocation optimization
- Target-specific passes
- Debug information emission
- Exception handling
```

#### 3.1.3 Training Data Dependency (훈련 데이터 의존성)
```
문제점:
- ARM, MIPS, X86 등 기존 백엔드 패턴에 의존
- 근본적으로 다른 아키텍처 (VLIW, Dataflow, Neuromorphic)에 대한 일반화 어려움
- Novel instruction patterns에 대한 취약성
```

### 3.2 Moderate Limitations (중간 수준 한계)

#### 3.2.1 Statement-Level Granularity
```
문제점:
- Cross-statement 최적화 기회 상실
- Context window 제한으로 인한 장거리 의존성 포착 어려움
```

#### 3.2.2 Confidence Score Limitations
```
문제점:
- Binary classification이 미묘한 오류를 포착하지 못함
- Calibration 문제 - 실제 correctness와 confidence의 불일치 가능
```

#### 3.2.3 TableGen Dependency
```
문제점:
- 타겟 설명 파일(.td)이 이미 존재해야 함
- TableGen의 한계가 VEGA의 한계로 전이됨
```

### 3.3 Quantitative Analysis of Errors (오류 유형 정량 분석)

| Error Type | Description | Proportion |
|------------|-------------|------------|
| **Err-Pred** | Target-specific 내용 예측 실패 | ~60% |
| **Err-Conf** | Confidence 점수 예측 실패 | ~25% |
| **Err-Def** | Function template 정의 오류 | ~15% |

---

## 4. 개선 아이디어 제안

### 4.1 Proposal: **VEGA-Verified** - Semantically Verified Neural Backend Generation

#### 핵심 아이디어
VEGA의 신경망 기반 코드 생성과 형식 검증(Formal Verification)을 결합하여, **생성된 코드의 의미론적 정확성을 보장**하는 시스템

```
Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    VEGA-Verified Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Neural Generation (VEGA)                                   │
│     ┌─────────────┐   ┌──────────────┐   ┌────────────────┐    │
│     │ Function    │──▶│ UniXcoder    │──▶│ Generated      │    │
│     │ Template    │   │ Fine-tuned   │   │ Code (Draft)   │    │
│     └─────────────┘   └──────────────┘   └────────────────┘    │
│                                                  │               │
│  2. Semantic Extraction                          ▼               │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ AST Parser → Control Flow Graph → Semantic Model     │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                  │               │
│  3. Specification Inference                      ▼               │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ Reference Backend Analysis → Specification Template   │    │
│     │ (Pre/Post conditions, Invariants)                    │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                  │               │
│  4. Verification Phase                           ▼               │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ SMT Solver (Z3) + Bounded Model Checking             │    │
│     │ → Verification Result + Counterexamples              │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                  │               │
│  5. Repair Loop (if verification fails)          ▼               │
│     ┌─────────────────────────────────────────────────────┐    │
│     │ Counterexample-Guided Neural Repair                  │    │
│     │ → Refined Code → Re-verify                           │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4.2 Main Contributions (vs Related Work)

#### Contribution 1: **Automated Semantic Specification Inference**

**Related Work와의 차별점:**

| Aspect | ACT | Hydride | VEGA | **VEGA-Verified** |
|--------|-----|---------|------|-------------------|
| Specification | Manual ISA description | Vendor pseudocode | Implicit in templates | **Auto-inferred from reference** |
| Coverage | Tensor accelerators | Vector instructions | General backend | **General backend** |
| Formal guarantee | Sound & Complete | Partial | None | **Sound** |

**핵심 기술:**
```python
# Specification Inference Algorithm
def infer_specification(reference_implementations):
    """
    여러 타겟의 동일 함수 구현을 분석하여
    공통 specification을 추론
    """
    # 1. Symbolic execution of reference implementations
    symbolic_traces = [symbolic_exec(impl) for impl in reference_implementations]
    
    # 2. Abstract interpretation to find common patterns
    common_pattern = abstract_merge(symbolic_traces)
    
    # 3. Generate verification conditions
    preconditions = extract_preconditions(common_pattern)
    postconditions = extract_postconditions(common_pattern)
    
    return Specification(preconditions, postconditions, invariants)
```

**이론적 근거:**
- Abstract Interpretation theory (Cousot & Cousot)
- Predicate Abstraction
- Program invariant inference (Daikon-style)

---

#### Contribution 2: **Counterexample-Guided Neural Repair (CGNR)**

**Related Work와의 차별점:**

| Approach | Method | Limitation |
|----------|--------|------------|
| Traditional CEGIS | SMT-only synthesis | Scalability issues |
| Neural repair (existing) | Heuristic-based | No formal guarantee |
| **CGNR** | **Hybrid neural + formal** | **Guided repair with formal feedback** |

**핵심 알고리즘:**
```
Algorithm: Counterexample-Guided Neural Repair
─────────────────────────────────────────────────
Input: Generated code C, Specification S
Output: Verified code C* or FAIL

1. (verified, counterexample) ← Verify(C, S)
2. while not verified and iterations < MAX:
   a. error_context ← ExtractErrorContext(C, counterexample)
   b. repair_prompt ← ConstructRepairPrompt(C, error_context)
   c. C ← NeuralRepair(repair_prompt)  // Fine-tuned repair model
   d. (verified, counterexample) ← Verify(C, S)
3. return C if verified else FAIL
```

**이론적 근거:**
- Counterexample-Guided Abstraction Refinement (CEGAR)
- Program Repair via Neural Networks
- Bounded Model Checking

---

#### Contribution 3: **Hierarchical Verification with Modular Composability**

**Related Work와의 차별점:**

| Approach | Verification Scope | Composability |
|----------|-------------------|---------------|
| VEGA | None | N/A |
| ACT | Full backend | Monolithic |
| **VEGA-Verified** | **Function → Module → Backend** | **Modular with interface contracts** |

**핵심 아이디어:**
```
Verification Hierarchy:
─────────────────────────────────────────────────

Level 3: Backend Integration Verification
         ┌─────────────────────────────────┐
         │ Cross-module consistency check  │
         │ End-to-end property verification│
         └─────────────────────────────────┘
                        ▲
Level 2: Module-level Verification
         ┌────────┐ ┌────────┐ ┌────────┐
         │AsmPrint│ │ISelDAG │ │MCEmit  │
         │Verified│ │Verified│ │Verified│
         └────────┘ └────────┘ └────────┘
                        ▲
Level 1: Function-level Verification  
         ┌───┐┌───┐┌───┐ ┌───┐┌───┐ ┌───┐
         │f1 ││f2 ││f3 │ │g1 ││g2 │ │h1 │
         └───┘└───┘└───┘ └───┘└───┘ └───┘
         
Benefits:
- Incremental verification (faster iteration)
- Localized repair (easier debugging)
- Formal interface contracts between modules
```

**이론적 근거:**
- Modular Verification (Hoare Logic extension)
- Assume-Guarantee Reasoning
- Compositional Semantics

---

### 4.3 Implementation Plan (구현 계획)

#### Phase 1: Lightweight Verification Layer (구현 난이도: 낮음)
```
Tasks:
1. Reference backend symbolic execution framework
2. Basic specification template generation
3. Z3-based verification of generated code

Estimated effort: 2-3 months
```

#### Phase 2: Neural Repair Integration (구현 난이도: 중간)
```
Tasks:
1. Counterexample encoder for neural model
2. Repair model fine-tuning
3. CGNR loop implementation

Estimated effort: 3-4 months
```

#### Phase 3: Hierarchical Verification (구현 난이도: 중간-높음)
```
Tasks:
1. Interface contract specification language
2. Modular verification infrastructure
3. Cross-module consistency checker

Estimated effort: 4-5 months
```

---

### 4.4 Expected Results (기대 결과)

| Metric | VEGA | **VEGA-Verified (Expected)** |
|--------|------|------------------------------|
| Function-level Accuracy | 71.5% | **85-90%** (with repair) |
| Semantic Correctness | Unknown | **100%** (verified portion) |
| Verification Coverage | 0% | **80-90%** |
| False Negative Rate | High | **Low** (formal guarantee) |
| Development Time | ~1 hour | ~2-3 hours (+verification) |

---

## 5. Alternative Improvement Ideas

### 5.1 VEGA + Equality Saturation
```
아이디어: E-graph 기반 instruction selection을 VEGA와 결합
장점: 
- Optimal instruction selection guarantee
- Target-independent optimization preservation
단점:
- E-graph scalability 문제
- Existing VEGA pipeline과의 통합 복잡성
```

### 5.2 VEGA + Active Learning
```
아이디어: 개발자 피드백을 활용한 점진적 모델 개선
장점:
- 도메인 전문가 지식 활용
- Continuous improvement
단점:
- Human-in-the-loop 요구
- 피드백 품질 의존성
```

### 5.3 VEGA + LLM Enhancement
```
아이디어: GPT-4/Claude 등 대형 LLM을 fine-tuning 또는 few-shot learning으로 활용
장점:
- 더 넓은 코드 패턴 이해
- Natural language specification 가능
단점:
- Inference cost
- Hallucination 문제
- 재현성 어려움
```

---

## 6. 결론

VEGA는 컴파일러 백엔드 자동 생성 분야에서 혁신적인 첫 걸음이지만, 실제 production 환경에서 사용되기 위해서는 **형식적 검증**이 필수적입니다.

제안하는 **VEGA-Verified**는:
1. **자동 specification 추론**을 통해 수동 작업 최소화
2. **Counterexample-guided neural repair**를 통해 정확도 향상
3. **계층적 모듈 검증**을 통해 확장성과 유지보수성 확보

이 세 가지 핵심 기여를 통해 기존 연구들과 명확히 차별화되며, 이론적으로 탄탄한 기반 위에서 구현 가능합니다.

---

## References

1. Zhong et al., "VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model", CGO 2025
2. Kothen et al., "Hydride: A Retargetable and Extensible Synthesis-based Compiler for Modern Hardware Architectures", ASPLOS 2024
3. ACT: "Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions", arXiv 2025
4. 3LA: "Application-Level Validation of Accelerator Designs Using a Formal Software/Hardware Interface"
5. Tate et al., "Equality Saturation: A New Approach to Optimization", POPL 2009
6. OpenVADL: "An Open Source Implementation of the Vienna Architecture Description Language", 2025
7. Guo et al., "UniXcoder: Unified Cross-Modal Pre-training for Code Representation", ACL 2022
8. LISA ADL: "Retargetable Code Generation based on an Architecture Description Language"
