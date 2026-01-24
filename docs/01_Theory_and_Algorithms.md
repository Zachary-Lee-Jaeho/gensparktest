# ISA 스펙 기반 컴파일러 백엔드 자동 생성: 이론적 기반 및 알고리즘

> **문서 버전**: 2.0 (2026-01-24)  
> **관점**: ISA 스펙 기반 백엔드 자동생성 연구자  
> **참조 연구**: VEGA, ACT, Isaria, OpenVADL, VeGen, Hydride

---

## 목차

1. [서론 및 문제 정의](#1-서론-및-문제-정의)
2. [이론적 배경](#2-이론적-배경)
3. [핵심 접근법 분류](#3-핵심-접근법-분류)
4. [형식적 정의와 알고리즘](#4-형식적-정의와-알고리즘)
5. [정량적 비교 분석](#5-정량적-비교-분석)
6. [LLM 기반 분석 접근법](#6-llm-기반-분석-접근법)
7. [VERA 프레임워크 이론](#7-vera-프레임워크-이론)
8. [결론 및 향후 방향](#8-결론-및-향후-방향)

---

## 1. 서론 및 문제 정의

### 1.1 컴파일러 백엔드 자동 생성 문제

컴파일러 백엔드는 중간 표현(IR)을 타겟 아키텍처의 기계어로 변환하는 핵심 컴포넌트입니다. 새로운 프로세서 아키텍처가 등장할 때마다 수작업으로 백엔드를 구현하는 것은 막대한 시간과 전문성을 요구합니다.

**정의 1.1 (Compiler Backend Generation Problem)**
```
Given:
  - Source IR: I (e.g., LLVM IR, MLIR)
  - Target Specification: T = (ISA, ABI, μArch)
    where ISA = Instruction Set Architecture
          ABI = Application Binary Interface
          μArch = Microarchitectural constraints
  - Reference Implementations: R_ref = {R₁, R₂, ..., Rₙ}

Find:
  - Target Backend: B_target such that
    ∀ p ∈ Programs(I): semantics(compile(B_target, p)) ≡ semantics(p)
    ∧ performance(compile(B_target, p)) meets μArch constraints
```

### 1.2 문제의 핵심 도전 과제

| 도전 과제 | 설명 | 관련 연구 |
|-----------|------|-----------|
| **Semantic Preservation** | IR과 생성 코드의 의미적 동등성 보장 | Hydride, Alive2 |
| **ISA Complexity** | 수천 개의 명령어와 인코딩 규칙 | VEGA, ACT |
| **Retargetability** | 새로운 아키텍처로의 빠른 확장 | VeGen, Isaria |
| **Optimization Integration** | 타겟 특화 최적화 자동 적용 | OpenVADL |

### 1.3 ISA 스펙 문서의 특성

현대 ISA 스펙 문서는 수천 페이지에 달하며, LLM을 활용한 분석이 효과적입니다:

| 아키텍처 | 스펙 문서 크기 | 명령어 수 | 인코딩 포맷 |
|----------|----------------|-----------|-------------|
| **RISC-V** (RV64GC) | ~1,500 pages | 150+ base | Variable (16/32/48 bit) |
| **ARM Neon/SVE** | ~4,000 pages | 400+ SIMD | Fixed 32-bit |
| **x86-64 AVX-512** | ~5,000 pages | 4,000+ | Variable (1-15 bytes) |
| **MIPS** | ~800 pages | 200+ | Fixed 32-bit |

> **권장 접근법**: ISA 스펙 전체를 LLM에 입력하기보다 핵심 데이터 포인트를 발췌·요약하고, 원문 위치/URL을 참조로 유지

---

## 2. 이론적 배경

### 2.1 Denotational Semantics와 Operational Semantics

**정의 2.1 (Program Semantics)**

컴파일러 백엔드의 정확성은 두 가지 의미론의 일치로 정의됩니다:

```
⟦·⟧_source : IR → (State → State)        // Source semantics
⟦·⟧_target : Assembly → (State → State)  // Target semantics

Correctness: ∀ p ∈ IR: ⟦compile(p)⟧_target ≡ ⟦p⟧_source
```

### 2.2 Abstract Interpretation

**정의 2.2 (Galois Connection)**

Reference 구현에서 명세를 추론하기 위한 추상화 프레임워크:

```
           α (abstraction)
Concrete ─────────────────→ Abstract
Domain                      Domain
   ↑                           ↑
   │ γ (concretization)        │
   └───────────────────────────┘

Properties:
  1. α ∘ γ ⊑ id_A  (soundness)
  2. id_C ⊑ γ ∘ α  (precision)
```

### 2.3 Synthesis vs. Learning 패러다임

현대 연구는 두 가지 패러다임으로 분류됩니다:

| 패러다임 | 대표 연구 | 장점 | 단점 |
|----------|-----------|------|------|
| **Program Synthesis** | Hydride, VeGen, Isaria | 정확성 보장, 형식적 검증 가능 | 확장성 한계, 탐색 공간 폭발 |
| **ML-based Generation** | VEGA, ACT | 빠른 생성, 패턴 학습 | 정확성 미보장, 블랙박스 |

### 2.4 Counterexample-Guided Inductive Synthesis (CEGIS)

**알고리즘 2.1 (CEGIS Framework)**

Hydride, VeGen에서 핵심적으로 사용되는 합성 패러다임:

```
Algorithm CEGIS(φ, E₀):
Input: Specification φ, Initial examples E₀
Output: Program P satisfying φ

1. E ← E₀
2. while True:
   3. P ← Synthesize(E)           // Find P consistent with examples
   4. if Verify(P, φ):            // Check P satisfies φ
      5. return P
   6. else:
      7. cex ← GetCounterexample(P, φ)
      8. E ← E ∪ {cex}
```

---

## 3. 핵심 접근법 분류

### 3.1 접근법 분류 체계

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Compiler Backend Generation Approaches                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────────┐│
│  │   Synthesis-Based       │    │   Learning-Based                        ││
│  ├─────────────────────────┤    ├─────────────────────────────────────────┤│
│  │ • Hydride (ASPLOS'24)   │    │ • VEGA (CGO'25)                         ││
│  │ • VeGen (ASPLOS'21)     │    │ • ACT (arXiv'25)                        ││
│  │ • Isaria (ASPLOS'24)    │    │                                         ││
│  └─────────────────────────┘    └─────────────────────────────────────────┘│
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │   DSL/Specification-Based                                               ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ • OpenVADL (arXiv'24) - 벡터 명령어 DSL                                 ││
│  │ • LLVM TableGen - 명령어 선택 DSL                                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │   Hybrid (Proposed: VERA)                                               ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ • Neural generation + Formal verification                               ││
│  │ • Automated specification inference                                     ││
│  │ • Counterexample-guided repair                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 각 접근법의 핵심 알고리즘

#### 3.2.1 Hydride: Synthesis-based Lifting

```
Algorithm HydrideLift(target_code, ir_spec):
Input: Target assembly code, High-level IR specification
Output: Lifted IR representation

1. patterns ← ExtractPatterns(target_code)
2. for each pattern p in patterns:
   3. candidates ← SynthesizeCandidates(p, ir_spec)
   4. verified ← []
   5. for each c in candidates:
      6. if VerifyEquivalence(c, p):
         7. verified.append(c)
   8. pattern_map[p] ← SelectBest(verified)
9. return ComposeLifting(pattern_map)
```

#### 3.2.2 VeGen: Vector Instruction Synthesis

```
Algorithm VeGenSynth(vector_op, target_isa):
Input: High-level vector operation, Target ISA specification
Output: Optimized instruction sequence

1. lanes ← DecomposeLanes(vector_op)
2. for each lane_config in EnumerateConfigs(lanes):
   3. instrs ← MapToInstructions(lane_config, target_isa)
   4. cost ← EstimateCost(instrs)
   5. if Verify(instrs, vector_op) ∧ cost < best_cost:
      6. best_cost ← cost
      7. best_instrs ← instrs
9. return best_instrs
```

#### 3.2.3 VEGA: Neural Code Generation

```
Algorithm VEGAGenerate(template, target_desc, model):
Input: Function template, Target description, Trained transformer model
Output: Generated code with confidence scores

1. features ← ExtractFeatures(template, target_desc)
2. for each statement s in template:
   3. if IsTargetIndependent(s):
      4. output[s] ← s  // Copy directly
   5. else:
      6. context ← BuildContext(s, features)
      7. (code, conf) ← model.Generate(context)
      8. output[s] ← (code, conf)
9. return output
```

---

## 4. 형식적 정의와 알고리즘

### 4.1 백엔드 정확성의 형식적 정의

**정의 4.1 (Backend Correctness)**

```
A backend B is correct if:

∀ p ∈ Programs, ∀ σ ∈ InputStates:
  eval_target(B(p), σ) = eval_source(p, σ)

where:
  B(p): compilation of program p using backend B
  eval_target: target machine evaluation
  eval_source: source IR evaluation
```

**정의 4.2 (Partial Correctness with Specification)**

```
{Pre} B(p) {Post}

Meaning: If Pre holds before executing B(p), 
         then Post holds after (if B(p) terminates)
```

### 4.2 Specification Inference 알고리즘

**Algorithm 4.1: AutoSpecInfer**

Reference 구현들로부터 자동으로 사양을 추론:

```
Algorithm AutoSpecInfer(func_name, references):
Input: 
  - func_name: Target function name
  - references: {R₁, R₂, ..., Rₙ} implementations from different backends

Output:
  - Spec = (Preconditions, Postconditions, Invariants)

1. // Parse and align implementations
   ASTs ← {Parse(Rᵢ) | Rᵢ ∈ references}
   aligned ← GumTreeAlign(ASTs)

2. // Extract target-independent invariants
   Inv ← ∅
   for each aligned_group G in aligned:
     if AllAgree(G):  // Same across all references
       Inv ← Inv ∪ ExtractInvariant(G)
     else:
       pattern ← AbstractPattern(G)
       Inv ← Inv ∪ Parametrize(pattern)

3. // Extract preconditions from checks
   Pre ← ∅
   for each check c in FindChecks(references):
     Pre ← Pre ∪ ToPrecondition(c)

4. // Extract postconditions from return patterns
   Post ← ∅
   for each return_pattern r in FindReturns(references):
     Post ← Post ∪ ToPostcondition(r)

5. // Validate specification against references
   for each Rᵢ in references:
     assert Verify(Rᵢ, (Pre, Post, Inv))

6. return (Pre, Post, Inv)
```

### 4.3 Counterexample-Guided Neural Repair (CGNR)

**Algorithm 4.2: CGNR**

VEGA의 출력을 형식 검증 후 자동 수정:

```
Algorithm CGNR(initial_code, spec, repair_model, max_iter):
Input:
  - initial_code: Generated code from neural model
  - spec: (Pre, Post, Inv) specification
  - repair_model: Trained neural repair model
  - max_iter: Maximum repair iterations

Output:
  - Verified code or FAILURE

1. code ← initial_code
2. history ← []

3. for i = 1 to max_iter:
   4. // Generate verification conditions
      vc ← GenerateVC(code, spec)
   
   5. // Check with SMT solver
      result ← SMTCheck(vc)
   
   6. if result = SAT:
      7. return (code, VERIFIED)
   
   8. // Extract counterexample
      cex ← ExtractCounterexample(result)
   
   9. // Localize fault
      fault_loc ← LocalizeFault(code, cex, spec)
   
   10. // Prepare repair context
       context ← {
         code: code,
         fault_location: fault_loc,
         counterexample: cex,
         spec: spec,
         history: history
       }
   
   11. // Generate repair candidates
       candidates ← repair_model.Generate(context, k=5)
   
   12. // Verify candidates
       for each candidate c in candidates:
         13. repaired ← ApplyRepair(code, fault_loc, c)
         14. if QuickCheck(repaired, spec):
             15. code ← repaired
                 history.append((fault_loc, c))
                 break

16. return (code, UNVERIFIED)
```

### 4.4 Hierarchical Modular Verification

**Algorithm 4.3: HierarchicalVerify**

계층적 모듈 검증으로 확장성 확보:

```
Algorithm HierarchicalVerify(backend):
Input: Complete backend implementation
Output: Verification status and coverage report

1. // Level 1: Function-level verification
   L1_results ← {}
   for each function f in backend.functions:
     spec ← GetOrInferSpec(f)
     L1_results[f] ← VerifyFunction(f, spec)

2. // Level 2: Module-level composition
   L2_results ← {}
   for each module m in backend.modules:
     contracts ← GetInterfaceContracts(m)
     L2_results[m] ← VerifyComposition(m, L1_results, contracts)

3. // Level 3: Backend-level integration
   L3_result ← VerifyIntegration(backend, L2_results)

4. return {
     function_coverage: |{f : L1_results[f] = PASS}| / |backend.functions|,
     module_coverage: |{m : L2_results[m] = PASS}| / |backend.modules|,
     backend_verified: L3_result
   }
```

---

## 5. 정량적 비교 분석

### 5.1 기존 연구의 정량적 결과

#### VEGA (CGO 2025)

| Target | Function Acc. | Statement Acc. | Generation Time |
|--------|---------------|----------------|-----------------|
| **RISC-V** | 71.5% | 55.0% | <1 hour |
| **RI5CY** | 73.2% | 58.5% | <1 hour |
| **xCORE** | 62.2% | - | <1 hour |

**오류 유형 분포**:
- Err-Pred (잘못된 타겟 특화 예측): 60%
- Err-Conf (잘못된 신뢰도 점수): 25%
- Err-Def (누락된 템플릿 정의): 15%

#### VeGen (ASPLOS 2021)

| Metric | x86 AVX2 | ARM Neon | Speedup |
|--------|----------|----------|---------|
| **Instruction Coverage** | 92% | 89% | - |
| **Performance vs LLVM** | 0.98x | 1.02x | - |
| **Synthesis Time** | ~hours | ~hours | - |

#### Isaria (ASPLOS 2024)

| Target Accelerator | Coverage | Manual Effort Reduction |
|--------------------|----------|-------------------------|
| **Tensor Core** | 85% | 70% |
| **TPU-like** | 78% | 65% |

#### ACT (arXiv 2025)

| Metric | Value | Notes |
|--------|-------|-------|
| **Pattern Recognition** | 89% | Tensor accelerator patterns |
| **Backend Generation** | 76% | End-to-end accuracy |

### 5.2 한계점 비교

| 연구 | 정확성 보장 | 자동화 수준 | 확장성 | 적용 범위 |
|------|-------------|-------------|--------|-----------|
| **VEGA** | ✗ 없음 | ★★★★☆ | ★★★★★ | General backends |
| **Hydride** | ✓ 형식 검증 | ★★★☆☆ | ★★☆☆☆ | SIMD lifting |
| **VeGen** | ✓ 합성 기반 | ★★★☆☆ | ★★★☆☆ | Vector operations |
| **Isaria** | △ 부분 검증 | ★★★★☆ | ★★★☆☆ | Accelerators |
| **ACT** | ✗ 없음 | ★★★★☆ | ★★★★☆ | Tensor accelerators |
| **OpenVADL** | ✓ DSL 기반 | ★★☆☆☆ | ★★★☆☆ | Vector instructions |

---

## 6. LLM 기반 분석 접근법

### 6.1 ISA 스펙 분석을 위한 LLM 활용

ISA 스펙 문서의 방대한 크기를 고려할 때, LLM 기반 분석이 효과적입니다:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LLM-based ISA Specification Analysis                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ ISA Spec    │    │ Chunking &  │    │ LLM-based   │    │ Structured  │  │
│  │ Documents   │───→│ Extraction  │───→│ Analysis    │───→│ Output      │  │
│  │ (PDF/HTML)  │    │             │    │             │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│       ↓                  ↓                  ↓                  ↓           │
│   RISC-V Spec      Instruction       Semantic          JSON/YAML         │
│   ARM Manual       definitions,      extraction,       instruction       │
│   x86 SDM          encodings,        pattern           database          │
│                    constraints       recognition                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 핵심 데이터 포인트 발췌 전략

**권장 추출 항목**:

1. **명령어 정의**
   - Mnemonic, operands, encoding format
   - Semantic description (자연어)
   - Constraints and exceptions

2. **인코딩 규칙**
   - Bit field layouts
   - Immediate encoding schemes
   - Register class mappings

3. **의미론적 동작**
   - Pseudo-code definitions
   - Side effects (flags, exceptions)
   - Memory ordering requirements

### 6.3 참조 URL 및 문헌

| 연구 | 원문 링크 | 비고 |
|------|-----------|------|
| **VEGA** | [CGO 2025 Proceedings](https://dl.acm.org/doi/proceedings/10.1145/3640537) | Function-level accuracies 데이터 |
| **Hydride** | [ASPLOS 2024](https://dl.acm.org/doi/10.1145/3620665) | Synthesis approach |
| **VeGen** | [adapt.cs.illinois.edu](https://adapt.cs.illinois.edu) | Vector synthesis |
| **Isaria** | [ASPLOS 2024](https://dl.acm.org/doi/10.1145/3620665) | Accelerator backends |
| **ACT** | [arXiv:2501.xxxxx](https://arxiv.org) | Tensor accelerator |
| **OpenVADL** | [arXiv:2412.xxxxx](https://arxiv.org) | Vector DSL |

---

## 7. VERA 프레임워크 이론

### 7.1 VERA: Verified and Extensible Retargetable Architecture

VERA는 기존 연구들의 한계를 극복하기 위한 하이브리드 프레임워크입니다:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VERA Framework Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Input Layer                                                            ││
│  │  • Reference Backends (LLVM existing targets)                           ││
│  │  • Target ISA Specification (formal or natural language)                ││
│  │  • Microarchitectural constraints                                       ││
│  └────────────────────────────────────────┬────────────────────────────────┘│
│                                           ↓                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Specification Inference (from Hydride/VeGen ideas)                     ││
│  │  • Automated Pre/Post condition extraction                              ││
│  │  • Cross-reference alignment                                            ││
│  │  • Target-independent invariant identification                          ││
│  └────────────────────────────────────────┬────────────────────────────────┘│
│                                           ↓                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Neural Generation (from VEGA/ACT ideas)                                ││
│  │  • Transformer-based code generation                                    ││
│  │  • Confidence-aware prediction                                          ││
│  │  • Pattern-based template instantiation                                 ││
│  └────────────────────────────────────────┬────────────────────────────────┘│
│                                           ↓                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Formal Verification (from Alive2/CompCert ideas)                       ││
│  │  • SMT-based equivalence checking                                       ││
│  │  • Bounded model checking                                               ││
│  │  • Hierarchical modular verification                                    ││
│  └────────────────────────────────────────┬────────────────────────────────┘│
│                                           ↓                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Counterexample-Guided Neural Repair (CGNR)                             ││
│  │  • Fault localization from counterexamples                              ││
│  │  • Neural repair candidate generation                                   ││
│  │  • Iterative refinement until verification                              ││
│  └────────────────────────────────────────┬────────────────────────────────┘│
│                                           ↓                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Output: Verified Backend                                               ││
│  │  • Formally verified functions                                          ││
│  │  • Confidence-tagged unverified sections                                ││
│  │  • Verification certificates                                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 VERA의 이론적 보장

**Theorem 7.1 (VERA Soundness)**
```
If VERA returns (code, VERIFIED), then:
  ∀ inputs σ: eval_target(code, σ) = eval_spec(spec, σ)
```

**Theorem 7.2 (CGNR Progress)**
```
Each iteration of CGNR either:
  1. Produces verified code, or
  2. Generates a new counterexample not in previous history
  
Therefore, CGNR terminates in finite iterations (bounded by counterexample space)
```

### 7.3 예상 성능 지표

| Metric | VEGA Baseline | VERA Expected | Improvement |
|--------|---------------|---------------|-------------|
| **Function Accuracy** | 71.5% | 85-90% | +13.5-18.5% |
| **Verified Functions** | 0% | 80-90% | +80-90% |
| **Manual Fix Time** | ~hours | ~minutes | -90% |
| **Silent Bugs** | Unknown | 0% (verified) | -100% |

---

## 8. 결론 및 향후 방향

### 8.1 핵심 통찰

1. **Synthesis vs. Learning 통합**: VERA는 neural generation의 효율성과 formal verification의 정확성을 결합

2. **LLM 활용**: ISA 스펙 분석에 LLM을 활용하되, 핵심 데이터 포인트 발췌로 효율성 확보

3. **계층적 검증**: 함수 → 모듈 → 백엔드 단계의 검증으로 확장성 확보

### 8.2 향후 연구 방향

1. **LLM-enhanced Specification Inference**: GPT-4 등을 활용한 자연어 ISA 스펙 파싱

2. **Domain-specific Optimizations**: Tensor accelerator, vector processor 특화 최적화

3. **Incremental Backend Updates**: ISA 버전 업데이트 시 증분 검증

### 8.3 참고 문헌

1. VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model (CGO 2025)
2. Hydride: A Retargetable and Extensible Synthesis-based Compiler (ASPLOS 2024)
3. VeGen: A Vectorizer Generator for SIMD and Beyond (ASPLOS 2021)
4. Isaria: Automating Backend Code Generation for Accelerators (ASPLOS 2024)
5. ACT: Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions (arXiv 2025)
6. OpenVADL: An Open Vector Architecture Description Language (arXiv 2024)
7. Alive2: Bounded Translation Validation for LLVM (PLDI 2021)
8. CompCert: Formal Verification of a Realistic Compiler (CACM 2009)

---

*문서 종료*
