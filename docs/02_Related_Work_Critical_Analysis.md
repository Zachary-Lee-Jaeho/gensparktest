# 관련 연구 비판적 분석

**저자**: VEGA-Verified Research Team  
**최종 수정**: 2026-01-23  
**관점**: 저명한 Compiler Backend Autogeneration 연구자

---

## 목차
1. [분석 개요](#1-분석-개요)
2. [VEGA (CGO 2025)](#2-vega-cgo-2025)
3. [VeGen (ASPLOS 2021)](#3-vegen-asplos-2021)
4. [Hydride (ASPLOS 2022)](#4-hydride-asplos-2022)
5. [ACT (arXiv 2025)](#5-act-arxiv-2025)
6. [Isaria (ASPLOS 2024)](#6-isaria-asplos-2024)
7. [OpenVADL (arXiv 2024)](#7-openvadl-arxiv-2024)
8. [종합 비교 및 Gap 분석](#8-종합-비교-및-gap-분석)

---

## 1. 분석 개요

### 1.1 분석 목표

본 문서는 다음 목표를 위해 기존 연구들을 비판적으로 분석합니다:

> **핵심 질문**: 새로운 하드웨어/ISA가 출시되었을 때, ISA 스펙 정도의 간단한 정보만으로 백엔드 컴파일러를 자동생성할 수 있는가?

### 1.2 분석 기준

| 기준 | 설명 | 이상적 수준 |
|------|------|------------|
| **입력 요구사항** | 무엇이 필요한가? | ISA 스펙 문서만 |
| **출력 완전성** | 무엇을 생성하는가? | 완전한 LLVM 백엔드 |
| **정확성 보장** | 어떻게 검증하는가? | 100% semantic preservation |
| **새 ISA 적용성** | 유사 ISA 없이 가능한가? | 완전히 새로운 ISA 지원 |
| **실용성** | 실제 프로덕션에 사용 가능한가? | -O2/-O3 최적화 수준 |

### 1.3 ISA 스펙 문서의 현실

```
문서 규모의 현실:
┌─────────────────────────────────────────────────────────────┐
│ RISC-V Unprivileged Spec    │ ~150 페이지                   │
│ ARM Architecture Ref Manual │ ~8,000+ 페이지               │
│ Intel SDM                   │ ~5,000+ 페이지               │
│ MIPS Architecture           │ ~1,000+ 페이지               │
└─────────────────────────────────────────────────────────────┘

LLM 처리의 한계:
- GPT-4: ~128K 토큰 ≈ ~100페이지 텍스트
- Claude: ~200K 토큰 ≈ ~150페이지 텍스트
- 결론: 대부분의 ISA 스펙을 한번에 처리 불가
```

---

## 2. VEGA (CGO 2025)

### 2.1 핵심 방법론

```
┌─────────────────────────────────────────────────────────────┐
│                      VEGA 워크플로우                          │
├─────────────────────────────────────────────────────────────┤
│  입력: Target Description Files (.td) + 기존 백엔드          │
│        ↓                                                    │
│  함수 그룹화: ARM, MIPS, X86 백엔드에서 동일 인터페이스 수집    │
│        ↓                                                    │
│  Feature Vector: Target-specific vs Target-independent 구분 │
│        ↓                                                    │
│  UniXcoder Fine-tuning                                      │
│        ↓                                                    │
│  출력: 새 타겟용 함수 구현체 + Confidence Score              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 평가 결과

| Target | Function-level Accuracy | Statement-level Accuracy |
|--------|------------------------|-------------------------|
| RISC-V | 71.5% | 55.0% |
| RI5CY | 73.2% | 58.5% |
| xCORE | 62.2% | - |

### 2.3 비판적 분석

#### ❌ 근본적 한계 1: 기존 백엔드 의존성

```
문제 시나리오:
  완전히 새로운 ISA "RISC-Y"가 출시됨
  
VEGA의 요구사항:
  1. .td 파일이 이미 작성되어 있어야 함
  2. RISC-Y와 "유사한" 기존 백엔드 (ARM, MIPS, X86) 필요
  3. Feature vector 추출을 위한 alignment 가능해야 함

현실:
  - .td 파일 작성 자체가 백엔드 개발의 상당 부분
  - 완전히 새로운 아키텍처면 "유사한" 것이 없음
  - Feature vector가 의미 없을 수 있음
```

#### ❌ 근본적 한계 2: 의미론적 이해 부재

```
VEGA가 학습하는 것:
  - 코드의 "구문적 패턴"
  - 기존 백엔드 간의 "표면적 유사성"

VEGA가 학습하지 못하는 것:
  - ISA의 의미론 (semantic)
  - 명령어의 정확한 동작
  - 메모리 일관성 모델
  - 예외 처리 규칙

예시:
  새 ISA가 unconventional한 메모리 모델을 가진다면?
  → VEGA는 기존 패턴을 복제할 뿐, 새 모델을 반영 못함
```

#### ❌ 근본적 한계 3: 정확도 문제

```
71.5% 함수 수준 정확도의 의미:
  
  100개 함수 중:
    ✓ 71-72개 정확
    ✗ 28-29개 부정확
  
  컴파일러에서의 영향:
    - 28개 함수가 틀리면 → 컴파일러 작동 불가
    - 각 함수를 수동 검토/수정 필요
    - "자동생성"의 의미가 퇴색

결론: VEGA는 "초안 생성 도구"이지, 
      "완전한 백엔드 생성기"가 아님
```

#### 📊 VEGA 평가 요약

| 기준 | 평가 | 설명 |
|------|------|------|
| 입력 요구사항 | ❌ 불충분 | .td + 기존 백엔드 필요 |
| 출력 완전성 | △ 부분적 | C++ 함수만 (전체 백엔드 아님) |
| 정확성 보장 | ❌ 없음 | 71% 확률적 정확도 |
| 새 ISA 적용성 | ❌ 불가 | 유사 ISA 필수 |
| 실용성 | △ 제한적 | 수동 수정 필수 |

---

## 3. VeGen (ASPLOS 2021)

### 3.1 핵심 방법론

```
┌─────────────────────────────────────────────────────────────┐
│                      VeGen 워크플로우                         │
├─────────────────────────────────────────────────────────────┤
│  입력: Intel Intrinsics Guide (XML)                         │
│        ↓                                                    │
│  Pseudocode 파싱 → SMT 공식 변환                            │
│        ↓                                                    │
│  VIDL (Vector Instruction Description Language) 생성        │
│        ↓                                                    │
│  Pattern Matcher + Lane-Binding Functions 자동 생성         │
│        ↓                                                    │
│  SLP + Beam Search 기반 벡터화                              │
│        ↓                                                    │
│  SMT 검증                                                   │
│        ↓                                                    │
│  출력: 검증된 Vectorizer                                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 핵심 기여: Lane Level Parallelism (LLP)

```
Traditional SIMD:
  [A₀] [A₁] [A₂] [A₃]     [B₀] [B₁] [B₂] [B₃]
    +    +    +    +   =     
  [A₀+B₀] [A₁+B₁] [A₂+B₂] [A₃+B₃]
  
  규칙: 모든 레인에서 동일 연산, 레인 간 통신 없음

VeGen의 LLP:
  - Non-isomorphic: 다른 연산이 각 레인에서 실행 가능
  - Cross-lane: 레인 간 데이터 이동 허용

예시 (vpmaddwd):
  [A₀] [A₁] [A₂] [A₃]     [B₀] [B₁] [B₂] [B₃]
                    ↓
  [A₀*B₀ + A₁*B₁]   [A₂*B₂ + A₃*B₃]
```

### 3.3 평가 결과

| Benchmark | VeGen vs LLVM | VeGen vs ICC |
|-----------|---------------|--------------|
| x265 idct4 | 3x faster | - |
| TVM 2D Conv | 5x faster | 11x faster |
| AVX512-VNNI kernels | First support | - |

### 3.4 비판적 분석

#### ✅ 장점 1: Vendor Spec 직접 활용

```
VeGen의 핵심 통찰:
  "Intel Intrinsics Guide XML은 이미 기계 판독 가능한 형식"
  
장점:
  - 새로운 명세 언어 학습 불필요
  - Vendor가 유지보수하는 공식 문서 활용
  - 새 명령어 추가 시 즉시 지원 가능
  
이것이 중요한 이유:
  - VADL: 새 언어를 배워야 함
  - Isaria: Rosette 인터프리터를 작성해야 함
  - VeGen: XML만 주면 됨
```

#### ✅ 장점 2: SMT 기반 정확성 검증

```
VeGen의 검증 파이프라인:

1. Pseudocode → SMT 공식 변환
   "dst[i+31:i] := a[i+31:i] + b[i+31:i]"
   → ∀i: dst[i] = a[i] + b[i]

2. Pattern → SMT 공식 변환
   generated pattern의 의미론 추출

3. SMT Equivalence Check
   Z3.check(pattern_sem ≠ target_sem) = UNSAT
   → 정확성 증명

장점:
  - 100% semantic correctness 보장 (검증 통과 시)
  - VEGA의 71% vs VeGen의 100%
```

#### ❌ 한계 1: 벡터화만 담당

```
VeGen이 하는 것:
  ✓ Vector instruction selection
  ✓ Vector code generation
  
VeGen이 하지 않는 것:
  ✗ Register allocation
  ✗ Calling convention
  ✗ Frame lowering
  ✗ ABI handling
  ✗ Scalar instruction selection
  ✗ Memory operations (general)
  ✗ Control flow

결론: VeGen은 "완전한 백엔드"의 일부만 담당
```

#### ❌ 한계 2: Vendor XML 형식 필요

```
VeGen이 작동하려면:
  - Intel Intrinsics Guide XML (Intel)
  - ARM Intrinsics 명세 (ARM)
  - 등등...

문제:
  완전히 새로운 ISA의 경우:
  1. Vendor가 XML 형식 문서를 제공해야 함
  2. 일반 PDF 스펙만 있으면 작동 안 함
  3. 소규모 벤더/학술 프로젝트에서는 XML 없을 수 있음

현실적 제약:
  - 대형 벤더 (Intel, ARM, AMD): XML 제공
  - 소형 벤더 / 새 스타트업: PDF만 제공 가능
  - RISC-V 확장: 표준 XML 형식 없음
```

#### ❌ 한계 3: Basic Block 범위

```
VeGen의 범위:
  - 단일 Basic Block 내에서만 벡터화
  - 크로스 블록 최적화 없음

영향:
  - Loop 전체를 벡터화하려면 별도 처리 필요
  - 실제 컴파일러는 loop-level 벡터화 중요
```

#### 📊 VeGen 평가 요약

| 기준 | 평가 | 설명 |
|------|------|------|
| 입력 요구사항 | △ 특수 형식 | Vendor XML 필요 |
| 출력 완전성 | ❌ 부분적 | 벡터화만 |
| 정확성 보장 | ✅ 우수 | SMT 검증 100% |
| 새 ISA 적용성 | △ 조건부 | XML 제공 시 |
| 실용성 | ✅ 높음 | 실제 성능 향상 입증 |

---

## 4. Hydride (ASPLOS 2022)

### 4.1 핵심 방법론

```
┌─────────────────────────────────────────────────────────────┐
│                     Hydride 워크플로우                        │
├─────────────────────────────────────────────────────────────┤
│  입력: Vendor Pseudocode Specs (Intel XML, ARM 등)          │
│        ↓                                                    │
│  Formal Semantics 파싱 (Z3 기반)                            │
│        ↓                                                    │
│  Similarity Analysis: 여러 ISA 간 "유사" 연산 식별           │
│        ↓                                                    │
│  AutoLLVM IR 생성 (파라미터화된 공통 IR)                     │
│        ↓                                                    │
│  SyGuS (Syntax-Guided Synthesis)                            │
│        ↓                                                    │
│  출력: x86/ARM/Hexagon 타겟 코드                            │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 핵심 기여: Cross-ISA Abstraction

```
Hydride의 핵심 통찰:
  "여러 ISA의 명령어들은 파라미터만 다른 동일한 연산이다"

예시:
  x86 AVX:  vaddps ymm0, ymm1, ymm2  (8×float32 add)
  ARM NEON: fadd v0.4s, v1.4s, v2.4s (4×float32 add)
  
  공통 추상화:
  AutoLLVM: @autollvm.add(lanes=N, elem_type=float32)
  
압축률:
  ┌────────────────┬──────────┬─────────────┬───────────┐
  │ Architecture   │ ISA Size │ AutoLLVM    │ Reduction │
  ├────────────────┼──────────┼─────────────┼───────────┤
  │ x86            │ 2,029    │ 136         │ 6.7%      │
  │ HVX            │ 307      │ 115         │ 37.5%     │
  │ ARM            │ 1,221    │ 177         │ 14.5%     │
  │ All Combined   │ 3,557    │ 397         │ 11.2%     │
  └────────────────┴──────────┴─────────────┴───────────┘
```

### 4.3 평가 결과

| Target | Hydride vs Halide Backend |
|--------|---------------------------|
| x86 | +8% |
| ARM | +3% |
| HVX | +100% |
| vs LLVM Backend | +12% (x86), +26% (ARM) |

### 4.4 비판적 분석

#### ✅ 장점 1: 진정한 Cross-ISA 추상화

```
Hydride가 발견하는 것:
  "x86의 vaddps와 ARM의 fadd는 본질적으로 같은 연산"

이것이 왜 강력한가:
  1. 새 ISA가 x86/ARM과 "유사"하면:
     → 기존 AutoLLVM IR 재사용 가능
     → 빠른 지원 가능
  
  2. Transfer Learning 효과:
     → 기존 ISA 지식이 새 ISA에 전이
     
  3. 유지보수 용이:
     → 공통 IR만 관리하면 됨
```

#### ✅ 장점 2: SyGuS 기반 정확성 보장

```
Hydride의 합성:
  
  입력: Halide IR expression
        max(maximum[...], input[...])
        
  출력: AutoLLVM IR
        @autollvm.max(lanes=64, bitwidth=8, signed=1)
        
  보장: SyGuS가 semantic equivalence 보장
        ∀ inputs: Halide_expr(inputs) = AutoLLVM_expr(inputs)
```

#### ❌ 한계 1: DSL 컴파일러 특화

```
Hydride가 잘하는 것:
  Halide → Target Code
  MLIR → Target Code
  (Domain-Specific Languages)

Hydride가 하지 않는 것:
  C/C++ → Target Code
  LLVM IR (general) → Target Code
  (범용 컴파일러)

이유:
  - SyGuS는 작은 expression에 효과적
  - 범용 C/C++의 복잡한 control flow 처리 어려움
  - DSL은 연산이 정형화되어 있어 합성 용이
```

#### ❌ 한계 2: 기존 ISA 지식 필요

```
Hydride의 전제:
  "새 ISA가 x86/ARM/HVX 중 하나와 유사해야 함"

문제 시나리오:
  완전히 새로운 ISA "RISC-Y":
  - 새로운 벡터 연산 방식
  - x86/ARM과 다른 메모리 모델
  - 비표준 레지스터 구조
  
  → Hydride의 similarity analysis가 실패
  → AutoLLVM IR에 매핑 안 됨
```

#### ❌ 한계 3: Synthesis 확장성

```
SyGuS의 한계:
  - 작은 윈도우 내에서만 합성
  - 큰 커널에서는 최적화 기회 놓침
  
현재 구현:
  "small window of input IR operations"
  
결과:
  일부 벤치마크에서 slowdown 발생
  → 더 큰 윈도우는 합성 시간 폭증
```

#### 📊 Hydride 평가 요약

| 기준 | 평가 | 설명 |
|------|------|------|
| 입력 요구사항 | △ 특수 형식 | Vendor Specs 필요 |
| 출력 완전성 | ❌ 부분적 | DSL 컴파일러만 |
| 정확성 보장 | ✅ 우수 | SyGuS 100% |
| 새 ISA 적용성 | △ 조건부 | 유사 ISA 필요 |
| 실용성 | ✅ 높음 | 실제 성능 향상 |

---

## 5. ACT (arXiv 2025)

### 5.1 핵심 방법론

```
┌─────────────────────────────────────────────────────────────┐
│                      ACT 워크플로우                          │
├─────────────────────────────────────────────────────────────┤
│  입력: ISA Description (Tensor IR 형식)                     │
│        ↓                                                    │
│  ISA → IR-to-ISA Rewrite Rules 변환                        │
│        ↓                                                    │
│  Parameterized Equality Saturation (Instruction Selection)  │
│        ↓                                                    │
│  Constraint Programming (Memory Allocation)                 │
│        ↓                                                    │
│  출력: Sound & Complete 백엔드                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 평가 결과

- Intel AMX, Gemmini: hand-optimized 대비 동등 또는 1.77x 향상
- 컴파일 시간: < 1초 (311ms for 390 nodes)
- Sound & Complete 수학적 증명 제공

### 5.3 비판적 분석

#### ✅ 장점: 수학적 정확성 보장

```
ACT의 핵심 강점:
  1. Soundness: 생성된 코드가 항상 올바름
  2. Completeness: 하드웨어가 지원하면 반드시 생성 가능
  
증명 방법:
  - Equality Saturation: 탐색 완전성
  - Constraint Programming: 할당 정확성
```

#### ❌ 근본적 한계: Tensor Accelerator 전용

```
ACT가 지원하는 것:
  ✓ Matrix multiply
  ✓ Convolution
  ✓ Tensor operations
  ✓ NPU/TPU workloads

ACT가 지원하지 않는 것:
  ✗ General control flow
  ✗ Scalar operations
  ✗ Exception handling
  ✗ System calls
  ✗ 범용 CPU 백엔드

결론: ACT는 "Tensor Accelerator 전용 도구"
      범용 ISA에는 적용 불가
```

#### 📊 ACT 평가 요약

| 기준 | 평가 | 설명 |
|------|------|------|
| 입력 요구사항 | △ 특수 형식 | Tensor IR ISA 필요 |
| 출력 완전성 | ✅ 완전 (도메인 내) | Tensor 연산 완전 지원 |
| 정확성 보장 | ✅ 최고 | Sound & Complete 증명 |
| 새 ISA 적용성 | ❌ 불가 | Tensor Accel만 |
| 실용성 | ✅ 높음 (도메인 내) | 실제 HW에서 검증 |

---

## 6. Isaria (ASPLOS 2024)

### 6.1 핵심 방법론

```
┌─────────────────────────────────────────────────────────────┐
│                     Isaria 워크플로우                         │
├─────────────────────────────────────────────────────────────┤
│  입력: ISA Specification (Rosette 인터프리터) + Cost Model   │
│        ↓                                                    │
│  Ruler로 Rewrite Rules 자동 합성                            │
│        ↓                                                    │
│  Phase Discovery (Expansion→Compilation→Optimization)       │
│        ↓                                                    │
│  Equality Saturation with Pruning                           │
│        ↓                                                    │
│  출력: DSP Vectorizing Compiler                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 평가 결과

- Tensilica SDK 대비 최대 6.9x 향상
- Clang auto-vectorizer 대비 최대 25x 향상
- Diospyros (수작업) 대비 평균 34% 향상

### 6.3 비판적 분석

#### ✅ 장점: 완전 자동 규칙 합성

```
Isaria의 핵심:
  "Rosette 인터프리터만 주면 rewrite rules 자동 생성"

기존 (Diospyros):
  - 28개 규칙을 수작업 작성
  - 각 규칙의 scheduling 수작업
  
Isaria:
  - 7,735개 후보 규칙 자동 생성
  - 300개로 자동 필터링
  - Phase 자동 발견
```

#### ❌ 한계 1: Rosette 인터프리터 필요

```
Isaria 사용을 위해 필요한 것:
  1. Rosette로 ISA 인터프리터 작성 (73 LoC in paper)
  2. Cost function 작성 (90 LoC)
  
문제:
  - Rosette 학습 필요
  - 인터프리터 작성 = ISA의 완전한 이해 필요
  - 사실상 반-수작업
  
비교:
  - VeGen: Intel XML (이미 존재)
  - Hydride: Vendor specs (이미 존재)
  - Isaria: Rosette 코드 (새로 작성해야 함)
```

#### ❌ 한계 2: 벡터화만, 확장성 문제

```
Isaria가 하는 것:
  ✓ DSP 벡터화

Isaria가 하지 않는 것:
  ✗ 완전한 백엔드
  ✗ Register allocation
  ✗ Scalar instruction selection

확장성 문제:
  - 오프라인 규칙 생성: 최대 220GiB 메모리
  - 타임아웃: 1일
  - 컴파일 시간: Diospyros 대비 2.1x 느림
```

#### 📊 Isaria 평가 요약

| 기준 | 평가 | 설명 |
|------|------|------|
| 입력 요구사항 | ❌ 높음 | Rosette 코드 작성 필요 |
| 출력 완전성 | ❌ 부분적 | 벡터화만 |
| 정확성 보장 | ✅ 우수 | Equality Saturation |
| 새 ISA 적용성 | △ 조건부 | 인터프리터 작성 시 |
| 실용성 | △ 제한적 | 확장성 문제 |

---

## 7. OpenVADL (arXiv 2024)

### 7.1 핵심 방법론

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenVADL 워크플로우                        │
├─────────────────────────────────────────────────────────────┤
│  입력: VADL 명세 (ISA + MiA + ABI)                          │
│        ↓                                                    │
│  VIR/VIAM 중간 표현 변환                                     │
│        ↓                                                    │
│  Pattern Inference (Operational Semantics → DFG → TableGen) │
│        ↓                                                    │
│  출력: LLVM 백엔드 + 어셈블러 + 시뮬레이터 + HDL             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 평가 결과

- Embench 22개 중 18개 벤치마크 컴파일 성공
- Upstream LLVM 대비 ~6% 명령어 수 증가
- -O0 최적화 수준만 지원

### 7.3 비판적 분석

#### ✅ 장점: 가장 완전한 출력

```
OpenVADL이 생성하는 것:
  ✓ LLVM 백엔드 (완전)
  ✓ 어셈블러
  ✓ 링커
  ✓ 시뮬레이터
  ✓ HDL (하드웨어 기술)
  ✓ 문서

이것이 중요한 이유:
  - 다른 모든 연구: 백엔드의 일부만 생성
  - OpenVADL: 완전한 툴체인 생성
```

#### ❌ 근본적 한계: VADL 명세 = 백엔드 설계

```
VADL 명세 예시:
  instruction set architecture RV32I = {
      register file X : Bits<32>[32] [ 0 -> 0 ]
      format R : Bits<32> = { 
          funct7: Bits<7>, rs2: Bits<5>, rs1: Bits<5>,
          funct3: Bits<3>, rd: Bits<5>, opcode: Bits<7> 
      }
      instruction ADD : R = { 
          semantics = X[rd] <- X[rs1] + X[rs2]
          encoding = { funct7 = 0b0000000, funct3 = 0b000, ... }
      }
  }

문제:
  - 이것을 작성하려면 ISA를 완전히 이해해야 함
  - 사실상 백엔드를 "다른 언어로" 작성하는 것
  - "자동생성"의 의미가 희석됨
  
비교:
  ISA 스펙 문서 → VADL 명세 작성 → 백엔드 생성
  ISA 스펙 문서 → LLVM 백엔드 직접 작성
  
  둘의 노력이 유사할 수 있음
```

#### ❌ 한계 2: -O0만 지원

```
현재 상태:
  - 최적화 수준 -O0만 지원
  - -O2, -O3는 미지원

실제 프로덕션에서:
  - -O0: 디버깅용, 성능 낮음
  - -O2/-O3: 프로덕션 필수
  
결론: 현재 OpenVADL은 프로덕션 사용 불가
```

#### 📊 OpenVADL 평가 요약

| 기준 | 평가 | 설명 |
|------|------|------|
| 입력 요구사항 | ❌ 높음 | VADL 명세 작성 필요 |
| 출력 완전성 | ✅ 최고 | 완전한 툴체인 |
| 정확성 보장 | △ 제한적 | 일부 벤치마크 실패 |
| 새 ISA 적용성 | △ 조건부 | VADL 작성 시 |
| 실용성 | ❌ 제한적 | -O0만 지원 |

---

## 8. 종합 비교 및 Gap 분석

### 8.1 종합 비교표

| 측면 | VEGA | VeGen | Hydride | ACT | Isaria | OpenVADL |
|------|------|-------|---------|-----|--------|----------|
| **연도** | 2025 | 2021 | 2022 | 2025 | 2024 | 2024 |
| **입력 형식** | .td + 기존 BE | Vendor XML | Vendor Specs | Tensor IR | Rosette | VADL |
| **출력 범위** | C++ 함수 | Vectorizer | DSL 컴파일러 | Tensor BE | Vectorizer | 완전 BE |
| **정확성** | 71% | 100% (SMT) | 100% (SyGuS) | 100% (증명) | 100% (E-graph) | ~82% |
| **새 ISA** | ❌ | △ | △ | ❌ | △ | △ |
| **최적화** | - | 높음 | 높음 | 높음 | 높음 | -O0만 |

### 8.2 "새로운 ISA" 시나리오 분석

```
시나리오: 완전히 새로운 ISA "RISC-Y" 출시
입력: ISA 스펙 문서 (PDF, 500페이지)

각 접근법의 현실적 적용:

VEGA: ❌ 불가능
├── 이유: 유사한 기존 백엔드 없음
└── 필요: .td 파일 수동 작성 + 학습 데이터

VeGen: △ 조건부 가능
├── 조건: RISC-Y 벤더가 XML 형식 명세 제공
├── 범위: 벡터 명령어만
└── 나머지: 다른 방법 필요

Hydride: △ 조건부 가능
├── 조건: RISC-Y가 x86/ARM과 어느 정도 유사
├── 범위: DSL 컴파일러만
└── 문제: 완전히 새로운 연산 지원 불가

ACT: ❌ 범위 밖
├── 이유: Tensor Accelerator 아님
└── 결론: 범용 CPU에 적용 불가

Isaria: △ 조건부 가능
├── 필요: Rosette 인터프리터 작성 (수작업)
├── 범위: 벡터화만
└── 문제: 확장성 (220GB 메모리)

OpenVADL: △ 가장 가까움
├── 필요: VADL 명세 작성 (수작업)
├── 장점: 완전한 백엔드 생성
└── 문제: VADL 작성 ≈ 백엔드 개발

결론: 현재 어느 연구도 
      "ISA 스펙만으로 완전한 백엔드 생성"을 
      완전히 해결하지 못함
```

### 8.3 근본적 Gap 분석

```
┌─────────────────────────────────────────────────────────────┐
│                    Gap 1: ISA 스펙 파싱                      │
├─────────────────────────────────────────────────────────────┤
│ 현재: 모든 연구가 특정 형식의 입력 요구                       │
│       (XML, VADL, Rosette, Tensor IR)                       │
│                                                             │
│ 필요: PDF/자연어 ISA 스펙 직접 처리                          │
│                                                             │
│ 도전: ISA 스펙이 수천 페이지 → LLM으로 한번에 처리 불가       │
│                                                             │
│ 해결책 방향:                                                 │
│   - VeGen/Hydride: Vendor XML 활용 (좋은 시작점)            │
│   - 구조적 파싱: 섹션별 점진적 처리                          │
│   - LLM + 형식적 방법 결합                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Gap 2: 의미론적 이해                        │
├─────────────────────────────────────────────────────────────┤
│ 현재: 구문적 패턴 매칭 또는 명시적 의미론 기술 필요            │
│                                                             │
│ VeGen의 접근 (좋은 시작점):                                  │
│   Pseudocode → SMT 공식 자동 변환                           │
│                                                             │
│ Hydride의 접근 (좋은 시작점):                                │
│   Cross-ISA similarity로 의미론 추론                        │
│                                                             │
│ 아직 부족한 것:                                              │
│   - 자연어 설명에서 의미론 추출                              │
│   - 암묵적 제약 조건 발견                                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Gap 3: 출력 완전성                        │
├─────────────────────────────────────────────────────────────┤
│ VeGen/Hydride/Isaria: 벡터화만                              │
│ ACT: Tensor만                                               │
│ VEGA: 일부 함수만                                           │
│ OpenVADL: 완전하지만 -O0                                    │
│                                                             │
│ 필요:                                                        │
│   - 완전한 LLVM 백엔드                                      │
│   - -O2/-O3 최적화 수준                                     │
│   - 모든 ABI/calling convention                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Gap 4: 정확성 검증                       │
├─────────────────────────────────────────────────────────────┤
│ VEGA: 71% 확률적 (불충분)                                    │
│ VeGen: SMT 검증 (우수, 벡터화에 한정)                        │
│ Hydride: SyGuS (우수, DSL에 한정)                           │
│ ACT: 수학적 증명 (최고, Tensor에 한정)                       │
│                                                             │
│ 필요:                                                        │
│   - 전체 백엔드에 대한 100% 검증                             │
│   - VeGen의 SMT 검증을 전체 백엔드로 확장                    │
└─────────────────────────────────────────────────────────────┘
```

### 8.4 제안하는 연구 방향

```
VeGen + Hydride + VADL의 장점 결합:

Phase 1: Scalable ISA Extraction
├── PDF: 구조적 파싱 (섹션별)
├── XML: VeGen 방식 직접 활용
└── 출력: ISA Model

Phase 2: Hybrid Backend Synthesis
├── VeGen 방식: 벡터 명령어
├── Hydride 방식: 유사 패턴 전이
├── VADL 방식: 의미론 기반 추론
└── 출력: 백엔드 구성 요소

Phase 3: Verification-Guided Refinement
├── VeGen SMT 검증 확장
├── CGNR 반례 기반 수정
└── 출력: 검증된 백엔드
```

---

## 참고문헌

1. Zhong et al., "VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model", CGO 2025
2. Chen et al., "VeGen: A Vectorizer Generator for SIMD and Beyond", ASPLOS 2021
3. Hydride Project, "Retargetable Compiler IR Generation", ASPLOS 2022
4. Jain et al., "ACT: Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions", arXiv 2025
5. Thomas and Bornholt, "Isaria: Automatic Generation of Vectorizing Compilers for Customizable DSPs", ASPLOS 2024
6. Freitag et al., "The Vienna Architecture Description Language", arXiv 2024
