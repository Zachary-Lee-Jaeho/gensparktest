# 컴파일러 백엔드 자동생성 연구 분석

## 저명한 백엔드 자동생성 연구자 관점의 비판적 리뷰

**작성일**: 2026-01-23  
**분석 범위**: VEGA (CGO 2025), ACT (arXiv 2025), Isaria (ASPLOS 2024), OpenVADL (arXiv 2024), **VeGen (ASPLOS 2021)**, **Hydride (ASPLOS 2022)**

---

## 1. 연구 목표 맥락

### 우리의 목표
> 새로운 하드웨어와 ISA가 출시되었을 때, **ISA 스펙 정도의 간단한 정보만으로** 해당 새로운 하드웨어의 백엔드 컴파일러를 자동생성

### 핵심 요구사항
1. **입력**: ISA 스펙 (명령어 정의, 레지스터, 인코딩)
2. **출력**: 완전한 LLVM 백엔드 (또는 유사한 컴파일러 백엔드)
3. **전제**: 기존 컴파일러가 없는 완전히 새로운 ISA

### ISA 스펙 문서의 현실
> ⚠️ **중요**: 실제 ISA 스펙 문서는 매우 방대함
> - RISC-V Unprivileged Spec: ~150 페이지
> - ARM Architecture Reference Manual: ~8,000+ 페이지
> - Intel SDM (Software Developer Manual): ~5,000+ 페이지
> 
> LLM의 context window로 전체를 처리하기 어려움 → **구조적 파싱 + 점진적 처리 필요**

---

## 2. 각 접근법 상세 분석

### 2.1 VEGA (CGO 2025) - Transformer 기반 코드 생성

#### 방법론
```
입력: Target Description Files (.td)
     ↓
함수 그룹화: 기존 백엔드에서 동일 인터페이스 함수들 수집
     ↓
Feature Vector 추출: Target-specific vs Target-independent 구분
     ↓
Pre-trained Transformer (UniXcoder) Fine-tuning
     ↓
출력: 새 타겟용 함수 구현체
```

#### 핵심 아이디어
- **Function Template Abstraction**: 기존 백엔드(ARM, MIPS, X86 등)에서 동일한 인터페이스를 가진 함수들을 수집
- **Feature Vector Mapping**: 각 statement를 feature vector로 변환하여 target-specific/independent 속성 구분
- **Confidence Score**: 생성된 코드의 신뢰도 점수 제공

#### 평가 결과
| Target | Function-level Accuracy | Statement-level Accuracy |
|--------|------------------------|-------------------------|
| RISC-V | 71.5% | 55.0% |
| RI5CY | 73.2% | 58.5% |
| xCORE | 62.2% | - |

#### ⚠️ 비판적 리뷰 (저명한 연구자 관점)

**근본적 한계 1: "기존 백엔드 의존성"**
> VEGA는 ARM, MIPS 등 **기존 백엔드가 이미 존재해야만** 학습할 수 있다. 완전히 새로운 ISA에 대해서는 유사한 기존 백엔드가 없으면 Feature Vector 추출 자체가 불가능하다.

```
문제: 새로운 ISA "RISC-Y"가 출시됨
VEGA 요구사항: RISC-Y와 유사한 기존 백엔드 필요
현실: 완전히 새로운 아키텍처면 유사한 것이 없음
```

**근본적 한계 2: "코드 패턴 복제, 의미론적 이해 부재"**
> VEGA는 기존 코드의 **구문적 패턴**을 학습할 뿐, ISA의 **의미론적 특성**을 이해하지 못한다. 예를 들어, 새로운 ISA가 기존과 다른 메모리 일관성 모델을 가진다면 이를 반영할 수 없다.

**근본적 한계 3: "Target Description 파일 수동 작성 필요"**
> VEGA는 `.td` 파일이 이미 존재한다고 가정한다. 그러나 새로운 ISA의 경우 이 `.td` 파일 자체를 작성하는 것이 백엔드 개발의 상당 부분을 차지한다.

**정확도 문제**
> 71.5%의 함수 수준 정확도는 **나머지 28.5%를 수동으로 수정해야 함**을 의미한다. 컴파일러에서 28.5%의 함수가 틀리면 사실상 작동하지 않는 컴파일러이다.

---

### 2.2 VeGen (ASPLOS 2021) - Vectorizer Generator ⭐

**GitHub**: https://github.com/ychen306/vegen

#### 방법론
```
입력: Instruction Semantics (Intel Intrinsics Guide 등)
     ↓
VIDL (Vector Instruction Description Language)로 변환
     ↓
Pattern Matcher + Lane-Binding Functions 자동 생성
     ↓
Target-Independent Vectorization Algorithm (SLP + Beam Search)
     ↓
출력: Non-SIMD 명령어까지 지원하는 벡터라이저
```

#### 핵심 아이디어
- **Lane Level Parallelism (LLP)**: SIMD를 넘어서는 새로운 병렬성 모델
  - Non-isomorphic operations (다른 연산이 각 레인에서 실행 가능)
  - Cross-lane communication (레인 간 데이터 이동)
- **VIDL**: 명령어 의미론을 스칼라 연산 리스트 + 레인 바인딩 규칙으로 기술
- **Intel Intrinsics Guide에서 자동 추출**: XML에서 pseudocode를 파싱하여 SMT 공식으로 변환

#### 평가 결과
- x265 idct4 커널: LLVM 대비 **3x 속도 향상**
- TVM 2D Conv 커널: ICC/GCC/LLVM 대비 **11x 속도 향상** (AVX512-VNNI 사용)
- 새 명령어(vpdpbusd 등) 자동 지원

#### ⚠️ 비판적 리뷰

**장점: "가장 실용적인 의미론 기반 접근"**
> VeGen은 Intel Intrinsics Guide라는 **이미 존재하는 기계 판독 가능 형식**에서 의미론을 추출한다. 새로운 명세 언어를 만들 필요가 없다.

**근본적 한계 1: "벡터화만 담당"**
> VeGen은 **벡터 명령어 선택**에만 집중한다. 완전한 백엔드(register allocation, calling convention, frame lowering 등)는 별도로 필요하다.

**근본적 한계 2: "Side-effect 없는 명령어만"**
> VIDL은 레지스터 읽기/쓰기만 모델링. 복잡한 side-effect가 있는 명령어(예: 메모리 배리어, 시스템 호출)는 지원 불가.

**근본적 한계 3: "Basic Block 범위"**
> 현재 구현은 기본 블록 내에서만 벡터화. 크로스 블록 벡터화 미지원.

**근본적 한계 4: "새 ISA의 Intrinsics Guide가 필요"**
> 완전히 새로운 ISA의 경우, Intel Intrinsics Guide 같은 기계 판독 가능 형식의 문서가 필요하다. ISA 스펙 PDF만으로는 동작하지 않는다.

---

### 2.3 Hydride (ASPLOS 2022) - Retargetable IR Generation ⭐

**Website**: https://hydride.cs.illinois.edu/

#### 방법론
```
입력: Vendor Pseudocode Specifications (Intel XML, ARM 등)
     ↓
Formal Semantics로 파싱 (Z3 기반)
     ↓
Similarity Analysis: 여러 ISA 간 "유사한" 연산 식별
     ↓
AutoLLVM IR 생성: 공통 추상화 IR
     ↓
SyGuS (Syntax-Guided Synthesis)로 코드 합성
     ↓
출력: x86/ARM/Hexagon 타겟 코드
```

#### 핵심 아이디어
- **AutoLLVM IR**: 여러 ISA의 공통점을 추상화한 중간 표현
  - 3,557개 ISA 명령어 → **397개 AutoLLVM 명령어**로 압축 (11.2%)
  - 파라미터화된 명령어 (비트폭, 레인 수 등)
- **Similarity Analysis**: 다른 ISA에서 "의미적으로 유사한" 명령어 자동 식별
- **SyGuS for Code Synthesis**: Halide/MLIR → AutoLLVM IR 자동 합성

#### 평가 결과
| Architecture | ISA Size | AutoLLVM Size | Reduction |
|-------------|----------|---------------|-----------|
| x86 | 2,029 | 136 | 6.7% |
| HVX | 307 | 115 | 37.5% |
| ARM | 1,221 | 177 | 14.5% |
| **All Combined** | 3,557 | 397 | **11.2%** |

- Halide 컴파일러 대비: x86 +8%, ARM +3%, HVX +100%

#### ⚠️ 비판적 리뷰

**장점: "진정한 Cross-ISA 추상화"**
> Hydride는 여러 ISA에 걸쳐 공통 패턴을 자동으로 발견한다. 이는 새 ISA가 기존 ISA와 어느 정도 유사할 경우 매우 유용하다.

**장점: "Vendor Pseudocode 직접 활용"**
> Intel XML, ARM 명세 등 **벤더가 이미 제공하는 형식**을 활용. 새 언어를 배울 필요 없음.

**근본적 한계 1: "DSL 컴파일러에 특화"**
> Hydride는 **Halide 같은 DSL**에서 타겟 코드로 가는 경로에 최적화. 범용 C/C++ 컴파일러의 전체 백엔드는 다루지 않음.

**근본적 한계 2: "기존 ISA 지식 필요"**
> 완전히 새로운 ISA의 경우, "유사한" 기존 ISA가 없으면 AutoLLVM IR 생성이 제한적. **x86/ARM/Hexagon 지식을 전제**로 함.

**근본적 한계 3: "SyGuS 확장성"**
> 합성 윈도우가 작아서 큰 커널에서는 최적화 기회를 놓칠 수 있음.

**근본적 한계 4: "Vendor Specification 형식 의존"**
> Intel XML, ARM 명세 등 **기계 판독 가능한 형식**이 필요. 일반 PDF ISA 스펙에서는 동작하지 않음.

---

### 2.4 ACT (arXiv 2025) - Tensor Accelerator ISA 기반 생성

#### 방법론
```
입력: ISA Description (Tensor IR 형식)
     ↓
ISA → IR-to-ISA Rewrite Rules 변환
     ↓
Phase 1: Parameterized Equality Saturation (Instruction Selection)
     ↓
Phase 2: Constraint Programming (Memory Allocation)
     ↓
출력: 백엔드 코드
```

#### 핵심 아이디어
- **Formal ISA Specification**: Tensor IR와 동일한 operator language로 ISA 기술
- **Parameterized Instructions**: 가변 크기 행렬 연산 등 복잡한 명령어 지원
- **Sound & Complete 보장**: 수학적으로 정확성 증명

#### 평가 결과
- Intel AMX, Gemmini에서 hand-optimized 라이브러리와 동등 또는 1.77x 향상
- 컴파일 시간 < 1초 (311ms for 390 nodes)

#### ⚠️ 비판적 리뷰

**근본적 한계 1: "Tensor Accelerator 전용"**
> ACT는 **Tensor IR**를 중심으로 설계되어, 범용 CPU/GPU 백엔드에는 적용 불가. 일반적인 제어 흐름, 복잡한 메모리 연산, 예외 처리 등을 다룰 수 없다.

**근본적 한계 2: "ISA Specification 형식의 제약"**
> ACT의 ISA specification은 Tensor computation에 최적화되어 있어, 일반 ISA 스펙(예: RISC-V 공식 스펙 문서)을 바로 사용할 수 없다.

**적용 가능 범위**
```
ACT 적용 가능: NPU, TPU, Matrix Extensions
ACT 적용 불가: 범용 CPU 백엔드, GPU compute shaders, DSP 일반 연산
```

---

### 2.5 Isaria (ASPLOS 2024) - DSP 벡터라이징 컴파일러 자동생성

#### 방법론
```
입력: ISA Specification (Rosette 인터프리터) + Cost Model
     ↓
Ruler로 Rewrite Rules 합성
     ↓
Phase Discovery (Expansion → Compilation → Optimization)
     ↓
Equality Saturation with Pruning
     ↓
출력: 벡터라이징 컴파일러
```

#### 핵심 아이디어
- **Automatic Rule Synthesis**: ISA 인터프리터에서 rewrite rules 자동 합성
- **Phase-Oriented Scheduling**: 규칙을 단계별로 분류하여 탐색 공간 축소
- **Cost-Based Extraction**: 추상 비용 모델로 최적 코드 추출

#### 평가 결과
- Tensilica SDK 대비 최대 6.9x, clang auto-vectorizer 대비 최대 25x 향상
- Diospyros(수작업) 대비 평균 34% 향상

#### ⚠️ 비판적 리뷰

**근본적 한계 1: "벡터화 전용"**
> Isaria는 DSP **벡터화**에만 집중한다. 일반적인 instruction selection, register allocation, calling convention 등은 다루지 않는다.

**근본적 한계 2: "ISA Interpreter 필요"**
> Rosette로 작성된 **실행 가능한 ISA 인터프리터**가 필요하다. 이 인터프리터 작성 자체가 상당한 노력을 요구한다.

**근본적 한계 3: "Lane-wise Instructions만 지원"**
> 크로스-레인 연산(예: reduction, shuffle)이 많은 ISA에서는 규칙 합성이 복잡해지고, 검증이 어렵다.

**확장성 문제**
```
오프라인 규칙 생성: 최대 220GiB 메모리, 1일 타임아웃
컴파일 시간: 평균 2.1x 느림 (Diospyros 대비)
```

---

### 2.6 OpenVADL (arXiv 2024) - Architecture Description Language 기반

#### 방법론
```
입력: VADL 명세 (ISA + MiA + ABI)
     ↓
VIR/VIAM 중간 표현 변환
     ↓
Pattern Inference (Operational Semantics → DFG → TableGen)
     ↓
출력: LLVM 백엔드 + 어셈블러 + 시뮬레이터 + HDL
```

#### 핵심 아이디어
- **Single Source of Truth**: 하나의 VADL 명세로 모든 아티팩트 생성
- **ISA/MiA 분리**: 명령어 집합과 마이크로아키텍처 독립적 기술
- **Operational Semantics → Pattern**: 동작 의미론에서 명령어 패턴 자동 추론

#### 평가 결과
- Embench 22개 중 18개 벤치마크 컴파일 성공
- upstream LLVM 대비 ~6% 명령어 수 증가

#### ⚠️ 비판적 리뷰

**근본적 한계 1: "VADL 명세 작성 비용"**
> VADL 명세를 작성하는 것 자체가 **전문 지식**을 요구한다. 새로운 ISA의 경우 VADL 명세 작성이 백엔드 개발과 비슷한 수준의 노력을 필요로 할 수 있다.

```vadl
// VADL 명세 예시 - 이것 자체가 복잡함
instruction set architecture RV32I = {
    register file X : Bits<32>[32] [ 0 -> 0 ]
    format R : Bits<32> = { funct7: Bits<7>, rs2: Bits<5>, ... }
    instruction ADD : R = { 
        semantics = X[rd] <- X[rs1] + X[rs2]
        encoding = { funct7 = 0b0000000, funct3 = 0b000, ... }
    }
}
```

**근본적 한계 2: "-O0만 지원"**
> 현재 구현은 **최적화 수준 -O0**만 지원한다. 실제 프로덕션 컴파일러는 -O2, -O3가 필수적이다.

**근본적 한계 3: "Multiple-result Instructions 미지원"**
> TableGen의 한계로, **다중 결과 명령어**(예: 일부 DSP의 동시 곱셈-덧셈)를 자동 생성할 수 없다.

**패턴 커버리지**
> 자동 추론된 패턴이 모든 경우를 커버하지 못해 일부 벤치마크가 컴파일 실패한다.

---

## 3. 종합 비교 분석

### 3.1 접근법 비교표

| 측면 | VEGA | VeGen | Hydride | ACT | Isaria | OpenVADL |
|------|------|-------|---------|-----|--------|----------|
| **입력 요구사항** | .td 파일 + 기존 백엔드 | Intel XML 등 | Vendor Specs | Tensor IR ISA | Rosette 인터프리터 | VADL 명세 |
| **출력** | C++ 함수 | Vectorizer | DSL 컴파일러 | 백엔드 코드 | Vectorizing compiler | LLVM 백엔드 전체 |
| **적용 범위** | 일반 ISA | Vector 명령어 | DSL (Halide) | Tensor Accelerator | DSP 벡터화 | 범용 ISA |
| **정확성 보장** | ❌ (확률적) | ✅ (SMT 검증) | ✅ (SyGuS) | ✅ (수학적 증명) | ✅ (검증 기반) | △ (제한적) |
| **새 ISA 적용** | △ (유사 ISA 필요) | △ (XML 필요) | △ (유사 ISA 필요) | △ (Tensor only) | △ (인터프리터 필요) | △ (VADL 필요) |
| **학습 필요** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **최적화 수준** | - | 높음 | 높음 | 높음 | 높음 | -O0만 |

### 3.2 "새로운 ISA" 시나리오 적합성

```
시나리오: 완전히 새로운 ISA "RISC-Y" 출시
입력: ISA 스펙 문서 (PDF/텍스트, 수천 페이지)

각 접근법의 현실적 적용:

VEGA: ❌ 불가능
  - 이유: RISC-Y와 유사한 기존 백엔드 없음
  - .td 파일을 누군가 먼저 작성해야 함

VeGen: △ 부분적
  - 이유: 벡터화만 담당, 전체 백엔드 아님
  - Intel XML 같은 기계 판독 형식 필요
  + 장점: 벤더가 XML 제공 시 즉시 벡터화 지원

Hydride: △ 부분적
  - 이유: DSL 컴파일러 특화, 전체 백엔드 아님
  - 새 ISA가 x86/ARM과 유사해야 AutoLLVM IR 활용 가능
  + 장점: 유사 ISA 있으면 빠르게 적용

ACT: ❌ 범위 밖
  - 이유: RISC-Y가 Tensor Accelerator가 아닌 한 적용 불가
  - 범용 CPU 명령어 지원 안 함

Isaria: △ 부분적
  - 이유: 벡터화만 담당, 전체 백엔드 아님
  - Rosette 인터프리터 작성 필요 (상당한 노력)

OpenVADL: △ 가장 가까움
  - 이유: ISA를 VADL로 변환하면 백엔드 생성 가능
  - 문제: VADL 명세 작성 = 사실상 백엔드 설계
  + 장점: 완전한 백엔드 생성 가능
```

---

## 4. 근본적 Gap 분석

### 4.1 현재 연구들이 해결하지 못한 문제

```
목표: ISA 스펙 → 백엔드 컴파일러

Gap 1: ISA 스펙 파싱
  - 현재: 모든 연구가 특정 형식의 입력 요구 (XML, VADL, Rosette 등)
  - 필요: PDF/자연어 ISA 스펙 직접 처리
  - 현실: ISA 스펙은 수천 페이지, LLM context로 한번에 처리 불가

Gap 2: 의미론적 이해
  - 현재: 구문적 패턴 매칭 또는 명시적 의미론 기술 필요
  - 필요: ISA의 의미를 자동으로 이해하고 추론
  - VeGen/Hydride 접근: vendor pseudocode에서 SMT 공식 추출 (좋은 시작점)

Gap 3: 완전성
  - 현재: 71% 함수 정확도(VEGA), -O0만(VADL), 벡터화만(VeGen/Isaria)
  - 필요: 100% 작동하는 프로덕션 수준 백엔드

Gap 4: 검증
  - 현재: 제한적 테스트, 수학적 증명(ACT만)
  - VeGen: SMT로 패턴 검증 (좋은 접근)
  - Hydride: SyGuS로 정확성 보장 (좋은 접근)
```

### 4.2 왜 현재 접근법들이 부족한가?

**Machine Learning 기반 (VEGA)**:
> "기존 백엔드에서 패턴을 학습"하는 접근은 **새로운 ISA가 기존과 다른 특성**을 가질 때 실패한다. 예를 들어:
> - 새로운 addressing mode
> - 새로운 calling convention
> - 비표준 레지스터 구조
> - 특수 목적 명령어

**Formal 방법 기반 (ACT, Isaria, VADL)**:
> 명세를 작성하는 것 자체가 **백엔드 개발과 동등한 노력**을 요구한다.

**Vendor Spec 기반 (VeGen, Hydride)**:
> 가장 실용적이지만:
> - 기계 판독 가능한 명세 형식 필요 (Intel XML 등)
> - 전체 백엔드가 아닌 특정 영역(벡터화, DSL)에 특화
> - 완전히 새로운 ISA면 유사한 기존 패턴 없음

---

## 5. 논문 수준의 개선 아이디어

### 5.1 제안: "VERA" - Verified End-to-end Retargetable Architecture

#### 핵심 통찰

**VeGen + Hydride의 접근을 확장**:
> VeGen/Hydride가 보여준 것처럼, vendor pseudocode에서 formal semantics를 추출하는 것은 가능하다. 
> 그러나 이를 **전체 백엔드**로 확장하고, **PDF/자연어 스펙**까지 처리하려면 새로운 접근이 필요.

#### 제안 아키텍처
```
ISA 스펙 문서 (PDF/Vendor XML/자연어)
       ↓
[Phase 1: Structured ISA Extraction]
- PDF: 구조적 파싱 (섹션별 처리, LLM chunk processing)
- XML: VeGen/Hydride 방식 직접 활용
- 출력: ISA Model (레지스터, 인코딩, 의미론)
       ↓
[Phase 2: Compositional Backend Synthesis]
- VeGen 방식: 벡터 명령어 선택기 자동 생성
- Hydride 방식: 공통 IR 추상화
- VADL 방식: 패턴 추론
       ↓
[Phase 3: Counterexample-Guided Refinement]
- SMT-based 의미론 검증 (VeGen의 검증 확장)
- 반례 기반 자동 수정 (현재 CGNR 프레임워크)
       ↓
출력: 검증된 LLVM 백엔드
```

#### 핵심 기여 1: "Scalable ISA Document Processing"

**문제**: ISA 스펙이 수천 페이지 → LLM context로 한번에 처리 불가

**해결책**:
```python
class ScalableISAExtractor:
    """
    대용량 ISA 문서를 구조적으로 처리
    """
    def process_large_spec(self, spec_path: str) -> ISAModel:
        # 1. 문서 구조 분석 (목차, 섹션)
        structure = self.analyze_document_structure(spec_path)
        
        # 2. 섹션별 처리 (Register, Instruction, Encoding 분리)
        for section_type, pages in structure.items():
            if section_type == "registers":
                self.extract_registers(pages)
            elif section_type == "instructions":
                # 명령어별로 개별 처리
                for instr_pages in self.split_by_instruction(pages):
                    self.extract_instruction(instr_pages)
            elif section_type == "encoding":
                self.extract_encoding(pages)
        
        # 3. 크로스-레퍼런스 검증
        return self.validate_and_merge()
    
    def extract_instruction(self, pages: List[Page]) -> Instruction:
        """
        단일 명령어 추출 (LLM context 내 처리 가능)
        
        VeGen 방식 활용: pseudocode → SMT 공식
        """
        pseudocode = self.extract_pseudocode(pages)
        semantics = self.pseudocode_to_smt(pseudocode)  # VeGen 기법
        encoding = self.extract_encoding(pages)
        return Instruction(pseudocode, semantics, encoding)
```

#### 핵심 기여 2: "Hybrid Synthesis Pipeline"

**VeGen + Hydride + VADL 통합**:
```python
class HybridSynthesizer:
    """
    각 연구의 장점을 결합
    """
    def synthesize_backend(self, isa: ISAModel) -> LLVMBackend:
        backend = LLVMBackend()
        
        # 1. VeGen 방식: 벡터 명령어 선택기
        if isa.has_vector_instructions():
            backend.vectorizer = VeGenApproach.generate_vectorizer(
                isa.vector_instructions,
                isa.vector_semantics
            )
        
        # 2. Hydride 방식: 공통 패턴 활용
        if self.has_similar_isa(isa):
            similar_isa = self.find_most_similar(isa)
            backend.patterns = HydrideApproach.transfer_patterns(
                from_isa=similar_isa,
                to_isa=isa
            )
        
        # 3. VADL 방식: 의미론에서 패턴 추론
        for instr in isa.instructions:
            if not backend.has_pattern(instr):
                pattern = VADLApproach.infer_pattern(instr.semantics)
                backend.add_pattern(instr, pattern)
        
        return backend
```

#### 핵심 기여 3: "VeGen-style Verification Extended"

**VeGen의 SMT 검증을 전체 백엔드로 확장**:
```python
class ExtendedVerification:
    """
    VeGen의 검증 기법을 전체 백엔드에 적용
    """
    def verify_instruction_selection(self, isa: ISAModel, backend: LLVMBackend):
        """
        VeGen 방식: 각 패턴이 ISA 의미론과 일치하는지 SMT 검증
        """
        for pattern in backend.patterns:
            ir_semantics = self.compute_ir_semantics(pattern.ir_pattern)
            target_semantics = isa.get_semantics(pattern.target_instr)
            
            # SMT로 동치성 검증
            if not self.smt_check_equivalence(ir_semantics, target_semantics):
                counterexample = self.get_counterexample()
                yield VerificationFailure(pattern, counterexample)
    
    def repair_with_counterexample(self, failure: VerificationFailure) -> Pattern:
        """
        현재 CGNR 프레임워크 활용
        """
        return self.cgnr_repair(failure.pattern, failure.counterexample)
```

### 5.2 예상 기여도

| 기존 연구 | 한계 | VERA 개선 |
|----------|------|-----------|
| VEGA | 기존 백엔드 필요 | → Vendor spec에서 직접 추출 |
| VeGen | 벡터화만 | → 전체 백엔드로 확장 |
| Hydride | DSL 특화 | → 범용 C/C++ 컴파일러 지원 |
| ACT | Tensor only | → 범용 ISA 지원 |
| Isaria | Rosette 필요 | → PDF/XML 직접 처리 |
| VADL | 명세 작성 필요 | → 자동 명세 추출 |

### 5.3 실현 가능성 평가

```
단기 (6개월):
- VeGen의 Intel XML 파서를 다른 벤더(ARM, RISC-V)로 확장
- PDF 구조 분석 및 섹션별 처리 파이프라인
- 기본 검증 프레임워크 (VeGen SMT 기반)

중기 (1년):
- Hydride의 similarity analysis를 새 ISA에 적용
- 완전한 백엔드 구성 요소 합성 (register allocation, frame lowering)
- CGNR 기반 반복 수정

장기 (2년):
- 프로덕션 수준 -O2/-O3 최적화
- 다양한 ISA 타입 (CISC, RISC, DSP, Accelerator) 지원
- 완전 자동화 파이프라인
```

---

## 6. 현재 시스템(VEGA-Verified)에 대한 제안

### 6.1 VeGen/Hydride에서 배울 점

```
VeGen의 핵심 기법:
1. Vendor pseudocode → SMT 공식 변환
2. SMT 기반 패턴 검증
3. LLVM 통합 (IR-level 변환)

Hydride의 핵심 기법:
1. 여러 ISA 간 similarity analysis
2. Parameterized IR (AutoLLVM)
3. SyGuS 기반 코드 합성
```

### 6.2 현재 시스템에 통합 가능한 부분

```
현재: 버그 수정 (Neural Repair) + CGNR
     ↓
추가 1: VeGen 방식 의미론 추출
  - Intel XML 파서 도입
  - Pseudocode → SMT 변환
     ↓
추가 2: Hydride 방식 패턴 전이
  - 기존 ISA에서 유사 패턴 탐색
  - AutoLLVM-style 추상화
     ↓
추가 3: 검증 강화
  - VeGen SMT 검증 통합
  - CGNR로 자동 수정
```

### 6.3 구체적 구현 로드맵

**Phase 1: 데이터 파이프라인 개선**
```python
# 현재: 단순 버그-수정 쌍
training_data = [
    ("buggy_code", "fixed_code"),
]

# VeGen 방식 추가: Vendor spec에서 추출
training_data = [
    {
        "instruction_semantics": extract_from_intel_xml(instr),
        "llvm_pattern": extract_from_llvm_backend(instr),
        "verification_result": smt_verify(semantics, pattern)
    }
]
```

**Phase 2: 모델 목적 전환**
```
현재: CodeT5 (Seq2Seq)
      buggy → fixed

개선 방향:
1. VeGen 방식: semantics → pattern (rule-based, SMT-guided)
2. Hydride 방식: similar_pattern → new_pattern (transfer learning)
3. Neural: fallback for complex cases
```

**Phase 3: 검증 통합**
```
현재: 단순 테스트 기반 검증

개선:
- VeGen SMT 검증 도입
- 반례 기반 수정 (기존 CGNR 활용)
- Hydride SyGuS로 대안 합성
```

---

## 7. 결론

### 7.1 현재 연구 동향 요약
- **VEGA**: ML 기반이지만 기존 백엔드 의존
- **VeGen**: 가장 실용적인 의미론 기반, 벡터화 특화 ⭐
- **Hydride**: Cross-ISA 추상화, DSL 특화 ⭐
- **ACT**: 형식적이지만 Tensor only
- **Isaria**: 자동화되지만 벡터화 only
- **OpenVADL**: 완전하지만 명세 작성 필요

### 7.2 우리의 목표를 위한 Gap
> "ISA 스펙만으로 백엔드 생성"은 현재 **어느 연구도 완전히 해결하지 못함**
> 
> 그러나 **VeGen과 Hydride는 좋은 시작점**을 제공:
> - Vendor spec에서 formal semantics 추출 가능
> - Cross-ISA 패턴 전이 가능
> - SMT/SyGuS 기반 검증 가능

### 7.3 제안하는 방향

1. **Scalable ISA Processing**: 대용량 스펙 문서의 구조적 처리
2. **Hybrid Synthesis**: VeGen + Hydride + VADL 장점 결합
3. **Extended Verification**: VeGen SMT 검증을 전체 백엔드로 확장
4. **CGNR Integration**: 반례 기반 반복 수정

### 7.4 현실적 시작점
- **단기**: VeGen의 Intel XML 파서 확장, 다른 벤더 지원
- **중기**: Hydride의 similarity analysis 도입
- **장기**: 완전 자동화 파이프라인

---

## 참고문헌

1. Zhong et al., "VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model", CGO 2025
2. **Chen et al., "VeGen: A Vectorizer Generator for SIMD and Beyond", ASPLOS 2021** ⭐
3. **Hydride Project, "Retargetable Compiler IR Generation", ASPLOS 2022** ⭐
4. Jain et al., "ACT: Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions", arXiv 2025
5. Thomas and Bornholt, "Automatic Generation of Vectorizing Compilers for Customizable Digital Signal Processors", ASPLOS 2024
6. Freitag et al., "The Vienna Architecture Description Language", arXiv 2024
