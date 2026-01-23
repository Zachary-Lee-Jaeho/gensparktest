# 컴파일러 백엔드 자동생성 연구 분석

## 저명한 백엔드 자동생성 연구자 관점의 비판적 리뷰

**작성일**: 2026-01-23  
**분석 범위**: VEGA (CGO 2025), ACT (arXiv 2025), Isaria (ASPLOS 2024), OpenVADL (arXiv 2024)

---

## 1. 연구 목표 맥락

### 우리의 목표
> 새로운 하드웨어와 ISA가 출시되었을 때, **ISA 스펙 정도의 간단한 정보만으로** 해당 새로운 하드웨어의 백엔드 컴파일러를 자동생성

### 핵심 요구사항
1. **입력**: ISA 스펙 (명령어 정의, 레지스터, 인코딩)
2. **출력**: 완전한 LLVM 백엔드 (또는 유사한 컴파일러 백엔드)
3. **전제**: 기존 컴파일러가 없는 완전히 새로운 ISA

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

### 2.2 ACT (arXiv 2025) - Tensor Accelerator ISA 기반 생성

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

### 2.3 Isaria (ASPLOS 2024) - DSP 벡터라이징 컴파일러 자동생성

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

### 2.4 OpenVADL (arXiv 2024) - Architecture Description Language 기반

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

| 측면 | VEGA | ACT | Isaria | OpenVADL |
|------|------|-----|--------|----------|
| **입력 요구사항** | .td 파일 + 기존 백엔드 | Tensor IR ISA | Rosette 인터프리터 | VADL 명세 |
| **출력** | C++ 함수 | 백엔드 코드 | Vectorizing compiler | LLVM 백엔드 전체 |
| **적용 범위** | 일반 ISA | Tensor Accelerator | DSP 벡터화 | 범용 ISA |
| **정확성 보장** | ❌ (확률적) | ✅ (수학적 증명) | ✅ (검증 기반) | △ (제한적) |
| **새 ISA 적용** | △ (유사 ISA 필요) | △ (Tensor only) | △ (인터프리터 필요) | △ (VADL 필요) |
| **학습 필요** | ✅ | ❌ | ❌ | ❌ |
| **최적화 수준** | - | 높음 | 높음 | -O0만 |

### 3.2 "새로운 ISA" 시나리오 적합성

```
시나리오: 완전히 새로운 ISA "RISC-Y" 출시
입력: ISA 스펙 문서 (PDF/텍스트)

각 접근법의 현실적 적용:

VEGA: ❌ 불가능
  - 이유: RISC-Y와 유사한 기존 백엔드 없음
  - .td 파일을 누군가 먼저 작성해야 함

ACT: ❌ 범위 밖
  - 이유: RISC-Y가 Tensor Accelerator가 아닌 한 적용 불가
  - 범용 CPU 명령어 지원 안 함

Isaria: △ 부분적
  - 이유: 벡터화만 담당, 전체 백엔드 아님
  - Rosette 인터프리터 작성 필요

OpenVADL: △ 가장 가까움
  - 이유: ISA를 VADL로 변환하면 백엔드 생성 가능
  - 문제: VADL 명세 작성 = 사실상 백엔드 설계
```

---

## 4. 근본적 Gap 분석

### 4.1 현재 연구들이 해결하지 못한 문제

```
목표: ISA 스펙 → 백엔드 컴파일러

Gap 1: ISA 스펙 파싱
  - 현재: 모든 연구가 특정 형식의 입력 요구
  - 필요: 자연어/PDF ISA 스펙 직접 처리

Gap 2: 의미론적 이해
  - 현재: 구문적 패턴 매칭 또는 명시적 의미론 기술 필요
  - 필요: ISA의 의미를 자동으로 이해하고 추론

Gap 3: 완전성
  - 현재: 71% 함수 정확도, -O0만 지원 등
  - 필요: 100% 작동하는 프로덕션 수준 백엔드

Gap 4: 검증
  - 현재: 제한적 테스트, 수학적 증명(ACT만)
  - 필요: 생성된 백엔드의 정확성 자동 검증
```

### 4.2 왜 현재 접근법들이 부족한가?

**VEGA의 근본적 한계**:
> "기존 백엔드에서 패턴을 학습"하는 접근은 **새로운 ISA가 기존과 다른 특성**을 가질 때 실패한다. 예를 들어:
> - 새로운 addressing mode
> - 새로운 calling convention
> - 비표준 레지스터 구조
> - 특수 목적 명령어

**형식적 방법(ACT, Isaria, VADL)의 한계**:
> 명세를 작성하는 것 자체가 **백엔드 개발과 동등한 노력**을 요구한다.

---

## 5. 논문 수준의 개선 아이디어

### 5.1 제안: "VERA" - Verified End-to-end Retargetable Architecture

#### 핵심 아이디어
```
ISA 스펙 문서 (자연어/형식적)
       ↓
[Phase 1: LLM-based ISA Understanding]
ISA의 의미론적 특성 추출 + 형식적 모델 생성
       ↓
[Phase 2: Template-based Synthesis]
기존 백엔드 구조를 템플릿으로, ISA 특성 기반 인스턴스화
       ↓
[Phase 3: Counterexample-Guided Refinement]
SMT-based 검증 + 반례 기반 자동 수정
       ↓
출력: 검증된 LLVM 백엔드
```

#### 핵심 기여 1: "ISA Semantic Extraction"
```python
class ISASemanticExtractor:
    """
    ISA 스펙에서 의미론적 특성을 자동 추출
    """
    def extract_from_manual(self, isa_document: str) -> ISAModel:
        # LLM으로 ISA 문서 분석
        # 1. 레지스터 구조 추출
        # 2. 명령어 형식 추출
        # 3. 인코딩 규칙 추출
        # 4. 의미론 추출
        pass
    
    def extract_from_formal(self, formal_spec: str) -> ISAModel:
        # 형식적 명세에서 직접 추출
        pass
```

#### 핵심 기여 2: "Compositional Backend Synthesis"
```python
class CompositionalSynthesizer:
    """
    백엔드를 구성 요소별로 분리하여 합성
    """
    COMPONENTS = [
        "RegisterInfo",      # 레지스터 정보
        "InstructionInfo",   # 명령어 정보
        "AsmParser",         # 어셈블러 파서
        "AsmPrinter",        # 어셈블리 출력
        "MCCodeEmitter",     # 기계어 인코딩
        "InstructionSelect", # 명령어 선택
        "RegisterAlloc",     # 레지스터 할당
        "FrameLowering",     # 스택 프레임
    ]
    
    def synthesize_component(self, component: str, isa: ISAModel) -> Code:
        # 각 구성 요소를 독립적으로 합성
        # 템플릿 + ISA 특성 기반 인스턴스화
        pass
```

#### 핵심 기여 3: "Verification-Guided Iteration"
```python
class VerificationLoop:
    """
    생성된 백엔드를 검증하고 자동 수정
    """
    def verify_and_refine(self, backend: GeneratedBackend) -> VerifiedBackend:
        while True:
            # 1. SMT-based 의미론 검증
            result = self.smt_verify(backend)
            if result.verified:
                return backend
            
            # 2. 반례에서 수정 힌트 추출
            hints = self.analyze_counterexample(result.counterexample)
            
            # 3. 백엔드 자동 수정
            backend = self.repair(backend, hints)
```

### 5.2 예상 기여도

| 기존 연구 | 개선점 |
|----------|--------|
| VEGA: 기존 백엔드 필요 | → ISA 스펙만으로 시작 가능 |
| ACT: Tensor only | → 범용 ISA 지원 |
| Isaria: 벡터화만 | → 완전한 백엔드 생성 |
| VADL: 명세 작성 필요 | → 자동 명세 추출 |

### 5.3 실현 가능성 평가

```
단기 (6개월):
- ISA 스펙에서 기본 구조 추출 (레지스터, 인코딩)
- 단순 명령어에 대한 백엔드 생성
- 기본 검증 프레임워크

중기 (1년):
- 복잡한 명령어 지원 (메모리, 분기)
- 최적화 패스 자동 생성
- 더 정교한 검증

장기 (2년):
- 완전한 프로덕션 수준 백엔드
- 다양한 ISA 타입 지원
- 자동 최적화 수준 선택
```

---

## 6. 현재 시스템(VEGA-Verified)에 대한 제안

### 6.1 현재 시스템의 위치
```
현재: 버그 수정 (Neural Repair)
     ↓
개선 1: ISA 스펙에서 학습 데이터 생성
     ↓
개선 2: 코드 생성으로 전환
     ↓
개선 3: 검증 기반 반복 수정
```

### 6.2 구체적 구현 로드맵

**Phase 1: 데이터 파이프라인 개선**
```python
# 현재: 단순 버그-수정 쌍
training_data = [
    ("buggy_code", "fixed_code"),
]

# 개선: ISA 스펙 → 백엔드 코드
training_data = [
    {
        "isa_spec": {
            "name": "RISC-V",
            "registers": [...],
            "instructions": [...]
        },
        "backend_component": "MCCodeEmitter",
        "generated_code": "..."
    }
]
```

**Phase 2: 모델 아키텍처 변경**
```
현재: CodeT5 (Seq2Seq)
      buggy → fixed

개선: ISA-Aware Encoder + Component-Specific Decoder
      ISA spec → backend component
```

**Phase 3: 검증 통합**
```
현재: 단순 테스트 기반 검증
개선: SMT-based 의미론 검증 + 반례 기반 수정
```

---

## 7. 결론

### 7.1 현재 연구 동향 요약
- **VEGA**: ML 기반이지만 기존 백엔드 의존
- **ACT**: 형식적이지만 Tensor only
- **Isaria**: 자동화되지만 벡터화 only
- **OpenVADL**: 완전하지만 명세 작성 필요

### 7.2 우리의 목표를 위한 Gap
> "ISA 스펙만으로 백엔드 생성"은 현재 **어느 연구도 완전히 해결하지 못함**

### 7.3 제안하는 방향
1. **ISA 의미론 자동 추출**: LLM + 형식적 방법 결합
2. **구성적 합성**: 백엔드를 구성 요소별로 분리
3. **검증 기반 반복**: 반례로 자동 수정

### 7.4 현재 시스템에서의 시작점
- 현재의 CGNR (반례 기반 수정) 프레임워크를 확장
- 버그 수정 → 코드 생성으로 모델 목적 전환
- ISA 스펙 파서 및 데이터 파이프라인 구축

---

## 참고문헌

1. Zhong et al., "VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model", CGO 2025
2. Jain et al., "ACT: Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions", arXiv 2025
3. Thomas and Bornholt, "Automatic Generation of Vectorizing Compilers for Customizable Digital Signal Processors", ASPLOS 2024
4. Freitag et al., "The Vienna Architecture Description Language", arXiv 2024
