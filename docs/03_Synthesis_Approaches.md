# 03. Synthesis Approaches for Compiler Backend Generation

**Author**: Compiler Backend Autogeneration Research Group  
**Version**: 2.0  
**Date**: 2026-01-23  
**Classification**: Research Analysis Document

---

## Executive Summary

이 문서는 컴파일러 백엔드 자동생성을 위한 다양한 합성(Synthesis) 접근법을 분석한다. 크게 세 가지 패러다임—**Program Synthesis**, **Machine Learning**, **Hybrid Approaches**—을 검토하고, 각각의 장단점과 실제 적용 가능성을 평가한다.

---

## 1. Program Synthesis Approaches

### 1.1 Equality Saturation 기반 접근

#### 1.1.1 기본 개념

Equality Saturation은 프로그램 변환을 위한 강력한 기법으로, E-graph 자료구조를 사용하여 등가 프로그램들을 효율적으로 표현한다.

```
Definition: E-graph
- E-node: 연산자와 자식 E-class의 조합
- E-class: 의미적으로 동등한 E-node들의 집합
- Saturation: 더 이상 새로운 등가 관계가 추가되지 않을 때까지 rewrite 적용
```

#### 1.1.2 ACT의 Parameterized Equality Saturation

ACT[2]는 텐서 가속기를 위해 **파라미터화된** Equality Saturation을 도입:

```python
# ACT의 Parameterized E-matching 개념
class ParameterizedPattern:
    """
    일반 패턴: matmul(A[M,K], B[K,N]) → result[M,N]
    파라미터화: matmul(A[m,k], B[k,n]) → result[m,n]
                where m, k, n are symbolic parameters
    """
    
    def __init__(self, pattern_template, constraints):
        self.template = pattern_template
        self.constraints = constraints  # e.g., k <= 256
    
    def match_with_parameters(self, expr):
        """
        Return: (match_success, parameter_bindings)
        Example: matmul(A[128,64], B[64,256]) 
                 → (True, {m: 128, k: 64, n: 256})
        """
        pass
```

**장점**:
- 다양한 타일 크기를 단일 규칙으로 처리
- 하드웨어 제약을 파라미터 제약으로 표현 가능

**한계**:
- 텐서 연산에 특화, 범용 ISA에 적용 어려움
- 파라미터 공간이 커지면 탐색 비용 급증

#### 1.1.3 Isaria의 Phased Equality Saturation

Isaria[3]는 위상 분리(Phase Separation)로 E-graph 폭발 문제를 해결:

```
Phase Structure:
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Expansion                                         │
│  - 벡터화 기회 노출                                           │
│  - 스칼라 → 잠재적 벡터 형태로 확장                           │
│  Rules: {r | cost_delta(r) > α}  (비용 증가 허용)           │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: Compilation                                       │
│  - 벡터 명령어로 하강                                        │
│  - 하드웨어 특화 패턴 매칭                                    │
│  Rules: {r | cost_delta(r) ≈ 0}  (비용 중립)               │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: Optimization                                      │
│  - 불필요한 연산 제거                                        │
│  - 레지스터 압력 최소화                                       │
│  Rules: {r | cost_delta(r) < -β}  (비용 감소만)            │
└─────────────────────────────────────────────────────────────┘
```

**핵심 통찰**: 모든 규칙을 동시에 적용하면 E-graph가 폭발하지만, 비용 변화 방향에 따라 위상을 분리하면 제어 가능.

### 1.2 Constraint Programming 기반 접근

#### 1.2.1 ACT의 메모리 할당

ACT는 메모리 할당을 Constraint Satisfaction Problem(CSP)으로 정형화:

```python
# ACT의 메모리 할당 CP 모델
class MemoryAllocationCP:
    def __init__(self, scratchpad_size, banks, access_patterns):
        self.size = scratchpad_size
        self.banks = banks
        self.patterns = access_patterns
    
    def formulate_constraints(self):
        """
        Variables:
        - base[tensor]: 각 텐서의 시작 주소
        - bank_assign[tensor]: 뱅크 할당
        
        Constraints:
        1. Non-overlap: 텐서들이 겹치지 않음
           ∀i,j: end[i] ≤ base[j] ∨ end[j] ≤ base[i]
        
        2. Bank conflict avoidance: 동시 접근 시 뱅크 충돌 없음
           ∀(i,j) ∈ concurrent_accesses: 
             bank_assign[i] ≠ bank_assign[j]
        
        3. Alignment: 하드웨어 정렬 요구사항
           ∀i: base[i] mod alignment[i] = 0
        
        4. Size limit: 전체 크기 제한
           ∀i: base[i] + size[i] ≤ scratchpad_size
        """
        pass
```

#### 1.2.2 VADL의 레지스터 할당

VADL[4]은 레지스터 할당을 Graph Coloring 문제로 해결:

```
Register Allocation Constraints:
1. Interference: 동시에 살아있는 값들은 다른 레지스터에 할당
2. Register class: 각 값은 허용된 레지스터 클래스 내에서 할당
3. Calling convention: ABI 호환 레지스터 사용
4. Spill cost minimization: 스필 비용 최소화
```

### 1.3 SMT 기반 합성

#### 1.3.1 VeGen의 의미론 추출

VeGen[5]은 SMT 솔버를 사용하여 벡터 명령어 의미론을 자동 추출:

```python
# VeGen의 SMT 기반 의미론 추출
class VeGenSemanticsExtractor:
    def __init__(self, solver="z3"):
        self.solver = solver
    
    def extract_lane_semantics(self, instruction_spec):
        """
        Input: 벤더 문서에서 파싱한 명령어 스펙
        Output: 레인별 연산의 SMT 공식
        
        Example: _mm_add_epi32
        
        For each lane i in 0..3:
            result[i] = SignExtend32(a[i]) + SignExtend32(b[i])
            
        SMT Formula:
            ∀i ∈ [0,3]: extract(result, i*32, 32) = 
                bvadd(sign_extend(extract(a, i*32, 32), 32),
                      sign_extend(extract(b, i*32, 32), 32))
        """
        pass
    
    def verify_equivalence(self, impl1, impl2):
        """
        두 구현의 의미론적 동등성을 SMT로 검증
        """
        formula = Not(Equals(impl1, impl2))
        return self.solver.check(formula) == UNSAT
```

#### 1.3.2 Hydride의 Cross-ISA Equivalence

Hydride[6]는 서로 다른 ISA 간의 동등성을 SMT로 검증:

```python
# Hydride의 Cross-ISA 동등성 검증
class HydrideEquivalenceChecker:
    def check_cross_isa_equivalence(self, x86_impl, hvx_impl):
        """
        x86 AVX 구현과 HVX 구현의 동등성 검증
        
        Challenge: 
        - 레인 크기가 다름 (AVX: 256-bit, HVX: 1024-bit)
        - 연산 의미론이 미묘하게 다를 수 있음
        
        Solution:
        1. 양쪽을 공통 IR (AutoLLVM)로 리프팅
        2. 공통 IR에서 동등성 검증
        """
        x86_ir = self.lift_to_auto_llvm(x86_impl)
        hvx_ir = self.lift_to_auto_llvm(hvx_impl)
        
        # Bounded model checking
        for input_size in [1, 4, 8, 16]:
            concrete_inputs = self.generate_test_vectors(input_size)
            for inp in concrete_inputs:
                if self.evaluate(x86_ir, inp) != self.evaluate(hvx_ir, inp):
                    return CounterExample(inp)
        
        return Equivalent()
```

---

## 2. Machine Learning Approaches

### 2.1 Pre-trained Transformer 기반 (VEGA)

#### 2.1.1 Architecture

```
VEGA Architecture:
┌────────────────────────────────────────────────────────────────┐
│                     Input Processing                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Target Desc  │    │ Function     │    │ Feature      │     │
│  │ File (.td)   │ → │ Template     │ → │ Vector       │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
├────────────────────────────────────────────────────────────────┤
│                     UniXcoder Backbone                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Pre-trained on: CodeSearchNet (6 languages)            │  │
│  │  Fine-tuned on: ComBack dataset (LLVM backends)         │  │
│  │                                                         │  │
│  │  Encoder: 12 layers, 768 hidden, 12 heads               │  │
│  │  Decoder: 12 layers, 768 hidden, 12 heads               │  │
│  └─────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────┤
│                     Output Generation                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Code         │    │ Confidence   │    │ Manual Edit  │     │
│  │ Generation   │ → │ Score        │ → │ Guidance     │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
└────────────────────────────────────────────────────────────────┘
```

#### 2.1.2 Feature Vector Design

VEGA의 핵심은 Target Description에서 Feature Vector를 추출하는 것:

```python
# VEGA의 Feature Vector 구성
class VEGAFeatureExtractor:
    def extract_features(self, target_desc_file):
        """
        Feature Categories:
        1. Register Features
           - 레지스터 클래스 수
           - 각 클래스의 레지스터 수
           - 특수 레지스터 (SP, FP, etc.)
        
        2. Instruction Features
           - 명령어 패밀리별 수
           - 복합 명령어 존재 여부
           - 조건부 실행 지원 여부
        
        3. ABI Features
           - 호출 규약 정보
           - 스택 정렬 요구사항
           - 가변 인자 처리 방식
        
        4. Target Features
           - 엔디안
           - 포인터 크기
           - 정렬 요구사항
        """
        features = {}
        
        # Parse .td file
        register_info = self.parse_registers(target_desc_file)
        instruction_info = self.parse_instructions(target_desc_file)
        abi_info = self.parse_calling_convention(target_desc_file)
        
        # Encode features
        features['register_vec'] = self.encode_registers(register_info)
        features['instruction_vec'] = self.encode_instructions(instruction_info)
        features['abi_vec'] = self.encode_abi(abi_info)
        
        return self.concatenate(features)
```

#### 2.1.3 한계 분석

```
VEGA의 근본적 한계:

1. 의미론 이해 부재
   - 코드의 구조적 패턴만 학습
   - 의미론적 정확성 보장 불가
   - 예: 조건 분기의 의미 vs 문법

2. 기존 백엔드 의존
   - 유사한 백엔드가 없으면 품질 저하
   - 완전히 새로운 ISA에 적용 어려움
   
3. 함수 경계 문제
   - 각 함수를 독립적으로 생성
   - 함수 간 일관성 보장 없음
   
4. 검증 부재
   - 생성된 코드의 정확성 검증 없음
   - 수동 검토 필수
```

### 2.2 Seq2Seq 기반 버그 수정 (현재 시스템)

현재 VEGA-Verified의 Neural Repair 모듈:

```python
# 현재 시스템의 학습 데이터 구조
class RepairDataset(Dataset):
    """
    데이터 형태: (buggy_code, fixed_code) 쌍
    
    확장 방식:
    - 5개 기본 패턴에서 시작
    - Kind 변수로 변형 생성 (Kind_0, Kind_1, ...)
    - 총 약 2000개 샘플로 확장
    
    패턴 유형:
    1. Missing case
    2. Wrong return  
    3. Missing null check
    4. Off-by-one
    5. Missing break
    """
    
    def __init__(self, data: List[Tuple[str, str]], tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        buggy, fixed = self.data[idx]
        
        # 입력 형식: "fix bug: {source}"
        input_text = f"fix bug: {buggy}"
        target_text = fixed
        
        return {
            'input_ids': self.tokenizer.encode(input_text),
            'labels': self.tokenizer.encode(target_text)
        }
```

#### 2.2.1 현재 접근법의 문제점

```
문제점 분석:

1. 데이터 다양성 부족
   - 5개 패턴의 단순 확장
   - 실제 버그의 복잡성 반영 못함
   - ISA 특화 버그 패턴 부재

2. 결정론적 생성 문제
   - do_sample=True로 인한 랜덤성
   - 동일 입력에 다른 출력
   - 검증 파이프라인과 충돌

3. 프롬프트-학습 데이터 불일치
   - 학습 시: "fix bug: {code}"
   - 추론 시: 다른 프롬프트 형식
   - 분포 이동(distribution shift) 발생

4. Counterexample 활용 미흡
   - 반례를 학습에 직접 반영하지 않음
   - 단순 재시도에 그침
```

### 2.3 제안: ISA-Aware Neural Generation

```python
# 제안하는 ISA-Aware 생성 모델
class ISAAwareGenerator:
    """
    핵심 개선사항:
    1. ISA 스펙을 직접 입력으로 받음
    2. 구조화된 출력 생성
    3. 검증 가능한 중간 표현 사용
    """
    
    def __init__(self, base_model="Salesforce/codet5-large"):
        self.encoder = ISASpecEncoder(base_model)
        self.decoder = StructuredCodeDecoder(base_model)
        self.verifier = IncrementalVerifier()
    
    def generate(self, isa_spec, component_type):
        """
        Input:
        - isa_spec: 파싱된 ISA 명세
        - component_type: 생성할 컴포넌트 (RegisterInfo, InstrInfo, etc.)
        
        Output:
        - 검증된 코드 조각
        """
        # 1. ISA 스펙 인코딩
        spec_embedding = self.encoder.encode(isa_spec)
        
        # 2. 구조화된 생성
        candidates = self.decoder.generate(
            spec_embedding, 
            component_type,
            num_candidates=5,
            do_sample=False,  # 결정론적 생성
            temperature=1.0
        )
        
        # 3. 점진적 검증
        for candidate in candidates:
            if self.verifier.verify_partial(candidate, component_type):
                return candidate
        
        return None
```

---

## 3. Hybrid Approaches

### 3.1 VeGen: Semantics + Code Generation

VeGen의 하이브리드 접근:

```
VeGen Pipeline:
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Semantic Extraction (Symbolic)                        │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐            │
│  │ Vendor     │    │ Symbolic   │    │ SMT        │            │
│  │ Pseudocode │ → │ Evaluator  │ → │ Formulas   │            │
│  └────────────┘    └────────────┘    └────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Pattern Generation (Template-based)                   │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐            │
│  │ SMT        │    │ VIDL       │    │ Pattern    │            │
│  │ Formulas   │ → │ Encoding   │ → │ Matchers   │            │
│  └────────────┘    └────────────┘    └────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Vectorization (Heuristic)                             │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐            │
│  │ Scalar     │    │ SLP        │    │ Vectorized │            │
│  │ Program    │ → │ Packing    │ → │ Program    │            │
│  └────────────┘    └────────────┘    └────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 VERA: 제안하는 통합 프레임워크

```
VERA (Verified End-to-end Retargetable Architecture):
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: ISA Understanding (LLM + Formal)                      │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ ISA Spec     │    │ LLM-based    │    │ Formal       │      │
│  │ Document     │ → │ Extraction   │ → │ Model        │      │
│  │ (PDF/XML)    │    │              │    │ (SMT-LIB)    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                           │                     │               │
│                           v                     v               │
│                    Cross-Validation via Semantic Matching       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Compositional Synthesis                               │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                 Component Templates                     │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │    │
│  │  │Register │ │Instr    │ │Calling  │ │MC       │      │    │
│  │  │Info     │ │Selection│ │Conv     │ │Emitter  │      │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘      │    │
│  └────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           v                                     │
│              ISA-Guided Instantiation + ML Assistance           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Verification-Guided Refinement                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │   SMT-based Verification                                │   │
│  │   ┌─────────┐     ┌─────────┐     ┌─────────┐          │   │
│  │   │ VC Gen  │ ──→ │ SMT     │ ──→ │ Counter │          │   │
│  │   │         │     │ Solver  │     │ Example │          │   │
│  │   └─────────┘     └─────────┘     └─────────┘          │   │
│  │                        │                │               │   │
│  │                        v                v               │   │
│  │              ┌─────────────────────────────────┐       │   │
│  │              │    CGNR (Neural Repair)         │       │   │
│  │              └─────────────────────────────────┘       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 각 Layer의 기술적 세부사항

#### Layer 1: ISA Understanding

```python
class ISAUnderstandingLayer:
    """
    대용량 ISA 스펙 문서 처리를 위한 계층적 접근
    """
    
    def __init__(self, llm_model, formal_verifier):
        self.llm = llm_model
        self.verifier = formal_verifier
        self.chunk_processor = ChunkBasedProcessor(
            max_chunk_size=8000,  # tokens
            overlap=500
        )
    
    def process_isa_spec(self, spec_document):
        """
        대용량 ISA 스펙 처리 전략:
        
        1. 구조 기반 분할
           - 레지스터 설명 섹션
           - 명령어 인코딩 섹션
           - 명령어 의미론 섹션
           - 예외 처리 섹션
        
        2. 각 섹션 독립 처리
           - LLM으로 구조화된 정보 추출
           - 형식 모델로 변환
        
        3. Cross-Reference 해결
           - 섹션 간 참조 연결
           - 일관성 검증
        """
        # 문서 구조 분석
        sections = self.analyze_structure(spec_document)
        
        # 섹션별 처리
        results = {}
        for section_type, content in sections.items():
            if section_type == 'registers':
                results['registers'] = self.extract_registers(content)
            elif section_type == 'instructions':
                results['instructions'] = self.extract_instructions(content)
            elif section_type == 'encodings':
                results['encodings'] = self.extract_encodings(content)
        
        # Cross-validation
        self.validate_consistency(results)
        
        return results
```

#### Layer 2: Compositional Synthesis

```python
class CompositionalSynthesisLayer:
    """
    백엔드 컴포넌트별 독립 합성
    """
    
    COMPONENTS = [
        'TargetInfo',
        'RegisterInfo', 
        'InstrInfo',
        'FrameLowering',
        'ISelDAGToDAG',
        'AsmPrinter',
        'MCCodeEmitter',
        'Disassembler'
    ]
    
    def synthesize_backend(self, isa_model, reference_backends=None):
        """
        컴포넌트별 합성 전략:
        
        1. 의존성 순서로 합성
           TargetInfo → RegisterInfo → InstrInfo → ...
        
        2. 각 컴포넌트에 대해:
           a. ISA 모델에서 관련 정보 추출
           b. 템플릿 기반 초기 생성
           c. ML 모델로 세부사항 채움
           d. 단위 검증
        
        3. 컴포넌트 간 일관성 검증
        """
        backend = {}
        
        for component in self.COMPONENTS:
            # 의존성 주입
            deps = {c: backend[c] for c in self.get_dependencies(component)}
            
            # 합성
            backend[component] = self.synthesize_component(
                component, isa_model, deps, reference_backends
            )
            
            # 단위 검증
            self.verify_component(component, backend[component])
        
        return backend
```

#### Layer 3: Verification-Guided Refinement

```python
class VerificationGuidedRefinementLayer:
    """
    검증 주도 반복 개선
    """
    
    def refine_until_verified(self, backend, isa_model, max_iterations=10):
        """
        CGNR (Counterexample-Guided Neural Repair) 확장
        """
        for iteration in range(max_iterations):
            # 검증
            result = self.verify_backend(backend, isa_model)
            
            if result.is_verified():
                return backend, VerificationReport(
                    status="VERIFIED",
                    iterations=iteration
                )
            
            # 반례 분석
            counterexample = result.get_counterexample()
            fault_location = self.localize_fault(counterexample, backend)
            
            # 신경망 기반 수정
            repair_context = RepairContext(
                faulty_code=fault_location.code,
                counterexample=counterexample,
                isa_constraint=isa_model.get_relevant_constraints(fault_location)
            )
            
            repaired_code = self.neural_repair(repair_context)
            backend = self.apply_repair(backend, fault_location, repaired_code)
        
        return backend, VerificationReport(
            status="TIMEOUT",
            iterations=max_iterations,
            remaining_issues=result.get_issues()
        )
```

---

## 4. Synthesis Quality Metrics

### 4.1 정확도 지표

```
┌──────────────────┬────────────┬────────────┬────────────┬────────────┐
│ Metric           │ VEGA       │ ACT        │ Isaria     │ VERA(est.) │
├──────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Function-level   │ 71.5%      │ N/A        │ N/A        │ 85%+       │
│ Statement-level  │ 55.0%      │ N/A        │ N/A        │ 90%+       │
│ Semantic correct │ Unknown    │ 100%*      │ 100%*      │ 100%*      │
│ End-to-end       │ <8%        │ ~100%      │ ~100%      │ 95%+       │
└──────────────────┴────────────┴────────────┴────────────┴────────────┘
* 제한된 도메인 내에서
```

### 4.2 효율성 지표

```
┌──────────────────┬────────────┬────────────┬────────────┐
│ Phase            │ ACT        │ Isaria     │ VERA(est.) │
├──────────────────┼────────────┼────────────┼────────────┤
│ Offline prep     │ Hours      │ ~1 day     │ ~1 week    │
│ Per-program      │ ~311ms     │ Minutes    │ Seconds    │
│ Memory (offline) │ <64GB      │ 220GB      │ <128GB     │
│ Memory (online)  │ <8GB       │ <16GB      │ <16GB      │
└──────────────────┴────────────┴────────────┴────────────┘
```

---

## 5. Implementation Roadmap

### 5.1 단기 (3개월)

```
Phase 1: 기반 구축
┌─────────────────────────────────────────────────────────────────┐
│ 1. ISA 스펙 파서 구현                                            │
│    - PDF/XML → 구조화된 형식                                     │
│    - RISC-V를 초기 타겟으로                                      │
│                                                                 │
│ 2. 컴포넌트 템플릿 정의                                          │
│    - 기존 LLVM 백엔드 분석                                       │
│    - 파라미터화된 템플릿 추출                                     │
│                                                                 │
│ 3. 단위 검증기 구현                                              │
│    - 각 컴포넌트별 specification                                │
│    - SMT 기반 검증 조건 생성                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 중기 (6개월)

```
Phase 2: 핵심 기능 구현
┌─────────────────────────────────────────────────────────────────┐
│ 1. ISA-Aware 인코더 훈련                                        │
│    - ISA 스펙 → 임베딩                                          │
│    - 다중 ISA 사전훈련                                          │
│                                                                 │
│ 2. 구조화된 디코더 구현                                          │
│    - 컴포넌트별 전문화된 디코더                                   │
│    - 문법 제약 통합                                              │
│                                                                 │
│ 3. CGNR 파이프라인 통합                                         │
│    - 검증 실패 → 수정 루프                                       │
│    - 반례 기반 학습 데이터 증강                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 장기 (12개월)

```
Phase 3: 완성 및 검증
┌─────────────────────────────────────────────────────────────────┐
│ 1. End-to-end 파이프라인 완성                                    │
│    - ISA 스펙 → 검증된 백엔드                                    │
│    - 자동화된 테스트 생성                                        │
│                                                                 │
│ 2. 다중 ISA 평가                                                │
│    - RISC-V, ARM, x86 subset                                   │
│    - 새로운 ISA (가상) 테스트                                    │
│                                                                 │
│ 3. 논문 작성 및 오픈소스 공개                                     │
│    - 주요 학회 제출 (PLDI, ASPLOS, CGO)                         │
│    - 재현 가능한 아티팩트                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Conclusion

### 6.1 핵심 통찰

1. **순수 ML 접근법의 한계**: VEGA와 같은 ML 기반 접근법은 의미론적 정확성을 보장하지 못함

2. **형식 기법의 범위 제한**: ACT, Isaria는 특정 도메인에서 강력하지만, 범용 백엔드 생성에는 부적합

3. **명세 작성의 부담**: VADL은 포괄적이지만, 명세 작성 자체가 백엔드 구현과 비슷한 수준의 노력 필요

4. **하이브리드의 가능성**: VeGen, Hydride의 접근법을 확장하여 의미론 추출과 ML 생성을 결합할 수 있음

### 6.2 연구 방향

```
권장 연구 우선순위:

1. ISA 의미론 자동 추출 (High Priority)
   - VeGen의 접근법을 범용 ISA로 확장
   - LLM을 활용한 자연어 스펙 이해

2. 검증 가능한 생성 (High Priority)
   - 생성 과정에 검증 통합
   - 점진적 검증으로 오류 조기 발견

3. 컴포넌트 템플릿 라이브러리 (Medium Priority)
   - 재사용 가능한 템플릿 축적
   - 새 ISA 적용 시 시작점 제공

4. 반례 기반 학습 (Medium Priority)
   - 검증 실패에서 학습 데이터 생성
   - 지속적 모델 개선
```

---

## References

[1] E. Schkufza et al., "Stochastic Superoptimization," ASPLOS 2013  
[2] M. Jain et al., "ACT: Compiler Backends from Tensor Accelerator ISA Descriptions," arXiv 2025  
[3] S. Thomas, J. Bornholt, "Automatic Generation of Vectorizing Compilers for Customizable DSPs," ASPLOS 2024  
[4] A. Freitag et al., "The Vienna Architecture Description Language," arXiv 2024  
[5] Y. Chen et al., "VeGen: A Vectorizer Generator for SIMD and Beyond," ASPLOS 2021  
[6] M. Willsey et al., "Hydride: Portably Lifting Vector Intrinsics to IR Level," ASPLOS 2022  
[7] M. Zhong et al., "VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model," CGO 2025  
