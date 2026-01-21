# VEGA 논문 재현 및 분석 프로젝트

## 개요

이 프로젝트는 CGO 2025에 발표된 **VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model** 논문을 재현하고, 저명한 compiler backend autogeneration 연구자의 관점에서 분석하여 개선점을 제안합니다.

## 프로젝트 구조

```
webapp/
├── Dockerfile              # GPU 지원 Docker 환경 (CUDA 11.7)
├── Dockerfile.light        # CPU 전용 경량 Docker 환경
├── build_and_run.sh        # Docker 빌드 및 실행 스크립트
├── run_vega_tests.sh       # 테스트 실행 스크립트
├── VEGA_Analysis_Report.md # 상세 분석 보고서
├── tests/
│   ├── matmul_test.cpp         # MatMul correctness 테스트
│   ├── vega_simulator.py       # VEGA 워크플로우 시뮬레이터
│   └── vega_verified_prototype.py  # VEGA-Verified 제안 프로토타입
└── results/                # 테스트 결과 저장소
```

## 빠른 시작

### 1. Docker 이미지 빌드
```bash
# 경량 버전 (CPU만)
docker build -t vega-light -f Dockerfile.light .

# 전체 버전 (GPU 지원)
docker build -t vega-reproduction .
```

### 2. 테스트 실행
```bash
# VEGA 시뮬레이터 실행
docker run --rm -v "$(pwd)/tests:/workspace/tests" vega-light \
    python3 /workspace/tests/vega_simulator.py

# MatMul 테스트
docker run --rm -v "$(pwd)/tests:/workspace/tests" vega-light \
    bash -c "cd /workspace/tests && g++ -O2 matmul_test.cpp -o matmul && ./matmul"

# VEGA-Verified 프로토타입
docker run --rm -v "$(pwd)/tests:/workspace/tests" vega-light \
    python3 /workspace/tests/vega_verified_prototype.py
```

## VEGA 논문 핵심 분석

### 강점
- AI 기반 컴파일러 백엔드 자동 생성의 첫 성공 사례
- Function template abstraction을 통한 cross-target 일반화
- 71.5%+ function-level accuracy (기존 fork-flow <8% 대비)
- Confidence score를 통한 human-in-the-loop 지원

### 약점 (8가지 주요 한계)
1. **형식적 의미론 검증 부재** - 생성 코드의 correctness 보장 불가
2. **제한된 함수 모듈 커버리지** - 7개 모듈만 지원 (~60%)
3. **훈련 데이터 의존성** - 기존 백엔드 패턴에서만 학습
4. **Statement-level 한계** - Cross-statement 최적화 어려움
5. **TableGen 의존성** - 타겟 설명 파일 필수
6. **LLVM 종속** - GCC 등 타 컴파일러 적용 불가
7. **Confidence Score 한계** - Binary classification의 표현력
8. **Instruction Selection 미지원** - 핵심 부분 수동 작업 필요

## 제안: VEGA-Verified

### 핵심 아이디어
VEGA의 신경망 기반 코드 생성과 형식 검증을 결합하여 **의미론적 정확성을 보장**

### 3가지 Main Contributions

#### 1. Automated Semantic Specification Inference
- 참조 백엔드에서 자동으로 formal specification 추론
- Abstract interpretation 및 invariant detection 활용
- **차별점**: ACT(manual), Hydride(vendor docs) 대비 fully automatic

#### 2. Counterexample-Guided Neural Repair (CGNR)
- 검증 실패 시 counterexample 기반 자동 수정
- CEGAR 원리를 neural repair에 적용
- **차별점**: 기존 CEGIS(scalability 문제) 대비 neural + formal hybrid

#### 3. Hierarchical Verification with Modular Composability
- Function → Module → Backend 3단계 계층적 검증
- Interface contract를 통한 모듈 간 composability
- **차별점**: 기존 monolithic 검증 대비 incremental, scalable

### 기대 효과

| Metric | VEGA | VEGA-Verified |
|--------|------|---------------|
| Function Accuracy | 71.5% | 85-90% |
| Semantic Correctness | Unknown | 100% (verified) |
| Verification Coverage | 0% | 80-90% |

## 관련 연구 비교

| 연구 | 접근법 | 강점 | 약점 |
|------|--------|------|------|
| VEGA | Neural | 자동화, 속도 | 검증 없음 |
| Hydride | Synthesis | 벡터 최적화 | 확장성 |
| ACT | E-graph | Formal guarantee | Tensor only |
| OpenVADL | ADL | 완전 자동 | 새 DSL 필요 |
| **VEGA-Verified** | **Hybrid** | **자동화 + 검증** | 추가 시간 |

## 파일 설명

### `tests/vega_simulator.py`
VEGA의 핵심 워크플로우를 시뮬레이션:
- Function template 생성
- Feature vector 추출 (TI/TS 구분)
- Target-specific 코드 생성
- Confidence score 분석

### `tests/vega_verified_prototype.py`
제안하는 VEGA-Verified 시스템 프로토타입:
- `SpecificationInferrer`: 자동 specification 추론
- `CounterexampleGuidedRepair`: CGNR 알고리즘
- `HierarchicalVerifier`: 3단계 계층적 검증
- `FormalVerifier`: 간이 SMT 기반 검증기

### `tests/matmul_test.cpp`
MatMul correctness 테스트:
- Naive, tiled, vectorized 구현
- Cross-validation을 통한 정확성 검증

## 향후 작업

1. **Z3 통합**: 실제 SMT solver 활용
2. **Neural Repair 모델 학습**: (buggy, counterexample, fixed) 데이터셋
3. **LLVM 통합**: 실제 백엔드 코드에 적용
4. **벤치마크**: SPEC 등 표준 벤치마크 평가

## 참고문헌

1. Zhong et al., "VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model", CGO 2025
2. Kothen et al., "Hydride: A Retargetable Synthesis-based Compiler", ASPLOS 2024
3. "ACT: Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions", 2025
4. Tate et al., "Equality Saturation: A New Approach to Optimization", POPL 2009
5. Guo et al., "UniXcoder: Unified Cross-Modal Pre-training for Code Representation", ACL 2022

## 라이선스

MIT License
