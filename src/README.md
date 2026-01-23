# src/ - 소스 코드 디렉토리

이 디렉토리는 VEGA-Verified 시스템의 핵심 소스 코드를 포함합니다.

---

## 📁 디렉토리 구조

```
src/
├── __init__.py              # 패키지 초기화
├── cli.py                   # CLI 진입점 (vega-verify 명령어)
├── main.py                  # 레거시 진입점
│
├── specification/           # 명세 언어 및 추론
│   ├── spec_language.py     # 형식 명세 DSL
│   ├── symbolic_exec.py     # Z3 기반 기호적 실행
│   └── inferrer.py          # 명세 자동 추론
│
├── verification/            # 검증 엔진
│   ├── verifier.py          # 메인 검증기 인터페이스
│   ├── smt_solver.py        # 확장된 SMT 솔버
│   ├── switch_verifier.py   # switch문 검증
│   ├── z3_backend.py        # Z3 통합
│   └── bmc.py               # 제한된 모델 체킹
│
├── repair/                  # 코드 수리
│   ├── cgnr.py              # CGNR 알고리즘
│   ├── neural_repair_engine.py  # GPU용 신경망 수리
│   ├── repair_model.py      # 규칙 기반 수리
│   ├── neural_model.py      # HuggingFace 백엔드
│   ├── fault_loc.py         # 결함 위치 추정
│   └── training_data.py     # 학습 데이터 생성
│
├── hierarchical/            # 계층적 검증
│   ├── function_verify.py   # L1: 함수 레벨 검증
│   ├── module_verify.py     # L2: 모듈 레벨 검증
│   ├── backend_verify.py    # L3: 백엔드 레벨 검증
│   └── hierarchical_verifier.py  # 통합 인터페이스
│
├── integration/             # 파이프라인 통합
│   ├── pipeline.py          # 메인 파이프라인
│   ├── cgnr_pipeline.py     # CGNR 통합
│   └── vega_adapter.py      # VEGA 모델 어댑터
│
├── parsing/                 # 코드 파싱
│   └── clang_ast_parser.py  # Clang AST 파서
│
├── llvm_extraction/         # LLVM 함수 추출
│   └── extractor.py         # LLVM 소스 추출기
│
└── utils/                   # 유틸리티
    └── config.py            # 설정 관리
```

---

## 🔧 주요 모듈 설명

### 1. specification/ - 명세 모듈

형식 명세를 정의하고 추론하는 모듈입니다.

```python
from src.specification import Specification, SpecificationInferrer

# 명세 생성
spec = Specification(
    function_name="getRelocType",
    preconditions=["valid_kind(Kind)"],
    postconditions=["result in {R_X86_64_32, R_X86_64_64, R_X86_64_NONE}"]
)

# 명세 추론
inferrer = SpecificationInferrer()
spec = inferrer.infer("getRelocType", references)
```

**주요 파일:**
- `spec_language.py`: Specification, Condition, Variable 등 핵심 데이터 클래스
- `symbolic_exec.py`: Z3 + Clang AST 기반 기호적 실행 (950+ LOC)
- `inferrer.py`: 참조 구현에서 명세 자동 추론

### 2. verification/ - 검증 모듈

SMT 기반 검증을 수행하는 모듈입니다.

```python
from src.verification import Verifier, SMTSolver

# 검증기 생성
verifier = Verifier(timeout_ms=30000)

# 검증 실행
result = verifier.verify(code, spec)
if result.is_verified():
    print("검증 성공!")
else:
    print(f"반례: {result.counterexample}")
```

**주요 파일:**
- `verifier.py`: 메인 검증 인터페이스
- `smt_solver.py`: Z3 기반 SMT 솔버 (메모리 모델, 함수 호출 지원)
- `switch_verifier.py`: switch문 전용 검증기 (968 LOC)
- `bmc.py`: 제한된 모델 체킹

### 3. repair/ - 수리 모듈

CGNR 알고리즘과 신경망 기반 수리를 구현합니다.

```python
from src.repair import CGNREngine, NeuralRepairEngine

# CGNR 수리
cgnr = CGNREngine(verifier=verifier, max_iterations=5)
result = cgnr.repair(buggy_code, spec)

# Neural 수리 (GPU 필요)
neural = NeuralRepairEngine(model_name="Salesforce/codet5-base")
neural.load()
candidates = neural.repair(buggy_code, counterexample)
```

**주요 파일:**
- `cgnr.py`: 반례 유도 신경망 수리 알고리즘
- `neural_repair_engine.py`: GPU용 CodeT5 기반 수리 (870 LOC)
- `repair_model.py`: 규칙 기반 수리 (CPU 폴백)
- `fault_loc.py`: 결함 위치 추정

### 4. hierarchical/ - 계층적 검증 모듈

3단계 계층적 검증을 구현합니다.

```python
from src.hierarchical import HierarchicalVerifier, Module, Backend

# 계층적 검증기
verifier = HierarchicalVerifier()

# 함수 레벨 검증
result = verifier.verify_function(code, spec)

# 모듈 레벨 검증
result = verifier.verify_module(module)

# 백엔드 레벨 검증
result = verifier.verify_backend(backend)
```

**검증 계층:**
- **L1 (함수)**: 개별 함수의 명세 준수 검증
- **L2 (모듈)**: 모듈 내 함수 간 계약 검증
- **L3 (백엔드)**: 전체 백엔드의 일관성 검증

### 5. integration/ - 통합 모듈

전체 파이프라인을 통합합니다.

```python
from src.integration import VEGAVerifiedPipeline, create_pipeline

# 파이프라인 생성
pipeline = create_pipeline(enable_repair=True)

# 실행
result = pipeline.run(code, references)
```

### 6. parsing/ - 파싱 모듈

Clang AST를 사용한 C++ 파싱을 지원합니다.

```python
from src.parsing import ClangASTParser

parser = ClangASTParser()
result = parser.parse_code(code)

# 함수 정보 추출
for func in result['functions']:
    print(f"함수: {func['name']}")
    print(f"파라미터: {func['parameters']}")
```

---

## 💻 사용 예제

### CLI 사용

```bash
# 검증
vega-verify verify --code function.cpp --spec spec.json

# 수리
vega-verify repair --code buggy.cpp --spec spec.json --strategy hybrid

# 실험 실행
vega-verify experiment --all
```

### Python API 사용

```python
# 전체 파이프라인 예제
from src.specification import Specification, SpecificationInferrer
from src.verification import Verifier
from src.repair import CGNREngine

# 1. 명세 추론
inferrer = SpecificationInferrer()
spec = inferrer.infer("getRelocType", [
    ("arm", arm_code),
    ("x86", x86_code)
])

# 2. 검증
verifier = Verifier(timeout_ms=30000)
result = verifier.verify(riscv_code, spec)

# 3. 수리 (검증 실패 시)
if not result.is_verified():
    cgnr = CGNREngine(verifier=verifier)
    repair_result = cgnr.repair(riscv_code, spec)
    if repair_result.is_successful():
        print("수리 성공!")
        print(repair_result.repaired_code)
```

---

## 📊 코드 통계

| 모듈 | 파일 수 | 코드 라인 |
|------|--------|----------|
| specification | 4 | 3,405 |
| verification | 6 | 7,037 |
| repair | 7 | 5,728 |
| hierarchical | 5 | 1,883 |
| integration | 4 | 3,987 |
| parsing | 2 | 1,423 |
| llvm_extraction | 3 | 4,568 |
| utils | 2 | 905 |
| **총합** | **33+** | **~33,000** |

---

## 🔗 관련 문서

- [메인 README](../README.md)
- [명령어 레퍼런스](../docs/COMMANDS_REFERENCE.md)
- [테스트 가이드](../tests/README.md)
