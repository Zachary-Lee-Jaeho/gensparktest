# configs/ - 설정 파일 디렉토리

이 디렉토리는 VEGA-Verified 시스템의 설정 파일을 포함합니다.

---

## 📁 디렉토리 구조

```
configs/
├── README.md                # 이 파일
├── default.yaml             # 기본 설정 파일
└── targets/                 # 타겟별 설정 (선택)
    └── riscv.yaml           # RISC-V 타겟 설정
```

---

## ⚙️ default.yaml - 기본 설정

### 설정 구조

```yaml
# 시스템 설정
system:
  name: "vega-verified"
  version: "0.1.0"
  log_level: "INFO"              # DEBUG, INFO, WARNING, ERROR
  log_file: "logs/vega_verified.log"

# 명세 추론 설정
specification:
  enabled: true
  min_references: 1              # 최소 참조 구현 수
  max_depth: 100                 # 기호적 실행 최대 깊이
  timeout_ms: 60000              # 타임아웃 (밀리초)
  min_similarity: 0.7            # 패턴 매칭 유사도 임계값
  min_confidence: 0.5            # 신뢰도 임계값

# 검증 설정
verification:
  solver: "z3"                   # SMT 솔버 (z3만 지원)
  timeout_ms: 30000              # 검증 타임아웃
  incremental: true              # 인크리멘탈 솔빙
  bmc:
    enabled: true
    max_bound: 10                # BMC 최대 bound
    auto_bound_detection: true

# 수리 설정 (CGNR)
repair:
  enabled: true
  max_iterations: 5              # 최대 반복 횟수
  beam_size: 5                   # 후보 생성 수
  temperature: 0.7               # 생성 온도
  model_type: "hybrid"           # rule_based, neural, hybrid

# 계층적 검증 설정
hierarchical:
  levels:
    - function                   # L1: 함수 레벨
    - module                     # L2: 모듈 레벨
    - backend                    # L3: 백엔드 레벨
  parallel_verification: false   # 병렬 검증
  max_workers: 4                 # 워커 수

# 파싱 설정
parsing:
  parser: "tree_sitter"          # tree_sitter 또는 clang
  cpp_standard: "c++17"          # C++ 표준
```

---

## 🔧 설정 사용법

### Python에서 사용

```python
from src.utils.config import load_config, Config

# 설정 파일 로드
config = load_config("configs/default.yaml")

# 또는 기본 설정 사용
config = Config()

# 설정 값 접근
print(config.verification.timeout_ms)  # 30000
print(config.repair.max_iterations)    # 5
```

### CLI에서 설정 지정

```bash
# 기본 설정 사용
vega-verify verify --code function.cpp

# 타임아웃 오버라이드
vega-verify verify --code function.cpp --timeout 60000
```

---

## 📝 하드코딩 권장 항목

다음 항목은 대부분의 경우 기본값을 유지하는 것을 권장합니다:

| 항목 | 권장 값 | 이유 |
|------|--------|------|
| `specification.max_depth` | 100 | 대부분 함수에서 적정 |
| `specification.timeout_ms` | 60000 | 1분 표준 |
| `verification.solver` | "z3" | Z3만 지원 |
| `verification.timeout_ms` | 30000 | 30초 표준 |
| `verification.bmc.max_bound` | 10 | BMC 기본 |
| `repair.max_iterations` | 5 | CGNR 기본 |
| `repair.model_type` | "hybrid" | CPU+GPU 지원 |
| `parsing.cpp_standard` | "c++17" | LLVM 요구사항 |

### 환경별 조정 필요 항목

| 항목 | CPU | GPU | 설명 |
|------|-----|-----|------|
| `repair.temperature` | 0.7 | 0.5-0.9 | 생성 다양성 |
| `hierarchical.max_workers` | 2 | 4-8 | 병렬 처리 |
| `parsing.parser` | tree_sitter | clang | libclang 필요시 |

---

## 📂 타겟별 설정

`targets/` 디렉토리에 백엔드별 설정을 추가할 수 있습니다:

```yaml
# configs/targets/riscv.yaml
target:
  name: "RISCV"
  triple: "riscv64-unknown-linux-gnu"
  
  # RISC-V 특정 설정
  verification:
    timeout_ms: 45000    # RISC-V는 더 긴 타임아웃
    
  # 참조 백엔드
  reference_backends:
    - ARM
    - MIPS
    - X86
```

### 타겟 설정 사용

```bash
# CLI에서 타겟 지정
vega-verify verify --code function.cpp --backend riscv

# Python에서 타겟 설정 로드
config = load_config("configs/targets/riscv.yaml")
```

---

## 🔗 관련 문서

- [메인 README](../README.md)
- [명령어 레퍼런스](../docs/COMMANDS_REFERENCE.md)
- [구현 작업 가이드](../docs/IMPLEMENTATION_TASKS_100_PERCENT.md)
