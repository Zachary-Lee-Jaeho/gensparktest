# VEGA-Verified: 의미적으로 검증된 신경망 컴파일러 백엔드 생성기

[![Tests](https://img.shields.io/badge/tests-150%20passing-brightgreen)]()
[![Phase](https://img.shields.io/badge/phase-2%20complete-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

> **논문 아티팩트**: 이 저장소는 VEGA-Verified 시스템의 구현 및 재현 자료를 포함합니다.

---

## 📋 목차

- [빠른 시작](#-빠른-시작)
- [구현 상태](#-구현-상태)
- [시스템 개요](#-시스템-개요)
- [설치 방법](#-설치-방법)
- [사용법](#-사용법)
- [테스트 실행](#-테스트-실행)
- [학습 실행](#-학습-실행)
- [프로젝트 구조](#-프로젝트-구조)
- [문서](#-문서)

---

## 🚀 빠른 시작

### 로컬 환경에서 실행 (Docker 없이)

```bash
# 1. 저장소 클론
git clone https://github.com/Zachary-Lee-Jaeho/gensparktest.git
cd gensparktest/webapp

# 2. 의존성 설치
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU용
pip install transformers accelerate

# 3. 패키지 설치
pip install -e .

# 4. 테스트 실행
python -m pytest tests/test_phase1_infrastructure.py tests/test_phase2_complete.py -v

# 5. CLI 확인
vega-verify --help
```

### Docker를 사용한 실행

```bash
# CPU용 이미지 빌드
docker build -f Dockerfile.unified -t vega-verified:cpu .

# 테스트 실행
docker run --rm vega-verified:cpu python -m pytest tests/ -v

# GPU용 이미지 빌드 및 실행
docker build -f Dockerfile.gpu -t vega-verified:gpu .
docker run --rm --gpus all vega-verified:gpu python -m pytest tests/ -v
```

---

## 📊 구현 상태

**최종 업데이트**: 2026-01-22

```
┌─────────────────────────────────────────────────────────────────┐
│                      구현 완성도 현황                              │
├─────────────────────────────────────────────────────────────────┤
│ 전체 완성도: ~90% (CPU MVP 기준)                                  │
│                                                                  │
│ ✅ 구조/인프라:              95% 완료                             │
│ ✅ 핵심 알고리즘 (CGNR, SMT):  95% 완료                           │
│ ✅ SMT 검증:                 100% 완료                            │
│ ✅ 명세 추론:                 85% 완료                             │
│ 🟡 Neural 컴포넌트:           45% 완료 (GPU 필요)                  │
│ ✅ 통합/테스트:               90% 완료                             │
│                                                                  │
│ 총 코드량: 33,000+ LOC (8개 모듈)                                 │
│ 테스트: 150개 핵심 테스트 통과                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 완전히 구현된 컴포넌트 (✅)

| 컴포넌트 | 파일 | 코드 라인 | 상태 |
|---------|------|----------|------|
| Neural Repair Engine | `neural_repair_engine.py` | 870 | ✅ GPU 준비 완료 |
| Symbolic Executor | `symbolic_exec.py` | 950+ | ✅ Z3 + Clang AST 통합 |
| SMT Solver | `smt_solver.py` | 550+ | ✅ 메모리 모델, 함수 호출 |
| Specification Language | `spec_language.py` | 510 | ✅ 완전 |
| CGNR Algorithm | `cgnr.py` | 340 | ✅ 통합 완료 |
| Switch Verifier | `switch_verifier.py` | 968 | ✅ 완전 |
| Fault Localizer | `fault_loc.py` | 400+ | ✅ 완전 |
| CLI Tool | `cli.py` | 1,200+ | ✅ 완전 |

### GPU 필요 컴포넌트 (🟡)

| 컴포넌트 | CPU 모드 | GPU 모드 |
|---------|---------|---------|
| NeuralRepairEngine | 규칙 기반 폴백 | CodeT5 추론 |
| 모델 학습 | Mock 학습 | 실제 학습 |

---

## 🔬 시스템 개요

VEGA-Verified는 VEGA 신경망 컴파일러 백엔드 생성기에 형식 검증 기능을 확장한 시스템입니다.

### 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                      VEGA-Verified 파이프라인                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │    LLVM      │───▶│   명세       │───▶│     SMT      │          │
│  │  추출기      │    │   추론       │    │   검증기     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  함수        │    │  기호적      │    │ 반례         │          │
│  │  데이터베이스│    │  실행        │    │  추출        │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                             │                   │                   │
│                             ▼                   ▼                   │
│                      ┌──────────────────────────────┐              │
│                      │      CGNR 수리 루프          │              │
│                      │  ┌────────────────────────┐  │              │
│                      │  │ NeuralRepairEngine     │  │              │
│                      │  │ ├─ CodeT5 (GPU)        │  │              │
│                      │  │ └─ 규칙 기반 (CPU)     │  │              │
│                      │  └────────────────────────┘  │              │
│                      └──────────────────────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 핵심 기능

- **Z3 기반 기호적 실행**: 경로 조건 만족도 검사
- **확장된 SMT 솔버**: 메모리 모델, 함수 호출, 루프 불변식
- **하이브리드 신경망 수리**: GPU 신경망 + CPU 규칙 기반 폴백
- **통합 CGNR**: 반례 유도 수리 파이프라인

---

## 📦 설치 방법

### 방법 1: 로컬 설치 (권장)

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# 핵심 의존성 설치
pip install -r requirements.txt

# PyTorch 설치 (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# PyTorch 설치 (GPU - CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Transformers 설치
pip install transformers accelerate

# 패키지 설치
pip install -e .

# 설치 확인
vega-verify status
```

### 방법 2: Docker 사용

```bash
# CPU용
docker build -f Dockerfile.unified -t vega-verified:cpu .

# GPU용
docker build -f Dockerfile.gpu -t vega-verified:gpu .
```

---

## 🖥️ 사용법

### CLI 명령어

```bash
# 도움말
vega-verify --help

# 시스템 상태 확인
vega-verify status

# 함수 검증
vega-verify verify --code function.cpp --spec spec.json

# 버그 수리
vega-verify repair --code buggy.cpp --spec spec.json --strategy hybrid

# 실험 실행
vega-verify experiment --all
vega-verify experiment --experiment verification --backend riscv
```

### Python API 사용

```python
# 검증
from src.verification import Verifier
from src.specification import Specification

verifier = Verifier(timeout_ms=30000)
spec = Specification(function_name="getRelocType")
result = verifier.verify(code, spec)
print(f"검증됨: {result.is_verified()}")

# 수리
from src.repair import CGNREngine

cgnr = CGNREngine(verifier=verifier, max_iterations=5)
repair_result = cgnr.repair(buggy_code, spec)
if repair_result.is_successful():
    print(repair_result.repaired_code)
```

---

## 🧪 테스트 실행

### 로컬 환경

```bash
# 모든 테스트
python -m pytest tests/ -v

# 핵심 테스트만 (150개)
python -m pytest tests/test_phase1_infrastructure.py tests/test_phase2_complete.py tests/integration/ -v

# 단위 테스트
python -m pytest tests/unit/ -v

# 통합 테스트
python -m pytest tests/integration/ -v

# 특정 패턴 테스트
python -m pytest tests/ -v -k "verification"
```

### Docker 환경

```bash
# CPU
docker run --rm vega-verified:cpu python -m pytest tests/ -v

# GPU
docker run --rm --gpus all vega-verified:gpu python -m pytest tests/ -v
```

---

## 🎯 학습 실행

### 빠른 테스트 (CPU)

```bash
# 최소 테스트 (10개 샘플, 1 에폭)
python scripts/train_neural_repair.py --test-only
```

### 빠른 테스트 (GPU)

```bash
python scripts/train_neural_repair.py --test-only --device cuda
```

### 전체 학습 (CPU) - 느림

```bash
./scripts/run_full_training.sh --cpu --epochs 10
```

### 전체 학습 (GPU) - 권장

```bash
./scripts/run_full_training.sh --gpu --epochs 10
```

### 체크포인트에서 재개

```bash
python scripts/train_neural_repair.py --resume models/repair_model/checkpoint-500 --epochs 5
```

### Docker에서 학습

```bash
# CPU
docker run --rm -v $(pwd)/models:/app/models vega-verified:cpu \
    python scripts/train_neural_repair.py --test-only

# GPU
docker run --rm --gpus all -v $(pwd)/models:/app/models vega-verified:gpu \
    ./scripts/run_full_training.sh --gpu --epochs 10
```

---

## 🔬 학습된 모델로 실험하기

모델 학습이 완료되면 `models/repair_model/final/` 디렉토리에 학습된 모델이 저장됩니다.

### 학습된 모델 확인

```bash
# 모델 파일 확인
ls -la models/repair_model/final/
# 예상 출력:
# config.json
# model.safetensors (또는 pytorch_model.bin)
# tokenizer_config.json
# tokenizer.json
```

### 학습된 모델로 실험 실행

```bash
# 기본: 학습된 모델로 repair 실험 (GPU 권장)
vega-verify experiment --experiment repair --model-path models/repair_model/final --device cuda

# CPU에서 실행 (느림, 테스트용)
vega-verify experiment --experiment repair --model-path models/repair_model/final --device cpu

# 전체 실험 + 학습된 모델
vega-verify experiment --all --model-path models/repair_model/final --device cuda

# 샘플 크기 조절
vega-verify experiment --experiment repair --model-path models/repair_model/final --device cuda --sample-size 200

# 특정 백엔드만 테스트
vega-verify experiment --experiment repair --model-path models/repair_model/final --device cuda --backend riscv
```

### Python API로 학습된 모델 사용

```python
from src.repair import NeuralRepairEngine, NeuralRepairConfig

# 학습된 모델 로드
config = NeuralRepairConfig(
    model_path="models/repair_model/final",
    model_name="Salesforce/codet5-large",  # 학습 시 사용한 모델
    device="cuda"  # 또는 "cpu"
)
engine = NeuralRepairEngine(config)
engine.load()

# 버그 수리
buggy_code = '''
switch (Kind) {
    case FK_Data_4: return R_X86_64_32;
    default: return R_X86_64_NONE;
}
'''
counterexample = {
    'Kind': 'FK_Data_8',
    'expected': 'R_X86_64_64',
    'actual': 'R_X86_64_NONE'
}

candidates = engine.repair(buggy_code, counterexample, num_candidates=5)
for i, (code, confidence) in enumerate(candidates):
    print(f"후보 {i+1} (신뢰도: {confidence:.3f}):")
    print(code)
```

### Docker에서 실험

```bash
# 학습된 모델이 models/ 디렉토리에 있을 때
docker run --rm --gpus all \
    -v $(pwd)/models:/app/models \
    vega-verified:gpu \
    vega-verify experiment --experiment repair --model-path /app/models/repair_model/final --device cuda
```

### 참고: 모델 없이 실행

`--model-path`를 지정하지 않으면 **규칙 기반 폴백(rule-based fallback)**을 사용합니다:

```bash
# 규칙 기반 폴백 사용 (Neural 모델 없음)
vega-verify experiment --experiment repair
```

---

## 📁 프로젝트 구조

```
webapp/
├── README.md                    # 이 파일
├── requirements.txt             # Python 의존성
├── setup.py                     # 패키지 설치
├── Dockerfile.unified           # CPU용 Docker 이미지
├── Dockerfile.gpu               # GPU용 Docker 이미지
│
├── src/                         # 소스 코드 (README 참조)
│   ├── cli.py                   # CLI 진입점
│   ├── specification/           # 명세 언어 및 추론
│   ├── verification/            # SMT 검증
│   ├── repair/                  # CGNR 및 Neural 수리
│   ├── hierarchical/            # 계층적 검증
│   ├── integration/             # 파이프라인 통합
│   ├── parsing/                 # Clang AST 파서
│   └── llvm_extraction/         # LLVM 함수 추출
│
├── tests/                       # 테스트 코드 (README 참조)
│   ├── test_phase1_infrastructure.py
│   ├── test_phase2_complete.py
│   ├── unit/                    # 단위 테스트
│   └── integration/             # 통합 테스트
│
├── scripts/                     # 스크립트 (README 참조)
│   ├── train_neural_repair.py   # 학습 스크립트
│   ├── run_full_training.sh     # 전체 학습 실행
│   └── reproduce_experiments.sh # 논문 재현
│
├── configs/                     # 설정 파일 (README 참조)
│   └── default.yaml             # 기본 설정
│
├── docs/                        # 문서 (README 참조)
│   ├── COMMANDS_REFERENCE.md    # 명령어 레퍼런스
│   └── IMPLEMENTATION_TASKS_100_PERCENT.md
│
├── data/                        # 데이터 파일
├── models/                      # 학습된 모델 저장
└── results/                     # 실험 결과 저장
```

---

## 📚 문서

| 문서 | 설명 |
|------|------|
| [docs/COMMANDS_REFERENCE.md](docs/COMMANDS_REFERENCE.md) | 모든 명령어 상세 레퍼런스 |
| [docs/IMPLEMENTATION_TASKS_100_PERCENT.md](docs/IMPLEMENTATION_TASKS_100_PERCENT.md) | 구현 작업 및 YAML 설정 가이드 |
| [docs/IMPLEMENTATION_VS_DESIGN_REPORT.md](docs/IMPLEMENTATION_VS_DESIGN_REPORT.md) | 설계 대비 구현 비교 |
| [src/README.md](src/README.md) | 소스 코드 구조 설명 |
| [tests/README.md](tests/README.md) | 테스트 실행 가이드 |
| [scripts/README.md](scripts/README.md) | 스크립트 사용법 |
| [configs/README.md](configs/README.md) | 설정 파일 가이드 |

---

## 📈 테스트 결과

```
테스트 현황 (2026-01-22)
├── 핵심 테스트 (Phase1 + Phase2 + Integration): 150 통과
├── 통합 테스트: 78 통과
├── 전체 통과 테스트: 258개
└── 실패: 0개 (핵심 테스트 기준)
```

---

## 🔗 참고 자료

1. Zhong et al., "VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model", CGO 2025
2. [LLVM Documentation](https://llvm.org/docs/)
3. [Z3 Solver Guide](https://microsoft.github.io/z3guide/)

---

## 📜 라이선스

MIT License

---

## 📧 문의

질문이 있으시면 GitHub Issue를 열어주세요.
