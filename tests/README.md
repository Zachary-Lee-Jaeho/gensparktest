# tests/ - 테스트 디렉토리

이 디렉토리는 VEGA-Verified 시스템의 모든 테스트 코드를 포함합니다.

---

## 📁 디렉토리 구조

```
tests/
├── test_phase1_infrastructure.py    # Phase 1: 인프라 테스트 (76개)
├── test_phase2_complete.py          # Phase 2: 통합 테스트 (47개)
├── test_switch_verifier.py          # Switch 검증기 테스트
├── test_switch_repair.py            # Switch 수리 테스트
├── test_llvm_infrastructure.py      # LLVM 인프라 테스트
├── test_llvm_extraction.py          # LLVM 추출 테스트
│
├── unit/                            # 단위 테스트
│   ├── test_verification.py         # 검증 모듈 테스트
│   ├── test_specification.py        # 명세 모듈 테스트
│   └── test_bmc.py                  # BMC 테스트
│
├── integration/                     # 통합 테스트
│   ├── test_pipeline.py             # 파이프라인 테스트
│   ├── test_verification_pipeline.py # 검증 파이프라인 테스트
│   ├── test_hierarchical_verification.py # 계층적 검증 테스트
│   └── test_comprehensive_pipeline.py # 종합 파이프라인 테스트
│
├── benchmarks/                      # 벤치마크 테스트
│   └── test_benchmarks.py
│
├── vega_simulator.py                # VEGA 시뮬레이터
└── vega_verified_prototype.py       # 프로토타입 코드
```

---

## 🧪 테스트 실행 방법

### 모든 테스트 실행

```bash
# 전체 테스트
python -m pytest tests/ -v

# 병렬 실행 (더 빠름)
pip install pytest-xdist
python -m pytest tests/ -v -n auto
```

### 핵심 테스트만 실행 (권장)

```bash
# 핵심 테스트 150개
python -m pytest \
    tests/test_phase1_infrastructure.py \
    tests/test_phase2_complete.py \
    tests/integration/ \
    -v
```

### 단위 테스트

```bash
# 모든 단위 테스트
python -m pytest tests/unit/ -v

# 특정 모듈 테스트
python -m pytest tests/unit/test_verification.py -v
python -m pytest tests/unit/test_specification.py -v
python -m pytest tests/unit/test_bmc.py -v
```

### 통합 테스트

```bash
# 모든 통합 테스트
python -m pytest tests/integration/ -v

# 특정 파일 테스트
python -m pytest tests/integration/test_pipeline.py -v
python -m pytest tests/integration/test_hierarchical_verification.py -v
```

### Phase별 테스트

```bash
# Phase 1: 인프라 (76개 테스트)
python -m pytest tests/test_phase1_infrastructure.py -v

# Phase 2: 통합 (47개 테스트)
python -m pytest tests/test_phase2_complete.py -v
```

### 특정 패턴 테스트

```bash
# "verification" 포함 테스트
python -m pytest tests/ -v -k "verification"

# "repair" 포함 테스트
python -m pytest tests/ -v -k "repair"

# "neural" 포함 테스트
python -m pytest tests/ -v -k "neural"

# 특정 클래스 테스트
python -m pytest tests/integration/test_pipeline.py::TestPipelineConfig -v

# 특정 메서드 테스트
python -m pytest tests/integration/test_pipeline.py::TestPipelineConfig::test_default_config -v
```

---

## 📊 테스트 현황

```
테스트 현황 (2026-01-22)
─────────────────────────────────
핵심 테스트: 150 통과 ✅
├── Phase 1 Infrastructure: 76 통과
├── Phase 2 Complete: 47 통과
└── Integration: 78 통과 (중복 포함)

전체 테스트: 258 통과
└── 일부 unit 테스트는 API 변경으로 조정 필요
```

---

## 🔍 주요 테스트 설명

### test_phase1_infrastructure.py

기본 인프라 구성요소 테스트:
- 명세 언어 (Specification, Condition)
- 검증기 기본 기능
- 파서 동작
- 유틸리티 함수

### test_phase2_complete.py

통합 기능 테스트:
- CGNR 알고리즘
- 신경망 수리 엔진
- SMT 솔버 확장
- 전체 파이프라인

### integration/test_pipeline.py

파이프라인 통합 테스트:
- 설정 관리
- 검증 흐름
- 배치 처리
- 통계 수집

### integration/test_hierarchical_verification.py

계층적 검증 테스트:
- L1 함수 레벨 검증
- L2 모듈 레벨 검증
- L3 백엔드 레벨 검증
- 자동 레벨 감지

---

## 📝 테스트 작성 가이드

### 새 테스트 추가

```python
# tests/unit/test_my_feature.py
import pytest
from src.my_module import MyClass

class TestMyClass:
    """MyClass 테스트."""
    
    @pytest.fixture
    def instance(self):
        """테스트 인스턴스 생성."""
        return MyClass()
    
    def test_basic_functionality(self, instance):
        """기본 기능 테스트."""
        result = instance.do_something()
        assert result is not None
    
    def test_edge_case(self, instance):
        """엣지 케이스 테스트."""
        with pytest.raises(ValueError):
            instance.do_something(invalid_input)
```

### 테스트 실행 옵션

```bash
# 상세 출력
python -m pytest tests/ -v -s

# 첫 실패시 중단
python -m pytest tests/ -v -x

# 실패 테스트만 재실행
python -m pytest tests/ -v --lf

# 커버리지 리포트
python -m pytest tests/ --cov=src --cov-report=html
```

---

## 🐳 Docker에서 테스트

```bash
# CPU 환경
docker run --rm vega-verified:cpu python -m pytest tests/ -v

# GPU 환경
docker run --rm --gpus all vega-verified:gpu python -m pytest tests/ -v

# 특정 테스트만
docker run --rm vega-verified:cpu python -m pytest tests/integration/ -v
```

---

## ⚠️ 알려진 이슈

1. **unit 테스트 일부 실패**: API 변경으로 인해 일부 단위 테스트 조정 필요
2. **GPU 테스트**: GPU 환경에서만 실행 가능한 테스트 존재
3. **시간 초과**: 일부 통합 테스트는 느린 환경에서 타임아웃 가능

---

## 🔗 관련 문서

- [메인 README](../README.md)
- [소스 코드 가이드](../src/README.md)
- [명령어 레퍼런스](../docs/COMMANDS_REFERENCE.md)
