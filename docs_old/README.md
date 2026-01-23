# docs/ - 문서 디렉토리

이 디렉토리는 VEGA-Verified 시스템의 문서를 포함합니다.

---

## 📁 문서 목록

```
docs/
├── README.md                              # 이 파일
├── COMMANDS_REFERENCE.md                  # 명령어 레퍼런스 (전체)
├── IMPLEMENTATION_TASKS_100_PERCENT.md    # 구현 작업 및 진행 상황
├── IMPLEMENTATION_VS_DESIGN_REPORT.md     # 설계 대비 구현 비교
└── IMPLEMENTATION_COMPLETION_ANALYSIS.md  # 완성도 분석
```

---

## 📚 문서 설명

### COMMANDS_REFERENCE.md - 명령어 레퍼런스

**모든 명령어의 상세 레퍼런스입니다.**

포함 내용:
- 환경 설정 (CPU/GPU)
- 테스트 명령어
- 학습 명령어
- 검증/수리 명령어
- 실험/평가 명령어
- Docker 명령어
- 개발 명령어
- 문제 해결

```bash
# 예시: 빠른 테스트
python scripts/train_neural_repair.py --test-only

# 예시: Docker GPU 학습
docker run --gpus all -v $(pwd)/models:/app/models vega-verified:gpu \
    ./scripts/run_full_training.sh --gpu
```

### IMPLEMENTATION_TASKS_100_PERCENT.md - 구현 작업 가이드

**100% 완성을 위한 작업 목록입니다.**

포함 내용:
- YAML 하드코딩 권장 항목
- CPU/GPU 환경 분리 계획
- 세부 Task 정의 및 상태
- Docker 명령어 정리

### IMPLEMENTATION_VS_DESIGN_REPORT.md - 설계 대비 구현 비교

**설계 문서와 실제 구현의 비교 분석입니다.**

포함 내용:
- 전체 완성도 현황
- 모듈별 구현 상태
- Mock/Placeholder 컴포넌트 목록
- Gap 분석 및 권장 사항

### IMPLEMENTATION_COMPLETION_ANALYSIS.md - 완성도 상세 분석

**구현 완성도의 상세 분석입니다.**

포함 내용:
- 컴포넌트별 완성도
- 100% 미달 사유
- 향후 개선 방향

---

## 🔍 빠른 참조

### 테스트 실행

```bash
# 핵심 테스트
python -m pytest tests/test_phase1_infrastructure.py tests/test_phase2_complete.py -v

# 전체 테스트
python -m pytest tests/ -v
```

### 학습 실행

```bash
# CPU 빠른 테스트
python scripts/train_neural_repair.py --test-only

# GPU 전체 학습
./scripts/run_full_training.sh --gpu --epochs 10
```

### CLI 사용

```bash
# 도움말
vega-verify --help

# 검증
vega-verify verify --code function.cpp --spec spec.json

# 수리
vega-verify repair --code buggy.cpp --spec spec.json
```

### Docker 사용

```bash
# CPU 이미지 빌드
docker build -f Dockerfile.unified -t vega-verified:cpu .

# GPU 이미지 빌드
docker build -f Dockerfile.gpu -t vega-verified:gpu .

# 테스트 실행
docker run --rm vega-verified:cpu python -m pytest tests/ -v
```

---

## 📊 구현 상태 요약

```
전체 완성도: ~90% (CPU MVP 기준)

✅ 완료:
- 구조/인프라: 95%
- 핵심 알고리즘: 95%
- SMT 검증: 100%
- 명세 추론: 85%
- 통합/테스트: 90%

🟡 진행 중:
- Neural 컴포넌트: 45% (GPU 필요)
```

---

## 🔗 관련 문서

- [메인 README](../README.md)
- [소스 코드 가이드](../src/README.md)
- [테스트 가이드](../tests/README.md)
- [스크립트 가이드](../scripts/README.md)
