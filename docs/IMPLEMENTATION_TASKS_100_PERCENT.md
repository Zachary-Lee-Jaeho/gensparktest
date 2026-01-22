# VEGA-Verified 100% 달성 Task 계획 및 YAML 설정 가이드

**작성일**: 2026-01-22  
**현재 진행률**: 종합 ~85-90%

---

## 1. YAML 하드코딩 항목 분석

### 1.1 현재 `configs/default.yaml`에 정의된 항목들

| 섹션 | 항목 | 현재 값 | 하드코딩 권장 여부 | 이유 |
|------|------|---------|-------------------|------|
| **specification** | max_depth | 100 | ✅ 하드코딩 | 대부분의 함수에서 적정 깊이 |
| | timeout_ms | 60000 | ✅ 하드코딩 | 1분 타임아웃이 표준 |
| | min_references | 1 | ✅ 하드코딩 | 최소 1개 레퍼런스 필요 |
| | min_similarity | 0.7 | ✅ 하드코딩 | 70% 유사도 임계값 |
| | min_confidence | 0.5 | ✅ 하드코딩 | 50% 신뢰도 임계값 |
| **verification** | solver | "z3" | ✅ 하드코딩 | Z3가 기본 SMT 솔버 |
| | timeout_ms | 30000 | ✅ 하드코딩 | 30초 검증 타임아웃 |
| | bmc.max_bound | 10 | ✅ 하드코딩 | BMC 기본 bound |
| | incremental | true | ✅ 하드코딩 | 성능 최적화 |
| **repair** | max_iterations | 5 | ✅ 하드코딩 | CGNR 기본 반복 |
| | beam_size | 5 | ✅ 하드코딩 | 후보 생성 수 |
| | temperature | 0.7 | ⚠️ 조건부 | GPU에서 조정 가능 |
| | model_type | "hybrid" | ✅ 하드코딩 | CPU+GPU 지원 |
| **hierarchical** | levels | [function, module, backend] | ✅ 하드코딩 | 3-level 검증 |
| | max_workers | 4 | ⚠️ 환경별 | CPU 코어에 따라 조정 |
| **parsing** | parser | "tree_sitter" | ⚠️ 조건부 | libclang 있으면 "clang" |
| | cpp_standard | "c++17" | ✅ 하드코딩 | LLVM 요구사항 |

### 1.2 하드코딩 권장 값 요약

```yaml
# configs/default.yaml - 하드코딩 권장 항목
specification:
  max_depth: 100          # 심볼릭 실행 최대 깊이 (고정)
  timeout_ms: 60000       # 1분 타임아웃 (고정)
  min_references: 1       # 최소 레퍼런스 수 (고정)
  min_similarity: 0.7     # 패턴 매칭 임계값 (고정)
  min_confidence: 0.5     # 신뢰도 임계값 (고정)

verification:
  solver: "z3"            # SMT 솔버 (고정 - Z3만 지원)
  timeout_ms: 30000       # 30초 타임아웃 (고정)
  bmc:
    enabled: true
    max_bound: 10         # BMC bound (고정)
  incremental: true       # 인크리멘탈 솔빙 (고정)

repair:
  max_iterations: 5       # CGNR 반복 횟수 (고정)
  beam_size: 5            # 후보 수 (고정)
  temperature: 0.7        # 생성 온도 (GPU에서 조정 가능)
  model_type: "hybrid"    # 하이브리드 모드 (고정)

hierarchical:
  levels:
    - function
    - module
    - backend
  parallel_verification: false  # CPU에서는 false 권장
  max_workers: 4                # 환경별 조정

parsing:
  cpp_standard: "c++17"   # LLVM 표준 (고정)
```

### 1.3 환경별 조정 필요 항목

| 항목 | CPU 환경 | GPU 환경 | Docker |
|------|---------|---------|--------|
| `repair.temperature` | 0.7 | 0.5-0.9 조정 가능 | 환경 변수로 오버라이드 |
| `hierarchical.max_workers` | 2-4 | 4-8 | 컨테이너 리소스에 따라 |
| `parsing.parser` | tree_sitter | clang (libclang 필요시) | clang 기본 |

---

## 2. CPU/GPU 환경 분리 계획

### 2.1 CPU MVP 구성 (현재 완료)

```bash
# CPU 환경 설치 (완료됨)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate

# CPU 테스트 실행
cd /home/jaeho/Projects/gensparktest/webapp
python scripts/train_neural_repair.py --test-only
```

### 2.2 GPU 서버 이동 시 설정

```bash
# GPU 환경 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate

# GPU 사용 여부 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2.3 Docker 명령어

```bash
# CPU Docker 빌드 및 실행
docker build -f Dockerfile.unified -t vega-verified:cpu .
docker run -it --rm vega-verified:cpu vega-verify --help

# GPU Docker 빌드 및 실행 (NVIDIA Container Toolkit 필요)
docker build -f Dockerfile.gpu -t vega-verified:gpu .
docker run -it --rm --gpus all vega-verified:gpu vega-verify --help

# 전체 학습 실행 (GPU)
docker run -it --rm --gpus all \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    vega-verified:gpu ./scripts/run_full_training.sh --gpu --epochs 10

# 빠른 테스트만 실행 (CPU)
docker run -it --rm vega-verified:cpu \
    python scripts/train_neural_repair.py --test-only
```

### 2.4 체크포인트/재개 기능

```python
# scripts/train_neural_repair.py에서 지원:
# --resume: 마지막 체크포인트에서 재개
# --checkpoint-dir: 체크포인트 저장 위치 지정

python scripts/train_neural_repair.py \
    --resume \
    --checkpoint-dir models/repair_model/checkpoint-latest \
    --epochs 5
```

---

## 3. 100% 달성을 위한 세부 Task 정의

### Task 목록 및 진행 상태

| Task ID | 제목 | 현재 상태 | 예상 소요 | 의존성 |
|---------|------|----------|----------|--------|
| T1 | SpecificationInferrer 빈 조건 처리 개선 | 🔴 미완료 | 30분 | 없음 |
| T2 | Config ↔ YAML 동기화 완성 | 🟡 진행중 | 20분 | 없음 |
| T3 | Clang AST ↔ SymbolicExecutor 통합 강화 | ✅ 완료 | - | 없음 |
| T4 | Neural Training 체크포인트/재개 | ✅ 완료 | - | 없음 |
| T5 | Integration Test 실패 수정 | 🔴 미완료 | 45분 | T1 |
| T6 | GPU Dockerfile 추가 | 🔴 미완료 | 30분 | 없음 |
| T7 | 전체 테스트 통과 확인 | 🔴 미완료 | 15분 | T1, T5 |

### T1: SpecificationInferrer 빈 조건 처리 개선

**문제**: `_validate_spec`에서 preconditions/postconditions가 모두 비어있으면 예외 발생  
**해결**: 최소한의 기본 조건 생성 또는 경고로 처리 변경

```python
# src/specification/inferrer.py 수정 필요
def _validate_spec(self, spec, functions):
    if not spec.preconditions and not spec.postconditions:
        # 예외 대신 기본 조건 생성 또는 경고
        if spec.invariants:
            # invariants가 있으면 유효한 스펙으로 처리
            return
        # 기본 precondition 추가: true (항상 참)
        spec.preconditions.append("true")
        spec.postconditions.append("result != undefined")
```

### T2: Config ↔ YAML 동기화

**문제**: `src/utils/config.py`와 `configs/default.yaml` 간 필드명 불일치  
**해결**: 필드명 매핑 추가

### T5: Integration Test 수정

**영향받는 테스트**:
- `test_statistics_tracking`
- `test_auto_level_detection`
- `test_complete_riscv_backend_verification`
- `test_verify_with_bmc`
- `test_full_pipeline_compiler_backend_function`

**원인**: 모두 `SpecificationInferrer.infer()`에서 빈 조건으로 인한 ValueError

### T6: GPU Dockerfile 추가

```dockerfile
# Dockerfile.gpu (생성 필요)
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
# ... GPU 전용 설정
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Import 오류 현황

### 4.1 해결된 오류

| 오류 | 해결 방법 | 상태 |
|------|----------|------|
| `cannot import 'VerificationLevel'` | `__init__.py`에 export 추가 | ✅ 해결됨 |
| `accelerate>=0.26.0 required` | `pip install accelerate` | ✅ 해결됨 |

### 4.2 현재 테스트 결과

```
Integration Tests: 73 passed, 5 failed
Core Tests (Phase 1+2): 72 passed, 0 failed
Total: 145+ tests
```

### 4.3 남은 실패 원인

5개 실패 모두 **동일 원인**:
```
ValueError: Could not infer any conditions for specification
```

이는 import 오류가 아니라 **로직 문제**로, T1에서 해결 예정.

---

## 5. 실행 명령어 요약

### 5.1 로컬 CPU 환경

```bash
# 설치
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate

# 테스트 실행
python -m pytest tests/ -v

# Neural 빠른 테스트
python scripts/train_neural_repair.py --test-only
```

### 5.2 GPU 서버

```bash
# GPU PyTorch 설치
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 전체 학습 실행
./scripts/run_full_training.sh --gpu --epochs 10

# 체크포인트에서 재개
python scripts/train_neural_repair.py --resume --epochs 5
```

### 5.3 Docker (CPU)

```bash
docker build -f Dockerfile.unified -t vega-verified .
docker run -it --rm vega-verified python -m pytest tests/ -v
docker run -it --rm vega-verified vega-verify experiment --all
```

### 5.4 Docker (GPU)

```bash
docker build -f Dockerfile.gpu -t vega-verified:gpu .
docker run -it --rm --gpus all \
    -v $(pwd)/models:/app/models \
    vega-verified:gpu ./scripts/run_full_training.sh --gpu
```

---

## 6. 100% 달성 기준

### CPU MVP 기준 (현재 타겟)
- [x] 72개 Core Test 통과
- [x] Neural 컴포넌트 CPU 로드/테스트 성공
- [x] Clang AST 파서 통합
- [ ] 78개 Integration Test 전체 통과 (5개 실패 수정 필요)
- [x] 체크포인트/재개 기능

### GPU 환경 기준 (추후)
- [ ] 전체 Neural Training 완료
- [ ] 학습된 모델로 Repair 정확도 검증
- [ ] End-to-end 파이프라인 GPU 실행

---

## 7. 다음 단계

1. **즉시 실행**: T1 (SpecificationInferrer 수정)으로 5개 테스트 실패 해결
2. **단기**: T6 (GPU Dockerfile) 추가
3. **중기**: GPU 환경에서 전체 학습 실행
4. **장기**: VEGA 비교 실험 (현재 보류)
