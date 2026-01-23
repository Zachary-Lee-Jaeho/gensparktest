# scripts/ - 실행 스크립트 디렉토리

이 디렉토리는 학습, 실험 재현, 자동화를 위한 스크립트를 포함합니다.

---

## 📁 파일 목록

```
scripts/
├── train_neural_repair.py      # 신경망 수리 모델 학습
├── run_full_training.sh        # 전체 학습 실행 (셸 래퍼)
└── reproduce_experiments.sh    # 논문 실험 재현
```

---

## 🎯 train_neural_repair.py - 신경망 학습 스크립트

CodeT5 기반 신경망 수리 모델을 학습합니다.

### 기본 사용법

```bash
# 빠른 테스트 (CPU, 10개 샘플, 1 에폭)
python scripts/train_neural_repair.py --test-only

# GPU 빠른 테스트
python scripts/train_neural_repair.py --test-only --device cuda
```

### 전체 학습

```bash
# CPU 학습 (매우 느림 - 수 시간 소요)
python scripts/train_neural_repair.py \
    --device cpu \
    --epochs 10 \
    --batch-size 4 \
    --train-size 1000

# GPU 학습 (권장)
python scripts/train_neural_repair.py \
    --device cuda \
    --epochs 10 \
    --batch-size 16 \
    --train-size 1000 \
    --fp16
```

### 체크포인트에서 재개

```bash
# 마지막 체크포인트에서 재개
python scripts/train_neural_repair.py \
    --resume models/repair_model/checkpoint-500 \
    --epochs 5

# GPU에서 재개
python scripts/train_neural_repair.py \
    --resume models/repair_model/checkpoint-500 \
    --device cuda \
    --epochs 10
```

### 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|-------|------|
| `--device` | auto | 장치 선택 (auto, cpu, cuda, mps) |
| `--model` | Salesforce/codet5-base | HuggingFace 모델명 |
| `--epochs` | 5 | 학습 에폭 수 |
| `--batch-size` | 8 | 배치 크기 |
| `--train-size` | 100 | 학습 샘플 수 |
| `--output-dir` | models/repair_model | 출력 디렉토리 |
| `--resume` | - | 체크포인트 경로 |
| `--fp16` | False | FP16 혼합 정밀도 (GPU만) |
| `--test-only` | False | 최소 테스트만 실행 |
| `--learning-rate` | 5e-5 | 학습률 |

---

## 🚀 run_full_training.sh - 전체 학습 셸 스크립트

`train_neural_repair.py`를 편리하게 실행하는 래퍼 스크립트입니다.

### 사용법

```bash
# GPU 학습 (권장)
./scripts/run_full_training.sh --gpu

# CPU 학습
./scripts/run_full_training.sh --cpu

# 체크포인트에서 재개
./scripts/run_full_training.sh --gpu --resume models/repair_model/checkpoint-500

# 커스텀 옵션
./scripts/run_full_training.sh --gpu --epochs 20 --batch-size 32
```

### 옵션

```
--gpu           GPU(CUDA) 사용 (권장)
--cpu           CPU만 사용 (느림)
--resume PATH   체크포인트에서 재개
--epochs N      에폭 수 (기본: 10)
--batch-size N  배치 크기 (기본: GPU 16, CPU 4)
--model NAME    모델명 (기본: Salesforce/codet5-base)
--output DIR    출력 디렉토리 (기본: models/repair_model)
--train-size N  학습 샘플 수 (기본: 1000)
--help          도움말 표시
```

### Docker에서 실행

```bash
# CPU
docker run --rm -v $(pwd)/models:/app/models vega-verified:cpu \
    ./scripts/run_full_training.sh --cpu

# GPU
docker run --rm --gpus all -v $(pwd)/models:/app/models vega-verified:gpu \
    ./scripts/run_full_training.sh --gpu --epochs 10
```

---

## 📊 reproduce_experiments.sh - 논문 재현 스크립트

논문의 모든 실험을 재현합니다.

### 사용법

```bash
# 전체 재현
./scripts/reproduce_experiments.sh --all

# 빠른 테스트
./scripts/reproduce_experiments.sh --quick

# 검증 실험만
./scripts/reproduce_experiments.sh --verification

# 수리 실험만
./scripts/reproduce_experiments.sh --repair

# 비교 실험
./scripts/reproduce_experiments.sh --comparison

# 소거 연구
./scripts/reproduce_experiments.sh --ablation
```

### 옵션

```
--all           모든 실험 실행
--quick         빠른 검증 (작은 샘플)
--verification  검증 실험만
--repair        수리 실험만
--comparison    VEGA vs VEGA-Verified 비교
--ablation      소거 연구
```

### Docker에서 실행

```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    vega-verified:cpu \
    ./scripts/reproduce_experiments.sh --all
```

---

## ⏱️ 예상 실행 시간

| 작업 | CPU | GPU |
|-----|-----|-----|
| `--test-only` | ~1분 | ~30초 |
| 전체 학습 (1000 샘플, 10 에폭) | ~10시간 | ~30분 |
| 전체 실험 재현 | ~2시간 | ~30분 |

---

## 📝 출력 파일

학습 후 생성되는 파일:

```
models/repair_model/
├── checkpoint-100/          # 중간 체크포인트
├── checkpoint-500/
├── final/                   # 최종 모델
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── vocab.json
└── training_config.json     # 학습 설정
```

---

## 🔗 관련 문서

- [메인 README](../README.md)
- [명령어 레퍼런스](../docs/COMMANDS_REFERENCE.md)
- [소스 코드 가이드](../src/README.md)
