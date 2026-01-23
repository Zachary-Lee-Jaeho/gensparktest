# Neural Repair Model 학습 가이드

## 모델 크기별 학습 명령어

GPU 서버에서 아래 세 가지 모델을 각각 학습하세요.

### 1. Small 모델 (Salesforce/codet5-small)
- **파라미터**: ~60M
- **학습 시간**: 가장 빠름
- **추천 용도**: 빠른 프로토타이핑, 테스트

```bash
python scripts/train_neural_repair.py \
    --device cuda \
    --model Salesforce/codet5-small \
    --epochs 10 \
    --batch-size 16 \
    --train-size 2000 \
    --fp16 \
    --output-dir models/repair_model_small
```

### 2. Base 모델 (Salesforce/codet5-base)
- **파라미터**: ~220M
- **학습 시간**: 중간
- **추천 용도**: 균형 잡힌 성능

```bash
python scripts/train_neural_repair.py \
    --device cuda \
    --model Salesforce/codet5-base \
    --epochs 10 \
    --batch-size 8 \
    --train-size 2000 \
    --fp16 \
    --output-dir models/repair_model_base
```

### 3. Large 모델 (Salesforce/codet5-large)
- **파라미터**: ~770M
- **학습 시간**: 가장 오래 걸림
- **추천 용도**: 최고 성능

```bash
python scripts/train_neural_repair.py \
    --device cuda \
    --model Salesforce/codet5-large \
    --epochs 10 \
    --batch-size 4 \
    --train-size 2000 \
    --fp16 \
    --output-dir models/repair_model_large
```

---

## 학습 후 모델 파일 구조

학습이 완료되면 각 디렉토리에 다음 파일들이 생성됩니다:

```
models/
├── repair_model_small/
│   └── final/
│       ├── config.json
│       ├── model.safetensors (또는 pytorch_model.bin)
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── special_tokens_map.json
├── repair_model_base/
│   └── final/
│       └── ... (동일 구조)
└── repair_model_large/
    └── final/
        └── ... (동일 구조)
```

---

## SSH 서버로 모델 전송

학습 완료 후 모델을 SSH 서버로 전송하세요:

```bash
# GPU 서버에서 실행
scp -r models/repair_model_small/final user@ssh-server:/path/to/webapp/models/repair_model_small/
scp -r models/repair_model_base/final user@ssh-server:/path/to/webapp/models/repair_model_base/
scp -r models/repair_model_large/final user@ssh-server:/path/to/webapp/models/repair_model_large/
```

또는 tar로 압축해서 전송:

```bash
# GPU 서버에서 압축
tar -czvf models_trained.tar.gz models/repair_model_*/final/

# 전송
scp models_trained.tar.gz user@ssh-server:/path/to/webapp/

# SSH 서버에서 압축 해제
tar -xzvf models_trained.tar.gz
```

---

## 실험 실행 (SSH 서버)

모델 전송 후 SSH 서버에서 실험:

```bash
# Small 모델로 실험
vega-verify experiment --experiment repair \
    --model-path models/repair_model_small/final \
    --device cpu \
    --sample-size 50

# Base 모델로 실험
vega-verify experiment --experiment repair \
    --model-path models/repair_model_base/final \
    --device cpu \
    --sample-size 50

# Large 모델로 실험
vega-verify experiment --experiment repair \
    --model-path models/repair_model_large/final \
    --device cpu \
    --sample-size 50

# 세 모델 비교 실험
for model in small base large; do
    echo "=== Testing $model model ==="
    vega-verify experiment --experiment repair \
        --model-path models/repair_model_$model/final \
        --device cpu \
        --sample-size 50
done
```

---

## 예상 학습 시간 (NVIDIA GPU 기준)

| 모델 | GPU (예: RTX 3090) | GPU (예: V100) |
|------|-------------------|----------------|
| Small | ~20분 | ~15분 |
| Base | ~1시간 | ~45분 |
| Large | ~3시간 | ~2시간 |

*train-size=2000, epochs=10 기준*

---

## 예상 Inference 시간 (CPU 기준)

| 모델 | 함수당 시간 |
|------|-----------|
| Small | ~1-2초 |
| Base | ~3-5초 |
| Large | ~10-15초 |

*6코어 CPU 기준, 실제 시간은 함수 복잡도에 따라 다름*

---

## 체크포인트에서 재학습

학습이 중단된 경우:

```bash
# Small 모델 재개
python scripts/train_neural_repair.py \
    --device cuda \
    --model Salesforce/codet5-small \
    --resume models/repair_model_small/checkpoint-latest \
    --epochs 5 \
    --output-dir models/repair_model_small
```

---

## 문제 해결

### CUDA Out of Memory
- `--batch-size`를 줄이세요 (예: 16 → 8 → 4)
- `--fp16` 옵션이 활성화되어 있는지 확인

### 학습이 너무 느림
- `--train-size`를 줄여서 테스트 (예: 500)
- `--epochs`를 줄여서 테스트 (예: 3)

### 모델 로딩 실패
- 모델 파일이 완전히 전송되었는지 확인
- `config.json`과 `model.safetensors` 파일 존재 확인
