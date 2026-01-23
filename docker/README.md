# docker/ - Docker 관련 파일 디렉토리

이 디렉토리는 Docker 빌드에 필요한 추가 파일들을 포함합니다.

---

## 📁 디렉토리 구조

```
docker/
├── README.md              # 이 파일
├── Dockerfile.llvm        # LLVM 전용 Docker 이미지
├── llvm-entrypoint.sh     # LLVM 컨테이너 진입점
└── tools/                 # 추가 도구
    └── ast_extractor.cpp  # AST 추출기 (C++)
```

---

## 🐳 Docker 이미지 종류

프로젝트 루트에 있는 주요 Dockerfile:

| 파일 | 용도 | 크기 |
|------|------|------|
| `Dockerfile.unified` | CPU 전체 환경 | ~2GB |
| `Dockerfile.gpu` | GPU 학습 환경 | ~8GB |
| `Dockerfile.light` | 최소 환경 | ~500MB |
| `docker/Dockerfile.llvm` | LLVM만 설치 | ~3GB |

---

## 🔧 Docker 빌드 명령어

### CPU 이미지 (권장)

```bash
# 프로젝트 루트에서 실행
docker build -f Dockerfile.unified -t vega-verified:cpu .

# 빌드 확인
docker run --rm vega-verified:cpu vega-verify status
```

### GPU 이미지

```bash
# GPU 이미지 빌드 (NVIDIA CUDA 13.0 기반)
docker build -f Dockerfile.gpu -t vega-verified:gpu .

# GPU 확인
docker run --rm --gpus all vega-verified:gpu \
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🚀 Docker 실행 명령어

### 테스트 실행

```bash
# CPU
docker run --rm vega-verified:cpu python -m pytest tests/ -v

# GPU
docker run --rm --gpus all vega-verified:gpu python -m pytest tests/ -v
```

### 학습 실행

```bash
# CPU 학습 (모델 저장)
docker run --rm \
    -v $(pwd)/models:/app/models \
    vega-verified:cpu \
    python scripts/train_neural_repair.py --test-only

# GPU 학습
docker run --rm --gpus all \
    -v $(pwd)/models:/app/models \
    vega-verified:gpu \
    ./scripts/run_full_training.sh --gpu --epochs 10
```

### 인터랙티브 셸

```bash
# 디버깅용
docker run -it --rm vega-verified:cpu /bin/bash
```

---

## 🔗 관련 문서

- [메인 README](../README.md)
- [명령어 레퍼런스](../docs/COMMANDS_REFERENCE.md)
