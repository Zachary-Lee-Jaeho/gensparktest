#!/usr/bin/env python3
"""
모델 비교 실험 스크립트.

세 가지 모델 크기(small, base, large)의 성능을 비교합니다.

사용법:
    # 모든 모델 비교
    python scripts/compare_models.py
    
    # 특정 샘플 크기로 비교
    python scripts/compare_models.py --sample-size 100
    
    # 특정 모델만 테스트
    python scripts/compare_models.py --models small base
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ModelTestResult:
    """단일 모델의 테스트 결과."""
    model_name: str
    model_path: str
    model_loaded: bool = False
    load_time_seconds: float = 0.0
    total_tests: int = 0
    successful_repairs: int = 0
    failed_repairs: int = 0
    errors: int = 0
    total_time_seconds: float = 0.0
    avg_inference_time_seconds: float = 0.0
    repair_accuracy: float = 0.0
    error_messages: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
            "load_time_seconds": self.load_time_seconds,
            "total_tests": self.total_tests,
            "successful_repairs": self.successful_repairs,
            "failed_repairs": self.failed_repairs,
            "errors": self.errors,
            "total_time_seconds": self.total_time_seconds,
            "avg_inference_time_seconds": self.avg_inference_time_seconds,
            "repair_accuracy": self.repair_accuracy,
        }


def check_model_exists(model_path: str) -> bool:
    """모델 파일이 존재하는지 확인."""
    path = Path(model_path)
    if not path.exists():
        return False
    
    # 필수 파일 확인
    required_files = ["config.json"]
    optional_model_files = ["model.safetensors", "pytorch_model.bin"]
    
    has_config = (path / "config.json").exists()
    has_model = any((path / f).exists() for f in optional_model_files)
    
    return has_config and has_model


def load_model(model_path: str, model_name: str, device: str = "cpu"):
    """모델 로드."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    
    print(f"  로딩 중: {model_path}")
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    load_time = time.time() - start
    print(f"  로드 완료: {load_time:.2f}초")
    
    return tokenizer, model, load_time


def run_inference(model, tokenizer, buggy_code: str, device: str = "cpu") -> str:
    """단일 추론 실행."""
    import torch
    
    prompt = f"fix bug: {buggy_code}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    if device == "cuda" and torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def get_test_cases() -> List[Dict[str, str]]:
    """테스트 케이스 생성."""
    return [
        {
            "name": "missing_case_1",
            "buggy": '''switch (Kind) {
    case FK_Data_4: return R_X86_64_32;
    default: return R_X86_64_NONE;
}''',
            "expected_fix": "FK_Data_8",  # 이 케이스가 추가되어야 함
        },
        {
            "name": "wrong_return_1",
            "buggy": '''switch (Kind) {
    case FK_Data_1: return R_X86_64_16;
    default: return R_X86_64_NONE;
}''',
            "expected_fix": "R_X86_64_8",  # 반환값이 수정되어야 함
        },
        {
            "name": "missing_case_2",
            "buggy": '''switch (Kind) {
    case FK_PCRel_1: return R_X86_64_PC8;
    case FK_PCRel_2: return R_X86_64_PC16;
    default: return R_X86_64_NONE;
}''',
            "expected_fix": "FK_PCRel_4",  # 이 케이스가 추가되어야 함
        },
        {
            "name": "wrong_return_2",
            "buggy": '''unsigned getRelocType(MCFixupKind Kind) {
    switch (Kind) {
        case RISCV::fixup_riscv_hi20: return ELF::R_RISCV_HI20;
        case RISCV::fixup_riscv_lo12_i: return ELF::R_RISCV_LO12_S;
        default: return ELF::R_RISCV_NONE;
    }
}''',
            "expected_fix": "R_RISCV_LO12_I",  # S를 I로 수정
        },
        {
            "name": "missing_null_check",
            "buggy": '''void process(Value *V) {
    auto *User = V->getUser();
    User->doSomething();
}''',
            "expected_fix": "if",  # null 체크가 추가되어야 함
        },
    ]


def test_model(
    model_name: str,
    model_path: str,
    test_cases: List[Dict],
    device: str = "cpu",
    verbose: bool = False
) -> ModelTestResult:
    """단일 모델 테스트."""
    result = ModelTestResult(model_name=model_name, model_path=model_path)
    
    # 모델 존재 확인
    if not check_model_exists(model_path):
        result.error_messages.append(f"모델을 찾을 수 없음: {model_path}")
        print(f"  ❌ 모델을 찾을 수 없음: {model_path}")
        return result
    
    # 모델 로드
    try:
        tokenizer, model, load_time = load_model(model_path, model_name, device)
        result.model_loaded = True
        result.load_time_seconds = load_time
    except Exception as e:
        result.error_messages.append(f"모델 로드 실패: {e}")
        print(f"  ❌ 모델 로드 실패: {e}")
        return result
    
    # 테스트 실행
    result.total_tests = len(test_cases)
    inference_times = []
    
    for i, tc in enumerate(test_cases):
        try:
            start = time.time()
            repaired = run_inference(model, tokenizer, tc["buggy"], device)
            inference_time = time.time() - start
            inference_times.append(inference_time)
            
            # 간단한 성공 판단 (expected_fix 문자열이 포함되어 있으면 성공)
            if tc["expected_fix"].lower() in repaired.lower():
                result.successful_repairs += 1
                status = "✅"
            else:
                result.failed_repairs += 1
                status = "❌"
            
            if verbose:
                print(f"    [{i+1}/{len(test_cases)}] {tc['name']}: {status} ({inference_time:.2f}s)")
                
        except Exception as e:
            result.errors += 1
            result.error_messages.append(f"{tc['name']}: {e}")
            if verbose:
                print(f"    [{i+1}/{len(test_cases)}] {tc['name']}: ⚠️ Error: {e}")
    
    # 통계 계산
    result.total_time_seconds = sum(inference_times)
    result.avg_inference_time_seconds = (
        result.total_time_seconds / len(inference_times) if inference_times else 0
    )
    result.repair_accuracy = (
        result.successful_repairs / result.total_tests if result.total_tests > 0 else 0
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(description="모델 비교 실험")
    parser.add_argument(
        "--models", nargs="+", 
        choices=["small", "base", "large"],
        default=["small", "base", "large"],
        help="테스트할 모델 (기본: 모두)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="추론 장치"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="상세 출력"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/model_comparison.json",
        help="결과 저장 경로"
    )
    
    args = parser.parse_args()
    
    # 장치 설정
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("VEGA-Verified 모델 비교 실험")
    print("=" * 60)
    print(f"장치: {device}")
    print(f"테스트 모델: {args.models}")
    print("=" * 60)
    
    # 모델 경로 설정
    model_paths = {
        "small": "models/repair_model_small/final",
        "base": "models/repair_model_base/final",
        "large": "models/repair_model_large/final",
    }
    
    # 테스트 케이스 로드
    test_cases = get_test_cases()
    print(f"테스트 케이스: {len(test_cases)}개")
    print()
    
    # 각 모델 테스트
    results = {}
    for model_name in args.models:
        print(f"\n>>> {model_name.upper()} 모델 테스트")
        print("-" * 40)
        
        result = test_model(
            model_name=model_name,
            model_path=model_paths[model_name],
            test_cases=test_cases,
            device=device,
            verbose=args.verbose
        )
        results[model_name] = result
        
        # 결과 출력
        print(f"\n  결과:")
        print(f"    로드됨: {result.model_loaded}")
        if result.model_loaded:
            print(f"    로드 시간: {result.load_time_seconds:.2f}초")
            print(f"    성공: {result.successful_repairs}/{result.total_tests}")
            print(f"    실패: {result.failed_repairs}/{result.total_tests}")
            print(f"    에러: {result.errors}")
            print(f"    정확도: {result.repair_accuracy*100:.1f}%")
            print(f"    평균 추론 시간: {result.avg_inference_time_seconds:.2f}초")
    
    # 비교 테이블 출력
    print("\n" + "=" * 60)
    print("결과 비교")
    print("=" * 60)
    print(f"{'모델':<10} {'로드':<6} {'정확도':<10} {'평균시간':<10} {'상태'}")
    print("-" * 60)
    
    for model_name in args.models:
        r = results[model_name]
        if r.model_loaded:
            status = "✅" if r.repair_accuracy >= 0.5 else "🟡"
            print(f"{model_name:<10} {'예':<6} {r.repair_accuracy*100:>6.1f}%    {r.avg_inference_time_seconds:>6.2f}s    {status}")
        else:
            print(f"{model_name:<10} {'아니오':<6} {'N/A':<10} {'N/A':<10} ❌ 모델 없음")
    
    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "test_cases_count": len(test_cases),
        "results": {name: r.to_dict() for name, r in results.items()},
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장됨: {output_path}")
    
    # 추천
    print("\n" + "=" * 60)
    print("추천")
    print("=" * 60)
    
    loaded_models = [name for name, r in results.items() if r.model_loaded]
    
    if not loaded_models:
        print("❌ 로드된 모델이 없습니다!")
        print("\n모델을 학습하고 다음 경로에 복사하세요:")
        for name, path in model_paths.items():
            print(f"  - {name}: {path}")
    else:
        best_accuracy = max(results[name].repair_accuracy for name in loaded_models)
        best_models = [name for name in loaded_models 
                       if results[name].repair_accuracy == best_accuracy]
        
        print(f"✅ 최고 정확도 모델: {', '.join(best_models)} ({best_accuracy*100:.1f}%)")
        
        if best_accuracy < 0.5:
            print("\n⚠️ 정확도가 낮습니다. 다음을 시도해보세요:")
            print("  - 학습 데이터 증가 (--train-size 5000)")
            print("  - 에폭 증가 (--epochs 20)")
            print("  - 더 큰 모델 사용 (large)")


if __name__ == "__main__":
    main()
