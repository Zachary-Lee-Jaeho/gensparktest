#!/usr/bin/env python3
"""
Neural Repair Model Training Script.

Supports:
- CPU training (slow but works everywhere)
- GPU training (CUDA for NVIDIA, MPS for Apple Silicon)
- Checkpoint resume for interrupted training
- Configurable model size (small/base/large)

Usage:
    # Quick test (CPU, minimal data)
    python scripts/train_neural_repair.py --test-only
    
    # Full training (CPU - will take hours)
    python scripts/train_neural_repair.py --epochs 5 --batch-size 4
    
    # Resume from checkpoint
    python scripts/train_neural_repair.py --resume models/repair_model/checkpoint-500
    
    # GPU training (when available)
    python scripts/train_neural_repair.py --device cuda --batch-size 16 --fp16
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")


def get_device(requested_device: str = "auto") -> str:
    """Detect best available device."""
    if requested_device != "auto":
        return requested_device
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def create_sample_data(size: int = 100) -> list:
    """Create sample bug-fix pairs for training."""
    # These are simplified examples for testing
    samples = [
        # Missing case in switch
        (
            '''switch (Kind) {
    case FK_Data_4: return R_X86_64_32;
    default: return R_X86_64_NONE;
}''',
            '''switch (Kind) {
    case FK_Data_4: return R_X86_64_32;
    case FK_Data_8: return R_X86_64_64;
    default: return R_X86_64_NONE;
}'''
        ),
        # Wrong return value
        (
            '''switch (Kind) {
    case FK_Data_1: return R_X86_64_16;
    default: return R_X86_64_NONE;
}''',
            '''switch (Kind) {
    case FK_Data_1: return R_X86_64_8;
    default: return R_X86_64_NONE;
}'''
        ),
        # Missing null check
        (
            '''void process(MCInst *MI) {
    unsigned Op = MI->getOpcode();
}''',
            '''void process(MCInst *MI) {
    if (!MI) return;
    unsigned Op = MI->getOpcode();
}'''
        ),
        # Off-by-one error
        (
            '''for (int i = 0; i <= size; i++) {
    data[i] = 0;
}''',
            '''for (int i = 0; i < size; i++) {
    data[i] = 0;
}'''
        ),
        # Missing break
        (
            '''switch (Kind) {
    case FK_Data_4:
        result = 32;
    case FK_Data_8:
        result = 64;
        break;
}''',
            '''switch (Kind) {
    case FK_Data_4:
        result = 32;
        break;
    case FK_Data_8:
        result = 64;
        break;
}'''
        ),
    ]
    
    # Replicate to reach desired size
    result = []
    for i in range(size):
        idx = i % len(samples)
        buggy, fixed = samples[idx]
        # Add slight variations
        if i >= len(samples):
            buggy = buggy.replace("Kind", f"Kind_{i % 10}")
            fixed = fixed.replace("Kind", f"Kind_{i % 10}")
        result.append((buggy, fixed))
    
    return result


def test_model_inference(model_path: str, device: str):
    """Quick test of model inference."""
    from src.repair.neural_repair_engine import NeuralRepairEngine, NeuralRepairConfig
    
    config = NeuralRepairConfig(
        model_path=model_path,
        device=device,
    )
    engine = NeuralRepairEngine(config)
    
    print(f"\n{'='*60}")
    print("Testing Model Inference")
    print('='*60)
    
    if engine.load(model_path):
        print(f"✅ Model loaded from {model_path}")
        print(f"   Device: {engine.device}")
        
        # Test repair
        buggy_code = '''switch (Kind) {
    case FK_Data_4: return R_X86_64_32;
    default: return R_X86_64_NONE;
}'''
        
        counterexample = {
            "input_values": {"Kind": "FK_Data_8"},
            "expected_output": "R_X86_64_64",
            "actual_output": "R_X86_64_NONE"
        }
        
        print(f"\nInput code:\n{buggy_code}")
        print(f"\nCounterexample: {counterexample}")
        
        start = time.time()
        candidates = engine.repair(buggy_code, counterexample, num_candidates=3)
        elapsed = time.time() - start
        
        print(f"\n✅ Generated {len(candidates)} candidates in {elapsed:.2f}s")
        for i, c in enumerate(candidates):
            print(f"\n[{i+1}] Confidence: {c.confidence:.3f}")
            print(f"    Code: {c.code[:200]}...")
        
        return True
    else:
        print("❌ Failed to load model")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train Neural Repair Model")
    parser.add_argument("--model", type=str, default="Salesforce/codet5-small",
                        help="Base model (codet5-small/base/large)")
    parser.add_argument("--output-dir", type=str, default="models/repair_model",
                        help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (reduce for CPU)")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to use")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 (GPU only)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--train-size", type=int, default=100,
                        help="Number of training samples")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run quick test (5 samples, 1 epoch)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only test inference")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to training data JSON file")
    
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"\n{'='*60}")
    print(f"Neural Repair Model Training")
    print('='*60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    
    # Quick test mode
    if args.test_only:
        args.train_size = 10
        args.epochs = 1
        args.batch_size = 2
        print("\n⚡ TEST MODE: 10 samples, 1 epoch")
    
    # Skip training if requested
    if args.skip_training:
        test_model_inference(f"{args.output_dir}/final", device)
        return 0
    
    # Load or create training data
    if args.data_file and Path(args.data_file).exists():
        print(f"\nLoading data from {args.data_file}...")
        with open(args.data_file, 'r') as f:
            data = json.load(f)
        train_data = [(d['buggy'], d['fixed']) for d in data.get('train', [])]
        eval_data = [(d['buggy'], d['fixed']) for d in data.get('eval', [])]
    else:
        print(f"\nCreating sample training data ({args.train_size} samples)...")
        all_data = create_sample_data(args.train_size)
        split = int(len(all_data) * 0.8)
        train_data = all_data[:split]
        eval_data = all_data[split:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}")
    
    # Import trainer
    from src.repair.neural_repair_engine import NeuralRepairTrainer
    
    # Create trainer
    trainer = NeuralRepairTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        fp16=args.fp16 and device == "cuda",
        gradient_accumulation_steps=4 if device == "cpu" else 1,
    )
    
    # Estimate time
    if device == "cpu":
        est_time = len(train_data) * args.epochs * 2  # ~2 sec/sample on CPU
        print(f"\n⚠️  CPU Training: Estimated time ~{est_time//60} minutes")
        print("   Consider using --test-only for quick validation")
        print("   For full training, use GPU server with --device cuda")
    
    # Train
    print(f"\n{'='*60}")
    print("Starting Training...")
    print('='*60)
    
    start_time = time.time()
    try:
        results = trainer.train(
            train_data=train_data,
            eval_data=eval_data if eval_data else None,
            resume_from_checkpoint=args.resume,
        )
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("Training Complete!")
        print('='*60)
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Train loss: {results.get('train_loss', 'N/A')}")
        print(f"Model saved to: {results.get('model_path', args.output_dir)}")
        
        # Test inference
        test_model_inference(f"{args.output_dir}/final", device)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
        print("   Resume with: python scripts/train_neural_repair.py --resume <checkpoint-path>")
        print(f"   Check {args.output_dir} for checkpoints")
        return 1
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
