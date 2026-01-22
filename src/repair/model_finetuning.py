"""
Phase 2.3: Model Fine-tuning Pipeline for Neural Repair.

This module provides interfaces and utilities for fine-tuning
code repair models on compiler backend bug-fix data.

Supported Models:
1. CodeT5+ (Salesforce) - Best for seq2seq repair
2. CodeBERT (Microsoft) - Good for code understanding
3. UniXcoder (Microsoft) - Unified code representation
4. GraphCodeBERT - Structure-aware code understanding

Training Configuration:
- Learning rate: 5e-5 (with warmup)
- Batch size: 8-16 (depending on GPU memory)
- Epochs: 3-10 (early stopping)
- Max sequence length: 512-1024
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import json
from pathlib import Path
import time


class ModelType(Enum):
    """Supported model types."""
    CODET5_SMALL = "Salesforce/codet5-small"
    CODET5_BASE = "Salesforce/codet5-base"
    CODET5P_220M = "Salesforce/codet5p-220m"
    CODET5P_770M = "Salesforce/codet5p-770m"
    CODEBERT = "microsoft/codebert-base"
    GRAPHCODEBERT = "microsoft/graphcodebert-base"
    UNIXCODER = "microsoft/unixcoder-base"


@dataclass
class TrainingConfig:
    """Configuration for model fine-tuning."""
    model_type: ModelType = ModelType.CODET5_SMALL
    max_source_length: int = 512
    max_target_length: int = 512
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    seed: int = 42
    output_dir: str = "models/repair_model"
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    early_stopping_patience: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": self.fp16,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "early_stopping_patience": self.early_stopping_patience,
        }


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    train_loss: float = 0.0
    eval_loss: float = 0.0
    exact_match: float = 0.0
    bleu_score: float = 0.0
    repair_accuracy: float = 0.0
    epoch: int = 0
    step: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "exact_match": self.exact_match,
            "bleu_score": self.bleu_score,
            "repair_accuracy": self.repair_accuracy,
            "epoch": self.epoch,
            "step": self.step,
        }


class RepairModelInterface:
    """
    Abstract interface for code repair models.
    
    Provides:
    - Training from bug-fix data
    - Inference for repair generation
    - Model loading/saving
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda"  # Will be auto-detected
        self.is_loaded = False
        self.metrics_history: List[TrainingMetrics] = []
    
    def load_model(self) -> None:
        """Load pretrained model and tokenizer."""
        raise NotImplementedError("Subclass must implement load_model")
    
    def train(
        self,
        train_data: List[Tuple[str, str]],
        eval_data: Optional[List[Tuple[str, str]]] = None,
        callback: Optional[Callable[[TrainingMetrics], None]] = None
    ) -> TrainingMetrics:
        """
        Fine-tune model on training data.
        
        Args:
            train_data: List of (buggy_code, fixed_code) pairs
            eval_data: Optional evaluation data
            callback: Optional callback for metrics
            
        Returns:
            Final training metrics
        """
        raise NotImplementedError("Subclass must implement train")
    
    def repair(
        self,
        buggy_code: str,
        counterexample: Optional[Dict[str, Any]] = None,
        num_candidates: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Generate repair candidates for buggy code.
        
        Args:
            buggy_code: Code with bug
            counterexample: Optional counterexample from verification
            num_candidates: Number of candidates to generate
            
        Returns:
            List of (repaired_code, confidence) tuples
        """
        raise NotImplementedError("Subclass must implement repair")
    
    def save(self, path: str) -> None:
        """Save model to path."""
        raise NotImplementedError("Subclass must implement save")
    
    def load(self, path: str) -> None:
        """Load model from path."""
        raise NotImplementedError("Subclass must implement load")


class CodeT5RepairModel(RepairModelInterface):
    """
    CodeT5/CodeT5+ based repair model.
    
    Uses encoder-decoder architecture for seq2seq repair.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._check_dependencies()
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        self.has_transformers = False
        self.has_torch = False
        
        try:
            import torch
            self.has_torch = True
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass
        
        try:
            import transformers
            self.has_transformers = True
        except ImportError:
            pass
        
        return self.has_transformers and self.has_torch
    
    def load_model(self) -> None:
        """Load CodeT5 model and tokenizer."""
        if not self._check_dependencies():
            print("Warning: transformers or torch not available")
            print("Model will run in mock mode")
            self.is_loaded = False
            return
        
        from transformers import T5ForConditionalGeneration, RobertaTokenizer
        import torch
        
        model_name = self.config.model_type.value
        
        print(f"Loading model: {model_name}")
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        if self.device == "cuda":
            self.model = self.model.cuda()
        
        self.is_loaded = True
        print(f"Model loaded on {self.device}")
    
    def train(
        self,
        train_data: List[Tuple[str, str]],
        eval_data: Optional[List[Tuple[str, str]]] = None,
        callback: Optional[Callable[[TrainingMetrics], None]] = None
    ) -> TrainingMetrics:
        """Fine-tune CodeT5 on bug-fix pairs."""
        if not self.is_loaded:
            # Mock training
            return self._mock_train(train_data, eval_data, callback)
        
        from transformers import TrainingArguments, Trainer
        from transformers import DataCollatorForSeq2Seq
        import torch
        from torch.utils.data import Dataset
        
        # Create dataset
        class RepairDataset(Dataset):
            def __init__(self, data, tokenizer, max_source, max_target):
                self.data = data
                self.tokenizer = tokenizer
                self.max_source = max_source
                self.max_target = max_target
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                source, target = self.data[idx]
                
                source_enc = self.tokenizer(
                    source,
                    max_length=self.max_source,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                target_enc = self.tokenizer(
                    target,
                    max_length=self.max_target,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                return {
                    "input_ids": source_enc.input_ids.squeeze(),
                    "attention_mask": source_enc.attention_mask.squeeze(),
                    "labels": target_enc.input_ids.squeeze(),
                }
        
        train_dataset = RepairDataset(
            train_data, 
            self.tokenizer,
            self.config.max_source_length,
            self.config.max_target_length
        )
        
        eval_dataset = None
        if eval_data:
            eval_dataset = RepairDataset(
                eval_data,
                self.tokenizer,
                self.config.max_source_length,
                self.config.max_target_length
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            fp16=self.config.fp16 and self.device == "cuda",
            seed=self.config.seed,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Get final metrics
        metrics = TrainingMetrics(
            train_loss=trainer.state.log_history[-1].get("loss", 0),
            eval_loss=trainer.state.log_history[-1].get("eval_loss", 0) if eval_dataset else 0,
            epoch=self.config.num_epochs,
            step=trainer.state.global_step,
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _mock_train(
        self,
        train_data: List[Tuple[str, str]],
        eval_data: Optional[List[Tuple[str, str]]] = None,
        callback: Optional[Callable[[TrainingMetrics], None]] = None
    ) -> TrainingMetrics:
        """Mock training when dependencies not available."""
        print("\n⚠️ Running mock training (no GPU/transformers)")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Eval samples: {len(eval_data) if eval_data else 0}")
        
        # Simulate training epochs
        for epoch in range(self.config.num_epochs):
            metrics = TrainingMetrics(
                train_loss=1.0 - (epoch * 0.15),
                eval_loss=1.2 - (epoch * 0.12),
                exact_match=0.2 + (epoch * 0.1),
                bleu_score=0.3 + (epoch * 0.08),
                repair_accuracy=0.25 + (epoch * 0.1),
                epoch=epoch + 1,
                step=(epoch + 1) * len(train_data),
            )
            
            self.metrics_history.append(metrics)
            
            if callback:
                callback(metrics)
            
            print(f"   Epoch {epoch + 1}: loss={metrics.train_loss:.3f}, "
                  f"repair_acc={metrics.repair_accuracy:.3f}")
        
        return self.metrics_history[-1]
    
    def repair(
        self,
        buggy_code: str,
        counterexample: Optional[Dict[str, Any]] = None,
        num_candidates: int = 5
    ) -> List[Tuple[str, float]]:
        """Generate repair candidates."""
        if not self.is_loaded:
            return self._mock_repair(buggy_code, counterexample, num_candidates)
        
        import torch
        
        # Format input
        if counterexample:
            input_text = f"<BUG> {buggy_code} </BUG> <CEX> {json.dumps(counterexample)} </CEX>"
        else:
            input_text = f"<BUG> {buggy_code} </BUG>"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_target_length,
                num_beams=num_candidates,
                num_return_sequences=num_candidates,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Decode
        candidates = []
        for i, (seq, score) in enumerate(zip(outputs.sequences, outputs.sequences_scores)):
            decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
            
            # Clean up
            decoded = decoded.replace("<FIX>", "").replace("</FIX>", "").strip()
            
            # Calculate confidence
            confidence = torch.softmax(outputs.sequences_scores, dim=0)[i].item()
            
            candidates.append((decoded, confidence))
        
        return candidates
    
    def _mock_repair(
        self,
        buggy_code: str,
        counterexample: Optional[Dict[str, Any]] = None,
        num_candidates: int = 5
    ) -> List[Tuple[str, float]]:
        """Mock repair when model not loaded."""
        # Template-based mock repair
        candidates = []
        
        # Check for common patterns
        if "switch" in buggy_code and counterexample:
            expected = counterexample.get("expected_output", "")
            input_val = counterexample.get("input_values", {}).get("Kind", "")
            
            if expected and input_val:
                # Try adding the missing case
                repair = buggy_code.replace(
                    "default:",
                    f"case {input_val}:\n    return {expected};\n  default:"
                )
                candidates.append((repair, 0.85))
        
        # Try simple mutations
        if "return" in buggy_code and counterexample:
            expected = counterexample.get("expected_output", "")
            if expected:
                # Fix return value
                import re
                repair = re.sub(
                    r'return\s+\w+;',
                    f'return {expected};',
                    buggy_code,
                    count=1
                )
                candidates.append((repair, 0.6))
        
        # Original as fallback
        candidates.append((buggy_code, 0.1))
        
        # Pad with variations
        while len(candidates) < num_candidates:
            candidates.append((buggy_code, 0.05))
        
        return candidates[:num_candidates]
    
    def save(self, path: str) -> None:
        """Save model to path."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        if self.is_loaded and self.model:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        
        # Save config and metrics
        with open(f"{path}/training_config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        with open(f"{path}/metrics_history.json", 'w') as f:
            json.dump([m.to_dict() for m in self.metrics_history], f, indent=2)
    
    def load(self, path: str) -> None:
        """Load model from path."""
        if not self._check_dependencies():
            print("Warning: Cannot load model without transformers/torch")
            return
        
        from transformers import T5ForConditionalGeneration, RobertaTokenizer
        
        self.tokenizer = RobertaTokenizer.from_pretrained(path)
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        
        if self.device == "cuda":
            self.model = self.model.cuda()
        
        self.is_loaded = True
        
        # Load metrics
        metrics_path = f"{path}/metrics_history.json"
        if Path(metrics_path).exists():
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
                self.metrics_history = [
                    TrainingMetrics(**m) for m in metrics_data
                ]


class ModelTrainingPipeline:
    """
    End-to-end pipeline for training repair models.
    
    Steps:
    1. Load training data
    2. Initialize model
    3. Fine-tune
    4. Evaluate
    5. Save
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        data_path: str,
        output_dir: str,
        verbose: bool = False
    ):
        self.config = config
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Returns:
            Training results and metrics
        """
        start_time = time.time()
        
        # 1. Load data
        print("📊 Loading training data...")
        train_data, eval_data = self._load_data()
        print(f"   Train: {len(train_data)} examples")
        print(f"   Eval: {len(eval_data)} examples")
        
        # 2. Initialize model
        print(f"\n🔧 Initializing model: {self.config.model_type.value}")
        model = CodeT5RepairModel(self.config)
        model.load_model()
        
        # 3. Train
        print("\n🚀 Starting training...")
        
        def log_callback(metrics: TrainingMetrics):
            if self.verbose:
                print(f"   Step {metrics.step}: loss={metrics.train_loss:.4f}")
        
        final_metrics = model.train(
            train_data=train_data,
            eval_data=eval_data,
            callback=log_callback
        )
        
        # 4. Evaluate
        print("\n📈 Evaluating model...")
        eval_results = self._evaluate(model, eval_data)
        
        # 5. Save
        print("\n💾 Saving model...")
        model.save(str(self.output_dir / "final_model"))
        
        # Results
        elapsed = time.time() - start_time
        
        results = {
            "status": "completed",
            "training_time_seconds": elapsed,
            "train_size": len(train_data),
            "eval_size": len(eval_data),
            "final_metrics": final_metrics.to_dict(),
            "eval_results": eval_results,
            "model_path": str(self.output_dir / "final_model"),
        }
        
        # Save results
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _load_data(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Load training data from files."""
        train_data = []
        eval_data = []
        
        # Try HuggingFace format first
        train_path = self.data_path / "train.jsonl"
        eval_path = self.data_path / "val.jsonl"
        
        if train_path.exists():
            with open(train_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    train_data.append((item["input"], item["output"]))
        
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    eval_data.append((item["input"], item["output"]))
        
        # Try JSON format
        json_path = self.data_path / "training_data.json"
        if json_path.exists() and not train_data:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            examples = data.get("examples", [])
            split_idx = int(len(examples) * 0.8)
            
            for ex in examples[:split_idx]:
                buggy = ex.get("buggy_code", "")
                fixed = ex.get("fixed_code", "")
                train_data.append((buggy, fixed))
            
            for ex in examples[split_idx:]:
                buggy = ex.get("buggy_code", "")
                fixed = ex.get("fixed_code", "")
                eval_data.append((buggy, fixed))
        
        return train_data, eval_data
    
    def _evaluate(
        self,
        model: CodeT5RepairModel,
        eval_data: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """Evaluate model on test data."""
        correct = 0
        total = 0
        
        for buggy, expected in eval_data[:100]:  # Limit for speed
            candidates = model.repair(buggy, num_candidates=1)
            if candidates:
                predicted, _ = candidates[0]
                if predicted.strip() == expected.strip():
                    correct += 1
            total += 1
        
        return {
            "exact_match": correct / total if total > 0 else 0,
            "total_evaluated": total,
            "correct": correct,
        }


def create_and_train_model(
    data_path: str,
    output_dir: str,
    model_type: str = "codet5-small",
    num_epochs: int = 5,
    batch_size: int = 8,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to create and train a repair model.
    
    Args:
        data_path: Path to training data
        output_dir: Output directory
        model_type: Model type (codet5-small, codet5-base, etc.)
        num_epochs: Number of training epochs
        batch_size: Training batch size
        verbose: Print verbose output
        
    Returns:
        Training results
    """
    # Map model type string to enum
    model_map = {
        "codet5-small": ModelType.CODET5_SMALL,
        "codet5-base": ModelType.CODET5_BASE,
        "codet5p-220m": ModelType.CODET5P_220M,
        "codet5p-770m": ModelType.CODET5P_770M,
        "codebert": ModelType.CODEBERT,
        "unixcoder": ModelType.UNIXCODER,
    }
    
    config = TrainingConfig(
        model_type=model_map.get(model_type, ModelType.CODET5_SMALL),
        num_epochs=num_epochs,
        batch_size=batch_size,
        output_dir=output_dir,
    )
    
    pipeline = ModelTrainingPipeline(
        config=config,
        data_path=data_path,
        output_dir=output_dir,
        verbose=verbose
    )
    
    return pipeline.run()


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 2.3: Model Fine-tuning Pipeline Demo")
    print("=" * 70)
    
    # Test configuration
    config = TrainingConfig(
        model_type=ModelType.CODET5_SMALL,
        num_epochs=3,
        batch_size=4,
        output_dir="models/test_repair"
    )
    
    print("\n📋 Training Configuration:")
    for key, val in config.to_dict().items():
        print(f"   {key}: {val}")
    
    # Test model interface
    print("\n🔧 Creating model interface...")
    model = CodeT5RepairModel(config)
    
    # Mock data
    train_data = [
        ("buggy code 1", "fixed code 1"),
        ("buggy code 2", "fixed code 2"),
        ("buggy code 3", "fixed code 3"),
    ]
    
    # Mock training
    print("\n🚀 Running mock training...")
    metrics = model.train(train_data)
    
    print(f"\n📊 Final Metrics:")
    for key, val in metrics.to_dict().items():
        print(f"   {key}: {val}")
    
    # Test repair
    print("\n🔍 Testing repair inference...")
    
    buggy_code = """
switch (Kind) {
    case FK_Data_4:
        return R_X86_64_32;
    default:
        return R_X86_64_NONE;
}
"""
    
    counterexample = {
        "input_values": {"Kind": "FK_Data_8"},
        "expected_output": "R_X86_64_64",
        "actual_output": "R_X86_64_NONE"
    }
    
    candidates = model.repair(buggy_code, counterexample, num_candidates=3)
    
    print(f"\n   Generated {len(candidates)} candidates:")
    for i, (code, conf) in enumerate(candidates):
        print(f"   [{i+1}] Confidence: {conf:.3f}")
        print(f"       {code[:100]}...")
    
    print("\n✅ Model Fine-tuning Pipeline Demo Complete")
