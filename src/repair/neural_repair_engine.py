"""
Neural Repair Engine for VEGA-Verified.

Complete implementation for neural code repair using transformer models.
Designed to work on GPU when available, with graceful CPU fallback.

Key Features:
1. CodeT5/CodeT5+ based repair generation
2. Counterexample-guided repair prompting  
3. Beam search with diverse decoding
4. Confidence scoring based on model logits
5. Batch inference support for efficiency

Usage:
    # Initialize
    engine = NeuralRepairEngine(model_name="Salesforce/codet5-base")
    
    # Load model (downloads if not cached)
    engine.load()
    
    # Generate repairs
    candidates = engine.repair(
        buggy_code="...",
        counterexample={"input": {...}, "expected": ...},
        num_candidates=5
    )
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Available device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class RepairCandidate:
    """A repair candidate with metadata."""
    code: str
    confidence: float
    beam_score: float = 0.0
    generation_time_ms: float = 0.0
    strategy: str = "neural"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "confidence": self.confidence,
            "beam_score": self.beam_score,
            "generation_time_ms": self.generation_time_ms,
            "strategy": self.strategy,
            "metadata": self.metadata,
        }


@dataclass
class NeuralRepairConfig:
    """Configuration for neural repair engine."""
    # Model settings
    model_name: str = "Salesforce/codet5-base"
    model_path: Optional[str] = None  # Path to fine-tuned model
    
    # Generation parameters
    max_source_length: int = 512
    max_target_length: int = 512
    num_beams: int = 10
    num_return_sequences: int = 5
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    length_penalty: float = 1.0
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    
    # Device settings
    device: Optional[str] = None  # None = auto-detect
    use_fp16: bool = True
    
    # Prompt templates
    prompt_template: str = "fix bug: {code}"
    cex_template: str = "fix bug with counterexample: {code} [CEX] input={input} expected={expected} actual={actual}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
            "num_beams": self.num_beams,
            "num_return_sequences": self.num_return_sequences,
            "temperature": self.temperature,
            "device": self.device,
        }


class NeuralRepairEngine:
    """
    Neural code repair engine using transformer models.
    
    This is the main class for neural-based code repair in VEGA-Verified.
    It handles model loading, prompt construction, inference, and result parsing.
    """
    
    # Supported model architectures
    SUPPORTED_MODELS = {
        "codet5": ["Salesforce/codet5-small", "Salesforce/codet5-base", "Salesforce/codet5-large"],
        "codet5p": ["Salesforce/codet5p-220m", "Salesforce/codet5p-770m", "Salesforce/codet5p-2b"],
        "unixcoder": ["microsoft/unixcoder-base"],
        "codebert": ["microsoft/codebert-base"],
    }
    
    def __init__(self, config: Optional[NeuralRepairConfig] = None):
        """
        Initialize the neural repair engine.
        
        Args:
            config: Configuration for the engine. If None, uses defaults.
        """
        self.config = config or NeuralRepairConfig()
        
        # Model components (lazy loaded)
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # State
        self.is_loaded = False
        self.model_type = self._detect_model_type()
        
        # Statistics
        self.stats = {
            "total_repairs": 0,
            "successful_repairs": 0,
            "total_time_ms": 0.0,
            "avg_candidates_per_repair": 0.0,
        }
        
        # Check dependencies
        self._torch_available = False
        self._transformers_available = False
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            import torch
            self._torch_available = True
            self._torch_version = torch.__version__
        except ImportError:
            logger.warning("PyTorch not available. Neural repair will not work.")
            self._torch_available = False
        
        try:
            import transformers
            self._transformers_available = True
            self._transformers_version = transformers.__version__
        except ImportError:
            logger.warning("Transformers not available. Neural repair will not work.")
            self._transformers_available = False
    
    def _detect_model_type(self) -> str:
        """Detect model type from model name."""
        model_name = self.config.model_name.lower()
        
        if "codet5p" in model_name:
            return "codet5p"
        elif "codet5" in model_name:
            return "codet5"
        elif "unixcoder" in model_name:
            return "unixcoder"
        elif "codebert" in model_name:
            return "codebert"
        else:
            return "unknown"
    
    def _detect_device(self) -> str:
        """Detect the best available device."""
        if self.config.device:
            return self.config.device
        
        if not self._torch_available:
            return "cpu"
        
        import torch
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def is_available(self) -> bool:
        """Check if the engine is ready for inference."""
        return self.is_loaded and self.model is not None and self.tokenizer is not None
    
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load the model and tokenizer.
        
        Args:
            model_path: Optional path to a fine-tuned model. If None, uses config.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self._torch_available or not self._transformers_available:
            logger.error("Cannot load model: PyTorch or Transformers not available")
            return False
        
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
        
        # Determine device
        self.device = self._detect_device()
        logger.info(f"Using device: {self.device}")
        
        # Determine model path
        load_path = model_path or self.config.model_path or self.config.model_name
        logger.info(f"Loading model from: {load_path}")
        
        try:
            # Load tokenizer
            if self.model_type in ["codet5", "codet5p"]:
                from transformers import RobertaTokenizer
                self.tokenizer = RobertaTokenizer.from_pretrained(load_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            
            # Load model
            if self.model_type in ["codet5", "codet5p"]:
                self.model = T5ForConditionalGeneration.from_pretrained(load_path)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Enable FP16 if configured and supported
            if self.config.use_fp16 and self.device == "cuda":
                self.model = self.model.half()
                logger.info("Using FP16 precision")
            
            # Set to eval mode
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def repair(
        self,
        buggy_code: str,
        counterexample: Optional[Dict[str, Any]] = None,
        specification: Optional[Any] = None,
        num_candidates: int = 5,
        return_scores: bool = False
    ) -> List[RepairCandidate]:
        """
        Generate repair candidates for buggy code.
        
        Args:
            buggy_code: The code containing the bug
            counterexample: Optional counterexample from verification
                           Format: {"input_values": {...}, "expected_output": ..., "actual_output": ...}
            specification: Optional specification (for context)
            num_candidates: Number of repair candidates to generate
            return_scores: If True, include detailed scores in metadata
            
        Returns:
            List of RepairCandidate objects, sorted by confidence
        """
        start_time = time.time()
        
        if not self.is_available():
            logger.warning("Model not loaded. Returning empty results.")
            return []
        
        import torch
        
        # Construct prompt
        prompt = self._construct_prompt(buggy_code, counterexample, specification)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=self.config.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with beam search
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.config.max_target_length,
                min_length=10,
                num_beams=max(self.config.num_beams, num_candidates),
                num_return_sequences=num_candidates,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                length_penalty=self.config.length_penalty,
                early_stopping=self.config.early_stopping,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Calculate beam scores
        if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
            beam_scores = outputs.sequences_scores.cpu().numpy().tolist()
        else:
            beam_scores = [0.0] * len(outputs.sequences)
        
        # Decode and create candidates
        candidates = []
        
        for i, (seq, score) in enumerate(zip(outputs.sequences, beam_scores)):
            # Decode
            decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
            
            # Clean up generated code
            cleaned_code = self._post_process_code(decoded, buggy_code)
            
            # Calculate confidence (softmax of beam scores)
            if beam_scores and max(beam_scores) != min(beam_scores):
                import numpy as np
                scores_array = np.array(beam_scores)
                exp_scores = np.exp(scores_array - np.max(scores_array))
                confidence = float(exp_scores[i] / exp_scores.sum())
            else:
                confidence = 1.0 / (i + 1)  # Decreasing by rank
            
            candidate = RepairCandidate(
                code=cleaned_code,
                confidence=confidence,
                beam_score=score,
                generation_time_ms=(time.time() - start_time) * 1000 / num_candidates,
                strategy="neural_beam_search",
                metadata={
                    "beam_rank": i,
                    "model": self.config.model_name,
                    "prompt_length": len(prompt),
                } if return_scores else {}
            )
            
            candidates.append(candidate)
        
        # Sort by confidence
        candidates.sort(key=lambda x: -x.confidence)
        
        # Update statistics
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["total_repairs"] += 1
        self.stats["total_time_ms"] += elapsed_ms
        self.stats["avg_candidates_per_repair"] = (
            (self.stats["avg_candidates_per_repair"] * (self.stats["total_repairs"] - 1) + len(candidates))
            / self.stats["total_repairs"]
        )
        
        return candidates
    
    def repair_batch(
        self,
        buggy_codes: List[str],
        counterexamples: Optional[List[Dict[str, Any]]] = None,
        num_candidates: int = 5
    ) -> List[List[RepairCandidate]]:
        """
        Generate repairs for multiple code snippets in batch.
        
        More efficient than calling repair() multiple times.
        
        Args:
            buggy_codes: List of buggy code snippets
            counterexamples: Optional list of counterexamples (one per code)
            num_candidates: Number of candidates per code
            
        Returns:
            List of candidate lists (one per input code)
        """
        if not self.is_available():
            return [[] for _ in buggy_codes]
        
        import torch
        
        # Prepare counterexamples
        if counterexamples is None:
            counterexamples = [None] * len(buggy_codes)
        
        # Construct prompts
        prompts = [
            self._construct_prompt(code, cex)
            for code, cex in zip(buggy_codes, counterexamples)
        ]
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            max_length=self.config.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate (this is tricky for batch with multiple outputs per input)
        # We'll process one at a time for now but could be optimized
        all_results = []
        
        for i in range(len(buggy_codes)):
            single_input = {k: v[i:i+1] for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **single_input,
                    max_length=self.config.max_target_length,
                    num_beams=max(self.config.num_beams, num_candidates),
                    num_return_sequences=num_candidates,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            # Process outputs
            candidates = []
            for j, seq in enumerate(outputs.sequences):
                decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
                cleaned = self._post_process_code(decoded, buggy_codes[i])
                
                candidates.append(RepairCandidate(
                    code=cleaned,
                    confidence=1.0 / (j + 1),
                    strategy="neural_batch"
                ))
            
            all_results.append(candidates)
        
        return all_results
    
    def _construct_prompt(
        self,
        buggy_code: str,
        counterexample: Optional[Dict[str, Any]] = None,
        specification: Optional[Any] = None
    ) -> str:
        """
        Construct the prompt for the repair model.
        
        Uses counterexample information when available to guide repair.
        """
        if counterexample:
            # Use counterexample template
            input_vals = counterexample.get("input_values", {})
            expected = counterexample.get("expected_output", "unknown")
            actual = counterexample.get("actual_output", "unknown")
            
            prompt = self.config.cex_template.format(
                code=buggy_code,
                input=json.dumps(input_vals) if isinstance(input_vals, dict) else str(input_vals),
                expected=str(expected),
                actual=str(actual)
            )
        else:
            # Simple template
            prompt = self.config.prompt_template.format(code=buggy_code)
        
        return prompt
    
    def _post_process_code(self, generated: str, original: str) -> str:
        """
        Post-process generated code to clean it up.
        
        - Removes model artifacts
        - Fixes common formatting issues
        - Falls back to original if generation is invalid
        """
        import re
        
        code = generated.strip()
        
        # Remove common artifacts
        code = code.replace("<pad>", "").replace("</s>", "").replace("<s>", "")
        code = code.replace("<extra_id_0>", "").replace("<extra_id_1>", "")
        
        # Try to extract code from markdown blocks
        code_match = re.search(r'```(?:cpp|c\+\+|c)?\n?(.*?)```', code, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        
        # Clean up whitespace
        code = code.strip()
        
        # Validate basic structure
        if not code:
            return original
        
        # Check for basic C++ validity (has braces, semicolons, etc.)
        if '{' not in code and '{' in original:
            # Generation seems incomplete, return original
            return original
        
        return code
    
    def save(self, path: str) -> None:
        """
        Save the model and configuration.
        
        Args:
            path: Directory to save to
        """
        if not self.is_available():
            logger.warning("Model not loaded, nothing to save")
            return
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save configuration
        with open(save_path / "repair_config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save statistics
        with open(save_path / "repair_stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self.stats,
            "model_loaded": self.is_loaded,
            "device": self.device,
            "model_name": self.config.model_name,
        }


class NeuralRepairTrainer:
    """
    Trainer for fine-tuning neural repair models.
    
    Usage:
        trainer = NeuralRepairTrainer(
            model_name="Salesforce/codet5-base",
            output_dir="models/my_repair_model"
        )
        
        trainer.train(
            train_data=[("buggy code", "fixed code"), ...],
            eval_data=[...]
        )
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/codet5-base",
        output_dir: str = "models/repair_model",
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        num_epochs: int = 5,
        warmup_steps: int = 100,
        max_source_length: int = 512,
        max_target_length: int = 512,
        fp16: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.fp16 = fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self) -> bool:
        """Check required dependencies."""
        try:
            import torch
            import transformers
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
    
    def train(
        self,
        train_data: List[Tuple[str, str]],
        eval_data: Optional[List[Tuple[str, str]]] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the repair model.
        
        Args:
            train_data: List of (buggy_code, fixed_code) pairs
            eval_data: Optional evaluation data
            resume_from_checkpoint: Path to resume from
            
        Returns:
            Training results dictionary
        """
        import torch
        from transformers import (
            T5ForConditionalGeneration,
            RobertaTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
        )
        from torch.utils.data import Dataset
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training on device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Create dataset class
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
                
                # Format source
                source_text = f"fix bug: {source}"
                
                source_encoding = self.tokenizer(
                    source_text,
                    max_length=self.max_source,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                target_encoding = self.tokenizer(
                    target,
                    max_length=self.max_target,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                labels = target_encoding.input_ids.squeeze()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                return {
                    "input_ids": source_encoding.input_ids.squeeze(),
                    "attention_mask": source_encoding.attention_mask.squeeze(),
                    "labels": labels,
                }
        
        # Create datasets
        train_dataset = RepairDataset(
            train_data, self.tokenizer,
            self.max_source_length, self.max_target_length
        )
        
        eval_dataset = None
        if eval_data:
            eval_dataset = RepairDataset(
                eval_data, self.tokenizer,
                self.max_source_length, self.max_target_length
            )
        
        # Training arguments
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            save_steps=500,
            save_total_limit=3,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            fp16=self.fp16 and self.device == "cuda",
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )
        
        # Callbacks
        callbacks = []
        if eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))
        
        # Save training config
        config = {
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "train_size": len(train_data),
            "eval_size": len(eval_data) if eval_data else 0,
        }
        
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Results
        results = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "model_path": str(self.output_dir / "final"),
        }
        
        logger.info(f"Training complete. Model saved to {self.output_dir / 'final'}")
        
        return results
    
    def evaluate(
        self,
        test_data: List[Tuple[str, str]],
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: List of (buggy_code, fixed_code) pairs
            model_path: Path to model (uses trained model if None)
            
        Returns:
            Evaluation metrics
        """
        # Load model for inference
        config = NeuralRepairConfig(
            model_path=model_path or str(self.output_dir / "final")
        )
        engine = NeuralRepairEngine(config)
        engine.load()
        
        # Evaluate
        correct = 0
        total = len(test_data)
        
        for buggy_code, expected_fix in test_data:
            candidates = engine.repair(buggy_code, num_candidates=1)
            
            if candidates and candidates[0].code.strip() == expected_fix.strip():
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }


def create_repair_engine(
    model_name: str = "Salesforce/codet5-base",
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> NeuralRepairEngine:
    """
    Factory function to create a neural repair engine.
    
    Args:
        model_name: HuggingFace model name
        model_path: Path to fine-tuned model (overrides model_name)
        device: Device to use (None = auto-detect)
        **kwargs: Additional configuration options
        
    Returns:
        Configured NeuralRepairEngine instance
    """
    config = NeuralRepairConfig(
        model_name=model_name,
        model_path=model_path,
        device=device,
        **{k: v for k, v in kwargs.items() if hasattr(NeuralRepairConfig, k)}
    )
    
    engine = NeuralRepairEngine(config)
    
    # Auto-load if model_path is specified
    if model_path:
        engine.load(model_path)
    
    return engine
