"""
Phase 2.3: Training Data Preparation for Neural Repair Model.

This module prepares training data for fine-tuning code repair models:
1. Bug-Fix Pair Extraction from LLVM commits
2. Synthetic Bug Generation from correct code
3. Counterexample-guided training data creation
4. Data augmentation and preprocessing

Training data format:
- Input: Buggy code + counterexample
- Output: Fixed code
- Metadata: Bug type, architecture, function type

Supported Models:
- CodeBERT/GraphCodeBERT for code understanding
- CodeT5/CodeT5+ for seq2seq repair
- UniXcoder for unified code representation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
import random
import re
from pathlib import Path
from datetime import datetime


class BugType(Enum):
    """Types of bugs for training data."""
    MISSING_CASE = "missing_case"
    WRONG_RETURN = "wrong_return"
    MISSING_CONDITION = "missing_condition"
    WRONG_CONDITION = "wrong_condition"
    MISSING_BREAK = "missing_break"
    OFF_BY_ONE = "off_by_one"
    WRONG_OPERATOR = "wrong_operator"
    MISSING_NULL_CHECK = "missing_null_check"
    TYPE_MISMATCH = "type_mismatch"
    SEMANTIC_ERROR = "semantic_error"


@dataclass
class TrainingExample:
    """A single training example for code repair."""
    id: str
    buggy_code: str
    fixed_code: str
    bug_type: BugType
    counterexample: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "buggy_code": self.buggy_code,
            "fixed_code": self.fixed_code,
            "bug_type": self.bug_type.value,
            "counterexample": self.counterexample,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingExample":
        return cls(
            id=data["id"],
            buggy_code=data["buggy_code"],
            fixed_code=data["fixed_code"],
            bug_type=BugType(data["bug_type"]),
            counterexample=data.get("counterexample"),
            metadata=data.get("metadata", {}),
        )
    
    def to_seq2seq(self, include_counterexample: bool = True) -> Tuple[str, str]:
        """Convert to seq2seq format (input, output)."""
        if include_counterexample and self.counterexample:
            input_text = f"<BUG> {self.buggy_code} </BUG> <CEX> {json.dumps(self.counterexample)} </CEX>"
        else:
            input_text = f"<BUG> {self.buggy_code} </BUG>"
        
        output_text = f"<FIX> {self.fixed_code} </FIX>"
        return input_text, output_text


@dataclass
class TrainingDataset:
    """Collection of training examples."""
    name: str
    examples: List[TrainingExample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_example(self, example: TrainingExample) -> None:
        self.examples.append(example)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        bug_type_counts = {}
        for ex in self.examples:
            bug_type_counts[ex.bug_type.value] = bug_type_counts.get(ex.bug_type.value, 0) + 1
        
        return {
            "total_examples": len(self.examples),
            "bug_type_distribution": bug_type_counts,
            "with_counterexample": sum(1 for ex in self.examples if ex.counterexample),
        }
    
    def save(self, path: str) -> None:
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "metadata": {
                **self.metadata,
                "created_at": datetime.now().isoformat(),
                "statistics": self.get_statistics(),
            },
            "examples": [ex.to_dict() for ex in self.examples],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingDataset":
        """Load dataset from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        dataset = cls(
            name=data["name"],
            metadata=data.get("metadata", {}),
        )
        
        for ex_data in data.get("examples", []):
            dataset.add_example(TrainingExample.from_dict(ex_data))
        
        return dataset
    
    def split(self, train_ratio: float = 0.8, seed: int = 42) -> Tuple["TrainingDataset", "TrainingDataset"]:
        """Split into train and validation sets."""
        random.seed(seed)
        shuffled = self.examples.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        
        train_set = TrainingDataset(
            name=f"{self.name}_train",
            examples=shuffled[:split_idx],
            metadata={"split": "train", "parent": self.name}
        )
        
        val_set = TrainingDataset(
            name=f"{self.name}_val",
            examples=shuffled[split_idx:],
            metadata={"split": "val", "parent": self.name}
        )
        
        return train_set, val_set


class SyntheticBugGenerator:
    """
    Generates synthetic bugs from correct code for training.
    
    Bug injection strategies:
    1. Remove case from switch
    2. Change return value
    3. Remove condition check
    4. Swap operators
    5. Off-by-one mutations
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.mutation_count = 0
    
    def generate_missing_case(self, code: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Generate bug by removing a case from switch."""
        # Find all cases
        case_pattern = re.compile(r'(case\s+([\w:]+):\s*\n(?:[^}]+?)return\s+([^;]+);)', re.MULTILINE)
        matches = list(case_pattern.finditer(code))
        
        if len(matches) < 2:  # Need at least 2 cases
            return None
        
        # Randomly select a case to remove
        idx = random.randint(0, len(matches) - 1)
        match = matches[idx]
        
        # Remove the case
        buggy_code = code[:match.start()] + code[match.end():]
        
        counterexample = {
            "input_values": {"Kind": match.group(2)},
            "expected_output": match.group(3),
            "actual_output": "default/unknown",
        }
        
        return buggy_code, {
            "bug_type": BugType.MISSING_CASE,
            "removed_case": match.group(2),
            "counterexample": counterexample,
        }
    
    def generate_wrong_return(self, code: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Generate bug by changing a return value."""
        # Find all return statements with relocation types
        return_pattern = re.compile(r'return\s+([\w:]+::R_\w+);', re.MULTILINE)
        matches = list(return_pattern.finditer(code))
        
        if not matches:
            return None
        
        # Randomly select a return to modify
        idx = random.randint(0, len(matches) - 1)
        match = matches[idx]
        original_return = match.group(1)
        
        # Generate a wrong return (swap some part)
        if '_32' in original_return:
            wrong_return = original_return.replace('_32', '_64')
        elif '_64' in original_return:
            wrong_return = original_return.replace('_64', '_32')
        elif 'HI20' in original_return:
            wrong_return = original_return.replace('HI20', 'LO12')
        elif 'LO12' in original_return:
            wrong_return = original_return.replace('LO12', 'HI20')
        else:
            # Generic mutation
            wrong_return = original_return + "_WRONG"
        
        buggy_code = code[:match.start()] + f"return {wrong_return};" + code[match.end():]
        
        counterexample = {
            "expected_output": original_return,
            "actual_output": wrong_return,
        }
        
        return buggy_code, {
            "bug_type": BugType.WRONG_RETURN,
            "original_return": original_return,
            "wrong_return": wrong_return,
            "counterexample": counterexample,
        }
    
    def generate_missing_condition(self, code: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Generate bug by removing a condition check."""
        # Find IsPCRel checks
        if_pattern = re.compile(r'if\s*\(\s*IsPCRel\s*\)\s*\{[^}]+\}', re.MULTILINE | re.DOTALL)
        match = if_pattern.search(code)
        
        if not match:
            return None
        
        # Remove the if block (keep just the else content if any)
        buggy_code = code[:match.start()] + code[match.end():]
        
        counterexample = {
            "input_values": {"IsPCRel": True},
            "expected_output": "PCRel relocation",
            "actual_output": "non-PCRel relocation",
        }
        
        return buggy_code, {
            "bug_type": BugType.MISSING_CONDITION,
            "removed_condition": "IsPCRel check",
            "counterexample": counterexample,
        }
    
    def generate_wrong_operator(self, code: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Generate bug by changing an operator."""
        # Find comparison operators
        patterns = [
            (r'(\w+)\s*==\s*(\w+)', '!='),
            (r'(\w+)\s*!=\s*(\w+)', '=='),
            (r'(\w+)\s*<\s*(\w+)', '<='),
            (r'(\w+)\s*>\s*(\w+)', '>='),
        ]
        
        for pattern, replacement in patterns:
            match = re.search(pattern, code)
            if match:
                # Replace operator
                original = match.group(0)
                left = match.group(1)
                right = match.group(2)
                
                wrong = f"{left} {replacement} {right}"
                buggy_code = code[:match.start()] + wrong + code[match.end():]
                
                return buggy_code, {
                    "bug_type": BugType.WRONG_OPERATOR,
                    "original": original,
                    "wrong": wrong,
                }
        
        return None
    
    def generate_bugs(
        self,
        code: str,
        num_bugs: int = 5,
        bug_types: Optional[List[BugType]] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate multiple synthetic bugs from code.
        
        Args:
            code: Correct source code
            num_bugs: Number of bugs to generate
            bug_types: Types of bugs to generate (None = all)
            
        Returns:
            List of (buggy_code, metadata) tuples
        """
        generators = {
            BugType.MISSING_CASE: self.generate_missing_case,
            BugType.WRONG_RETURN: self.generate_wrong_return,
            BugType.MISSING_CONDITION: self.generate_missing_condition,
            BugType.WRONG_OPERATOR: self.generate_wrong_operator,
        }
        
        if bug_types:
            generators = {k: v for k, v in generators.items() if k in bug_types}
        
        bugs = []
        attempts = 0
        max_attempts = num_bugs * 10
        
        while len(bugs) < num_bugs and attempts < max_attempts:
            attempts += 1
            
            # Random generator
            gen_type = random.choice(list(generators.keys()))
            generator = generators[gen_type]
            
            result = generator(code)
            if result:
                bugs.append(result)
                self.mutation_count += 1
        
        return bugs


class TrainingDataPipeline:
    """
    Pipeline for creating training data from LLVM sources.
    
    Steps:
    1. Extract correct functions from ground truth
    2. Generate synthetic bugs
    3. Create counterexamples via verification
    4. Format as training examples
    """
    
    def __init__(
        self,
        ground_truth_path: str,
        output_dir: str,
        verbose: bool = False
    ):
        self.ground_truth_path = Path(ground_truth_path)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.bug_generator = SyntheticBugGenerator()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth database."""
        with open(self.ground_truth_path, 'r') as f:
            return json.load(f)
    
    def create_training_dataset(
        self,
        bugs_per_function: int = 5,
        min_code_length: int = 100,
        max_functions: Optional[int] = None
    ) -> TrainingDataset:
        """
        Create training dataset from ground truth functions.
        
        Args:
            bugs_per_function: Number of synthetic bugs per function
            min_code_length: Minimum code length to consider
            max_functions: Maximum number of functions to process
            
        Returns:
            TrainingDataset with synthetic bug-fix pairs
        """
        db = self.load_ground_truth()
        dataset = TrainingDataset(
            name="vega_synthetic_bugs",
            metadata={
                "source": str(self.ground_truth_path),
                "bugs_per_function": bugs_per_function,
            }
        )
        
        functions = list(db.get("functions", {}).items())
        if max_functions:
            functions = functions[:max_functions]
        
        example_id = 0
        
        for func_id, func_data in functions:
            body = func_data.get("body", "")
            
            # Skip short functions
            if len(body) < min_code_length:
                continue
            
            # Skip functions without switches
            if "switch" not in body:
                continue
            
            if self.verbose:
                print(f"Processing: {func_data.get('name', func_id)}")
            
            # Generate synthetic bugs
            bugs = self.bug_generator.generate_bugs(body, bugs_per_function)
            
            for buggy_code, bug_info in bugs:
                example = TrainingExample(
                    id=f"synthetic_{example_id}",
                    buggy_code=buggy_code,
                    fixed_code=body,  # Original is the fix
                    bug_type=bug_info["bug_type"],
                    counterexample=bug_info.get("counterexample"),
                    metadata={
                        "source_function": func_data.get("name", func_id),
                        "backend": func_data.get("backend", "unknown"),
                        "bug_info": {k: v for k, v in bug_info.items() if k != "counterexample"},
                    }
                )
                dataset.add_example(example)
                example_id += 1
        
        return dataset
    
    def create_from_commits(
        self,
        commit_data: List[Dict[str, Any]]
    ) -> TrainingDataset:
        """
        Create training dataset from LLVM commit data.
        
        Expected format:
        {
            "commit_hash": "abc123",
            "before_code": "...",
            "after_code": "...",
            "commit_message": "...",
            "files_changed": ["..."]
        }
        """
        dataset = TrainingDataset(
            name="llvm_commits",
            metadata={"source": "llvm_commits"}
        )
        
        for i, commit in enumerate(commit_data):
            # Determine bug type from commit message
            msg = commit.get("commit_message", "").lower()
            bug_type = BugType.SEMANTIC_ERROR  # Default
            
            if "missing case" in msg:
                bug_type = BugType.MISSING_CASE
            elif "wrong return" in msg or "return value" in msg:
                bug_type = BugType.WRONG_RETURN
            elif "pcrel" in msg or "pc-rel" in msg:
                bug_type = BugType.MISSING_CONDITION
            elif "off by one" in msg or "off-by-one" in msg:
                bug_type = BugType.OFF_BY_ONE
            
            example = TrainingExample(
                id=f"commit_{commit.get('commit_hash', i)[:8]}",
                buggy_code=commit.get("before_code", ""),
                fixed_code=commit.get("after_code", ""),
                bug_type=bug_type,
                metadata={
                    "commit_hash": commit.get("commit_hash"),
                    "commit_message": commit.get("commit_message"),
                    "files_changed": commit.get("files_changed", []),
                }
            )
            dataset.add_example(example)
        
        return dataset
    
    def augment_dataset(
        self,
        dataset: TrainingDataset,
        augmentation_factor: int = 2
    ) -> TrainingDataset:
        """
        Augment training dataset with variations.
        
        Augmentation strategies:
        1. Variable renaming
        2. Whitespace variations
        3. Comment addition/removal
        """
        augmented = TrainingDataset(
            name=f"{dataset.name}_augmented",
            metadata={
                **dataset.metadata,
                "augmentation_factor": augmentation_factor,
            }
        )
        
        # Keep original examples
        for ex in dataset.examples:
            augmented.add_example(ex)
        
        # Add augmented examples
        for ex in dataset.examples:
            for i in range(augmentation_factor - 1):
                aug_ex = self._augment_example(ex, i)
                augmented.add_example(aug_ex)
        
        return augmented
    
    def _augment_example(self, example: TrainingExample, variation: int) -> TrainingExample:
        """Create an augmented version of an example."""
        buggy = example.buggy_code
        fixed = example.fixed_code
        
        if variation % 3 == 0:
            # Whitespace variation
            buggy = re.sub(r'\s+', ' ', buggy)
            fixed = re.sub(r'\s+', ' ', fixed)
        elif variation % 3 == 1:
            # Add extra newlines
            buggy = buggy.replace(';', ';\n')
            fixed = fixed.replace(';', ';\n')
        else:
            # Remove comments
            buggy = re.sub(r'//.*?\n', '\n', buggy)
            buggy = re.sub(r'/\*.*?\*/', '', buggy, flags=re.DOTALL)
            fixed = re.sub(r'//.*?\n', '\n', fixed)
            fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        
        return TrainingExample(
            id=f"{example.id}_aug{variation}",
            buggy_code=buggy,
            fixed_code=fixed,
            bug_type=example.bug_type,
            counterexample=example.counterexample,
            metadata={
                **example.metadata,
                "augmented_from": example.id,
                "augmentation_type": variation % 3,
            }
        )
    
    def export_for_huggingface(
        self,
        dataset: TrainingDataset,
        output_path: str
    ) -> None:
        """
        Export dataset in HuggingFace format.
        
        Creates:
        - train.jsonl
        - val.jsonl
        """
        train_set, val_set = dataset.split()
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write train
        with open(output_dir / "train.jsonl", 'w') as f:
            for ex in train_set.examples:
                input_text, output_text = ex.to_seq2seq()
                f.write(json.dumps({
                    "input": input_text,
                    "output": output_text,
                    "bug_type": ex.bug_type.value,
                }) + "\n")
        
        # Write validation
        with open(output_dir / "val.jsonl", 'w') as f:
            for ex in val_set.examples:
                input_text, output_text = ex.to_seq2seq()
                f.write(json.dumps({
                    "input": input_text,
                    "output": output_text,
                    "bug_type": ex.bug_type.value,
                }) + "\n")
        
        # Write metadata
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump({
                "name": dataset.name,
                "train_size": len(train_set.examples),
                "val_size": len(val_set.examples),
                "statistics": dataset.get_statistics(),
            }, f, indent=2)


def create_training_data_from_ground_truth(
    ground_truth_path: str,
    output_dir: str,
    bugs_per_function: int = 5,
    augmentation_factor: int = 2,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to create training data.
    
    Args:
        ground_truth_path: Path to ground truth database
        output_dir: Output directory
        bugs_per_function: Number of synthetic bugs per function
        augmentation_factor: Data augmentation factor
        verbose: Print verbose output
        
    Returns:
        Summary statistics
    """
    pipeline = TrainingDataPipeline(
        ground_truth_path=ground_truth_path,
        output_dir=output_dir,
        verbose=verbose
    )
    
    # Create base dataset
    dataset = pipeline.create_training_dataset(
        bugs_per_function=bugs_per_function
    )
    
    if verbose:
        print(f"\nBase dataset: {len(dataset.examples)} examples")
    
    # Augment
    if augmentation_factor > 1:
        dataset = pipeline.augment_dataset(dataset, augmentation_factor)
        if verbose:
            print(f"Augmented dataset: {len(dataset.examples)} examples")
    
    # Save
    dataset.save(f"{output_dir}/training_data.json")
    
    # Export for HuggingFace
    pipeline.export_for_huggingface(dataset, f"{output_dir}/huggingface")
    
    return dataset.get_statistics()


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 2.3: Training Data Pipeline Demo")
    print("=" * 70)
    
    # Test synthetic bug generation
    test_code = """
unsigned getRelocType(unsigned Kind, bool IsPCRel) {
    if (IsPCRel) {
        switch (Kind) {
        case FK_Data_4:
            return ELF::R_RISCV_32_PCREL;
        default:
            return ELF::R_RISCV_NONE;
        }
    }
    
    switch (Kind) {
    case FK_NONE:
        return ELF::R_RISCV_NONE;
    case FK_Data_4:
        return ELF::R_RISCV_32;
    case FK_Data_8:
        return ELF::R_RISCV_64;
    case RISCV::fixup_riscv_hi20:
        return ELF::R_RISCV_HI20;
    default:
        llvm_unreachable("Unknown fixup kind");
    }
}
"""
    
    generator = SyntheticBugGenerator()
    
    print("\n🐛 Generating synthetic bugs...")
    bugs = generator.generate_bugs(test_code, num_bugs=5)
    
    for i, (buggy_code, info) in enumerate(bugs):
        print(f"\n--- Bug {i+1}: {info['bug_type'].value} ---")
        if info.get('counterexample'):
            print(f"Counterexample: {info['counterexample']}")
        print(f"Buggy code preview: {buggy_code[:200]}...")
    
    # Test training example
    print("\n📝 Creating training example...")
    example = TrainingExample(
        id="test_001",
        buggy_code=bugs[0][0] if bugs else test_code,
        fixed_code=test_code,
        bug_type=bugs[0][1]["bug_type"] if bugs else BugType.SEMANTIC_ERROR,
        counterexample=bugs[0][1].get("counterexample") if bugs else None,
        metadata={"source": "test"}
    )
    
    input_text, output_text = example.to_seq2seq()
    print(f"Input length: {len(input_text)}")
    print(f"Output length: {len(output_text)}")
    
    # Test dataset
    print("\n📊 Creating dataset...")
    dataset = TrainingDataset(name="test_dataset")
    dataset.add_example(example)
    
    stats = dataset.get_statistics()
    print(f"Dataset statistics: {stats}")
    
    print("\n✅ Training Data Pipeline Demo Complete")
