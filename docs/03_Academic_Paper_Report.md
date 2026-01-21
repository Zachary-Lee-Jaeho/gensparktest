# VEGA-Verified: Semantically Verified Neural Compiler Backend Generation

## A Technical Report

---

**Authors**: [Anonymous for Review]

**Affiliation**: [Anonymous]

**Date**: January 2026

---

## Abstract

Automatic compiler backend generation has emerged as a critical research area, with recent AI-driven approaches such as VEGA demonstrating promising results. However, these neural approaches lack formal guarantees of semantic correctness, limiting their applicability in safety-critical domains. We propose **VEGA-Verified**, a hybrid system that combines neural code generation with formal verification to provide semantic correctness guarantees. Our approach introduces three key contributions: (1) **Automated Semantic Specification Inference** that extracts formal specifications from reference implementations, (2) **Counterexample-Guided Neural Repair (CGNR)** that iteratively refines generated code based on verification failures, and (3) **Hierarchical Modular Verification** that enables scalable, compositional verification of complete compiler backends. We present the theoretical foundations, detailed algorithms, and implementation design for VEGA-Verified, demonstrating how it addresses the fundamental limitations of existing approaches while maintaining the efficiency benefits of neural code generation.

**Keywords**: Compiler Backend Generation, Formal Verification, Neural Code Generation, Program Synthesis, Counterexample-Guided Refinement

---

## 1. Introduction

### 1.1 Motivation

The development of compiler backends is a notoriously labor-intensive task, requiring deep expertise in both the target architecture and the compiler infrastructure. A typical LLVM backend consists of tens of thousands of lines of code across multiple functional modules, including instruction selection, register allocation, code emission, and assembly parsing [1].

Recent advances in AI-driven code generation have opened new possibilities for automating this process. VEGA [2], presented at CGO 2025, represents a significant breakthrough by using pre-trained transformer models (UniXcoder) to automatically generate compiler backend code. VEGA achieves 71.5% function-level accuracy on RISC-V, substantially outperforming traditional fork-flow approaches (<8%).

However, **VEGA's neural approach provides no formal guarantees of correctness**. The generated code may be syntactically valid but semantically incorrect, potentially introducing subtle bugs that are difficult to detect through testing alone. This limitation is particularly concerning for:

- **Safety-critical systems** where compiler correctness is essential
- **Novel architectures** where test coverage may be limited
- **Security-sensitive applications** where compiler bugs can introduce vulnerabilities

### 1.2 Problem Statement

We address the following research question:

> *How can we combine the efficiency of neural code generation with the reliability of formal verification to produce compiler backends that are both automatically generated and semantically correct?*

### 1.3 Contributions

We propose **VEGA-Verified**, a hybrid system with three main contributions:

1. **Automated Semantic Specification Inference**: We develop an algorithm that automatically extracts formal specifications (preconditions, postconditions, invariants) from existing reference implementations, eliminating the need for manual specification writing.

2. **Counterexample-Guided Neural Repair (CGNR)**: We introduce a novel repair algorithm that combines formal verification with neural code repair, using counterexamples from failed verifications to guide the repair model.

3. **Hierarchical Modular Verification**: We present a three-level verification architecture (function → module → backend) with interface contracts, enabling scalable and incremental verification of complete compiler backends.

### 1.4 Paper Organization

Section 2 reviews related work. Section 3 presents the theoretical background. Section 4 describes our approach in detail. Section 5 presents the implementation design. Section 6 discusses expected evaluation methodology. Section 7 concludes.

---

## 2. Related Work

### 2.1 Traditional Compiler Backend Generation

**Architecture Description Languages (ADLs)** have been the traditional approach for retargetable compiler generation:

- **LISA** [3]: Language for Instruction Set Architectures, used in CoWare's commercial tools
- **nML** [4]: Used in the Target Compiler Technologies toolchain
- **VADL/OpenVADL** [5]: Vienna ADL with LLVM backend generation support

These approaches require manually written architecture descriptions and have limited expressiveness for complex instruction patterns.

**TableGen** [6] is LLVM's domain-specific language for defining target-specific information. While powerful, TableGen requires substantial manual effort and expertise to use effectively.

### 2.2 Synthesis-Based Approaches

**Program synthesis** techniques have been applied to instruction selection:

- **Diospyros** [7]: Synthesis-based vectorization for DSPs using equality saturation
- **VeGen** [8]: Automatic vectorizer generation from instruction semantics
- **Hydride** [9]: Synthesis-based compiler for modern hardware architectures (ASPLOS 2024)

These approaches provide formal guarantees but are limited to specific domains (primarily vector instructions) and face scalability challenges with complex ISAs.

**ACT** [10] (2025) generates compiler backends for tensor accelerators using equality saturation and constraint programming. ACT provides soundness and completeness guarantees but requires manual ISA specifications and is limited to tensor accelerators.

### 2.3 Neural Code Generation

**Pre-trained code models** have shown remarkable capabilities:

- **Codex/Copilot** [11]: GPT-based code generation
- **CodeBERT** [12]: BERT-based code understanding
- **UniXcoder** [13]: Unified cross-modal pre-training for code

**VEGA** [2] applies UniXcoder to compiler backend generation, achieving significant improvements over traditional approaches but without formal correctness guarantees.

### 2.4 Formal Verification for Compilers

**CompCert** [14] is a formally verified C compiler, but requires manual proof development. **Alive2** [15] verifies LLVM optimizations but focuses on IR transformations rather than backend generation.

**3LA** [16] uses Instruction-Level Abstraction (ILA) for accelerator validation but requires manual formal models.

### 2.5 Positioning of VEGA-Verified

| Approach | Automation | Formality | Domain |
|----------|------------|-----------|--------|
| LISA/ADL | Low | Medium | General |
| Hydride | Medium | High | Vectors |
| ACT | High | High | Tensors |
| VEGA | High | Low | General |
| **VEGA-Verified** | **High** | **High** | **General** |

VEGA-Verified uniquely combines high automation (neural generation) with high formality (SMT verification) for general-purpose compiler backends.

---

## 3. Background

### 3.1 LLVM Compiler Backend Architecture

An LLVM backend transforms LLVM IR to target-specific machine code through several phases:

```
LLVM IR → SelectionDAG → MachineInstr → MCInst → Assembly/Object
```

Key components include:
- **ISelDAGToDAG**: Instruction selection patterns
- **AsmPrinter**: Assembly output generation
- **MCCodeEmitter**: Machine code encoding
- **RegisterInfo/InstrInfo**: Target resource descriptions

VEGA targets seven function modules containing ~1,500 functions for a complete backend.

### 3.2 Abstract Interpretation

**Abstract interpretation** [17] provides a framework for sound program analysis by computing over abstract domains that over-approximate concrete program behaviors.

**Definition 3.1 (Abstract Domain)**: An abstract domain is a complete lattice $(A, \sqsubseteq)$ with a Galois connection $(\alpha, \gamma)$ to concrete states.

We use abstract interpretation to extract specifications from reference implementations by:
1. Abstracting concrete values to symbolic patterns
2. Identifying invariant relationships across implementations
3. Deriving preconditions and postconditions

### 3.3 Hoare Logic and Verification Conditions

**Hoare logic** [18] provides a foundation for program verification through Hoare triples $\{P\}\ S\ \{Q\}$.

**Definition 3.2 (Verification Condition)**: For a program $S$ with specification $(Pre, Post)$, the verification condition is:
$$VC = Pre \Rightarrow wp(S, Post)$$
where $wp$ is the weakest precondition transformer.

We generate VCs from compiler backend functions and discharge them using SMT solvers.

### 3.4 Counterexample-Guided Abstraction Refinement

**CEGAR** [19] iteratively refines abstractions based on spurious counterexamples:

```
Abstract Model → Check → Counterexample → Real? → Refine
```

We adapt this paradigm to neural code repair, using verification counterexamples to guide the repair model.

---

## 4. VEGA-Verified Approach

### 4.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    VEGA-Verified Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Reference Backends ──► Specification Inference                 │
│                              │                                  │
│                              ▼                                  │
│  Target Description ──► VEGA Neural Generation                  │
│                              │                                  │
│                              ▼                                  │
│                    Formal Verification ◄──────┐                │
│                         │        │            │                │
│                    VERIFIED    FAILED         │                │
│                         │        │            │                │
│                         │        ▼            │                │
│                         │   CGNR Repair ──────┘                │
│                         │                                       │
│                         ▼                                       │
│                Hierarchical Verification                        │
│                (Function → Module → Backend)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Contribution 1: Automated Specification Inference

#### 4.2.1 Problem Formulation

Given reference implementations $R = \{r_1, ..., r_n\}$ of a function $f$ from different backends, infer a specification $Spec(f) = (Pre, Post, Inv)$ such that:

$$\forall r_i \in R: \{Pre\}\ r_i\ \{Post\} \text{ is valid}$$

#### 4.2.2 Algorithm

**Algorithm 1: SpecificationInference**

```
Input: Function name f, Reference implementations R
Output: Specification Spec(f) = (Pre, Post, Inv)

1. Parse each r_i into AST_i
2. Align implementations using GumTree edit distance
3. For aligned statement groups:
   - If target-independent: extract invariant
   - If target-specific: abstract to parametric pattern
4. Extract preconditions from guards (null checks, bounds checks)
5. Extract postconditions from return patterns
6. Validate: verify each r_i against inferred spec
7. Return (Pre, Post, Inv)
```

#### 4.2.3 Specification Language

We define a specification language suitable for compiler backend functions:

```
Spec ::= (Pre, Post, Inv)
Pre  ::= Condition | Pre ∧ Pre
Post ::= Condition | Post ∧ Post
Inv  ::= Condition | Inv ∧ Inv

Condition ::= Expr RelOp Expr
            | isValid(Var) | isInRange(Var, Lo, Hi)
            | implies(Condition, Condition)
```

#### 4.2.4 Soundness

**Theorem 4.1 (Inference Soundness)**: If Algorithm 1 succeeds, the inferred specification is sound with respect to all reference implementations.

*Proof Sketch*: The algorithm only includes conditions that hold across all references (step 3-5), and explicitly validates against each reference (step 6). □

### 4.3 Contribution 2: Counterexample-Guided Neural Repair

#### 4.3.1 Problem Formulation

Given code $C$ and specification $Spec$ where $verify(C, Spec) = FAIL$, find $C'$ such that $verify(C', Spec) = VERIFIED$ while minimizing $edit\_distance(C, C')$.

#### 4.3.2 CGNR Algorithm

**Algorithm 2: CGNR**

```
Input: Initial code C₀, Specification Spec, Max iterations K
Output: Verified code C* or FAIL

1. C ← C₀, history ← []
2. For i = 1 to K:
   3. VC ← generateVC(C, Spec)
   4. (result, model) ← SMTSolve(VC)
   5. If result = UNSAT: return (C, VERIFIED)
   6. CE ← extractCounterexample(model)
   7. history.append((C, CE))
   8. fault_loc ← localizeFault(C, CE)
   9. context ← buildRepairContext(C, CE, fault_loc, history)
   10. C ← NeuralRepair(context)
11. Return FAIL
```

#### 4.3.3 Repair Context Construction

The repair context provides the neural model with structured information:

```python
RepairContext = {
    "original_code": C,
    "counterexample": {
        "inputs": CE.inputs,
        "expected": CE.expected_output,
        "actual": CE.actual_output,
        "trace": CE.execution_trace
    },
    "fault_location": {
        "line": fault_loc.line,
        "statement": fault_loc.stmt
    },
    "specification": Spec,
    "repair_history": history[-3:]
}
```

#### 4.3.4 Theoretical Properties

**Theorem 4.2 (CGNR Soundness)**: If CGNR returns $(C^*, VERIFIED)$, then $\{Pre\}\ C^*\ \{Post\}$ holds.

*Proof*: CGNR returns VERIFIED only when $SMTSolve(\neg VC) = UNSAT$, meaning $VC = Pre \Rightarrow wp(C^*, Post)$ is valid. By wp semantics, the Hoare triple holds. □

**Theorem 4.3 (CGNR Progress)**: Each CGNR iteration either produces a correct fix or provides new information (counterexample) that was not previously seen.

*Proof Sketch*: If the same counterexample recurs, the repair model receives updated history context, changing its output. Duplicate counterexamples are handled by repair history. □

### 4.4 Contribution 3: Hierarchical Modular Verification

#### 4.4.1 Motivation

Verifying a complete backend monolithically is:
1. **Computationally expensive**: SMT solving scales poorly
2. **Non-incremental**: Changes require complete re-verification
3. **Non-compositional**: No reuse of verification results

#### 4.4.2 Three-Level Architecture

**Level 1 (Function)**: Verify each function against its specification
```
∀ f ∈ Functions: verify(f, Spec(f)) = VERIFIED
```

**Level 2 (Module)**: Verify module consistency and interface contracts
```
∀ M ∈ Modules: 
  - InternalConsistency(M)
  - SatisfiesContract(M, IC_M)
```

**Level 3 (Backend)**: Verify cross-module compatibility
```
∀ (M_i, M_j) ∈ Dependencies:
  Compatible(IC_i.Guarantees, IC_j.Assumptions)
```

#### 4.4.3 Interface Contracts

**Definition 4.1 (Interface Contract)**:
$$IC(M) = (Assumptions, Guarantees, Dependencies)$$

where:
- $Assumptions$: Conditions module $M$ expects from inputs
- $Guarantees$: Conditions module $M$ ensures for outputs
- $Dependencies$: Other modules $M$ depends on

**Definition 4.2 (Contract Compatibility)**:
$$Compatible(IC_1, IC_2) \Leftrightarrow \forall a \in IC_1.Assumptions: \exists g \in IC_2.Guarantees: g \Rightarrow a$$

#### 4.4.4 Compositional Soundness

**Theorem 4.4 (Compositional Verification)**: If hierarchical verification succeeds at all three levels, the composed backend is correct.

*Proof Sketch*: By assume-guarantee reasoning [20]:
1. Level 1 ensures each function is locally correct
2. Level 2 ensures modules satisfy their contracts
3. Level 3 ensures contract compatibility
4. By composition: full backend correctness follows □

---

## 5. Implementation Design

### 5.1 System Architecture

```
vega-verified/
├── src/
│   ├── specification/       # Contribution 1
│   │   ├── inferrer.py
│   │   ├── symbolic_exec.py
│   │   └── pattern_abstract.py
│   ├── verification/
│   │   ├── vcgen.py
│   │   └── smt_solver.py
│   ├── repair/              # Contribution 2
│   │   ├── cgnr.py
│   │   ├── fault_loc.py
│   │   └── repair_model.py
│   └── hierarchical/        # Contribution 3
│       ├── function_verify.py
│       ├── module_verify.py
│       └── interface_contract.py
├── models/
│   └── repair_model/
└── specs/
```

### 5.2 Key Components

#### 5.2.1 Verification Condition Generator

```python
class VCGenerator:
    def generate(self, code: str, spec: Specification) -> VC:
        """Generate verification conditions for compiler backend code."""
        ast = parse_cpp(code)
        cfg = build_cfg(ast)
        
        # Weakest precondition computation
        wp = self.compute_wp(cfg, spec.postconditions)
        
        # VC = Pre => wp
        vc = Implies(spec.preconditions, wp)
        return vc
    
    def compute_wp(self, cfg: CFG, post: Condition) -> Condition:
        """Compute weakest precondition through CFG."""
        # Backward analysis from exit nodes
        for node in cfg.reverse_postorder():
            if node.is_return():
                node.wp = substitute(post, result=node.return_value)
            elif node.is_assignment():
                node.wp = substitute(successor.wp, node.lhs=node.rhs)
            elif node.is_branch():
                node.wp = And(
                    Implies(node.condition, true_branch.wp),
                    Implies(Not(node.condition), false_branch.wp)
                )
            elif node.is_switch():
                node.wp = And(*[
                    Implies(Eq(node.expr, case.value), case.wp)
                    for case in node.cases
                ])
        return cfg.entry.wp
```

#### 5.2.2 SMT Solver Interface

```python
class SMTVerifier:
    def __init__(self):
        self.solver = z3.Solver()
    
    def verify(self, vc: VC) -> Tuple[Result, Optional[Model]]:
        """Check if VC is valid by checking unsatisfiability of negation."""
        self.solver.push()
        self.solver.add(z3.Not(vc.to_z3()))
        
        result = self.solver.check()
        
        if result == z3.unsat:
            return (VERIFIED, None)
        elif result == z3.sat:
            model = self.solver.model()
            ce = self.extract_counterexample(model)
            return (FAILED, ce)
        else:
            return (UNKNOWN, None)
```

#### 5.2.3 Neural Repair Model

```python
class RepairModel:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2Seq.from_pretrained(model_path)
    
    def repair(self, context: RepairContext) -> str:
        """Generate repaired code from context."""
        prompt = self.format_prompt(context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            num_return_sequences=5
        )
        
        candidates = [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]
        
        return self.select_best(candidates, context.specification)
    
    def format_prompt(self, context: RepairContext) -> str:
        return f"""
        [REPAIR TASK]
        Original code:
        {context.original_code}
        
        Counterexample:
        Input: {context.counterexample.inputs}
        Expected: {context.counterexample.expected}
        Actual: {context.counterexample.actual}
        
        Violated condition: {context.specification.violated}
        
        Fix the code to satisfy the specification:
        """
```

### 5.3 Integration with VEGA

VEGA-Verified integrates with VEGA as a post-processing verification and repair layer:

```python
class VEGAVerifiedPipeline:
    def __init__(self, vega_model, repair_model, verifier):
        self.vega = vega_model
        self.repair = repair_model
        self.verifier = verifier
        self.spec_inferrer = SpecificationInferrer()
    
    def generate_verified(self, target: str, references: List[str]) -> Backend:
        """Generate and verify a complete backend."""
        backend = Backend(target)
        
        for module in MODULES:
            for func in module.functions:
                # Step 1: Infer specification
                spec = self.spec_inferrer.infer(func, references)
                
                # Step 2: Generate with VEGA
                code = self.vega.generate(func, target)
                
                # Step 3: Verify and repair
                verified_code, result = self.cgnr(code, spec)
                
                if result == VERIFIED:
                    backend.add_function(module, func, verified_code)
                else:
                    backend.add_function(module, func, code, verified=False)
        
        # Step 4: Hierarchical verification
        self.hierarchical_verify(backend)
        
        return backend
```

---

## 6. Evaluation Plan

### 6.1 Research Questions

- **RQ1**: How effective is automated specification inference?
- **RQ2**: What is the repair success rate of CGNR?
- **RQ3**: How does VEGA-Verified compare to VEGA in accuracy?
- **RQ4**: What is the verification coverage and overhead?

### 6.2 Benchmarks

We plan to evaluate on VEGA's original benchmark:
- **RISC-V**: Standard RISC-V backend
- **RI5CY**: PULP platform with extensions
- **xCORE**: XMOS IoT processor

### 6.3 Metrics

| Metric | Definition |
|--------|------------|
| Spec Inference Rate | % functions with successfully inferred specs |
| CGNR Success Rate | % failed verifications successfully repaired |
| Function Accuracy | % functions passing verification |
| Verification Coverage | % code covered by verification |
| Overhead | Additional time over base VEGA |

### 6.4 Expected Results

Based on our preliminary analysis:

| Metric | VEGA | VEGA-Verified (Expected) |
|--------|------|--------------------------|
| Function Accuracy | 71.5% | 85-90% |
| Verified Functions | 0% | 80-90% |
| Generation Time | <1 hour | ~2 hours |

---

## 7. Conclusion

We have presented VEGA-Verified, a hybrid approach to compiler backend generation that combines neural code generation with formal verification. Our three contributions—automated specification inference, counterexample-guided neural repair, and hierarchical modular verification—address the fundamental limitations of existing approaches while maintaining practical efficiency.

The key insight is that neural models excel at generating plausible code quickly, while formal methods excel at identifying subtle correctness issues. By combining these complementary strengths, VEGA-Verified can achieve both the efficiency of neural approaches and the reliability of formal methods.

### Future Work

1. **Specification Refinement**: Iteratively improve specifications based on verification feedback
2. **Transfer Learning**: Apply learned repair patterns across different targets
3. **Incremental Verification**: Efficiently re-verify after small changes
4. **Extension to Other Compilers**: Adapt the approach to GCC and other compiler infrastructures

---

## References

[1] LLVM Project. "Writing an LLVM Backend." https://llvm.org/docs/WritingAnLLVMBackend.html

[2] Zhong, M., et al. "VEGA: Automatically Generating Compiler Backends Using a Pre-Trained Transformer Model." CGO 2025.

[3] Hoffmann, A., et al. "A novel methodology for the design of application-specific instruction-set processors (ASIPs) using a machine description language." TCAD 2001.

[4] Fauth, A., et al. "Describing instruction set processors using nML." EDTC 1995.

[5] Per, K., et al. "OpenVADL: An Open Source Implementation of the Vienna Architecture Description Language." 2025.

[6] LLVM Project. "TableGen Overview." https://llvm.org/docs/TableGen/

[7] VanHattum, A., et al. "Vectorization for digital signal processors via equality saturation." ASPLOS 2021.

[8] Chen, Y., et al. "VeGen: A Vectorizer Generator for SIMD and Beyond." ASPLOS 2021.

[9] Kothen, A., et al. "Hydride: A Retargetable and Extensible Synthesis-based Compiler for Modern Hardware Architectures." ASPLOS 2024.

[10] "ACT: Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions." arXiv 2025.

[11] Chen, M., et al. "Evaluating Large Language Models Trained on Code." arXiv 2021.

[12] Feng, Z., et al. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages." EMNLP 2020.

[13] Guo, D., et al. "UniXcoder: Unified Cross-Modal Pre-training for Code Representation." ACL 2022.

[14] Leroy, X. "Formal Verification of a Realistic Compiler." CACM 2009.

[15] Lopes, N., et al. "Alive2: Bounded Translation Validation for LLVM." PLDI 2021.

[16] Huang, B., et al. "Application-Level Validation of Accelerator Designs Using a Formal Software/Hardware Interface." MICRO 2022.

[17] Cousot, P., Cousot, R. "Abstract interpretation: a unified lattice model for static analysis of programs." POPL 1977.

[18] Hoare, C.A.R. "An axiomatic basis for computer programming." CACM 1969.

[19] Clarke, E., et al. "Counterexample-guided abstraction refinement." CAV 2000.

[20] Jones, C.B. "Tentative steps toward a development method for interfering programs." TOPLAS 1983.

---

## Appendix A: Specification Examples

### A.1 getRelocType Specification

```
Function: getRelocType(Fixup, Target, IsPCRel) -> unsigned

Preconditions:
  P1: Fixup.isValid()
  P2: Fixup.getTargetKind() ∈ {FK_NONE, FK_Data_1, FK_Data_2, FK_Data_4, FK_Data_8, ...}
  P3: Target.isInitialized()

Postconditions:
  Q1: result ∈ ValidRelocationTypes(Target.getArch())
  Q2: result >= 0
  Q3: sizeOf(result) = expectedSize(Fixup.getTargetKind())

Invariants:
  I1: FK_NONE → result = R_*_NONE
  I2: FK_Data_1 → result ∈ {R_*_8, ...}
  I3: FK_Data_4 ∧ IsPCRel → result ∈ PCRelativeRelocations
  I4: FK_Data_4 ∧ ¬IsPCRel → result ∈ AbsoluteRelocations
```

### A.2 emitInstruction Specification

```
Function: emitInstruction(MI, Streamer) -> void

Preconditions:
  P1: MI.isValid()
  P2: Streamer.isReady()
  P3: MI.getOpcode() ∈ ValidOpcodes

Postconditions:
  Q1: Streamer.bytesEmitted() > 0
  Q2: encodingCorrect(MI, Streamer.lastEmitted())

Invariants:
  I1: hasRelocation(MI) → Streamer.pendingFixups().size() > 0
```

---

## Appendix B: CGNR Repair Examples

### B.1 Missing Case Repair

**Original (buggy)**:
```cpp
case FK_Data_4: return ELF::R_RISCV_32;
```

**Counterexample**:
```
Input: {Fixup.kind: FK_Data_4, IsPCRel: true}
Expected: R_RISCV_PC32
Actual: R_RISCV_32
Violated: "IsPCRel implies PC-relative relocation"
```

**Repaired**:
```cpp
case FK_Data_4: return IsPCRel ? ELF::R_RISCV_PC32 : ELF::R_RISCV_32;
```

### B.2 Wrong Relocation Type Repair

**Original (buggy)**:
```cpp
case FK_Data_8: return ELF::R_RISCV_32;
```

**Counterexample**:
```
Input: {Fixup.kind: FK_Data_8}
Expected: R_RISCV_64
Actual: R_RISCV_32
Violated: "FK_Data_8 requires 64-bit relocation"
```

**Repaired**:
```cpp
case FK_Data_8: return ELF::R_RISCV_64;
```
