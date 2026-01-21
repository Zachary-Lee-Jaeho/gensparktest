# VEGA-Verified 구현 설계 문서

## 목차
1. [시스템 아키텍처](#1-시스템-아키텍처)
2. [모듈 설계](#2-모듈-설계)
3. [데이터 모델](#3-데이터-모델)
4. [인터페이스 설계](#4-인터페이스-설계)
5. [구현 로드맵](#5-구현-로드맵)
6. [테스트 계획](#6-테스트-계획)

---

## 1. 시스템 아키텍처

### 1.1 고수준 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VEGA-Verified System                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Input Layer                                  │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │   │
│  │  │ Reference     │  │ Target        │  │ VEGA          │           │   │
│  │  │ Backends      │  │ Description   │  │ Model         │           │   │
│  │  │ (C++ code)    │  │ (.td files)   │  │ (UniXcoder)   │           │   │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘           │   │
│  └──────────┼──────────────────┼──────────────────┼─────────────────────┘   │
│             │                  │                  │                         │
│             ▼                  ▼                  ▼                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Processing Layer                                │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │   │
│  │  │ Specification   │  │ Neural          │  │ Verification    │     │   │
│  │  │ Inference       │  │ Generation      │  │ Engine          │     │   │
│  │  │ Engine          │  │ (VEGA)          │  │ (SMT/BMC)       │     │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │   │
│  │           │                    │                    │              │   │
│  │           └────────────────────┼────────────────────┘              │   │
│  │                                │                                    │   │
│  │                                ▼                                    │   │
│  │                    ┌─────────────────────┐                         │   │
│  │                    │   CGNR Engine       │                         │   │
│  │                    │   (Repair Loop)     │                         │   │
│  │                    └─────────────────────┘                         │   │
│  │                                │                                    │   │
│  │                                ▼                                    │   │
│  │                    ┌─────────────────────┐                         │   │
│  │                    │   Hierarchical      │                         │   │
│  │                    │   Verifier          │                         │   │
│  │                    └─────────────────────┘                         │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Output Layer                                 │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │   │
│  │  │ Verified      │  │ Verification  │  │ Repair        │           │   │
│  │  │ Backend       │  │ Report        │  │ Log           │           │   │
│  │  │ Code          │  │               │  │               │           │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 디렉토리 구조

```
vega-verified/
│
├── src/                           # 소스 코드
│   ├── __init__.py
│   ├── main.py                    # 메인 엔트리 포인트
│   │
│   ├── specification/             # Contribution 1: Spec Inference
│   │   ├── __init__.py
│   │   ├── inferrer.py           # 메인 추론 엔진
│   │   ├── symbolic_exec.py      # 심볼릭 실행
│   │   ├── pattern_abstract.py   # 패턴 추상화
│   │   ├── condition_extract.py  # 조건 추출
│   │   └── spec_language.py      # Specification 언어 정의
│   │
│   ├── verification/              # 검증 엔진
│   │   ├── __init__.py
│   │   ├── vcgen.py              # VC 생성기
│   │   ├── smt_solver.py         # Z3 인터페이스
│   │   ├── bmc.py                # Bounded Model Checking
│   │   └── verifier.py           # 메인 검증기
│   │
│   ├── repair/                    # Contribution 2: CGNR
│   │   ├── __init__.py
│   │   ├── cgnr.py               # CGNR 알고리즘
│   │   ├── fault_loc.py          # 결함 위치 추정
│   │   ├── repair_context.py     # 수리 컨텍스트 구성
│   │   ├── repair_model.py       # 신경망 수리 모델
│   │   └── candidate_select.py   # 후보 선택
│   │
│   ├── hierarchical/              # Contribution 3: 계층적 검증
│   │   ├── __init__.py
│   │   ├── function_verify.py    # Level 1: 함수 검증
│   │   ├── module_verify.py      # Level 2: 모듈 검증
│   │   ├── backend_verify.py     # Level 3: 백엔드 검증
│   │   └── interface_contract.py # 인터페이스 계약
│   │
│   ├── parsing/                   # 코드 파싱
│   │   ├── __init__.py
│   │   ├── cpp_parser.py         # C++ 파싱
│   │   ├── ast_utils.py          # AST 유틸리티
│   │   └── cfg_builder.py        # CFG 구축
│   │
│   ├── integration/               # 외부 시스템 통합
│   │   ├── __init__.py
│   │   ├── vega_adapter.py       # VEGA 모델 어댑터
│   │   ├── llvm_adapter.py       # LLVM 통합
│   │   └── tablegen_parser.py    # TableGen 파싱
│   │
│   └── utils/                     # 유틸리티
│       ├── __init__.py
│       ├── logging.py
│       ├── config.py
│       └── metrics.py
│
├── models/                        # 모델 파일
│   ├── repair_model/             # 수리 모델
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   └── vega_model/               # VEGA 원본 모델
│
├── specs/                         # Specification 파일
│   ├── templates/                # Spec 템플릿
│   │   ├── asmprinter.spec
│   │   ├── iseldagtodag.spec
│   │   └── mccodeemitter.spec
│   └── inferred/                 # 추론된 Spec
│
├── tests/                         # 테스트
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
│
├── docs/                          # 문서
│
├── configs/                       # 설정 파일
│   ├── default.yaml
│   └── targets/
│       ├── riscv.yaml
│       ├── pulp.yaml
│       └── xcore.yaml
│
├── scripts/                       # 스크립트
│   ├── train_repair_model.py
│   ├── evaluate.py
│   └── generate_backend.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 2. 모듈 설계

### 2.1 Specification Inference Module

#### 2.1.1 클래스 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                    specification/                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │ SpecificationInferrer│◄──────│ PatternAbstractor   │         │
│  ├─────────────────────┤       ├─────────────────────┤         │
│  │ - references        │       │ - patterns          │         │
│  │ - spec_templates    │       │ + abstract()        │         │
│  ├─────────────────────┤       │ + generalize()      │         │
│  │ + infer()           │       └─────────────────────┘         │
│  │ + validate()        │                                        │
│  └──────────┬──────────┘                                        │
│             │                                                    │
│             │ uses                                               │
│             ▼                                                    │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │ SymbolicExecutor    │       │ ConditionExtractor  │         │
│  ├─────────────────────┤       ├─────────────────────┤         │
│  │ - state             │       │ + extract_pre()     │         │
│  │ - path_constraints  │       │ + extract_post()    │         │
│  ├─────────────────────┤       │ + extract_inv()     │         │
│  │ + execute()         │       └─────────────────────┘         │
│  │ + get_constraints() │                                        │
│  └─────────────────────┘                                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Specification                                            │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ - preconditions: List[Condition]                         │   │
│  │ - postconditions: List[Condition]                        │   │
│  │ - invariants: List[Condition]                            │   │
│  │ - function_name: str                                     │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ + to_smt(): z3.Formula                                   │   │
│  │ + validate(code: str): bool                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.1.2 핵심 인터페이스

```python
# src/specification/inferrer.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

@dataclass
class Specification:
    """Formal specification for a function."""
    function_name: str
    preconditions: List['Condition']
    postconditions: List['Condition']
    invariants: List['Condition']
    
    def to_smt(self) -> 'z3.Formula':
        """Convert to SMT formula for verification."""
        pass
    
    def to_json(self) -> Dict:
        """Serialize to JSON for storage."""
        pass


class SpecificationInferrer:
    """Main engine for specification inference."""
    
    def __init__(self, config: Dict):
        self.symbolic_executor = SymbolicExecutor()
        self.pattern_abstractor = PatternAbstractor()
        self.condition_extractor = ConditionExtractor()
    
    def infer(self, function_name: str, 
              references: List[str]) -> Specification:
        """
        Infer specification from reference implementations.
        
        Args:
            function_name: Name of the function
            references: List of reference implementations (C++ code)
            
        Returns:
            Inferred specification
        """
        # Step 1: Parse and execute symbolically
        traces = []
        for ref in references:
            ast = parse_cpp(ref)
            trace = self.symbolic_executor.execute(ast)
            traces.append(trace)
        
        # Step 2: Abstract patterns
        patterns = self.pattern_abstractor.abstract(traces)
        
        # Step 3: Extract conditions
        pre = self.condition_extractor.extract_pre(patterns)
        post = self.condition_extractor.extract_post(patterns)
        inv = self.condition_extractor.extract_inv(patterns)
        
        spec = Specification(
            function_name=function_name,
            preconditions=pre,
            postconditions=post,
            invariants=inv
        )
        
        # Step 4: Validate against references
        self._validate(spec, references)
        
        return spec
    
    def _validate(self, spec: Specification, 
                  references: List[str]) -> None:
        """Validate that spec holds for all references."""
        for ref in references:
            if not spec.validate(ref):
                raise SpecificationError(
                    f"Inferred spec does not hold for reference"
                )
```

### 2.2 Verification Module

#### 2.2.1 클래스 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                      verification/                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │ Verifier            │◄──────│ VCGenerator         │         │
│  ├─────────────────────┤       ├─────────────────────┤         │
│  │ - solver            │       │ + generate()        │         │
│  │ - vcgen             │       │ + compute_wp()      │         │
│  ├─────────────────────┤       └─────────────────────┘         │
│  │ + verify()          │                                        │
│  │ + get_counterex()   │       ┌─────────────────────┐         │
│  └──────────┬──────────┘       │ SMTSolver           │         │
│             │                  ├─────────────────────┤         │
│             │ uses             │ - z3_solver         │         │
│             ▼                  │ + check()           │         │
│  ┌─────────────────────┐       │ + get_model()       │         │
│  │ BoundedModelChecker │       └─────────────────────┘         │
│  ├─────────────────────┤                                        │
│  │ - bound             │                                        │
│  │ + check()           │                                        │
│  └─────────────────────┘                                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ VerificationResult                                       │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ - status: VerificationStatus                             │   │
│  │ - counterexample: Optional[Counterexample]               │   │
│  │ - verified_properties: List[str]                         │   │
│  │ - time_ms: float                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Counterexample                                           │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ - input_values: Dict[str, Any]                          │   │
│  │ - expected_output: Any                                   │   │
│  │ - actual_output: Any                                     │   │
│  │ - violated_condition: str                                │   │
│  │ - trace: List[str]                                       │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ + to_repair_context(): RepairContext                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2.2 핵심 인터페이스

```python
# src/verification/verifier.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import z3

class VerificationStatus(Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class Counterexample:
    """Counterexample from failed verification."""
    input_values: Dict[str, any]
    expected_output: any
    actual_output: any
    violated_condition: str
    trace: List[str]
    
    def to_repair_context(self) -> 'RepairContext':
        """Convert to repair context for CGNR."""
        pass


@dataclass
class VerificationResult:
    """Result of verification."""
    status: VerificationStatus
    counterexample: Optional[Counterexample]
    verified_properties: List[str]
    time_ms: float


class Verifier:
    """Main verification engine."""
    
    def __init__(self, timeout_ms: int = 30000):
        self.vcgen = VCGenerator()
        self.solver = SMTSolver(timeout_ms)
    
    def verify(self, code: str, 
               spec: Specification) -> VerificationResult:
        """
        Verify code against specification.
        
        Args:
            code: C++ code to verify
            spec: Specification to verify against
            
        Returns:
            VerificationResult with status and potential counterexample
        """
        import time
        start = time.time()
        
        # Generate verification conditions
        vc = self.vcgen.generate(code, spec)
        
        # Check with SMT solver
        result, model = self.solver.check(vc)
        
        elapsed = (time.time() - start) * 1000
        
        if result == z3.unsat:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                counterexample=None,
                verified_properties=self._extract_properties(spec),
                time_ms=elapsed
            )
        elif result == z3.sat:
            ce = self._extract_counterexample(model, code, spec)
            return VerificationResult(
                status=VerificationStatus.FAILED,
                counterexample=ce,
                verified_properties=[],
                time_ms=elapsed
            )
        else:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                counterexample=None,
                verified_properties=[],
                time_ms=elapsed
            )


class VCGenerator:
    """Verification Condition Generator."""
    
    def generate(self, code: str, 
                 spec: Specification) -> z3.Formula:
        """Generate verification condition."""
        ast = parse_cpp(code)
        cfg = build_cfg(ast)
        
        # Compute weakest precondition
        wp = self.compute_wp(cfg, spec.postconditions)
        
        # VC = Pre => wp
        pre_formula = z3.And(*[c.to_smt() for c in spec.preconditions])
        vc = z3.Implies(pre_formula, wp)
        
        return vc
    
    def compute_wp(self, cfg: CFG, 
                   post: List[Condition]) -> z3.Formula:
        """Compute weakest precondition through CFG."""
        # Implementation of wp computation
        pass
```

### 2.3 CGNR Module

#### 2.3.1 클래스 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                         repair/                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │ CGNREngine          │◄──────│ FaultLocalizer     │         │
│  ├─────────────────────┤       ├─────────────────────┤         │
│  │ - verifier          │       │ + localize()        │         │
│  │ - repair_model      │       │ + rank_suspicious() │         │
│  │ - max_iterations    │       └─────────────────────┘         │
│  ├─────────────────────┤                                        │
│  │ + repair()          │       ┌─────────────────────┐         │
│  │ + repair_iteration()│◄──────│ RepairModel         │         │
│  └─────────────────────┘       ├─────────────────────┤         │
│                                │ - model             │         │
│                                │ - tokenizer         │         │
│                                │ + generate()        │         │
│                                │ + format_prompt()   │         │
│                                └─────────────────────┘         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ RepairContext                                            │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ - original_code: str                                     │   │
│  │ - counterexample: Counterexample                         │   │
│  │ - fault_location: FaultLocation                          │   │
│  │ - specification: Specification                           │   │
│  │ - repair_history: List[RepairAttempt]                   │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ + to_prompt(): str                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ FaultLocation                                            │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ - line: int                                              │   │
│  │ - statement: str                                         │   │
│  │ - relevant_vars: List[str]                              │   │
│  │ - suspiciousness: float                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.3.2 핵심 인터페이스

```python
# src/repair/cgnr.py

from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class RepairContext:
    """Context for neural repair."""
    original_code: str
    counterexample: Counterexample
    fault_location: 'FaultLocation'
    specification: Specification
    repair_history: List['RepairAttempt']
    
    def to_prompt(self) -> str:
        """Convert to prompt for repair model."""
        return f"""
[REPAIR TASK]
Original code:
{self.original_code}

Counterexample:
- Input: {self.counterexample.input_values}
- Expected: {self.counterexample.expected_output}
- Actual: {self.counterexample.actual_output}

Fault location: line {self.fault_location.line}
  {self.fault_location.statement}

Violated condition: {self.counterexample.violated_condition}

Previous attempts: {len(self.repair_history)}

Generate fixed code:
"""


@dataclass
class RepairAttempt:
    """Record of a repair attempt."""
    code: str
    counterexample: Counterexample
    iteration: int


class CGNREngine:
    """Counterexample-Guided Neural Repair Engine."""
    
    DEFAULT_MAX_ITERATIONS = 5
    
    def __init__(self, 
                 verifier: Verifier,
                 repair_model: 'RepairModel',
                 max_iterations: int = DEFAULT_MAX_ITERATIONS):
        self.verifier = verifier
        self.repair_model = repair_model
        self.fault_localizer = FaultLocalizer()
        self.max_iterations = max_iterations
    
    def repair(self, code: str, 
               spec: Specification) -> Tuple[str, VerificationResult]:
        """
        Main CGNR algorithm.
        
        Args:
            code: Initial code to repair
            spec: Specification to satisfy
            
        Returns:
            Tuple of (repaired_code, verification_result)
        """
        current_code = code
        history: List[RepairAttempt] = []
        
        for iteration in range(self.max_iterations):
            # Step 1: Verify
            result = self.verifier.verify(current_code, spec)
            
            if result.status == VerificationStatus.VERIFIED:
                return current_code, result
            
            if result.counterexample is None:
                break  # Cannot proceed without counterexample
            
            # Step 2: Localize fault
            fault_loc = self.fault_localizer.localize(
                current_code, result.counterexample
            )
            
            # Step 3: Build repair context
            context = RepairContext(
                original_code=current_code,
                counterexample=result.counterexample,
                fault_location=fault_loc,
                specification=spec,
                repair_history=history[-3:]  # Keep last 3 attempts
            )
            
            # Step 4: Generate repair candidates
            candidates = self.repair_model.generate(
                context, num_candidates=5
            )
            
            # Step 5: Select best candidate
            current_code = self._select_best(candidates, spec)
            
            # Record attempt
            history.append(RepairAttempt(
                code=current_code,
                counterexample=result.counterexample,
                iteration=iteration
            ))
        
        # Return best effort
        return current_code, self.verifier.verify(current_code, spec)
    
    def _select_best(self, candidates: List[str], 
                     spec: Specification) -> str:
        """Select best repair candidate."""
        # Try verification on each candidate
        for candidate in candidates:
            result = self.verifier.verify(candidate, spec)
            if result.status == VerificationStatus.VERIFIED:
                return candidate
        
        # If none verified, return first candidate
        return candidates[0] if candidates else ""
```

### 2.4 Hierarchical Verification Module

#### 2.4.1 클래스 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                      hierarchical/                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐                                       │
│  │ HierarchicalVerifier│                                       │
│  ├─────────────────────┤                                       │
│  │ - function_verifier │                                       │
│  │ - module_verifier   │                                       │
│  │ - backend_verifier  │                                       │
│  ├─────────────────────┤                                       │
│  │ + verify_backend()  │                                       │
│  └─────────┬───────────┘                                       │
│            │                                                    │
│   ┌────────┴────────┬────────────────┐                         │
│   ▼                 ▼                ▼                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │FunctionVerify│  │ModuleVerify  │  │BackendVerify │         │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤         │
│  │ + verify()   │  │ + verify()   │  │ + verify()   │         │
│  └──────────────┘  │ + check_     │  │ + check_     │         │
│                    │   internal() │  │   compat()   │         │
│                    └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ InterfaceContract                                        │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ - module_name: str                                       │   │
│  │ - assumptions: List[Condition]                           │   │
│  │ - guarantees: List[Condition]                            │   │
│  │ - dependencies: List[str]                                │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ + check_compatibility(other: InterfaceContract): bool    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 데이터 모델

### 3.1 Specification Data Model

```python
# src/specification/spec_language.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import z3

class ConditionType(Enum):
    EQUALITY = "eq"
    INEQUALITY = "neq"
    LESS_THAN = "lt"
    LESS_EQUAL = "le"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "ge"
    IS_VALID = "valid"
    IS_IN_RANGE = "range"
    IMPLIES = "implies"
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class Expression:
    """Base class for expressions."""
    pass


@dataclass
class Variable(Expression):
    name: str
    type: str = "int"


@dataclass
class Constant(Expression):
    value: Any
    type: str = "int"


@dataclass
class BinaryOp(Expression):
    op: str  # +, -, *, /, %, &, |
    left: Expression
    right: Expression


@dataclass
class FunctionCall(Expression):
    name: str
    args: List[Expression]


@dataclass
class Condition:
    """Formal condition in specification."""
    type: ConditionType
    operands: List[Union[Expression, 'Condition']]
    
    def to_smt(self) -> z3.ExprRef:
        """Convert condition to Z3 formula."""
        if self.type == ConditionType.EQUALITY:
            return self.operands[0].to_smt() == self.operands[1].to_smt()
        elif self.type == ConditionType.IMPLIES:
            return z3.Implies(
                self.operands[0].to_smt(),
                self.operands[1].to_smt()
            )
        # ... other types
    
    def __str__(self) -> str:
        """Human-readable representation."""
        pass


@dataclass
class Specification:
    """Complete specification for a function."""
    function_name: str
    preconditions: List[Condition] = field(default_factory=list)
    postconditions: List[Condition] = field(default_factory=list)
    invariants: List[Condition] = field(default_factory=list)
    
    def to_smt(self) -> z3.ExprRef:
        """Convert entire specification to SMT formula."""
        pre = z3.And(*[c.to_smt() for c in self.preconditions]) \
              if self.preconditions else z3.BoolVal(True)
        post = z3.And(*[c.to_smt() for c in self.postconditions]) \
               if self.postconditions else z3.BoolVal(True)
        return z3.Implies(pre, post)
    
    def to_json(self) -> Dict:
        """Serialize to JSON."""
        return {
            "function_name": self.function_name,
            "preconditions": [str(c) for c in self.preconditions],
            "postconditions": [str(c) for c in self.postconditions],
            "invariants": [str(c) for c in self.invariants]
        }
    
    @classmethod
    def from_json(cls, data: Dict) -> 'Specification':
        """Deserialize from JSON."""
        pass
```

### 3.2 Backend Data Model

```python
# src/integration/backend_model.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

class ModuleType(Enum):
    ASM_PRINTER = "AsmPrinter"
    ISEL_DAG_TO_DAG = "ISelDAGToDAG"
    MC_CODE_EMITTER = "MCCodeEmitter"
    ASM_PARSER = "AsmParser"
    DISASSEMBLER = "Disassembler"
    REGISTER_INFO = "RegisterInfo"
    INSTR_INFO = "InstrInfo"


@dataclass
class Function:
    """A function in the backend."""
    name: str
    module: ModuleType
    code: str
    specification: Optional[Specification] = None
    verified: bool = False
    
    def to_json(self) -> Dict:
        return {
            "name": self.name,
            "module": self.module.value,
            "code": self.code,
            "specification": self.specification.to_json() if self.specification else None,
            "verified": self.verified
        }


@dataclass
class Module:
    """A module in the backend."""
    type: ModuleType
    functions: List[Function] = field(default_factory=list)
    interface_contract: Optional['InterfaceContract'] = None
    verified: bool = False
    
    def add_function(self, func: Function):
        self.functions.append(func)
    
    def all_verified(self) -> bool:
        return all(f.verified for f in self.functions)


@dataclass
class Backend:
    """Complete compiler backend."""
    target: str
    modules: Dict[ModuleType, Module] = field(default_factory=dict)
    
    def add_module(self, module: Module):
        self.modules[module.type] = module
    
    def get_function(self, module: ModuleType, name: str) -> Optional[Function]:
        if module in self.modules:
            for f in self.modules[module].functions:
                if f.name == name:
                    return f
        return None
    
    def verification_summary(self) -> Dict:
        """Get verification status summary."""
        total_functions = sum(len(m.functions) for m in self.modules.values())
        verified_functions = sum(
            sum(1 for f in m.functions if f.verified) 
            for m in self.modules.values()
        )
        return {
            "target": self.target,
            "total_modules": len(self.modules),
            "verified_modules": sum(1 for m in self.modules.values() if m.verified),
            "total_functions": total_functions,
            "verified_functions": verified_functions,
            "verification_rate": verified_functions / total_functions if total_functions > 0 else 0
        }
```

---

## 4. 인터페이스 설계

### 4.1 External Interfaces

#### 4.1.1 CLI Interface

```python
# src/main.py

import click

@click.group()
def cli():
    """VEGA-Verified: Semantically Verified Neural Backend Generation"""
    pass


@cli.command()
@click.argument('target')
@click.option('--references', '-r', multiple=True, help='Reference backend paths')
@click.option('--output', '-o', default='output', help='Output directory')
@click.option('--config', '-c', default='configs/default.yaml', help='Config file')
def generate(target, references, output, config):
    """Generate verified backend for TARGET."""
    from vega_verified import VEGAVerifiedPipeline
    
    pipeline = VEGAVerifiedPipeline.from_config(config)
    backend = pipeline.generate(target, list(references))
    backend.save(output)
    
    click.echo(f"Generated backend for {target}")
    click.echo(f"Verification: {backend.verification_summary()}")


@cli.command()
@click.argument('code_file')
@click.argument('spec_file')
def verify(code_file, spec_file):
    """Verify CODE_FILE against SPEC_FILE."""
    from vega_verified import Verifier, Specification
    
    with open(code_file) as f:
        code = f.read()
    
    spec = Specification.load(spec_file)
    verifier = Verifier()
    result = verifier.verify(code, spec)
    
    click.echo(f"Status: {result.status.value}")
    if result.counterexample:
        click.echo(f"Counterexample: {result.counterexample}")


@cli.command()
@click.argument('function_name')
@click.option('--references', '-r', multiple=True, required=True)
@click.option('--output', '-o', required=True)
def infer_spec(function_name, references, output):
    """Infer specification for FUNCTION_NAME from REFERENCES."""
    from vega_verified import SpecificationInferrer
    
    inferrer = SpecificationInferrer()
    ref_code = [open(r).read() for r in references]
    spec = inferrer.infer(function_name, ref_code)
    spec.save(output)
    
    click.echo(f"Inferred specification saved to {output}")


if __name__ == '__main__':
    cli()
```

#### 4.1.2 Python API

```python
# Public API

from vega_verified import (
    # Core classes
    VEGAVerifiedPipeline,
    Specification,
    Verifier,
    CGNREngine,
    HierarchicalVerifier,
    
    # Data models
    Backend,
    Module,
    Function,
    VerificationResult,
    Counterexample,
    
    # Utilities
    parse_cpp,
    build_cfg,
)

# Example usage
pipeline = VEGAVerifiedPipeline(config)
backend = pipeline.generate("riscv", references)

# Manual verification
verifier = Verifier()
result = verifier.verify(code, spec)

# Manual repair
cgnr = CGNREngine(verifier, repair_model)
fixed_code, result = cgnr.repair(code, spec)
```

---

## 5. 구현 로드맵

### 5.1 Phase 1: Core Infrastructure (Week 1-4)

**목표**: 기본 검증 인프라 구축

**Tasks**:
- [ ] C++ 파서 통합 (tree-sitter 또는 libclang)
- [ ] CFG 빌더 구현
- [ ] Z3 인터페이스 구현
- [ ] 기본 VC 생성기 구현
- [ ] 간단한 함수에 대한 검증 테스트

**Deliverable**: 수동 spec으로 간단한 함수 검증 가능

### 5.2 Phase 2: Specification Inference (Week 5-8)

**목표**: 자동 specification 추론

**Tasks**:
- [ ] Symbolic execution 엔진 구현
- [ ] Pattern abstraction 알고리즘 구현
- [ ] Condition extraction 로직 구현
- [ ] Reference backend 파서 구현
- [ ] Spec 검증 로직 구현

**Deliverable**: Reference에서 자동으로 spec 추론 가능

### 5.3 Phase 3: CGNR (Week 9-12)

**목표**: 자동 수리 기능 구현

**Tasks**:
- [ ] Fault localization 알고리즘 구현
- [ ] Repair context 구성 로직 구현
- [ ] Repair model 훈련 데이터 준비
- [ ] Repair model fine-tuning
- [ ] CGNR 메인 루프 구현

**Deliverable**: 검증 실패 시 자동 수리 가능

### 5.4 Phase 4: Hierarchical Verification (Week 13-16)

**목표**: 계층적 검증 완성

**Tasks**:
- [ ] Interface contract 정의 언어 구현
- [ ] Module-level 검증 구현
- [ ] Cross-module compatibility check 구현
- [ ] Backend-level 검증 구현
- [ ] Full pipeline 통합

**Deliverable**: 완전한 백엔드 계층적 검증 가능

### 5.5 Phase 5: Evaluation & Optimization (Week 17-20)

**목표**: 평가 및 최적화

**Tasks**:
- [ ] VEGA 벤치마크로 평가
- [ ] 성능 병목 식별 및 최적화
- [ ] 문서화 완성
- [ ] 오픈소스 공개 준비

---

## 6. 테스트 계획

### 6.1 Unit Tests

```python
# tests/unit/test_spec_inference.py

import pytest
from vega_verified.specification import SpecificationInferrer, Specification

class TestSpecificationInference:
    
    def test_infer_simple_function(self):
        """Test inference on simple function."""
        inferrer = SpecificationInferrer()
        
        # Simple reference implementations
        refs = [
            "int abs(int x) { return x < 0 ? -x : x; }",
            "int abs(int x) { if (x < 0) return -x; return x; }"
        ]
        
        spec = inferrer.infer("abs", refs)
        
        assert len(spec.postconditions) > 0
        assert any("result >= 0" in str(p) for p in spec.postconditions)
    
    def test_infer_reloc_type(self):
        """Test inference on getRelocType function."""
        inferrer = SpecificationInferrer()
        
        # Load ARM and MIPS implementations
        arm_impl = load_fixture("arm_getRelocType.cpp")
        mips_impl = load_fixture("mips_getRelocType.cpp")
        
        spec = inferrer.infer("getRelocType", [arm_impl, mips_impl])
        
        assert "Fixup.isValid()" in str(spec.preconditions)
        assert "result >= 0" in str(spec.postconditions)
```

### 6.2 Integration Tests

```python
# tests/integration/test_cgnr.py

import pytest
from vega_verified import CGNREngine, Verifier, Specification

class TestCGNR:
    
    def test_repair_simple_bug(self):
        """Test CGNR can repair a simple bug."""
        verifier = Verifier()
        cgnr = CGNREngine(verifier, mock_repair_model())
        
        buggy_code = """
        int abs(int x) {
            return x;  // Bug: should return -x when x < 0
        }
        """
        
        spec = Specification(
            function_name="abs",
            preconditions=[],
            postconditions=[Condition.parse("result >= 0")]
        )
        
        fixed_code, result = cgnr.repair(buggy_code, spec)
        
        assert result.status == VerificationStatus.VERIFIED
    
    def test_repair_max_iterations(self):
        """Test CGNR respects max iterations."""
        verifier = Verifier()
        cgnr = CGNREngine(verifier, failing_repair_model(), max_iterations=3)
        
        buggy_code = "int f() { return -1; }"
        spec = Specification(
            function_name="f",
            postconditions=[Condition.parse("result >= 0")]
        )
        
        fixed_code, result = cgnr.repair(buggy_code, spec)
        
        # Should fail after max iterations
        assert result.status == VerificationStatus.FAILED
```

### 6.3 Benchmark Tests

```python
# tests/benchmarks/test_vega_benchmark.py

import pytest
from vega_verified import VEGAVerifiedPipeline

class TestVEGABenchmark:
    
    @pytest.mark.benchmark
    def test_riscv_backend(self):
        """Benchmark on RISC-V backend."""
        pipeline = VEGAVerifiedPipeline.from_config("configs/riscv.yaml")
        backend = pipeline.generate("riscv", REFERENCE_BACKENDS)
        
        summary = backend.verification_summary()
        
        # Expected: 85-90% verification rate
        assert summary["verification_rate"] >= 0.85
    
    @pytest.mark.benchmark  
    def test_generation_time(self):
        """Test generation time is reasonable."""
        import time
        
        pipeline = VEGAVerifiedPipeline.from_config("configs/default.yaml")
        
        start = time.time()
        backend = pipeline.generate("riscv", REFERENCE_BACKENDS)
        elapsed = time.time() - start
        
        # Expected: < 3 hours total
        assert elapsed < 3 * 3600
```

---

## 부록: 설정 파일 예시

### configs/default.yaml

```yaml
# VEGA-Verified Configuration

# Specification Inference
specification:
  symbolic_execution:
    max_depth: 100
    timeout_ms: 5000
  pattern_abstraction:
    min_similarity: 0.7
  condition_extraction:
    include_null_checks: true
    include_bounds_checks: true

# Verification
verification:
  solver: z3
  timeout_ms: 30000
  bmc_bound: 10

# CGNR
repair:
  max_iterations: 5
  model_path: models/repair_model
  beam_size: 5
  temperature: 0.7

# Hierarchical Verification
hierarchical:
  levels: [function, module, backend]
  parallel_verification: true
  max_workers: 4

# Logging
logging:
  level: INFO
  file: logs/vega_verified.log
```

### configs/targets/riscv.yaml

```yaml
# RISC-V Target Configuration

target:
  name: RISCV
  triple: riscv64-unknown-linux-gnu

references:
  - backends/ARM
  - backends/MIPS
  - backends/X86

modules:
  - AsmPrinter
  - ISelDAGToDAG
  - MCCodeEmitter
  - AsmParser
  - Disassembler
  - RegisterInfo
  - InstrInfo

interface_contracts:
  MCCodeEmitter:
    assumptions:
      - validMachineInstr(MI)
      - streamerReady(Streamer)
    guarantees:
      - emittedBytesCorrect(MI)
      - relocationsRegistered()
    dependencies:
      - RegisterInfo
      - InstrInfo
```
