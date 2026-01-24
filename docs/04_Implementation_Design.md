# VERA 프레임워크 구현 설계서

> **문서 버전**: 2.0 (2026-01-24)  
> **관점**: ISA 스펙 기반 백엔드 자동생성 연구자  
> **참조 연구**: VEGA, ACT, Isaria, OpenVADL, VeGen, Hydride

---

## 목차

1. [설계 개요](#1-설계-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [모듈 상세 설계](#3-모듈-상세-설계)
4. [데이터 모델](#4-데이터-모델)
5. [핵심 알고리즘 구현](#5-핵심-알고리즘-구현)
6. [인터페이스 정의](#6-인터페이스-정의)
7. [구현 로드맵](#7-구현-로드맵)
8. [테스트 전략](#8-테스트-전략)
9. [배포 및 통합](#9-배포-및-통합)
10. [부록: 설정 및 참조](#10-부록-설정-및-참조)

---

## 1. 설계 개요

### 1.1 설계 목표

VERA (Verified and Extensible Retargetable Architecture) 프레임워크는 다음 목표를 달성하도록 설계됩니다:

| 목표 | 설명 | 측정 기준 |
|------|------|-----------|
| **Verified Correctness** | 형식 검증을 통한 semantic correctness 보장 | >80% 함수 검증 |
| **High Accuracy** | VEGA 대비 높은 생성 정확도 | >85% function accuracy |
| **Automated Repair** | CGNR을 통한 자동 오류 수정 | >90% 수정 성공률 |
| **Scalability** | 새로운 타겟으로의 확장 용이성 | <1일 새 타겟 적응 |
| **Modularity** | 계층적 모듈 구조로 유지보수성 확보 | Independent modules |

### 1.2 설계 원칙

1. **Hydride-like Compositional Design**: 합성 기반 접근법의 모듈성 차용
2. **VEGA-compatible Pipeline**: 기존 VEGA 인프라와의 호환성 유지
3. **Incremental Verification**: 점진적 검증으로 확장성 확보
4. **LLM-assisted Specification**: ISA 스펙 분석에 LLM 활용

### 1.3 관련 연구 설계 참조

| 연구 | 참조 설계 요소 | 적용 방식 |
|------|----------------|-----------|
| **VEGA** | Template extraction, Neural generation | 생성 엔진 기반 |
| **Hydride** | CEGIS loop, Formal verification | CGNR 설계 |
| **VeGen** | Lane-based decomposition | Vector handling |
| **Alive2** | SMT-based verification | 검증 조건 생성 |

---

## 2. 시스템 아키텍처

### 2.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VERA Framework Architecture                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         INPUT LAYER                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │   │
│  │  │ Reference       │  │ Target ISA      │  │ Configuration           │  │   │
│  │  │ Backends        │  │ Specification   │  │ (YAML/JSON)             │  │   │
│  │  │ (LLVM sources)  │  │ (.td files)     │  │                         │  │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘  │   │
│  └───────────┼────────────────────┼────────────────────────┼────────────────┘   │
│              │                    │                        │                    │
│              ▼                    ▼                        ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    SPECIFICATION INFERENCE MODULE                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │ Parser      │  │ Aligner     │  │ Extractor   │  │ Validator   │    │   │
│  │  │ (Clang AST) │→ │ (GumTree)   │→ │ (Spec)      │→ │ (Spec)      │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └────────────────────────────────────────┬────────────────────────────────┘   │
│                                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    NEURAL GENERATION MODULE                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │ Template    │  │ Feature     │  │ Transformer │  │ Confidence  │    │   │
│  │  │ Extractor   │→ │ Encoder     │→ │ Model       │→ │ Estimator   │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └────────────────────────────────────────┬────────────────────────────────┘   │
│                                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    VERIFICATION MODULE                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │ VC          │  │ SMT         │  │ BMC         │  │ Result      │    │   │
│  │  │ Generator   │→ │ Solver      │→ │ Engine      │→ │ Analyzer    │    │   │
│  │  │             │  │ (Z3)        │  │             │  │             │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────┬───────────────────────────────────────┬───────────────┘   │
│                    │ VERIFIED                              │ COUNTEREXAMPLE    │
│                    ▼                                       ▼                   │
│  ┌─────────────────────────────┐    ┌──────────────────────────────────────┐   │
│  │    OUTPUT: Verified Code    │    │         CGNR MODULE                  │   │
│  │    + Certificates           │    │  ┌─────────┐  ┌─────────┐  ┌──────┐ │   │
│  └─────────────────────────────┘    │  │ Fault   │→ │ Repair  │→ │ Re-  │ │   │
│                                     │  │ Locator │  │ Model   │  │verify│ │   │
│                                     │  └─────────┘  └─────────┘  └──────┘ │   │
│                                     └──────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    HIERARCHICAL VERIFICATION MODULE                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │   │
│  │  │ L1: Function│→ │ L2: Module  │→ │ L3: Backend │                      │   │
│  │  │ Contracts   │  │ Composition │  │ Integration │                      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 디렉토리 구조

```
vera-framework/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # 설정 관리
│   │   ├── pipeline.py            # 메인 파이프라인
│   │   └── types.py               # 공통 타입 정의
│   │
│   ├── spec_inference/
│   │   ├── __init__.py
│   │   ├── parser.py              # AST 파싱 (libclang)
│   │   ├── aligner.py             # GumTree 정렬
│   │   ├── extractor.py           # Spec 추출
│   │   └── validator.py           # Spec 검증
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── template.py            # 템플릿 추출
│   │   ├── feature.py             # Feature encoding
│   │   ├── model.py               # Transformer 모델
│   │   └── confidence.py          # 신뢰도 추정
│   │
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── vc_generator.py        # VC 생성
│   │   ├── smt_solver.py          # SMT 인터페이스 (Z3)
│   │   ├── bmc.py                 # Bounded model checking
│   │   └── result.py              # 결과 분석
│   │
│   ├── repair/
│   │   ├── __init__.py
│   │   ├── fault_locator.py       # 결함 위치 파악
│   │   ├── repair_model.py        # 수정 모델
│   │   └── cgnr.py                # CGNR 메인 루프
│   │
│   └── hierarchical/
│       ├── __init__.py
│       ├── function_level.py      # L1 검증
│       ├── module_level.py        # L2 검증
│       ├── backend_level.py       # L3 검증
│       └── contracts.py           # 인터페이스 계약
│
├── models/
│   ├── generation/                # 생성 모델 체크포인트
│   │   └── unixcoder_finetuned/
│   └── repair/                    # 수정 모델 체크포인트
│       └── cgnr_model/
│
├── specs/
│   ├── inferred/                  # 추론된 명세
│   ├── templates/                 # 명세 템플릿
│   └── validated/                 # 검증된 명세
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmark/
│
├── configs/
│   ├── default.yaml
│   └── targets/
│       ├── riscv.yaml
│       ├── arm.yaml
│       └── mips.yaml
│
├── scripts/
│   ├── train_generation.py
│   ├── train_repair.py
│   └── evaluate.py
│
├── docs/
│   └── api/
│
├── requirements.txt
├── setup.py
└── README.md
```

### 2.3 의존성

```yaml
# requirements.txt
# Core
python>=3.9
torch>=2.0
transformers>=4.30

# Parsing & AST
libclang>=16.0
tree-sitter>=0.20

# Verification
z3-solver>=4.12
pysmt>=0.9

# Utilities
pyyaml>=6.0
tqdm>=4.65
numpy>=1.24
pandas>=2.0

# Testing
pytest>=7.0
pytest-cov>=4.0
```

---

## 3. 모듈 상세 설계

### 3.1 Specification Inference Module

#### 3.1.1 Parser (parser.py)

```python
from typing import List, Optional
from dataclasses import dataclass
import clang.cindex as clang

@dataclass
class ASTNode:
    kind: str
    spelling: str
    location: tuple  # (file, line, column)
    children: List['ASTNode']
    tokens: List[str]

class BackendParser:
    """
    Reference backend C++ 소스를 파싱하여 AST 생성
    
    참조: VEGA의 GumTree 기반 파싱 접근법
    """
    
    def __init__(self, clang_path: Optional[str] = None):
        if clang_path:
            clang.Config.set_library_file(clang_path)
        self.index = clang.Index.create()
    
    def parse_file(self, filepath: str) -> ASTNode:
        """
        C++ 파일을 파싱하여 AST 반환
        
        Args:
            filepath: 소스 파일 경로
            
        Returns:
            ASTNode: 루트 AST 노드
        """
        tu = self.index.parse(filepath, args=['-std=c++17'])
        return self._convert_cursor(tu.cursor)
    
    def _convert_cursor(self, cursor: clang.Cursor) -> ASTNode:
        """Clang cursor를 ASTNode로 변환"""
        children = [self._convert_cursor(c) for c in cursor.get_children()]
        tokens = [t.spelling for t in cursor.get_tokens()]
        
        return ASTNode(
            kind=cursor.kind.name,
            spelling=cursor.spelling,
            location=(
                cursor.location.file.name if cursor.location.file else "",
                cursor.location.line,
                cursor.location.column
            ),
            children=children,
            tokens=tokens
        )
    
    def extract_functions(self, ast: ASTNode) -> List[ASTNode]:
        """AST에서 함수 정의 추출"""
        functions = []
        
        def visit(node: ASTNode):
            if node.kind == 'FUNCTION_DECL':
                functions.append(node)
            for child in node.children:
                visit(child)
        
        visit(ast)
        return functions
```

#### 3.1.2 Spec Extractor (extractor.py)

```python
from dataclasses import dataclass
from typing import List, Set, Dict
from enum import Enum

class ConditionType(Enum):
    PRECONDITION = "pre"
    POSTCONDITION = "post"
    INVARIANT = "inv"

@dataclass
class Condition:
    type: ConditionType
    expression: str
    source_location: str
    is_target_independent: bool

@dataclass
class Specification:
    function_name: str
    preconditions: List[Condition]
    postconditions: List[Condition]
    invariants: List[Condition]
    
    def to_smt(self) -> str:
        """SMT-LIB 형식으로 변환"""
        smt_parts = []
        
        # Preconditions as assertions
        for pre in self.preconditions:
            smt_parts.append(f"(assert {self._expr_to_smt(pre.expression)})")
        
        return "\n".join(smt_parts)
    
    def _expr_to_smt(self, expr: str) -> str:
        """표현식을 SMT-LIB 형식으로 변환"""
        # Simplified conversion - actual implementation would be more complex
        return expr.replace("&&", "and").replace("||", "or")

class SpecificationExtractor:
    """
    Reference 구현들에서 자동으로 명세 추론
    
    알고리즘:
    1. 다중 백엔드 함수 정렬 (GumTree)
    2. Target-independent 패턴 식별
    3. Pre/Post condition 추출
    4. Invariant 추론
    
    참조: Hydride의 specification inference 접근법
    """
    
    def __init__(self, aligner):
        self.aligner = aligner
    
    def infer_specification(
        self, 
        func_name: str, 
        implementations: Dict[str, ASTNode]
    ) -> Specification:
        """
        여러 백엔드 구현에서 명세 추론
        
        Args:
            func_name: 함수 이름
            implementations: {target_name: AST} 딕셔너리
            
        Returns:
            Specification: 추론된 명세
        """
        # Step 1: 구현들 정렬
        aligned = self.aligner.align(list(implementations.values()))
        
        # Step 2: Preconditions 추출
        preconditions = self._extract_preconditions(aligned)
        
        # Step 3: Postconditions 추출
        postconditions = self._extract_postconditions(aligned)
        
        # Step 4: Invariants 추론
        invariants = self._extract_invariants(aligned)
        
        return Specification(
            function_name=func_name,
            preconditions=preconditions,
            postconditions=postconditions,
            invariants=invariants
        )
    
    def _extract_preconditions(self, aligned) -> List[Condition]:
        """
        정렬된 구현에서 preconditions 추출
        
        패턴:
        - Null checks: if (ptr == nullptr) return error;
        - Bounds checks: assert(index < size);
        - Type checks: if (!isa<Type>(value)) ...
        """
        preconditions = []
        
        for group in aligned.groups:
            if self._is_null_check(group):
                preconditions.append(Condition(
                    type=ConditionType.PRECONDITION,
                    expression=self._extract_null_check_expr(group),
                    source_location=group.location,
                    is_target_independent=group.is_ti
                ))
            elif self._is_bounds_check(group):
                preconditions.append(Condition(
                    type=ConditionType.PRECONDITION,
                    expression=self._extract_bounds_check_expr(group),
                    source_location=group.location,
                    is_target_independent=group.is_ti
                ))
        
        return preconditions
    
    def _extract_postconditions(self, aligned) -> List[Condition]:
        """
        정렬된 구현에서 postconditions 추출
        
        패턴:
        - Return value constraints
        - Output parameter constraints
        """
        postconditions = []
        
        for group in aligned.groups:
            if self._is_return_statement(group):
                postconditions.append(Condition(
                    type=ConditionType.POSTCONDITION,
                    expression=self._extract_return_constraint(group),
                    source_location=group.location,
                    is_target_independent=group.is_ti
                ))
        
        return postconditions
    
    def _extract_invariants(self, aligned) -> List[Condition]:
        """
        정렬된 구현에서 invariants 추론
        
        패턴:
        - Loop invariants
        - Switch coverage invariants
        - Data structure invariants
        """
        invariants = []
        
        for group in aligned.groups:
            if self._is_switch_statement(group):
                invariants.append(Condition(
                    type=ConditionType.INVARIANT,
                    expression=self._extract_switch_coverage(group),
                    source_location=group.location,
                    is_target_independent=True  # Switch structure is TI
                ))
        
        return invariants
    
    # Helper methods (simplified)
    def _is_null_check(self, group) -> bool:
        return "nullptr" in str(group) or "NULL" in str(group)
    
    def _is_bounds_check(self, group) -> bool:
        return "assert" in str(group) and ("<" in str(group) or ">" in str(group))
    
    def _is_return_statement(self, group) -> bool:
        return group.kind == "RETURN_STMT"
    
    def _is_switch_statement(self, group) -> bool:
        return group.kind == "SWITCH_STMT"
```

### 3.2 Verification Module

#### 3.2.1 VC Generator (vc_generator.py)

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

@dataclass
class VerificationCondition:
    """검증 조건 (Verification Condition)"""
    precondition: str      # SMT-LIB format
    program: str           # 프로그램 representation
    postcondition: str     # SMT-LIB format
    
    def to_smt_query(self) -> str:
        """
        SMT 쿼리 생성: Pre ∧ Program ⟹ Post 의 validity 검사
        
        Returns:
            SMT-LIB 형식의 쿼리 문자열
        """
        return f"""
; Verification Condition
; Check: Pre ∧ Program ⟹ Post

; Declarations
(declare-sort Value)
(declare-fun eval (Value) Value)

; Precondition
{self.precondition}

; Program encoding
{self.program}

; Negated postcondition (for SAT check)
(assert (not {self.postcondition}))

; Check satisfiability
(check-sat)
(get-model)
"""

class VCGenerator:
    """
    Verification Condition 생성기
    
    Hoare Logic 기반:
    {Pre} S {Post}  ⟺  Pre ⟹ wp(S, Post)
    
    참조: Alive2의 VC 생성 방식
    """
    
    def generate_vc(
        self, 
        code: str, 
        spec: 'Specification'
    ) -> VerificationCondition:
        """
        코드와 명세에서 VC 생성
        
        Args:
            code: 생성된 C++ 코드
            spec: 추론된 명세
            
        Returns:
            VerificationCondition: 생성된 VC
        """
        # Parse code to IR
        ir = self._parse_to_ir(code)
        
        # Generate precondition
        pre_smt = self._encode_preconditions(spec.preconditions)
        
        # Encode program
        prog_smt = self._encode_program(ir)
        
        # Generate postcondition
        post_smt = self._encode_postconditions(spec.postconditions)
        
        return VerificationCondition(
            precondition=pre_smt,
            program=prog_smt,
            postcondition=post_smt
        )
    
    def _encode_preconditions(self, preconditions: List['Condition']) -> str:
        """Preconditions을 SMT-LIB로 인코딩"""
        parts = []
        for pre in preconditions:
            parts.append(f"(assert {pre.expression})")
        return "\n".join(parts)
    
    def _encode_program(self, ir) -> str:
        """
        프로그램 IR을 SMT로 인코딩
        
        지원 구문:
        - Assignment
        - Conditional (if-else)
        - Switch
        - Return
        """
        smt_parts = []
        
        for stmt in ir.statements:
            if stmt.kind == 'ASSIGN':
                smt_parts.append(self._encode_assignment(stmt))
            elif stmt.kind == 'IF':
                smt_parts.append(self._encode_conditional(stmt))
            elif stmt.kind == 'SWITCH':
                smt_parts.append(self._encode_switch(stmt))
            elif stmt.kind == 'RETURN':
                smt_parts.append(self._encode_return(stmt))
        
        return "\n".join(smt_parts)
    
    def _encode_switch(self, switch_stmt) -> str:
        """
        Switch문 인코딩
        
        Example:
        switch(x) {
            case A: return R_A;
            case B: return R_B;
            default: return R_DEFAULT;
        }
        
        SMT:
        (ite (= x A) R_A
          (ite (= x B) R_B R_DEFAULT))
        """
        cases = switch_stmt.cases
        default = switch_stmt.default
        
        # Build nested ITE
        result = default.value if default else "undefined"
        
        for case in reversed(cases):
            result = f"(ite (= {switch_stmt.selector} {case.value}) {case.result} {result})"
        
        return f"(define-fun switch_result () Value {result})"
    
    def _encode_postconditions(self, postconditions: List['Condition']) -> str:
        """Postconditions을 SMT-LIB로 인코딩"""
        if not postconditions:
            return "true"
        
        parts = [f"({post.expression})" for post in postconditions]
        return f"(and {' '.join(parts)})"
```

#### 3.2.2 SMT Solver Interface (smt_solver.py)

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
import z3

class VerificationResult(Enum):
    VERIFIED = "verified"
    COUNTEREXAMPLE = "counterexample"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

@dataclass
class Counterexample:
    """반례 (Counterexample) 정보"""
    input_values: Dict[str, Any]
    expected_output: Any
    actual_output: Any
    trace: str

@dataclass
class SMTResult:
    """SMT solver 결과"""
    status: VerificationResult
    counterexample: Optional[Counterexample]
    time_ms: float
    solver_stats: Dict[str, Any]

class SMTSolver:
    """
    Z3 기반 SMT solver 인터페이스
    
    참조: Alive2, Hydride의 SMT 기반 검증
    """
    
    def __init__(self, timeout_ms: int = 30000):
        self.timeout_ms = timeout_ms
        self.solver = z3.Solver()
        self.solver.set("timeout", timeout_ms)
    
    def check(self, vc: 'VerificationCondition') -> SMTResult:
        """
        VC의 validity 검사
        
        전략:
        1. VC를 Z3 표현식으로 변환
        2. Negated postcondition의 satisfiability 검사
        3. SAT → counterexample, UNSAT → verified
        
        Args:
            vc: 검증 조건
            
        Returns:
            SMTResult: 검증 결과
        """
        import time
        start_time = time.time()
        
        # Parse and add VC to solver
        self.solver.reset()
        z3_expr = self._parse_smt_lib(vc.to_smt_query())
        self.solver.add(z3_expr)
        
        # Check
        result = self.solver.check()
        elapsed_ms = (time.time() - start_time) * 1000
        
        if result == z3.sat:
            # Found counterexample
            model = self.solver.model()
            cex = self._extract_counterexample(model)
            return SMTResult(
                status=VerificationResult.COUNTEREXAMPLE,
                counterexample=cex,
                time_ms=elapsed_ms,
                solver_stats=self._get_stats()
            )
        elif result == z3.unsat:
            # Verified
            return SMTResult(
                status=VerificationResult.VERIFIED,
                counterexample=None,
                time_ms=elapsed_ms,
                solver_stats=self._get_stats()
            )
        else:
            # Timeout or unknown
            return SMTResult(
                status=VerificationResult.TIMEOUT if elapsed_ms >= self.timeout_ms 
                       else VerificationResult.UNKNOWN,
                counterexample=None,
                time_ms=elapsed_ms,
                solver_stats=self._get_stats()
            )
    
    def _extract_counterexample(self, model: z3.ModelRef) -> Counterexample:
        """Z3 model에서 counterexample 추출"""
        input_values = {}
        
        for decl in model.decls():
            name = decl.name()
            value = model[decl]
            input_values[name] = self._z3_to_python(value)
        
        return Counterexample(
            input_values=input_values,
            expected_output=None,  # Will be computed
            actual_output=None,
            trace=str(model)
        )
    
    def _z3_to_python(self, z3_val) -> Any:
        """Z3 값을 Python 값으로 변환"""
        if z3.is_int(z3_val):
            return z3_val.as_long()
        elif z3.is_bool(z3_val):
            return bool(z3_val)
        else:
            return str(z3_val)
    
    def _get_stats(self) -> Dict[str, Any]:
        """Solver 통계 반환"""
        stats = self.solver.statistics()
        return {str(k): stats.get_key_value(k) for k in stats.keys()}
```

### 3.3 CGNR Module

#### 3.3.1 CGNR Main Loop (cgnr.py)

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class RepairContext:
    """수정 컨텍스트"""
    code: str
    fault_location: Tuple[int, int]  # (start_line, end_line)
    counterexample: 'Counterexample'
    spec: 'Specification'
    history: List[Tuple[str, str]]  # (old_code, new_code) pairs

@dataclass
class CGNRResult:
    """CGNR 결과"""
    success: bool
    final_code: str
    iterations: int
    repair_history: List[Tuple[str, str]]
    verification_status: 'VerificationResult'

class CGNR:
    """
    Counterexample-Guided Neural Repair
    
    알고리즘:
    1. 초기 코드 검증
    2. 실패 시 counterexample 추출
    3. Fault localization
    4. Neural repair model로 수정 후보 생성
    5. 후보 검증, 성공 시 종료
    6. 실패 시 2로 반복
    
    참조: Hydride의 CEGIS, VEGA의 error patterns
    """
    
    def __init__(
        self,
        verifier: 'SMTSolver',
        repair_model: 'RepairModel',
        fault_locator: 'FaultLocator',
        max_iterations: int = 10
    ):
        self.verifier = verifier
        self.repair_model = repair_model
        self.fault_locator = fault_locator
        self.max_iterations = max_iterations
    
    def repair(
        self, 
        initial_code: str, 
        spec: 'Specification'
    ) -> CGNRResult:
        """
        CGNR 메인 루프
        
        Args:
            initial_code: VEGA가 생성한 초기 코드
            spec: 추론된 명세
            
        Returns:
            CGNRResult: 수정 결과
        """
        code = initial_code
        history = []
        
        for iteration in range(self.max_iterations):
            logger.info(f"CGNR iteration {iteration + 1}/{self.max_iterations}")
            
            # Step 1: Generate VC and verify
            vc = self._generate_vc(code, spec)
            result = self.verifier.check(vc)
            
            # Step 2: Check result
            if result.status == VerificationResult.VERIFIED:
                logger.info("Verification successful!")
                return CGNRResult(
                    success=True,
                    final_code=code,
                    iterations=iteration + 1,
                    repair_history=history,
                    verification_status=result.status
                )
            
            if result.status == VerificationResult.TIMEOUT:
                logger.warning("Verification timeout, trying simplification...")
                # Try with simplified VC or BMC
                continue
            
            # Step 3: Extract counterexample and localize fault
            cex = result.counterexample
            fault_loc = self.fault_locator.localize(code, cex, spec)
            
            logger.info(f"Fault localized to lines {fault_loc}")
            
            # Step 4: Generate repair candidates
            context = RepairContext(
                code=code,
                fault_location=fault_loc,
                counterexample=cex,
                spec=spec,
                history=history
            )
            
            candidates = self.repair_model.generate_candidates(context, k=5)
            
            # Step 5: Try candidates
            for candidate in candidates:
                repaired_code = self._apply_repair(code, fault_loc, candidate)
                
                # Quick check before full verification
                if self._quick_check(repaired_code, cex):
                    logger.info(f"Candidate passed quick check, full verify...")
                    
                    # Full verification
                    repair_vc = self._generate_vc(repaired_code, spec)
                    repair_result = self.verifier.check(repair_vc)
                    
                    if repair_result.status == VerificationResult.VERIFIED:
                        history.append((code, repaired_code))
                        return CGNRResult(
                            success=True,
                            final_code=repaired_code,
                            iterations=iteration + 1,
                            repair_history=history,
                            verification_status=repair_result.status
                        )
                    
                    # Candidate passed quick check but failed full verify
                    # Still update code to make progress
                    if repair_result.status == VerificationResult.COUNTEREXAMPLE:
                        # New counterexample - we made progress
                        if repair_result.counterexample != cex:
                            history.append((code, repaired_code))
                            code = repaired_code
                            break
            
            logger.warning(f"No successful repair in iteration {iteration + 1}")
        
        # Max iterations reached
        return CGNRResult(
            success=False,
            final_code=code,
            iterations=self.max_iterations,
            repair_history=history,
            verification_status=VerificationResult.UNKNOWN
        )
    
    def _generate_vc(self, code: str, spec: 'Specification') -> 'VerificationCondition':
        """VC 생성"""
        from .vc_generator import VCGenerator
        generator = VCGenerator()
        return generator.generate_vc(code, spec)
    
    def _apply_repair(
        self, 
        code: str, 
        fault_loc: Tuple[int, int], 
        repair: str
    ) -> str:
        """수정 적용"""
        lines = code.split('\n')
        start, end = fault_loc
        
        # Replace faulty lines with repair
        repaired_lines = lines[:start] + [repair] + lines[end+1:]
        return '\n'.join(repaired_lines)
    
    def _quick_check(self, code: str, cex: 'Counterexample') -> bool:
        """
        빠른 검사: counterexample에 대해 올바른지 확인
        
        실제 SMT 검증 전 간단한 concrete execution으로 필터링
        """
        # Simplified: would actually execute code with cex inputs
        return True  # Placeholder
```

#### 3.3.2 Fault Locator (fault_locator.py)

```python
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class SuspiciousLocation:
    """의심 위치"""
    start_line: int
    end_line: int
    suspiciousness: float  # 0.0 ~ 1.0
    reason: str

class FaultLocator:
    """
    결함 위치 파악 (Fault Localization)
    
    기법:
    1. Counterexample-based: cex가 영향을 주는 statement 추적
    2. Spec-based: specification 위반 지점 식별
    3. Pattern-based: VEGA error patterns 기반 추론
    
    참조: VEGA의 error type analysis
    """
    
    def __init__(self):
        # VEGA error patterns (from paper)
        self.error_patterns = {
            'Err-Pred': self._localize_prediction_error,
            'Err-Conf': self._localize_confidence_error,
            'Err-Def': self._localize_definition_error,
        }
    
    def localize(
        self, 
        code: str, 
        cex: 'Counterexample', 
        spec: 'Specification'
    ) -> Tuple[int, int]:
        """
        결함 위치 파악
        
        Args:
            code: 결함이 있는 코드
            cex: Counterexample
            spec: 명세
            
        Returns:
            (start_line, end_line): 의심 위치
        """
        suspicious = []
        
        # Method 1: Counterexample trace analysis
        trace_based = self._analyze_cex_trace(code, cex)
        suspicious.extend(trace_based)
        
        # Method 2: Spec violation analysis
        spec_based = self._analyze_spec_violation(code, cex, spec)
        suspicious.extend(spec_based)
        
        # Method 3: Pattern-based analysis
        pattern_based = self._analyze_error_patterns(code)
        suspicious.extend(pattern_based)
        
        # Combine and rank
        ranked = self._rank_suspicious(suspicious)
        
        if ranked:
            top = ranked[0]
            return (top.start_line, top.end_line)
        else:
            # Fallback: return first non-trivial line
            return (1, 1)
    
    def _analyze_cex_trace(
        self, 
        code: str, 
        cex: 'Counterexample'
    ) -> List[SuspiciousLocation]:
        """Counterexample trace에서 의심 위치 분석"""
        suspicious = []
        
        # Parse trace to find divergence point
        trace_lines = cex.trace.split('\n')
        
        for i, line in enumerate(code.split('\n'), 1):
            # Check if this line is in the execution path
            if self._line_in_trace(line, trace_lines):
                # Check if this is a decision point
                if self._is_decision_point(line):
                    suspicious.append(SuspiciousLocation(
                        start_line=i,
                        end_line=i,
                        suspiciousness=0.8,
                        reason="Decision point in counterexample trace"
                    ))
        
        return suspicious
    
    def _analyze_spec_violation(
        self, 
        code: str, 
        cex: 'Counterexample',
        spec: 'Specification'
    ) -> List[SuspiciousLocation]:
        """명세 위반 지점 분석"""
        suspicious = []
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check return statements against postconditions
            if 'return' in line:
                for post in spec.postconditions:
                    if not self._satisfies_postcondition(line, post, cex):
                        suspicious.append(SuspiciousLocation(
                            start_line=i,
                            end_line=i,
                            suspiciousness=0.9,
                            reason=f"Return violates postcondition: {post.expression}"
                        ))
        
        return suspicious
    
    def _analyze_error_patterns(self, code: str) -> List[SuspiciousLocation]:
        """VEGA error patterns 기반 분석"""
        suspicious = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Pattern: Target-specific identifiers (Err-Pred)
            if self._has_target_specific_pattern(line):
                suspicious.append(SuspiciousLocation(
                    start_line=i,
                    end_line=i,
                    suspiciousness=0.6,
                    reason="Target-specific pattern (potential Err-Pred)"
                ))
            
            # Pattern: Switch case (common error location)
            if 'case' in line and 'return' in line:
                suspicious.append(SuspiciousLocation(
                    start_line=i,
                    end_line=i,
                    suspiciousness=0.5,
                    reason="Switch case with return (common error pattern)"
                ))
        
        return suspicious
    
    def _rank_suspicious(
        self, 
        locations: List[SuspiciousLocation]
    ) -> List[SuspiciousLocation]:
        """의심 위치 순위 매기기"""
        # Remove duplicates and sort by suspiciousness
        unique = {}
        for loc in locations:
            key = (loc.start_line, loc.end_line)
            if key not in unique or unique[key].suspiciousness < loc.suspiciousness:
                unique[key] = loc
        
        return sorted(unique.values(), key=lambda x: -x.suspiciousness)
    
    # Helper methods
    def _line_in_trace(self, line: str, trace: List[str]) -> bool:
        return any(line.strip() in t for t in trace)
    
    def _is_decision_point(self, line: str) -> bool:
        keywords = ['if', 'switch', 'case', '?', ':']
        return any(kw in line for kw in keywords)
    
    def _has_target_specific_pattern(self, line: str) -> bool:
        patterns = ['R_RISCV', 'R_ARM', 'R_MIPS', 'R_X86']
        return any(p in line for p in patterns)
    
    def _satisfies_postcondition(
        self, 
        return_line: str, 
        post: 'Condition',
        cex: 'Counterexample'
    ) -> bool:
        # Simplified check
        return True  # Placeholder
```

---

## 4. 데이터 모델

### 4.1 Core Data Classes

```python
# src/core/types.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

class TargetArch(Enum):
    """지원 타겟 아키텍처"""
    RISCV = "riscv"
    RISCV_RI5CY = "ri5cy"
    ARM = "arm"
    AARCH64 = "aarch64"
    MIPS = "mips"
    X86_64 = "x86_64"
    XCORE = "xcore"

@dataclass
class Function:
    """백엔드 함수"""
    name: str
    signature: str
    body: str
    file_path: str
    target: TargetArch
    is_generated: bool = False
    
    # Metadata
    line_count: int = 0
    complexity: int = 0  # Cyclomatic complexity

@dataclass
class Module:
    """백엔드 모듈 (관련 함수들의 집합)"""
    name: str
    functions: List[Function]
    dependencies: List[str]
    target: TargetArch

@dataclass
class Backend:
    """완전한 백엔드"""
    name: str
    target: TargetArch
    modules: List[Module]
    
    # Verification status
    verified_functions: int = 0
    total_functions: int = 0
    
    @property
    def verification_coverage(self) -> float:
        if self.total_functions == 0:
            return 0.0
        return self.verified_functions / self.total_functions

@dataclass
class GenerationResult:
    """생성 결과"""
    function: Function
    generated_code: str
    confidence: float
    generation_time_ms: float
    model_version: str

@dataclass
class VerificationCertificate:
    """검증 인증서"""
    function_name: str
    verified_at: datetime
    spec_hash: str
    vc_hash: str
    solver_result: str
    proof_trace: Optional[str] = None

@dataclass
class VERAResult:
    """VERA 파이프라인 최종 결과"""
    backend: Backend
    generation_results: List[GenerationResult]
    verification_certificates: List[VerificationCertificate]
    repair_history: List[Dict[str, Any]]
    
    # Metrics
    total_time_seconds: float
    accuracy: float
    verification_coverage: float
    repair_success_rate: float
```

---

## 5. 핵심 알고리즘 구현

### 5.1 Main Pipeline

```python
# src/core/pipeline.py

from typing import List, Optional
from dataclasses import dataclass
import logging
import time

from .types import Backend, TargetArch, VERAResult
from ..spec_inference import SpecificationExtractor
from ..generation import NeuralGenerator
from ..verification import SMTSolver, VCGenerator
from ..repair import CGNR
from ..hierarchical import HierarchicalVerifier

logger = logging.getLogger(__name__)

class VERAPipeline:
    """
    VERA 메인 파이프라인
    
    Flow:
    1. Reference backends 로드 및 파싱
    2. 명세 자동 추론
    3. Neural generation
    4. Formal verification
    5. CGNR (필요 시)
    6. Hierarchical verification
    7. 결과 및 인증서 생성
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize components
        self.spec_extractor = SpecificationExtractor()
        self.generator = NeuralGenerator(config['generation'])
        self.vc_generator = VCGenerator()
        self.verifier = SMTSolver(timeout_ms=config['verification']['timeout_ms'])
        self.cgnr = CGNR(
            verifier=self.verifier,
            repair_model=self._load_repair_model(),
            fault_locator=FaultLocator(),
            max_iterations=config['repair']['max_iterations']
        )
        self.hierarchical_verifier = HierarchicalVerifier()
    
    def run(
        self, 
        reference_backends: List[Backend],
        target_arch: TargetArch,
        target_description: str
    ) -> VERAResult:
        """
        VERA 파이프라인 실행
        
        Args:
            reference_backends: 참조 백엔드들
            target_arch: 생성할 타겟 아키텍처
            target_description: 타겟 설명 (.td 파일 등)
            
        Returns:
            VERAResult: 최종 결과
        """
        start_time = time.time()
        
        results = []
        certificates = []
        repair_history = []
        
        logger.info(f"Starting VERA pipeline for {target_arch.value}")
        
        # Step 1: 함수 템플릿 추출
        templates = self._extract_templates(reference_backends)
        logger.info(f"Extracted {len(templates)} function templates")
        
        for template in templates:
            # Step 2: 명세 추론
            spec = self.spec_extractor.infer_specification(
                template.name,
                {b.target.value: self._get_function(b, template.name) 
                 for b in reference_backends}
            )
            logger.info(f"Inferred specification for {template.name}")
            
            # Step 3: 코드 생성
            gen_result = self.generator.generate(
                template=template,
                target=target_arch,
                target_desc=target_description
            )
            logger.info(f"Generated code for {template.name} "
                       f"(confidence: {gen_result.confidence:.2f})")
            
            # Step 4: 검증
            vc = self.vc_generator.generate_vc(gen_result.generated_code, spec)
            ver_result = self.verifier.check(vc)
            
            if ver_result.status == VerificationResult.VERIFIED:
                logger.info(f"Verified: {template.name}")
                certificates.append(self._create_certificate(template.name, spec, vc))
            
            elif ver_result.status == VerificationResult.COUNTEREXAMPLE:
                # Step 5: CGNR
                logger.info(f"Verification failed for {template.name}, starting CGNR")
                
                cgnr_result = self.cgnr.repair(
                    gen_result.generated_code,
                    spec
                )
                
                if cgnr_result.success:
                    gen_result.generated_code = cgnr_result.final_code
                    repair_history.extend(cgnr_result.repair_history)
                    certificates.append(self._create_certificate(
                        template.name, spec, 
                        self.vc_generator.generate_vc(cgnr_result.final_code, spec)
                    ))
                    logger.info(f"CGNR successful for {template.name} "
                               f"({cgnr_result.iterations} iterations)")
                else:
                    logger.warning(f"CGNR failed for {template.name}")
            
            results.append(gen_result)
        
        # Step 6: Hierarchical verification
        backend = self._assemble_backend(target_arch, results)
        hier_result = self.hierarchical_verifier.verify(backend)
        
        logger.info(f"Hierarchical verification: "
                   f"L1={hier_result.l1_coverage:.1%}, "
                   f"L2={hier_result.l2_coverage:.1%}, "
                   f"L3={'PASS' if hier_result.l3_pass else 'FAIL'}")
        
        total_time = time.time() - start_time
        
        return VERAResult(
            backend=backend,
            generation_results=results,
            verification_certificates=certificates,
            repair_history=repair_history,
            total_time_seconds=total_time,
            accuracy=len(certificates) / len(results) if results else 0,
            verification_coverage=hier_result.l1_coverage,
            repair_success_rate=self._calc_repair_rate(repair_history)
        )
```

---

## 6. 인터페이스 정의

### 6.1 Python API

```python
# src/__init__.py

from .core.pipeline import VERAPipeline
from .core.types import Backend, TargetArch, VERAResult

def generate_backend(
    reference_path: str,
    target_arch: str,
    target_description_path: str,
    config_path: str = "configs/default.yaml"
) -> VERAResult:
    """
    VERA를 사용한 백엔드 생성
    
    Args:
        reference_path: Reference backends 디렉토리 경로
        target_arch: 타겟 아키텍처 (예: "riscv", "arm")
        target_description_path: 타겟 설명 파일 경로
        config_path: 설정 파일 경로
        
    Returns:
        VERAResult: 생성 결과
        
    Example:
        >>> result = generate_backend(
        ...     reference_path="/path/to/llvm/backends",
        ...     target_arch="riscv",
        ...     target_description_path="/path/to/RISCV.td",
        ... )
        >>> print(f"Accuracy: {result.accuracy:.1%}")
        >>> print(f"Verified: {result.verification_coverage:.1%}")
    """
    import yaml
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load reference backends
    references = _load_references(reference_path)
    target = TargetArch(target_arch)
    
    with open(target_description_path) as f:
        target_desc = f.read()
    
    # Run pipeline
    pipeline = VERAPipeline(config)
    return pipeline.run(references, target, target_desc)
```

### 6.2 CLI Interface

```python
# scripts/vera_cli.py

import argparse
import logging
from vera import generate_backend

def main():
    parser = argparse.ArgumentParser(
        description="VERA: Verified and Extensible Retargetable Architecture"
    )
    
    parser.add_argument(
        "--reference", "-r",
        required=True,
        help="Path to reference backends directory"
    )
    
    parser.add_argument(
        "--target", "-t",
        required=True,
        choices=["riscv", "arm", "aarch64", "mips", "x86_64"],
        help="Target architecture"
    )
    
    parser.add_argument(
        "--target-desc", "-d",
        required=True,
        help="Path to target description file (.td)"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Run VERA
    result = generate_backend(
        reference_path=args.reference,
        target_arch=args.target,
        target_description_path=args.target_desc,
        config_path=args.config
    )
    
    # Output results
    print("\n" + "="*60)
    print("VERA Generation Complete")
    print("="*60)
    print(f"Target: {args.target}")
    print(f"Functions generated: {len(result.generation_results)}")
    print(f"Accuracy: {result.accuracy:.1%}")
    print(f"Verification coverage: {result.verification_coverage:.1%}")
    print(f"Repair success rate: {result.repair_success_rate:.1%}")
    print(f"Total time: {result.total_time_seconds:.1f}s")
    print("="*60)
    
    # Save results
    _save_results(result, args.output)

if __name__ == "__main__":
    main()
```

---

## 7. 구현 로드맵

### 7.1 Phase별 계획

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VERA Implementation Roadmap                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Phase 1: Foundation (Week 1-4)                                                 │
│  ═══════════════════════════════                                                │
│  □ Core infrastructure setup                                                    │
│  □ Parser and AST utilities (libclang integration)                              │
│  □ Basic SMT solver interface (Z3)                                              │
│  □ Unit test framework                                                          │
│                                                                                 │
│  Phase 2: Specification Inference (Week 5-8)                                    │
│  ══════════════════════════════════════════                                     │
│  □ GumTree alignment implementation                                             │
│  □ Precondition extractor                                                       │
│  □ Postcondition extractor                                                      │
│  □ Invariant inference                                                          │
│  □ Spec validation                                                              │
│                                                                                 │
│  Phase 3: Verification (Week 9-12)                                              │
│  ═════════════════════════════════                                              │
│  □ VC generation for compiler code patterns                                     │
│  □ SMT encoding for C++ constructs                                              │
│  □ BMC integration                                                              │
│  □ Counterexample extraction                                                    │
│                                                                                 │
│  Phase 4: CGNR (Week 13-16)                                                     │
│  ═════════════════════════                                                      │
│  □ Fault localization algorithms                                                │
│  □ Neural repair model training                                                 │
│  □ CGNR main loop                                                               │
│  □ Repair validation                                                            │
│                                                                                 │
│  Phase 5: Hierarchical Verification (Week 17-20)                                │
│  ══════════════════════════════════════════════                                 │
│  □ Function-level contracts                                                     │
│  □ Module composition verification                                              │
│  □ Backend integration verification                                             │
│  □ Certificate generation                                                       │
│                                                                                 │
│  Phase 6: Evaluation & Optimization (Week 21-24)                                │
│  ════════════════════════════════════════════════                               │
│  □ VEGA benchmark evaluation                                                    │
│  □ Extended target benchmarks                                                   │
│  □ Performance optimization                                                     │
│  □ Documentation and release                                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 마일스톤

| Milestone | 목표 | 완료 기준 |
|-----------|------|-----------|
| **M1** (Week 4) | Infrastructure | Parser, SMT interface working |
| **M2** (Week 8) | Spec Inference | 70% functions에서 spec 추론 |
| **M3** (Week 12) | Verification | RISC-V 10개 함수 검증 성공 |
| **M4** (Week 16) | CGNR | 50% 실패 케이스 자동 수정 |
| **M5** (Week 20) | Hierarchical | 전체 파이프라인 동작 |
| **M6** (Week 24) | Release | VEGA 대비 +10% accuracy |

---

## 8. 테스트 전략

### 8.1 테스트 계층

```python
# tests/unit/test_spec_extractor.py

import pytest
from vera.spec_inference import SpecificationExtractor

class TestSpecificationExtractor:
    """Specification Extractor 단위 테스트"""
    
    @pytest.fixture
    def extractor(self):
        return SpecificationExtractor()
    
    def test_extract_null_check_precondition(self, extractor):
        """Null check에서 precondition 추출 테스트"""
        code = """
        unsigned getRelocType(const MCFixup &Fixup) {
            if (!Fixup.getValue()) return 0;
            // ...
        }
        """
        spec = extractor.infer_from_code(code)
        
        assert len(spec.preconditions) > 0
        assert any("Fixup.getValue()" in p.expression for p in spec.preconditions)
    
    def test_extract_switch_invariant(self, extractor):
        """Switch문에서 invariant 추출 테스트"""
        code = """
        unsigned getRelocType(const MCFixup &Fixup) {
            switch (Fixup.getTargetKind()) {
                case FK_NONE: return R_RISCV_NONE;
                case FK_Data_4: return R_RISCV_32;
                default: llvm_unreachable("unknown fixup kind");
            }
        }
        """
        spec = extractor.infer_from_code(code)
        
        assert len(spec.invariants) > 0
        assert any("switch" in inv.expression.lower() for inv in spec.invariants)

# tests/integration/test_cgnr_pipeline.py

class TestCGNRPipeline:
    """CGNR 통합 테스트"""
    
    def test_repair_wrong_relocation(self):
        """잘못된 relocation type 수정 테스트"""
        initial_code = """
        case FK_Data_4: return R_RISCV_32;  // Bug: should check is64Bit
        """
        
        spec = Specification(
            function_name="getRelocType",
            preconditions=[],
            postconditions=[Condition(
                type=ConditionType.POSTCONDITION,
                expression="is64Bit ⟹ result = R_RISCV_64",
                source_location="",
                is_target_independent=False
            )],
            invariants=[]
        )
        
        cgnr = CGNR(verifier, repair_model, fault_locator)
        result = cgnr.repair(initial_code, spec)
        
        assert result.success
        assert "is64Bit" in result.final_code
        assert "R_RISCV_64" in result.final_code
```

### 8.2 벤치마크 테스트

```python
# tests/benchmark/test_vega_targets.py

import pytest
from vera import generate_backend

class TestVEGABenchmarks:
    """VEGA 논문의 벤치마크 재현"""
    
    @pytest.mark.benchmark
    def test_riscv_accuracy(self, reference_backends, riscv_desc):
        """RISC-V 정확도 테스트 (VEGA: 71.5%, VERA target: >85%)"""
        result = generate_backend(
            reference_backends,
            target_arch="riscv",
            target_description=riscv_desc
        )
        
        assert result.accuracy >= 0.85, \
            f"Accuracy {result.accuracy:.1%} below target 85%"
    
    @pytest.mark.benchmark
    def test_riscv_verification_coverage(self, reference_backends, riscv_desc):
        """RISC-V 검증 커버리지 테스트"""
        result = generate_backend(
            reference_backends,
            target_arch="riscv",
            target_description=riscv_desc
        )
        
        assert result.verification_coverage >= 0.80, \
            f"Verification coverage {result.verification_coverage:.1%} below target 80%"
```

---

## 9. 배포 및 통합

### 9.1 LLVM 통합

```cpp
// llvm/lib/Target/RISCV/VERAIntegration.cpp

namespace llvm {

/// VERA-generated code integration point
class VERABackendIntegration {
public:
    /// Load VERA-generated backend components
    static bool loadGeneratedComponents(const std::string &path);
    
    /// Verify component at runtime (optional)
    static bool verifyComponent(const std::string &name);
    
    /// Get verification certificate
    static std::optional<Certificate> getCertificate(const std::string &name);
};

} // namespace llvm
```

### 9.2 Docker 배포

```dockerfile
# Dockerfile

FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    libclang-dev \
    z3 \
    && rm -rf /var/lib/apt/lists/*

# Install VERA
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

# Entry point
ENTRYPOINT ["python", "-m", "vera.cli"]
```

---

## 10. 부록: 설정 및 참조

### 10.1 기본 설정 파일

```yaml
# configs/default.yaml

# General settings
project_name: "vera"
log_level: "INFO"

# Specification inference
spec_inference:
  alignment_algorithm: "gumtree"
  min_reference_count: 2
  confidence_threshold: 0.7

# Neural generation
generation:
  model_name: "microsoft/unixcoder-base-nine"
  checkpoint_path: "models/generation/unixcoder_finetuned"
  max_length: 512
  beam_size: 5
  temperature: 0.8

# Verification
verification:
  solver: "z3"
  timeout_ms: 30000
  bmc_bound: 10
  
# CGNR
repair:
  model_path: "models/repair/cgnr_model"
  max_iterations: 10
  candidates_per_iteration: 5

# Hierarchical verification
hierarchical:
  l1_timeout_ms: 10000
  l2_timeout_ms: 60000
  l3_timeout_ms: 300000
```

### 10.2 타겟별 설정

```yaml
# configs/targets/riscv.yaml

target:
  name: "riscv"
  variants: ["rv32", "rv64"]
  
  # ISA-specific settings
  register_classes:
    - name: "GPR"
      size: 32
      count: 32
    - name: "FPR"
      size: 64
      count: 32
  
  # Relocation types
  relocations:
    - "R_RISCV_NONE"
    - "R_RISCV_32"
    - "R_RISCV_64"
    - "R_RISCV_BRANCH"
    - "R_RISCV_JAL"
    # ...

  # Verification hints
  verification_hints:
    critical_functions:
      - "getRelocType"
      - "applyFixup"
      - "encodeInstruction"
```

### 10.3 참조 문헌 및 URL

| 자료 | URL | 설명 |
|------|-----|------|
| **VEGA Paper** | [CGO 2025](https://dl.acm.org/doi/proceedings/10.1145/3640537) | Function-level accuracies |
| **Hydride** | [ASPLOS 2024](https://dl.acm.org/doi/10.1145/3620665) | CEGIS 기반 합성 |
| **VeGen** | [adapt.cs.illinois.edu](https://adapt.cs.illinois.edu) | Vector synthesis |
| **Z3 Solver** | [github.com/Z3Prover](https://github.com/Z3Prover/z3) | SMT solver |
| **UniXcoder** | [github.com/microsoft](https://github.com/microsoft/CodeBERT) | Pre-trained model |
| **LLVM** | [llvm.org](https://llvm.org) | Compiler infrastructure |

---

*문서 종료*
