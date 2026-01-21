#!/usr/bin/env python3
"""
VEGA-Verified: Semantically Verified Neural Compiler Backend Generation

Main entry point with support for different execution modes:
- vega: Original VEGA neural generation only
- vega-verified: VEGA + formal verification + CGNR repair
- verify-only: Verification only (no generation)

Usage:
    python -m src.main --mode vega-verified --target riscv
    python -m src.main verify --code code.cpp --spec spec.json
    python -m src.main compare --modes vega,vega-verified --target riscv
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from . import ExecutionMode
from .utils.config import Config, load_config, save_config
from .utils.logging import setup_logger, get_logger, ExperimentLogger
from .utils.metrics import MetricsCollector, FunctionMetrics


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="VEGA-Verified: Semantically Verified Neural Compiler Backend Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with VEGA-Verified mode
  python -m src.main --mode vega-verified --target riscv

  # Compare VEGA vs VEGA-Verified
  python -m src.main compare --modes vega,vega-verified --target riscv

  # Verify a single function
  python -m src.main verify --code function.cpp --spec function.spec

  # Infer specification from references
  python -m src.main infer-spec --function getRelocType --refs arm.cpp mips.cpp
        """
    )
    
    # Global options
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['vega', 'vega-verified', 'verify-only'],
        default='vega-verified',
        help='Execution mode (default: vega-verified)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--target', '-t',
        type=str,
        default='riscv',
        help='Target architecture (default: riscv)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify a single function')
    verify_parser.add_argument('--code', required=True, help='Code file')
    verify_parser.add_argument('--spec', required=True, help='Specification file')
    verify_parser.add_argument('--repair', action='store_true', help='Attempt repair if verification fails')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare different modes')
    compare_parser.add_argument(
        '--modes',
        type=str,
        default='vega,vega-verified',
        help='Modes to compare (comma-separated)'
    )
    
    # Infer spec command
    infer_parser = subparsers.add_parser('infer-spec', help='Infer specification')
    infer_parser.add_argument('--function', '-f', required=True, help='Function name')
    infer_parser.add_argument('--refs', nargs='+', required=True, help='Reference files')
    infer_parser.add_argument('--out', default='spec.json', help='Output spec file')
    
    # Run experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run full experiment')
    exp_parser.add_argument('--name', default=None, help='Experiment name')
    
    return parser


def run_verification(args: argparse.Namespace, config: Config) -> int:
    """Run single function verification."""
    from .verification import Verifier
    from .specification import Specification
    from .repair import CGNREngine
    
    logger = get_logger()
    
    # Load code
    code_path = Path(args.code)
    if not code_path.exists():
        logger.error(f"Code file not found: {args.code}")
        return 1
    
    with open(code_path) as f:
        code = f.read()
    
    # Load spec
    spec_path = Path(args.spec)
    if not spec_path.exists():
        logger.error(f"Specification file not found: {args.spec}")
        return 1
    
    spec = Specification.load(str(spec_path))
    
    # Run verification
    logger.info(f"Verifying {spec.function_name}...")
    
    verifier = Verifier(
        timeout_ms=config.verification.timeout_ms,
        verbose=args.verbose
    )
    
    result = verifier.verify(code, spec)
    
    # Print result
    print(f"\nVerification Result: {result.status.value}")
    print(f"Time: {result.time_ms:.2f}ms")
    
    if result.verified_properties:
        print(f"Verified properties: {result.verified_properties}")
    
    if result.counterexample:
        print(f"\nCounterexample:")
        print(f"  Inputs: {result.counterexample.input_values}")
        print(f"  Violated: {result.counterexample.violated_condition}")
        
        # Attempt repair if requested
        if args.repair and config.mode == 'vega-verified':
            logger.info("Attempting repair...")
            
            cgnr = CGNREngine(
                verifier=verifier,
                max_iterations=config.repair.max_iterations,
                verbose=args.verbose
            )
            
            repair_result = cgnr.repair(code, spec)
            
            print(f"\nRepair Result: {repair_result.status.value}")
            print(f"Iterations: {repair_result.iterations}")
            
            if repair_result.is_successful():
                print("\nRepaired code:")
                print(repair_result.repaired_code)
                
                # Save repaired code
                output_path = Path(args.output) / f"{spec.function_name}_repaired.cpp"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(repair_result.repaired_code)
                print(f"\nRepaired code saved to: {output_path}")
    
    return 0 if result.status.value == 'verified' else 1


def run_comparison(args: argparse.Namespace, config: Config) -> int:
    """Run comparison between different modes."""
    from .verification import Verifier
    from .specification import SpecificationInferrer, Specification
    from .repair import CGNREngine
    
    logger = get_logger()
    modes = args.modes.split(',')
    
    logger.info(f"Comparing modes: {modes}")
    logger.info(f"Target: {config.target}")
    
    # Initialize metrics collector
    metrics = MetricsCollector(output_dir=args.output)
    
    # Create experiment logger
    exp_name = f"compare_{config.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_logger = ExperimentLogger(exp_name, args.output)
    
    results_by_mode: Dict[str, Dict[str, Any]] = {}
    
    for mode in modes:
        logger.info(f"\n=== Running with mode: {mode} ===")
        
        # Update config for this mode
        config.mode = mode
        
        # Start experiment tracking
        exp_metrics = metrics.start_experiment(
            experiment_id=f"{exp_name}_{mode}",
            mode=mode,
            target=config.target
        )
        
        # Run pipeline (simplified for demo)
        mode_results = run_pipeline_for_mode(mode, config, args.verbose)
        
        # End experiment
        metrics.end_experiment()
        
        # Save metrics
        metrics.save_metrics()
        
        results_by_mode[mode] = mode_results
    
    # Generate comparison report
    if len(modes) >= 2:
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        
        for mode, results in results_by_mode.items():
            print(f"\n{mode.upper()}:")
            print(f"  Functions: {results.get('total_functions', 0)}")
            print(f"  Accuracy: {results.get('accuracy', 0):.1%}")
            print(f"  Verified: {results.get('verified_rate', 0):.1%}")
            print(f"  Time: {results.get('total_time_ms', 0):.0f}ms")
        
        # Calculate improvements
        if 'vega' in results_by_mode and 'vega-verified' in results_by_mode:
            vega = results_by_mode['vega']
            verified = results_by_mode['vega-verified']
            
            print("\n" + "-" * 40)
            print("VEGA-Verified vs VEGA:")
            
            acc_improvement = (verified.get('accuracy', 0) - vega.get('accuracy', 0)) * 100
            print(f"  Accuracy improvement: {acc_improvement:+.1f}pp")
            
            ver_rate = verified.get('verified_rate', 0) * 100
            print(f"  Verification coverage: {ver_rate:.1f}%")
            
            time_overhead = 0
            if vega.get('total_time_ms', 0) > 0:
                time_overhead = ((verified.get('total_time_ms', 0) / vega.get('total_time_ms', 1)) - 1) * 100
            print(f"  Time overhead: {time_overhead:+.1f}%")
    
    # Save experiment summary
    summary_path = exp_logger.save_summary()
    print(f"\nExperiment summary saved to: {summary_path}")
    
    return 0


def run_pipeline_for_mode(mode: str, config: Config, verbose: bool) -> Dict[str, Any]:
    """Run the pipeline for a specific mode and return results."""
    from .verification import Verifier
    from .repair import CGNREngine
    
    results = {
        'mode': mode,
        'total_functions': 0,
        'generated_functions': 0,
        'verified_functions': 0,
        'repaired_functions': 0,
        'accuracy': 0.0,
        'verified_rate': 0.0,
        'total_time_ms': 0.0,
    }
    
    start_time = time.time()
    
    # Create test functions (in real implementation, would load from VEGA)
    test_functions = create_test_functions()
    
    verifier = Verifier(verbose=verbose)
    cgnr = CGNREngine(verifier=verifier, verbose=verbose) if mode == 'vega-verified' else None
    
    for func_name, (code, spec) in test_functions.items():
        results['total_functions'] += 1
        
        if mode == 'vega':
            # Just count as generated (VEGA mode)
            results['generated_functions'] += 1
            # Simulate accuracy based on VEGA paper
            if hash(func_name) % 10 < 7:  # ~70% accuracy
                results['accuracy'] = 0.715
        
        elif mode == 'vega-verified':
            results['generated_functions'] += 1
            
            # Verify
            ver_result = verifier.verify(code, spec)
            
            if ver_result.is_verified():
                results['verified_functions'] += 1
            elif cgnr:
                # Try repair
                repair_result = cgnr.repair(code, spec)
                if repair_result.is_successful():
                    results['repaired_functions'] += 1
                    results['verified_functions'] += 1
        
        elif mode == 'verify-only':
            ver_result = verifier.verify(code, spec)
            if ver_result.is_verified():
                results['verified_functions'] += 1
    
    results['total_time_ms'] = (time.time() - start_time) * 1000
    
    if results['total_functions'] > 0:
        results['accuracy'] = results['generated_functions'] / results['total_functions']
        results['verified_rate'] = results['verified_functions'] / results['total_functions']
    
    return results


def create_test_functions() -> Dict[str, tuple]:
    """Create test functions for demonstration."""
    from .specification.spec_language import Specification, Condition, Variable, Constant
    
    # Example: getRelocType function
    code = """
unsigned RISCVELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                            const MCValue &Target,
                                            bool IsPCRel) const {
    switch (Fixup.getTargetKind()) {
    case FK_NONE: return ELF::R_RISCV_NONE;
    case FK_Data_1: return ELF::R_RISCV_8;
    case FK_Data_2: return ELF::R_RISCV_16;
    case FK_Data_4: return ELF::R_RISCV_32;
    case FK_Data_8: return ELF::R_RISCV_64;
    default:
        llvm_unreachable("Unknown fixup kind!");
    }
}
"""
    
    spec = Specification(
        function_name="getRelocType",
        module="MCCodeEmitter",
        preconditions=[
            Condition.valid(Variable("Fixup")),
            Condition.valid(Variable("Target")),
        ],
        postconditions=[
            Condition.ge(Variable("result"), Constant(0)),
        ],
        invariants=[
            Condition.implies(
                Condition.eq(Variable("Fixup_kind"), Constant("FK_NONE", "enum")),
                Condition.eq(Variable("result"), Constant("R_RISCV_NONE", "enum"))
            ),
        ]
    )
    
    return {
        "getRelocType": (code, spec),
    }


def run_infer_spec(args: argparse.Namespace, config: Config) -> int:
    """Run specification inference."""
    from .specification import SpecificationInferrer
    
    logger = get_logger()
    
    logger.info(f"Inferring specification for: {args.function}")
    logger.info(f"Reference files: {args.refs}")
    
    # Load references
    references = []
    for ref_path in args.refs:
        path = Path(ref_path)
        if not path.exists():
            logger.warning(f"Reference file not found: {ref_path}")
            continue
        
        with open(path) as f:
            code = f.read()
        
        # Extract target name from filename
        target = path.stem.upper()
        references.append((target, code))
    
    if not references:
        logger.error("No valid reference files found")
        return 1
    
    # Infer specification
    inferrer = SpecificationInferrer()
    spec = inferrer.infer(args.function, references)
    
    # Save specification
    output_path = Path(args.out)
    spec.save(str(output_path))
    
    print(f"\nInferred Specification for {args.function}:")
    print(spec)
    print(f"\nSaved to: {output_path}")
    
    return 0


def run_experiment(args: argparse.Namespace, config: Config) -> int:
    """Run full experiment."""
    logger = get_logger()
    
    exp_name = args.name or f"vega_verified_{config.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Target: {config.target}")
    
    # Run comparison
    args.modes = 'vega,vega-verified'
    return run_comparison(args, config)


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()
    
    # Override config with command line args
    config.mode = args.mode
    config.target = args.target
    config.experiment.output_dir = args.output
    
    if args.verbose:
        config.log_level = "DEBUG"
    
    # Setup logging
    setup_logger(
        name="vega_verified",
        level=config.log_level,
        log_file=config.log_file
    )
    
    logger = get_logger()
    
    # Dispatch to command handlers
    if args.command == 'verify':
        return run_verification(args, config)
    elif args.command == 'compare':
        return run_comparison(args, config)
    elif args.command == 'infer-spec':
        return run_infer_spec(args, config)
    elif args.command == 'experiment':
        return run_experiment(args, config)
    else:
        # Default: show info and run simple demo
        logger.info("VEGA-Verified: Semantically Verified Neural Compiler Backend Generation")
        logger.info(f"Mode: {config.mode}")
        logger.info(f"Target: {config.target}")
        
        print("\nRun with --help for usage information")
        print("\nQuick demo:")
        
        # Run a quick demo
        args.modes = 'vega,vega-verified'
        return run_comparison(args, config)


if __name__ == "__main__":
    sys.exit(main())
