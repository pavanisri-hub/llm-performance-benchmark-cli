"""Command-line entry point for the LLM performance benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark import BenchmarkOrchestrator
from config_parser import ConfigurationError, load_config
from dataset_loader import DatasetError, load_prompts
from visualization import generate_visualizations


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Hugging Face causal language models using a YAML "
            "or JSON configuration file."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a YAML or JSON benchmark configuration file.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the full benchmark pipeline (default: preflight only).",
    )
    return parser


def run_preflight(config_path: str) -> int:
    """Validate configuration and dataset before running model benchmarks."""
    try:
        config = load_config(config_path)
        prompts = load_prompts(config.dataset_path)
    except (ConfigurationError, DatasetError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    print("Benchmark preflight completed successfully.")
    print(f"Configuration: {Path(config_path).resolve()}")
    print(f"Models configured: {len(config.models)}")
    for model_id in config.models:
        print(f"  - {model_id}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Prompts loaded: {len(prompts)}")
    print(f"Maximum new tokens: {config.max_new_tokens}")
    print(f"Warm-up prompts per model: {config.warmup_prompts}")
    print(f"Output directory: {config.output_dir}")

    return 0


def run_benchmark(config_path: str) -> int:
    """Run the full benchmark pipeline."""
    try:
        config = load_config(config_path)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    config_path_obj = Path(config_path).resolve()
    orchestrator = BenchmarkOrchestrator(config, config_path_obj)

    print("Starting full benchmark pipeline...")
    print(f"Models: {len(config.models)}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output directory: {config.output_dir}")

    try:
        report = orchestrator.run()
        report_path = orchestrator.save_report(config.output_dir)

        print("\nBenchmark completed successfully.")
        print(f"Report saved to: {report_path}")

        print("\nGenerating visualizations...")
        chart_paths = generate_visualizations(report_path, config.output_dir)
        for chart_path in chart_paths:
            print(f"  - {chart_path}")

        print("\n=== Benchmark Summary ===")
        for model_result in report.models:
            print(f"\nModel: {model_result.model_id}")
            print(f"  Device: {model_result.device}")
            print(f"  Total prompts: {model_result.total_prompts}")
            print(f"  Mean latency: {model_result.mean_latency:.3f} s")
            print(f"  Median latency: {model_result.median_latency:.3f} s")
            print(f"  P95 latency: {model_result.p95_latency:.3f} s")
            print(f"  Min latency: {model_result.min_latency:.3f} s")
            print(f"  Max latency: {model_result.max_latency:.3f} s")
            print(f"  Mean tokens/second: {model_result.mean_tokens_per_second:.2f}")

        return 0

    except Exception as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1


def main() -> int:
    """Run CLI argument parsing and benchmark execution."""
    parser = build_parser()
    args = parser.parse_args()

    if args.run:
        return run_benchmark(args.config)
    else:
        return run_preflight(args.config)


if __name__ == "__main__":
    raise SystemExit(main())