"""Command-line entry point for the LLM performance benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config_parser import ConfigurationError, load_config
from dataset_loader import DatasetError, load_prompts


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


def main() -> int:
    """Run CLI argument parsing and benchmark preflight validation."""
    parser = build_parser()
    args = parser.parse_args()
    return run_preflight(args.config)


if __name__ == "__main__":
    raise SystemExit(main())