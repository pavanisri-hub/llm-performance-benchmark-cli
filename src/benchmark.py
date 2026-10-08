"""Benchmark orchestration for multi-model LLM performance evaluation."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from config_parser import BenchmarkConfig
from dataset_loader import Prompt
from model_runner import HuggingFaceRunner, InferenceError

logger = logging.getLogger(__name__)


@dataclass
class PromptResult:
    """Result of running inference on a single prompt."""

    prompt_id: str
    prompt_length: int
    generated_tokens: int
    latency_seconds: float
    tokens_per_second: float


@dataclass
class ModelResult:
    """Aggregated benchmark results for a single model."""

    model_id: str
    device: str
    prompt_results: list[PromptResult] = field(default_factory=list)
    mean_latency: float = 0.0
    median_latency: float = 0.0
    p95_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0
    mean_tokens_per_second: float = 0.0
    total_prompts: int = 0

    def compute_statistics(self) -> None:
        """Compute aggregated statistics from prompt results."""
        if not self.prompt_results:
            return

        latencies = [r.latency_seconds for r in self.prompt_results]
        tps_values = [r.tokens_per_second for r in self.prompt_results]

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        self.mean_latency = sum(latencies) / n
        self.median_latency = sorted_latencies[n // 2] if n % 2 == 1 else (sorted_latencies[n // 2 - 1] + sorted_latencies[n // 2]) / 2
        p95_index = math.ceil(0.95 * n) - 1
        self.p95_latency = sorted_latencies[max(0, p95_index)]
        self.min_latency = min(latencies)
        self.max_latency = max(latencies)
        self.mean_tokens_per_second = sum(tps_values) / n
        self.total_prompts = n


@dataclass
class BenchmarkReport:
    """Full benchmark report for all models."""

    timestamp: str
    config_path: str
    models: list[ModelResult] = field(default_factory=list)
    system_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "config_path": self.config_path,
            "models": [
                {
                    "model_id": m.model_id,
                    "device": m.device,
                    "prompt_results": [
                        {
                            "prompt_id": r.prompt_id,
                            "prompt_length": r.prompt_length,
                            "generated_tokens": r.generated_tokens,
                            "latency_seconds": r.latency_seconds,
                            "tokens_per_second": r.tokens_per_second,
                        }
                        for r in m.prompt_results
                    ],
                    "mean_latency": m.mean_latency,
                    "median_latency": m.median_latency,
                    "p95_latency": m.p95_latency,
                    "min_latency": m.min_latency,
                    "max_latency": m.max_latency,
                    "mean_tokens_per_second": m.mean_tokens_per_second,
                    "total_prompts": m.total_prompts,
                }
                for m in self.models
            ],
            "system_info": self.system_info,
        }


class BenchmarkOrchestrator:
    """
    Orchestrate benchmarking across multiple models.

    This orchestrator:
      - Loads each model dynamically via HuggingFaceRunner.
      - Runs warm-up prompts.
      - Runs full benchmark prompts.
      - Records per-prompt latency and throughput.
      - Aggregates per-model statistics.
      - Saves structured JSON results.
      - Explicitly releases memory between models.
    """

    def __init__(self, config: BenchmarkConfig, config_path: Path) -> None:
        """
        Initialize the orchestrator.

        Args:
            config: Validated benchmark configuration.
            config_path: Path to the configuration file (for reporting).
        """
        self.config = config
        self.config_path = config_path
        self.report = BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_path=str(config_path.resolve()),
        )
        self._capture_system_info()

    def _capture_system_info(self) -> None:
        """Capture system information for the report."""
        self.report.system_info = {
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
        }

    def _run_model_benchmark(self, model_id: str, prompts: list[Prompt]) -> ModelResult:
        """Run benchmark for a single model."""
        logger.info("Starting benchmark for model %s", model_id)

        runner = HuggingFaceRunner(model_id, self.config.max_new_tokens)
        result = ModelResult(model_id=model_id, device=str(runner.device))

        try:
            runner.load()

            # Warm-up phase
            warmup_count = min(self.config.warmup_prompts, len(prompts))
            logger.info("Running %d warm-up prompts for %s", warmup_count, model_id)
            for i in range(warmup_count):
                try:
                    runner.generate(prompts[i].text)
                except InferenceError as error:
                    logger.warning("Warm-up failed for %s: %s", model_id, error)

            # Benchmark phase
            logger.info("Running %d benchmark prompts for %s", len(prompts), model_id)
            for prompt in prompts:
                start_time = time.perf_counter()
                generated_tokens = runner.generate(prompt.text)
                latency = time.perf_counter() - start_time

                tokens_per_second = generated_tokens / latency if latency > 0 else 0.0

                prompt_result = PromptResult(
                    prompt_id=prompt.prompt_id,
                    prompt_length=len(prompt.text.split()),
                    generated_tokens=generated_tokens,
                    latency_seconds=latency,
                    tokens_per_second=tokens_per_second,
                )
                result.prompt_results.append(prompt_result)
                logger.info(
                    "Prompt %s: %d tokens, %.3f s, %.2f tokens/s",
                    prompt.prompt_id,
                    generated_tokens,
                    latency,
                    tokens_per_second,
                )

            result.compute_statistics()

        except Exception as error:
            logger.error("Benchmark failed for %s: %s", model_id, error)
            raise
        finally:
            runner.unload()

        logger.info("Benchmark completed for model %s", model_id)
        return result

    def run(self) -> BenchmarkReport:
        """Run the full benchmark across all configured models."""
        from dataset_loader import load_prompts

        prompts = load_prompts(self.config.dataset_path)
        logger.info("Loaded %d prompts for benchmarking", len(prompts))

        for model_id in self.config.models:
            model_result = self._run_model_benchmark(model_id, prompts)
            self.report.models.append(model_result)

        return self.report

    def save_report(self, output_dir: Path) -> Path:
        """Save the benchmark report to a JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{timestamp}.json"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.report.to_dict(), f, indent=2)

        logger.info("Benchmark report saved to %s", output_path)
        return output_path