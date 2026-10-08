"""Visualization utilities for benchmark results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def load_benchmark_report(report_path: Path) -> dict[str, Any]:
    """Load a benchmark report from a JSON file."""
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_latency_bar_chart(report: dict[str, Any], output_dir: Path) -> Path:
    """Generate a grouped bar chart of mean latency per model."""
    models = report["models"]
    model_ids = [m["model_id"] for m in models]
    latencies = [m["mean_latency"] for m in models]

    plt.figure(figsize=(8, 5))
    plt.bar(model_ids, latencies, color="#4C72B0")
    plt.ylabel("Mean Latency (seconds)")
    plt.title("Mean Latency per Model")
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = output_dir / "latency_bar_chart.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info("Latency bar chart saved to %s", output_path)
    return output_path


def plot_throughput_bar_chart(report: dict[str, Any], output_dir: Path) -> Path:
    """Generate a grouped bar chart of mean tokens/second per model."""
    models = report["models"]
    model_ids = [m["model_id"] for m in models]
    tps_values = [m["mean_tokens_per_second"] for m in models]

    plt.figure(figsize=(8, 5))
    plt.bar(model_ids, tps_values, color="#55A36B")
    plt.ylabel("Mean Tokens/Second")
    plt.title("Mean Throughput per Model")
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = output_dir / "throughput_bar_chart.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info("Throughput bar chart saved to %s", output_path)
    return output_path


def generate_visualizations(report_path: Path, output_dir: Path) -> list[Path]:
    """Generate all visualizations from a benchmark report."""
    report = load_benchmark_report(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = [
        plot_latency_bar_chart(report, output_dir),
        plot_throughput_bar_chart(report, output_dir),
    ]

    return charts