"""Tests for benchmark configuration parsing and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config_parser import (
    BenchmarkConfig,
    ConfigurationError,
    load_config,
)


def write_yaml(path: Path, content: str) -> Path:
    """Write YAML fixture content and return its file path."""
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_valid_yaml_configuration(tmp_path: Path) -> None:
    """A valid YAML file should create a typed benchmark configuration."""
    config_file = write_yaml(
        tmp_path / "benchmark.yaml",
        """
models:
  - distilgpt2
  - gpt2
  - EleutherAI/pythia-160m
dataset_path: prompts.jsonl
max_new_tokens: 25
warmup_prompts: 1
output_dir: output
min_output_tokens: 2
generation:
  do_sample: false
  use_cache: true
  repetition_penalty: 1.2
""",
    )

    config = load_config(config_file)

    assert isinstance(config, BenchmarkConfig)
    assert config.models == [
        "distilgpt2",
        "gpt2",
        "EleutherAI/pythia-160m",
    ]
    assert config.dataset_path == (tmp_path / "prompts.jsonl").resolve()
    assert config.output_dir == (tmp_path / "output").resolve()
    assert config.max_new_tokens == 25
    assert config.warmup_prompts == 1
    assert config.min_output_tokens == 2
    assert config.generation.do_sample is False
    assert config.generation.use_cache is True
    assert config.generation.repetition_penalty == 1.2


def test_loads_valid_json_configuration(tmp_path: Path) -> None:
    """A valid JSON file should create a typed benchmark configuration."""
    config_file = tmp_path / "benchmark.json"
    config_file.write_text(
        json.dumps(
            {
                "models": [
                    "distilgpt2",
                    "gpt2",
                    "EleutherAI/pythia-160m",
                ],
                "dataset_path": "prompts.jsonl",
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.models[0] == "distilgpt2"
    assert config.dataset_path == (tmp_path / "prompts.jsonl").resolve()
    assert config.max_new_tokens == 50
    assert config.warmup_prompts == 0


@pytest.mark.parametrize(
    ("models", "expected_message"),
    [
        (
            ["distilgpt2", "gpt2"],
            "at least three model identifiers",
        ),
        (
            ["distilgpt2", "gpt2", "gpt2"],
            "distinct model identifiers",
        ),
        (
            ["distilgpt2", "", "EleutherAI/pythia-160m"],
            "non-empty string",
        ),
    ],
)
def test_rejects_invalid_model_lists(
    tmp_path: Path,
    models: list[str],
    expected_message: str,
) -> None:
    """The model list must contain three distinct non-empty strings."""
    config_file = tmp_path / "invalid_models.json"
    config_file.write_text(
        json.dumps(
            {
                "models": models,
                "dataset_path": "prompts.jsonl",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match=expected_message):
        load_config(config_file)


def test_rejects_missing_dataset_path(tmp_path: Path) -> None:
    """A configuration must include a non-empty dataset path."""
    config_file = write_yaml(
        tmp_path / "missing_dataset.yaml",
        """
models:
  - distilgpt2
  - gpt2
  - EleutherAI/pythia-160m
""",
    )

    with pytest.raises(ConfigurationError, match="dataset_path"):
        load_config(config_file)


def test_rejects_invalid_numeric_settings(tmp_path: Path) -> None:
    """Numeric generation and warm-up settings must be valid."""
    config_file = write_yaml(
        tmp_path / "invalid_numbers.yaml",
        """
models:
  - distilgpt2
  - gpt2
  - EleutherAI/pythia-160m
dataset_path: prompts.jsonl
max_new_tokens: 0
""",
    )

    with pytest.raises(ConfigurationError, match="max_new_tokens"):
        load_config(config_file)


def test_rejects_unsupported_file_extension(tmp_path: Path) -> None:
    """Only YAML and JSON configuration files are supported."""
    config_file = (tmp_path / "benchmark.txt")
    config_file.write_text("models: []", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Unsupported configuration"):
        load_config(config_file)