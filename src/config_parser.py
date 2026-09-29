"""Configuration loading and validation for benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a benchmark configuration is missing or invalid."""


@dataclass(frozen=True)
class GenerationConfig:
    """Controls passed to Hugging Face model generation."""

    do_sample: bool = False
    use_cache: bool = True
    repetition_penalty: float = 1.0


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated settings for a complete benchmark run."""

    models: list[str]
    dataset_path: Path
    max_new_tokens: int = 50
    warmup_prompts: int = 0
    output_dir: Path = Path("outputs")
    min_output_tokens: int = 3
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def load_config(file_path: str | Path) -> BenchmarkConfig:
    """Load and validate a YAML or JSON benchmark configuration file.

    Args:
        file_path: Path to a .yaml, .yml, or .json configuration file.

    Returns:
        A validated BenchmarkConfig instance.

    Raises:
        ConfigurationError: If the file is absent, unsupported, malformed,
            or does not satisfy the expected schema.
    """
    config_path = Path(file_path).expanduser()

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file was not found: {config_path}"
        )

    if not config_path.is_file():
        raise ConfigurationError(
            f"Configuration path is not a file: {config_path}"
        )

    suffix = config_path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ConfigurationError(
            "Unsupported configuration format. "
            "Use a .yaml, .yml, or .json file."
        )

    try:
        raw_config = _read_config(config_path, suffix)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ConfigurationError(
            f"Could not parse configuration file '{config_path}': {error}"
        ) from error

    return _validate_config(raw_config, config_path.parent)


def _read_config(config_path: Path, suffix: str) -> dict[str, Any]:
    """Read raw mapping data from a supported configuration file."""
    with config_path.open("r", encoding="utf-8") as config_file:
        if suffix == ".json":
            content = json.load(config_file)
        else:
            content = yaml.safe_load(config_file)

    if content is None:
        return {}

    if not isinstance(content, dict):
        raise ConfigurationError(
            "Configuration root must be a mapping/object."
        )

    return content


def _validate_config(
    raw_config: dict[str, Any],
    config_directory: Path,
) -> BenchmarkConfig:
    """Validate raw configuration values and create a typed config object."""
    models = _validate_models(raw_config.get("models"))
    dataset_path = _validate_dataset_path(
        raw_config.get("dataset_path"),
        config_directory,
    )
    max_new_tokens = _validate_positive_integer(
        raw_config.get("max_new_tokens", 50),
        "max_new_tokens",
    )
    warmup_prompts = _validate_non_negative_integer(
        raw_config.get("warmup_prompts", 0),
        "warmup_prompts",
    )
    min_output_tokens = _validate_positive_integer(
        raw_config.get("min_output_tokens", 3),
        "min_output_tokens",
    )
    output_dir = _validate_output_dir(
        raw_config.get("output_dir", "outputs"),
        config_directory,
    )
    generation = _validate_generation(raw_config.get("generation", {}))

    return BenchmarkConfig(
        models=models,
        dataset_path=dataset_path,
        max_new_tokens=max_new_tokens,
        warmup_prompts=warmup_prompts,
        output_dir=output_dir,
        min_output_tokens=min_output_tokens,
        generation=generation,
    )


def _validate_models(value: Any) -> list[str]:
    """Validate that at least three distinct model identifiers are supplied."""
    if not isinstance(value, list):
        raise ConfigurationError(
            "'models' must be a list containing at least three model IDs."
        )

    normalized_models = [
        model.strip() for model in value if isinstance(model, str)
    ]

    if len(normalized_models) != len(value) or any(
        not model for model in normalized_models
    ):
        raise ConfigurationError(
            "Every item in 'models' must be a non-empty string."
        )

    if len(normalized_models) < 3:
        raise ConfigurationError(
            "'models' must contain at least three model identifiers."
        )

    if len(set(normalized_models)) != len(normalized_models):
        raise ConfigurationError(
            "'models' must contain distinct model identifiers."
        )

    return normalized_models


def _validate_dataset_path(value: Any, config_directory: Path) -> Path:
    """Validate and resolve a dataset file path."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            "'dataset_path' must be a non-empty string."
        )

    dataset_path = Path(value).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (config_directory / dataset_path).resolve()

    return dataset_path


def _validate_output_dir(value: Any, config_directory: Path) -> Path:
    """Validate and resolve an output directory path."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            "'output_dir' must be a non-empty string."
        )

    output_dir = Path(value).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_directory / output_dir).resolve()

    return output_dir


def _validate_positive_integer(value: Any, field_name: str) -> int:
    """Validate a strictly positive integer setting."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(
            f"'{field_name}' must be a positive integer."
        )

    return value


def _validate_non_negative_integer(value: Any, field_name: str) -> int:
    """Validate a non-negative integer setting."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(
            f"'{field_name}' must be a non-negative integer."
        )

    return value


def _validate_generation(value: Any) -> GenerationConfig:
    """Validate optional Hugging Face generation settings."""
    if value is None:
        value = {}

    if not isinstance(value, dict):
        raise ConfigurationError(
            "'generation' must be a mapping/object when provided."
        )

    do_sample = value.get("do_sample", False)
    use_cache = value.get("use_cache", True)
    repetition_penalty = value.get("repetition_penalty", 1.0)

    if not isinstance(do_sample, bool):
        raise ConfigurationError(
            "'generation.do_sample' must be a boolean."
        )

    if not isinstance(use_cache, bool):
        raise ConfigurationError(
            "'generation.use_cache' must be a boolean."
        )

    if (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, (int, float))
        or repetition_penalty <= 0
    ):
        raise ConfigurationError(
            "'generation.repetition_penalty' must be a positive number."
        )

    return GenerationConfig(
        do_sample=do_sample,
        use_cache=use_cache,
        repetition_penalty=float(repetition_penalty),
    )