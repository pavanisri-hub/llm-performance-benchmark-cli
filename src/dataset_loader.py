"""Prompt dataset loading for CSV and JSONL benchmark inputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


class DatasetError(ValueError):
    """Raised when a benchmark prompt dataset is missing or invalid."""


@dataclass(frozen=True)
class PromptRecord:
    """A single benchmark prompt with a stable identifier."""

    prompt_id: str
    prompt: str


def load_prompts(dataset_path: str | Path) -> list[PromptRecord]:
    """Load validated prompt records from a JSONL or CSV dataset.

    JSONL records must contain a non-empty ``prompt`` field and may provide
    an optional ``id`` field. CSV files must include a ``prompt`` column and
    may include an ``id`` column.

    Args:
        dataset_path: Path to a .jsonl or .csv prompt dataset.

    Returns:
        A non-empty list of validated PromptRecord objects.

    Raises:
        DatasetError: If the path is invalid, format is unsupported, or
            records do not contain usable prompts.
    """
    path = Path(dataset_path).expanduser()

    if not path.exists():
        raise DatasetError(f"Dataset file was not found: {path}")

    if not path.is_file():
        raise DatasetError(f"Dataset path is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        prompts = _load_jsonl(path)
    elif suffix == ".csv":
        prompts = _load_csv(path)
    else:
        raise DatasetError(
            "Unsupported dataset format. Use a .jsonl or .csv file."
        )

    if not prompts:
        raise DatasetError("Dataset must contain at least one valid prompt.")

    prompt_ids = [record.prompt_id for record in prompts]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise DatasetError("Dataset contains duplicate prompt IDs.")

    return prompts


def _load_jsonl(path: Path) -> list[PromptRecord]:
    """Load prompts from a JSON Lines dataset."""
    prompts: list[PromptRecord] = []

    try:
        with path.open("r", encoding="utf-8") as dataset_file:
            for line_number, raw_line in enumerate(dataset_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetError(
                        f"Invalid JSON on line {line_number}: {error.msg}"
                    ) from error

                prompts.append(_build_prompt_record(record, line_number))
    except OSError as error:
        raise DatasetError(
            f"Could not read dataset '{path}': {error}"
        ) from error

    return prompts


def _load_csv(path: Path) -> list[PromptRecord]:
    """Load prompts from a CSV dataset."""
    prompts: list[PromptRecord] = []

    try:
        with path.open("r", encoding="utf-8", newline="") as dataset_file:
            reader = csv.DictReader(dataset_file)

            if not reader.fieldnames or "prompt" not in reader.fieldnames:
                raise DatasetError(
                    "CSV dataset must contain a 'prompt' column."
                )

            for row_number, row in enumerate(reader, start=2):
                prompts.append(_build_prompt_record(row, row_number))
    except OSError as error:
        raise DatasetError(
            f"Could not read dataset '{path}': {error}"
        ) from error

    return prompts


def _build_prompt_record(
    raw_record: object,
    record_number: int,
) -> PromptRecord:
    """Validate one raw dataset record and return a PromptRecord."""
    if not isinstance(raw_record, dict):
        raise DatasetError(
            f"Record {record_number} must be a JSON object or CSV row."
        )

    raw_prompt = raw_record.get("prompt")
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise DatasetError(
            f"Record {record_number} must contain a non-empty 'prompt'."
        )

    raw_id = raw_record.get("id")
    if raw_id is None or not str(raw_id).strip():
        prompt_id = f"prompt_{record_number:03d}"
    else:
        prompt_id = str(raw_id).strip()

    return PromptRecord(prompt_id=prompt_id, prompt=raw_prompt.strip())