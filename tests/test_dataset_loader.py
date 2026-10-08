"""Tests for CSV and JSONL prompt dataset loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset_loader import DatasetError, PromptRecord, load_prompts


def test_loads_jsonl_prompts_with_explicit_and_generated_ids(
    tmp_path: Path,
) -> None:
    """JSONL datasets should preserve IDs and generate missing ones."""
    dataset_file = tmp_path / "prompts.jsonl"
    dataset_file.write_text(
        (
            '{"id": "first", "prompt": "Explain test-driven development."}\n'
            '{"prompt": "Describe containerization."}\n'
        ),
        encoding="utf-8",
    )

    prompts = load_prompts(dataset_file)

    assert prompts == [
        PromptRecord(
            prompt_id="first",
            prompt="Explain test-driven development.",
        ),
        PromptRecord(
            prompt_id="prompt_002",
            prompt="Describe containerization.",
        ),
    ]


def test_loads_csv_prompts(tmp_path: Path) -> None:
    """CSV datasets should load prompt and optional ID columns."""
    dataset_file = tmp_path / "prompts.csv"
    dataset_file.write_text(
        "id,prompt\n"
        "csv_1,Explain type hints.\n"
        ",Explain GPU memory profiling.\n",
        encoding="utf-8",
    )

    prompts = load_prompts(dataset_file)

    assert prompts == [
        PromptRecord(
            prompt_id="csv_1",
            prompt="Explain type hints.",
        ),
        PromptRecord(
            prompt_id="prompt_003",
            prompt="Explain GPU memory profiling.",
        ),
    ]


def test_rejects_csv_without_prompt_column(tmp_path: Path) -> None:
    """CSV files must contain a prompt column."""
    dataset_file = tmp_path / "invalid.csv"
    dataset_file.write_text(
        "id,text\nprompt_001,Missing required field\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="prompt"):
        load_prompts(dataset_file)


def test_rejects_blank_prompt(tmp_path: Path) -> None:
    """Blank prompt values should not reach the benchmark engine."""
    dataset_file = tmp_path / "blank.jsonl"
    dataset_file.write_text(
        '{"id": "blank", "prompt": "   "}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="non-empty 'prompt'"):
        load_prompts(dataset_file)


def test_rejects_duplicate_prompt_ids(tmp_path: Path) -> None:
    """Prompt IDs must be unique for reliable report rows."""
    dataset_file = tmp_path / "duplicates.jsonl"
    dataset_file.write_text(
        (
            '{"id": "same", "prompt": "First prompt"}\n'
            '{"id": "same", "prompt": "Second prompt"}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="duplicate prompt IDs"):
        load_prompts(dataset_file)


def test_rejects_invalid_jsonl_record(tmp_path: Path) -> None:
    """Invalid JSON should produce a line-specific error."""
    dataset_file = tmp_path / "invalid.jsonl"
    dataset_file.write_text('{"prompt": "Valid"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(DatasetError, match="Invalid JSON on line 2"):
        load_prompts(dataset_file)