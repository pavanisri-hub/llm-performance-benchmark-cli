"""Hugging Face model abstraction for causal language model inference."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Final

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised when model inference fails."""


class HuggingFaceRunner:
    """
    Run inference on a Hugging Face causal language model.

    This runner:
      - Loads the model and tokenizer dynamically from a model ID.
      - Uses CUDA if available, otherwise CPU.
      - Handles missing padding tokens safely.
      - Returns only the count of newly generated tokens.
      - Explicitly releases GPU/CPU memory after each model.
    """

    _DEFAULT_PADDING_TOKEN: Final[str] = "<|endoftext|>"

    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        """
        Initialize the runner for a specific model.

        Args:
            model_id: Hugging Face model identifier (e.g., "gpt2").
            max_new_tokens: Maximum number of tokens to generate.
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initialized HuggingFaceRunner for model %s on %s", model_id, self._device)

    def load(self) -> None:
        """Load the model and tokenizer onto the device."""
        if self._model is not None:
            logger.debug("Model %s already loaded; skipping reload.", self.model_id)
            return

        logger.info("Loading tokenizer for %s", self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if tokenizer.pad_token is None:
            logger.warning(
                "Tokenizer for %s has no padding token; setting pad_token to %s",
                self.model_id,
                self._DEFAULT_PADDING_TOKEN,
            )
            tokenizer.pad_token = self._DEFAULT_PADDING_TOKEN

        logger.info("Loading model for %s", self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )
        model = model.to(self._device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        logger.info("Model %s loaded successfully on %s", self.model_id, self._device)

    def unload(self) -> None:
        """Explicitly release model and tokenizer memory."""
        if self._model is None and self._tokenizer is None:
            return

        logger.info("Unloading model %s", self.model_id)
        self._model = None
        self._tokenizer = None

        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        gc.collect()
        logger.debug("Memory cleanup completed for %s", self.model_id)

    def generate(self, prompt: str) -> int:
        """
        Generate text for a single prompt and return new token count.

        Args:
            prompt: Input text prompt.

        Returns:
            Number of newly generated tokens.

        Raises:
            InferenceError: If model or tokenizer is not loaded, or generation fails.
        """
        if self._model is None or self._tokenizer is None:
            raise InferenceError(f"Model {self.model_id} is not loaded. Call load() first.")

        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = inputs.input_ids.to(self._device)
            attention_mask = inputs.attention_mask.to(self._device)
            prompt_length = input_ids.shape[1]

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                )

            generated_length = outputs.shape[1] - prompt_length
            return max(0, generated_length)

        except Exception as error:
            logger.error("Inference failed for %s: %s", self.model_id, error)
            raise InferenceError(f"Inference failed for {self.model_id}: {error}") from error

    @property
    def device(self) -> torch.device:
        """Return the device used for inference."""
        return self._device