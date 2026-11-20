"""LLM back-ends behind one interface.

* :class:`~halludetect.llm.openai_compat.OpenAICompatLLM` — any OpenAI-compatible
  chat endpoint (Ollama, vLLM, LM Studio, llama.cpp server, OpenAI, Azure OpenAI).
* :class:`~halludetect.llm.hf_local.HFLocalLLM` — a local Hugging Face causal LM
  (e.g. ``Qwen/Qwen2.5-1.5B-Instruct``); exposes token log-probabilities.
* Simulators (``halludetect.ugrav.simulator`` and ``halludetect.gita.simulator``)
  implement the same interface for fully offline, reproducible demos.
"""

from __future__ import annotations

from .base import LLM, Generation

__all__ = ["LLM", "Generation"]
