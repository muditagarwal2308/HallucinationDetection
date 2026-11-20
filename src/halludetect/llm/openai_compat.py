"""OpenAI-compatible chat-completions client (Ollama, vLLM, LM Studio, OpenAI, Azure OpenAI)."""

from __future__ import annotations

import math
import os

from . import prompts
from .base import LLM, Generation


class OpenAICompatLLM(LLM):
    """Client for any ``/v1/chat/completions`` endpoint.

    Examples
    --------
    Ollama (local, free)::

        OpenAICompatLLM(model="qwen2.5:7b", base_url="http://localhost:11434/v1")

    OpenAI::

        OpenAICompatLLM(model="gpt-4o-mini", base_url="https://api.openai.com/v1", api_key=os.environ["OPENAI_API_KEY"])

    Token log-probabilities are requested when available (``logprobs=true``);
    endpoints that ignore the flag still work, but predictive entropy and
    P(True) are then reported as unavailable.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", api_key: str | None = None,
                 timeout: float = 120.0, request_logprobs: bool = True):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")
        self.timeout = timeout
        self.request_logprobs = request_logprobs
        self.name = f"openai-compat:{model}"
        self.supports_logprobs = request_logprobs

    def _post(self, payload: dict) -> dict:
        import requests

        resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout,
                             headers={"Authorization": f"Bearer {self.api_key}"})
        resp.raise_for_status()
        return resp.json()

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.0, max_tokens: int = 48,
                 system: str | None = None) -> list[Generation]:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        out: list[Generation] = []
        # Many local servers ignore ``n``; request one completion at a time for portability.
        for i in range(n):
            payload = {"model": self.model, "messages": messages, "temperature": temperature,
                       "max_tokens": max_tokens, "seed": 1234 + i}
            if self.request_logprobs:
                payload["logprobs"] = True
            data = self._post(payload)
            choice = data["choices"][0]
            lps: list[float] = []
            content = (choice.get("logprobs") or {}).get("content") or []
            for tok in content:
                if tok.get("logprob") is not None:
                    lps.append(float(tok["logprob"]))
            out.append(Generation(choice["message"]["content"] or "", lps))
        if not any(g.token_logprobs for g in out):
            self.supports_logprobs = False
        return out

    def p_true(self, question: str, answer: str, samples: list[str] | None = None) -> float | None:
        payload = {
            "model": self.model, "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
            "messages": [{"role": "user", "content": prompts.p_true_prompt(question, answer, samples)}],
        }
        try:
            data = self._post(payload)
        except Exception:
            return None
        content = (data["choices"][0].get("logprobs") or {}).get("content") or []
        if not content:
            text = (data["choices"][0]["message"]["content"] or "").strip().upper()
            return 1.0 if text.startswith("A") else 0.0 if text.startswith("B") else None
        p_a = p_b = 0.0
        for cand in content[0].get("top_logprobs", []):
            tok = cand["token"].strip().upper().strip("(")
            if tok == "A":
                p_a += math.exp(cand["logprob"])
            elif tok == "B":
                p_b += math.exp(cand["logprob"])
        return p_a / (p_a + p_b) if p_a + p_b > 0 else None
