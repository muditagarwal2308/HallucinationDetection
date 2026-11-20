"""Local Hugging Face causal language model with token log-probabilities."""

from __future__ import annotations

from . import prompts
from .base import LLM, Generation


class HFLocalLLM(LLM):
    """Runs an instruction-tuned causal LM locally (CPU or GPU).

    Uses the model's chat template, samples ``n`` completions in one batch and
    returns the log-probability of every generated token via
    ``model.compute_transition_scores`` — exactly what predictive entropy and the
    weighted semantic-entropy estimator need. ``load_in_4bit=True`` enables
    bitsandbytes quantisation on CUDA (as planned for the Kaggle runs).
    """

    supports_logprobs = True

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str | None = None,
                 dtype: str = "auto", load_in_4bit: bool = False, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = f"hf:{model_name}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs: dict = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = (torch.float16 if self.device == "cuda" else torch.float32) if dtype == "auto" \
                else getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not load_in_4bit:
            self.model.to(self.device)
        self.model.eval()
        self.seed = seed
        self._calls = 0

    def _encode(self, prompt: str, system: str | None):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.0, max_tokens: int = 48,
                 system: str | None = None) -> list[Generation]:
        """Sample ``n`` completions sharing a single prompt prefill.

        ``model.generate(num_return_sequences=n)`` re-encodes the prompt n times; on a CPU
        that dominates the cost of retrieval-augmented prompts. Here the prompt is encoded
        once, its key/value cache is replicated n times, and tokens are decoded in a batch.
        Log-probabilities are those of the sampling distribution (temperature-scaled).
        """
        torch = self.torch
        enc = self._encode(prompt, system)
        self._calls += 1
        gen = torch.Generator(device="cpu").manual_seed(self.seed * 100003 + self._calls)
        eos = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        eos |= set(getattr(self.model.generation_config, "eos_token_id", None) or []) \
            if isinstance(getattr(self.model.generation_config, "eos_token_id", None), list) else set()
        with torch.no_grad():
            out = self.model(**enc, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1, :].float()
            if n > 1:
                cache.batch_repeat_interleave(n)
                logits = logits.expand(n, -1)
            attn = enc["attention_mask"].repeat(n, 1)
            tokens = [[] for _ in range(n)]
            lps = [[] for _ in range(n)]
            done = torch.zeros(n, dtype=torch.bool)
            for _ in range(max_tokens):
                if temperature > 0:
                    logp = torch.log_softmax(logits / temperature, dim=-1)
                    nxt = torch.multinomial(logp.exp().cpu(), 1, generator=gen).squeeze(1).to(logits.device)
                else:
                    logp = torch.log_softmax(logits, dim=-1)
                    nxt = logp.argmax(-1)
                chosen = logp.gather(1, nxt[:, None]).squeeze(1)
                for i in range(n):
                    if done[i]:
                        continue
                    t = int(nxt[i])
                    if t in eos:
                        done[i] = True
                        continue
                    tokens[i].append(t)
                    lps[i].append(float(chosen[i]))
                    if "\n" in self.tokenizer.decode([t]) and tokens[i][:-1]:
                        done[i] = True  # short answers end at the first line break
                if bool(done.all()):
                    break
                attn = torch.cat([attn, torch.ones((n, 1), dtype=attn.dtype, device=attn.device)], dim=1)
                out = self.model(input_ids=nxt[:, None], attention_mask=attn, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits = out.logits[:, -1, :].float()
        return [Generation(self.tokenizer.decode(t, skip_special_tokens=True), lp) for t, lp in zip(tokens, lps)]

    def answer(self, question: str, passages: list[str] | None = None, n: int = 1,
               temperature: float = 0.0, max_tokens: int = 16) -> list[Generation]:
        """Short answers need few tokens; capping them keeps CPU runs affordable."""
        return super().answer(question, passages, n=n, temperature=temperature, max_tokens=max_tokens)

    def p_true(self, question: str, answer: str, samples: list[str] | None = None) -> float | None:
        torch = self.torch
        enc = self._encode(prompts.p_true_prompt(question, answer, samples), None)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1]
        probs = logits.float().softmax(-1)
        ids_a = {self.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("A", " A")}
        ids_b = {self.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("B", " B")}
        p_a = float(sum(probs[i] for i in ids_a))
        p_b = float(sum(probs[i] for i in ids_b))
        return p_a / (p_a + p_b) if p_a + p_b > 0 else None
