"""Hugging Face cross-encoder NLI (e.g. DeBERTa-large-MNLI)."""

from __future__ import annotations

from .base import NLIModel, NLIResult


class HFNLI(NLIModel):
    """Wraps any sequence-classification NLI checkpoint from the Hugging Face Hub.

    Recommended checkpoints:
    ``microsoft/deberta-large-mnli`` (semantic-entropy reference),
    ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`` (stronger),
    ``cross-encoder/nli-deberta-v3-small`` (fast on CPU).
    """

    # The semantic-uncertainty reference implementation clusters with the relaxed rule by default.
    default_strict = False

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli", device: str | None = None,
                 batch_size: int = 16, max_length: int = 512, cache_path=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self._idx = {}
        for i, lab in id2label.items():
            for key in ("entail", "neutral", "contradict"):
                if key in lab:
                    self._idx[key] = i
        if len(self._idx) != 3:
            raise ValueError(f"Cannot map labels of {model_name}: {id2label}")
        self._cache: dict[tuple[str, str], NLIResult] = {}
        self.cache_path = cache_path
        self._dirty = 0
        if cache_path is not None:
            import json
            from pathlib import Path

            p = Path(cache_path)
            if p.exists():
                for prem, hyp, e, n, c in json.loads(p.read_text(encoding="utf-8")):
                    self._cache[(prem, hyp)] = NLIResult(e, n, c)

    def save_cache(self) -> None:
        """Persist predictions so that long benchmark runs can be resumed."""
        if self.cache_path is None or not self._dirty:
            return
        import json
        from pathlib import Path

        rows = [[p, h, r.entailment, r.neutral, r.contradiction] for (p, h), r in self._cache.items()]
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.cache_path).write_text(json.dumps(rows), encoding="utf-8")
        self._dirty = 0

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        todo = [p for p in dict.fromkeys(pairs) if p not in self._cache]
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start: start + self.batch_size]
            enc = self.tokenizer([p for p, _ in chunk], [h for _, h in chunk], truncation="only_first",
                                 max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.model(**enc).logits.softmax(-1).cpu().numpy()
            for pair, pr in zip(chunk, probs):
                self._cache[pair] = NLIResult(
                    entailment=float(pr[self._idx["entail"]]),
                    neutral=float(pr[self._idx["neutral"]]),
                    contradiction=float(pr[self._idx["contradict"]]),
                )
            self._dirty += len(chunk)
            if self._dirty >= 500:
                self.save_cache()
        return [self._cache[p] for p in pairs]
