"""Natural Language Inference (NLI) back-ends.

Two implementations share one interface:

* :class:`~halludetect.nli.lexical.LexicalNLI` — a transparent, dependency-free
  approximation (weighted lexical coverage + contradiction cues). It makes the
  whole project runnable offline, in CI and in the demo app.
* :class:`~halludetect.nli.hf.HFNLI` — a Hugging Face cross-encoder such as
  ``microsoft/deberta-large-mnli`` (the model used for semantic-entropy
  clustering by Kuhn et al., 2023). Requires ``pip install -e .[ml]``.
"""

from __future__ import annotations

from .base import NLIModel, NLIResult
from .lexical import LexicalNLI


def get_nli(name: str = "lexical", **kwargs) -> NLIModel:
    """Factory: ``"lexical"`` or any Hugging Face NLI model id."""
    if name in ("lexical", "offline", "", None):
        return LexicalNLI(**kwargs)
    from .hf import HFNLI

    return HFNLI(model_name=name, **kwargs)


def __getattr__(name: str):
    # ``HFNLI`` needs torch; import it lazily so the offline package stays light.
    if name == "HFNLI":
        from .hf import HFNLI

        return HFNLI
    raise AttributeError(name)


__all__ = ["NLIModel", "NLIResult", "LexicalNLI", "HFNLI", "get_nli"]
