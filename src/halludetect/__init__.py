"""halludetect: hallucination detection and mitigation for LLM question answering.

Two research frameworks are implemented on a shared core:

* **UG-RAV**: Uncertainty-Gated Retrieval-Augmented Verification for factual
  hallucinations in open-domain QA (semantic-entropy gate, NLI evidence
  verification, ΔSE signal, logistic fusion, accept / correct / abstain policy).
* **VGHD**: Verse-Grounded Hallucination Detection for retrieval-augmented QA on
  the Bhagavad Gita (citation existence, claim-verse support, quotation and
  speaker checks, false-premise / out-of-scope detection, repair or abstain).
"""

__version__ = "1.0.0"
