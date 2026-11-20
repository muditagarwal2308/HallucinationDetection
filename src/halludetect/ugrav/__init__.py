"""UG-RAV: Uncertainty-Gated Retrieval-Augmented Verification.

Pipeline (Algorithm 1 of the dissertation):

1. closed-book sampling: greedy answer a0 + K samples;
2. uncertainty signals: semantic entropy (SE), predictive entropy, self-consistency, P(True);
3. gate: if the closed-book risk r̂0 < τ_low, accept a0 (no retrieval);
4. otherwise retrieve top-k passages, re-sample with evidence, and compute
   evidence support (NLI), ΔSE = SE_closed − SE_RAG and parametric–evidence agreement;
5. fuse the signals with logistic regression into a risk r̂ and answer from the
   evidence if r̂ < τ_high and the answer is entailed, else abstain.
"""

from .data import WorldItem, load_passages, load_world
from .pipeline import UGRAV

__all__ = ["UGRAV", "WorldItem", "load_world", "load_passages"]
