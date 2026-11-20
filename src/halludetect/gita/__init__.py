"""Verse-Grounded Hallucination Detection (VGHD) for retrieval-augmented QA on the Bhagavad Gita.

A scripture is a closed corpus with a fixed reference system (BG chapter.verse),
so an answer can be verified *verse by verse* rather than as a whole:

1. every cited verse must exist (fabricated citation);
2. every cited verse must support the claim attributed to it (misattribution);
3. every quotation must match the text (misquotation);
4. every attributed speaker must be the verse's speaker (wrong speaker);
5. the answer must not accept a false premise or answer what the text does not
   address (false premise / out of scope);

and the detected problems drive a repair-or-abstain policy.
"""

from .corpus import GitaCorpus, Verse
from .verifier import VerificationReport, VerseVerifier

__all__ = ["GitaCorpus", "Verse", "VerseVerifier", "VerificationReport"]
