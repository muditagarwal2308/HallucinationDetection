import pytest

from halludetect.gita import GitaCorpus, VerseVerifier
from halludetect.gita.citations import extract_citations
from halludetect.gita.claims import segment_claims
from halludetect.gita.pipeline import GitaQA
from halludetect.gita.policy import RepairPolicy

SOUL_Q = "Which verse of the Bhagavad Gita says that the soul cannot be cut by weapons?"


@pytest.fixture(scope="module")
def corpus():
    return GitaCorpus.load()


@pytest.fixture(scope="module")
def verifier(corpus):
    return VerseVerifier(corpus)


def test_corpus_size_and_edition_mapping(corpus):
    assert len(corpus) == 701
    assert corpus.chapter_lengths[2] == 72 and corpus.chapter_lengths[18] == 78
    assert sum(corpus.chapter_lengths_700.values()) == 700
    assert corpus.get("13.1").ref_700 is None  # Arjuna's question, unnumbered in the 700-verse recension
    assert corpus.get_700(13, 1).ref == "13.2"
    assert corpus.get("2.47").speaker == "Krishna" and corpus.get("1.1").speaker == "Dhritarashtra"
    assert "karmaṇy" in corpus.get("2.47").iast


@pytest.mark.parametrize("text,expected", [
    ("as stated in BG 2.47.", [(2, 47)]),
    ("Bhagavad Gita 18:66 says", [(18, 66)]),
    ("see Gita 2.22-24", [(2, 22), (2, 23), (2, 24)]),
    ("Chapter 2, verse 23", [(2, 23)]),
    ("verse 5 of chapter 6", [(6, 5)]),
    ("Chapter II, verse 20", [(2, 20)]),
    ("(4.7)", [(4, 7)]),
])
def test_citation_extraction(text, expected):
    refs = [r for c in extract_citations(text) for r in c.refs]
    assert refs == expected


def test_pointer_sentence_attaches_citation_to_previous_claim():
    claims = segment_claims("The Gita states that the soul cannot be cut. This is described in BG 2.20.")
    assert len(claims) == 1
    assert claims[0].citations[0].start == 20


def test_retrieval_multilingual(corpus):
    assert corpus.search("weapons cannot cut the soul", 3)[0][0].ref == "2.23"
    assert corpus.search("कर्म और उसके फल के बारे में कृष्ण क्या कहते हैं?", 3)[0][0].ref == "2.47"
    assert corpus.search("karmanye vadhikaraste ma phaleshu", 3)[0][0].ref == "2.47"


def test_motivating_example_misattribution_is_detected_and_repaired(corpus, verifier):
    wrong = "The Gita states that the soul cannot be cut by weapons or burnt by fire. This is described in BG 2.20."
    rep = verifier.verify(SOUL_Q, wrong)
    assert rep.types == {"misattributed_citation"}
    assert rep.claims[0].best_ref == "2.23"
    decision = RepairPolicy(corpus).decide(rep)
    assert decision.action == "repair" and "BG 2.23" in decision.answer
    assert verifier.verify(SOUL_Q, decision.answer).verdict == "faithful"


def test_fabricated_wrong_speaker_and_misquote(verifier):
    assert "fabricated_citation" in verifier.verify("What does Krishna say about action?",
                                                    "Krishna says to act without attachment (BG 2.80).").types
    rep = verifier.verify("What does Krishna say about action and its fruits?",
                          "Arjuna says that one's concern is with action alone, never with its fruits (BG 2.47).")
    assert "wrong_speaker" in rep.types
    rep = verifier.verify("Quote the verse on surrender.",
                          'Krishna says, "Abandoning all wealth come unto Me alone for gold" (BG 18.66).')
    assert "misquotation" in rep.types


def test_premise_checks(verifier):
    rep = verifier.verify("What does the Gita say about smartphones?", "Use smartphones with detachment (BG 2.47).")
    assert "out_of_scope_answered" in rep.types
    rep = verifier.verify("What does BG 2.80 say about meditation?",
                          "There is no BG 2.80; chapter 2 has only 72 verses.")
    assert rep.verdict == "correct_abstention"


def test_simulated_rag_pipeline_catches_injected_errors():
    qa = GitaQA(simulator_mode="fabricated_citation")
    out = qa.ask("What does Krishna say about action and its fruits?")
    assert out["report"]["hallucinated"] and "fabricated_citation" in out["report"]["types"]
    out = qa.ask("What does Krishna say about action and its fruits?", simulator_mode="none")
    assert out["report"]["verdict"] == "faithful" and out["decision"]["action"] == "accept"
