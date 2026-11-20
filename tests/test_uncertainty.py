import math

import pytest

from halludetect.nli import LexicalNLI
from halludetect.text import answer_to_statement, contains_match, normalize_answer
from halludetect.uncertainty import (
    SemanticClusters,
    cluster_answers,
    lexical_entropy,
    max_entropy,
    self_consistency,
    semantic_entropy,
)

NLI = LexicalNLI()


def test_paraphrases_form_one_cluster_and_zero_entropy():
    answers = ["Canberra"] * 6 + ["It is Canberra"] * 3 + ["Canberra."]
    cl = cluster_answers(answers, NLI, "What is the capital of Australia?")
    assert cl.n_clusters == 1
    assert semantic_entropy(cl) == pytest.approx(0.0)
    assert lexical_entropy(answers) > 0.5  # string-level entropy overstates uncertainty


def test_semantic_entropy_worked_example_from_notes():
    # four meaning clusters of sizes 4, 3, 2, 1 out of K = 10 → 1.28 nats
    cl = SemanticClusters(answers=[str(i) for i in range(10)], assignments=[0] * 4 + [1] * 3 + [2] * 2 + [3])
    assert semantic_entropy(cl) == pytest.approx(1.2799, abs=1e-3)
    assert max_entropy(10) == pytest.approx(math.log(10))
    assert self_consistency(cl) == pytest.approx(0.4)


def test_weighted_semantic_entropy_uses_probability_mass():
    cl = SemanticClusters(answers=["a", "b"], assignments=[0, 1])
    assert semantic_entropy(cl, logprobs=[0.0, 0.0]) == pytest.approx(math.log(2))
    assert semantic_entropy(cl, logprobs=[0.0, -20.0]) < 1e-6


def test_distinct_people_are_not_clustered():
    cl = cluster_answers(["Premchand", "Munshi Premchand", "Rabindranath Tagore"], NLI, "Who wrote Godan?")
    assert cl.assignments[0] == cl.assignments[1] != cl.assignments[2]


def test_answer_matching_and_statements():
    assert normalize_answer("The Taj  Mahal!") == "taj mahal"
    assert contains_match("It was written by Munshi Premchand.", ["Premchand"])
    assert not contains_match("Premchandra", ["Premchand"])
    assert answer_to_statement("Who wrote Godan?", "Premchand") == "Premchand wrote Godan."
    assert answer_to_statement("What is the capital of Australia?", "Canberra") == \
        "The capital of Australia is Canberra."
