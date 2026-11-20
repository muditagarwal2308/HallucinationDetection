import numpy as np
import pytest

from halludetect.nli import LexicalNLI
from halludetect.ugrav import UGRAV, load_passages, load_world
from halludetect.ugrav import policy as P
from halludetect.ugrav.benchmark import evaluate, simulate_records
from halludetect.ugrav.data import PassageRetriever
from halludetect.ugrav.fusion import LogisticFusion, out_of_fold
from halludetect.ugrav.simulator import SimulatedQALLM


@pytest.fixture(scope="module")
def world():
    return load_world()


def test_world_dataset(world):
    assert len(world) >= 190
    assert {w.tier for w in world} == {"high", "mid", "low"}
    assert any(w.kind == "ambiguous" for w in world) and any(w.misconception for w in world)


def test_simulator_is_deterministic_and_popularity_sensitive(world):
    a, b = SimulatedQALLM(world, seed=3), SimulatedQALLM(world, seed=3)
    q = world[0].question
    assert [g.text for g in a.answer(q, n=5, temperature=1.0)] == [g.text for g in b.answer(q, n=5, temperature=1.0)]
    know = {t: np.mean([SimulatedQALLM(world, seed=s).closed_belief(w)[0] == "knows"
                        for s in range(5) for w in world if w.tier == t and w.kind != "ambiguous"])
            for t in ("high", "low")}
    assert know["high"] > know["low"] + 0.3


def test_logistic_fusion_learns_and_cross_validates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + 0.2 * rng.normal(size=200) > 0).astype(float)
    m = LogisticFusion().fit(X, y)
    assert ((m.predict_proba(X) > 0.5) == y).mean() > 0.9
    oof = out_of_fold(X, y, [f"q{i}" for i in range(200)])
    assert ((oof > 0.5) == y).mean() > 0.85


def test_algorithm1_routing():
    a = P.PolicyArrays(closed_correct=np.array([1.0, 0, 0]), rag_correct=np.array([1.0, 1, 0]),
                       se=np.zeros(3), se_rag=np.zeros(3), sc=np.ones(3), min_tok=np.ones(3),
                       support_rag=np.array([0.9, 0.9, 0.1]), r0=np.array([0.1, 0.8, 0.8]),
                       r_rag=np.array([0.1, 0.2, 0.2]))
    d = P.ugrav(a, tau_low=0.3, tau_high=0.5, min_support=0.5)
    assert d.source.tolist() == [0, 1, -1]  # accept closed, answer from evidence, abstain (unsupported)
    s = d.summary(a)
    assert s["coverage"] == pytest.approx(2 / 3) and s["hallucination_rate"] == 0.0


def test_live_pipeline_trace(world):
    llm = SimulatedQALLM(world, seed=0)
    pipe = UGRAV(llm, LexicalNLI(), PassageRetriever(load_passages()), k=6)
    item = world[0]
    out = pipe.run(item.question, aliases=item.aliases)
    assert out["decision"]["action"] in {"accept", "answer_from_evidence", "abstain"}
    assert out["steps"][0]["step"] == "closed_book"


def test_end_to_end_evaluation_on_one_seed():
    res = evaluate(simulate_records(seeds=(0,), k=6))
    det = res["detection"]
    assert det["semantic_entropy"]["auroc"] > det["lexical_entropy"]["auroc"]
    assert det["fused_with_evidence"]["auroc"] > det["semantic_entropy"]["auroc"]
    pol = res["policies_test"]
    assert pol["ug_rav"]["hallucination_rate"] < pol["never_retrieve"]["hallucination_rate"]
