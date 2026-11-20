"""Smoke and interaction tests for the Streamlit app (``app/``), using Streamlit's AppTest harness.

Every page must render without an exception using only the offline components (lexical NLI,
simulators, bundled data and result files); key buttons are clicked to exercise the live pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = sorted((APP / "pages").glob("[0-9]_*.py"))
TIMEOUT = 240


def _page(number: int) -> Path:
    return next(p for p in PAGES if p.name.startswith(f"{number}_"))


def _run(path: Path) -> AppTest:
    at = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    at.run()
    return at


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _html(at: AppTest) -> str:
    return "\n".join(el.proto.body for el in at.get("html"))


def test_eight_pages_are_present():
    labels = [p.stem.split("_", 2)[-1] for p in PAGES]
    assert labels == ["Hallucination_Explorer", "Semantic_Entropy_Lab", "UG-RAV_Live", "UG-RAV_Results",
                      "Gita_Verse_Verifier", "Gita_RAG_Assistant", "VGHD_Results", "Corpus_Browser"]


@pytest.mark.parametrize("page", [APP / "Home.py", *PAGES], ids=lambda p: p.stem.split("_", 2)[-1])
def test_page_renders_without_exception(page: Path):
    _assert_clean(_run(page))


def test_navigation_reaches_every_page():
    at = _run(APP / "Home.py")
    _assert_clean(at)
    assert "Hallucination Detection Studio" in _html(at)
    for page in PAGES:
        at.switch_page(f"pages/{page.name}").run()
        _assert_clean(at)


def test_explorer_quiz_explains_the_answer():
    at = _run(_page(1))
    at.radio(key="quiz_guess_0").set_value("Factual inconsistency").run()
    _assert_clean(at)
    assert at.success and "Correct" in at.success[0].value


def test_semantic_entropy_lab_presets_and_sampling():
    at = _run(_page(2))
    _assert_clean(at)
    assert ">0.00<" in _html(at)  # paraphrases: one meaning, zero semantic entropy

    at.segmented_control(key="se_preset").set_value("confabulation").run()
    _assert_clean(at)
    assert ">1.28<" in _html(at)  # shares 4/3/2/1 → SE ≈ 1.28 nats

    at.segmented_control(key="se_preset").set_value("godan").run()
    _assert_clean(at)
    assert ">0.50<" in _html(at)

    before = at.text_area(key="se_answers").value
    at.button(key="se_sample").click().run()
    _assert_clean(at)
    sampled = at.text_area(key="se_answers").value
    assert sampled != before and len(sampled.splitlines()) == 10


def test_ugrav_live_single_run_and_batch():
    at = _run(_page(3))
    _assert_clean(at)
    at.button(key="ug_run").click().run()
    _assert_clean(at)
    page = _html(at)
    assert "Closed-book sampling" in page and "Decision" in page
    assert "Answer from evidence" in page  # Godan, seed 0: confused model, evidence corrects it

    at.button(key="ug_batch").click().run()
    _assert_clean(at)
    assert at.dataframe and at.dataframe[0].value.shape[0] == 25


def test_ugrav_live_accepts_confident_answer_without_retrieval():
    at = _run(_page(3))
    at.selectbox(key="ug_item").set_value("w052").run()  # capital of Australia
    at.button(key="ug_run").click().run()
    _assert_clean(at)
    assert "Closed-book sampling" in _html(at)


def test_verifier_flags_repairs_and_accepts():
    at = _run(_page(5))
    _assert_clean(at)
    page = _html(at)
    assert "Hallucinated" in page and "Misattributed citation" in page
    assert "<ins>2.23.</ins>" in page  # repaired citation BG 2.20 → BG 2.23

    at.pills(key="v_preset").set_value("corrected").run()
    _assert_clean(at)
    assert ">Faithful<" in _html(at)

    at.text_area(key="va").input("The Gita says the soul cannot be cut by weapons (BG 2.80).")
    at.button(key="v_submit").click().run()
    _assert_clean(at)
    assert "Fabricated citation" in _html(at)

    at.pills(key="v_preset").set_value("premise").run()
    _assert_clean(at)
    assert "False premise accepted" in _html(at)


def test_rag_assistant_catches_injected_hallucination():
    at = _run(_page(6))
    _assert_clean(at)
    at.selectbox(key="rag_inject").set_value("fabricated_citation").run()
    at.button(key="rag_ask").click().run()
    _assert_clean(at)
    assert at.success and any("Caught" in s.value for s in at.success)

    at.button(key="rag_ex_3").click().run()  # Hindi example
    at.selectbox(key="rag_inject").set_value("none").run()
    at.button(key="rag_ask").click().run()
    _assert_clean(at)
    assert "BG 2.47" in _html(at)


def test_vghd_results_switches_heatmap_metric():
    at = _run(_page(7))
    at.radio(key="vr_heat_metric").set_value("type_accuracy").run()
    _assert_clean(at)


def test_corpus_browser_navigation_and_search():
    at = _run(_page(8))
    at.button(key="cb_next").click().run()
    _assert_clean(at)
    assert at.slider(key="cb_v_1").value == 2
    at.button(key="cb_ex_0").click().run()
    _assert_clean(at)
    assert "BG 2.23" in _html(at)


def test_common_helpers():
    sys.path.insert(0, str(APP))
    import common

    left, right = common.diff_html("cut by weapons (BG 2.20).", "cut by weapons (BG 2.23).")
    assert "<del>2.20).</del>" in left and "<ins>2.23).</ins>" in right
    assert common.fmt(-0.0, 2) == "0.00" and common.fmt(None) == "—"
    assert common.pct(0.125) == "12.5%"
