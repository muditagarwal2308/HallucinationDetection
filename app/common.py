"""Shared building blocks of the Hallucination Detection Studio (Streamlit app).

Everything that more than one page needs lives here: page set-up and styling, cached
loaders for the corpora, models and result files, human-readable labels, and small
HTML components (badges, chips, verse cards, verdict banners, diffs, gauges).
"""

from __future__ import annotations

import html
import json
import math
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

try:  # running from a checkout without ``pip install -e .``
    import halludetect  # noqa: F401
except ImportError:  # pragma: no cover - depends on the environment
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from halludetect.paths import DATA, RESULTS, ROOT  # noqa: E402

APP_NAME = "Hallucination Detection Studio"
REPO_URL = "https://github.com/muditagarwal2308/HallucinationDetection"
APP_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- palette
INDIGO = "#4338CA"
INDIGO_DARK = "#1E1B4B"
INDIGO_SOFT = "#E0E7FF"
SAFFRON = "#F59E0B"
SAFFRON_DARK = "#B45309"
TEAL = "#0D9488"
ROSE = "#DB2777"
VIOLET = "#7C3AED"
GREEN = "#16A34A"
RED = "#DC2626"
AMBER = "#D97706"
SLATE = "#64748B"
GRID = "#ECEAF6"

#: Solid colours for semantic clusters (bars, chip borders) and their light chip backgrounds.
CLUSTER_SOLID = ["#4F46E5", "#F59E0B", "#14B8A6", "#EC4899", "#8B5CF6", "#22C55E", "#F43F5E", "#0EA5E9",
                 "#64748B", "#F97316", "#A3A3A3", "#84CC16"]
CLUSTER_LIGHT = ["#E0E7FF", "#FEF3C7", "#CCFBF1", "#FCE7F3", "#EDE9FE", "#DCFCE7", "#FFE4E6", "#E0F2FE",
                 "#F1F5F9", "#FFEDD5", "#F5F5F5", "#ECFCCB"]
SPEAKER_COLORS = {"Krishna": INDIGO, "Arjuna": SAFFRON, "Sanjaya": TEAL, "Dhritarashtra": ROSE}

# ---------------------------------------------------------------- labels
SIGNAL_LABELS = {
    "semantic_entropy": "Semantic entropy",
    "semantic_entropy_weighted": "Semantic entropy (weighted)",
    "predictive_entropy": "Predictive entropy",
    "length_normalised_pe": "Length-normalised PE",
    "lexical_entropy": "Lexical entropy",
    "one_minus_self_consistency": "1 − self-consistency",
    "one_minus_greedy_agreement": "1 − greedy agreement",
    "one_minus_p_true": "1 − P(True)",
    "min_token_surprisal": "Min-token surprisal (FLARE)",
    "one_minus_evidence_support": "1 − evidence support",
    "fused_closed_book": "Fused closed-book risk r̂₀",
    "fused_with_evidence": "Fused risk with evidence r̂",
    "fused_closed": "Fused closed-book risk r̂₀",
    "fused_full": "Fused risk with evidence r̂",
}
POLICY_LABELS = {
    "never_retrieve": "Never retrieve (closed-book)",
    "always_retrieve": "Always retrieve (vanilla RAG)",
    "sc_abstention": "Self-consistency abstention",
    "flare_style": "FLARE-style gate",
    "se_gated_no_verification": "SE-gated RAG, no verification",
    "ug_rav": "UG-RAV (proposed)",
    "ug_rav_without_uncertainty_shift": "UG-RAV without ΔSE signals",
    "ug_rav_without_evidence_signals": "UG-RAV without evidence signals",
    "oracle": "Oracle (upper bound)",
}
FEATURE_LABELS = {
    "se": "Semantic entropy",
    "one_minus_sc": "1 − self-consistency",
    "one_minus_agreement": "1 − greedy agreement",
    "one_minus_p_true": "1 − P(True)",
    "neg_log_min_tok": "−log min-token prob.",
    "ln_pe": "Length-normalised PE",
    "se_rag": "SE with evidence",
    "delta_se": "ΔSE (closed − RAG)",
    "support_rag": "Evidence support of RAG answer",
    "contradiction_rag": "Evidence contradiction of RAG answer",
    "support_closed": "Evidence support of closed answer",
    "contradiction_closed": "Evidence contradiction of closed answer",
    "agreement": "Closed ↔ RAG agreement",
    "one_minus_sc_rag": "1 − self-consistency (RAG)",
}
GATE_LABELS = {
    "fused_closed_book_risk": "Fused closed-book risk r̂₀",
    "semantic_entropy": "Semantic entropy",
    "flare_min_token_prob": "FLARE min-token probability",
    "random": "Random gate",
}
ISSUE_LABELS = {
    "fabricated_citation": "Fabricated citation",
    "misattributed_citation": "Misattributed citation",
    "unsupported_claim": "Unsupported claim",
    "misquotation": "Misquotation",
    "wrong_speaker": "Wrong speaker",
    "false_premise_accepted": "False premise accepted",
    "out_of_scope_answered": "Out-of-scope question answered",
    "imprecise_citation": "Imprecise citation",
    "uncited_claim": "Uncited claim",
    "correct": "Correct answer",
    "correct_refusal": "Correct refusal",
    "none": "No injection",
    "random": "Random type",
}
DETECTOR_LABELS = {
    "citation_existence": "Citation-existence check",
    "answer_level_nli": "Answer-level NLI (RAGAS-style)",
    "vghd": "VGHD (proposed)",
    "vghd_no_premise": "VGHD without premise check",
    "vghd_no_quotes": "VGHD without quotation check",
    "vghd_no_speaker": "VGHD without speaker check",
}
SIM_STATES = {
    "knows": ("Knows the fact", "Most probability mass is on the gold answer; samples agree."),
    "confident_wrong": ("Confidently wrong", "A wrong answer (often a common misconception) is firmly memorised: "
                                             "samples agree, so semantic entropy stays low — the blind spot of "
                                             "uncertainty-only detection."),
    "confused": ("Confabulating", "The model does not know; mass is spread over plausible distractors, so semantic "
                                  "entropy is high."),
    "ambiguous": ("Ambiguous question", "Many answers are valid; high entropy without error (an aleatoric false "
                                        "alarm)."),
    "ambiguous_grounded": ("Ambiguous, grounded", "The passages name several valid answers and the model picks "
                                                  "among them."),
    "grounded": ("Grounded in evidence", "The gold answer is in the passages and the model adopts it."),
    "knowledge_conflict": ("Knowledge conflict", "The gold answer is in the passages but the model partly sticks "
                                                 "to its confident wrong belief."),
    "ignored_evidence": ("Ignored the evidence", "The gold answer is in the passages but the model keeps its prior "
                                                 "belief."),
    "misled": ("Misled by retrieval", "Only a wrong candidate appears in the passages and the model adopts it "
                                      "(an external hallucination)."),
    "unhelpful_evidence": ("Unhelpful evidence", "The passages contain no candidate answer; the model becomes less "
                                                 "certain."),
}
ACTION_STYLES = {
    # action: (label, icon, background, foreground, border)
    "accept": ("Accept", "✅", "#ECFDF5", "#065F46", "#A7F3D0"),
    "answer_from_evidence": ("Answer from evidence", "📚", "#EEF2FF", "#3730A3", "#C7D2FE"),
    "repair": ("Repair", "🛠️", "#FFFBEB", "#92400E", "#FDE68A"),
    "regenerate": ("Regenerate", "🔁", "#F5F3FF", "#5B21B6", "#DDD6FE"),
    "abstain": ("Abstain", "🛑", "#F8FAFC", "#334155", "#CBD5E1"),
}
VERDICT_STYLES = {
    # verdict: (label, icon, background, foreground, border, explanation)
    "faithful": ("Faithful", "✅", "#ECFDF5", "#065F46", "#A7F3D0",
                 "Every claim is supported by the verse it cites."),
    "hallucinated": ("Hallucinated", "⚠️", "#FEF2F2", "#991B1B", "#FECACA",
                     "At least one critical or major verse-level error was found."),
    "correct_abstention": ("Correct abstention", "🛡️", "#EEF2FF", "#3730A3", "#C7D2FE",
                           "The question is problematic and the answer rightly declines or corrects it."),
    "over_abstention": ("Over-abstention", "🤐", "#FFFBEB", "#92400E", "#FDE68A",
                        "The answer refuses although the Gita does address the question."),
}
SEVERITY_STYLES = {
    "critical": ("Critical", "#FEE2E2", "#991B1B", "#DC2626"),
    "major": ("Major", "#FFEDD5", "#9A3412", "#EA580C"),
    "minor": ("Minor", "#FEF9C3", "#854D0E", "#CA8A04"),
}
PREMISE_LABELS = {
    "ok": ("No problem detected", GREEN),
    "nonexistent_reference": ("Non-existent reference", RED),
    "out_of_scope": ("Out of scope", RED),
    "false_premise": ("False premise", RED),
}
NLI_CHOICES = {
    "lexical": "Lexical NLI (offline, instant)",
    "microsoft/deberta-large-mnli": "DeBERTa-large-MNLI (~1.6 GB, slow)",
}
NLI_HELP = ("The lexical NLI runs offline and instantly. DeBERTa-large-MNLI is downloaded from Hugging Face on first "
            "use (~1.6 GB, needs torch + transformers) and is slow on CPU; if it cannot be loaded the app falls "
            "back to the lexical NLI.")


def label(name: str, table: dict[str, str]) -> str:
    """Human-readable label with a tidy fallback for unknown keys."""
    return table.get(name, name.replace("_", " ").capitalize())


# ---------------------------------------------------------------- formatting
def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt(x, digits: int = 3) -> str:
    """Number with fixed decimals; '—' for missing values; never prints −0.000."""
    if is_missing(x):
        return "—"
    v = float(x)
    if abs(v) < 0.5 * 10 ** (-digits):
        v = 0.0
    return f"{v:.{digits}f}"


def pct(x, digits: int = 1) -> str:
    return "—" if is_missing(x) else f"{100 * float(x):.{digits}f}%"


def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


# ---------------------------------------------------------------- page set-up
_CSS = """
<style>
.block-container {padding-top: 2.4rem; padding-bottom: 3.5rem; max-width: 1320px;}
.hd-eyebrow {font-size: .76rem; letter-spacing: .14em; text-transform: uppercase; color: #B45309; font-weight: 700;
  margin-bottom: .2rem;}
.hd-title {font-size: 2.15rem; font-weight: 800; color: #1E1B4B; line-height: 1.15; margin: 0 0 .4rem 0;}
.hd-subtitle {font-size: 1.04rem; color: #4B5563; max-width: 64rem; line-height: 1.55;}
.hd-rule {height: 4px; width: 84px; border-radius: 4px; background: linear-gradient(90deg, #4338CA, #F59E0B);
  margin: .7rem 0 .2rem 0;}
.hd-chip {display: inline-block; padding: .2rem .65rem; border-radius: 999px; font-size: .86rem; font-weight: 600;
  margin: 0 .35rem .4rem 0; line-height: 1.45; border: 1px solid transparent; white-space: normal;}
.hd-chip small {font-weight: 800; opacity: .75; margin-right: .3rem; font-size: .72rem;}
.hd-badge {display: inline-block; padding: .1rem .55rem; border-radius: .45rem; font-size: .78rem; font-weight: 700;
  letter-spacing: .02em; line-height: 1.5; vertical-align: middle; border: 1px solid transparent;}
.hd-card {border: 1px solid #E4E2F3; border-radius: .9rem; padding: 1rem 1.15rem; background: #FFFFFF;
  box-shadow: 0 1px 2px rgba(30, 27, 75, .05); margin-bottom: .8rem;}
.hd-card h4 {margin: 0 0 .35rem 0; color: #1E1B4B; font-size: 1.05rem;}
.hd-metric {border: 1px solid #E4E2F3; border-radius: .85rem; padding: .8rem 1rem .85rem 1rem; background: #FFFFFF;
  box-shadow: 0 1px 2px rgba(30, 27, 75, .04); height: 100%; box-sizing: border-box;}
.hd-metric .hd-metric-value {font-size: 1.7rem; font-weight: 800; color: #1E1B4B; line-height: 1.2;}
.hd-metric .hd-metric-label {font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: #4338CA;
  font-weight: 700; margin-top: .3rem; line-height: 1.35;}
.hd-metric .hd-metric-sub {font-size: .84rem; color: #6B7280; margin-top: .2rem; line-height: 1.4;}
.hd-metric-grid {display: grid; gap: .75rem; align-items: stretch; margin-bottom: .4rem;}
@media (max-width: 900px) {.hd-metric-grid {grid-template-columns: repeat(2, minmax(0, 1fr)) !important;}}
.hd-verse {border: 1px solid #F1E3C2; border-left: 5px solid #F59E0B; border-radius: .8rem; padding: .85rem 1.05rem;
  background: #FFFCF5; margin-bottom: .8rem;}
.hd-verse.hd-verse-bad {border-color: #FECACA; border-left-color: #DC2626; background: #FFF7F7;}
.hd-verse-head {display: flex; align-items: center; gap: .45rem; flex-wrap: wrap; margin-bottom: .45rem;}
.hd-ref {font-weight: 800; color: #1E1B4B; font-size: 1.02rem; margin-right: .15rem;}
.hd-sanskrit {font-family: "Noto Sans Devanagari", "Noto Serif Devanagari", "Nirmala UI", "Mangal",
  "Kohinoor Devanagari", "Devanagari Sangam MN", "Lohit Devanagari", sans-serif; font-size: 1.14rem;
  line-height: 1.95; color: #4C1D95;}
.hd-iast {font-style: italic; color: #6B7280; font-size: .93rem; margin: .2rem 0 .5rem 0; line-height: 1.5;}
.hd-english {color: #111827; font-size: 1.0rem; line-height: 1.6;}
.hd-note {margin-top: .55rem; font-size: .88rem; color: #4B5563;}
.hd-banner {border-radius: 1rem; padding: 1rem 1.25rem; display: flex; gap: 1rem; align-items: center;
  border: 1px solid; margin-bottom: .9rem;}
.hd-banner-icon {font-size: 2.1rem; line-height: 1;}
.hd-banner-title {font-weight: 800; font-size: 1.3rem; line-height: 1.2;}
.hd-banner-text {font-size: .97rem; opacity: .92; margin-top: .15rem;}
.hd-banner-side {margin-left: auto; text-align: right; font-size: .85rem;}
.hd-banner-side b {font-size: 1.5rem; display: block; line-height: 1.1;}
.hd-step {display: flex; align-items: center; gap: .65rem; margin: .1rem 0 .55rem 0;}
.hd-step-num {min-width: 2.05rem; height: 2.05rem; border-radius: 999px; background: #4338CA; color: #FFFFFF;
  font-weight: 800; display: flex; align-items: center; justify-content: center; font-size: 1rem;}
.hd-step-num.hd-step-done {background: #F59E0B;}
.hd-step-title {font-size: 1.16rem; font-weight: 800; color: #1E1B4B;}
.hd-step-sub {font-size: .9rem; color: #6B7280;}
.hd-answer {font-size: 1.04rem; padding: .65rem .95rem; border-radius: .65rem; background: #F4F3FB;
  border: 1px solid #E4E2F3; line-height: 1.6; color: #111827;}
.hd-diff del {background: #FEE2E2; color: #991B1B; text-decoration: line-through; border-radius: .25rem;
  padding: 0 .15rem;}
.hd-diff ins {background: #DCFCE7; color: #14532D; text-decoration: none; border-radius: .25rem; padding: 0 .15rem;
  font-weight: 700;}
.hd-issue {border-left: 5px solid; border-radius: .6rem; padding: .6rem .85rem; margin-bottom: .6rem;
  background: #FCFCFF; border-top: 1px solid #EEECF8; border-right: 1px solid #EEECF8;
  border-bottom: 1px solid #EEECF8;}
.hd-issue-title {font-weight: 750; color: #1E1B4B; margin-left: .35rem;}
.hd-issue-msg {font-size: .95rem; color: #1F2937; margin-top: .3rem;}
.hd-issue-fix {font-size: .88rem; color: #065F46; margin-top: .25rem;}
.hd-bar {height: .5rem; border-radius: 999px; background: #EEF0F6; overflow: hidden; margin: .15rem 0 .35rem 0;}
.hd-bar > span {display: block; height: 100%; border-radius: 999px;}
.hd-muted {color: #6B7280; font-size: .88rem;}
.hd-kv {display: grid; grid-template-columns: max-content 1fr; gap: .25rem .9rem; font-size: .93rem;}
.hd-kv dt {color: #6B7280; font-weight: 600;}
.hd-kv dd {margin: 0; color: #111827;}
.hd-callout {border-radius: .85rem; padding: .85rem 1.05rem; background: #FFF8EB; border: 1px solid #FCE3B4;
  color: #3F2A06; font-size: .95rem; line-height: 1.55; margin: .3rem 0 .9rem 0;}
.hd-callout.hd-callout-indigo {background: #F3F4FF; border-color: #D9DCFB; color: #1E1B4B;}
.hd-pipeline {display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin: .2rem 0 .6rem 0;}
.hd-node {padding: .35rem .75rem; border-radius: .6rem; background: #F4F3FB; border: 1px solid #E4E2F3;
  font-size: .88rem; font-weight: 650; color: #1E1B4B;}
.hd-arrow {color: #9CA3AF; font-weight: 800;}
.hd-legend {font-size: .82rem; color: #6B7280; margin-top: .25rem;}
.hd-footer {margin-top: 2.5rem; padding-top: .8rem; border-top: 1px solid #ECEAF6; font-size: .84rem;
  color: #6B7280;}
.hd-table {width: 100%; border-collapse: separate; border-spacing: 0; font-size: .92rem; border: 1px solid #E4E2F3;
  border-radius: .8rem; overflow: hidden; margin-bottom: .6rem;}
.hd-table th {background: #F4F3FB; color: #3730A3; text-align: left; font-weight: 700; padding: .55rem .8rem;
  border-bottom: 1px solid #E4E2F3; font-size: .76rem; text-transform: uppercase; letter-spacing: .05em;}
.hd-table td {padding: .62rem .8rem; vertical-align: top; border-bottom: 1px solid #F0EEF8; color: #1F2937;
  line-height: 1.5;}
.hd-table tr:last-child td {border-bottom: none;}
.hd-table tr.hd-row-hl td {background: #FFFBEB;}
.hd-table td.hd-strong {font-weight: 700; color: #1E1B4B;}
.hd-chart-title {font-weight: 750; color: #1E1B4B; font-size: 1rem; margin: .2rem 0 -.2rem 0;}
</style>
"""


@dataclass
class PageContext:
    """Handles returned by :func:`setup_page`; ``sidebar`` sits above the sidebar's About box."""

    sidebar: st.delta_generator.DeltaGenerator


def setup_page(title: str, icon: str, subtitle: str, eyebrow: str = "") -> PageContext:
    """Configure the page, inject the stylesheet, draw the header and the sidebar About box."""
    page_title = title if title == APP_NAME else f"{title} · {APP_NAME}"
    st.set_page_config(page_title=page_title, page_icon=icon, layout="wide")
    st.html(_CSS)
    settings = st.sidebar.container()
    with st.sidebar:
        _sidebar_about()
    eyebrow_html = f'<div class="hd-eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    st.html(f'{eyebrow_html}<div class="hd-title">{esc(icon)} {esc(title)}</div>'
            f'<div class="hd-subtitle">{subtitle}</div><div class="hd-rule"></div>')
    return PageContext(sidebar=settings)


def _sidebar_about() -> None:
    st.divider()
    st.markdown("**About**")
    st.caption(
        "Hallucination detection and mitigation for LLM question answering — **UG-RAV** for factual QA and "
        "**VGHD** for retrieval-augmented QA over the Bhagavad Gita. From the M.Tech dissertation of "
        "*Manju Pal* (Gautam Buddha University), supervised by *Dr. Raju Pal*."
    )
    st.caption(f"Runs fully offline. No telemetry, no keys. · [Source code on GitHub]({REPO_URL})")


def footer() -> None:
    st.html(f'<div class="hd-footer">{APP_NAME} · offline research demo · '
            f'<a href="{REPO_URL}" target="_blank">github.com/muditagarwal2308/HallucinationDetection</a></div>')


# ---------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading the Bhagavad Gita corpus …")
def get_corpus():
    from halludetect.gita import GitaCorpus

    return GitaCorpus.load()


@st.cache_resource(show_spinner=False)
def _load_nli(name: str):
    from halludetect.nli import get_nli

    return get_nli(name)


def get_nli_safe(name: str):
    """Return ``(nli_model, warning)``; heavy models fall back to the lexical NLI with a friendly message."""
    if name == "lexical":
        return _load_nli("lexical"), None
    try:
        with st.spinner(f"Loading {name} (the first use downloads the model) …"):
            return _load_nli(name), None
    except Exception as exc:  # missing torch/transformers, no network, out of memory, ...
        return _load_nli("lexical"), (f"Could not load **{name}** ({type(exc).__name__}: {str(exc)[:160]}). "
                                      "Using the offline lexical NLI instead.")


@st.cache_resource(show_spinner=False)
def cached_verifier(nli_name: str, support_threshold: float):
    from halludetect.gita import VerseVerifier

    nli = None if nli_name == "lexical" else _load_nli(nli_name)
    return VerseVerifier(get_corpus(), nli, support_threshold=support_threshold)


def get_verifier_safe(nli_name: str = "lexical", support_threshold: float = 0.5):
    """Return ``(verifier, effective_nli_name, warning)``."""
    if nli_name == "lexical":
        return cached_verifier("lexical", support_threshold), "lexical", None
    _, warning = get_nli_safe(nli_name)
    if warning:
        return cached_verifier("lexical", support_threshold), "lexical", warning
    return cached_verifier(nli_name, support_threshold), nli_name, None


@st.cache_resource(show_spinner=False)
def get_repair_policy():
    from halludetect.gita.policy import RepairPolicy

    return RepairPolicy(get_corpus())


@st.cache_resource(show_spinner="Loading the open-domain QA world …")
def get_world():
    from halludetect.ugrav import load_world

    return load_world()


@st.cache_resource(show_spinner=False)
def world_by_id() -> dict:
    return {w.id: w for w in get_world()}


def find_world_item(question: str):
    from halludetect.text import normalize_answer

    key = normalize_answer(question or "")
    return next((w for w in get_world() if normalize_answer(w.question) == key), None)


@st.cache_resource(show_spinner=False)
def get_retriever():
    from halludetect.ugrav import load_passages
    from halludetect.ugrav.data import PassageRetriever

    return PassageRetriever(load_passages())


def make_simulator(seed: int = 0, profile: str = "sim-7b"):
    """A fresh simulator (cheap); fresh objects keep repeated runs reproducible."""
    from halludetect.ugrav.simulator import SimulatedQALLM

    return SimulatedQALLM(get_world(), seed=int(seed), profile=profile)


def make_openai_llm(model: str, base_url: str, timeout: float = 60.0):
    from halludetect.llm.openai_compat import OpenAICompatLLM

    return OpenAICompatLLM(model=model.strip(), base_url=base_url.strip(), timeout=timeout)


def check_server(base_url: str, timeout: float = 3.0) -> str | None:
    """Return an error message if the OpenAI-compatible server cannot be reached, else ``None``."""
    import requests

    try:
        r = requests.get(base_url.rstrip("/") + "/models", timeout=timeout)
        if r.status_code >= 500:
            return f"The server at {base_url} answered with HTTP {r.status_code}."
        return None
    except Exception as exc:
        return f"Cannot reach {base_url} ({type(exc).__name__}). Is the server (e.g. `ollama serve`) running?"


# ---------------------------------------------------------------- result files
@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json(path: Path) -> dict | None:
    """Load a result file, or ``None`` if it is missing or unreadable (e.g. still being written)."""
    try:
        if not path.exists():
            return None
        return _read_json(str(path), path.stat().st_mtime)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _read_jsonl(path: str, mtime: float) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_jsonl(path: Path) -> list[dict]:
    try:
        return _read_jsonl(str(path), path.stat().st_mtime) if path.exists() else []
    except Exception:
        return []


def ugrav_results() -> dict[str, tuple[str, dict]]:
    """Available UG-RAV benchmark files: ``{key: (label, payload)}`` (simulation first, then real models)."""
    out: dict[str, tuple[str, dict]] = {}
    for p in sorted((RESULTS / "ugrav").glob("benchmark_*.json")):
        d = load_json(p)
        if d and "detection" in d:
            name = p.stem.replace("benchmark_", "")
            out[f"sim:{name}"] = (f"Simulation · {name}", d)
    for p in sorted((RESULTS / "real").glob("benchmark_*.json")):
        d = load_json(p)
        if d and "detection" in d:
            name = p.stem.replace("benchmark_", "").replace("popqa_", "PopQA · ")
            out[f"real:{p.stem}"] = (f"Real model · {name}", d)
    return out


def gita_results() -> dict[str, dict]:
    """Available VGHD benchmark files keyed by NLI name."""
    out: dict[str, dict] = {}
    for p in sorted((RESULTS / "gita").glob("benchmark_*.json")):
        d = load_json(p)
        if d and "detectors" in d:
            out[d.get("nli") or p.stem.replace("benchmark_", "")] = d
    return out


def missing_results_note(command: str, what: str) -> None:
    st.info(f"No {what} found under `results/`. Generate them with `{command}` (run from the repository root) "
            "and reload this page.", icon="ℹ️")


# ---------------------------------------------------------------- HTML components
def chip(text: str, bg: str = INDIGO_SOFT, fg: str = INDIGO_DARK, border: str | None = None, tag: str = "") -> str:
    tag_html = f"<small>{esc(tag)}</small>" if tag else ""
    return (f'<span class="hd-chip" style="background:{bg};color:{fg};border-color:{border or bg}">'
            f"{tag_html}{esc(text)}</span>")


def badge(text: str, bg: str = INDIGO_SOFT, fg: str = INDIGO_DARK, border: str | None = None) -> str:
    return f'<span class="hd-badge" style="background:{bg};color:{fg};border-color:{border or bg}">{esc(text)}</span>'


def severity_chip(severity: str) -> str:
    name, bg, fg, border = SEVERITY_STYLES.get(severity, (severity.title(), "#F1F5F9", "#334155", "#CBD5E1"))
    return badge(name, bg, fg, border)


def speaker_badge(speaker: str | None) -> str:
    if not speaker:
        return ""
    c = SPEAKER_COLORS.get(speaker, SLATE)
    return badge(f"🗣 {speaker}", "#FFFFFF", c, c)


def action_badge(action: str) -> str:
    lab, icon, bg, fg, border = ACTION_STYLES.get(action, (action.title(), "•", "#F1F5F9", "#334155", "#CBD5E1"))
    return badge(f"{icon} {lab}", bg, fg, border)


def bar(value: float, color: str = INDIGO, max_value: float = 1.0) -> str:
    v = 0.0 if is_missing(value) else max(0.0, min(1.0, float(value) / max_value))
    return f'<div class="hd-bar"><span style="width:{100 * v:.1f}%;background:{color}"></span></div>'


def metric_card(label_: str, value: str, sub: str = "") -> str:
    """Value-first card, so that values line up even when labels wrap."""
    sub_html = f'<div class="hd-metric-sub">{sub}</div>' if sub else ""
    return (f'<div class="hd-metric"><div class="hd-metric-value">{esc(value)}</div>'
            f'<div class="hd-metric-label">{esc(label_)}</div>{sub_html}</div>')


def metric_row(cards: list[tuple[str, str, str]]) -> None:
    """A row of equal-height metric cards ``(label, value, sub-text)`` laid out with a CSS grid."""
    inner = "".join(metric_card(lab, val, sub) for lab, val, sub in cards)
    st.html(f'<div class="hd-metric-grid" style="grid-template-columns:repeat({len(cards)},minmax(0,1fr))">'
            f"{inner}</div>")


def step_header(number: int | str, title: str, subtitle: str = "", done: bool = False) -> None:
    cls = "hd-step-num hd-step-done" if done else "hd-step-num"
    sub = f'<div class="hd-step-sub">{subtitle}</div>' if subtitle else ""
    st.html(f'<div class="hd-step"><div class="{cls}">{esc(number)}</div>'
            f'<div><div class="hd-step-title">{esc(title)}</div>{sub}</div></div>')


def callout(text_html: str, tone: str = "saffron") -> None:
    cls = "hd-callout hd-callout-indigo" if tone == "indigo" else "hd-callout"
    st.html(f'<div class="{cls}">{text_html}</div>')


def answer_box(text: str) -> str:
    return f'<div class="hd-answer">{esc(text)}</div>'


def banner(title: str, text: str, icon: str, bg: str, fg: str, border: str, side_html: str = "") -> str:
    side = f'<div class="hd-banner-side">{side_html}</div>' if side_html else ""
    return (f'<div class="hd-banner" style="background:{bg};color:{fg};border-color:{border}">'
            f'<div class="hd-banner-icon">{icon}</div><div><div class="hd-banner-title">{esc(title)}</div>'
            f'<div class="hd-banner-text">{text}</div></div>{side}</div>')


def verdict_banner(verdict: str, score: float | None = None, extra: str = "") -> str:
    lab, icon, bg, fg, border, expl = VERDICT_STYLES.get(
        verdict, (verdict.replace("_", " ").title(), "•", "#F8FAFC", "#334155", "#CBD5E1", ""))
    side = f"risk score<b>{fmt(score, 2)}</b>" if score is not None else ""
    return banner(lab, esc(expl) + (f" {extra}" if extra else ""), icon, bg, fg, border, side)


def action_banner(action: str, text: str) -> str:
    lab, icon, bg, fg, border = ACTION_STYLES.get(action, (action.title(), "•", "#F8FAFC", "#334155", "#CBD5E1"))
    return banner(lab, text, icon, bg, fg, border)


def _verse_lines(text: str, sep: str) -> str:
    parts = [p.strip() for p in text.split(sep) if p.strip()]
    return "<br>".join(esc(p) for p in parts)


def verse_card(verse, *, badges: str = "", note: str = "", show_sanskrit: bool = True) -> str:
    """Card with reference, speaker, Devanagari, IAST and Besant's English translation."""
    alt = ""
    if verse.ref_700 is None:
        alt = badge("not numbered in the 700-verse recension", "#F1F5F9", "#475569")
    elif verse.ref_700 != verse.ref:
        alt = badge(f"= {verse.ref_700} in the 700-verse numbering", "#F1F5F9", "#475569")
    sans = ""
    if show_sanskrit:
        sans = (f'<div class="hd-sanskrit" lang="sa">{_verse_lines(verse.sanskrit, "।")}</div>'
                f'<div class="hd-iast">{_verse_lines(verse.iast, "|")}</div>')
    note_html = f'<div class="hd-note">{note}</div>' if note else ""
    return (f'<div class="hd-verse"><div class="hd-verse-head"><span class="hd-ref">BG {esc(verse.ref)}</span>'
            f"{speaker_badge(verse.speaker)}{alt}{badges}</div>{sans}"
            f'<div class="hd-english">{esc(verse.english)}</div>{note_html}</div>')


def missing_verse_card(ref: str, reason: str) -> str:
    return (f'<div class="hd-verse hd-verse-bad"><div class="hd-verse-head"><span class="hd-ref">BG {esc(ref)}</span>'
            f'{badge("does not exist", "#FEE2E2", "#991B1B", "#FECACA")}</div>'
            f'<div class="hd-english">{esc(reason)}</div></div>')


def cluster_chips(answers: list[str], assignments: list[int]) -> str:
    """Answers as chips coloured by their semantic cluster (C1, C2, …)."""
    out = []
    for a, c in zip(answers, assignments):
        i = c % len(CLUSTER_SOLID)
        out.append(chip(a, CLUSTER_LIGHT[i], INDIGO_DARK, CLUSTER_SOLID[i], tag=f"C{c + 1}"))
    return "<div>" + "".join(out) + "</div>"


def diff_html(before: str, after: str) -> tuple[str, str]:
    """Word-level diff: removed words struck through in ``before``, inserted words highlighted in ``after``."""
    a, b = before.split(), after.split()
    left, right = [], []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            left.append(esc(" ".join(a[i1:i2])))
            right.append(esc(" ".join(b[j1:j2])))
            continue
        if i2 > i1:
            left.append(f"<del>{esc(' '.join(a[i1:i2]))}</del>")
        if j2 > j1:
            right.append(f"<ins>{esc(' '.join(b[j1:j2]))}</ins>")
    return " ".join(left), " ".join(right)


def diff_block(before: str, after: str, before_label: str = "Before", after_label: str = "After") -> None:
    left, right = diff_html(before, after)
    c1, c2 = st.columns(2)
    with c1:
        st.caption(before_label)
        st.html(f'<div class="hd-answer hd-diff">{left}</div>')
    with c2:
        st.caption(after_label)
        st.html(f'<div class="hd-answer hd-diff">{right}</div>')


def pipeline_strip(nodes: list[str]) -> str:
    parts = []
    for i, n in enumerate(nodes):
        if i:
            parts.append('<span class="hd-arrow">→</span>')
        parts.append(f'<span class="hd-node">{n}</span>')
    return f'<div class="hd-pipeline">{"".join(parts)}</div>'


def html_table(rows: list[dict], columns: list[str], *, widths: dict[str, str] | None = None,
               raw: tuple[str, ...] = (), strong: tuple[str, ...] = (), highlight: str | None = None) -> str:
    """A wrapping, styled HTML table (better than a data grid for text-heavy tables).

    ``raw`` columns contain trusted HTML; all other cells are escaped. Rows whose ``highlight`` key is
    truthy get a saffron tint.
    """
    widths = widths or {}
    head = "".join(f'<th style="width:{widths[c]}">{esc(c)}</th>' if c in widths else f"<th>{esc(c)}</th>"
                   for c in columns)
    body = []
    for r in rows:
        cells = []
        for c in columns:
            val = r.get(c, "")
            cls = ' class="hd-strong"' if c in strong else ""
            cells.append(f"<td{cls}>{val if c in raw else esc(val)}</td>")
        tr = '<tr class="hd-row-hl">' if highlight and r.get(highlight) else "<tr>"
        body.append(tr + "".join(cells) + "</tr>")
    return f'<table class="hd-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


# ---------------------------------------------------------------- charts
def style_fig(fig: go.Figure, height: int = 380, legend: bool = True, title: str | None = None) -> go.Figure:
    """House style for Plotly figures. ``title`` is rendered by :func:`show_fig` above the chart, so it can
    never collide with the legend."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36 if legend else 16, b=8),
        font=dict(size=13), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=12),
        meta={"hd_title": title} if title else None,
    )
    return fig


def show_fig(fig: go.Figure, key: str | None = None) -> None:
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    if meta.get("hd_title"):
        st.html(f'<div class="hd-chart-title">{esc(meta["hd_title"])}</div>')
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def gauge(value: float, threshold: float, title: str, *, safe_below: bool = True, height: int = 210) -> go.Figure:
    """Risk gauge on [0, 1] with the decision threshold marked in saffron."""
    good, bad = "#DCFCE7", "#FEE2E2"
    steps = [{"range": [0, threshold], "color": good if safe_below else bad},
             {"range": [threshold, 1], "color": bad if safe_below else good}]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value),
        number={"valueformat": ".2f", "font": {"size": 34, "color": INDIGO_DARK}},
        title={"text": title, "font": {"size": 14, "color": "#4B5563"}},
        gauge={"axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#9CA3AF"},
               "bar": {"color": INDIGO, "thickness": 0.28}, "bgcolor": "#FFFFFF", "borderwidth": 0,
               "steps": steps,
               "threshold": {"line": {"color": SAFFRON, "width": 5}, "thickness": 0.9, "value": float(threshold)}},
    ))
    fig.update_layout(height=height, margin=dict(l=24, r=24, t=48, b=4))
    return fig


def cluster_bar(sizes: list[int], labels: list[str], title: str = "") -> go.Figure:
    k = max(1, sum(sizes))
    probs = [s / k for s in sizes]
    colors = [CLUSTER_SOLID[i % len(CLUSTER_SOLID)] for i in range(len(sizes))]
    fig = go.Figure(go.Bar(
        x=[f"C{i + 1}" for i in range(len(sizes))], y=probs, marker_color=colors,
        text=[f"{p:.2f}" for p in probs], textposition="outside", cliponaxis=False,
        customdata=labels, hovertemplate="%{x}: %{customdata}<br>p = %{y:.2f}<extra></extra>",
    ))
    fig.update_yaxes(range=[0, 1.12], title="p(cluster | question)", gridcolor=GRID)
    fig.update_xaxes(title="meaning cluster")
    fig.update_layout(bargap=0.55 if len(sizes) < 3 else 0.3)
    return style_fig(fig, height=300, legend=False, title=title or None)


def relpath(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def data_path(*parts: str) -> Path:
    return DATA.joinpath(*parts)
