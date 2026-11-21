"""Gita Verse Verifier — verse-level verification of an answer, with repair or abstention."""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ctx = C.setup_page(
    "Gita Verse Verifier", "📜",
    "Paste a question about the Bhagavad Gita and an answer — from any chatbot. <b>VGHD</b> splits the answer into "
    "claims and checks each against the verse it cites: does the verse exist, does it entail the claim, does the "
    "quotation match, is the speaker right, and is the question's premise true? Then it accepts, repairs or "
    "abstains.",
    eyebrow="VGHD · verification",
)

MOTIVATING_Q = "Which verse of the Bhagavad Gita says that the soul cannot be cut by weapons?"
PRESETS = {
    "motivating": ("Motivating example · BG 2.20", MOTIVATING_Q,
                   "The Gita states that the soul cannot be cut by weapons or burnt by fire. This is described in "
                   "BG 2.20."),
    "corrected": ("Corrected · BG 2.23", MOTIVATING_Q,
                  "The Gita states that the soul cannot be cut by weapons or burnt by fire. This is described in "
                  "BG 2.23."),
    "fabricated": ("Fabricated · BG 2.80", MOTIVATING_Q,
                   "The Gita states that the soul cannot be cut by weapons or burnt by fire. This is described in "
                   "BG 2.80."),
    "speaker": ("Wrong speaker", "What does Krishna say about action and its fruits?",
                "Arjuna teaches that one's business is with the action only, never with its fruits (BG 2.47)."),
    "misquote": ("Misquotation · BG 18.66", "What does Krishna promise to those who take refuge in him?",
                 "Krishna says, \"Abandon all duties and come to me; I will make you rich and powerful\" "
                 "(BG 18.66)."),
    "commentary": ("Unsupported commentary", "What does Krishna say about action and its fruits?",
                   "Krishna says that action without attachment to its fruits guarantees material wealth and "
                   "success in this life (BG 2.47)."),
    "premise": ("False premise · “Time am I”", "Why does Arjuna say \"Time am I, laying desolate the world\"?",
                "Arjuna says these words because he is terrified by the universal form (BG 11.32)."),
    "scope": ("Out of scope · smartphones", "What does the Bhagavad Gita say about using smartphones?",
              "The Gita advises that one should limit smartphone use to keep the mind steady (BG 6.26)."),
    "quote_ok": ("Correct quotation · BG 18.66", "What does Krishna promise to those who take refuge in him?",
                 "Krishna says, \"Abandoning all duties, come unto Me alone for shelter; sorrow not, I will liberate "
                 "thee from all sins\" (BG 18.66)."),
    "refusal": ("Correct refusal", "Why does Arjuna say \"Time am I, laying desolate the world\"?",
                "Arjuna does not say this; these words are spoken by Krishna in the vision of the universal form "
                "(BG 11.32)."),
}
CLAIM_STATUS = {
    "supported": "✅ supported", "misattributed": "🔀 misattributed", "misquoted": "✂️ misquoted",
    "fabricated_citation": "👻 fabricated citation", "imprecise_citation": "📍 imprecise citation",
    "unsupported": "❓ unsupported", "uncited": "🔖 uncited", "context": "💬 context (not checked)",
    "wrong_speaker": "🗣️ wrong speaker",
}
QUOTE_STATUS = {"verified": "✅ verified", "misquoted": "✂️ misquoted", "misattributed": "🔀 misattributed",
                "not_found": "❓ not found"}


def _apply_preset() -> None:
    key = st.session_state.get("v_preset")
    if key in PRESETS:
        _, q, a = PRESETS[key]
        st.session_state.vq, st.session_state.va = q, a
        st.session_state.v_inputs = (q, a)


if "vq" not in st.session_state:
    st.session_state.v_preset = "motivating"
    _apply_preset()

with ctx.sidebar:
    st.markdown("**Verifier settings**")
    nli_name = st.selectbox("NLI back-end", list(C.NLI_CHOICES), format_func=C.NLI_CHOICES.get, key="v_nli",
                            help=C.NLI_HELP)
    tuned = C.tuned_threshold(nli_name)
    threshold = st.slider("Support threshold", 0.05, 0.9, tuned, 0.05, key=f"v_threshold_{nli_name}",
                          help="Entailment probability at which a verse counts as supporting a claim. The default "
                               "is the value tuned on the dev split for this NLI back-end (VGHD Results page).")
verifier, eff_nli, warning = C.get_verifier_safe(nli_name, float(threshold))
if warning:
    st.warning(warning, icon="⚠️")


@st.cache_data(show_spinner=False, max_entries=256)
def verify(question: str, answer: str, nli: str, support_threshold: float) -> dict:
    ver = C.cached_verifier(nli, support_threshold)
    rep = ver.verify(question, answer)
    dec = C.get_repair_policy().decide(rep)
    out = {"report": rep.as_dict(), "decision": dec.as_dict(), "repaired": None}
    if dec.action == "repair":
        out["repaired"] = ver.verify(question, dec.answer).as_dict()
    return out


# ------------------------------------------------------------------ inputs
st.pills("Examples", list(PRESETS), format_func=lambda k: PRESETS[k][0], key="v_preset", on_change=_apply_preset)
with st.form("verify_form", border=True):
    st.text_area("Question", key="vq", height=75)
    st.text_area("Answer to verify", key="va", height=110,
                 help="Cite verses as BG 2.47, Gita 2:47 or “chapter 2, verse 47”; quote with “…”.")
    submitted = st.form_submit_button("Verify answer", type="primary", icon=":material/fact_check:", key="v_submit")
if submitted:
    st.session_state.v_inputs = (st.session_state.vq, st.session_state.va)

question, answer = st.session_state.get("v_inputs", ("", ""))
if not question.strip() or not answer.strip():
    st.info("Enter both a question and an answer, then press **Verify answer**.", icon="✍️")
    C.footer()
    st.stop()

with st.spinner("Verifying verse by verse …"):
    out = verify(question, answer, eff_nli, float(threshold))
rep, dec = out["report"], out["decision"]
corpus = C.get_corpus()

# ------------------------------------------------------------------ verdict
types = "".join(C.chip(C.label(t, C.ISSUE_LABELS), "#FEE2E2", "#991B1B", "#FECACA") for t in rep["types"])
st.html(C.verdict_banner(rep["verdict"], rep["score"], extra=f"<div style='margin-top:.4rem'>{types}</div>"
                         if types else ""))
n_cit = sum(len(c["cited"]) for c in rep["claims"])
C.metric_row([
    ("Claims checked", str(len(rep["claims"])), "sentences making a statement"),
    ("Verses cited", str(n_cit), f"{sum(1 for c in rep['claims'] for x in c['cited'] if not x['exists'])} "
                                 "non-existent"),
    ("Issues", str(len(rep["issues"])), " · ".join(f"{sum(1 for i in rep['issues'] if i['severity'] == s)} {s}"
                                                  for s in ("critical", "major", "minor"))),
    ("Risk score", C.fmt(rep["score"], 2), "max claim risk (0 = safe, 1 = certain error)"),
])
st.space("small")

# ------------------------------------------------------------------ issues and premise
left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("##### Issues")
    if not rep["issues"]:
        st.success("No issues: every claim is supported by the verse it cites.", icon="✅")
    for issue in rep["issues"]:
        _, _, _, border = C.SEVERITY_STYLES.get(issue["severity"], ("", "", "", C.SLATE))
        claim = (C.badge(f"claim {issue['claim_index'] + 1}", "#F1F5F9", "#334155")
                 if issue.get("claim_index") is not None else "")
        fix = (f'<div class="hd-issue-fix">💡 Suggestion: {C.esc(issue["suggestion"])}</div>'
               if issue.get("suggestion") else "")
        st.html(f'<div class="hd-issue" style="border-left-color:{border}">{C.severity_chip(issue["severity"])}'
                f'<span class="hd-issue-title">{C.esc(C.label(issue["type"], C.ISSUE_LABELS))}</span> {claim}'
                f'<div class="hd-issue-msg">{C.esc(issue["message"])}</div>{fix}</div>')
with right:
    st.markdown("##### Premise analysis")
    prem = rep["premise"]
    name, color = C.PREMISE_LABELS.get(prem["kind"], (prem["kind"], C.SLATE))
    body = [f'<div>{C.badge(name, "#FFFFFF", color, color)}</div>']
    if prem.get("presupposition"):
        body.append(f'<div class="hd-note">The question presupposes: “{C.esc(prem["presupposition"])}”'
                    + (f' — spoken by <b>{C.esc(prem["presupposed_speaker"])}</b>'
                       if prem.get("presupposed_speaker") else "") + "</div>")
    if prem.get("message"):
        body.append(f'<div class="hd-issue-msg">{C.esc(prem["message"])}</div>')
    if prem["kind"] == "ok" and not prem.get("presupposition"):
        body.append('<div class="hd-note">No presupposition to check: the question does not assume that a verse '
                    "says something, refers only to existing verses and is within the Gita's scope.</div>")
    for ev in prem.get("evidence", [])[:3]:
        v = corpus.get(ev["ref"])
        score = (f"entailment {ev['entailment']:.2f} · contradiction {ev.get('contradiction', 0):.2f}"
                 if "entailment" in ev else f"quotation match {ev.get('quote_match', 0):.2f}")
        if v is not None:
            body.append(f'<div class="hd-note"><b>BG {C.esc(v.ref)}</b> ({C.esc(v.speaker)}) — {score}<br>'
                        f'<span class="hd-muted">{C.esc(v.english[:220])}{"…" if len(v.english) > 220 else ""}'
                        "</span></div>")
    st.html(f'<div class="hd-card">{"".join(body)}</div>')

# ------------------------------------------------------------------ claims
def _speaker(c: dict) -> str:
    if c.get("speaker") is None:
        return "—"
    if c.get("speaker_ok") is None:
        return c["speaker"]
    return f"{c['speaker']} {'✅' if c['speaker_ok'] else '❌'}"


st.markdown("##### Claims")
if rep["claims"]:
    claims_df = pd.DataFrame([{
        "#": c["index"] + 1, "Claim": c["core"] or c["claim"], "Cites": ", ".join(c["citations"]) or "—",
        "Status": CLAIM_STATUS.get(c["status"], c["status"]), "Support": c["support"],
        "Best verse": f"BG {c['best_ref']}" if c.get("best_ref") else "—", "Best support": c["best_support"],
        "Speaker": _speaker(c), "Risk": c["risk"],
    } for c in rep["claims"]])
    st.dataframe(claims_df, hide_index=True, width="stretch", column_config={
        "Claim": st.column_config.TextColumn(width="large"),
        "Support": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0,
                                                   help="Entailment of the claim by the cited verse(s)."),
        "Best support": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0,
                                                        help="Entailment by the best verse in the whole Gita."),
        "Risk": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0),
    })
else:
    st.caption("No claims to check — the answer declines the question.")

# ------------------------------------------------------------------ cited verses
cards = []
for c in rep["claims"]:
    cited_refs = set()
    for x in c["cited"]:
        cited_refs.add(x["ref"])
        if x["exists"]:
            v = corpus.get(x["ref"])
            if v is None:
                continue
            ok = x["entailment"] >= threshold
            note = (f"Claim {c['index'] + 1} · entailment <b>{x['entailment']:.2f}</b> · contradiction "
                    f"{x['contradiction']:.2f}{C.bar(x['entailment'], C.GREEN if ok else C.RED)}")
            cards.append(C.verse_card(v, badges=C.badge(f"cited as {x['citation']}", C.INDIGO_SOFT, C.INDIGO_DARK),
                                      note=note))
        else:
            ch = int(x["ref"].split(".")[0])
            n = corpus.chapter_lengths.get(ch)
            why = (f"Chapter {ch} has only {n} verses." if n else "The Bhagavad Gita has only 18 chapters.")
            cards.append(C.missing_verse_card(x["ref"], f"Cited as {x['citation']} in claim {c['index'] + 1}. {why}"))
    best = c.get("best_ref")
    if best and best not in cited_refs and c["status"] not in ("context",):
        v = corpus.get(best)
        if v is not None:
            note = (f"Best match for claim {c['index'] + 1} in the whole Gita · entailment "
                    f"<b>{c['best_support']:.2f}</b>{C.bar(c['best_support'], C.GREEN)}")
            cards.append(C.verse_card(v, badges=C.badge("best supporting verse", "#DCFCE7", "#14532D"), note=note))
if cards:
    st.markdown("##### Verses")
    cols = st.columns(2, gap="medium")
    for i, card in enumerate(cards):
        with cols[i % 2]:
            st.html(card)

quotes = [(c["index"], q) for c in rep["claims"] for q in c["quotes"]]
if quotes:
    st.markdown("##### Quotations")
    st.dataframe(pd.DataFrame([{
        "Claim": i + 1, "Quotation": q["quote"], "Status": QUOTE_STATUS.get(q["status"], q["status"]),
        "Match with cited verse": q["match_cited"],
        "Closest verse": f"BG {q['best_ref']}" if q.get("best_ref") else "—", "Closest match": q["best_match"],
    } for i, q in quotes]), hide_index=True, width="stretch", column_config={
        "Quotation": st.column_config.TextColumn(width="large"),
        "Match with cited verse": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0),
        "Closest match": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0),
    })

# ------------------------------------------------------------------ repair decision
st.markdown("##### Repair-or-abstain decision")
st.html(C.action_banner(dec["action"], C.esc(dec["rationale"])))
if dec["action"] == "repair":
    C.diff_block(answer, dec["answer"], "Original answer (removed words struck through)",
                 "Repaired answer (changes highlighted)")
    if dec["edits"]:
        st.markdown("Edits: " + " · ".join(f"`{e}`" for e in dec["edits"]))
    rv = out.get("repaired")
    if rv:
        lab = C.VERDICT_STYLES.get(rv["verdict"], (rv["verdict"],))[0]
        if rv["verdict"] == "faithful":
            st.success(f"Re-verification of the repaired answer: **{lab}** (risk {C.fmt(rv['score'], 2)}).",
                       icon="🔁")
        else:
            st.warning(f"Re-verification of the repaired answer: **{lab}** (risk {C.fmt(rv['score'], 2)}).",
                       icon="🔁")
elif dec["action"] == "abstain":
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Original answer")
        st.html(f'<div class="hd-answer hd-diff"><del>{C.esc(answer)}</del></div>')
    with c2:
        st.caption("System response")
        st.html(C.answer_box(dec["answer"]))
    if dec["edits"]:
        st.markdown("Attempted edits: " + " · ".join(f"`{e}`" for e in dec["edits"]))
else:
    st.caption("The answer is shown unchanged.")
    st.html(C.answer_box(dec["answer"]))

with st.expander("How VGHD checks an answer"):
    st.markdown(
        "1. **Premise** — non-existent references (“BG 19.5”), topics the Gita cannot address (smartphones, "
        "cricket) and presuppositions (“Which verse says X?”, “Why does Arjuna say …?”) are checked against the "
        "verses first.\n"
        "2. **Claims** — the answer is split into sentences; pointer sentences such as “This is described in BG "
        "2.20.” pass their citation to the previous claim.\n"
        "3. **Citations** — every cited verse must exist (in Besant's 701-verse or the common 700-verse "
        "numbering) and must *entail* the claim (NLI). If another verse entails it, the citation is "
        "*misattributed* (or *imprecise* if it is a neighbour).\n"
        "4. **Quotations** — quoted text is fuzzy-matched against the English, IAST and Devanagari of the verse.\n"
        "5. **Speaker** — “Arjuna says …” must match the speaker of the cited verse.\n"
        "6. **Policy** — problematic premises → abstain; citation, quotation and speaker errors → deterministic "
        "repair, then re-verification; unsupported claims → removed (abstain if nothing remains)."
    )

C.footer()
