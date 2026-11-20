"""Corpus Browser — read, search and summarise the 701-verse Bhagavad Gita corpus."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

C.setup_page(
    "Corpus Browser", "📖",
    "The verse-aligned corpus behind VGHD: all 701 verses of Annie Besant's 1922 translation with the Sanskrit in "
    "Devanagari and IAST transliteration, the speaker of every verse, and a mapping to the common 700-verse "
    "numbering. Read chapter by chapter or search in English, Hindi or romanised Sanskrit.",
    eyebrow="VGHD · corpus",
)

corpus = C.get_corpus()
chapters = sorted(corpus.chapters)
lengths = corpus.chapter_lengths
speakers = Counter(v.speaker for v in corpus.ordered)
SEARCH_EXAMPLES = ["weapons cannot cut the soul", "कर्म का फल", "karmaṇy evādhikāras te", "dharmakṣetre kurukṣetre",
                   "steady wisdom like a tortoise", "क्रोध से मोह"]

C.metric_row([
    ("Verses", f"{len(corpus)}", "Besant numbering"),
    ("Chapters", f"{len(chapters)}", "18 discourses"),
    ("Speakers", f"{len(speakers)}", " · ".join(f"{s} {n}" for s, n in speakers.most_common())),
    ("700-verse mapping", f"{sum(1 for v in corpus.ordered if v.ref_700)}", "verses with a common-recension ref"),
])
st.space("small")


def _step(delta: int) -> None:
    ch = st.session_state.get("cb_ch", 1)
    v = st.session_state.get(f"cb_v_{ch}", 1) + delta
    if v < 1:
        if ch > chapters[0]:
            ch -= 1
            v = lengths[ch]
        else:
            v = 1
    elif v > lengths[ch]:
        if ch < chapters[-1]:
            ch += 1
            v = 1
        else:
            v = lengths[ch]
    st.session_state.cb_ch = ch
    st.session_state[f"cb_v_{ch}"] = v


def _set_query(q: str) -> None:
    st.session_state.cb_query = q


read_tab, search_tab, stats_tab, about_tab = st.tabs(["📖 Read", "🔍 Search", "🗣️ Speakers & chapters",
                                                      "ℹ️ Edition & provenance"])

# ------------------------------------------------------------------ read
with read_tab:
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        ch = st.selectbox("Chapter", chapters, key="cb_ch",
                          format_func=lambda c: f"{c} · {corpus.chapters[c]['name']} ({lengths[c]} verses)")
        vnum = st.slider("Verse", 1, lengths[ch], 1, key=f"cb_v_{ch}")
        b1, b2 = st.columns(2)
        b1.button("Previous", icon=":material/chevron_left:", on_click=_step, args=(-1,), key="cb_prev",
                  width="stretch")
        b2.button("Next", icon=":material/chevron_right:", on_click=_step, args=(1,), key="cb_next", width="stretch")
        chapter_speakers = Counter(v.speaker for v in corpus.ordered if v.chapter == ch)
        st.html(
            f'<div class="hd-card"><h4>Chapter {ch}: {C.esc(corpus.chapters[ch]["name"])}</h4>'
            f'<div class="hd-muted">{lengths[ch]} verses (Besant) · {corpus.chapter_lengths_700[ch]} in the '
            "700-verse numbering</div><div style='margin-top:.5rem'>"
            + "".join(C.speaker_badge(s) + f' <span class="hd-muted">{n}</span>&nbsp;&nbsp;'
                      for s, n in chapter_speakers.most_common()) + "</div></div>"
        )
    with c2:
        verse = corpus.get(ch, vnum)
        if verse is not None:
            st.html(C.verse_card(verse))
            with st.expander("Context: the neighbouring verses"):
                for nb in corpus.neighbours(verse):
                    st.html(C.verse_card(nb, show_sanskrit=False))

# ------------------------------------------------------------------ search
with search_tab:
    st.caption("Hybrid search: English BM25 over Besant's translation, a Hindi→English concept glossary, and "
               "character n-grams over the IAST and Devanagari text, fused by reciprocal rank.")
    ex_cols = st.columns(len(SEARCH_EXAMPLES))
    for i, (col, q) in enumerate(zip(ex_cols, SEARCH_EXAMPLES)):
        col.button(q, key=f"cb_ex_{i}", on_click=_set_query, args=(q,), width="stretch")
    s1, s2 = st.columns([5, 1], vertical_alignment="bottom")
    query = s1.text_input("Search the Gita", key="cb_query", placeholder="e.g. the mind is restless like the wind")
    k = s2.number_input("Results", 1, 20, 5, key="cb_k")
    if query.strip():
        hits = corpus.search(query.strip(), k=int(k))
        if not hits:
            st.info("No verse matches this query.", icon="🔍")
        cols = st.columns(2, gap="medium")
        for i, (v, score) in enumerate(hits):
            with cols[i % 2]:
                st.html(C.verse_card(v, badges=C.badge(f"#{i + 1} · score {score:.3f}", "#F1F5F9", "#334155")))
    else:
        st.info("Type a query or pick an example above.", icon="🔍")

# ------------------------------------------------------------------ stats
with stats_tab:
    left, right = st.columns([2, 3], gap="large")
    order = [s for s in ("Krishna", "Arjuna", "Sanjaya", "Dhritarashtra") if s in speakers] + \
        [s for s in speakers if s not in ("Krishna", "Arjuna", "Sanjaya", "Dhritarashtra")]
    with left:
        fig = go.Figure(go.Pie(labels=order, values=[speakers[s] for s in order], hole=0.55, sort=False,
                               marker=dict(colors=[C.SPEAKER_COLORS.get(s, C.SLATE) for s in order]),
                               textinfo="label+value"))
        C.show_fig(C.style_fig(fig, height=360, legend=False, title="Who speaks the 701 verses"), key="cb_pie")
    with right:
        by_ch = {s: [sum(1 for v in corpus.ordered if v.chapter == c and v.speaker == s) for c in chapters]
                 for s in order}
        fig = go.Figure()
        for s in order:
            fig.add_bar(x=chapters, y=by_ch[s], name=s, marker_color=C.SPEAKER_COLORS.get(s, C.SLATE))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="chapter", tickmode="linear", dtick=1)
        fig.update_yaxes(title="verses", gridcolor=C.GRID)
        C.show_fig(C.style_fig(fig, height=360, title="Speakers per chapter"), key="cb_stack")
    st.caption("**What this shows —** Krishna speaks most of the text; Arjuna's questions open many chapters, and "
               "Sanjaya narrates the frame story to the blind king Dhritarashtra, who speaks only the first verse. "
               "VGHD uses these attributions to catch answers that put Krishna's words in Arjuna's mouth.")
    table = pd.DataFrame([{
        "Chapter": c, "Name": corpus.chapters[c]["name"], "Verses (Besant)": lengths[c],
        "Verses (700)": corpus.chapter_lengths_700[c], **{s: by_ch[s][i] for s in order},
    } for i, c in enumerate(chapters)])
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Name": st.column_config.TextColumn(width="large")})

# ------------------------------------------------------------------ about
with about_tab:
    v131 = corpus.get(13, 1)
    C.callout(
        "<b>Edition note — why 701 verses?</b> Besant's edition numbers Arjuna's opening question in chapter 13 "
        "(“Matter and Spirit … these I fain would learn, O Keshava”) as <b>13.1</b>, giving 35 verses in chapter 13 "
        "and 701 in total. Many popular editions leave that verse unnumbered, giving the canonical count of "
        "<b>700</b> (34 verses in chapter 13), so Besant's 13.<i>n</i> is 13.<i>n</i>−1 there. The corpus keeps "
        "Besant's numbering and stores the 700-verse equivalent of every verse, so VGHD accepts a citation in "
        "either convention.", tone="indigo")
    if v131 is not None:
        st.html(C.verse_card(v131))
    C.callout(
        "<b>Provenance.</b> Annie Besant, <i>The Bhagavad-Gita, or The Lord's Song</i> (4th ed., G. A. Natesan, "
        "Madras, 1922) — in the public domain. The transcription, with the Devanagari text, comes from English "
        "Wikisource (<i>Bhagavad-Gita (Besant 4th)</i>); each English paragraph is anchored to the number of the "
        "Devanagari verse that precedes it, which yields a complete, gap-free alignment. IAST is transliterated "
        "from the Devanagari. Build script: <code>scripts/build_gita_corpus.py</code>; data: "
        "<code>data/gita/verses.json</code>.")

C.footer()
