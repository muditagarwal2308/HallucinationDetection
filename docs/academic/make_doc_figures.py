"""Figures for the academic documents (dissertation, paper, progress reports, work plan).

Every result figure is drawn from the benchmark JSON files in ``results/``; nothing is typed in by
hand. Diagrams (architecture, pipelines, taxonomies) and work-plan Gantt charts are drawn with
matplotlib patches. Output: ``docs/academic/figures/*.png`` at 200 dpi.

Usage::

    python docs/academic/make_doc_figures.py
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
RES = ROOT / "results"
DPI = 200

INDIGO = "#4338A8"
TEAL = "#1F9E8F"
ORANGE = "#F29F1F"
RED = "#D7263D"
GREEN = "#2E9E57"
GREY = "#8A8F98"
LIGHT = {"blue": "#DCE8F7", "green": "#DDF1E3", "orange": "#FDEBD0", "red": "#F9DADD",
         "grey": "#EEF0F3", "indigo": "#E4E2F6", "teal": "#D5F0EC", "yellow": "#FFF6D6"}
EDGE = {"blue": "#2F5F98", "green": "#2E7D4A", "orange": "#C27A12", "red": "#A61E2F",
        "grey": "#5B616B", "indigo": INDIGO, "teal": "#167A6F", "yellow": "#B38F1D"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    plt.close(fig)
    print(f"  figures/{name}.png")


# ---------------------------------------------------------------------------- diagram helpers


def canvas(w: float, h: float, xmax: float = 100.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, xmax * h / w)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color="blue", size=9.5, bold=False, round_=1.2, lw=1.1, style=None):
    """Rounded box centred at (x, y)."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={round_}",
                       fc=LIGHT[color], ec=EDGE[color], lw=lw, linestyle=style or "-")
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=size, fontweight="bold" if bold else "normal",
            linespacing=1.15, wrap=False)
    return (x, y, w, h)


def diamond(ax, x, y, w, h, text, color="yellow", size=9):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(pts, closed=True, fc=LIGHT[color], ec=EDGE[color], lw=1.1))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, linespacing=1.1)
    return (x, y, w, h)


def arrow(ax, p1, p2, text=None, color="#333333", tpos=0.5, toff=(0, 0), size=8.5, style="-|>", ls="-",
          rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11, color=color, lw=1.1, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if text:
        tx = p1[0] + (p2[0] - p1[0]) * tpos + toff[0]
        ty = p1[1] + (p2[1] - p1[1]) * tpos + toff[1]
        ax.text(tx, ty, text, fontsize=size, ha="center", va="center", style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6))


def poly_arrow(ax, pts, text=None, tidx=0, toff=(0, 0), size=8.5, color="#333333"):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.1)
    arrow(ax, pts[-2], pts[-1], color=color)
    if text:
        a, b = pts[tidx], pts[tidx + 1]
        ax.text((a[0] + b[0]) / 2 + toff[0], (a[1] + b[1]) / 2 + toff[1], text, fontsize=size, ha="center",
                va="center", style="italic", bbox=dict(fc="white", ec="none", pad=0.6))


def bottom(b):
    return (b[0], b[1] - b[3] / 2)


def top(b):
    return (b[0], b[1] + b[3] / 2)


def left(b):
    return (b[0] - b[2] / 2, b[1])


def right(b):
    return (b[0] + b[2] / 2, b[1])


# ---------------------------------------------------------------------------- conceptual diagrams


def fig_taxonomy(shaded: set[str], name: str, note: str) -> None:
    fig, ax = canvas(6.6, 3.15)
    root = box(ax, 50, 42.5, 34, 6.5, "Hallucination in LLMs", "grey", 10.5, True)
    fa = box(ax, 20.5, 29.5, 37, 8, "Factuality hallucination\nconflicts with world knowledge", "blue", 9.5, True)
    fb = box(ax, 70.5, 29.5, 43, 8, "Faithfulness hallucination\nbreaks instruction, context or logic", "indigo",
             9.5, True)
    leaves = [
        ("inconsistency", 10, "Factual\ninconsistency\nwrong fact stated"),
        ("fabrication", 30, "Factual\nfabrication\ninvented fact\nor source"),
        ("instruction", 51, "Instruction\ninconsistency\ntask ignored"),
        ("context", 70.5, "Context\ninconsistency\ncontradicts the\ngiven context"),
        ("logical", 90, "Logical\ninconsistency\nself-\ncontradiction"),
    ]
    for key, x, label in leaves:
        on = key in shaded
        b = box(ax, x, 11, 18, 13.5, label, "orange" if on else "grey", 8.6, False, lw=1.6 if on else 0.9)
        parent = fa if key in ("inconsistency", "fabrication") else fb
        arrow(ax, bottom(parent), top(b))
    arrow(ax, bottom(root), top(fa))
    arrow(ax, bottom(root), top(fb))
    ax.text(50, 1.2, note, ha="center", va="center", fontsize=8.5, style="italic", color="#444444")
    save(fig, name)


def fig_categorisation(spec: dict, name: str, note: str) -> None:
    fig, ax = canvas(6.6, 3.0)
    root = box(ax, 50, 40, 44, 6.5, spec["root"], "grey", 10.5, True)
    xs = {0: [8.5, 25.5, 42.5], 1: [58, 75, 91.5]}
    for i, (head, leaves) in enumerate(spec["branches"]):
        hx = 25.5 if i == 0 else 75
        hb = box(ax, hx, 27.5, 26, 6.5, head, "blue" if i == 0 else "indigo", 10, True)
        arrow(ax, bottom(root), top(hb))
        for x, (label, on) in zip(xs[i], leaves):
            b = box(ax, x, 10.5, 15, 14, label, "orange" if on else "grey", 8.0, lw=1.6 if on else 0.9)
            arrow(ax, bottom(hb), top(b))
    ax.text(50, 1.0, note, ha="center", va="center", fontsize=8.5, style="italic", color="#444444")
    save(fig, name)


def fig_ugrav_architecture() -> None:
    fig, ax = canvas(6.6, 8.4)
    H = 100 * 8.4 / 6.6
    cx = 40
    q = box(ax, cx, H - 5, 30, 5.5, "Question $q$", "grey", 10, True)
    s1 = box(ax, cx, H - 15, 50, 7.5, "Closed-book sampling\ngreedy answer $a_0$ + $K$ samples ($T = 1.0$)", "blue")
    s2 = box(ax, cx, H - 26.5, 50, 7.5,
             "Uncertainty signals: SE, self-consistency,\nPE, LN-PE, P(True), min-token probability", "blue")
    s3 = box(ax, cx, H - 38, 50, 7.5,
             "Closed-book risk  $\\hat{r}_0 = \\sigma(\\mathbf{w}_0^{\\top}\\mathbf{x}_0 + b_0)$", "indigo")
    d1 = diamond(ax, cx, H - 51, 34, 12, "$\\hat{r}_0 < \\tau_{low}$ ?")
    acc = box(ax, 84, H - 51, 24, 7.5, "Accept $a_0$\n(no retrieval)", "green", 9.5, True)
    s4 = box(ax, cx, H - 65, 50, 7.5, "Retrieve top-$k$ passages\n(BM25 or Wikipedia search)", "teal")
    db = box(ax, 84, H - 65, 24, 7.5, "Passage\ncollection", "grey", 9)
    s5 = box(ax, cx, H - 76.5, 50, 7.5, "Evidence-conditioned sampling\nanswer $a_D$ + $K$ samples with passages",
             "teal")
    s6 = box(ax, cx, H - 88, 50, 7.5,
             "Evidence signals (NLI)\nSup, Con, SE$_{RAG}$, $\\Delta$SE, agreement", "teal")
    s7 = box(ax, cx, H - 99.5, 50, 7.5,
             "Fused risk  $\\hat{r} = \\sigma(\\mathbf{w}^{\\top}\\mathbf{x} + b)$", "orange")
    d2 = diamond(ax, cx, H - 112, 40, 12, "$\\hat{r} < \\tau_{high}$  and\nSup$(a_D) \\geq \\theta_{sup}$ ?")
    ans = box(ax, 84, H - 112, 24, 7.5, "Answer $a_D$\nfrom evidence", "green", 9.5, True)
    ab = box(ax, cx, H - 124, 34, 5.5, "Abstain: \u201cI don't know\u201d", "red", 9.5, True)
    nli = box(ax, 84, H - 82, 24, 7.5, "NLI model\n(DeBERTa-MNLI)", "grey", 9)
    for a, b in [(q, s1), (s1, s2), (s2, s3), (s3, d1), (s4, s5), (s5, s6), (s6, s7), (s7, d2)]:
        arrow(ax, bottom(a), top(b))
    arrow(ax, right(d1), left(acc), "yes", tpos=0.45, toff=(0, 1.6))
    arrow(ax, bottom(d1), top(s4), "no", tpos=0.45, toff=(2.5, 0))
    arrow(ax, left(db), right(s4))
    arrow(ax, right(d2), left(ans), "yes", tpos=0.45, toff=(0, 1.6))
    arrow(ax, bottom(d2), top(ab), "no", tpos=0.5, toff=(2.5, 0))
    arrow(ax, (nli[0] - 12, nli[1] + 1.5), (cx + 25, H - 87), color=GREY, ls="--")
    # closed-book features also feed the fused model
    poly_arrow(ax, [(cx - 25, H - 26.5), (5, H - 26.5), (5, H - 99.5), (cx - 25, H - 99.5)],
               "closed-book signals $\\mathbf{x}_0$", tidx=1, toff=(0, 0), size=8.5)
    ax.text(84, H - 5, "Stage 1: detect\n(closed book)", ha="center", fontsize=9, color=EDGE["blue"],
            fontweight="bold")
    ax.text(84, H - 92.5, "Stage 2: verify\nand decide", ha="center", fontsize=9, color=EDGE["teal"],
            fontweight="bold")
    save(fig, "ugrav_architecture")


def fig_se_godan() -> None:
    fig, ax = canvas(6.6, 1.95)
    ax.text(17, 30.0, "Step 1: sample $K = 5$ answers", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(50, 30.0, "Step 2: cluster by meaning", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(84, 30.0, "Step 3: entropy over clusters", ha="center", fontsize=9.5, fontweight="bold")
    samples = ["Premchand", "Munshi Premchand", "Premchand", "Rabindranath Tagore", "Sarat Chandra"]
    colors = ["blue", "blue", "blue", "red", "orange"]
    for i, (s, c) in enumerate(zip(samples, colors)):
        box(ax, 17, 25.5 - i * 4.4, 27, 3.4, s, c, 8.8, round_=0.8)
    clusters = [("$C_1$: Premchand    $|C_1| = 3$", "blue"), ("$C_2$: Tagore    $|C_2| = 1$", "red"),
                ("$C_3$: Sarat Chandra    $|C_3| = 1$", "orange")]
    for i, (t, c) in enumerate(clusters):
        box(ax, 50, 25 - i * 6.0, 29, 4.4, t, c, 8.8, round_=0.8)
    ax.text(50, 6.0, "bidirectional NLI entailment\n(question as context)", ha="center", fontsize=8.2,
            style="italic", color="#444444")
    ax.text(84, 24.5, "$p(C) = (0.6,\\ 0.2,\\ 0.2)$", ha="center", fontsize=10)
    ax.text(84, 18.0, "$SE = -\\sum_c\\, p(c)\\,\\ln p(c)$", ha="center", fontsize=10)
    ax.text(84, 11.5, "$SE \\approx 0.95$ nats  (max $\\ln 5 = 1.61$)", ha="center", fontsize=10)
    ax.text(84, 6.5, "uncertain: retrieval is useful", ha="center", fontsize=8.5, style="italic", color=EDGE["red"])
    arrow(ax, (31.5, 16.8), (35, 16.8))
    arrow(ax, (65, 16.8), (68.5, 16.8))
    save(fig, "se_godan")


def fig_exp_pipeline() -> None:
    fig, ax = canvas(6.6, 1.85)
    steps = [("Load questions\nand gold aliases", "grey"), ("Closed-book and\nRAG generation", "blue"),
             ("Cache answers,\nlog-probs, P(True)", "blue"), ("Signals: SE, PE,\nNLI support, $\\Delta$SE", "green"),
             ("Fusion, policies\nand evaluation", "green")]
    xs = [10, 30, 50, 70, 90]
    for i, ((t, c), x) in enumerate(zip(steps, xs)):
        box(ax, x, 15, 17, 10, t, c, 8.8)
        if i:
            arrow(ax, (xs[i - 1] + 8.5, 15), (x - 8.5, 15))
    ax.add_patch(FancyBboxPatch((20.5, 6.5), 39, 17.5, boxstyle="round,pad=0,rounding_size=1.5", fc="none",
                                ec=EDGE["blue"], lw=0.9, ls="--"))
    ax.add_patch(FancyBboxPatch((60.5, 6.5), 39, 17.5, boxstyle="round,pad=0,rounding_size=1.5", fc="none",
                                ec=EDGE["green"], lw=0.9, ls="--"))
    ax.text(40, 3.4, "GPU / model server: run once, cached to JSONL", ha="center", fontsize=8.5,
            color=EDGE["blue"], style="italic")
    ax.text(80, 3.4, "CPU: repeatable analysis (seconds)", ha="center", fontsize=8.5, color=EDGE["green"],
            style="italic")
    save(fig, "exp_pipeline")


def fig_vghd_pipeline() -> None:
    fig, ax = canvas(6.6, 4.3)
    q = box(ax, 13, 50, 23, 9, "Question\n(English, Hindi or\nromanised Sanskrit)", "grey", 8.6, True)
    pr = box(ax, 13, 30, 22, 8, "Premise and\nscope check", "orange", 8.8)
    rt = box(ax, 42, 50, 27, 9, "Hybrid verse retrieval\nBM25, IAST, Devanagari\n(rank fusion)", "teal", 8.4)
    cp = box(ax, 42, 61.5, 24, 5, "Corpus: 701 verses", "grey", 8.6)
    g = box(ax, 72, 50, 24, 9, "LLM answer citing\nverses as BG c.v", "blue", 8.8)
    sg = box(ax, 71, 35, 26, 8, "Claim segmentation and\ncitation extraction", "blue", 8.6)
    checks = ["Citation\nexists?", "Verse entails\nclaim?", "Quotation\nmatches?", "Speaker\ncorrect?"]
    cx = [46, 60, 74, 88]
    cbs = [box(ax, x, 20, 12.5, 8, t, "indigo", 8.4) for x, t in zip(cx, checks)]
    rep = box(ax, 55, 7, 26, 7, "Verification report", "yellow", 9, True)
    pol = box(ax, 86, 7, 26, 9, "Policy: accept, repair,\nregenerate or abstain", "green", 8.8, True)
    arrow(ax, right(q), left(rt))
    arrow(ax, bottom(cp), top(rt))
    arrow(ax, right(rt), left(g))
    arrow(ax, bottom(g), top(sg))
    arrow(ax, bottom(q), top(pr))
    for b in cbs:
        arrow(ax, (b[0], 28), top(b))
    ax.plot([46, 88], [28, 28], color="#333333", lw=1.1)
    ax.plot([71, 71], [31, 28], color="#333333", lw=1.1)
    for b, xe in zip(cbs, (49, 53, 58, 63)):
        arrow(ax, bottom(b), (xe, 10.5), color=GREY)
    poly_arrow(ax, [bottom(pr), (13, 7), left(rep)], color=GREY)
    arrow(ax, right(rep), left(pol))
    save(fig, "vghd_pipeline")


def fig_gita_rag() -> None:
    fig, ax = canvas(6.6, 1.95)
    q = box(ax, 8, 10, 13, 8, "Question\n$q$", "grey", 9, True)
    r = box(ax, 28, 10, 18, 8, "Retriever\n(sparse / dense)", "teal", 9)
    c = box(ax, 28, 22.5, 18, 5.5, "Corpus: 701 verses", "grey", 8.6)
    t = box(ax, 48, 10, 14, 8, "Top-$k$ verses\n$d_1, \\dots, d_k$", "teal", 9)
    g = box(ax, 66, 10, 14, 8, "LLM\ngenerator", "blue", 9)
    a = box(ax, 88, 10, 20, 8, "Answer with\nverse citations", "orange", 9)
    arrow(ax, right(q), left(r))
    arrow(ax, bottom(c), top(r))
    arrow(ax, right(r), left(t))
    arrow(ax, right(t), left(g))
    arrow(ax, right(g), left(a))
    poly_arrow(ax, [(8, 6), (8, 1.8), (66, 1.8), (66, 6)])
    ax.text(37, 3.4, "question passed to the generator", fontsize=8, style="italic", ha="center", color="#444444")
    save(fig, "gita_rag_pipeline")


def fig_gita_example() -> None:
    verses = {v["ref"]: v for v in json.loads((ROOT / "data" / "gita" / "verses.json").read_text(encoding="utf-8"))}
    fig, ax = canvas(6.6, 3.7)
    ax.add_patch(FancyBboxPatch((1, 41), 98, 13.5, boxstyle="round,pad=0,rounding_size=1.2", fc=LIGHT["grey"],
                                ec=EDGE["grey"], lw=1))
    ax.text(3, 51.3, "Question:", fontweight="bold", fontsize=9, va="center")
    ax.text(19.5, 51.3, "Which verse of the Bhagavad Gita says that the soul cannot be cut by weapons?",
            fontsize=8.4, va="center")
    ax.text(3, 47.0, "LLM answer:", fontweight="bold", fontsize=9, va="center")
    ax.text(19.5, 47.0, "The Gita states that the soul cannot be cut by weapons or burnt by fire.", fontsize=9,
            va="center")
    ax.text(19.5, 43.3, "This is described in BG 2.20.", fontsize=9, va="center", color=EDGE["red"],
            fontweight="bold")

    def verse_box(x0, ref, colour, verdict, iast_cut, eng):
        ax.add_patch(FancyBboxPatch((x0, 2), 47, 33, boxstyle="round,pad=0,rounding_size=1.2", fc=LIGHT[colour],
                                    ec=EDGE[colour], lw=1.3))
        ax.text(x0 + 23.5, 31.5, ref, ha="center", fontweight="bold", fontsize=10)
        v = verses[ref.split()[1]]
        ax.text(x0 + 23.5, 25.5, " ".join(v["iast"].split()[:iast_cut]) + " \u2026", ha="center", fontsize=8.6,
                style="italic")
        words, lines, cur = eng.split(), [], ""
        for w_ in words:
            if len(cur) + len(w_) > 44:
                lines.append(cur)
                cur = w_
            else:
                cur = (cur + " " + w_).strip()
        lines.append(cur)
        ax.text(x0 + 23.5, 16.5, "\u201c" + "\n".join(lines) + "\u201d", ha="center", va="center", fontsize=8.6)
        ax.text(x0 + 23.5, 6.0, verdict, ha="center", va="center", fontsize=9, color=EDGE[colour],
                fontweight="bold")

    verse_box(1.5, "BG 2.20 (cited verse)", "red", "claim not entailed: misattributed citation", 4,
              "He is not born, nor doth he die; \u2026 he is not slain when the body is slaughtered.")
    verse_box(51.5, "BG 2.23 (correct verse)", "green", "claim entailed: suggested citation", 4,
              "Weapons cleave him not, nor fire burneth him, nor waters wet him, nor wind drieth him away.")
    arrow(ax, (30, 41), (25, 35.2), color=EDGE["red"])
    arrow(ax, (60, 41), (75, 35.2), color=EDGE["green"], ls="--")
    ax.text(73, 38.4, "verifier searches the corpus", fontsize=8, style="italic", color=EDGE["green"])
    save(fig, "gita_example")


def fig_software_architecture() -> None:
    fig, ax = canvas(6.6, 4.0)
    layers = [
        (52, "Interfaces", "grey", ["CLI: halludetect\nverify, gita-ask,\nugrav-ask, bench-*",
                                    "Streamlit studio\nHome + 8 pages", "Kaggle GPU\nnotebook",
                                    "Scripts: corpus,\ndatasets, real pilot,\nfigures"]),
        (37, "Frameworks", "orange", ["halludetect.ugrav\nsignals \u00b7 fusion \u00b7 policy\nbenchmark \u00b7 pipeline",
                                      "halludetect.gita\ncorpus \u00b7 citations \u00b7 claims\npremise \u00b7 verifier "
                                      "\u00b7 policy",
                                      "simulators\nSimulatedQALLM\nSimulatedGitaLLM"]),
        (22, "Core library", "blue", ["uncertainty\nclustering, SE,\nPE, SC",
                                      "nli\nLexicalNLI,\nHFNLI (DeBERTa)",
                                      "retrieval\nBM25, reciprocal\nrank fusion",
                                      "llm\nHF local, OpenAI-\ncompatible",
                                      "metrics, text\nAUROC, ECE,\nAURAC, CIs"]),
        (7, "Data and results", "green", ["Gita corpus (701 v.)\nQA (106) \u00b7 test set (524)",
                                          "ODQA world (197 q.)\npassages (198)",
                                          "PopQA + Wikipedia\ncache (real pilot)",
                                          "results/*.json\nrecords, benchmarks"]),
    ]
    for y, title, colour, items in layers:
        ax.add_patch(FancyBboxPatch((1, y - 6.6), 98, 13.2, boxstyle="round,pad=0,rounding_size=1.0", fc="white",
                                    ec=EDGE[colour], lw=1.0, ls="--"))
        ax.text(2.5, y + 5.0, title, fontsize=9, fontweight="bold", color=EDGE[colour], va="center")
        n = len(items)
        w = (93 - 2 * (n - 1)) / n
        for i, t in enumerate(items):
            box(ax, 5 + w / 2 + i * (w + 2), y - 0.8, w, 9.6, t, colour, 7.8, round_=0.8)
    for y in (45.2, 30.2, 15.2):
        arrow(ax, (50, y + 0.2), (50, y - 1.7), style="<|-|>")
    save(fig, "software_architecture")


# ---------------------------------------------------------------------------- Gantt charts

STATUS_COL = {"done": INDIGO, "active": ORANGE, "planned": "#9AA5B1", "writing": GREEN, "milestone": RED}


def gantt(tasks, start, end, name, title=None, markers=(), unit="month", width=6.6, legend=None):
    fig, ax = plt.subplots(figsize=(width, 0.36 * len(tasks) + 1.35))
    for i, t in enumerate(tasks):
        label, s, e, status = t[:4]
        y = len(tasks) - 1 - i
        if status == "milestone":
            ax.plot([s], [y], marker="D", color=STATUS_COL[status], ms=7)
        else:
            ax.barh(y, (e - s).days, left=s, height=0.56, color=STATUS_COL[status], edgecolor="white")
        ax.text(start - dt.timedelta(days=2), y, label, ha="right", va="center", fontsize=9)
    ax.set_yticks([])
    ax.set_ylim(-0.7, len(tasks) - 0.3)
    ax.set_xlim(start, end)
    if unit == "month":
        ticks, d = [], dt.date(start.year, start.month, 1)
        while d <= end:
            ticks.append(d)
            d = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([d.strftime("%b\n%Y") if (d.month in (1, 8) or d == ticks[0]) else d.strftime("%b")
                            for d in ticks], fontsize=8.5)
        ax.grid(axis="x", color="#DDDDDD", lw=0.6)
    else:
        ticks = [start + dt.timedelta(weeks=k) for k in range(0, ((end - start).days // 7) + 1)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"W{k + 1}\n{d.strftime('%d %b')}" if k % 2 == 0 else f"W{k + 1}"
                            for k, d in enumerate(ticks)], fontsize=7.2)
        ax.grid(axis="x", color="#E3E3E3", lw=0.5)
    ax.set_axisbelow(True)
    for d, text in markers:
        ax.axvline(d, color=RED, lw=1.1, ls="--")
        ax.text(d, len(tasks) - 0.35, text, color=RED, fontsize=8, ha="center", va="bottom",
                bbox=dict(fc="white", ec="none", pad=0.5))
    ax.spines["left"].set_visible(False)
    if title:
        ax.set_title(title, pad=18)
    from matplotlib.patches import Patch
    present = {t[3] for t in tasks}
    items = legend or [(k, v) for k, v in [("done", "completed"), ("active", "in progress"),
                                           ("planned", "planned"), ("writing", "writing"),
                                           ("milestone", "milestone")] if k in present]
    handles = [Patch(color=STATUS_COL[k], label=v) for k, v in items]
    ax_h = fig.get_figheight() * ax.get_position().height
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -(0.5 if unit == "month" else 0.62) / ax_h),
              ncol=len(handles), fontsize=8.5)
    save(fig, name)


def fig_gantts() -> None:
    D = dt.date
    ugrav = [
        ("Literature survey", D(2026, 8, 3), D(2026, 9, 25), "done"),
        ("Problem formulation and objectives", D(2026, 9, 1), D(2026, 9, 30), "done"),
        ("Design of the UG-RAV framework", D(2026, 9, 15), D(2026, 10, 20), "done"),
        ("Implementation of the framework", D(2026, 10, 15), D(2026, 11, 15), "done"),
        ("Simulation study and benchmark", D(2026, 11, 5), D(2026, 11, 18), "done"),
        ("Real-model pilot (CPU, PopQA)", D(2026, 11, 12), D(2026, 11, 22), "done"),
        ("VGHD extension (corpus, verifier)", D(2026, 10, 20), D(2026, 11, 20), "done"),
        ("Progress report and documentation", D(2026, 11, 10), D(2026, 11, 28), "writing"),
        ("Full-scale GPU runs (3 datasets)", D(2027, 1, 4), D(2027, 3, 5), "planned"),
        ("Ablations, label audit, failure analysis", D(2027, 2, 8), D(2027, 3, 26), "planned"),
        ("Conformal thresholds, extensions", D(2027, 3, 1), D(2027, 4, 2), "planned"),
        ("Dissertation and research paper", D(2027, 2, 15), D(2027, 5, 7), "writing"),
        ("Submission and viva", D(2027, 5, 10), D(2027, 5, 10), "milestone"),
    ]
    gantt(ugrav, D(2026, 8, 1), D(2027, 5, 31), "gantt_ugrav_part1",
          markers=[(D(2026, 11, 23), "end-semester evaluation")])
    vghd = [
        ("Literature survey", D(2026, 8, 3), D(2026, 9, 25), "done"),
        ("Topic finalisation and objectives", D(2026, 9, 1), D(2026, 9, 30), "done"),
        ("Corpus construction (701 verses)", D(2026, 10, 12), D(2026, 11, 2), "done"),
        ("QA dataset and test set", D(2026, 10, 26), D(2026, 11, 12), "done"),
        ("Verifier and repair policy", D(2026, 10, 26), D(2026, 11, 16), "done"),
        ("Benchmark (lexical and DeBERTa NLI)", D(2026, 11, 9), D(2026, 11, 22), "done"),
        ("Progress report", D(2026, 11, 12), D(2026, 11, 28), "writing"),
        ("Real LLM answers + human annotation", D(2027, 1, 11), D(2027, 3, 5), "planned"),
        ("Dense multilingual retrieval", D(2027, 2, 1), D(2027, 3, 12), "planned"),
        ("Integration with UG-RAV", D(2027, 3, 1), D(2027, 3, 31), "planned"),
        ("Dissertation chapter and paper", D(2027, 3, 1), D(2027, 5, 7), "writing"),
    ]
    gantt(vghd, D(2026, 8, 1), D(2027, 5, 31), "gantt_vghd",
          markers=[(D(2026, 11, 23), "progress review")])
    w0 = D(2027, 1, 4)

    def wk(a, b):
        return w0 + dt.timedelta(weeks=a - 1), w0 + dt.timedelta(weeks=b)

    part2 = [
        ("T1 Kaggle setup and smoke test", *wk(1, 1), "planned"),
        ("T2 Data loaders, labels, splits", *wk(1, 2), "planned"),
        ("T3 Closed-book generation (2 models)", *wk(2, 5), "planned"),
        ("T4 RAG generation + BM25 index", *wk(4, 7), "planned"),
        ("T5 NLI signals and caching", *wk(5, 8), "planned"),
        ("T6 Fusion, thresholds, baselines", *wk(7, 9), "planned"),
        ("T7 Label audit (200 answers)", *wk(8, 10), "planned"),
        ("T8 Ablations (K, NLI, retriever)", *wk(10, 12), "planned"),
        ("T9 Distractor study for \u0394SE", *wk(11, 13), "planned"),
        ("T10 Conformal thresholds", *wk(12, 14), "planned"),
        ("T11 VGHD: real LLM answers, annotation", *wk(6, 13), "planned"),
        ("T12 Failure analysis", *wk(13, 15), "planned"),
        ("T13 Dissertation writing", *wk(9, 19), "writing"),
        ("T14 Research paper", *wk(14, 18), "writing"),
        ("T15 Demo, code release, viva slides", *wk(17, 20), "writing"),
        ("Mid-semester review (Part II)", w0 + dt.timedelta(weeks=9, days=3), None, "milestone"),
        ("Pre-submission seminar", w0 + dt.timedelta(weeks=16, days=3), None, "milestone"),
        ("Final viva", w0 + dt.timedelta(weeks=19, days=3), None, "milestone"),
    ]
    gantt(part2, w0, w0 + dt.timedelta(weeks=20), "gantt_part2", unit="week", width=6.8,
          legend=[("planned", "experiments"), ("writing", "writing and release"), ("milestone", "milestone")])


# ---------------------------------------------------------------------------- result figures

SIGNAL_LABELS = {
    "fused_with_evidence": "Fused: uncertainty + evidence",
    "one_minus_evidence_support": "1 \u2212 evidence support of $a_0$",
    "min_token_surprisal": "Min-token surprisal (FLARE)",
    "fused_closed_book": "Fused: closed-book only ($\\hat{r}_0$)",
    "semantic_entropy": "Semantic entropy",
    "predictive_entropy": "Predictive entropy",
    "one_minus_greedy_agreement": "1 \u2212 greedy agreement",
    "one_minus_self_consistency": "1 \u2212 self-consistency",
    "semantic_entropy_weighted": "Semantic entropy (weighted)",
    "one_minus_p_true": "1 \u2212 P(True)",
    "lexical_entropy": "Lexical entropy",
    "length_normalised_pe": "Length-normalised PE",
}


def signal_colour(k: str) -> str:
    return ORANGE if k.startswith("fused") else TEAL if "evidence" in k else INDIGO


def fig_detection(res: dict, name: str, title: str) -> None:
    det = res["detection"]
    keys = sorted(det, key=lambda k: det[k]["auroc"])
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(keys) + 0.9))
    y = np.arange(len(keys))
    vals = [det[k]["auroc"] for k in keys]
    lo = [det[k]["auroc"] - det[k]["auroc_ci"][0] for k in keys]
    hi = [det[k]["auroc_ci"][1] - det[k]["auroc"] for k in keys]
    ax.barh(y, vals, color=[signal_colour(k) for k in keys], height=0.66)
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor="#222222", lw=0.9, capsize=2.5)
    for yi, k in zip(y, keys):
        ax.text(det[k]["auroc_ci"][1] + 0.008, yi, f"{det[k]['auroc']:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([SIGNAL_LABELS.get(k, k) for k in keys], fontsize=9)
    lo_x = max(0.0, min(min(det[k]["auroc_ci"][0] for k in keys) - 0.05, 0.45))
    ax.set_xlim(lo_x, 1.04)
    ax.axvline(0.5, color=GREY, ls="--", lw=0.9)
    ax.set_xlabel("AUROC for detecting a wrong closed-book answer (95% bootstrap CI)")
    if title:
        ax.set_title(title)
    save(fig, name)


def fig_se_by_state() -> None:
    path = RES / "ugrav" / "records_sim-7b.jsonl"
    if not path.exists():
        return
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    order = [("knows", "knows"), ("confident_wrong", "confident error"), ("confused", "confabulation"),
             ("ambiguous", "ambiguous")]
    data = [[r["closed"]["se"] for r in recs if r["meta"]["sim_state"] == s] for s, _ in order]
    acc = [np.mean([r["closed_correct"] for r in recs if r["meta"]["sim_state"] == s]) for s, _ in order]
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="black"),
                    flierprops=dict(marker=".", ms=3, color=GREY))
    for patch, c in zip(bp["boxes"], [GREEN, RED, ORANGE, INDIGO]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for i, (d, a) in enumerate(zip(data, acc)):
        ax.text(i + 1, 2.45, f"n = {len(d)}\nmean SE = {np.mean(d):.2f}\naccuracy = {a:.2f}", ha="center",
                fontsize=8)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([lbl for _, lbl in order])
    ax.set_ylabel("Semantic entropy (nats)")
    ax.set_ylim(-0.1, 3.0)
    ax.set_xlabel("Hidden knowledge state of the simulator")
    save(fig, "sim_se_by_state")


POLICY_LABELS = {
    "never_retrieve": "Never retrieve", "always_retrieve": "Always retrieve", "sc_abstention": "SC abstention",
    "flare_style": "FLARE-style gate", "se_gated_no_verification": "SE gate, no verification",
    "ug_rav": "UG-RAV", "ug_rav_without_uncertainty_shift": "UG-RAV \u2212 post-retrieval unc.",
    "ug_rav_without_evidence_signals": "UG-RAV \u2212 evidence signals", "oracle": "Oracle",
}


def fig_policies(res: dict, name: str) -> None:
    pol = res["policies_test"]
    keys = [k for k in ["never_retrieve", "always_retrieve", "sc_abstention", "flare_style",
                        "se_gated_no_verification", "ug_rav", "oracle"] if k in pol]
    metrics = [("hallucination_rate", "Hallucination rate (wrong / all)"), ("coverage", "Coverage"),
               ("retrieval_rate", "Retrieval rate")]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.9), sharey=True)
    y = np.arange(len(keys))[::-1]
    for ax, (m, lbl) in zip(axes, metrics):
        vals = [pol[k][m] for k in keys]
        cols = [ORANGE if k == "ug_rav" else GREY if k == "oracle" else INDIGO for k in keys]
        ax.barh(y, vals, color=cols, height=0.65)
        for yi, v in zip(y, vals):
            ax.text(v + 0.02, yi, f"{v:.3f}", va="center", fontsize=7.8)
        ax.set_xlim(0, 1.25 if m != "hallucination_rate" else max(vals) * 1.45 + 0.02)
        ax.set_title(lbl, fontsize=9.5)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([POLICY_LABELS[k] for k in keys], fontsize=8.8)
    fig.tight_layout(w_pad=0.6)
    save(fig, name)


def fig_risk_coverage(res: dict, name: str) -> None:
    rc = res["risk_coverage_test"]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    style = {"never_retrieve": (INDIGO, "--"), "always_retrieve": (TEAL, "--"), "sc_abstention": (GREY, ":"),
             "flare_style": ("#7B61C9", "-."), "se_gated_no_verification": ("#2F5F98", "-."),
             "ug_rav": (ORANGE, "-"), "oracle": ("#222222", ":")}
    for k, (c, ls) in style.items():
        if k in rc:
            lw = 2.2 if k == "ug_rav" else 1.3
            aur = res["policies_test"][k]["aurac"]
            ax.plot(rc[k]["coverage"], rc[k]["risk"], color=c, ls=ls, lw=lw,
                    label=f"{POLICY_LABELS[k]} (AURAC {aur:.3f})")
    ax.set_xlabel("Coverage (fraction of questions answered)")
    ax.set_ylabel("Selective risk (error rate on answered)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.35, max(max(v["risk"]) for v in rc.values()) * 1.05))
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(color="#EEEEEE", lw=0.6)
    save(fig, name)


def fig_gate_budget(res: dict, name: str) -> None:
    gb = res["gate_budget_test"]
    labels = {"fused_closed_book_risk": ("Fused $\\hat{r}_0$ gate (UG-RAV)", ORANGE, "o"),
              "flare_min_token_prob": ("Min-token probability (FLARE)", "#7B61C9", "s"),
              "semantic_entropy": ("Semantic entropy (SUGAR-like)", INDIGO, "^"),
              "random": ("Random gate", GREY, "x")}
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    for k, (lbl, c, mk) in labels.items():
        if k in gb:
            ax.plot([p["retrieval_rate"] for p in gb[k]], [p["accuracy"] for p in gb[k]], marker=mk, color=c,
                    label=lbl, lw=1.6 if k.startswith("fused") else 1.2)
    ax.set_xlabel("Retrieval rate (fraction of questions sent to retrieval)")
    ax.set_ylabel("Accuracy (answer from evidence when gated)")
    ax.grid(color="#EEEEEE", lw=0.6)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, name)


def fig_popularity(res: dict, name: str) -> None:
    pop = res["popularity"]
    tiers = [t for t in ("high", "mid", "low") if t in pop]
    x = np.arange(len(tiers))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    ax.bar(x - 0.2, [pop[t]["closed_book_accuracy"] for t in tiers], 0.38, color=INDIGO, label="closed book")
    ax.bar(x + 0.2, [pop[t]["rag_accuracy"] for t in tiers], 0.38, color=TEAL, label="with retrieval")
    for i, t in enumerate(tiers):
        ax.text(i - 0.2, pop[t]["closed_book_accuracy"] + 0.02, f"{pop[t]['closed_book_accuracy']:.2f}",
                ha="center", fontsize=7.8)
        ax.text(i + 0.2, pop[t]["rag_accuracy"] + 0.02, f"{pop[t]['rag_accuracy']:.2f}", ha="center", fontsize=7.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n = {pop[t]['n']})" for t in tiers])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by popularity tier", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax = axes[1]
    ax.bar(x - 0.2, [pop[t]["retrieval_helps"] for t in tiers], 0.38, color=GREEN, label="retrieval helps")
    ax.bar(x + 0.2, [pop[t]["retrieval_hurts"] for t in tiers], 0.38, color=RED, label="retrieval hurts")
    ax.plot(x, [pop[t]["ugrav_retrieval_rate"] for t in tiers], color=ORANGE, marker="o", lw=1.4,
            label="UG-RAV retrieval rate")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylim(0, 1.15)
    ax.set_title("When retrieval helps or hurts", fontsize=10)
    ax.legend(fontsize=7.6, loc="upper left")
    fig.tight_layout(w_pad=1.5)
    save(fig, name)


FEATURE_LABELS = {
    "se": "SE (closed)", "one_minus_sc": "1 \u2212 SC", "one_minus_agreement": "1 \u2212 greedy agreement",
    "one_minus_p_true": "1 \u2212 P(True)", "neg_log_min_tok": "\u2212log min-token prob.", "ln_pe": "LN-PE",
    "se_rag": "SE$_{RAG}$", "delta_se": "$\\Delta$SE", "support_rag": "Sup($a_D$)", "contradiction_rag": "Con($a_D$)",
    "support_closed": "Sup($a_0$)", "contradiction_closed": "Con($a_0$)", "agreement": "agreement",
    "one_minus_sc_rag": "1 \u2212 SC$_{RAG}$",
}


def fig_fusion_weights(res: dict, name: str) -> None:
    fw = res["fusion_weights"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    for ax, key, title in [(axes[0], "closed", "Closed-book risk $\\hat{r}_0$"),
                           (axes[1], "rag", "Evidence-grounded risk $\\hat{r}$")]:
        w = {k: v for k, v in fw[key].items() if k != "bias"}
        ks = sorted(w, key=lambda k: w[k])
        cols = [TEAL if k in ("support_rag", "contradiction_rag", "support_closed", "contradiction_closed",
                              "agreement") else ORANGE if k in ("se_rag", "delta_se", "one_minus_sc_rag")
                else INDIGO for k in ks]
        ax.barh(range(len(ks)), [w[k] for k in ks], color=cols, height=0.65)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([FEATURE_LABELS.get(k, k) for k in ks], fontsize=8.5)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("standardised weight", fontsize=8.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, name)


DET_LABELS = {"citation_existence": "Citation existence only", "answer_level_nli": "Answer-level NLI",
              "vghd_no_speaker": "VGHD \u2212 speaker check", "vghd_no_premise": "VGHD \u2212 premise check",
              "vghd_no_quotes": "VGHD \u2212 quotation check", "vghd": "VGHD (full)"}
TYPE_LABELS = {"fabricated_citation": "Fabricated citation", "misattributed_citation": "Misattributed citation",
               "misquotation": "Misquotation", "wrong_speaker": "Wrong speaker",
               "unsupported_claim": "Unsupported claim", "false_premise_accepted": "False premise accepted",
               "out_of_scope_answered": "Out-of-scope answered"}


def fig_vghd_detectors(g: dict, name: str, title: str | None = None) -> None:
    det = g["detectors"]
    keys = [k for k in DET_LABELS if k in det]
    metrics = [("auroc", "AUROC", INDIGO), ("f1", "F1", TEAL), ("false_positive_rate", "FPR on correct", RED)]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(keys))
    for i, (m, lbl, c) in enumerate(metrics):
        vals = [det[k][m] for k in keys]
        ax.bar(x + (i - 1) * 0.27, vals, 0.26, color=c, label=lbl)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * 0.27, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([DET_LABELS[k].replace(" \u2212 ", "\n\u2212 ").replace(" only", "\nonly")
                        .replace("-level NLI", "-level\nNLI") for k in keys], fontsize=8)
    ax.set_ylim(0, 1.22)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    if title:
        ax.set_title(title)
    save(fig, name)


def fig_vghd_per_type(g: dict, name: str) -> None:
    det = g["detectors"]
    keys = [k for k in DET_LABELS if k in det]
    types = list(TYPE_LABELS)
    m = np.array([[det[k]["per_type"][t]["recall"] for k in keys] for t in types])
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    im = ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(types)):
        for j in range(len(keys)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if 0.25 < m[i, j] < 0.85 else "white" if m[i, j] <= 0.25 else "black")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([DET_LABELS[k].replace(" \u2212 ", "\n\u2212 ").replace(" only", "\nonly")
                        .replace("-level NLI", "-level\nNLI") for k in keys], fontsize=7.8)
    n = {t: det["vghd"]["per_type"][t]["n"] for t in types}
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{TYPE_LABELS[t]} ({n[t]})" for t in types], fontsize=8.5)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("recall", fontsize=8)
    save(fig, name)


def fig_vghd_repair(g: dict, name: str) -> None:
    rep = g["repair"]
    types = [t for t in TYPE_LABELS if t in rep]
    cats = [("repaired_and_verified", "repaired and verified", GREEN), ("abstained", "abstained", INDIGO),
            ("repair_failed", "repair failed", ORANGE), ("missed", "missed (not detected)", RED)]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    y = np.arange(len(types))[::-1]
    leftv = np.zeros(len(types))
    for key, lbl, c in cats:
        vals = np.array([rep[t].get(key, 0) for t in types], dtype=float)
        ax.barh(y, vals, left=leftv, color=c, label=lbl, height=0.62)
        for yi, lv, v in zip(y, leftv, vals):
            if v >= 3:
                ax.text(lv + v / 2, yi, f"{int(v)}", ha="center", va="center", fontsize=7.8, color="white")
        leftv += vals
    ax.set_yticks(y)
    ax.set_yticklabels([TYPE_LABELS[t] for t in types], fontsize=8.8)
    ax.set_xlabel("Hallucinated answers in the test split")
    ax.legend(ncol=4, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.42, -0.2))
    save(fig, name)


def fig_vghd_retrieval(g: dict, name: str) -> None:
    r = g["retrieval"]
    groups = [("all", "All"), ("en", "English"), ("hi", "Hindi")]
    ms = [("hit_at_1", "Hit@1", INDIGO), ("recall_at_5", "Recall@5", TEAL), ("mrr", "MRR", ORANGE)]
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    x = np.arange(len(groups))
    for i, (m, lbl, c) in enumerate(ms):
        vals = [r[gk][m] for gk, _ in groups]
        ax.bar(x + (i - 1) * 0.26, vals, 0.25, color=c, label=lbl)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * 0.26, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl} (n = {r[gk]['n']})" for gk, lbl in groups])
    ax.set_ylim(0, 1.18)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    save(fig, name)


def fig_nli_comparison(lex: dict, deb: dict, name: str) -> None:
    keys = [k for k in DET_LABELS if k in lex["detectors"] and k in deb["detectors"]]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.8), sharey=True)
    y = np.arange(len(keys))[::-1]
    for ax, (m, lbl) in zip(axes, [("auroc", "AUROC"), ("f1", "F1"), ("false_positive_rate", "FPR on correct")]):
        a = [lex["detectors"][k][m] for k in keys]
        b = [deb["detectors"][k][m] for k in keys]
        ax.barh(y + 0.19, a, 0.36, color=INDIGO, label="lexical NLI")
        ax.barh(y - 0.19, b, 0.36, color=TEAL, label="DeBERTa-large-MNLI")
        for yi, va, vb in zip(y, a, b):
            ax.text(va + 0.015, yi + 0.19, f"{va:.2f}", va="center", fontsize=6.8)
            ax.text(vb + 0.015, yi - 0.19, f"{vb:.2f}", va="center", fontsize=6.8)
        ax.set_title(lbl, fontsize=9.5)
        ax.set_xlim(0, 1.2)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([DET_LABELS[k] for k in keys], fontsize=8.2)
    axes[0].legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(-0.9, -0.32), ncol=2)
    fig.tight_layout(w_pad=0.5)
    save(fig, name)


def fig_sim_vs_real(sim: dict, real: dict, name: str) -> None:
    keys = [k for k in SIGNAL_LABELS if k in sim["detection"] and k in real["detection"]]
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(keys) + 1.0))
    y = np.arange(len(keys))[::-1]
    for off, res, c, lbl in [(0.19, sim, GREY, "simulation (985 instances)"),
                             (-0.19, real, ORANGE, "real pilot")]:
        v = [res["detection"][k]["auroc"] for k in keys]
        lo = [res["detection"][k]["auroc"] - res["detection"][k]["auroc_ci"][0] for k in keys]
        hi = [res["detection"][k]["auroc_ci"][1] - res["detection"][k]["auroc"] for k in keys]
        ax.barh(y + off, v, 0.36, color=c, label=lbl)
        ax.errorbar(v, y + off, xerr=[lo, hi], fmt="none", ecolor="#333333", lw=0.7, capsize=1.8)
    ax.set_yticks(y)
    ax.set_yticklabels([SIGNAL_LABELS[k] for k in keys], fontsize=8.6)
    ax.axvline(0.5, color=GREY, ls="--", lw=0.9)
    ax.set_xlim(0.2, 1.05)
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, name)


def main() -> None:
    print("Diagrams")
    fig_taxonomy({"inconsistency", "fabrication"}, "taxonomy_ugrav",
                 "shaded: factuality hallucination, the focus of UG-RAV")
    fig_taxonomy({"fabrication", "context"}, "taxonomy_vghd",
                 "shaded: the types most relevant to retrieval-augmented QA on a scripture")
    fig_taxonomy({"inconsistency", "fabrication", "context"}, "taxonomy_combined",
                 "shaded: types addressed in this dissertation (UG-RAV: factuality; VGHD: fabrication and context)")
    fig_categorisation({"root": "Handling hallucination in LLMs", "branches": [
        ("Detection", [("Uncertainty-\nbased\n(entropy, SE)", True),
                       ("Consistency-\nbased\n(SelfCheckGPT)", True),
                       ("Evidence-\nbased\n(NLI, MiniCheck)", True)]),
        ("Mitigation", [("Adaptive\nretrieval\n(FLARE, Rowen)", True),
                        ("Training-\nbased\n(Self-RAG)", False),
                        ("Abstention\n(selective\nanswering)", True)])]},
        "categorisation_ugrav", "shaded: training-free approaches combined in UG-RAV")
    fig_categorisation({"root": "Hallucination detection approaches", "branches": [
        ("Reference-free", [("Uncertainty-\nbased\n(entropy, SE)", False),
                            ("Consistency-\nbased\n(SelfCheckGPT)", False),
                            ("Internal-state\n(hidden-state\nprobes)", False)]),
        ("Reference-based", [("Faithfulness\n(NLI, MiniCheck,\nFActScore)", True),
                             ("Attribution /\ncitation\n(AIS, ALCE)", True),
                             ("LLM-as-judge\n(RAGAS)", False)])]},
        "categorisation_vghd", "shaded: approaches most relevant to verse-grounded detection")
    fig_ugrav_architecture()
    fig_se_godan()
    fig_exp_pipeline()
    fig_vghd_pipeline()
    fig_gita_rag()
    fig_gita_example()
    fig_software_architecture()
    print("Work plans")
    fig_gantts()
    print("Results")
    sim = load(RES / "ugrav" / "benchmark_sim-7b.json")
    if sim:
        fig_detection(sim, "sim_detection_auroc", None)
        fig_se_by_state()
        fig_policies(sim, "sim_policies")
        fig_risk_coverage(sim, "sim_risk_coverage")
        fig_gate_budget(sim, "sim_gate_budget")
        fig_popularity(sim, "sim_popularity")
        fig_fusion_weights(sim, "sim_fusion_weights")
    real = load(RES / "real" / "benchmark_popqa_qwen2.5-1.5b-instruct.json")
    if real:
        fig_detection(real, "real_detection_auroc", None)
        fig_policies(real, "real_policies")
        fig_risk_coverage(real, "real_risk_coverage")
        if sim:
            fig_sim_vs_real(sim, real, "sim_vs_real_auroc")
    lex = load(RES / "gita" / "benchmark_lexical.json")
    if lex:
        fig_vghd_detectors(lex, "vghd_lexical_detectors")
        fig_vghd_per_type(lex, "vghd_lexical_per_type")
        fig_vghd_repair(lex, "vghd_lexical_repair")
        fig_vghd_retrieval(lex, "vghd_retrieval")
    deb = load(RES / "gita" / "benchmark_microsoft_deberta_large_mnli.json")
    if deb:
        fig_vghd_per_type(deb, "vghd_deberta_per_type")
        fig_vghd_repair(deb, "vghd_deberta_repair")
        if lex:
            fig_nli_comparison(lex, deb, "vghd_nli_comparison")


if __name__ == "__main__":
    main()
