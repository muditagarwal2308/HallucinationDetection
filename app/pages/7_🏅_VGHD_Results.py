"""VGHD Results — detector comparison, per-type recall, ablations, tuning and repair outcomes."""

from __future__ import annotations

import re
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

C.setup_page(
    "VGHD Results", "🏅",
    "How does verse-grounded verification compare with the checks a general RAG pipeline would use? Detectors are "
    "evaluated on a synthetic test set of correct and hallucinated answers to Gita questions, covering seven "
    "hallucination types; repair and abstention are evaluated on the detected hallucinations.",
    eyebrow="VGHD · benchmark",
)

TYPE_ORDER = ["fabricated_citation", "misattributed_citation", "misquotation", "wrong_speaker", "unsupported_claim",
              "false_premise_accepted", "out_of_scope_answered"]
MAIN = ["citation_existence", "answer_level_nli", "vghd"]
ABLATIONS = ["vghd_no_premise", "vghd_no_quotes", "vghd_no_speaker"]
DET_COLORS = {"citation_existence": C.SLATE, "answer_level_nli": C.SAFFRON, "vghd": C.INDIGO,
              "vghd_no_premise": "#818CF8", "vghd_no_quotes": "#A5B4FC", "vghd_no_speaker": "#C7D2FE"}
NLI_PRETTY = {"lexical": "Lexical NLI", "microsoft/deberta-large-mnli": "DeBERTa-large-MNLI"}


def what_this_shows(text: str) -> None:
    st.caption(f"**What this shows —** {text}")


def split_of(qid: str) -> str:
    digits = re.sub(r"\D", "", qid or "") or "0"
    return "dev" if int(digits) % 2 == 1 else "test"


runs = C.gita_results()
if not runs:
    C.missing_results_note("halludetect bench-gita", "VGHD benchmark results")
    C.footer()
    st.stop()

keys = list(runs)
default = "lexical" if "lexical" in runs else keys[0]
sel = st.segmented_control("NLI back-end of the run", keys, default=default, required=True, key="vr_nli",
                           format_func=lambda k: NLI_PRETTY.get(k, k))
R = runs[sel]
det = R["detectors"]
if len(runs) < 2:
    st.caption("Only one NLI back-end has been benchmarked. Run `halludetect bench-gita --nli "
               "microsoft/deberta-large-mnli` to add the DeBERTa comparison.")

C.callout(
    "<b>Protocol.</b> Items are split by question id: odd ids form the <b>dev</b> split, even ids the held-out "
    f"<b>test</b> split. On dev, VGHD's support threshold (chosen: <b>{C.fmt(R.get('support_threshold'), 2)}</b>) "
    "and the answer-level baseline's decision threshold (chosen: "
    f"<b>{C.fmt(R.get('answer_level_threshold'), 2)}</b>) are tuned by <b>Youden's J</b> = TPR − FPR, which is "
    "insensitive to the 4 : 1 class imbalance. All numbers below are on the test split unless noted.<br><br>"
    "<b>Caveat.</b> The rules of the offline <i>lexical</i> NLI were refined while inspecting this synthetic test "
    "set, so its scores are optimistic. The DeBERTa run uses an off-the-shelf model without any tuning to the "
    "test set and is the fairer estimate of out-of-the-box performance.",
)

v = det.get("vghd", {})
alb = det.get("answer_level_nli", {})
C.metric_row([
    ("VGHD F1", C.fmt(v.get("f1")), f"answer-level NLI: {C.fmt(alb.get('f1'))}"),
    ("VGHD AUROC", C.fmt(v.get("auroc")), f"answer-level NLI: {C.fmt(alb.get('auroc'))}"),
    ("VGHD recall", C.pct(v.get("recall")), f"of {v.get('positives', '—')} hallucinated test answers"),
    ("False-positive rate", C.pct(v.get("false_positive_rate")),
     f"answer-level NLI: {C.pct(alb.get('false_positive_rate'))}"),
    ("All items (dev + test)", C.fmt(R.get("vghd_all_items", {}).get("f1")),
     f"F1 on {R.get('vghd_all_items', {}).get('n', '—')} answers"),
])
st.space("small")

tab_names = ["🏁 Detectors", "🧬 Per-type recall", "🧪 Ablations", "🎚️ Dev tuning", "🛠️ Repair & abstain"]
if len(runs) >= 2:
    tab_names.append("⚖️ Lexical vs DeBERTa")
tab_names.append("🗂️ Test set")
tabs = dict(zip(tab_names, st.tabs(tab_names)))

# ------------------------------------------------------------------ detectors
with tabs["🏁 Detectors"]:
    names = [d for d in MAIN + ABLATIONS if d in det] + [d for d in det if d not in MAIN + ABLATIONS]
    ddf = pd.DataFrame([{
        "Detector": C.label(d, C.DETECTOR_LABELS), "AUROC": det[d].get("auroc"), "AUPRC": det[d].get("auprc"),
        "Precision": det[d].get("precision"), "Recall": det[d].get("recall"), "F1": det[d].get("f1"),
        "FPR (correct answers flagged)": det[d].get("false_positive_rate"), "Accuracy": det[d].get("accuracy"),
    } for d in names])
    rules = {"AUROC": True, "AUPRC": True, "Precision": True, "Recall": True, "F1": True,
             "FPR (correct answers flagged)": False, "Accuracy": True}

    def _style(col: pd.Series) -> list[str]:
        best = col.max() if rules[col.name] else col.min()
        return ["background-color:#DCFCE7;color:#14532D;font-weight:700" if abs(x - best) < 1e-9 else ""
                for x in col]

    st.dataframe(ddf.style.apply(_style, subset=list(rules), axis=0).format({c: "{:.3f}" for c in rules}),
                 hide_index=True, width="stretch")
    fig = go.Figure()
    for d in names:
        fig.add_bar(x=["F1", "AUROC", "FPR"], y=[det[d].get("f1"), det[d].get("auroc"),
                                                  det[d].get("false_positive_rate")],
                    name=C.label(d, C.DETECTOR_LABELS), marker_color=DET_COLORS.get(d, C.SLATE))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.05], gridcolor=C.GRID)
    C.show_fig(C.style_fig(fig, height=400, title="F1, AUROC and false-positive rate"), key="vr_bars")
    what_this_shows(
        "a citation-existence check is precise but only catches invented verse numbers; an answer-level "
        "faithfulness check (what RAGAS-style RAG evaluation sees) flags many correct answers because it ignores "
        "which verse each claim cites. Checking every claim against *its own* cited verse gives VGHD both high "
        "recall and a low false-positive rate.")

# ------------------------------------------------------------------ per type
with tabs["🧬 Per-type recall"]:
    metric = st.radio("Cell value", ["recall", "type_accuracy"], horizontal=True, key="vr_heat_metric",
                      format_func={"recall": "Recall — the answer is flagged",
                                   "type_accuracy": "Type accuracy — flagged with the right type"}.get)
    names = [d for d in MAIN + ABLATIONS if d in det]
    types = [t for t in TYPE_ORDER if any(t in det[d].get("per_type", {}) for d in names)]
    z = [[det[d].get("per_type", {}).get(t, {}).get(metric) for t in types] for d in names]
    counts = {t: next((det[d]["per_type"][t]["n"] for d in names if t in det[d].get("per_type", {})), None)
              for t in types}
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{C.label(t, C.ISSUE_LABELS)}<br>(n = {counts[t]})" for t in types],
        y=[C.label(d, C.DETECTOR_LABELS) for d in names], zmin=0, zmax=1,
        colorscale=[[0, "#FFF7ED"], [0.5, "#FCD34D"], [1, "#4338CA"]],
        text=[[("—" if x is None else f"{x:.0%}") for x in row] for row in z], texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}<br>%{z:.0%}<extra></extra>", colorbar=dict(tickformat=".0%"),
    ))
    fig.update_yaxes(autorange="reversed")
    C.show_fig(C.style_fig(fig, height=120 + 52 * len(names), legend=False), key="vr_heat")
    what_this_shows(
        "which hallucination types each detector catches (test split). Each VGHD check owns a type: the premise "
        "check covers false premises and out-of-scope questions, the quotation check misquotations, the speaker "
        "check wrong speakers. Switch to *type accuracy* to see that the ablation rows lose exactly those types, "
        "even where other checks still flag the answer.")

# ------------------------------------------------------------------ ablations
with tabs["🧪 Ablations"]:
    present = [a for a in ABLATIONS if a in det]
    if not present or "vghd" not in det:
        st.info("This run has no ablation results.")
    else:
        rows = []
        for a in present:
            rows.append({"Ablation": C.label(a, C.DETECTOR_LABELS),
                         "ΔF1": det[a]["f1"] - v["f1"], "ΔAUROC": det[a]["auroc"] - v["auroc"],
                         "ΔRecall": det[a]["recall"] - v["recall"],
                         "ΔFPR": det[a]["false_positive_rate"] - v["false_positive_rate"]})
        adf = pd.DataFrame(rows)
        fig = go.Figure()
        for metric_name, color in (("ΔF1", C.INDIGO), ("ΔAUROC", C.TEAL), ("ΔRecall", C.SAFFRON), ("ΔFPR", C.RED)):
            fig.add_bar(x=adf["Ablation"], y=adf[metric_name], name=metric_name, marker_color=color,
                        text=[f"{x:+.3f}" for x in adf[metric_name]], textposition="outside", cliponaxis=False)
        fig.add_hline(y=0, line_color="#9CA3AF")
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="change vs full VGHD", gridcolor=C.GRID)
        C.show_fig(C.style_fig(fig, height=400, title="Impact of removing one check"), key="vr_ablation")
        lost = []
        for a in present:
            for t, s in det[a].get("per_type", {}).items():
                full = v.get("per_type", {}).get(t, {})
                if not full:
                    continue
                if full["recall"] - s["recall"] > 0.05 or full["type_accuracy"] - s["type_accuracy"] > 0.05:
                    lost.append({"Ablation": C.label(a, C.DETECTOR_LABELS), "Type": C.label(t, C.ISSUE_LABELS),
                                 "Recall": f"{full['recall']:.0%} → {s['recall']:.0%}",
                                 "Type accuracy": f"{full['type_accuracy']:.0%} → {s['type_accuracy']:.0%}"})
        if lost:
            st.dataframe(pd.DataFrame(lost), hide_index=True, width="stretch")
        what_this_shows(
            "every check earns its place. Without the premise check, problematic questions are often still flagged "
            "through other issues but are no longer recognised as false premises or out-of-scope, and correct "
            "refusals start to be flagged (higher FPR). Without the speaker check, wrong-speaker answers slip "
            "through. Without the quotation check, misquotations are still flagged as unsupported but not "
            "recognised as misquotations, so they cannot be repaired precisely.")

# ------------------------------------------------------------------ dev tuning
with tabs["🎚️ Dev tuning"]:
    tuning = R.get("tuning", {})
    if not tuning:
        st.info("This run has no tuning record.")
    else:
        ths = sorted(tuning, key=float)
        x = [float(t) for t in ths]
        tpr = [tuning[t]["recall"] for t in ths]
        fpr = [tuning[t]["false_positive_rate"] for t in ths]
        f1 = [tuning[t]["f1"] for t in ths]
        j = [a - b for a, b in zip(tpr, fpr)]
        fig = go.Figure()
        for name, ys, color, dash in (("TPR (recall)", tpr, C.GREEN, "solid"), ("FPR", fpr, C.RED, "solid"),
                                      ("F1", f1, C.INDIGO, "solid"), ("Youden's J = TPR − FPR", j, C.SAFFRON, "dash")):
            fig.add_scatter(x=x, y=ys, mode="lines+markers", name=name, line=dict(color=color, width=3, dash=dash))
        chosen = R.get("support_threshold")
        if chosen is not None:
            fig.add_vline(x=float(chosen), line_dash="dot", line_color=C.INDIGO_DARK,
                          annotation_text=f"chosen {float(chosen):.2f}", annotation_position="top right")
        fig.update_xaxes(title="support threshold (entailment needed to accept a cited verse)", gridcolor=C.GRID)
        fig.update_yaxes(range=[0, 1.05], gridcolor=C.GRID)
        C.show_fig(C.style_fig(fig, height=400, title="Tuning on the dev split"), key="vr_tuning")
        what_this_shows("how strict the entailment requirement should be. A higher threshold rejects more cited "
                        "verses (more recall, more false alarms); the threshold maximising Youden's J on the dev "
                        "split is then frozen for the test split.")

# ------------------------------------------------------------------ repair
with tabs["🛠️ Repair & abstain"]:
    repair = R.get("repair", {})
    if not repair:
        st.info("This run has no repair evaluation.")
    else:
        types = [t for t in TYPE_ORDER if t in repair] + [t for t in repair if t not in TYPE_ORDER]
        outcomes = [("repaired_and_verified", "Repaired and re-verified", C.GREEN),
                    ("abstained", "Abstained", C.INDIGO), ("repair_failed", "Repair failed verification", C.AMBER),
                    ("missed", "Not detected", C.RED)]
        fig = go.Figure()
        for key, name, color in outcomes:
            fig.add_bar(y=[C.label(t, C.ISSUE_LABELS) for t in types], x=[repair[t].get(key, 0) for t in types],
                        name=name, marker_color=color, orientation="h")
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="hallucinated test answers", gridcolor=C.GRID)
        fig.update_yaxes(autorange="reversed")
        C.show_fig(C.style_fig(fig, height=380, title="What happens to each detected hallucination"),
                   key="vr_repair")
        total = sum(sum(repair[t].get(k, 0) for k, _, _ in outcomes) for t in types)
        resolved = sum(repair[t].get("resolved", 0) for t in types)
        st.metric("Resolved (repaired and verified, or correctly abstained)",
                  C.pct(resolved / total if total else None), help=f"{resolved} of {total} hallucinated test answers.")
        what_this_shows(
            "citation, speaker and unsupported-claim errors are repaired deterministically and pass re-verification; "
            "false-premise and out-of-scope answers are replaced by an abstention that explains the problem. "
            "Misquotation repair restores the closest span of the verse, which occasionally still fails the "
            "re-check.")

        examples = R.get("repair_examples", [])
        if examples:
            st.markdown("##### Repair examples")
            i = st.selectbox("Example", range(len(examples)), key="vr_example",
                             format_func=lambda k: f"{k + 1}. {C.label(examples[k]['type'], C.ISSUE_LABELS)} — "
                                                   f"{'; '.join(examples[k].get('edits', []))[:80]}")
            ex = examples[i]
            C.diff_block(ex["before"], ex["after"], "Hallucinated answer", "Repaired answer")
            st.caption("Edits: " + " · ".join(ex.get("edits", [])))

# ------------------------------------------------------------------ NLI comparison
if "⚖️ Lexical vs DeBERTa" in tabs:
    with tabs["⚖️ Lexical vs DeBERTa"]:
        metric = st.radio("Metric", ["f1", "auroc", "false_positive_rate", "recall"], horizontal=True, key="vr_cmp",
                          format_func={"f1": "F1", "auroc": "AUROC", "false_positive_rate": "FPR",
                                       "recall": "Recall"}.get)
        fig = go.Figure()
        palette = [C.INDIGO, C.SAFFRON, C.TEAL, C.ROSE]
        for n_i, (nli, run) in enumerate(runs.items()):
            names = [d for d in MAIN + ABLATIONS if d in run["detectors"]]
            fig.add_bar(x=[C.label(d, C.DETECTOR_LABELS) for d in names],
                        y=[run["detectors"][d].get(metric) for d in names], name=NLI_PRETTY.get(nli, nli),
                        marker_color=palette[n_i % len(palette)])
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 1.05], gridcolor=C.GRID)
        fig.update_xaxes(tickangle=-25)
        C.show_fig(C.style_fig(fig, height=420, title="The same detectors with two NLI back-ends"), key="vr_cmp_fig")
        cmp = pd.DataFrame([{"NLI": NLI_PRETTY.get(nli, nli), "Support threshold": run.get("support_threshold"),
                             "VGHD F1": run["detectors"].get("vghd", {}).get("f1"),
                             "VGHD AUROC": run["detectors"].get("vghd", {}).get("auroc"),
                             "VGHD FPR": run["detectors"].get("vghd", {}).get("false_positive_rate"),
                             "Answer-level F1": run["detectors"].get("answer_level_nli", {}).get("f1"),
                             "Runtime (s)": run.get("runtime_seconds")} for nli, run in runs.items()])
        st.dataframe(cmp, hide_index=True, width="stretch")
        what_this_shows("whether the gains come from the verse-level design or from the NLI model. The lexical NLI "
                        "was refined on this test set; DeBERTa was not, so its numbers are the conservative "
                        "estimate.")

# ------------------------------------------------------------------ test set
with tabs["🗂️ Test set"]:
    rows = C.load_jsonl(C.data_path("gita", "hallucination_testset.jsonl"))
    if not rows:
        st.info("`data/gita/hallucination_testset.jsonl` was not found.")
    else:
        tdf = pd.DataFrame(rows)
        tdf["split"] = tdf["qid"].map(split_of)
        tdf["Type"] = tdf["type"].map(lambda t: C.label(t, C.ISSUE_LABELS))
        comp = tdf.groupby(["Type", "split"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for split, color in (("dev", C.SAFFRON), ("test", C.INDIGO)):
            if split in comp.columns:
                fig.add_bar(y=comp.index, x=comp[split], name=f"{split} split", orientation="h", marker_color=color)
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="answers", gridcolor=C.GRID)
        C.show_fig(C.style_fig(fig, height=360, title=f"Composition of the test set ({len(tdf)} answers)"),
                   key="vr_comp")
        what_this_shows("correct answers and correct refusals are the negatives; each hallucinated answer is a "
                        "controlled perturbation of a reference answer, so every type is measured in isolation.")
        f1c, f2c = st.columns([3, 1])
        chosen_types = f1c.multiselect("Types", sorted(tdf["Type"].unique()), key="vr_types")
        split_f = f2c.selectbox("Split", ["all", "dev", "test"], key="vr_split")
        view = tdf
        if chosen_types:
            view = view[view["Type"].isin(chosen_types)]
        if split_f != "all":
            view = view[view["split"] == split_f]
        st.dataframe(view[["id", "Type", "label", "split", "question", "answer"]], hide_index=True,
                     width="stretch", height=380, column_config={
                         "label": st.column_config.NumberColumn("hallucinated", help="1 = hallucinated"),
                         "answer": st.column_config.TextColumn(width="large"),
                         "question": st.column_config.TextColumn(width="medium")})
        st.caption("Copy any question and answer into the **Gita Verse Verifier** to see the full report.")

C.footer()
