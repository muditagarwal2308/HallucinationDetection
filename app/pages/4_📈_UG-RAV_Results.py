"""UG-RAV Results — detection, policies, retrieval budgets, popularity and fusion analysis."""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

C.setup_page(
    "UG-RAV Results", "📈",
    "How well do the uncertainty signals detect wrong answers, and what does gating retrieval on them buy? "
    "All policy numbers are measured on the held-out <b>test split</b> (even question ids); thresholds and "
    "fusion weights are fitted on the <b>dev split</b> or out-of-fold.",
    eyebrow="UG-RAV · benchmark",
)

POLICY_COLORS = {
    "never_retrieve": C.SLATE, "always_retrieve": C.TEAL, "sc_abstention": "#A855F7", "flare_style": C.ROSE,
    "se_gated_no_verification": C.SAFFRON, "ug_rav": C.INDIGO, "ug_rav_without_uncertainty_shift": "#818CF8",
    "ug_rav_without_evidence_signals": "#93C5FD", "oracle": "#111827",
}
GATE_COLORS = {"fused_closed_book_risk": C.INDIGO, "semantic_entropy": C.SAFFRON, "flare_min_token_prob": C.ROSE,
               "random": C.SLATE}
TIER_ORDER = ["high", "mid", "low"]


def what_this_shows(text: str) -> None:
    st.caption(f"**What this shows —** {text}")


def highlight_best(df: pd.DataFrame, rules: dict[str, bool], skip: pd.Series, formats: dict):
    """Styler that highlights the best value of each column in ``rules`` (True = higher is better)."""

    def style(col: pd.Series) -> list[str]:
        vals = col[~skip].dropna()
        if vals.empty:
            return [""] * len(col)
        best = vals.max() if rules[col.name] else vals.min()
        return ["background-color:#DCFCE7;color:#14532D;font-weight:700"
                if (not s and pd.notna(v) and abs(v - best) < 1e-9) else "" for v, s in zip(col, skip)]

    return df.style.apply(style, subset=list(rules), axis=0).format(formats, na_rep="—")


results = C.ugrav_results()
if not results:
    C.missing_results_note("halludetect bench-ugrav", "UG-RAV benchmark results")
    C.footer()
    st.stop()

keys = list(results)
has_real = any(k.startswith("real:") for k in keys)
sel = st.segmented_control("Experiment", keys, format_func=lambda k: results[k][0], default=keys[0], required=True,
                           key="ur_run")
label, R = results[sel]
is_sim = sel.startswith("sim:")
if not has_real:
    st.caption("Real-model results are not available yet — run `python scripts/run_real_experiment.py` "
               "(Qwen2.5-1.5B-Instruct on PopQA with Wikipedia retrieval and DeBERTa-MNLI) and reload.")

if is_sim:
    C.callout(
        f"<b>Simulation</b> · {R.get('n_records', '—')} records = {R.get('n_questions', '—')} questions × "
        f"{max(1, (R.get('n_records') or 0) // max(1, R.get('n_questions') or 1))} simulator seeds · "
        f"<code>{C.esc(R.get('label', ''))}</code>. The simulator reproduces documented behaviours (long-tail "
        "knowledge, confident misconceptions, confabulation, knowledge conflicts); these numbers demonstrate the "
        "mechanisms, not the accuracy of a particular model.", tone="indigo")
else:
    C.callout(
        f"<b>Real model</b> · {R.get('n_records', '—')} questions · <code>{C.esc(R.get('label', ''))}</code>. "
        "Produced by <code>scripts/run_real_experiment.py</code>; small-sample CPU run, so confidence intervals "
        "are wide.", tone="indigo")

pol = R.get("policies_test", {})
ug = pol.get("ug_rav", {})
cards = [
    ("Closed-book accuracy", C.pct(R.get("closed_book_accuracy")), "greedy answer, no retrieval"),
    ("RAG accuracy", C.pct(R.get("rag_accuracy")), "greedy answer with top-3 passages"),
    ("UG-RAV selective acc.", C.pct(ug.get("selective_accuracy")), f"at {C.pct(ug.get('coverage'))} coverage"),
    ("UG-RAV hallucination rate", C.pct(ug.get("hallucination_rate")),
     f"never {C.pct(pol.get('never_retrieve', {}).get('hallucination_rate'))} · "
     f"always {C.pct(pol.get('always_retrieve', {}).get('hallucination_rate'))}"),
    ("UG-RAV retrieval rate", C.pct(ug.get("retrieval_rate")), "vanilla RAG retrieves 100%"),
]
if R.get("retrieval_recall_at_k") is not None:
    cards.append(("Retrieval recall@3", C.pct(R["retrieval_recall_at_k"]), "gold passage retrieved"))
C.metric_row(cards)
st.space("small")

tabs = st.tabs(["🎯 Detection", "⚖️ Policies", "🚦 Retrieval gate", "🌐 Popularity", "🧮 Fusion & confident errors"])

# ------------------------------------------------------------------ detection
with tabs[0]:
    det = R.get("detection", {})
    if not det:
        st.info("This result file has no detection table.")
    else:
        rows = []
        for name, s in det.items():
            ci = s.get("auroc_ci") or [None, None]
            rows.append({"key": name, "Signal": C.label(name, C.SIGNAL_LABELS), "AUROC": s.get("auroc"),
                         "CI low": ci[0], "CI high": ci[1], "AUPRC": s.get("auprc"), "AURAC": s.get("aurac"),
                         "Acc@80%": s.get("acc_at_80"), "Acc@90%": s.get("acc_at_90"), "ECE": s.get("ece")})
        df = pd.DataFrame(rows).sort_values("AUROC")
        colors = [C.INDIGO if k.startswith("fused") else C.TEAL if "evidence" in k else
                  C.SAFFRON if k.startswith("semantic") else "#A5B4FC" for k in df["key"]]
        lo = (df["AUROC"] - df["CI low"]).fillna(0).clip(lower=0)
        hi = (df["CI high"] - df["AUROC"]).fillna(0).clip(lower=0)
        fig = go.Figure(go.Bar(
            x=df["AUROC"], y=df["Signal"], orientation="h", marker_color=colors,
            error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color="#334155", thickness=1.3),
            hovertemplate="%{y}<br>AUROC %{x:.3f}<extra></extra>",
        ))
        fig.add_scatter(x=[1.075] * len(df), y=df["Signal"], mode="text", text=[f"{v:.3f}" for v in df["AUROC"]],
                        textposition="middle left", textfont=dict(color=C.INDIGO_DARK, size=12), hoverinfo="skip")
        fig.add_vline(x=0.5, line_dash="dot", line_color=C.SLATE, annotation_text="chance",
                      annotation_position="top")
        lows = pd.to_numeric(df["CI low"], errors="coerce").dropna()
        xmin = min(0.4, float(lows.min() if len(lows) else df["AUROC"].min()) - 0.03)
        fig.update_xaxes(range=[xmin, 1.08], title="AUROC for detecting a wrong closed-book answer (95% bootstrap CI)",
                         gridcolor=C.GRID, dtick=0.1)
        C.show_fig(C.style_fig(fig, height=90 + 34 * len(df), legend=False), key="ur_auroc")
        best = df.iloc[-1]
        se_row = det.get("semantic_entropy", {})
        le_row = det.get("lexical_entropy", {})
        what_this_shows(
            f"how well each signal ranks wrong answers above right ones (0.5 = chance). Best: **{best['Signal']}** "
            f"(AUROC {best['AUROC']:.3f}). Semantic entropy ({C.fmt(se_row.get('auroc'))}) versus lexical entropy "
            f"({C.fmt(le_row.get('auroc'))}) isolates the benefit of clustering by meaning; signals that look at "
            "retrieved evidence (teal, indigo) can also catch *confident* errors that uncertainty alone misses.")
        with st.expander("Full detection table"):
            show = df.drop(columns=["key"]).sort_values("AUROC", ascending=False)
            st.dataframe(show, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in show.columns if c != "Signal"})
            st.caption("AUPRC: precision–recall area for the *wrong* class. AURAC: accuracy averaged over all "
                       "rejection rates. Acc@80%: accuracy when the riskiest 20% are rejected. ECE: calibration "
                       "error of the probabilistic scores (lower is better).")

# ------------------------------------------------------------------ policies
with tabs[1]:
    if not pol:
        st.info("This result file has no policy comparison.")
    else:
        prow = []
        for name, s in pol.items():
            prow.append({"Policy": C.label(name, C.POLICY_LABELS), "Coverage": s.get("coverage"),
                         "Accuracy (all)": s.get("accuracy"), "Selective accuracy": s.get("selective_accuracy"),
                         "Hallucination rate": s.get("hallucination_rate"), "Retrieval rate": s.get("retrieval_rate"),
                         "AURAC": s.get("aurac"), "Utility": s.get("utility")})
        pdf = pd.DataFrame(prow)
        skip = pd.Series([n == "oracle" for n in pol], index=pdf.index)
        rules = {"Coverage": True, "Accuracy (all)": True, "Selective accuracy": True, "Hallucination rate": False,
                 "Retrieval rate": False, "AURAC": True, "Utility": True}
        fmts = {c: "{:.1%}" for c in ["Coverage", "Accuracy (all)", "Selective accuracy", "Hallucination rate",
                                      "Retrieval rate"]}
        fmts.update({"AURAC": "{:.3f}", "Utility": "{:.3f}"})
        st.dataframe(highlight_best(pdf, rules, skip, fmts), hide_index=True, width="stretch")
        what_this_shows(
            "each policy's decisions on the test split. Green marks the best value per column (the oracle is "
            "excluded — it is an upper bound). *Hallucination rate* counts wrong answers over all questions; "
            "*utility* = (correct − wrong) / n rewards abstaining instead of being wrong.")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            names = list(pol)
            fig = go.Figure()
            fig.add_bar(x=[C.label(n, C.POLICY_LABELS) for n in names],
                        y=[pol[n].get("hallucination_rate") for n in names], name="Hallucination rate",
                        marker_color=C.RED)
            fig.add_bar(x=[C.label(n, C.POLICY_LABELS) for n in names],
                        y=[pol[n].get("retrieval_rate") for n in names], name="Retrieval rate",
                        marker_color=C.INDIGO)
            fig.update_layout(barmode="group")
            fig.update_yaxes(tickformat=".0%", gridcolor=C.GRID)
            fig.update_xaxes(tickangle=-35)
            C.show_fig(C.style_fig(fig, height=440, title="Hallucination rate vs retrieval rate"), key="ur_hall")
            what_this_shows("the cost–risk trade-off: vanilla RAG retrieves for every question; UG-RAV spends "
                            "retrieval only on uncertain questions and abstains when the evidence does not entail "
                            "the answer, which drives the hallucination rate down.")
        with c2:
            curves = R.get("risk_coverage_test", {})
            default = [p for p in ("never_retrieve", "always_retrieve", "flare_style", "ug_rav", "oracle")
                       if p in curves]
            chosen = st.multiselect("Policies", list(curves), default=default,
                                    format_func=lambda p: C.label(p, C.POLICY_LABELS), key="ur_rc_policies")
            fig = go.Figure()
            for p in chosen:
                cv = curves[p]
                fig.add_scatter(x=cv["coverage"], y=cv["risk"], mode="lines", name=C.label(p, C.POLICY_LABELS),
                                line=dict(color=POLICY_COLORS.get(p, C.SLATE), width=4 if p == "ug_rav" else 2,
                                          dash="dash" if p == "oracle" else "solid"))
            fig.update_xaxes(title="coverage (share of questions answered)", tickformat=".0%", gridcolor=C.GRID)
            fig.update_yaxes(title="selective risk (error rate of answered)", tickformat=".0%", gridcolor=C.GRID)
            C.show_fig(C.style_fig(fig, height=380, title="Risk–coverage curves"), key="ur_rc")
            what_this_shows("answering the lowest-risk questions first: a lower curve means the policy's risk score "
                            "orders questions better, so it can trade coverage for accuracy more efficiently.")

# ------------------------------------------------------------------ gate budget
with tabs[2]:
    budget = R.get("gate_budget_test", {})
    if not budget:
        st.info("This result file has no retrieval-budget analysis.")
    else:
        fig = go.Figure()
        for gate, pts in budget.items():
            fig.add_scatter(x=[p["retrieval_rate"] for p in pts], y=[p["accuracy"] for p in pts],
                            mode="lines+markers", name=C.label(gate, C.GATE_LABELS),
                            line=dict(color=GATE_COLORS.get(gate, C.SLATE), width=3 if gate.startswith("fused") else 2,
                                      dash="dot" if gate == "random" else "solid"),
                            hovertemplate="retrieval %{x:.0%}<br>accuracy %{y:.1%}<extra></extra>")
        fig.update_xaxes(title="retrieval budget actually used (share of questions)", tickformat=".0%",
                         gridcolor=C.GRID)
        fig.update_yaxes(title="accuracy (retrieve if the gate fires, else closed-book)", tickformat=".0%",
                         gridcolor=C.GRID)
        C.show_fig(C.style_fig(fig, height=430, title="Accuracy at matched retrieval budgets"), key="ur_budget")
        what_this_shows(
            "for each budget, the gate retrieves for the questions it scores as riskiest (threshold set on the dev "
            "split). A good gate climbs quickly: it reaches near-RAG accuracy while retrieving for far fewer "
            "questions. The dotted random gate is the baseline any useful signal must beat.")
        th = R.get("thresholds", {})
        if th.get("ug_rav"):
            t = th["ug_rav"]
            st.caption(f"Tuned UG-RAV thresholds: τ_low = {t.get('tau_low')}, τ_high = {t.get('tau_high')}, "
                       f"minimum support = {t.get('min_support')} · matched dev retrieval rate for the baselines: "
                       f"{C.pct(th.get('matched_retrieval_rate_dev'))}.")

# ------------------------------------------------------------------ popularity
with tabs[3]:
    popd = R.get("popularity", {})
    if not popd:
        st.info("This result file has no popularity analysis.")
    else:
        tiers = [t for t in TIER_ORDER if t in popd] + [t for t in popd if t not in TIER_ORDER]
        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig = go.Figure()
            fig.add_bar(x=tiers, y=[popd[t]["closed_book_accuracy"] for t in tiers], name="Closed-book",
                        marker_color=C.SLATE)
            fig.add_bar(x=tiers, y=[popd[t]["rag_accuracy"] for t in tiers], name="With retrieval",
                        marker_color=C.TEAL)
            fig.update_layout(barmode="group")
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05], gridcolor=C.GRID, title="accuracy")
            fig.update_xaxes(title="entity popularity tier")
            C.show_fig(C.style_fig(fig, height=360, title="Accuracy by popularity"), key="ur_pop_acc")
            what_this_shows("the long-tail effect: models know popular facts but fail on rare ones; retrieval "
                            "closes most of the gap for low-popularity entities.")
        with c2:
            fig = go.Figure()
            fig.add_bar(x=tiers, y=[popd[t]["retrieval_helps"] for t in tiers], name="Retrieval helps (✗ → ✓)",
                        marker_color=C.GREEN)
            fig.add_bar(x=tiers, y=[popd[t]["retrieval_hurts"] for t in tiers], name="Retrieval hurts (✓ → ✗)",
                        marker_color=C.RED)
            fig.update_layout(barmode="group")
            fig.update_yaxes(tickformat=".0%", gridcolor=C.GRID, title="share of questions")
            fig.update_xaxes(title="entity popularity tier")
            C.show_fig(C.style_fig(fig, height=360, title="When retrieval helps — and when it hurts"),
                       key="ur_pop_help")
            what_this_shows("retrieval mostly helps rare facts and can *hurt* popular ones (misleading passages, "
                            "knowledge conflicts) — the reason to retrieve selectively rather than always.")
        tdf = pd.DataFrame([{
            "Tier": t, "Questions": popd[t].get("n"), "UG-RAV retrieval rate": popd[t].get("ugrav_retrieval_rate"),
            "UG-RAV coverage": popd[t].get("ugrav_coverage"),
            "UG-RAV selective accuracy": popd[t].get("ugrav_selective_accuracy"),
            "UG-RAV hallucination rate": popd[t].get("ugrav_hallucination_rate"),
        } for t in tiers])
        st.dataframe(tdf, hide_index=True, width="stretch", column_config={
            c: st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0)
            for c in tdf.columns if c.startswith("UG-RAV")})
        what_this_shows("UG-RAV adapts to popularity on its own: it retrieves more often for rare entities, where "
                        "the closed-book model is unsure, and keeps the hallucination rate low in every tier.")

# ------------------------------------------------------------------ fusion & confident errors
with tabs[4]:
    weights = R.get("fusion_weights", {})
    if weights:
        st.markdown("##### Fusion weights")
        cols = st.columns(len(weights), gap="large")
        titles = {"closed": "Closed-book risk r̂₀", "rag": "Risk with evidence r̂"}
        for col, (name, w) in zip(cols, weights.items()):
            with col:
                items = sorted(((k, v) for k, v in w.items() if k != "bias"), key=lambda kv: abs(kv[1]))
                fig = go.Figure(go.Bar(
                    x=[v for _, v in items], y=[C.label(k, C.FEATURE_LABELS) for k, _ in items], orientation="h",
                    marker_color=[C.RED if v > 0 else C.GREEN for _, v in items],
                    text=[f"{v:+.2f}" for _, v in items], textposition="outside", cliponaxis=False,
                ))
                fig.add_vline(x=0, line_color="#9CA3AF")
                fig.update_xaxes(title="standardised logistic weight", gridcolor=C.GRID)
                C.show_fig(C.style_fig(fig, height=90 + 30 * len(items), legend=False,
                                       title=titles.get(name, name)), key=f"ur_w_{name}")
        what_this_shows("the logistic-regression weights on standardised signals. Red bars raise the predicted "
                        "risk, green bars lower it. With evidence, *support* of the evidence-grounded answer "
                        "dominates; weights of correlated signals share credit, so read them jointly.")

    ce = R.get("confident_errors", {})
    if ce:
        st.markdown("##### Confident errors")
        C.metric_row([
            ("Wrong closed-book answers", f"{ce.get('n_wrong', '—')}", "all seeds / questions"),
            ("Confident errors", f"{ce.get('n_confident_errors', '—')}", "wrong and semantic entropy ≤ median"),
            ("Share of errors that are confident", C.pct(ce.get("share_of_errors_that_are_confident")),
             "invisible to uncertainty alone"),
        ])
        within = ce.get("auroc_within_low_se", {})
        if within:
            names = list(within)
            fig = go.Figure(go.Bar(
                x=[C.label(n, C.SIGNAL_LABELS) for n in names], y=[within[n] for n in names],
                marker_color=[C.SAFFRON if n == "semantic_entropy" else C.TEAL if "evidence" in n else C.INDIGO
                              for n in names],
                text=[f"{within[n]:.3f}" for n in names], textposition="outside", cliponaxis=False,
            ))
            fig.add_hline(y=0.5, line_dash="dot", line_color=C.SLATE, annotation_text="chance")
            fig.update_yaxes(range=[0, 1.08], title="AUROC among low-entropy answers", gridcolor=C.GRID)
            C.show_fig(C.style_fig(fig, height=360, legend=False,
                                   title="Separating right from wrong among confident (low-SE) answers"),
                       key="ur_confident")
        C.callout(
            "<b>Why this matters.</b> When a model has firmly memorised a wrong answer — often a popular "
            "misconception such as <i>Sydney</i> for the capital of Australia — its samples agree, semantic entropy "
            "is low and an uncertainty-only detector is at chance. Checking the answer against retrieved evidence "
            "breaks the tie: among low-entropy answers, the evidence-aware signals still separate right from "
            "wrong. This is why UG-RAV verifies instead of trusting the gate alone.")

    states = R.get("simulator_states", {})
    if states:
        st.markdown("##### Simulator knowledge states")
        sdf = pd.DataFrame([{"State": C.SIM_STATES.get(k, (k,))[0], "Records": v.get("n"),
                             "Mean semantic entropy": v.get("mean_se"), "Closed-book accuracy": v.get("accuracy")}
                            for k, v in states.items()])
        st.dataframe(sdf, hide_index=True, width="stretch", column_config={
            "Mean semantic entropy": st.column_config.NumberColumn(format="%.2f"),
            "Closed-book accuracy": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
        })
        what_this_shows("*confidently wrong* answers have about the same semantic entropy as answers the model "
                        "truly knows, while *ambiguous* questions have high entropy although every answer is "
                        "right — the two failure modes of entropy-only detection.")

C.footer()
