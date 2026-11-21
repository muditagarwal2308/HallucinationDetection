"""Generate the report figures (PNG, 200 dpi) from the benchmark result files.

Usage: python scripts/make_figures.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INDIGO, SAFFRON, TEAL, ROSE, GREY, GREEN = "#3730a3", "#f59e0b", "#0d9488", "#e11d48", "#6b7280", "#16a34a"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11})

NICE = {
    "fused_with_evidence": "Fused (uncertainty + evidence)", "one_minus_evidence_support": "1 − evidence support",
    "fused_closed_book": "Fused (closed-book only)", "semantic_entropy": "Semantic entropy",
    "semantic_entropy_weighted": "Semantic entropy (weighted)", "predictive_entropy": "Predictive entropy",
    "length_normalised_pe": "Length-normalised PE", "lexical_entropy": "Lexical entropy",
    "one_minus_self_consistency": "1 − self-consistency", "one_minus_greedy_agreement": "1 − greedy agreement",
    "one_minus_p_true": "1 − P(True)", "min_token_surprisal": "Min-token surprisal (FLARE)",
    "never_retrieve": "Never retrieve", "always_retrieve": "Always retrieve", "sc_abstention": "SC abstention",
    "flare_style": "FLARE-style gate", "se_gated_no_verification": "SE gate, no verification (SUGAR-like)",
    "ug_rav": "UG-RAV", "ug_rav_without_uncertainty_shift": "UG-RAV − ΔSE/SE_RAG",
    "ug_rav_without_evidence_signals": "UG-RAV − evidence signals", "oracle": "Oracle",
    "citation_existence": "Citation existence only", "answer_level_nli": "Answer-level NLI (RAGAS/ALCE-style)",
    "vghd": "VGHD (proposed)", "vghd_no_premise": "VGHD − premise check", "vghd_no_quotes": "VGHD − quotation check",
    "vghd_no_speaker": "VGHD − speaker check",
}


def nice(k: str) -> str:
    return NICE.get(k, k.replace("_", " "))


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


# ----------------------------------------------------------------- UG-RAV
def ugrav_figures(res: dict, tag: str, title: str) -> None:
    det = sorted(res["detection"].items(), key=lambda kv: kv[1]["auroc"])
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(det) + 1.2))
    y = np.arange(len(det))
    au = [v["auroc"] for _, v in det]
    lo = [v["auroc"] - v["auroc_ci"][0] for _, v in det]
    hi = [v["auroc_ci"][1] - v["auroc"] for _, v in det]
    colors = [SAFFRON if k.startswith("fused") else TEAL if "evidence" in k else INDIGO for k, _ in det]
    ax.barh(y, au, xerr=[lo, hi], color=colors, alpha=0.9, capsize=3)
    ax.set_yticks(y, [nice(k) for k, _ in det])
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC for detecting a wrong closed-book answer (95% bootstrap CI)")
    ax.set_title(f"Detection signals — {title}")
    for i, (v, h) in enumerate(zip(au, hi)):
        ax.text(min(v + h + 0.008, 0.965), i, f"{v:.3f}", va="center", fontsize=8)
    save(fig, f"ugrav_{tag}_detection_auroc.png")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    styles = {"ug_rav": (SAFFRON, "-", 2.6), "always_retrieve": (TEAL, "-", 1.6), "never_retrieve": (GREY, "-", 1.6),
              "se_gated_no_verification": (INDIGO, "--", 1.4), "flare_style": (ROSE, ":", 1.6),
              "oracle": ("black", ":", 1.0)}
    for name, (c, ls, lw) in styles.items():
        cur = res["risk_coverage_test"].get(name)
        if cur:
            ax.plot(cur["coverage"], cur["risk"], color=c, ls=ls, lw=lw, label=nice(name))
    ax.set_xlabel("Coverage (fraction of questions answered)")
    ax.set_ylabel("Selective risk (error rate of answered)")
    ax.set_title(f"Risk–coverage (test split) — {title}")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(bottom=0)
    save(fig, f"ugrav_{tag}_risk_coverage.png")

    pol = res["policies_test"]
    names = [n for n in ("never_retrieve", "always_retrieve", "sc_abstention", "flare_style",
                         "se_gated_no_verification",
                         "ug_rav", "ug_rav_without_uncertainty_shift", "ug_rav_without_evidence_signals") if n in pol]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(names))
    w = 0.27
    ax.bar(x - w, [pol[n]["hallucination_rate"] for n in names], w, color=ROSE,
           label="Hallucination rate (wrong / all)")
    ax.bar(x, [pol[n]["accuracy"] for n in names], w, color=GREEN, label="Accuracy (correct / all)")
    ax.bar(x + w, [pol[n]["retrieval_rate"] for n in names], w, color=INDIGO, alpha=0.7, label="Retrieval rate")
    ax.set_xticks(x, [nice(n) for n in names], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.set_title(f"Policies on the test split — {title}")
    save(fig, f"ugrav_{tag}_policies.png")

    if res.get("gate_budget_test"):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        cols = {"fused_closed_book_risk": SAFFRON, "semantic_entropy": INDIGO, "flare_min_token_prob": ROSE,
                "random": GREY}
        labels = {"fused_closed_book_risk": "Fused closed-book risk r̂0 (UG-RAV gate)",
                  "semantic_entropy": "Semantic entropy (SUGAR-like)", "flare_min_token_prob": "Min token prob (FLARE)",
                  "random": "Random"}
        for g, rows in res["gate_budget_test"].items():
            ax.plot([r["budget"] for r in rows], [r["accuracy"] for r in rows], marker="o", color=cols.get(g, "k"),
                    label=labels.get(g, g))
        ax.set_xlabel("Retrieval budget (fraction of questions retrieved)")
        ax.set_ylabel("Accuracy (no abstention)")
        ax.set_title(f"When to retrieve — matched budgets — {title}")
        ax.legend(frameon=False, fontsize=8)
        save(fig, f"ugrav_{tag}_gate_budget.png")

    popn = res["popularity"]
    tiers = [t for t in ("low", "mid", "high") if t in popn]
    if tiers:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        x = np.arange(len(tiers))
        axes[0].bar(x - 0.2, [popn[t]["closed_book_accuracy"] for t in tiers], 0.4, color=GREY, label="Closed-book")
        axes[0].bar(x + 0.2, [popn[t]["rag_accuracy"] for t in tiers], 0.4, color=TEAL, label="With retrieval")
        axes[0].set_xticks(x, [f"{t}\n(n={popn[t]['n']})" for t in tiers])
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Accuracy by entity popularity")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].bar(x - 0.2, [popn[t]["retrieval_helps"] for t in tiers], 0.4, color=GREEN, label="Retrieval helps")
        axes[1].bar(x + 0.2, [popn[t]["retrieval_hurts"] for t in tiers], 0.4, color=ROSE, label="Retrieval hurts")
        axes[1].set_xticks(x, tiers)
        axes[1].set_title("Share of questions where retrieval helps / hurts")
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Objective O4 — {title}", fontweight="bold")
        save(fig, f"ugrav_{tag}_popularity.png")

    wts = res["fusion_weights"]["rag"]
    items = sorted(((k, v) for k, v in wts.items() if k != "bias"), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(items) + 1.0))
    ax.barh([k for k, _ in items], [v for _, v in items], color=[ROSE if v > 0 else TEAL for _, v in items])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardised logistic-regression weight (positive → higher risk)")
    ax.set_title(f"Fusion weights for the evidence-grounded answer — {title}")
    save(fig, f"ugrav_{tag}_fusion_weights.png")


def simulator_state_figure() -> None:
    path = RES / "ugrav" / "records_sim-7b.jsonl"
    if not path.exists():
        return
    states: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        states.setdefault(r["meta"]["sim_state"], []).append(r["closed"]["se"])
    order = [s for s in ("knows", "confident_wrong", "confused", "ambiguous") if s in states]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bp = ax.boxplot([states[s] for s in order], patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], [GREEN, ROSE, SAFFRON, INDIGO]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_xticks(range(1, len(order) + 1),
                  ["knows\n(correct)", "confident error\n(wrong)", "confabulation\n(mostly wrong)",
                   "ambiguous\n(correct)"][: len(order)])
    ax.set_ylabel("Semantic entropy (nats)")
    ax.set_title("Why uncertainty alone is not enough (simulation)")
    save(fig, "ugrav_sim_se_by_state.png")


# ------------------------------------------------------------------- VGHD
def vghd_figures(res: dict, tag: str, title: str) -> None:
    dets = res["detectors"]
    order = [d for d in ("citation_existence", "answer_level_nli", "vghd_no_speaker", "vghd_no_premise",
                         "vghd_no_quotes", "vghd") if d in dets]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    x = np.arange(len(order))
    w = 0.27
    ax.bar(x - w, [dets[d]["auroc"] for d in order], w, color=INDIGO, label="AUROC")
    ax.bar(x, [dets[d]["f1"] for d in order], w, color=SAFFRON, label="F1")
    ax.bar(x + w, [dets[d]["false_positive_rate"] for d in order], w, color=ROSE, label="False-positive rate")
    ax.set_xticks(x, [nice(d) for d in order], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    ax.set_title(f"Detectors on the held-out test split — {title}")
    save(fig, f"vghd_{tag}_detectors.png")

    types = ["fabricated_citation", "misattributed_citation", "misquotation", "wrong_speaker", "unsupported_claim",
             "false_premise_accepted", "out_of_scope_answered"]
    mat = np.array([[dets[d]["per_type"].get(t, {}).get("recall", np.nan) for t in types] for d in order])
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(order) + 1.6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(types)), [t.replace("_", "\n") for t in types], fontsize=8)
    ax.set_yticks(range(len(order)), [nice(d) for d in order], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, label="Recall")
    ax.set_title(f"Recall per hallucination type — {title}")
    save(fig, f"vghd_{tag}_per_type_recall.png")

    rep = res.get("repair", {})
    if rep:
        kinds = ["repaired_and_verified", "abstained", "repair_failed", "missed"]
        colors = [GREEN, INDIGO, SAFFRON, ROSE]
        ts = list(rep)
        fig, ax = plt.subplots(figsize=(8.0, 3.6))
        left = np.zeros(len(ts))
        for k, c in zip(kinds, colors):
            vals = np.array([rep[t].get(k, 0) for t in ts], float)
            ax.barh([t.replace("_", " ") for t in ts], vals, left=left, color=c, label=k.replace("_", " "))
            left += vals
        ax.set_xlabel("Hallucinated answers in the test split")
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.38))
        ax.set_title(f"What the repair-or-abstain policy did — {title}")
        save(fig, f"vghd_{tag}_repair.png")


def real_pilot_tier_figure(analysis: dict, model: str) -> None:
    """Closed-book vs RAG accuracy and retrieval quality per popularity tier (O4 on a real model)."""
    tiers = [t for t in ("high", "mid", "low") if t in analysis["by_tier"]]
    d = analysis["by_tier"]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = np.arange(len(tiers))
    w = 0.2
    series = [("closed_accuracy", "Closed-book accuracy", INDIGO), ("rag_accuracy", "RAG accuracy", TEAL),
              ("answer_in_passages", "Answer in retrieved passages", SAFFRON),
              ("gold_page_retrieved", "Gold Wikipedia page retrieved", GREY)]
    for i, (k, label, c) in enumerate(series):
        vals = [d[t][k] for t in tiers]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=label)
        ax.bar_label(bars, [f"{v:.0%}" for v in vals], fontsize=7, padding=1)
    ax.set_xticks(x, [f"{t} popularity\n(n = {d[t]['n']})" for t in tiers])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of questions")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    ax.set_title(f"Real pilot by popularity tier — {model}, PopQA")
    save(fig, "ugrav_real_tiers.png")


def vghd_nli_comparison_figure(lex: dict, deb: dict) -> None:
    """Same detectors and test split, two NLI back-ends."""
    order = ["citation_existence", "answer_level_nli", "vghd"]
    metrics = [("auroc", "AUROC"), ("f1", "F1"), ("false_positive_rate", "False-positive rate")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, (m, label) in zip(axes, metrics):
        x = np.arange(len(order))
        a = [lex["detectors"][k][m] for k in order]
        b = [deb["detectors"][k][m] for k in order]
        b1 = ax.bar(x - 0.2, a, 0.38, color=SAFFRON, label="Lexical NLI (refined on the test set)")
        b2 = ax.bar(x + 0.2, b, 0.38, color=INDIGO, label="DeBERTa-large-MNLI (off the shelf)")
        ax.bar_label(b1, [f"{v:.2f}" for v in a], fontsize=6.5, padding=1)
        ax.bar_label(b2, [f"{v:.2f}" for v in b], fontsize=6.5, padding=1)
        ax.set_xticks(x, ["Citation\nexistence", "Answer-level\nNLI", "VGHD"], fontsize=8)
        ax.set_title(label)
        ax.set_ylim(0, 1.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("VGHD with two NLI back-ends — held-out test split", fontweight="bold", fontsize=11)
    save(fig, "vghd_nli_comparison.png")


def main() -> None:
    sim = load(RES / "ugrav" / "benchmark_sim-7b.json")
    if sim:
        ugrav_figures(sim, "sim", "simulation")
    simulator_state_figure()
    for path in sorted(glob.glob(str(RES / "real" / "benchmark_popqa_*.json"))):
        real = load(Path(path))
        model = Path(path).stem.replace("benchmark_popqa_", "")
        ugrav_figures(real, "real", f"real model ({model}, PopQA)")
        analysis = load(RES / "real" / f"analysis_popqa_{model}.json")
        if analysis:
            real_pilot_tier_figure(analysis, model)
    for path, tag, title in ((RES / "gita" / "benchmark_lexical.json", "lexical", "lexical NLI"),
                             (RES / "gita" / "benchmark_microsoft_deberta_large_mnli.json", "deberta",
                              "DeBERTa-large-MNLI")):
        r = load(path)
        if r:
            vghd_figures(r, tag, title)
    lex = load(RES / "gita" / "benchmark_lexical.json")
    deb = load(RES / "gita" / "benchmark_microsoft_deberta_large_mnli.json")
    if lex and deb:
        vghd_nli_comparison_figure(lex, deb)


if __name__ == "__main__":
    main()
