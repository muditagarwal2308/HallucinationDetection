"""UG-RAV experiments: signal benchmarking (O1), fusion (O2), policies (O3) and popularity analysis (O4).

The same evaluation runs on cached :class:`QuestionRecord` files produced either
by the offline simulator (``simulate_records``) or by a real LLM
(``scripts/run_real_experiment.py``). Expensive generation happens once; all
analysis below is cheap and repeatable, exactly as planned in the dissertation's
experimental pipeline.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import metrics
from ..nli import get_nli
from ..paths import RESULTS
from . import policy as P
from .data import PassageRetriever, load_passages, load_world
from .fusion import (
    BLACKBOX_CLOSED,
    BLACKBOX_FULL,
    CLOSED_FEATURES,
    FULL_FEATURES,
    LogisticFusion,
    available,
    feature_row,
    out_of_fold,
)
from .signals import QuestionRecord, build_record
from .simulator import SimulatedQALLM


# ------------------------------------------------------------------ records
def simulate_records(seeds=(0, 1, 2, 3, 4), k: int = 10, temperature: float = 1.0, top_k: int = 3,
                     profile: str = "sim-7b", nli_name: str = "lexical") -> list[QuestionRecord]:
    world = load_world()
    retriever = PassageRetriever(load_passages())
    nli = get_nli(nli_name)
    records = []
    for seed in seeds:
        llm = SimulatedQALLM(world, seed=seed, profile=profile)
        for item in world:
            meta = {"seed": seed, "tier": item.tier, "popularity": item.popularity, "kind": item.kind,
                    "group": item.group, "gold_in_corpus": item.in_corpus,
                    "sim_state": llm.closed_belief(item)[0]}
            rec = build_record(item.id, item.question, item.aliases, llm, nli, retriever, k=k,
                               temperature=temperature, top_k=top_k, meta=meta)
            rec.meta["sim_rag_state"] = llm.rag_belief(item, rec.rag.passages)[0]
            rec.meta["gold_retrieved"] = f"p-{item.id}" in rec.rag.passage_ids
            records.append(rec)
    return records


def save_records(records: list[QuestionRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")


def load_records(path: Path) -> list[QuestionRecord]:
    return [QuestionRecord.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def split_of(qid: str) -> str:
    digits = "".join(ch for ch in qid if ch.isdigit()) or "0"
    return "dev" if int(digits) % 2 == 1 else "test"


# --------------------------------------------------------------- evaluation
def _ci(metric, *arrays):
    lo, hi = metrics.bootstrap_ci(metric, *arrays, n_boot=500)
    return [round(lo, 4), round(hi, 4)]


def detection_table(records: list[QuestionRecord], fused: dict[str, np.ndarray]) -> dict:
    """AUROC / AUPRC / AURAC / ECE of every signal for detecting wrong closed-book answers."""
    correct = np.array([r.closed_correct for r in records], float)
    wrong = 1 - correct
    c = [r.closed for r in records]
    signals = {
        "semantic_entropy": np.array([x.se for x in c]),
        "semantic_entropy_weighted": np.array([x.se_weighted if x.se_weighted is not None else np.nan for x in c]),
        "predictive_entropy": np.array([x.pe if x.pe is not None else np.nan for x in c]),
        "length_normalised_pe": np.array([x.ln_pe if x.ln_pe is not None else np.nan for x in c]),
        "lexical_entropy": np.array([x.lexical_entropy for x in c]),
        "one_minus_self_consistency": np.array([1 - x.self_consistency for x in c]),
        "one_minus_greedy_agreement": np.array([1 - x.greedy_agreement for x in c]),
        "one_minus_p_true": np.array([1 - x.p_true if x.p_true is not None else np.nan for x in c]),
        "min_token_surprisal": np.array([-np.log(max(x.min_token_prob, 1e-6)) if x.min_token_prob is not None
                                         else np.nan for x in c]),
        "one_minus_evidence_support": np.array([1 - r.rag.support_closed for r in records]),
    }
    signals.update(fused)
    out = {}
    for name, s in signals.items():
        if np.isnan(s).any():
            continue
        row = {"auroc": metrics.auroc(s, wrong), "auroc_ci": _ci(metrics.auroc, s, wrong),
               "auprc": metrics.auprc(s, wrong), "aurac": metrics.aurac(s, correct),
               "acc_at_80": metrics.accuracy_at_coverage(s, correct, 0.8),
               "acc_at_90": metrics.accuracy_at_coverage(s, correct, 0.9)}
        if name.startswith("fused") or name == "one_minus_p_true":
            row["ece"] = metrics.ece(1 - s, correct)
        out[name] = row
    return out


def confident_error_analysis(records, fused_full: np.ndarray, fused_closed: np.ndarray) -> dict:
    """How each detector ranks *confident* errors (wrong answers with low semantic entropy)."""
    se = np.array([r.closed.se for r in records])
    wrong = np.array([not r.closed_correct for r in records])
    low = se <= np.quantile(se, 0.5)
    conf_err = wrong & low
    res = {"n_wrong": int(wrong.sum()), "n_confident_errors": int(conf_err.sum()),
           "share_of_errors_that_are_confident": float(conf_err.sum() / max(wrong.sum(), 1))}
    subset = low  # among low-entropy answers, can we still separate right from wrong?
    if wrong[subset].any() and (~wrong[subset]).any():
        res["auroc_within_low_se"] = {
            "semantic_entropy": metrics.auroc(se[subset], wrong[subset]),
            "fused_closed": metrics.auroc(fused_closed[subset], wrong[subset]),
            "fused_full": metrics.auroc(fused_full[subset], wrong[subset]),
            "one_minus_evidence_support": metrics.auroc(
                1 - np.array([r.rag.support_closed for r in records])[subset], wrong[subset]),
        }
    return res


def popularity_analysis(records, decisions: P.Decisions, arrays: P.PolicyArrays) -> dict:
    """Objective O4: when does retrieval help or hurt, by popularity tier."""
    out = {}
    tiers = np.array([r.meta.get("tier") or r.meta.get("pop_tier", "all") for r in records])
    corr = decisions.correct(arrays)
    for t in [x for x in ("high", "mid", "low") if x in set(tiers)] or sorted(set(tiers)):
        m = tiers == t
        cc, rc = arrays.closed_correct[m], arrays.rag_correct[m]
        ans = decisions.source[m] >= 0
        out[t] = {
            "n": int(m.sum()), "closed_book_accuracy": float(cc.mean()), "rag_accuracy": float(rc.mean()),
            "retrieval_helps": float(((cc == 0) & (rc == 1)).mean()),
            "retrieval_hurts": float(((cc == 1) & (rc == 0)).mean()),
            "ugrav_retrieval_rate": float(decisions.retrieved[m].mean()),
            "ugrav_coverage": float(ans.mean()),
            "ugrav_selective_accuracy": float(corr[m][ans].mean()) if ans.any() else float("nan"),
            "ugrav_hallucination_rate": float(((1 - corr[m]) * ans).mean()),
        }
    return out


def evaluate(records: list[QuestionRecord], label: str = "simulation") -> dict:
    t0 = time.time()
    groups = [r.id for r in records]
    y_closed = np.array([not r.closed_correct for r in records], float)
    y_rag = np.array([not r.rag_correct for r in records], float)
    closed_feats = available(records, CLOSED_FEATURES)
    full_feats = available(records, FULL_FEATURES)
    Xc = np.array([feature_row(r, closed_feats) for r in records])
    Xf = np.array([feature_row(r, full_feats) for r in records])
    r0 = out_of_fold(Xc, y_closed, groups)
    r_full_closed = out_of_fold(Xf, y_closed, groups)
    r_rag = out_of_fold(Xf, y_rag, groups)
    no_shift = [f for f in full_feats if f not in ("delta_se", "se_rag", "one_minus_sc_rag")]
    no_evid = [f for f in full_feats if f not in ("support_rag", "contradiction_rag", "support_closed",
                                                  "contradiction_closed", "agreement")]
    r_rag_no_shift = out_of_fold(np.array([feature_row(r, no_shift) for r in records]), y_rag, groups)
    r_rag_no_evid = out_of_fold(np.array([feature_row(r, no_evid) for r in records]), y_rag, groups)

    weights = {
        "closed": LogisticFusion().fit(Xc, y_closed).weights(closed_feats),
        "rag": LogisticFusion().fit(Xf, y_rag).weights(full_feats),
    }
    det = detection_table(records, {"fused_closed_book": r0, "fused_with_evidence": r_full_closed})

    arrays = P.PolicyArrays.from_records(records, r0, r_rag)
    split = np.array([split_of(g) for g in groups])
    dev, test = np.where(split == "dev")[0], np.where(split == "test")[0]
    a_dev, a_test = arrays.subset(dev), arrays.subset(test)

    th = P.tune_ugrav(a_dev)
    th_ns = P.tune_ugrav(a_dev, r_rag=r_rag_no_shift[dev])
    th_ne = P.tune_ugrav(a_dev, r_rag=r_rag_no_evid[dev])
    sc_theta = P.tune_sc(a_dev)
    ug_dev = P.ugrav(a_dev, **th)
    rate = float(ug_dev.retrieved.mean())
    flare_theta = P.match_rate(a_dev.min_tok, rate, retrieve_if_below=True)
    se_tau = P.match_rate(a_dev.se, rate, retrieve_if_below=False)

    policies = [
        P.never_retrieve(a_test), P.always_retrieve(a_test), P.sc_abstention(a_test, sc_theta),
        P.flare_gate(a_test, flare_theta), P.se_gate(a_test, se_tau),
        P.ugrav(a_test, **th),
        P.ugrav(a_test, name="ug_rav_without_uncertainty_shift", r_rag=r_rag_no_shift[test], **th_ns),
        P.ugrav(a_test, name="ug_rav_without_evidence_signals", r_rag=r_rag_no_evid[test], **th_ne),
        P.oracle(a_test),
    ]
    pol = {d.name: d.summary(a_test) for d in policies}
    rng = np.random.default_rng(0)
    budget = P.gate_budget_curve(a_dev, a_test, {
        "fused_closed_book_risk": (a_dev.r0, a_test.r0),
        "semantic_entropy": (a_dev.se, a_test.se),
        "flare_min_token_prob": (-np.nan_to_num(a_dev.min_tok, nan=1.0), -np.nan_to_num(a_test.min_tok, nan=1.0)),
        "random": (rng.random(len(a_dev.se)), rng.random(len(a_test.se))),
    })
    curves = {}
    for d in policies:
        cov, risk = metrics.risk_coverage_curve(d.risk, d.correct(a_test))
        idx = np.linspace(0, len(cov) - 1, min(50, len(cov))).astype(int)
        curves[d.name] = {"coverage": cov[idx].round(4).tolist(), "risk": risk[idx].round(4).tolist()}
    ug_full = P.ugrav(arrays, **th)

    states = defaultdict(lambda: {"n": 0, "se": [], "correct": []})
    for r in records:
        s = r.meta.get("sim_state")
        if s:
            states[s]["n"] += 1
            states[s]["se"].append(r.closed.se)
            states[s]["correct"].append(r.closed_correct)
    state_summary = {k: {"n": v["n"], "mean_se": float(np.mean(v["se"])), "accuracy": float(np.mean(v["correct"]))}
                     for k, v in states.items()}

    res = {
        "label": label, "n_records": len(records), "n_questions": len(set(groups)),
        "closed_book_accuracy": float(1 - y_closed.mean()), "rag_accuracy": float(1 - y_rag.mean()),
        "features": {"closed": closed_feats, "full": full_feats}, "fusion_weights": weights,
        "detection": det,
        "confident_errors": confident_error_analysis(records, r_full_closed, r0),
        "thresholds": {"ug_rav": th, "ug_rav_without_uncertainty_shift": th_ns,
                       "ug_rav_without_evidence_signals": th_ne, "sc_theta": sc_theta,
                       "flare_theta": flare_theta, "se_tau": se_tau, "matched_retrieval_rate_dev": rate},
        "policies_test": pol, "risk_coverage_test": curves, "gate_budget_test": budget,
        "retrieval_cost": 0.05,
        "popularity": popularity_analysis(records, ug_full, arrays),
        "simulator_states": state_summary,
        "seconds": round(time.time() - t0, 2),
    }
    if any("gold_retrieved" in r.meta for r in records):
        res["retrieval_recall_at_k"] = float(np.mean([r.meta.get("gold_retrieved", False) for r in records]))
    return res


def run_simulation(out_dir: Path | None = None, seeds=(0, 1, 2, 3, 4), profile: str = "sim-7b",
                   nli_name: str = "lexical", verbose: bool = True) -> dict:
    t = time.time()
    records = simulate_records(seeds=seeds, profile=profile, nli_name=nli_name)
    out = out_dir or RESULTS / "ugrav"
    save_records(records, out / f"records_{profile}.jsonl")
    res = evaluate(records, label=f"simulation:{profile}:{len(seeds)} seeds:{nli_name}")
    res["generation_seconds"] = round(time.time() - t, 1)
    (out / f"benchmark_{profile}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    save_fusion_models(records, res["thresholds"]["ug_rav"], out / "fusion_model.json")
    if verbose:
        print(policy_markdown(res))
    return res


def policy_markdown(res: dict) -> str:
    rows = ["| Policy | Coverage | Accuracy (all) | Selective acc. | Hallucination rate | Retrieval rate | AURAC |",
            "|---|---|---|---|---|---|---|"]
    for name, s in res["policies_test"].items():
        rows.append(f"| {name} | {s['coverage']:.3f} | {s['accuracy']:.3f} | {s['selective_accuracy']:.3f} | "
                    f"{s['hallucination_rate']:.3f} | {s['retrieval_rate']:.3f} | {s['aurac']:.3f} |")
    return "\n".join(rows)


def detection_markdown(res: dict) -> str:
    rows = ["| Signal | AUROC (95% CI) | AUPRC | AURAC | Acc@80% | Acc@90% |", "|---|---|---|---|---|---|"]
    for name, s in sorted(res["detection"].items(), key=lambda kv: -kv[1]["auroc"]):
        ci = s["auroc_ci"]
        rows.append(f"| {name} | {s['auroc']:.3f} ({ci[0]:.3f}–{ci[1]:.3f}) | {s['auprc']:.3f} | {s['aurac']:.3f} | "
                    f"{s['acc_at_80']:.3f} | {s['acc_at_90']:.3f} |")
    return "\n".join(rows)


def save_fusion_models(records: list[QuestionRecord], thresholds: dict, path: Path) -> dict:
    """Fit the fusion models on all records and save them for the live pipeline (app / CLI)."""
    y_closed = np.array([not r.closed_correct for r in records], float)
    y_rag = np.array([not r.rag_correct for r in records], float)
    models = {}
    for key, names, y in (("closed", CLOSED_FEATURES, y_closed), ("closed_blackbox", BLACKBOX_CLOSED, y_closed),
                          ("rag", FULL_FEATURES, y_rag), ("rag_blackbox", BLACKBOX_FULL, y_rag)):
        feats = available(records, names)
        X = np.array([feature_row(r, feats) for r in records])
        models[key] = LogisticFusion().fit(X, y).to_dict(feats)
    payload = {"thresholds": thresholds, "models": models, "trained_on": len(records)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
