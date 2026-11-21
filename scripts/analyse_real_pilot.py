"""Analysis of the real-model UG-RAV pilot beyond ``evaluate()``.

It covers accuracy and retrieval quality by popularity tier, confident errors, ΔSE, case studies, and an audit of
the automatic correctness labels.

Usage::

    python scripts/analyse_real_pilot.py [--records results/real/records_popqa_qwen2.5-1.5b-instruct.jsonl]

Writes ``results/real/analysis_<tag>.json`` and prints Markdown tables used in report 03, §6.

Label audit: correctness is decided automatically by normalised alias containment (``contains_match``). A relaxed
matcher (token-F1 ≥ 0.5 against any alias, or a shared content stem) flags answers whose label might be wrong. The
flagged answers are then adjudicated by hand in ``results/real/label_audit_<tag>.json`` (``{"<id>:<closed|rag>":
true|false}``). This script reports the headline numbers both with the automatic labels (primary) and with the
audited labels (sensitivity analysis).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from halludetect.metrics import auroc, bootstrap_ci
from halludetect.text import content_tokens, max_token_f1

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results" / "real" / "records_popqa_qwen2.5-1.5b-instruct.jsonl"
TIERS = ("high", "mid", "low")


def relaxed_match(answer: str, aliases: list[str]) -> bool:
    if max_token_f1(answer, aliases) >= 0.5:
        return True
    a = set(content_tokens(answer))
    return any(a & set(content_tokens(x)) for x in aliases)


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=DEFAULT)
    args = ap.parse_args()
    recs = [json.loads(x) for x in args.records.read_text(encoding="utf-8").splitlines() if x.strip()]
    tag = args.records.stem.replace("records_", "")
    audit_path = args.records.with_name(f"label_audit_{tag}.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    audit = {k: v for k, v in audit.items() if not k.startswith("_")}

    n = len(recs)
    closed_ok = np.array([r["closed_correct"] for r in recs], dtype=bool)
    rag_ok = np.array([r["rag_correct"] for r in recs], dtype=bool)
    tiers = np.array([r["meta"]["tier"] for r in recs])
    se = np.array([r["closed"]["se"] for r in recs])
    sc = np.array([r["closed"]["self_consistency"] for r in recs])
    support = np.array([r["rag"]["support_closed"] for r in recs])
    dse = np.array([r["rag"]["delta_se"] for r in recs])
    se_rag = np.array([r["rag"]["se_rag"] for r in recs])
    in_pass = np.array([r["meta"].get("answer_in_passages", False) for r in recs], dtype=bool)
    gold_ret = np.array([r["meta"].get("gold_retrieved", False) for r in recs], dtype=bool)

    out: dict = {"n": n, "tag": tag, "k": len(recs[0]["closed"]["samples"]),
                 "mean_seconds_per_question": float(np.mean([r["seconds"] for r in recs])),
                 "relations": dict(Counter(r["meta"]["relation"] for r in recs).most_common())}

    # ---- accuracy and retrieval by tier (O4)
    by_tier = {}
    for t in [x for x in TIERS if x in set(tiers)]:
        m = tiers == t
        by_tier[t] = {"n": int(m.sum()), "closed_accuracy": float(closed_ok[m].mean()),
                      "rag_accuracy": float(rag_ok[m].mean()),
                      "retrieval_helps": float((~closed_ok[m] & rag_ok[m]).mean()),
                      "retrieval_hurts": float((closed_ok[m] & ~rag_ok[m]).mean()),
                      "answer_in_passages": float(in_pass[m].mean()), "gold_page_retrieved": float(gold_ret[m].mean()),
                      "mean_se": float(se[m].mean())}
    out["by_tier"] = by_tier
    out["overall"] = {"closed_accuracy": float(closed_ok.mean()), "rag_accuracy": float(rag_ok.mean()),
                      "retrieval_helps": float((~closed_ok & rag_ok).mean()),
                      "retrieval_hurts": float((closed_ok & ~rag_ok).mean()),
                      "answer_in_passages": float(in_pass.mean()), "gold_page_retrieved": float(gold_ret.mean()),
                      "rag_accuracy_when_answer_in_passages": float(rag_ok[in_pass].mean()) if in_pass.any() else None,
                      "rag_accuracy_when_answer_not_in_passages":
                          float(rag_ok[~in_pass].mean()) if (~in_pass).any() else None}

    # ---- uncertainty: SE of correct vs wrong, confident errors
    wrong = ~closed_ok
    low_se = se <= 0.5
    out["uncertainty"] = {
        "mean_se_correct": float(se[closed_ok].mean()) if closed_ok.any() else None,
        "mean_se_wrong": float(se[wrong].mean()) if wrong.any() else None,
        "share_wrong_with_low_se": float((wrong & low_se).sum() / max(1, wrong.sum())),
        "n_wrong": int(wrong.sum()), "n_confident_errors": int((wrong & low_se).sum()),
        "auroc_se_closed": auroc(se, wrong.astype(int)),
        "auroc_se_closed_ci": bootstrap_ci(auroc, se, wrong.astype(int), n_boot=500),
        "auroc_one_minus_sc_closed": auroc(1 - sc, wrong.astype(int)),
        "auroc_one_minus_support_closed": auroc(1 - support, wrong.astype(int)),
        "auroc_one_minus_support_closed_ci": bootstrap_ci(auroc, 1 - support, wrong.astype(int), n_boot=500),
    }
    # ΔSE: does the drop in uncertainty after retrieval indicate a correct RAG answer?
    rag_wrong = ~rag_ok
    out["delta_se"] = {
        "mean_delta_se_rag_correct": float(dse[rag_ok].mean()) if rag_ok.any() else None,
        "mean_delta_se_rag_wrong": float(dse[rag_wrong].mean()) if rag_wrong.any() else None,
        "auroc_neg_delta_se_for_rag_error": auroc(-dse, rag_wrong.astype(int)),
        "auroc_se_rag_for_rag_error": auroc(se_rag, rag_wrong.astype(int)),
        "auroc_one_minus_support_rag_for_rag_error": auroc(
            1 - np.array([r["rag"]["support_rag"] for r in recs]), rag_wrong.astype(int)),
    }

    # ---- case studies
    def case(r: dict) -> dict:
        return {"id": r["id"], "question": r["question"], "tier": r["meta"]["tier"], "gold": r["aliases"][:3],
                "closed": r["closed"]["greedy"], "closed_correct": r["closed_correct"],
                "se": round(r["closed"]["se"], 2), "rag": r["rag"]["greedy"], "rag_correct": r["rag_correct"],
                "support": round(r["rag"]["support_rag"], 2), "delta_se": round(r["rag"]["delta_se"], 2)}

    out["cases"] = {
        "retrieval_fixed": [case(r) for r in recs if not r["closed_correct"] and r["rag_correct"]][:6],
        "retrieval_hurt": [case(r) for r in recs if r["closed_correct"] and not r["rag_correct"]][:6],
        "confident_errors": [case(r) for r in recs if not r["closed_correct"] and r["closed"]["se"] <= 0.5][:6],
        "both_wrong": [case(r) for r in recs if not r["closed_correct"] and not r["rag_correct"]][:6],
    }

    # ---- label audit
    flagged = []
    for r in recs:
        for kind in ("closed", "rag"):
            ans, auto = r[kind]["greedy"], r[f"{kind}_correct"]
            if not auto and relaxed_match(ans, r["aliases"]):
                flagged.append({"key": f"{r['id']}:{kind}", "question": r["question"], "answer": ans,
                                "aliases": r["aliases"][:6], "auto": auto, "audited": audit.get(f"{r['id']}:{kind}")})
    out["label_audit"] = {"n_flagged": len(flagged), "n_adjudicated": sum(f["audited"] is not None for f in flagged),
                          "flagged": flagged}
    if audit:
        c2 = np.array([audit.get(f"{r['id']}:closed", r["closed_correct"]) for r in recs], dtype=bool)
        r2 = np.array([audit.get(f"{r['id']}:rag", r["rag_correct"]) for r in recs], dtype=bool)
        out["label_audit"]["audited"] = {
            "closed_accuracy": float(c2.mean()), "rag_accuracy": float(r2.mean()),
            "n_closed_labels_changed": int((c2 != closed_ok).sum()), "n_rag_labels_changed": int((r2 != rag_ok).sum()),
            "auroc_se_closed": auroc(se, (~c2).astype(int)),
            "auroc_one_minus_support_closed": auroc(1 - support, (~c2).astype(int))}

    path = args.records.with_name(f"analysis_{tag}.json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- markdown
    print(f"## Real pilot analysis — {tag} (N = {n}, K = {out['k']})\n")
    print("| Tier | N | Closed-book acc. | RAG acc. | Retrieval helps | Retrieval hurts | Answer in passages "
          "| Gold page retrieved | Mean SE |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t, d in [*by_tier.items(), ("all", {**out["overall"], "n": n, "mean_se": float(se.mean()),
                                            "closed_accuracy": out["overall"]["closed_accuracy"]})]:
        print(f"| {t} | {d['n']} | {pct(d['closed_accuracy'])} | {pct(d['rag_accuracy'])} | "
              f"{pct(d['retrieval_helps'])} | {pct(d['retrieval_hurts'])} | {pct(d['answer_in_passages'])} | "
              f"{pct(d['gold_page_retrieved'])} | {d['mean_se']:.2f} |")
    u = out["uncertainty"]
    print(f"\nWrong closed-book answers: {u['n_wrong']}; with SE ≤ 0.5 (confident errors): {u['n_confident_errors']} "
          f"({pct(u['share_wrong_with_low_se'])}). Mean SE correct {u['mean_se_correct']:.2f} vs wrong "
          f"{u['mean_se_wrong']:.2f}.")
    print(f"AUROC (closed-book error): SE {u['auroc_se_closed']:.3f} {u['auroc_se_closed_ci']}, "
          f"1−SC {u['auroc_one_minus_sc_closed']:.3f}, 1−evidence support {u['auroc_one_minus_support_closed']:.3f} "
          f"{u['auroc_one_minus_support_closed_ci']}")
    d = out["delta_se"]
    print(f"ΔSE: mean {d['mean_delta_se_rag_correct']:.2f} when the RAG answer is correct vs "
          f"{d['mean_delta_se_rag_wrong']:.2f} when wrong; AUROC(−ΔSE → RAG error) "
          f"{d['auroc_neg_delta_se_for_rag_error']:.3f}; SE_rag {d['auroc_se_rag_for_rag_error']:.3f}; "
          f"1−support_rag {d['auroc_one_minus_support_rag_for_rag_error']:.3f}")
    print(f"\nLabel audit: {len(flagged)} answers flagged by the relaxed matcher; "
          f"{out['label_audit']['n_adjudicated']} adjudicated.")
    for f in flagged:
        print(f"  {f['key']:<14} {f['answer'][:40]!r:44} gold={f['aliases'][:3]} audited={f['audited']}")
    if audit:
        a = out["label_audit"]["audited"]
        print(f"Audited labels: closed {pct(a['closed_accuracy'])}, RAG {pct(a['rag_accuracy'])} "
              f"({a['n_closed_labels_changed']} + {a['n_rag_labels_changed']} labels changed); AUROC SE "
              f"{a['auroc_se_closed']:.3f}, 1−support {a['auroc_one_minus_support_closed']:.3f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
