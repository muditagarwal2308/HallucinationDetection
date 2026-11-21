// Content shared by the four decks: references, diagrams and result panels.
// Every number comes from the result JSON passed in as `D` (see lib/data.js).
const K = require("../lib/kit");
const { f2, f3, pct, int, parseRealLabel, repairSummary } = require("../lib/data");

const { C, F, txt, bullets, card, iconCircle, icon, numCircle, box, arrow, line, pill, table, forest, diverging,
  pending, barChart, lineChart, legendRow, stat } = K;

// ------------------------------------------------------------------ references
const REFS = {
  huang: "L. Huang et al., “A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions,” ACM Trans. Inf. Syst., vol. 43, no. 2, 2025.",
  ji: "Z. Ji et al., “Survey of hallucination in natural language generation,” ACM Comput. Surv., vol. 55, no. 12, 2023.",
  kuhn: "L. Kuhn, Y. Gal and S. Farquhar, “Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation,” in Proc. ICLR, 2023.",
  farquhar: "S. Farquhar, J. Kossen, L. Kuhn and Y. Gal, “Detecting hallucinations in large language models using semantic entropy,” Nature, vol. 630, pp. 625–630, 2024.",
  manakul: "P. Manakul, A. Liusie and M. Gales, “SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models,” in Proc. EMNLP, 2023.",
  kadavath: "S. Kadavath et al., “Language models (mostly) know what they know,” arXiv:2207.05221, 2022.",
  lewis: "P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Proc. NeurIPS, 2020.",
  mallen: "A. Mallen et al., “When not to trust language models: Investigating effectiveness of parametric and non-parametric memories,” in Proc. ACL, 2023.",
  jiang: "Z. Jiang et al., “Active retrieval augmented generation,” in Proc. EMNLP, 2023.",
  su: "W. Su et al., “DRAGIN: Dynamic retrieval augmented generation based on the real-time information needs of large language models,” in Proc. ACL, 2024.",
  ding: "H. Ding et al., “Retrieve only when it needs: Adaptive retrieval augmentation for hallucination mitigation in large language models,” arXiv:2402.10612, 2024.",
  zubkova: "H. Zubkova, J.-H. Park and S.-W. Lee, “SUGAR: Leveraging contextual confidence for smarter retrieval,” in Proc. ICASSP, 2025.",
  tang: "L. Tang, P. Laban and G. Durrett, “MiniCheck: Efficient fact-checking of LLMs on grounding documents,” in Proc. EMNLP, 2024.",
  asai: "A. Asai, Z. Wu, Y. Wang, A. Sil and H. Hajishirzi, “Self-RAG: Learning to retrieve, generate, and critique through self-reflection,” in Proc. ICLR, 2024.",
  zha: "Y. Zha, Y. Yang, R. Li and Z. Hu, “AlignScore: Evaluating factual consistency with a unified alignment function,” in Proc. ACL, 2023.",
  niu: "C. Niu et al., “RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models,” in Proc. ACL, 2024.",
  es: "S. Es, J. James, L. Espinosa-Anke and S. Schockaert, “RAGAS: Automated evaluation of retrieval augmented generation,” in Proc. EACL (Demos), 2024.",
  gao: "T. Gao, H. Yen, J. Yu and D. Chen, “Enabling large language models to generate text with citations,” in Proc. EMNLP, 2023.",
  chandra: "R. Chandra and V. Kulkarni, “Semantic and sentiment analysis of selected Bhagavad Gita translations using BERT-based language framework,” IEEE Access, vol. 10, pp. 21291–21315, 2022.",
  malhas: "R. Malhas, W. Mansour and T. Elsayed, “Qur’an QA 2022: Overview of the first shared task on question answering over the Holy Qur’an,” in Proc. OSACT, pp. 79–87, 2022.",
  besant: "A. Besant, The Bhagavad-Gita, or The Lord’s Song, 4th ed. Madras: G. A. Natesan, 1922 (public domain; English Wikisource transcription).",
  telang: "K. T. Telang, The Bhagavadgîtâ, Sacred Books of the East, vol. 8. Oxford: Clarendon Press, 1882.",
  arnold: "E. Arnold, The Song Celestial, or Bhagavad-Gîtâ. London: Trübner & Co., 1885.",
  gretil: "GRETIL — Göttingen Register of Electronic Texts in Indian Languages, SUB Göttingen.",
  cormack: "G. V. Cormack, C. L. A. Clarke and S. Büttcher, “Reciprocal rank fusion outperforms Condorcet and individual rank learning methods,” in Proc. SIGIR, 2009.",
  he: "P. He, X. Liu, J. Gao and W. Chen, “DeBERTa: Decoding-enhanced BERT with disentangled attention,” in Proc. ICLR, 2021.",
  halueval: "J. Li et al., “HaluEval: A large-scale hallucination evaluation benchmark for large language models,” in Proc. EMNLP, 2023.",
  joshi: "M. Joshi, E. Choi, D. Weld and L. Zettlemoyer, “TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension,” in Proc. ACL, 2017.",
  kwiat: "T. Kwiatkowski et al., “Natural Questions: A benchmark for question answering research,” TACL, vol. 7, pp. 453–466, 2019.",
  karpukhin: "V. Karpukhin et al., “Dense passage retrieval for open-domain question answering,” in Proc. EMNLP, 2020.",
  robertson: "S. Robertson and H. Zaragoza, “The probabilistic relevance framework: BM25 and beyond,” Found. Trends Inf. Retr., vol. 3, no. 4, 2009.",
  dubey: "A. Dubey et al., “The Llama 3 herd of models,” arXiv:2407.21783, 2024.",
  qwen: "Qwen Team, “Qwen2.5 technical report,” arXiv:2412.15115, 2024.",
  xie: "J. Xie et al., “Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts,” in Proc. ICLR, 2024.",
  angelopoulos: "A. N. Angelopoulos and S. Bates, “A gentle introduction to conformal prediction and distribution-free uncertainty quantification,” arXiv:2107.07511, 2021.",
  min: "S. Min et al., “FActScore: Fine-grained atomic evaluation of factual precision in long form text generation,” in Proc. EMNLP, 2023.",
  kossen: "J. Kossen et al., “Semantic entropy probes: Robust and cheap hallucination detection in LLMs,” arXiv:2406.15927, 2024.",
  wen: "B. Wen et al., “Know your limits: A survey of abstention in large language models,” TACL, vol. 13, 2025.",
};

function makeRefs(keys) {
  keys.forEach((k) => { if (!REFS[k]) throw new Error(`Unknown reference ${k}`); });
  const idx = Object.fromEntries(keys.map((k, i) => [k, i + 1]));
  const c = (...ks) => `[${ks.map((k) => { if (!idx[k]) throw new Error(`Reference ${k} not in this deck`); return idx[k]; }).join(", ")}]`;
  return { list: keys.map((k) => REFS[k]), c };
}

// ------------------------------------------------------------------ labels
const SIGNAL = {
  fused_with_evidence: "Fused: uncertainty + evidence",
  one_minus_evidence_support: "1 − evidence support of a_{0}",
  min_token_surprisal: "Min-token surprisal (FLARE)",
  fused_closed_book: "Fused closed-book risk r̂_{0}",
  semantic_entropy: "Semantic entropy",
  predictive_entropy: "Predictive entropy",
  one_minus_greedy_agreement: "1 − greedy agreement",
  one_minus_self_consistency: "1 − self-consistency",
  semantic_entropy_weighted: "Semantic entropy (weighted)",
  one_minus_p_true: "1 − P(True)",
  lexical_entropy: "Lexical entropy",
  length_normalised_pe: "Length-normalised PE",
};
const signalColor = (k) =>
  k.startsWith("fused") ? C.saffron : k === "one_minus_evidence_support" ? C.teal
    : k === "lexical_entropy" || k === "length_normalised_pe" ? C.slate : C.indigo;

const POLICY = {
  never_retrieve: "Never retrieve",
  always_retrieve: "Always retrieve",
  sc_abstention: "SC abstention",
  flare_style: "FLARE-style gate",
  se_gated_no_verification: "SE gate, no verification",
  ug_rav: "UG-RAV",
  ug_rav_without_uncertainty_shift: "UG-RAV − ΔSE / SE_{RAG}",
  ug_rav_without_evidence_signals: "UG-RAV − evidence signals",
  oracle: "Oracle (upper bound)",
};
const POLICY_SHORT = {
  never_retrieve: "Never retrieve", always_retrieve: "Always retrieve", sc_abstention: "SC abstention",
  flare_style: "FLARE-style", se_gated_no_verification: "SE gate (no verif.)", ug_rav: "UG-RAV", oracle: "Oracle",
};

const FEATURE = {
  se: "Semantic entropy", one_minus_sc: "1 − self-consistency", one_minus_agreement: "1 − greedy agreement",
  one_minus_p_true: "1 − P(True)", neg_log_min_tok: "Min-token surprisal", ln_pe: "Length-normalised PE",
  se_rag: "SE with evidence", delta_se: "ΔSE (uncertainty shift)", support_rag: "Support of a_{D}",
  contradiction_rag: "Contradiction of a_{D}", support_closed: "Support of a_{0}", contradiction_closed: "Contradiction of a_{0}",
  agreement: "Closed-book / RAG agreement", one_minus_sc_rag: "1 − SC with evidence",
};

const HTYPES = ["fabricated_citation", "misattributed_citation", "misquotation", "wrong_speaker", "unsupported_claim",
  "false_premise_accepted", "out_of_scope_answered"];
const HTYPE = {
  fabricated_citation: "Fabricated citation", misattributed_citation: "Misattributed citation", misquotation: "Misquotation",
  wrong_speaker: "Wrong speaker", unsupported_claim: "Unsupported claim", false_premise_accepted: "False premise accepted",
  out_of_scope_answered: "Out-of-scope answered", correct: "Correct answer", correct_refusal: "Correct refusal",
};
const HTYPE_SHORT = {
  fabricated_citation: "Fabricated", misattributed_citation: "Misattributed", misquotation: "Misquotation",
  wrong_speaker: "Wrong speaker", unsupported_claim: "Unsupported", false_premise_accepted: "False premise",
  out_of_scope_answered: "Out of scope",
};
const DETECTOR = {
  citation_existence: "Citation existence only", answer_level_nli: "Answer-level NLI (RAGAS/ALCE-style)",
  vghd: "VGHD (proposed)", vghd_no_premise: "VGHD − premise check", vghd_no_quotes: "VGHD − quotation check",
  vghd_no_speaker: "VGHD − speaker check",
};

// ------------------------------------------------------------------ diagrams
// Factuality / faithfulness taxonomy (Huang et al.) with highlighted leaves.
function taxonomy(slide, pres, { x, y, w, hl = [], hlParent, refTag = "" }) {
  const rootW = 2.5;
  box(slide, pres, `**Hallucination in LLMs** ${refTag}`, { x: x + (w - rootW) / 2, y, w: rootW, h: 0.4, fill: C.ink, color: C.white, border: C.ink, fontSize: 11 });
  const colW = (w - 0.3) / 2;
  const cols = [
    ["Factuality", "true in the world?", ["Factual inconsistency", "Factual fabrication"]],
    ["Faithfulness", "true to the input and itself?", ["Instruction inconsistency", "Context inconsistency", "Logical inconsistency"]],
  ];
  const rcx = x + w / 2;
  line(slide, pres, rcx, y + 0.4, rcx, y + 0.6, { color: C.slate, width: 1 });
  line(slide, pres, x + colW / 2, y + 0.6, x + colW + 0.3 + colW / 2, y + 0.6, { color: C.slate, width: 1 });
  cols.forEach(([name, q, leaves], ci) => {
    const cx = x + ci * (colW + 0.3);
    line(slide, pres, cx + colW / 2, y + 0.6, cx + colW / 2, y + 0.78, { color: C.slate, width: 1 });
    const isHl = hlParent === name;
    box(slide, pres, `**${name}**\n${q}`, { x: cx, y: y + 0.78, w: colW, h: 0.5, fill: isHl ? C.indigo : C.indigoSoft, color: isHl ? C.white : C.indigo, border: C.indigo, fontSize: 9.5 });
    const ly0 = y + 1.4;
    const lh = 0.34;
    const gap = 0.09;
    const spine = cx + 0.14;
    line(slide, pres, spine, y + 1.28, spine, ly0 + (leaves.length - 1) * (lh + gap) + lh / 2, { color: C.slate, width: 1 });
    leaves.forEach((lf, li) => {
      const ly = ly0 + li * (lh + gap);
      line(slide, pres, spine, ly + lh / 2, cx + 0.3, ly + lh / 2, { color: C.slate, width: 1 });
      const on = hl.includes(lf);
      box(slide, pres, lf, { x: cx + 0.3, y: ly, w: colW - 0.3, h: lh, fill: on ? C.saffronSoft : C.white, color: on ? C.saffronDark : C.text, border: on ? C.saffron : C.slateLight, fontSize: 9.5, bold: on });
    });
  });
}

// Numbered stage flow (cards with icons, arrows in between).
function stageFlow(slide, pres, stages, { x, y, w, h, gap = 0.28, titleSize = 11.5, bodySize = 9.5 }) {
  const n = stages.length;
  const cw = (w - gap * (n - 1)) / n;
  stages.forEach((s, i) => {
    const cx = x + i * (cw + gap);
    card(slide, pres, { x: cx, y, w: cw, h, accent: s.accent || C.indigo, top: true });
    iconCircle(slide, pres, s.icon, { x: cx + 0.14, y: y + 0.2, d: 0.44, bg: s.accent || C.indigo });
    txt(slide, `${i + 1}`, { x: cx + cw - 0.45, y: y + 0.16, w: 0.32, h: 0.3, fontFace: F.head, fontSize: 16, bold: true, color: "D6D0C4", align: "right" });
    txt(slide, s.title, { x: cx + 0.14, y: y + 0.72, w: cw - 0.24, h: 0.5, fontFace: F.head, fontSize: titleSize, bold: true, color: C.ink, valign: "top" });
    bullets(slide, s.lines, { x: cx + 0.14, y: y + 1.22, w: cw - 0.22, h: h - 1.32, fontSize: bodySize, gap: 3, indent: 9 });
    if (i < n - 1) arrow(slide, pres, cx + cw + 0.03, y + 0.42, cx + cw + gap - 0.03, y + 0.42, { color: C.saffron, width: 1.75 });
  });
}

// UG-RAV architecture (Algorithm 1 as implemented), drawn with editable shapes.
function ugravArch(slide, pres, { y0 = 1.28, withEquations = true } = {}) {
  const hA = 0.62;
  const yA = y0;
  const mid = (yy, hh) => yy + hh / 2;
  const nb = (t, o) => box(slide, pres, t, { fontSize: 9, h: hA, ...o });
  nb("**Question q**", { x: 0.5, y: yA, w: 0.9, fill: C.ink, color: C.white, border: C.ink });
  nb("**Closed-book sampling**\ngreedy a_{0} + K = 10 samples", { x: 1.6, y: yA, w: 1.62, fill: C.white, border: C.indigo });
  nb("**Uncertainty signals**\nSE · SC · PE · P(True) · min-token p", { x: 3.42, y: yA, w: 1.78, fill: C.indigoSoft, border: C.indigo });
  nb("**Closed-book risk**\nr̂_{0} = σ(w_{0}ᵀx_{0} + b_{0})", { x: 5.4, y: yA, w: 1.4, fill: C.saffronSoft, border: C.saffron });
  const dA = { x: 7.0, y: mid(yA, hA) - 0.45, w: 1.42, h: 0.9 };
  box(slide, pres, "r̂_{0} < τ_{low} ?", { ...dA, shape: "diamond", fill: C.white, border: C.ink, fontSize: 9.5, bold: true });
  nb("**Accept a_{0}**\n(no retrieval)", { x: 8.62, y: yA, w: 0.88, fill: C.greenSoft, border: C.green, color: C.green });
  const ay = mid(yA, hA);
  arrow(slide, pres, 1.4, ay, 1.6, ay); arrow(slide, pres, 3.22, ay, 3.42, ay); arrow(slide, pres, 5.2, ay, 5.4, ay);
  arrow(slide, pres, 6.8, ay, 7.0, ay); arrow(slide, pres, 8.42, ay, 8.62, ay, { color: C.green });
  txt(slide, "yes", { x: 8.36, y: ay - 0.26, w: 0.3, h: 0.18, fontSize: 8, bold: true, color: C.green, align: "center" });

  const yB = y0 + 1.3;
  const cxD = dA.x + dA.w / 2;
  arrow(slide, pres, cxD, dA.y + dA.h, cxD, yB, { color: C.rose });
  txt(slide, "no", { x: cxD + 0.06, y: dA.y + dA.h + 0.02, w: 0.3, h: 0.18, fontSize: 8, bold: true, color: C.rose });
  nb("**Retrieve top-k passages**\nBM25 corpus or Wikipedia", { x: 6.92, y: yB, w: 1.58, fill: C.tealSoft, border: C.teal });
  nb("**Re-sample with evidence**\na_{D} + K samples", { x: 5.05, y: yB, w: 1.62, fill: C.white, border: C.teal });
  nb("**Evidence signals**\nSup · Con · SE_{RAG} · ΔSE · agreement", { x: 2.95, y: yB, w: 1.85, fill: C.tealSoft, border: C.teal });
  nb("**Fused risk**\nr̂ = σ(wᵀx + b)", { x: 1.25, y: yB, w: 1.45, fill: C.saffronSoft, border: C.saffron });
  const by = mid(yB, hA);
  arrow(slide, pres, 6.92, by, 6.67, by); arrow(slide, pres, 5.05, by, 4.8, by); arrow(slide, pres, 2.95, by, 2.7, by);

  const yC = yB + 0.95;
  const dB = { x: 1.08, y: yC, w: 1.8, h: 0.98 };
  arrow(slide, pres, 1.97, yB + hA, 1.97, yC);
  box(slide, pres, "r̂ < τ_{high} and\nSup(a_{D}) ≥ θ_{sup} ?", { ...dB, shape: "diamond", fill: C.white, border: C.ink, fontSize: 8.5, bold: true });
  const ans = { x: 3.35, y: yC - 0.06, w: 2.0, h: 0.46 };
  const abs = { x: 3.35, y: yC + 0.58, w: 2.0, h: 0.46 };
  box(slide, pres, "**Answer a_{D}** from verified evidence", { ...ans, fill: C.tealSoft, border: C.teal, color: C.tealDark, fontSize: 9.5 });
  box(slide, pres, "**Abstain:** “I don’t know”", { ...abs, fill: C.roseSoft, border: C.rose, color: C.rose, fontSize: 9.5 });
  const dRx = dB.x + dB.w;
  const dCy = dB.y + dB.h / 2;
  arrow(slide, pres, dRx, dCy, ans.x, ans.y + ans.h / 2, { color: C.teal });
  arrow(slide, pres, dRx, dCy, abs.x, abs.y + abs.h / 2, { color: C.rose });
  txt(slide, "yes", { x: 2.95, y: dCy - 0.34, w: 0.35, h: 0.18, fontSize: 8, bold: true, color: C.teal });
  txt(slide, "no", { x: 2.98, y: dCy + 0.16, w: 0.35, h: 0.18, fontSize: 8, bold: true, color: C.rose });

  if (withEquations) {
    const ex = 5.75;
    const ew = 3.75;
    card(slide, pres, { x: ex, y: yC - 0.06, w: ew, h: 1.1, accent: C.indigo });
    txt(slide, [
      ...K.runs("SE(q) = −Σ_{c} p(c|q) ln p(c|q),  p(c|q) ≈ |c| / K", { breakLine: false }),
    ], { x: ex + 0.18, y: yC + 0.0, w: ew - 0.25, h: 0.24, fontSize: 9.5, color: C.text });
    txt(slide, "Sup(a) = max_{i} P_{NLI}(entail | d_{i}, q ⊕ a)", { x: ex + 0.18, y: yC + 0.24, w: ew - 0.25, h: 0.24, fontSize: 9.5 });
    txt(slide, "ΔSE(q) = SE_{closed}(q) − SE_{RAG}(q)", { x: ex + 0.18, y: yC + 0.48, w: ew - 0.25, h: 0.24, fontSize: 9.5 });
    txt(slide, "Only the two logistic models are trained; no LLM weight changes.", { x: ex + 0.18, y: yC + 0.73, w: ew - 0.25, h: 0.24, fontSize: 9, italic: true, color: C.saffronDark });
  }
}

// ------------------------------------------------------------------ result panels: UG-RAV
function detectionRows(res, keys) {
  const det = res.detection || {};
  const ks = (keys || Object.keys(det)).filter((k) => det[k]);
  return ks
    .map((k) => ({ key: k, label: SIGNAL[k] || k, value: det[k].auroc, lo: det[k].auroc_ci[0], hi: det[k].auroc_ci[1], color: signalColor(k), bold: k.startsWith("fused") }))
    .sort((a, b) => b.value - a.value);
}

function forestPanel(slide, pres, res, box_, opts = {}) {
  const rows = detectionRows(res, opts.keys);
  const min = opts.min !== undefined ? opts.min : Math.min(0.5, Math.floor(Math.min(...rows.map((r) => r.lo)) * 10) / 10);
  forest(slide, pres, rows, { ...box_, min, max: 1.0, step: 0.1, labelW: opts.labelW || 2.35, valueW: opts.valueW || 1.3, fontSize: opts.fontSize || 9.5 });
  return rows;
}

function seByStateChart(slide, pres, sim, b) {
  const st = sim.simulator_states;
  const order = [["knows", "Knows"], ["confident_wrong", "Confident error"], ["confused", "Confabulation"], ["ambiguous", "Ambiguous"]].filter(([k]) => st[k]);
  barChart(slide, pres, [{ name: "Mean semantic entropy (nats)", labels: order.map((o) => o[1]), values: order.map(([k]) => st[k].mean_se) }], {
    ...b, chartColors: [C.indigo, C.rose, C.saffron, C.teal], showLegend: false, valAxisMaxVal: 1.4, valAxisLabelFormatCode: "0.0",
    showValAxisTitle: true, valAxisTitle: "Mean SE (nats)", valAxisTitleFontSize: 8.5, valAxisTitleColor: C.muted,
  });
}

function lowSeChart(slide, pres, res, b) {
  const a = (res.confident_errors || {}).auroc_within_low_se;
  if (!a) return false;
  const items = [["semantic_entropy", "Semantic entropy", C.indigo], ["fused_closed", "Closed-book r̂₀", C.indigoMid],
    ["one_minus_evidence_support", "Evidence support", C.teal], ["fused_full", "Fused + evidence", C.saffron]].filter(([k]) => a[k] !== undefined);
  barChart(slide, pres, [{ name: "AUROC within the low-SE half", labels: items.map((i) => i[1]), values: items.map(([k]) => a[k]) }], {
    ...b, chartColors: items.map((i) => i[2]), showLegend: false, dataLabelFormatCode: "0.000",
  });
  return true;
}

function weightsPanel(slide, pres, res, b, { model = "rag", top = 10 } = {}) {
  const w = res.fusion_weights[model];
  const rows = Object.entries(w).filter(([k]) => k !== "bias")
    .map(([k, v]) => ({ label: FEATURE[k] || k, value: v, key: k }))
    .sort((a, b2) => Math.abs(b2.value) - Math.abs(a.value)).slice(0, top);
  diverging(slide, pres, rows.map((r) => ({ ...r, label: r.label })), { ...b, labelW: 2.2 });
  return rows;
}

function policiesChart(slide, pres, res, b, keys = ["never_retrieve", "always_retrieve", "sc_abstention", "flare_style", "se_gated_no_verification", "ug_rav", "oracle"]) {
  const p = res.policies_test;
  const ks = keys.filter((k) => p[k]);
  const labels = ks.map((k) => POLICY_SHORT[k] || POLICY[k]);
  barChart(slide, pres, [
    { name: "Hallucination rate (wrong / all)", labels, values: ks.map((k) => p[k].hallucination_rate) },
    { name: "Retrieval rate", labels, values: ks.map((k) => p[k].retrieval_rate) },
    { name: "Coverage (answered / all)", labels, values: ks.map((k) => p[k].coverage) },
  ], {
    ...b, chartColors: [C.rose, "9AA5B8", C.indigo], dataLabelFormatCode: "0%", valAxisLabelFormatCode: "0%",
    dataLabelFontSize: 7, barGapWidthPct: 45, catAxisLabelFontSize: 8.5,
  });
}

function riskCoverageChart(slide, pres, res, b) {
  const rc = res.risk_coverage_test;
  const keys = [["ug_rav", "UG-RAV", C.saffron], ["always_retrieve", "Always retrieve", C.teal], ["flare_style", "FLARE-style gate", C.rose],
    ["se_gated_no_verification", "SE gate, no verification", C.indigo], ["never_retrieve", "Never retrieve", "9AA5B8"]].filter(([k]) => rc[k]);
  const labels = rc[keys[0][0]].coverage.map((c) => c.toFixed(1));
  lineChart(slide, pres, keys.map(([k, name]) => ({ name, labels, values: rc[k].risk })), {
    ...b, chartColors: keys.map((k) => k[2]), catAxisLabelFrequency: Math.max(1, Math.round(labels.length / 5)),
    valAxisMinVal: 0, valAxisLabelFormatCode: "0.00",
    showCatAxisTitle: true, catAxisTitle: "Coverage (fraction of questions answered)", catAxisTitleFontSize: 8.5, catAxisTitleColor: C.muted,
    showValAxisTitle: true, valAxisTitle: "Selective risk", valAxisTitleFontSize: 8.5, valAxisTitleColor: C.muted, legendFontSize: 8.5,
  });
}

function gateBudgetChart(slide, pres, res, b) {
  const g = res.gate_budget_test;
  const keys = [["fused_closed_book_risk", "Fused r̂₀ gate (UG-RAV)", C.saffron], ["flare_min_token_prob", "Min-token prob (FLARE)", C.rose],
    ["semantic_entropy", "Semantic entropy (SUGAR-like)", C.indigo], ["random", "Random", "9AA5B8"]].filter(([k]) => g[k]);
  const labels = g[keys[0][0]].map((r) => `${Math.round(r.budget * 100)}%`);
  const all = keys.flatMap(([k]) => g[k].map((r) => r.accuracy));
  lineChart(slide, pres, keys.map(([k, name]) => ({ name, labels, values: g[k].map((r) => r.accuracy) })), {
    ...b, chartColors: keys.map((k) => k[2]), lineDataSymbol: "circle", lineDataSymbolSize: 6,
    valAxisMinVal: Math.floor(Math.min(...all) * 20) / 20, valAxisMaxVal: Math.ceil(Math.max(...all) * 20) / 20, valAxisLabelFormatCode: "0.00",
    showCatAxisTitle: true, catAxisTitle: "Retrieval budget (share of questions)", catAxisTitleFontSize: 8.5, catAxisTitleColor: C.muted,
    showValAxisTitle: true, valAxisTitle: "Accuracy", valAxisTitleFontSize: 8.5, valAxisTitleColor: C.muted, legendFontSize: 8.5,
  });
}

function popularityChart(slide, pres, res, b) {
  const p = res.popularity;
  const tiers = [["high", "High popularity"], ["mid", "Mid"], ["low", "Low (long tail)"]].filter(([k]) => p[k]);
  const labels = tiers.map((t) => t[1]);
  barChart(slide, pres, [
    { name: "Closed-book accuracy", labels, values: tiers.map(([k]) => p[k].closed_book_accuracy) },
    { name: "Retrieval-augmented accuracy", labels, values: tiers.map(([k]) => p[k].rag_accuracy) },
    { name: "UG-RAV retrieval rate", labels, values: tiers.map(([k]) => p[k].ugrav_retrieval_rate) },
  ], { ...b, chartColors: [C.indigo, C.teal, C.saffron], dataLabelFormatCode: "0.00", dataLabelFontSize: 8 });
}

// ------------------------------------------------------------------ result panels: real-model pilot
function realInfo(D) {
  if (!D.real) return null;
  const r = D.real;
  const L = parseRealLabel(r.label);
  return {
    ...L, n: r.n_questions, closed: r.closed_book_accuracy, rag: r.rag_accuracy,
    secs: r.mean_seconds_per_question, inPassages: r.answer_in_passages_rate, goldRetrieved: r.retrieval_recall_at_k,
    nTest: D.realRecords ? D.realRecords.test : null, tiers: D.realRecords ? D.realRecords.tiers : null,
    det: r.detection || {}, pol: r.policies_test || {}, pop: r.popularity || {},
  };
}

function pendingReal(slide, pres, D, b, what = "Real-model pilot") {
  pending(slide, pres, {
    ...b,
    title: `${what}: Qwen2.5-1.5B-Instruct on PopQA`,
    body: "60 PopQA questions stratified by popularity · live Wikipedia retrieval (top-3 lead sections) · DeBERTa-large-MNLI for clustering and evidence support · K = 10 samples · laptop CPU. The run is in progress; this slide fills in automatically when it finishes.",
    file: D.files.real, cmd: "node presentations/build_decks.js",
  });
}

function pendingDeberta(slide, pres, D, b) {
  pending(slide, pres, {
    ...b,
    title: "VGHD with DeBERTa-large-MNLI (no lexicon tuning)",
    body: "The independent check of the lexical results: the same 524-answer test set and protocol, with microsoft/deberta-large-mnli as the NLI back-end and no hand-tuned synonym lexicon. The CPU run is in progress.",
    file: D.files.gitaDeb, cmd: "node presentations/build_decks.js",
  });
}

// ------------------------------------------------------------------ result panels: VGHD
function vghdDetectorChart(slide, pres, res, b, keys = ["citation_existence", "answer_level_nli", "vghd"], colors = [C.slate, C.teal, C.saffron]) {
  const d = res.detectors;
  const ks = keys.filter((k) => d[k]);
  const labels = ["AUROC", "F1", "False-positive rate"];
  barChart(slide, pres, ks.map((k) => ({ name: DETECTOR[k], labels, values: [d[k].auroc, d[k].f1, d[k].false_positive_rate] })), {
    ...b, chartColors: colors, dataLabelFormatCode: "0.00", dataLabelFontSize: 8,
  });
}

function vghdPerTypeChart(slide, pres, series, b) {
  // series: [{ name, res, key, color }]
  const labels = HTYPES.map((t) => HTYPE_SHORT[t]);
  barChart(slide, pres, series.map((s) => ({
    name: s.name, labels, values: HTYPES.map((t) => ((s.res.detectors[s.key].per_type || {})[t] || { recall: 0 }).recall),
  })), { ...b, chartColors: series.map((s) => s.color), dataLabelFormatCode: "0.00", dataLabelFontSize: 7, catAxisLabelFontSize: 8.5, barGapWidthPct: 50 });
}

function retrievalChart(slide, pres, res, b) {
  const r = res.retrieval;
  const groups = [["all", `All (n=${r.all.n})`, C.indigo], ["en", `English (n=${r.en.n})`, "9AA5B8"], ["hi", `Hindi (n=${r.hi.n})`, C.saffron]].filter(([k]) => r[k]);
  const labels = ["Hit@1", "Recall@5", "MRR"];
  barChart(slide, pres, groups.map(([k, name]) => ({ name, labels, values: [r[k].hit_at_1, r[k].recall_at_5, r[k].mrr] })), {
    ...b, chartColors: groups.map((g) => g[2]), dataLabelFormatCode: "0.00", dataLabelFontSize: 8,
  });
}

function repairChart(slide, pres, res, b) {
  const { rows } = repairSummary(res.repair);
  const order = HTYPES.filter((t) => rows.find((r) => r.type === t));
  const get = (t, k) => rows.find((r) => r.type === t)[k];
  const labels = order.map((t) => HTYPE_SHORT[t]);
  const series = [
    ["Repaired and re-verified", "repaired", C.teal], ["Correct abstention", "correctAbstain", C.indigo],
    ["Abstained (safe)", "otherAbstain", "9AA5B8"], ["Repair failed", "failed", C.saffron], ["Missed", "missed", C.rose],
  ].filter(([, k]) => order.some((t) => get(t, k) > 0));
  slide.addChart(pres.charts.BAR, series.map(([name, k]) => ({ name, labels, values: order.map((t) => get(t, k)) })), K.chartBase({
    ...b, barDir: "bar", barGrouping: "stacked", chartColors: series.map((s) => s[2]), barGapWidthPct: 45,
    showValue: true, dataLabelFormatCode: "0;-0;;", dataLabelColor: C.white, dataLabelFontSize: 8, dataLabelFontBold: true,
    valAxisLabelFormatCode: "0", catAxisLabelFontSize: 8.5, catAxisOrientation: "maxMin", legendFontSize: 8,
  }));
}

function ablationTable(slide, pres, res, b) {
  const d = res.detectors;
  const keys = ["vghd", "vghd_no_speaker", "vghd_no_premise", "vghd_no_quotes"].filter((k) => d[k]);
  const note = {
    vghd: "all checks",
    vghd_no_speaker: `wrong-speaker recall ${f2(d.vghd.per_type.wrong_speaker.recall)} → ${f2((d.vghd_no_speaker || d.vghd).per_type.wrong_speaker.recall)}`,
    vghd_no_premise: `FPR ${f3(d.vghd.false_positive_rate)} → ${f3((d.vghd_no_premise || d.vghd).false_positive_rate)}`,
    vghd_no_quotes: `misquote type acc. ${f2(d.vghd.per_type.misquotation.type_accuracy)} → ${f2((d.vghd_no_quotes || d.vghd).per_type.misquotation.type_accuracy)}`,
  };
  const rows = [["Detector", "AUROC", "F1", "FPR", "What changes"]];
  keys.forEach((k) => rows.push([k === "vghd" ? `**${DETECTOR[k]}**` : DETECTOR[k], f3(d[k].auroc), f3(d[k].f1), f3(d[k].false_positive_rate), note[k]]));
  table(slide, pres, rows, { ...b, colW: [1.75, 0.62, 0.55, 0.55, b.w - 3.47], fontSize: 9, align: ["left", "center", "center", "center", "left"], highlightRows: [1], rowH: 0.34 });
}

// ------------------------------------------------------------------ status tables
function statusTable(slide, pres, rows, b, { fontSize = 9.5, rowH = 0.27, objCol = false } = {}) {
  const head = objCol ? ["Task", "Obj.", "Status"] : ["Task", "Status"];
  const out = [head];
  rows.forEach((r) => {
    const [task, status, obj] = r;
    const [fill, col] = K.STATUS[status] || K.STATUS.Planned;
    const st = { text: status, fill, color: col, bold: true, align: "center" };
    out.push(objCol ? [task, { text: obj || "–", align: "center" }, st] : [task, st]);
  });
  const colW = objCol ? [b.w - 1.75, 0.55, 1.2] : [b.w - 1.2, 1.2];
  table(slide, pres, out, { ...b, colW, fontSize, rowH, zebra: true });
}

module.exports = {
  REFS, makeRefs, SIGNAL, POLICY, POLICY_SHORT, FEATURE, HTYPES, HTYPE, HTYPE_SHORT, DETECTOR,
  taxonomy, stageFlow, ugravArch, detectionRows, forestPanel, seByStateChart, lowSeChart, weightsPanel, policiesChart,
  riskCoverageChart, gateBudgetChart, popularityChart, realInfo, pendingReal, pendingDeberta, vghdDetectorChart,
  vghdPerTypeChart, retrievalChart, repairChart, ablationTable, statusTable,
};
