// M1 — Dissertation Part I, Mid-Semester Evaluation (October 2026):
// Verse-Grounded Hallucination Detection for Bhagavad Gita RAG (VGHD).
const K = require("../lib/kit");
const S = require("./shared");
const { f2, f3, pct, repairSummary } = require("../lib/data");

const { C, F, txt, bullets, card, iconCircle, icon, numCircle, box, arrow, line, pill, table, stat } = K;

const SECTIONS = ["Introduction", "Background", "Methodology", "Results & Discussion", "Conclusion"];

function build(D, assets) {
  const R = S.makeRefs(["huang", "ji", "kuhn", "farquhar", "manakul", "zha", "tang", "niu", "es", "gao", "asai", "chandra",
    "malhas", "halueval", "lewis", "besant", "telang", "arnold", "gretil", "cormack", "he"]);
  const g = D.gita;
  const L = D.gitaLex;
  const det = L.detectors;
  const v = det.vghd;
  const al = det.answer_level_nli;
  const ce = det.citation_existence;
  const deb = D.gitaDeb;
  const rep = repairSummary(L.repair);
  const nTestAns = v.n;
  const nDev = g.tsSplit.dev || 0;
  const nTestSplit = g.tsSplit.test || 0;
  const pt = (d, t) => d.per_type[t];

  const deck = new K.Deck({
    title: "Hallucination Detection in Retrieval-Augmented Question Answering on the Bhagavad Gita",
    milestone: "Dissertation Part I · Mid-Semester Evaluation", sections: SECTIONS,
  });

  // 1 ─ Title
  deck.add(null, (ctx) => K.titleSlide(ctx, {
    title: "Hallucination Detection in Retrieval-Augmented Question Answering on the Bhagavad Gita",
    titleSize: 28,
    subtitle: "Verse-Grounded Hallucination Detection (VGHD): design, prototype and first results",
    milestone: "Dissertation Presentation Part – I · Mid-Semester Evaluation",
    date: "October 2026",
    notes: "Good morning. I am Manju Pal, M.Tech CSE with specialisation in Artificial Intelligence and Robotics, working under the supervision of Dr. Raju Pal. " +
      "This is Part I of my dissertation, the mid-semester evaluation. The topic is hallucination detection when a large language model answers questions about the Bhagavad Gita with retrieved verses. " +
      "Since the proposal, the design has been completed and a working prototype has been built, so in addition to the problem and literature I will show the corpus, the datasets, the verifier and the first results.",
  }));

  // 2 ─ Outline
  deck.add(null, (ctx) => K.outline(ctx, {
    sections: [
      ["Introduction", "Research area · Motivation · Problem statement", "FaLightbulb"],
      ["Background", "Literature review · Research gaps · Research objectives", "FaBookOpen"],
      ["Methodology", "Proposed approach · Corpus and datasets · Verifier and policy · Setup", "FaCogs"],
      ["Results & Discussion", "Detection · Per-type recall · Ablations · Retrieval · Repair", "FaChartBar"],
      ["Conclusion", "Conclusion and future scope · Status · Publications · Next plan", "FaFlagCheckered"],
    ],
    notes: "The talk follows the standard structure: introduction, background, methodology, results and discussion, and conclusion. " +
      "The results section is new compared with the proposal — every number on those slides is read from the benchmark files of the prototype.",
  }));

  // 3 ─ Divider
  deck.add("Introduction", (ctx) => K.divider(ctx, { num: 1, title: "Introduction", items: ["Research Area", "Motivation", "Problem Statement"],
    notes: "First, the research area, a motivating example, and the problem statement." }));

  // 4 ─ Research area
  deck.add("Introduction", (ctx) => {
    const { slide, pres } = ctx;
    K.content(ctx, {
      kicker: "Introduction · Research area", title: "Research Area: Hallucination in LLMs",
      notes: "A hallucination is output that sounds fluent and confident but is not supported by facts, by the context given to the model, or by the instruction. " +
        "People increasingly ask chat assistants about texts such as the Bhagavad Gita and rarely check the answer. " +
        "In the standard taxonomy of Huang et al., two types matter most here, shown shaded: fabrication — a verse or quotation that does not exist — and context inconsistency — the answer goes beyond the verses that were retrieved. " +
        "What makes a scripture special is that it has a fixed reference system, chapter and verse, so many claims can be checked exactly.",
    });
    bullets(slide, [
      `**Hallucination:** LLM output that is fluent and confident but not supported by facts, the given context or the instruction ${R.c("huang", "ji")}`,
      "**Growing use:** students and general readers ask LLM assistants about texts such as the Bhagavad Gita and often accept the answer without verification",
      "**Most relevant types:** fabrication (a verse or quotation that does not exist) and context inconsistency (the answer goes beyond the retrieved verses)",
      "**Why a scripture is special:** a closed corpus with a fixed reference system (BG chapter.verse), so many claims can be checked exactly",
    ], { x: 0.5, y: 1.3, w: 4.35, h: 3.7, fontSize: 12.5, gap: 9 });
    S.taxonomy(slide, pres, { x: 5.1, y: 1.3, w: 4.4, hl: ["Factual fabrication", "Context inconsistency"], refTag: R.c("huang") });
    txt(slide, "Shaded: the types most relevant to question answering on a scripture.", { x: 5.1, y: 4.62, w: 4.4, h: 0.25, fontSize: 9, italic: true, color: C.muted, align: "center" });
  });

  // 5 ─ Motivation
  deck.add("Introduction", (ctx) => {
    const { slide, pres } = ctx;
    const v20 = g.verseByRef["2.20"].english;
    const v23 = g.verseByRef["2.23"].english;
    K.content(ctx, {
      kicker: "Introduction · Motivation", title: "Motivation: A Verse-Level Hallucination",
      notes: "Here is the example that motivates the whole work. The question asks which verse says the soul cannot be cut by weapons. " +
        "The answer states the right teaching and the right chapter, but cites BG 2.20. Verse 2.20 says the soul is never born and never dies; it is verse 2.23 that says weapons do not cleave it. " +
        "If we judge the answer as a whole against the retrieved verses, it looks faithful, because the teaching is in the retrieved context. " +
        "Only when each claim is checked against the specific verse it cites does the error appear. The prototype flags this as a misattributed citation and repairs it to BG 2.23.",
    });
    card(slide, pres, { x: 0.5, y: 1.25, w: 4.1, h: 2.35, accent: C.indigo });
    txt(slide, "QUESTION", { x: 0.7, y: 1.35, w: 3.8, h: 0.2, fontSize: 8.5, bold: true, color: C.indigo, charSpacing: 2 });
    txt(slide, "Which verse says that the soul cannot be cut by weapons?", { x: 0.7, y: 1.55, w: 3.8, h: 0.45, fontFace: F.head, fontSize: 12, italic: true });
    txt(slide, "LLM ANSWER (WITH RETRIEVED VERSES)", { x: 0.7, y: 2.07, w: 3.8, h: 0.2, fontSize: 8.5, bold: true, color: C.indigo, charSpacing: 2 });
    txt(slide, [
      { text: "The Gita states that the soul cannot be cut by weapons or burnt by fire. This is described in ", options: {} },
      { text: "BG 2.20", options: { bold: true, color: C.rose, underline: { style: "sng", color: C.rose } } },
      { text: ".", options: {} },
    ], { x: 0.7, y: 2.3, w: 3.8, h: 0.9, fontSize: 12 });
    txt(slide, "Correct teaching, correct chapter, wrong verse.", { x: 0.7, y: 3.2, w: 3.8, h: 0.28, fontSize: 10, bold: true, color: C.saffronDark });

    const vc = (yy, ref, text, ok) => {
      card(slide, pres, { x: 4.85, y: yy, w: 4.65, h: 1.12, accent: ok ? C.teal : C.rose });
      pill(slide, pres, `BG ${ref}`, { x: 5.05, y: yy + 0.13, w: 0.85, h: 0.26, fill: ok ? C.tealSoft : C.roseSoft, color: ok ? C.tealDark : C.rose });
      txt(slide, ok ? "entails the claim" : "does not entail the claim", { x: 6.0, y: yy + 0.13, w: 2.4, h: 0.26, fontSize: 9.5, bold: true, color: ok ? C.tealDark : C.rose, valign: "middle" });
      icon(slide, ok ? "FaCheckCircle" : "FaTimesCircle", { x: 9.1, y: yy + 0.12, s: 0.28, color: ok ? C.teal : C.rose });
      txt(slide, `“${text}”`, { x: 5.05, y: yy + 0.45, w: 4.3, h: 0.62, fontSize: 9.5, italic: true, color: C.muted });
    };
    vc(1.25, "2.20", v20, false);
    vc(2.48, "2.23", v23, true);

    card(slide, pres, { x: 0.5, y: 3.85, w: 4.35, h: 1.15, accent: C.slate, noShadow: true, fill: "F3F1EC" });
    iconCircle(slide, pres, "FaEye", { x: 0.68, y: 4.0, d: 0.4, bg: C.slate });
    txt(slide, "**Answer-level check** — is the answer supported by the retrieved verses?", { x: 1.2, y: 3.95, w: 3.55, h: 0.45, fontSize: 10.5 });
    txt(slide, "Yes: the teaching is in the context → error missed", { x: 1.2, y: 4.45, w: 3.55, h: 0.45, fontSize: 10.5, bold: true, color: C.rose });
    card(slide, pres, { x: 5.1, y: 3.85, w: 4.4, h: 1.15, accent: C.saffron, noShadow: true, fill: "FFF7EC" });
    iconCircle(slide, pres, "FaSearch", { x: 5.28, y: 4.0, d: 0.4, bg: C.saffron });
    txt(slide, "**Verse-level check** — does each cited verse support its claim?", { x: 5.8, y: 3.95, w: 3.6, h: 0.45, fontSize: 10.5 });
    txt(slide, "No: misattributed citation → repaired to BG 2.23", { x: 5.8, y: 4.45, w: 3.6, h: 0.45, fontSize: 10.5, bold: true, color: C.tealDark });
  });

  // 6 ─ Problem statement
  deck.add("Introduction", (ctx) => {
    const { slide, pres } = ctx;
    K.content(ctx, {
      kicker: "Introduction · Problem statement", title: "Problem Statement",
      notes: "Formally: given a question on the Gita, its verses and an open LLM used without any parameter update, decide whether the retrieval-augmented answer contains a verse-level hallucination. " +
        "Four checks define it: every cited verse must exist and support its claim; every quotation must match the text; speakers must be attributed correctly; and the answer must not accept a false premise or answer something the text does not address. " +
        "Note the verse count: the common recension has 700 verses, Besant's edition numbers 701, and the checker accepts both. " +
        "The goal is not only a score: the detection is used to accept the answer, repair it, or abstain.",
    });
    txt(slide, "Given a question q on the Bhagavad Gita, its verses (700 in the common recension; 701 in Besant’s numbering) and an open LLM used **without any parameter update**, detect whether the retrieval-augmented answer contains a **verse-level hallucination**:", { x: 0.5, y: 1.22, w: 9, h: 0.62, fontSize: 12.5 });
    const checks = [
      ["FaLink", "Citations", "Does every cited verse exist, and does it support the claim attributed to it?"],
      ["FaQuoteLeft", "Quotations", "Does every quotation match the original text?"],
      ["FaUserCheck", "Speakers", "Is each statement attributed to the right speaker?"],
      ["FaQuestionCircle", "Premise and scope", "Does the answer accept a false premise, or answer what the text does not address?"],
    ];
    checks.forEach(([ic, t, d], i) => {
      const x = 0.5 + i * 2.3;
      card(slide, pres, { x, y: 1.98, w: 2.1, h: 1.45, accent: C.indigo, top: true });
      iconCircle(slide, pres, ic, { x: x + 0.15, y: 2.14, d: 0.38, bg: C.indigo });
      txt(slide, t, { x: x + 0.62, y: 2.14, w: 1.4, h: 0.38, fontFace: F.head, fontSize: 11.5, bold: true, color: C.ink, valign: "middle" });
      txt(slide, d, { x: x + 0.15, y: 2.6, w: 1.85, h: 0.78, fontSize: 10, color: C.muted });
    });
    const fy = 3.68;
    box(slide, pres, "**Question q** on the Gita", { x: 0.5, y: fy, w: 1.65, h: 0.55, fill: C.ink, color: C.white, border: C.ink, fontSize: 10 });
    box(slide, pres, "**Retrieve verses**\n+ open LLM", { x: 2.45, y: fy, w: 1.65, h: 0.55, fill: C.white, border: C.indigo, fontSize: 10 });
    box(slide, pres, "**Verse-level check:**\nhallucination?", { x: 4.4, y: fy, w: 1.75, h: 0.55, fill: C.saffronSoft, border: C.saffron, fontSize: 10 });
    arrow(slide, pres, 2.15, fy + 0.275, 2.45, fy + 0.275); arrow(slide, pres, 4.1, fy + 0.275, 4.4, fy + 0.275);
    const outs = [["Accept the answer", C.greenSoft, C.green], ["Repair the citation / quote / speaker", C.tealSoft, C.tealDark], ["Abstain and flag the claim", C.roseSoft, C.rose]];
    outs.forEach(([t, f, c], i) => {
      const oy = 3.52 + i * 0.3;
      box(slide, pres, t, { x: 6.55, y: oy, w: 2.95, h: 0.26, fill: f, border: c, color: c, fontSize: 9, bold: true });
      arrow(slide, pres, 6.15, fy + 0.275, 6.55, oy + 0.13, { color: c });
    });
    card(slide, pres, { x: 0.5, y: 4.48, w: 9, h: 0.52, accent: C.saffron, fill: "FFF7EC", noShadow: true });
    txt(slide, "**Goal:** use the detected hallucinations to repair the answer or to abstain, so that every answer stays faithful to the verses it cites.", { x: 0.7, y: 4.48, w: 8.7, h: 0.52, fontSize: 11, valign: "middle" });
  });

  // 7 ─ Divider
  deck.add("Background", (ctx) => K.divider(ctx, { num: 2, title: "Background", items: ["Literature Review", "Research Gaps", "Research Objectives"],
    notes: "Next, the background: what existing work does, where it falls short for this problem, and the objectives." }));

  // 8 ─ Literature
  deck.add("Background", (ctx) => {
    const { slide, pres } = ctx;
    K.content(ctx, {
      kicker: "Background · Literature review", title: "Literature Review: Key Works",
      notes: "The literature falls into three groups. Reference-free detectors, such as semantic entropy and SelfCheckGPT, use the model's own uncertainty or consistency; they miss errors the model makes consistently and never look at citations. " +
        "Reference-based detectors — AlignScore, MiniCheck, RAGTruth, RAGAS, ALCE and Self-RAG — compare the answer with documents, but they treat evidence as free text and do not exploit an exact reference system. " +
        "Work on scriptures, such as the BERT analysis of Gita translations and the Qur'an QA shared task, does not study LLM hallucination at all. " +
        "Compared with the proposal deck, AlignScore and MiniCheck, and RAGTruth and RAGAS, now have separate reference numbers.",
    });
    const fam = { free: C.indigoSoft, ref: C.tealSoft, scr: C.saffronSoft };
    const rows = [
      ["Work", "Approach", "Granularity", "Limitation for this work"],
      [{ text: `Semantic entropy ${R.c("kuhn", "farquhar")}`, fill: fam.free }, "Entropy over meaning clusters of sampled answers", "Answer", "Misses consistent errors; no citation check"],
      [{ text: `SelfCheckGPT ${R.c("manakul")}`, fill: fam.free }, "Consistency among sampled responses", "Sentence", "No external evidence"],
      [{ text: `AlignScore ${R.c("zha")}; MiniCheck ${R.c("tang")}`, fill: fam.ref }, "Factual consistency against a document", "Claim", "Do not use verse numbering"],
      [{ text: `RAGTruth ${R.c("niu")}`, fill: fam.ref }, "Annotated hallucinations in RAG outputs", "Span", "General domain; English only"],
      [{ text: `RAGAS ${R.c("es")}`, fill: fam.ref }, "LLM-based faithfulness and relevance scores", "Answer", "No citation-level verification"],
      [{ text: `ALCE ${R.c("gao")}`, fill: fam.ref }, "Citation recall and precision with NLI", "Sentence", "Web documents; no exact references"],
      [{ text: `Self-RAG ${R.c("asai")}`, fill: fam.ref }, "Learned retrieval and self-critique", "Segment", "Requires fine-tuning the LLM"],
      [{ text: `Chandra & Kulkarni ${R.c("chandra")}`, fill: fam.scr }, "BERT-based analysis of Gita translations", "Verse", "No question answering or hallucination"],
      [{ text: `Qur’an QA 2022 ${R.c("malhas")}`, fill: fam.scr }, "Shared task on QA over a scripture", "Passage", "Extractive QA; no LLM hallucination"],
    ];
    table(slide, pres, rows, { x: 0.5, y: 1.22, w: 9, colW: [2.35, 3.0, 1.0, 2.65], fontSize: 9.5, rowH: 0.31, zebra: false, firstColBold: true });
    K.legendRow(slide, pres, [["Reference-free detection", C.indigoSoft], ["Reference-based detection", C.tealSoft], ["NLP on scriptures", C.saffronSoft]], { x: 0.5, y: 4.48 });
    icon(slide, "FaBookOpen", { x: 9.15, y: 4.45, s: 0.3, color: C.indigo });
  });

  // 9 ─ Research gaps
  deck.add("Background", (ctx) => {
    const { slide, pres } = ctx;
    K.content(ctx, {
      kicker: "Background · Research gaps", title: "Research Gaps",
      notes: "Four gaps follow from the literature. First, no benchmark or detector checks answers on the Gita verse by verse. " +
        "Second, attribution methods such as ALCE and RAGAS check free-form documents and do not use a fixed chapter.verse numbering. " +
        "Third, retrieval for English and Hindi questions over Sanskrit verses and their translations is rarely evaluated. " +
        "Fourth, most detectors only output a score; using it to repair the answer or abstain is largely unexplored. " +
        "Each card also shows how the prototype addresses the gap; the third gap is now covered by objective O1, which the proposal did not do.",
    });
    const gaps = [
      ["No verse-level evaluation", `Existing benchmarks and detectors (HaluEval ${R.c("halueval")}, RAGTruth ${R.c("niu")}) target general-domain text; none checks answers on the Gita verse by verse`, `Labelled test set of ${g.nTest} answers, 7 hallucination types`],
      ["Structured references not exploited", `Faithfulness and attribution methods (ALCE ${R.c("gao")}, RAGAS ${R.c("es")}) check free-form documents, not a fixed chapter.verse numbering`, "Existence, claim–verse entailment, quotation and speaker checks"],
      ["Multilingual retrieval over Sanskrit", "Retrievers are rarely evaluated on English and Hindi questions over Sanskrit verses and their translations", "Hybrid English / IAST / Devanagari retrieval; Hit@1 by language"],
      ["Detection without correction", "Most detectors only report a score; their use for repairing the answer or abstaining is largely unexplored", "Repair-or-abstain policy; every repair is re-verified"],
    ];
    gaps.forEach(([t, d, p], i) => {
      const x = 0.5 + (i % 2) * 4.6;
      const y = 1.22 + Math.floor(i / 2) * 1.5;
      card(slide, pres, { x, y, w: 4.4, h: 1.35, accent: C.saffron });
      numCircle(slide, pres, i + 1, { x: x + 0.2, y: y + 0.14, d: 0.36, fontSize: 12 });
      txt(slide, t, { x: x + 0.68, y: y + 0.12, w: 3.6, h: 0.4, fontFace: F.head, fontSize: 12, bold: true, color: C.ink, valign: "middle" });
      txt(slide, d, { x: x + 0.22, y: y + 0.55, w: 4.05, h: 0.52, fontSize: 9.5, color: C.muted });
      txt(slide, [{ text: "→ Prototype: ", options: { bold: true, color: C.tealDark } }, { text: p, options: { color: C.tealDark } }], { x: x + 0.22, y: y + 1.06, w: 4.05, h: 0.24, fontSize: 9 });
    });
    card(slide, pres, { x: 0.5, y: 4.3, w: 9, h: 0.7, accent: C.indigo, fill: C.indigoPale, noShadow: true });
    txt(slide, "**Problem formulation:** verse-grounded detection of hallucinations in retrieval-augmented question answering on the Bhagavad Gita — with the detections used to repair the answer or to abstain — remains an open and challenging task.", { x: 0.72, y: 4.3, w: 8.6, h: 0.7, fontSize: 11, valign: "middle" });
  });

  // 10 ─ Objectives
  deck.add("Background", (ctx) => {
    const { slide, pres } = ctx;
    K.content(ctx, {
      kicker: "Background · Research objectives", title: "Research Objectives",
      notes: "There are two objectives. O1 is about data: a verse-aligned corpus and a verse-level QA dataset, and an analysis of the hallucinations an LLM makes with and without retrieval. " +
        "O1 now explicitly includes retrieval quality for English and Hindi questions, which closes the multilingual gap. " +
        "O2 is the method: a verse-grounded detection framework, used to repair or abstain, evaluated against baselines and against human judgement. " +
        "The ticks show what the prototype already delivers; the hourglasses show what remains — real LLM answers and the human evaluation.",
    });
    const obj = [
      ["O1", "Corpus, dataset and analysis", "To prepare a verse-aligned corpus of the Bhagavad Gita and a verse-level QA dataset (English and Hindi), and to analyse the hallucinations of an LLM with and without retrieval of the relevant verses — including retrieval quality by question language.",
        [[`${g.nVerses}-verse aligned corpus: Devanagari · IAST · English`, true], [`${g.nQa}-question QA set, incl. ${g.qaLang.hi || 0} in Hindi`, true], ["Hybrid multilingual retrieval, evaluated by language", true], ["Real LLM answers with / without retrieval", false]]],
      ["O2", "Detection, repair and evaluation", "To develop a verse-grounded hallucination detection framework for retrieval-augmented QA, to use its output to repair the answer or abstain, and to evaluate it against baselines and human judgement.",
        [["VGHD verifier: citation, support, quotation, speaker, premise", true], ["Repair-or-abstain policy with re-verification", true], ["Benchmark with 2 baselines and 3 ablations", true], ["Agreement with human annotation", false]]],
    ];
    obj.forEach(([code, t, d, items], i) => {
      const x = 0.5 + i * 4.6;
      card(slide, pres, { x, y: 1.22, w: 4.4, h: 3.78, accent: i ? C.teal : C.indigo, top: true });
      numCircle(slide, pres, code, { x: x + 0.2, y: 1.42, d: 0.55, bg: i ? C.teal : C.indigo, fontSize: 15 });
      txt(slide, t, { x: x + 0.9, y: 1.42, w: 3.35, h: 0.55, fontFace: F.head, fontSize: 14, bold: true, color: C.ink, valign: "middle" });
      txt(slide, d, { x: x + 0.22, y: 2.08, w: 4.0, h: 1.15, fontSize: 10.5, color: C.text });
      txt(slide, "STATUS IN THE PROTOTYPE", { x: x + 0.22, y: 3.25, w: 4.0, h: 0.2, fontSize: 8, bold: true, color: C.saffronDark, charSpacing: 2 });
      items.forEach(([it, done], j) => {
        const yy = 3.5 + j * 0.36;
        icon(slide, done ? "FaCheckCircle" : "FaHourglassHalf", { x: x + 0.24, y: yy + 0.04, s: 0.2, color: done ? C.green : C.saffron });
        txt(slide, it, { x: x + 0.55, y: yy, w: 3.7, h: 0.3, fontSize: 9.5, color: done ? C.text : C.muted, valign: "middle" });
      });
    });
  });

  // 11 ─ Divider
  deck.add("Methodology", (ctx) => K.divider(ctx, { num: 3, title: "Methodology", items: ["Proposed Approach", "Corpus and Datasets", "Verifier and Policy", "Experimental Setup"],
    notes: "The methodology section covers the proposed approach, the corpus and datasets that were built, the verification rules and repair policy, and the experimental setup." }));

  // 12 ─ Proposed approach
  deck.add("Methodology", (ctx) => {
    const { slide, pres } = ctx;
    K.content(ctx, {
      kicker: "Methodology · Proposed approach", title: "Proposed Approach: Verse-Grounded Verification",
      notes: "The idea is simple: because the Gita has a fixed reference system, an answer can be verified verse by verse. " +
        "Step one retrieves verses with a hybrid of three indexes — English BM25, IAST character 4-grams for romanised Sanskrit, and Devanagari 3-grams — fused by reciprocal rank; Hindi questions are expanded through a glossary. " +
        "Step two: an open LLM answers from those verses and must cite every verse as BG chapter.verse. Step three splits the answer into claims and extracts citations, quotations and speakers. " +
        "Step four runs the checks, and step five decides: accept, repair, regenerate with the verifier's feedback, or abstain. The hypothesis is that this catches errors that answer-level methods cannot see.",
    });
    card(slide, pres, { x: 0.5, y: 1.2, w: 9, h: 0.62, accent: C.saffron, fill: "FFF7EC", noShadow: true });
    txt(slide, "**Idea:** a scripture is a closed corpus with a fixed reference system (BG chapter.verse), so an answer can be verified **verse by verse**. **Hypothesis:** this catches errors that answer-level methods miss, such as a correct teaching cited to the wrong verse.", { x: 0.72, y: 1.2, w: 8.65, h: 0.62, fontSize: 10.5, valign: "middle" });
    S.stageFlow(slide, pres, [
      { icon: "FaSearch", title: "Retrieve verses", lines: ["English BM25 + IAST and Devanagari n-grams", `Reciprocal-rank fusion ${R.c("cormack")}`, "Hindi → English concept glossary"] },
      { icon: "FaFeatherAlt", title: "Generate answer", lines: ["Open LLM answers from the verses", "Must cite each verse as BG c.v", "No fine-tuning"] },
      { icon: "FaListUl", title: "Segment", lines: ["Split into claims", "Extract citations and quotations", "Keep the named speaker"] },
      { icon: "FaClipboardCheck", title: "Verify", lines: ["Verse exists?", "Cited verse entails claim (NLI)?", "Quote and speaker match?", "Premise grounded?"] },
      { icon: "FaBalanceScale", title: "Decide", accent: C.saffron, lines: ["Accept", "Repair and re-verify", "Regenerate with feedback", "Abstain with reason"] },
    ], { x: 0.5, y: 2.0, w: 9, h: 3.0, gap: 0.25, titleSize: 11.5, bodySize: 9.5 });
  });

  // 13 ─ Corpus
  deck.add("Methodology", (ctx) => {
    const { slide, pres } = ctx;
    const sp = g.speakers;
    K.content(ctx, {
      kicker: "Methodology · Data sources and verse corpus", title: "Data Sources and the Verse Corpus",
      notes: `Four public-domain sources were identified in the proposal. The corpus uses Annie Besant's 1922 fourth edition, transcribed on Wikisource, because it gives the Sanskrit text and the English translation verse by verse and is in the public domain. ` +
        "Telang's translation was considered but dropped because its prose is not verse-numbered; Arnold's is poetic and not verse-aligned. " +
        `The result is ${g.nVerses} verses in ${g.chapters} chapters, each with Devanagari, IAST and English text and a speaker. Each English paragraph is anchored to the number of the Devanagari verse before it, which fixes OCR slips in the English numbering. ` +
        "One edition detail matters for a citation checker: Besant numbers Arjuna's opening question of chapter 13 as verse 13.1, giving 701 verses; the common recension has 700. Both references are stored and a citation is accepted under either.",
    });
    const rows = [
      ["Source", "Content", "Decision"],
      [`GRETIL ${R.c("gretil")}`, "Sanskrit text of the verses", "Identified; Devanagari taken from the bilingual Besant edition"],
      [`Telang, 1882 ${R.c("telang")}`, "English prose translation", "Considered; dropped (not verse-numbered)"],
      [{ text: `**Besant, 1922, 4th ed.** ${R.c("besant")}`, fill: C.saffronSoft }, { text: "Sanskrit + English, verse by verse", fill: C.saffronSoft }, { text: "**Primary source** — public domain (Wikisource)", fill: C.saffronSoft }],
      [`Arnold, 1885 ${R.c("arnold")}`, "Poetic translation", "Not verse-aligned; not used"],
    ];
    table(slide, pres, rows, { x: 0.5, y: 1.22, w: 5.55, colW: [1.75, 1.75, 2.05], fontSize: 9.5, rowH: 0.42, zebra: false });
    card(slide, pres, { x: 0.5, y: 3.55, w: 5.55, h: 1.45, accent: C.indigo });
    bullets(slide, [
      "**Alignment:** each English paragraph is anchored to the number of the preceding Devanagari verse (॥ २३ ॥); speakers come from the Sanskrit uvāca lines; IAST is generated automatically",
      "**Edition note:** Besant numbers Arjuna’s question of ch. 13 as 13.1 → 701 verses; the common recension has 700. Both references are stored and either is accepted",
    ], { x: 0.72, y: 3.65, w: 5.2, h: 1.3, fontSize: 9.5, gap: 5 });
    stat(slide, pres, { x: 6.3, y: 1.22, w: 1.5, h: 1.05, value: String(g.nVerses), label: "Verses", sub: `${g.chapters} chapters`, valueSize: 24 });
    stat(slide, pres, { x: 8.0, y: 1.22, w: 1.5, h: 1.05, value: "3", label: "Scripts", sub: "Devanagari · IAST · English", valueSize: 24 });
    const spk = ["Krishna", "Arjuna", "Sanjaya", "Dhritarashtra"].filter((k) => sp[k]);
    txt(slide, "VERSES BY SPEAKER", { x: 6.3, y: 2.45, w: 3.2, h: 0.2, fontSize: 8.5, bold: true, color: C.indigo, charSpacing: 2 });
    slide.addChart(pres.charts.BAR, [{ name: "Verses", labels: spk, values: spk.map((k) => sp[k]) }], K.chartBase({
      x: 6.2, y: 2.62, w: 3.35, h: 2.4, barDir: "bar", chartColors: [C.indigo, C.teal, C.saffron, C.slate], showLegend: false,
      showValue: true, dataLabelFormatCode: "0", dataLabelFontSize: 9, valAxisHidden: true, valGridLine: { style: "none" },
      catAxisOrientation: "maxMin", barGapWidthPct: 40, chartArea: { fill: { color: C.bg } }, plotArea: { fill: { color: C.bg } },
    }));
  });

  // 14 ─ Datasets
  deck.add("Methodology", (ctx) => {
    const { slide, pres } = ctx;
    const catOrder = [["concept", "Concept"], ["false_premise", "False premise"], ["out_of_scope", "Out of scope"], ["speaker", "Speaker"],
      ["hindi", "Hindi"], ["quote", "Quotation"], ["nonexistent_reference", "Non-existent ref."], ["verse_lookup", "Verse lookup"]].filter(([k]) => g.qaCats[k]);
    const tOrder = ["correct", "correct_refusal", ...S.HTYPES].filter((t) => g.tsTypes[t]);
    const neg = g.tsLabels[0] || g.tsLabels["0"] || 0;
    const pos = g.tsLabels[1] || g.tsLabels["1"] || 0;
    K.content(ctx, {
      kicker: "Methodology · Datasets", title: "Datasets: Verse-Level QA and a Hallucination Test Set",
      notes: `No verse-level QA dataset existed, so two were built. The QA set has ${g.nQa} questions with gold verses and reference answers: concept, verse-lookup, quotation and speaker questions, ${g.qaLang.hi || 0} Hindi questions, and three kinds of problematic question — false premise, non-existent reference and out of scope — whose correct answer is a refusal. ` +
        `From it, a labelled test set of ${g.nTest} answers was generated by controlled perturbation: each answerable question contributes its correct reference answer and five corruptions — fabricated citation, misattributed citation, misquotation, wrong speaker and unsupported claim; problematic questions contribute a correct refusal and an answer that accepts the premise. ` +
        `That gives ${neg} negatives and ${pos} hallucinated answers. The split is by question, so all versions of one question stay in the same half: ${nDev} answers for development and ${nTestSplit} for test. These are synthetic errors — ideal for per-type measurement, but easier than errors found in the wild.`,
      badge: ["SYNTHETIC PERTURBATIONS", "synth"],
    });
    txt(slide, `QA DATASET · ${g.nQa} QUESTIONS`, { x: 0.5, y: 1.2, w: 4.3, h: 0.22, fontSize: 9, bold: true, color: C.indigo, charSpacing: 2 });
    slide.addChart(pres.charts.BAR, [{ name: "Questions", labels: catOrder.map((c) => c[1]), values: catOrder.map(([k]) => g.qaCats[k]) }], K.chartBase({
      x: 0.4, y: 1.42, w: 4.4, h: 2.95, barDir: "bar", catAxisOrientation: "maxMin", showLegend: false, showValue: true, dataLabelFormatCode: "0",
      dataLabelFontSize: 9, chartColors: catOrder.map(([k]) => ["false_premise", "out_of_scope", "nonexistent_reference"].includes(k) ? C.rose : k === "hindi" ? C.saffron : C.indigo),
      valAxisHidden: true, valGridLine: { style: "none" }, barGapWidthPct: 35, chartArea: { fill: { color: C.bg } }, plotArea: { fill: { color: C.bg } },
    }));
    txt(slide, `HALLUCINATION TEST SET · ${g.nTest} ANSWERS`, { x: 5.2, y: 1.2, w: 4.3, h: 0.22, fontSize: 9, bold: true, color: C.indigo, charSpacing: 2 });
    slide.addChart(pres.charts.BAR, [{ name: "Answers", labels: tOrder.map((t) => S.HTYPE[t]), values: tOrder.map((t) => g.tsTypes[t]) }], K.chartBase({
      x: 5.1, y: 1.42, w: 4.45, h: 2.95, barDir: "bar", catAxisOrientation: "maxMin", showLegend: false, showValue: true, dataLabelFormatCode: "0",
      dataLabelFontSize: 9, chartColors: tOrder.map((t) => (t.startsWith("correct") ? C.green : C.rose)),
      valAxisHidden: true, valGridLine: { style: "none" }, barGapWidthPct: 30, chartArea: { fill: { color: C.bg } }, plotArea: { fill: { color: C.bg } },
    }));
    K.legendRow(slide, pres, [["Answerable", C.indigo], ["Hindi", C.saffron], ["Problematic (refusal is correct)", C.rose]], { x: 0.5, y: 4.4, fontSize: 8 });
    K.legendRow(slide, pres, [[`Negatives (${neg})`, C.green], [`Hallucinated (${pos})`, C.rose]], { x: 5.2, y: 4.4, fontSize: 8 });
    txt(slide, `Built by controlled perturbation (seed 2026) · split by question: ${nDev} dev / ${nTestSplit} test answers · gold verses checked against Besant’s text`, { x: 0.5, y: 4.7, w: 9, h: 0.3, fontSize: 9.5, italic: true, color: C.muted });
  });

  // 15 ─ Checks and policy
  deck.add("Methodology", (ctx) => {
    const { slide, pres } = ctx;
    K.content(ctx, {
      kicker: "Methodology · Verifier and policy", title: "Verification Checks and the Repair-or-Abstain Policy",
      notes: "This table is the heart of the verifier. Each row is an issue type with a decidable rule and the action the policy takes. " +
        "A fabricated citation does not exist in either numbering — chapter 2 has 72 verses, so BG 2.80 is fabricated. A misattributed citation exists, but the verse does not entail the claim while another verse does; if that verse is within two of the cited one, the citation is only imprecise. " +
        "Quotations are matched in English, IAST and Devanagari with a similarity threshold of 0.92; speakers are compared with the speaker of the cited verse; false premises and out-of-scope questions are checked from the question itself. " +
        "Support is the maximum NLI entailment of the claim given each cited verse, alone and jointly. And importantly, every repair is verified again before it is returned — the policy never returns an answer it could not verify.",
    });
    const sev = (s) => ({ text: s, color: s === "Critical" ? C.rose : s === "Major" ? C.saffronDark : C.slate, bold: true, align: "center" });
    const rows = [
      ["Issue", "Rule (decidable check)", "Severity", "Policy action"],
      ["Fabricated citation", "Reference exists in neither numbering (e.g. BG 2.80 — chapter 2 has 72 verses)", sev("Critical"), "Retarget to a supporting verse, else abstain"],
      ["Misattributed citation", "Cited verse does not entail the claim; another verse does", sev("Major"), "Retarget the citation (BG 2.20 → BG 2.23)"],
      ["Imprecise citation", "As above, but the supporting verse is within ±2 of the cited one", sev("Minor"), "Retarget the citation"],
      ["Unsupported claim", "No verse entails a claim presented as the text’s teaching", sev("Major"), "Remove the claim; abstain if none remains"],
      ["Misquotation", "Similarity to the cited verse < 0.92 (English, IAST or Devanagari)", sev("Major"), "Restore the verse’s wording"],
      ["Wrong speaker", "Named speaker differs from the speaker of every cited verse", sev("Major"), "Replace the speaker"],
      ["False premise accepted", "Presupposition of the question not grounded; answer does not reject it", sev("Critical"), "Abstain, explaining why"],
      ["Out-of-scope answered", "Question outside the text; answer does not decline", sev("Critical"), "Abstain"],
    ];
    table(slide, pres, rows, { x: 0.5, y: 1.2, w: 9, colW: [1.7, 4.05, 0.85, 2.4], fontSize: 9.5, rowH: 0.345, firstColBold: true });
    iconCircle(slide, pres, "FaShieldAlt", { x: 0.5, y: 4.47, d: 0.42, bg: C.teal });
    txt(slide, "**Support** = max NLI entailment of the claim given each cited verse (alone and jointly) · **risk score** = max over claims · **every repair is re-verified** before it is returned.", { x: 1.05, y: 4.42, w: 8.45, h: 0.55, fontSize: 10, valign: "middle" });
  });

  // 16 ─ Experimental setup
  deck.add("Methodology", (ctx) => {
    const { slide, pres } = ctx;
    K.content(ctx, {
      kicker: "Methodology · Experimental setup", title: "Experimental Setup and Evaluation Protocol",
      notes: "All VGHD experiments run on the laptop CPU — the corpus is tiny, so no GPU is needed for verification. The Kaggle free GPU tier is reserved for generating real LLM answers at scale. " +
        "Two NLI back-ends are used: a deterministic lexical NLI that runs offline, and DeBERTa-large-MNLI, which serves as the independent check because it involves no hand-tuning. " +
        `The protocol: split by question into ${nDev} development and ${nTestSplit} test answers; thresholds are tuned on development by Youden's J; the baselines are citation existence only and an answer-level NLI faithfulness check in the style of RAGAS and ALCE; and three ablations each remove one check. ` +
        "Metrics include AUROC, F1, the false-positive rate on correct answers, recall per hallucination type, retrieval Hit@1, Recall@5 and MRR, and repair outcomes.",
    });
    const cols = [
      ["FaLaptopCode", "Hardware and software", C.indigo, [
        "Laptop: HP Pavilion x360, Core i5-1335U, 16 GB RAM — all VGHD runs (CPU)",
        "Kaggle free GPU tier (≤ 30 GPU-h/week) — real LLM answers at scale",
        `Python package halludetect · Streamlit app (${D.repo.appPages} pages) · CLI · ${D.repo.nTests} tests with CI`,
      ]],
      ["FaMicrochip", "Models", C.teal, [
        "NLI: lexical NLI (offline, deterministic) and DeBERTa-large-MNLI " + R.c("he"),
        "LLMs (answer generation): Llama-3.1-8B or Qwen2.5-7B in 4-bit, via Ollama / Hugging Face",
        "Offline simulator with error injection for demos and tests",
      ]],
      ["FaClipboardCheck", "Protocol and metrics", C.saffron, [
        `Split by question: ${nDev} dev / ${nTestSplit} test answers`,
        "Thresholds tuned on dev by Youden’s J",
        "Baselines: citation existence; answer-level NLI. Ablations: − premise, − quotation, − speaker check",
        "AUROC, F1, FPR on correct answers, per-type recall, Hit@1 / Recall@5 / MRR, repair outcomes",
      ]],
    ];
    cols.forEach(([ic, t, col, items], i) => {
      const x = 0.5 + i * 3.05;
      card(slide, pres, { x, y: 1.22, w: 2.85, h: 3.78, accent: col, top: true });
      iconCircle(slide, pres, ic, { x: x + 0.18, y: 1.42, d: 0.46, bg: col });
      txt(slide, t, { x: x + 0.75, y: 1.42, w: 2.0, h: 0.46, fontFace: F.head, fontSize: 12.5, bold: true, color: C.ink, valign: "middle" });
      bullets(slide, items, { x: x + 0.18, y: 2.05, w: 2.55, h: 2.85, fontSize: 10, gap: 6 });
    });
  });

  // 17 ─ Divider
  deck.add("Results & Discussion", (ctx) => K.divider(ctx, { num: 4, title: "Results & Discussion", items: ["Detection", "Per-Type Recall", "Ablations and Retrieval", "Repair and Demo"],
    notes: "Now the results. All of them are on the held-out test split of the synthetic test set, with the lexical NLI unless stated otherwise." }));

  // 18 ─ Detection vs baselines
  deck.add("Results & Discussion", (ctx) => {
    const { slide, pres } = ctx;
    const debTxt = deb
      ? `Independent check without any lexicon tuning (DeBERTa-large-MNLI): VGHD AUROC ${f3(deb.detectors.vghd.auroc)}, F1 ${f3(deb.detectors.vghd.f1)}, FPR ${f3(deb.detectors.vghd.false_positive_rate)}.`
      : "Independent check without lexicon tuning (DeBERTa-large-MNLI): PENDING — run in progress.";
    K.content(ctx, {
      kicker: "Results · Detection", title: "VGHD versus Answer-Level Baselines",
      badge: ["SYNTHETIC TEST SET · LEXICAL NLI", "synth"],
      notes: `On the ${nTestAns} test answers, of which ${v.positives} are hallucinated, VGHD reaches an AUROC of ${f3(v.auroc)} and an F1 of ${f3(v.f1)}, while flagging only ${pct(v.false_positive_rate)} of the correct answers. ` +
        `The answer-level NLI baseline — essentially what RAGAS- or ALCE-style faithfulness sees — reaches an AUROC of ${f3(al.auroc)} and flags ${pct(al.false_positive_rate)} of correct answers. Citation existence alone has high precision but catches only ${pct(ce.recall, 0)} of hallucinations. ` +
        "An honest caveat: the lexical NLI's synonym lexicon was refined while looking at false positives on this test set, so these lexical numbers are optimistic. " +
        (deb ? `The DeBERTa run, which has no such tuning, gives an AUROC of ${f3(deb.detectors.vghd.auroc)} and serves as the independent check.` : "The DeBERTa-large-MNLI run, which involves no tuning, is the independent check; it is still running and will be added."),
    });
    S.vghdDetectorChart(slide, pres, L, { x: 0.4, y: 1.18, w: 5.75, h: 3.3 });
    stat(slide, pres, { x: 6.45, y: 1.22, w: 3.05, h: 1.0, value: f3(v.auroc), label: "VGHD AUROC", sub: `answer-level NLI: ${f3(al.auroc)}`, valueSize: 24 });
    stat(slide, pres, { x: 6.45, y: 2.35, w: 3.05, h: 1.0, value: f3(v.f1), label: "VGHD F1", sub: `answer-level NLI: ${f3(al.f1)}`, valueSize: 24 });
    stat(slide, pres, { x: 6.45, y: 3.48, w: 3.05, h: 1.0, value: pct(v.false_positive_rate), label: "Correct answers flagged", sub: `answer-level NLI: ${pct(al.false_positive_rate)}`, valueSize: 24, color: C.tealDark, accent: C.teal });
    card(slide, pres, { x: 0.5, y: 4.58, w: 9, h: 0.45, accent: C.rose, fill: C.roseSoft, noShadow: true });
    txt(slide, `**Caveat:** lexicon tuned while inspecting this test set → optimistic. ${debTxt}`, { x: 0.68, y: 4.58, w: 8.75, h: 0.45, fontSize: 9.5, valign: "middle" });
  });

  // 19 ─ Per-type recall
  deck.add("Results & Discussion", (ctx) => {
    const { slide, pres } = ctx;
    const citeTypes = ["fabricated_citation", "misattributed_citation", "wrong_speaker"];
    const alCite = Math.max(...citeTypes.map((t) => pt(al, t).recall));
    const vCite = Math.min(...citeTypes.map((t) => pt(v, t).recall));
    const fp = pt(v, "false_premise_accepted");
    K.content(ctx, {
      kicker: "Results · What each detector catches", title: "Recall per Hallucination Type",
      badge: ["SYNTHETIC TEST SET · LEXICAL NLI", "synth"],
      notes: `This is the central result. When the teaching is correct but the citation is invented, wrong, or attributed to the wrong speaker, the answer-level detector flags at most ${pct(alCite, 0)} of such answers — no more often than it flags correct answers, ${pct(al.false_positive_rate, 0)}. To it, these errors are invisible. ` +
        `VGHD catches ${pct(vCite, 0)} of them and names the correct verse. Misquotations and unsupported claims are caught by both, because the wording itself changes. ` +
        `The weakest category for VGHD is false premises: ${Math.round(fp.recall * fp.n)} of ${fp.n}. A lexical model cannot see word order — "the senses are greater than the mind" uses the words of BG 3.42, which says the opposite — so this is where a trained NLI model should help.`,
    });
    S.vghdPerTypeChart(slide, pres, [
      { name: S.DETECTOR.citation_existence, res: L, key: "citation_existence", color: C.slate },
      { name: S.DETECTOR.answer_level_nli, res: L, key: "answer_level_nli", color: C.teal },
      { name: S.DETECTOR.vghd, res: L, key: "vghd", color: C.saffron },
    ], { x: 0.4, y: 1.15, w: 9.2, h: 2.85 });
    const take = [
      ["FaEye", C.teal, `Answer-level NLI flags at most **${pct(alCite, 0)}** of fabricated, misattributed or wrong-speaker answers — vs **${pct(al.false_positive_rate, 0)}** of correct ones`],
      ["FaBullseye", C.saffron, `VGHD catches **${pct(vCite, 0)}** of them and names the correct verse (BG 2.20 → 2.23)`],
      ["FaExclamationTriangle", C.rose, `Weakest: false premises (**${Math.round(fp.recall * fp.n)} of ${fp.n}**) — a lexical model cannot see word order`],
    ];
    take.forEach(([ic, col, t], i) => {
      const x = 0.5 + i * 3.05;
      card(slide, pres, { x, y: 4.1, w: 2.85, h: 0.92, accent: col });
      iconCircle(slide, pres, ic, { x: x + 0.15, y: 4.33, d: 0.42, bg: col });
      txt(slide, t, { x: x + 0.65, y: 4.14, w: 2.12, h: 0.84, fontSize: 9.5, valign: "middle" });
    });
  });

  // 20 ─ Ablations and retrieval
  deck.add("Results & Discussion", (ctx) => {
    const { slide, pres } = ctx;
    const r = L.retrieval;
    K.content(ctx, {
      kicker: "Results · Ablations and retrieval", title: "Every Check Matters; Hindi Retrieval Works",
      badge: ["SYNTHETIC TEST SET · LEXICAL NLI", "synth"],
      notes: `The ablations remove one check at a time. Without the speaker check, wrong-speaker recall drops from ${f2(pt(v, "wrong_speaker").recall)} to ${f2(pt(det.vghd_no_speaker, "wrong_speaker").recall)} and AUROC to ${f3(det.vghd_no_speaker.auroc)}. ` +
        `Without the premise check, the false-positive rate rises from ${f3(v.false_positive_rate)} to ${f3(det.vghd_no_premise.false_positive_rate)}, because correct refusals can no longer be recognised. Without quotation matching, misquotations are still flagged but mostly under the wrong type. ` +
        `On the right is retrieval on the ${r.all.n} answerable questions: Hit@1 ${f3(r.all.hit_at_1)}, Recall@5 ${f3(r.all.recall_at_5)}. The ${r.hi.n} Hindi questions do better than English — Hit@1 ${f3(r.hi.hit_at_1)} — because the glossary expands them into precise concept words; English misses are mostly concept questions phrased in modern words that Besant's translation does not use. A dense multilingual retriever is the natural next step.`,
    });
    txt(slide, "ABLATIONS (TEST SPLIT)", { x: 0.5, y: 1.2, w: 5, h: 0.22, fontSize: 9, bold: true, color: C.indigo, charSpacing: 2 });
    S.ablationTable(slide, pres, L, { x: 0.5, y: 1.45, w: 5.2 });
    card(slide, pres, { x: 0.5, y: 3.35, w: 5.2, h: 1.65, accent: C.saffron });
    bullets(slide, [
      `**Speaker check** is the only defence against wrong-speaker answers (recall ${f2(pt(v, "wrong_speaker").recall)} → ${f2(pt(det.vghd_no_speaker, "wrong_speaker").recall)})`,
      `**Premise check** keeps correct refusals from being flagged (FPR ${pct(v.false_positive_rate)} → ${pct(det.vghd_no_premise.false_positive_rate)})`,
      "**Quotation check** turns a vague flag into the right diagnosis (needed for repair)",
    ], { x: 0.72, y: 3.47, w: 4.85, h: 1.45, fontSize: 10, gap: 5 });
    txt(slide, `VERSE RETRIEVAL (${r.all.n} ANSWERABLE QUESTIONS)`, { x: 6.0, y: 1.2, w: 3.5, h: 0.22, fontSize: 9, bold: true, color: C.indigo, charSpacing: 1 });
    S.retrievalChart(slide, pres, L, { x: 5.9, y: 1.45, w: 3.65, h: 3.55 });
  });

  // 21 ─ Repair and demo
  deck.add("Results & Discussion", (ctx) => {
    const { slide, pres } = ctx;
    const t = rep.tot;
    K.content(ctx, {
      kicker: "Results · From detection to action", title: "Repair, Abstention and the Prototype",
      badge: ["SYNTHETIC TEST SET · LEXICAL NLI", "synth"],
      notes: `Detection drives action. Of the ${t.n} hallucinated test answers, ${t.repaired} were repaired and verified again as faithful, and ${t.correctAbstain} problematic questions were correctly turned into an abstention — ${pct(t.resolved / t.n)} resolved in total. ` +
        `${t.otherAbstain} more answers ended in a safe abstention, ${t.failed} repairs failed re-verification and were not returned as verified, and ${t.missed} answers were missed by the detector. Failed repairs are mostly misquotations whose restored span did not pass the similarity check. ` +
        "On the right is the Gita Verse Verifier page of the prototype app: for the motivating example it flags BG 2.20 as a misattributed citation and suggests BG 2.23, with the entailment scores of the candidate verses. I can show it live after the talk.",
    });
    txt(slide, `OUTCOMES FOR THE ${t.n} HALLUCINATED TEST ANSWERS`, { x: 0.5, y: 1.2, w: 5, h: 0.22, fontSize: 9, bold: true, color: C.indigo, charSpacing: 1 });
    S.repairChart(slide, pres, L, { x: 0.4, y: 1.42, w: 5.2, h: 2.85 });
    stat(slide, pres, { x: 0.5, y: 4.33, w: 2.5, h: 0.72, value: pct(t.resolved / t.n), label: "Resolved", valueSize: 18 });
    txt(slide, "repaired and re-verified, or a correct abstention", { x: 3.1, y: 4.33, w: 2.5, h: 0.72, fontSize: 9.5, color: C.muted, valign: "middle" });
    const img = assets.verifier;
    K.imageFit(slide, { data: img.data, size: img.size, x: 5.85, y: 1.25, w: 3.65, h: 3.25, pres });
    txt(slide, "Gita Verse Verifier (Streamlit app): BG 2.20 flagged as a misattributed citation; BG 2.23 suggested.", { x: 5.85, y: 4.55, w: 3.65, h: 0.45, fontSize: 9, italic: true, color: C.muted, align: "center" });
  });

  // 22 ─ Divider
  deck.add("Conclusion", (ctx) => K.divider(ctx, { num: 5, title: "Conclusion", items: ["Conclusion and Future Scope", "Status", "Publications", "Next Plan"],
    notes: "Finally, the conclusion, the status of the work, publications and the plan for the remaining period." }));

  // 23 ─ Conclusion and future scope
  deck.add("Conclusion", (ctx) => {
    const { slide, pres } = ctx;
    K.content(ctx, {
      kicker: "Conclusion", title: "Conclusion and Future Scope",
      notes: "To conclude: LLMs answering questions on the Gita can fabricate or misattribute verses, misquote the text, attribute words to the wrong speaker, or accept a false premise. " +
        "Existing detectors work on general-domain text and do not use the fixed reference system of a scripture. VGHD does, and the prototype shows the difference exactly where the proposal predicted — correct teachings with wrong references. " +
        "The results so far are on synthetic perturbations with a tuned lexical model, so they are an upper bound; the next steps are real LLM answers, human annotation and the DeBERTa check. " +
        "Future scope includes commentary attribution, other texts such as the Upanishads, Ramayana and Mahabharata, more Indian languages, and dense multilingual retrieval.",
    });
    card(slide, pres, { x: 0.5, y: 1.22, w: 5.3, h: 3.8, accent: C.indigo, top: true });
    iconCircle(slide, pres, "FaFlagCheckered", { x: 0.7, y: 1.42, d: 0.42, bg: C.indigo });
    txt(slide, "Conclusion", { x: 1.22, y: 1.42, w: 4, h: 0.42, fontFace: F.head, fontSize: 14, bold: true, color: C.ink, valign: "middle" });
    bullets(slide, [
      "LLMs answering questions on the Gita may fabricate or misattribute verses, misquote the text, or attribute words to the wrong speaker",
      "Existing detectors target general-domain text and ignore the fixed reference system of a scripture",
      `**VGHD** verifies each claim against the verse it cites; a working prototype (corpus, datasets, verifier, repair policy, app) is complete`,
      `On the synthetic test set (lexical NLI): AUROC **${f3(v.auroc)}** vs **${f3(al.auroc)}** for answer-level faithfulness; **${pct(pt(v, "misattributed_citation").recall, 0)}** of misattributed citations caught vs **${pct(pt(al, "misattributed_citation").recall, 0)}**`,
      "Lexical numbers are an upper bound; real LLM answers and human labels are the next test",
    ], { x: 0.7, y: 1.98, w: 4.95, h: 3.0, fontSize: 10.5, gap: 6 });
    card(slide, pres, { x: 6.05, y: 1.22, w: 3.45, h: 3.8, accent: C.saffron, top: true });
    iconCircle(slide, pres, "FaRoute", { x: 6.25, y: 1.42, d: 0.42, bg: C.saffron });
    txt(slide, "Future Scope", { x: 6.77, y: 1.42, w: 2.6, h: 0.42, fontFace: F.head, fontSize: 14, bold: true, color: C.ink, valign: "middle" });
    bullets(slide, [
      "Attribute an interpretation to a specific commentator rather than flag it as unsupported",
      "Other texts: Upanishads, Ramayana, Mahabharata",
      "More Indian languages; Hindi answers, not only questions",
      "Long-form explanations via atomic-claim decomposition",
      "Dense multilingual retriever (e.g. BGE-M3)",
    ], { x: 6.25, y: 1.98, w: 3.15, h: 3.0, fontSize: 10.5, gap: 6 });
  });

  // 24 ─ Status, publications and next plan
  deck.add("Conclusion", (ctx) => {
    const { slide, pres } = ctx;
    K.content(ctx, {
      kicker: "Conclusion · Status, publications and plan", title: "Status, Publications and Next Semester Plan",
      notes: "Compared with the proposal, where only the topic, the survey and the sources were complete, the design, the corpus, both datasets, the verifier, the repair policy and the benchmark are now complete. " +
        "What remains is the experiment with real LLM answers, with and without retrieval, and the evaluation against human judgement. " +
        `The independent DeBERTa check is ${deb ? "complete" : "running"}. ` +
        "On publications: nothing has been published or submitted yet. A paper draft based on the prototype results has been prepared; it will be submitted once results on real LLM answers are available.",
    });
    S.statusTable(slide, pres, [
      ["Topic finalisation and problem formulation", "Completed", "–"],
      ["Literature survey and research gaps", "Completed", "–"],
      ["Sources of the text identified", "Completed", "O1"],
      ["Detailed design of the framework", "Completed", "O2"],
      [`Verse corpus (${g.nVerses}) and QA dataset (${g.nQa})`, "Completed", "O1"],
      [`Hallucination test set (${g.nTest}) and benchmark`, "Completed", "O2"],
      ["Verifier and repair policy (prototype)", "Completed", "O2"],
      ["Independent check with DeBERTa-large-MNLI", deb ? "Completed" : "In progress", "O2"],
      ["Real LLM answers with / without retrieval", "Planned", "O1"],
      ["Evaluation against human judgement", "Planned", "O2"],
    ], { x: 0.5, y: 1.2, w: 5.45 }, { objCol: true, fontSize: 9, rowH: 0.34 });
    card(slide, pres, { x: 6.2, y: 1.2, w: 3.3, h: 1.4, accent: C.indigo });
    iconCircle(slide, pres, "FaFileAlt", { x: 6.38, y: 1.35, d: 0.4, bg: C.indigo });
    txt(slide, "List of Publications", { x: 6.88, y: 1.35, w: 2.5, h: 0.4, fontFace: F.head, fontSize: 12.5, bold: true, color: C.ink, valign: "middle" });
    txt(slide, "**Nil** — nothing published or submitted yet. A paper draft on the prototype results is prepared (not submitted).", { x: 6.4, y: 1.82, w: 2.98, h: 0.72, fontSize: 9.5 });
    card(slide, pres, { x: 6.2, y: 2.78, w: 3.3, h: 2.22, accent: C.saffron });
    txt(slide, "Next Semester Plan", { x: 6.4, y: 2.86, w: 3.0, h: 0.36, fontFace: F.head, fontSize: 12.5, bold: true, color: C.ink, valign: "middle" });
    const steps = ["Generate answers from 2–3 open LLMs, with and without retrieval (O1)", "Annotate them for the 7 types; measure agreement with VGHD (O2)", "Analyse results; finalise the paper", "Write the dissertation"];
    steps.forEach((s, i) => {
      numCircle(slide, pres, i + 1, { x: 6.4, y: 3.3 + i * 0.42, d: 0.28, fontSize: 9 });
      txt(slide, s, { x: 6.78, y: 3.27 + i * 0.42, w: 2.65, h: 0.36, fontSize: 9, valign: "middle" });
    });
  });

  // 25 ─ References
  deck.add("References", (ctx) => K.references(ctx, { refs: R.list, fontSize: 7.5,
    notes: "The references, numbered in order of first citation. The Huang et al. survey is cited in its ACM TOIS 2025 version, and every work has its own number." }));

  // 26 ─ Thank you
  deck.add(null, (ctx) => K.thanks(ctx, {
    lines: [
      { icon: "FaGithub", text: "Code, data and app: github.com/muditagarwal2308/HallucinationDetection" },
      { icon: "FaDesktop", text: "Live demo: streamlit run app/Home.py → Gita Verse Verifier" },
      { icon: "FaUserTie", text: "Supervisor: Dr. Raju Pal · USICT, Gautam Buddha University" },
    ],
    notes: "Thank you. I am happy to take questions, and I can demonstrate the verifier live.",
  }));

  return deck;
}

module.exports = { build, file: "M1_PartI_MidSem_VGHD_BhagavadGita", label: "M1 — Part I Mid-Semester (VGHD)" };
