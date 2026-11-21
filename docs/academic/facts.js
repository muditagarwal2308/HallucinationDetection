// Facts for the academic documents: every number quoted in the Markdown sources is resolved here
// from the result files in results/ and the datasets in data/, so a re-run of a benchmark followed by
// a rebuild updates every document consistently.
//
// Template syntax in the Markdown sources:  {{source:path.to.value|format}}
//   source  sim | gita | real | deb | calc
//   format  0..4 (fixed decimals), pct0 / pct1 (percentage), int, ci2 / ci3 (CI array "a\u2013b"), raw

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const R = (...p) => path.join(ROOT, ...p);

function readJSON(p) {
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null;
}

function readJSONL(p) {
  if (!fs.existsSync(p)) return null;
  return fs.readFileSync(p, 'utf8').split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l));
}

function count(arr, fn) {
  const out = {};
  for (const x of arr) {
    const k = fn(x);
    out[k] = (out[k] || 0) + 1;
  }
  return out;
}

function loadSources() {
  const src = {
    sim: readJSON(R('results', 'ugrav', 'benchmark_sim-7b.json')),
    gita: readJSON(R('results', 'gita', 'benchmark_lexical.json')),
    real: readJSON(R('results', 'real', 'benchmark_popqa_qwen2.5-1.5b-instruct.json')),
    deb: readJSON(R('results', 'gita', 'benchmark_microsoft_deberta_large_mnli.json')),
  };
  src.calc = derive(src);
  return src;
}

function repairSummary(g) {
  if (!g || !g.repair) return null;
  const s = { total: 0, resolved: 0, repaired: 0, abstained: 0, failed: 0, missed: 0 };
  for (const v of Object.values(g.repair)) {
    s.repaired += v.repaired_and_verified || 0;
    s.abstained += v.abstained || 0;
    s.failed += v.repair_failed || 0;
    s.missed += v.missed || 0;
    s.resolved += v.resolved || 0;
  }
  s.total = s.repaired + s.abstained + s.failed + s.missed;
  s.resolved_share = s.total ? s.resolved / s.total : 0;
  return s;
}

function derive(src) {
  const c = {};
  // datasets ---------------------------------------------------------------------------------
  const verses = readJSON(R('data', 'gita', 'verses.json')) || [];
  c.n_verses = verses.length;
  c.n_verses_700 = verses.filter((v) => v.ref_700).length;
  const spk = count(verses, (v) => v.speaker);
  c.spk_krishna = spk.Krishna || 0;
  c.spk_arjuna = spk.Arjuna || 0;
  c.spk_sanjaya = spk.Sanjaya || 0;
  c.spk_dhritarashtra = spk.Dhritarashtra || 0;
  const qa = readJSONL(R('data', 'gita', 'qa.jsonl')) || [];
  c.n_qa = qa.length;
  const qcat = count(qa, (q) => q.category);
  for (const [k, v] of Object.entries(qcat)) c[`qa_${k}`] = v;
  c.qa_answerable = qa.filter((q) => q.answerable).length;
  c.qa_english_answerable = qa.filter((q) => q.answerable && q.lang === 'en').length;
  c.qa_problematic = qa.filter((q) => !q.answerable).length;
  const ts = readJSONL(R('data', 'gita', 'hallucination_testset.jsonl')) || [];
  c.n_testset = ts.length;
  const tcount = count(ts, (t) => t.type);
  for (const [k, v] of Object.entries(tcount)) c[`ts_${k}`] = v;
  c.ts_negatives = ts.filter((t) => t.label === 0).length;
  c.ts_positives = ts.filter((t) => t.label === 1).length;
  const world = readJSONL(R('data', 'odqa', 'world.jsonl')) || [];
  c.n_world = world.length;
  const tiers = count(world, (w) => w.tier);
  c.world_high = tiers.high || 0;
  c.world_mid = tiers.mid || 0;
  c.world_low = tiers.low || 0;
  c.world_misconceptions = world.filter((w) => w.misconception).length;
  c.world_ambiguous = world.filter((w) => w.kind === 'ambiguous').length;
  const passages = readJSONL(R('data', 'odqa', 'passages.jsonl')) || [];
  c.n_passages = passages.length;
  c.world_withheld = world.filter((w) => w.in_corpus === false).length;
  // tests in the repository
  const testDir = R('tests');
  c.n_tests = fs.existsSync(testDir)
    ? fs.readdirSync(testDir).filter((f) => f.endsWith('.py'))
      .map((f) => (fs.readFileSync(path.join(testDir, f), 'utf8').match(/^\s*def test_/gm) || []).length)
      .reduce((a, b) => a + b, 0)
    : 0;
  // simulation ---------------------------------------------------------------------------------
  const sim = src.sim;
  if (sim) {
    const recs = readJSONL(R('results', 'ugrav', 'records_sim-7b.jsonl')) || [];
    c.sim_seeds = new Set(recs.map((r) => r.meta.seed)).size;
    const isTest = (r) => parseInt(String(r.id).replace(/\D/g, ''), 10) % 2 === 0;
    c.sim_test_instances = recs.filter(isTest).length;
    c.sim_test_questions = new Set(recs.filter(isTest).map((r) => r.id)).size;
    c.sim_dev_instances = recs.length - c.sim_test_instances;
    const p = sim.policies_test;
    c.sim_ugrav_abstain = 1 - p.ug_rav.coverage;
    c.sim_hall_reduction_vs_always = 1 - p.ug_rav.hallucination_rate / p.always_retrieve.hallucination_rate;
    c.sim_hall_reduction_vs_se = 1 - p.ug_rav.hallucination_rate / p.se_gated_no_verification.hallucination_rate;
    c.sim_hall_reduction_vs_never = 1 - p.ug_rav.hallucination_rate / p.never_retrieve.hallucination_rate;
    const gb = sim.gate_budget_test;
    const at = (k, b) => gb[k].find((x) => Math.abs(x.budget - b) < 1e-9);
    c.sim_gb40_fused = at('fused_closed_book_risk', 0.4).accuracy;
    c.sim_gb40_se = at('semantic_entropy', 0.4).accuracy;
    c.sim_gb40_flare = at('flare_min_token_prob', 0.4).accuracy;
    c.sim_gb40_random = at('random', 0.4).accuracy;
    c.sim_gb40_rate = at('fused_closed_book_risk', 0.4).retrieval_rate;
    c.sim_gb20_fused = at('fused_closed_book_risk', 0.2).accuracy;
    c.sim_gb20_se = at('semantic_entropy', 0.2).accuracy;
    c.sim_gb20_random = at('random', 0.2).accuracy;
    c.sim_gb0 = at('random', 0.0).accuracy;
    c.sim_gb100 = at('random', 1.0).accuracy;
    c.sim_gb40_gain_share = (c.sim_gb40_fused - c.sim_gb0) / (c.sim_gb100 - c.sim_gb0);
    c.sim_rag_gain = sim.rag_accuracy - sim.closed_book_accuracy;
    const det = sim.detection;
    c.sim_fusion_gain = det.fused_with_evidence.auroc - det.fused_closed_book.auroc;
    c.sim_se_vs_lex = det.semantic_entropy.auroc - det.lexical_entropy.auroc;
    c.sim_n_det_signals = Object.keys(det).length;
  }
  // VGHD ---------------------------------------------------------------------------------------
  for (const [key, g] of [['gita', src.gita], ['deb', src.deb]]) {
    if (!g) continue;
    const r = repairSummary(g);
    if (r) {
      c[`${key}_rep_total`] = r.total;
      c[`${key}_rep_resolved`] = r.resolved;
      c[`${key}_rep_resolved_share`] = r.resolved_share;
      c[`${key}_rep_repaired`] = r.repaired;
      c[`${key}_rep_abstained`] = r.abstained;
      c[`${key}_rep_failed`] = r.failed;
      c[`${key}_rep_missed`] = r.missed;
    }
    const v = g.detectors.vghd;
    c[`${key}_test_n`] = v.n;
    c[`${key}_test_pos`] = v.positives;
    c[`${key}_test_neg`] = v.n - v.positives;
    c[`${key}_vghd_fp`] = Math.round(v.false_positive_rate * (v.n - v.positives));
    c[`${key}_ans_fp`] = Math.round(g.detectors.answer_level_nli.false_positive_rate * (v.n - v.positives));
    c[`${key}_fp_detected`] = Math.round(v.per_type.false_premise_accepted.recall * v.per_type.false_premise_accepted.n);
    c[`${key}_fp_n`] = v.per_type.false_premise_accepted.n;
    c[`${key}_misq_detected`] = Math.round(v.per_type.misquotation.recall * v.per_type.misquotation.n);
    c[`${key}_tuning_grid`] = Object.keys(g.tuning || {}).join(', ');
    for (const [t, d] of Object.entries(v.per_type)) {
      c[`${key}_n_${t}`] = d.n;
    }
  }
  if (src.gita && src.gita.retrieval) {
    c.gita_ret_miss = Math.round((1 - src.gita.retrieval.all.hit_at_1) * src.gita.retrieval.all.n);
  }
  // real pilot ---------------------------------------------------------------------------------
  const real = src.real;
  if (real) {
    const recs = readJSONL(R('results', 'real', 'records_popqa_qwen2.5-1.5b-instruct.jsonl')) || [];
    c.real_n = real.n_records;
    c.real_k = recs.length ? recs[0].closed.samples.length : null;
    const tierc = count(recs, (r) => r.meta.tier);
    c.real_tier_low = tierc.low || 0;
    c.real_tier_mid = tierc.mid || 0;
    c.real_tier_high = tierc.high || 0;
    c.real_closed_wrong = recs.filter((r) => !r.closed_correct).length;
    c.real_rag_wrong = recs.filter((r) => !r.rag_correct).length;
    c.real_rag_fixes = recs.filter((r) => !r.closed_correct && r.rag_correct).length;
    c.real_rag_breaks = recs.filter((r) => r.closed_correct && !r.rag_correct).length;
    c.real_gold_retrieved = recs.filter((r) => r.meta.gold_retrieved).length / Math.max(recs.length, 1);
    c.real_answer_in_passages = recs.filter((r) => r.meta.answer_in_passages).length / Math.max(recs.length, 1);
    c.real_mean_minutes = recs.reduce((a, r) => a + (r.seconds || 0), 0) / Math.max(recs.length, 1) / 60;
    c.real_total_hours = recs.reduce((a, r) => a + (r.seconds || 0), 0) / 3600;
    const relations = count(recs, (r) => r.meta.relation);
    c.real_relations = Object.entries(relations).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} (${v})`).join(', ');
    c.real_n_relations = Object.keys(relations).length;
    const det = real.detection;
    if (det) {
      const ranked = Object.entries(det).sort((a, b) => b[1].auroc - a[1].auroc);
      c.real_best_signal = ranked[0][0];
      c.real_best_auroc = ranked[0][1].auroc;
    }
  }
  return c;
}

function getPath(obj, p) {
  let cur = obj;
  for (const part of p.split('.')) {
    if (cur === null || cur === undefined) return undefined;
    cur = cur[part];
  }
  return cur;
}

function format(v, fmt, ref) {
  if (v === undefined || v === null) throw new Error(`template value missing: ${ref}`);
  if (!fmt || fmt === 'raw') return String(v);
  if (/^\d$/.test(fmt)) return Number(v).toFixed(Number(fmt));
  if (fmt === 'int') return Math.round(Number(v)).toLocaleString('en-IN');
  if (fmt === 'pct0') return `${(100 * v).toFixed(0)}%`;
  if (fmt === 'pct1') return `${(100 * v).toFixed(1)}%`;
  if (fmt === 'ci2' || fmt === 'ci3') {
    const d = Number(fmt.slice(2));
    return `${Number(v[0]).toFixed(d)}\u2013${Number(v[1]).toFixed(d)}`;
  }
  throw new Error(`unknown format ${fmt} in ${ref}`);
}

function resolveTemplates(text, src, docName) {
  return text.replace(/\{\{\s*([a-z]+):([A-Za-z0-9_.\-]+)(?:\|([A-Za-z0-9]+))?\s*\}\}/g, (m, s, p, f) => {
    if (!(s in src)) throw new Error(`${docName}: unknown source ${s} in ${m}`);
    if (!src[s]) throw new Error(`${docName}: source '${s}' not available (result file missing) for ${m}`);
    return format(getPath(src[s], p), f, `${docName}: ${m}`);
  });
}

module.exports = { loadSources, resolveTemplates };
