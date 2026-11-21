// Loads every number shown on the slides from the repository at build time.
// Nothing numeric about results is hard-coded in the deck sources: if a result file changes,
// re-running `node presentations/build_decks.js` refreshes every slide and speaker note.
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const P = (...p) => path.join(ROOT, ...p);

const FILES = {
  sim: P("results", "ugrav", "benchmark_sim-7b.json"),
  gitaLex: P("results", "gita", "benchmark_lexical.json"),
  // Pending results: the decks render a "PENDING" panel until these files exist.
  real: process.env.HD_REAL_JSON || P("results", "real", "benchmark_popqa_qwen2.5-1.5b-instruct.json"),
  realRecords: process.env.HD_REAL_RECORDS || P("results", "real", "records_popqa_qwen2.5-1.5b-instruct.jsonl"),
  gitaDeb: process.env.HD_DEBERTA_JSON || P("results", "gita", "benchmark_microsoft_deberta_large_mnli.json"),
};

const rel = (f) => path.relative(ROOT, f).split(path.sep).join("/");

function readJson(file, required) {
  if (!fs.existsSync(file)) {
    if (required) throw new Error(`Required result file missing: ${file}`);
    return null;
  }
  return JSON.parse(fs.readFileSync(file, "utf-8"));
}

function readJsonl(file) {
  if (!fs.existsSync(file)) return [];
  return fs
    .readFileSync(file, "utf-8")
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));
}

const count = (arr, key) =>
  arr.reduce((m, x) => {
    const k = typeof key === "function" ? key(x) : x[key];
    m[k] = (m[k] || 0) + 1;
    return m;
  }, {});

// split rule used by both benchmarks: odd question number -> dev, even -> test
const splitOf = (qid) => {
  const digits = String(qid).replace(/\D/g, "") || "0";
  return BigInt(digits) % 2n === 1n ? "dev" : "test";
};

function load() {
  const sim = readJson(FILES.sim, true);
  const gitaLex = readJson(FILES.gitaLex, true);
  const real = readJson(FILES.real, false);
  const gitaDeb = readJson(FILES.gitaDeb, false);

  const verses = readJson(P("data", "gita", "verses.json"), true);
  const qa = readJsonl(P("data", "gita", "qa.jsonl"));
  const testset = readJsonl(P("data", "gita", "hallucination_testset.jsonl"));
  const world = readJsonl(P("data", "odqa", "world.jsonl"));
  const passages = readJsonl(P("data", "odqa", "passages.jsonl"));

  let realRecords = null;
  if (real) {
    const recs = readJsonl(FILES.realRecords);
    if (recs.length) {
      realRecords = {
        n: recs.length,
        test: recs.filter((r) => splitOf(r.id) === "test").length,
        tiers: count(recs, (r) => (r.meta && (r.meta.tier || r.meta.pop_tier)) || "all"),
      };
    }
  }

  const verseByRef = Object.fromEntries(verses.map((v) => [v.ref, v]));
  const chapters = new Set(verses.map((v) => v.chapter)).size;
  const speakers = count(verses, "speaker");
  const qaCats = count(qa, "category");
  const qaLang = count(qa, "lang");
  const tsTypes = count(testset, "type");
  const tsLabels = count(testset, "label");
  const tsSplit = count(testset, (r) => splitOf(r.qid));
  const worldTiers = count(world, "tier");
  const worldKinds = count(world, "kind");
  const misconceptions = world.filter((w) => w.misconception).length;
  const withheld = world.filter((w) => w.in_corpus === false).length;

  const testFiles = fs.readdirSync(P("tests")).filter((f) => f.endsWith(".py"));
  const nTests = testFiles.reduce(
    (s, f) => s + (fs.readFileSync(P("tests", f), "utf-8").match(/^\s*def test_/gm) || []).length,
    0
  );
  const appPages = 1 + fs.readdirSync(P("app", "pages")).filter((f) => f.endsWith(".py")).length;
  const cliCmds = (fs.readFileSync(P("src", "halludetect", "cli.py"), "utf-8").match(/add_parser\(/g) || []).length;
  const subpackages = fs
    .readdirSync(P("src", "halludetect"), { withFileTypes: true })
    .filter((d) => d.isDirectory() && !d.name.startsWith("__") && fs.existsSync(P("src", "halludetect", d.name, "__init__.py")))
    .map((d) => d.name);

  return {
    sim, gitaLex, real, gitaDeb, realRecords,
    files: Object.fromEntries(Object.entries(FILES).map(([k, v]) => [k, rel(v)])),
    gita: {
      nVerses: verses.length, chapters, speakers, verseByRef,
      nQa: qa.length, qaCats, qaLang,
      nTest: testset.length, tsTypes, tsLabels, tsSplit,
    },
    odqa: {
      nQuestions: world.length, tiers: worldTiers, kinds: worldKinds, misconceptions,
      nPassages: passages.length, withheld,
    },
    repo: { nTests, appPages, cliCmds, subpackages },
  };
}

// ---------------------------------------------------------------- formatting
const f2 = (x) => Number(x).toFixed(2);
const f3 = (x) => Number(x).toFixed(3);
const pct = (x, d = 1) => `${(100 * Number(x)).toFixed(d)}%`;
const int = (x) => Number(x).toLocaleString("en-IN");

// "real:popqa:Qwen/Qwen2.5-1.5B-Instruct:microsoft/deberta-large-mnli:K=10"
function parseRealLabel(label) {
  const parts = String(label || "").split(":");
  return {
    dataset: (parts[1] || "popqa").toUpperCase() === "POPQA" ? "PopQA" : parts[1],
    model: (parts[2] || "").split("/").pop() || "Qwen2.5-1.5B-Instruct",
    nli: (parts[3] || "").split("/").pop() || "deberta-large-mnli",
    k: (parts[4] || "K=10").replace("K=", ""),
  };
}

// Repair outcomes on the hallucinated test answers (see gita/benchmark.py):
// resolved = repaired and re-verified, or a correct abstention on a problematic question.
function repairSummary(repair) {
  const types = Object.keys(repair);
  const tot = { n: 0, repaired: 0, correctAbstain: 0, otherAbstain: 0, failed: 0, missed: 0, resolved: 0 };
  const rows = types.map((t) => {
    const r = repair[t];
    const problematic = t === "false_premise_accepted" || t === "out_of_scope_answered";
    const abst = r.abstained || 0;
    const row = {
      type: t,
      repaired: r.repaired_and_verified || 0,
      correctAbstain: problematic ? abst : 0,
      otherAbstain: problematic ? 0 : abst,
      failed: r.repair_failed || 0,
      missed: r.missed || 0,
      resolved: r.resolved || 0,
    };
    row.n = row.repaired + abst + row.failed + row.missed;
    for (const k of Object.keys(tot)) tot[k] += row[k];
    return row;
  });
  return { rows, tot };
}

module.exports = { load, f2, f3, pct, int, parseRealLabel, repairSummary, ROOT, P };
