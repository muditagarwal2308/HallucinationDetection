<div align="center">

# Hallucination Detection Studio

**Detecting and mitigating hallucinations in LLM question answering — two research frameworks, implemented, tested and evaluated**

[![CI](https://github.com/muditagarwal2308/HallucinationDetection/actions/workflows/ci.yml/badge.svg)](https://github.com/muditagarwal2308/HallucinationDetection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Offline](https://img.shields.io/badge/runs-fully%20offline-0d9488)

</div>

This repository turns an M.Tech dissertation on **hallucination in large language models** into working
software, data, experiments and reports. It analyses the six source documents — study notes, two
progress reports and two slide decks by **Manju Pal** (M.Tech CSE, AI & Robotics, Gautam Buddha University;
supervisor **Dr. Raju Pal**) — and implements both research directions they propose:

| | **UG-RAV** — Uncertainty-Gated Retrieval-Augmented Verification | **VGHD** — Verse-Grounded Hallucination Detection |
|---|---|---|
| Task | Factual open-domain QA (closed-book LLM) | Retrieval-augmented QA on the Bhagavad Gita |
| Idea | Semantic entropy decides *when* to retrieve; NLI checks the evidence; ΔSE + logistic fusion decide to **accept, answer from evidence, or abstain** | A scripture has a fixed reference system, so verify an answer **verse by verse** — citations, claim support, quotations, speakers, premises — then **repair or abstain** |
| Code | [`src/halludetect/ugrav`](src/halludetect/ugrav) | [`src/halludetect/gita`](src/halludetect/gita) |
| Report | [03 UG-RAV Technical Report](reports/03_UG-RAV_Technical_Report.md) | [04 VGHD Technical Report](reports/04_VGHD_Technical_Report.md) |

<!-- KEY_RESULTS -->

---

## Contents

- [Quick start](#quick-start)
- [The app](#the-app)
- [Command line](#command-line)
- [Python API](#python-api)
- [Reports](#reports)
- [Results at a glance](#results-at-a-glance)
- [Repository layout](#repository-layout)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Data and licences](#data-and-licences)
- [Design principles](#design-principles)
- [Acknowledgements and citation](#acknowledgements-and-citation)

---

## Quick start

```bash
git clone https://github.com/muditagarwal2308/HallucinationDetection.git
cd HallucinationDetection
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[app,data,dev]"                        # offline components (no GPU, no downloads)
streamlit run app/Home.py                               # the interactive studio
pytest -q                                               # the test suite
```

Everything above runs **offline** with a deterministic lexical NLI and two simulators. For real models:

```bash
pip install -e ".[ml]"                                  # torch + transformers
halludetect verify --nli microsoft/deberta-large-mnli -q "..." -a "..."
python scripts/run_real_experiment.py --n 60 --k 10     # Qwen2.5-1.5B + Wikipedia + DeBERTa on CPU
```

Any OpenAI-compatible server works as the LLM back-end — e.g. [Ollama](https://ollama.com) (`ollama pull qwen2.5:7b`,
then `--llm qwen2.5:7b`), vLLM, LM Studio or OpenAI. Docker: `docker build -t halludetect . && docker run -p 8501:8501 halludetect`.

## The app

`streamlit run app/Home.py` opens a multi-page studio:

<table>
<tr>
<td width="33%"><img src="docs/screenshots/home.png" alt="Home: headline results and architecture"></td>
<td width="33%"><img src="docs/screenshots/ugrav_live.png" alt="UG-RAV Live: one question traced end to end"></td>
<td width="33%"><img src="docs/screenshots/verifier.png" alt="Gita Verse Verifier: BG 2.20 flagged, BG 2.23 suggested"></td>
</tr>
<tr>
<td align="center"><sub>Home — headline results and architecture</sub></td>
<td align="center"><sub>UG-RAV Live — <i>Godan</i>: wrong closed-book answer → retrieve → verified answer</sub></td>
<td align="center"><sub>Gita Verse Verifier — BG 2.20 flagged, BG 2.23 suggested</sub></td>
</tr>
</table>

| Page | What you can do |
|---|---|
| **Hallucination Explorer** | Taxonomy (factuality vs faithfulness), examples, detector families |
| **Semantic Entropy Lab** | Type answers or sample them; see meaning clusters, semantic vs lexical entropy, the worked examples from the notes |
| **UG-RAV Live** | Trace one question through sampling → semantic entropy → gate → retrieval → NLI support → ΔSE → fused risk → decision |
| **UG-RAV Results** | Detection AUROCs with CIs, risk–coverage curves, policies, retrieval budgets, popularity analysis — simulation and real model |
| **Gita Verse Verifier** | Paste a question and an answer; get per-claim verse checks, issues, the correct verse and a repaired answer |
| **Gita RAG Assistant** | Ask in English, Hindi or romanised Sanskrit; watch retrieval, generation (simulator with hallucination injection, or your LLM) and verification |
| **VGHD Results** | Detector comparison, per-type recall, ablations, repairs |
| **Corpus Browser** | All 701 verses in Devanagari, IAST and English, with speakers |

## Command line

```bash
halludetect verify -q "Which verse says the soul cannot be cut by weapons?" \
                   -a "The soul cannot be cut by weapons. This is described in BG 2.20."
# → misattributed_citation: BG 2.20 does not support this claim; BG 2.23 does. Repair: "... BG 2.23."

halludetect gita-ask "कर्म और उसके फल के बारे में कृष्ण क्या कहते हैं?" --inject wrong_speaker
halludetect ugrav-ask "Who wrote the Hindi novel Godan?"
halludetect se -q "What is the capital of Australia?" Canberra "It is Canberra." Sydney
halludetect search "karmanye vadhikaraste ma phaleshu"
halludetect bench-gita            # VGHD benchmark (lexical NLI)
halludetect bench-ugrav           # UG-RAV simulation benchmark
```

## Python API

```python
from halludetect.gita import GitaCorpus, VerseVerifier
from halludetect.gita.policy import RepairPolicy

corpus = GitaCorpus.load()
report = VerseVerifier(corpus).verify(
    "Which verse says that the soul cannot be cut by weapons?",
    "The Gita states that the soul cannot be cut by weapons or burnt by fire. This is described in BG 2.20.")
report.types                                  # {'misattributed_citation'}
report.claims[0].best_ref                     # '2.23'
RepairPolicy(corpus).decide(report).answer    # '... This is described in BG 2.23.'
```

```python
from halludetect.nli import get_nli
from halludetect.ugrav import UGRAV, load_passages, load_world
from halludetect.ugrav.data import PassageRetriever
from halludetect.ugrav.simulator import SimulatedQALLM   # or llm.hf_local.HFLocalLLM / llm.openai_compat.OpenAICompatLLM

world = load_world()
pipe = UGRAV(SimulatedQALLM(world, seed=0), get_nli("lexical"), PassageRetriever(load_passages()))
trace = pipe.run("Who wrote the Hindi novel Godan?", aliases=["Premchand", "Munshi Premchand"])
trace["decision"]   # {'action': 'answer_from_evidence', 'answer': 'Munshi Premchand', 'retrieved': True, ...}
```

## Reports

| # | Report | What it covers | PDF |
|---|---|---|---|
| 01 | [Source Document Analysis](reports/01_Source_Document_Analysis.md) | Critical review of the six documents: the two research directions compared, methodology, a 27-item consistency audit, feasibility and risk, prioritised recommendations | [PDF](reports/pdf/01_Source_Document_Analysis.pdf) |
| 02 | [Hallucination Primer](reports/02_Hallucination_Primer.md) | A study guide: taxonomy, causes, uncertainty- and evidence-based detection with equations and worked examples, evaluation metrics, datasets, viva Q&A | [PDF](reports/pdf/02_Hallucination_Primer.pdf) |
| 03 | [UG-RAV Technical Report](reports/03_UG-RAV_Technical_Report.md) | Implementation, simulation study, real-model pilot, ablations, popularity analysis | [PDF](reports/pdf/03_UG-RAV_Technical_Report.pdf) |
| 04 | [VGHD Technical Report](reports/04_VGHD_Technical_Report.md) | Corpus, datasets, detector, repair policy, benchmark with baselines and ablations | [PDF](reports/pdf/04_VGHD_Technical_Report.pdf) |

## Results at a glance

<!-- RESULTS_GLANCE -->

## Repository layout

```text
HallucinationDetection/
├── src/halludetect/
│   ├── metrics.py            AUROC, AUPRC, ECE, risk–coverage, AURAC, bootstrap CIs
│   ├── text.py               normalisation, answer matching, concept lexicon, q⊕a statements
│   ├── uncertainty/          semantic clustering, semantic/predictive/lexical entropy, self-consistency
│   ├── nli/                  LexicalNLI (offline) and HFNLI (DeBERTa etc.)
│   ├── retrieval/            BM25, reciprocal-rank fusion
│   ├── llm/                  LLM interface, prompts, Hugging Face (shared-prefill sampler), OpenAI-compatible
│   ├── ugrav/                world data, simulator, signals, fusion, Algorithm 1 + baselines, benchmark, live pipeline
│   ├── gita/                 corpus, citations, claims, premise checks, verifier, repair policy, RAG pipeline, benchmark
│   └── cli.py                `halludetect` command
├── app/                      Streamlit studio (Home + 8 pages)
├── data/                     Gita corpus, QA and test sets; ODQA world and passages (see data/README.md)
├── results/                  benchmark outputs (JSON) and cached records
├── reports/                  Markdown reports, figures/, pdf/
├── scripts/                  corpus/dataset builders, real experiment, figures, PDF builder
├── notebooks/                Kaggle GPU notebook for the full-scale UG-RAV run
└── tests/                    unit, integration and app tests
```

## Reproducing the experiments

| Result | Command | Time |
|---|---|---|
| Gita corpus | `python scripts/build_gita_corpus.py` | 1 min (network, cached) |
| Gita QA and test sets | `python scripts/build_gita_datasets.py` | seconds |
| ODQA world | `python scripts/build_odqa_world.py` | seconds |
| VGHD benchmark (lexical) | `halludetect bench-gita` | ~3 min |
| VGHD benchmark (DeBERTa) | `halludetect bench-gita --nli microsoft/deberta-large-mnli --search-k 6` | ~1 h CPU |
| UG-RAV simulation | `halludetect bench-ugrav --seeds 5` | ~30 s |
| UG-RAV real-model pilot | `python scripts/run_real_experiment.py --n 60 --k 10 --threads 12` | ~1–2 h CPU |
| UG-RAV full scale | [`notebooks/ugrav_kaggle_gpu.ipynb`](notebooks/ugrav_kaggle_gpu.ipynb) on a Kaggle T4 | ~3 GPU-h |
| Figures / PDFs | `python scripts/make_figures.py` · `python scripts/build_pdfs.py` | 1 min / 5 min |

All randomness is seeded; real-model runs cache every record and resume after interruption.

## Data and licences

Code: [MIT](LICENSE). The Gita corpus uses **Annie Besant's 1922 translation (public domain)** with the
Sanskrit text, transcribed on English Wikisource; retrieved Wikipedia excerpts are CC BY-SA 4.0; PopQA is
downloaded at run time and not redistributed. Details, dataset statistics and the 700/701 verse-numbering
note are in [data/README.md](data/README.md). The original dissertation documents are not included in this
public repository.

## Design principles

* **Inference-only.** No LLM is trained or fine-tuned; the only trained component is a logistic regression over a few scalar signals — as required by the dissertation.
* **Honest evaluation.** Dev/test splits by question, out-of-fold fusion, bootstrap CIs, matched retrieval budgets, and explicit caveats wherever a number is optimistic (synthetic perturbations, simulation, tuned lexicon).
* **Explainable decisions.** Every flag names its evidence (the verse, the passage, the signal values), and every repair is re-verified before it is returned.
* **Runs anywhere.** Offline by default; real models plug in through one interface (Hugging Face, Ollama/OpenAI-compatible).

## Acknowledgements and citation

The research questions, frameworks (UG-RAV, VGHD) and literature review originate from the dissertation work of
**Manju Pal** under the supervision of **Dr. Raju Pal**, University School of Information and Communication
Technology, Gautam Buddha University. This repository provides an independent implementation and evaluation.

```bibtex
@software{halludetect2026,
  title  = {Hallucination Detection Studio: UG-RAV and VGHD implementations},
  author = {Agarwal, Mudit},
  year   = {2026},
  url    = {https://github.com/muditagarwal2308/HallucinationDetection}
}
```
