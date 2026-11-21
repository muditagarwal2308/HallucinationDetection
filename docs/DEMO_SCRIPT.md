# Demo script — Hallucination detection & mitigation (UG-RAV + VGHD)

A rehearsed, timed walkthrough for the viva, milestone reviews or any live showcase. It runs **fully offline
on a laptop CPU**. It uses the answer simulator and the lexical NLI, so it needs no GPU, no network and no API
key.

| | |
|---|---|
| **Duration** | 15 min core walkthrough + 5 min optional (CLI, real-model results, Q&A crib) |
| **Audience** | Examiners and supervisor (viva / milestone review), or a technical audience |
| **Story** | *LLMs answer fluently even when they are wrong → measure uncertainty → check the evidence → accept, correct or abstain*, first for open-domain QA (UG-RAV), then for scripture QA where every claim must match a verse (VGHD) |
| **Backup** | `python scripts/run_demo.py --pause` replays every scene in the terminal; screenshots in `docs/screenshots/`; decks in `presentations/` |

---

## 0. Before you start (5 minutes, the day before and again 10 minutes before)

```powershell
# one-time setup
git clone https://github.com/muditagarwal2308/HallucinationDetection
cd HallucinationDetection
python -m venv .venv; .\.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -e ".[app]"

# health check: data, CLI, demo, app, artefacts (about 3 minutes); expect "N/N checks passed"
python scripts/e2e_check.py --skip-tests

# start the app (keep this terminal open)
streamlit run app/Home.py
```

Checklist:

- [ ] Browser at **http://localhost:8501**, zoom 110–125 %, sidebar expanded, dark/light theme as rehearsed.
- [ ] A second terminal is open in the repo with the venv active and a large font (≥ 16 pt) for the CLI segment.
- [ ] Notifications are muted, and other browser tabs are closed.
- [ ] Open each app page once before the audience arrives (warms the caches; the first page load takes ≈ 2 s).
- [ ] The milestone deck (`presentations/M4_*.pptx`) is open on the *Live demo* agenda slide.

> **If something breaks:** say *“let me show the same scene from the recorded run”*. Then run
> `python scripts/run_demo.py --scene N` (N = 1–8), or show the screenshot. Every scene below lists its
> fallback scene number.

---

## 1. Opening — the problem (1 min) · *Home* page

**Do:** show the **Home** page hero and the two project cards (UG-RAV and VGHD).

**Say:**
> “Large language models are fluent even when they are wrong. Ask one *‘Who wrote Godan?’* and it may say
> *Rabindranath Tagore*. That is a perfect sentence, but the author is Munshi Premchand, and nothing in the text
> reveals the error. My dissertation asks two questions. First, *can we tell when an answer is likely to be
> wrong without retraining the model?* Second, *once we suspect it, can we fix it, or decline to answer?*
> I study this in two settings: open-domain question answering (UG-RAV), and question answering over the
> Bhagavad Gita, where every claim must be grounded in a specific verse (VGHD).”

**Expect:** headline metric tiles on Home, read live from `results/*.json`.

---

## 2. What is a hallucination? (1.5 min) · *🧩 Hallucination Explorer*

**Do:**

1. Scroll through **A taxonomy of hallucinations**, pausing on the factuality vs. faithfulness split.
2. In **Classify the hallucination**, select example **6. “Godan and Tagore”**.
3. Pick the answer, then reveal it.

**Say:**
> “Hallucinations differ in kind. This one is a *factual* error from the model's own memory: an internal
> hallucination. Retrieval can fix internal errors but can introduce *external* ones, where the model copies
> a misleading passage. That is why the method verifies evidence instead of trusting it.”

**Fallback:** none needed (static content).

---

## 3. Semantic entropy — measuring uncertainty in meaning (2 min) · *🎲 Semantic Entropy Lab*

**Do:**

1. Choose preset **“Paraphrases · Canberra”**. Point out several surface forms but *one* meaning cluster:
   semantic entropy ≈ 0 while lexical entropy is high.
2. Choose preset **“Confabulation · rare novel (4/3/2/1)”**. There are many clusters and semantic entropy is
   high.
3. Choose preset **“Godan · K = 5, one misconception”**. There are two clusters (Premchand, Tagore).

**Say:**
> “We sample several answers and group them by *meaning*. Two answers share a cluster if each entails the
> other. Semantic entropy is the entropy over clusters. Paraphrases do not count as disagreement, but genuine
> confusion does. That is the Nature 2024 method I build on. Its blind spot is a model that is *consistently*
> wrong: all samples agree, so entropy is low.”

**Fallback:** `python scripts/run_demo.py --scene 1`.

---

## 4. UG-RAV live — gate, verify, decide (3 min) · *🧭 UG-RAV Live*

Backend: **Offline answer simulator**, seed 0, K = 10. Use the tuned thresholds (leave “Override” off).

| Step | World question to pick | What happens | Talking point |
|---|---|---|---|
| a | **Who wrote the Hindi novel Godan?** (w001) | closed-book *Rabindranath Tagore* ✗, SE 1.09, risk r̂₀ 0.63 ≥ τ_low → **retrieve** → passage “Godan is a Hindi novel written by Munshi Premchand…”, NLI support 0.99 → **ANSWER FROM EVIDENCE: Munshi Premchand ✓** | “Uncertainty opens the gate; evidence decides.” |
| b | **Who invented the crescograph?** (w117) | all 10 samples say *Satyendra Nath Bose* ✗, **SE = 0.00**, yet the fused closed-book risk (0.23) still opens the gate → evidence contradicts → **Jagadish Chandra Bose ✓** | “A *confident error* that semantic entropy alone would accept. 32 % of the simulator's errors are confident; on low-SE answers SE has AUROC 0.51 (chance) and the fused detector 0.93.” |
| c | **Who wrote the collection of poems Gitanjali?** (w003) | consistent and correct, r̂₀ 0.15 < τ_low → **ACCEPT without retrieval** | “We pay for retrieval only when needed: 62 % of questions on the test split.” |
| d | *(type your own)* **Who wrote Godan?** | the paraphrase is matched to world question w001, with the same trace as (a) | “The simulator recognises paraphrases of its 197 questions.” |

Then open the expanders: **per-passage NLI**, **ΔSE**, **fusion features**, and **decision rationale**.

**Say:**
> “Three signals are fused by a logistic regression, the only trained component: closed-book uncertainty,
> NLI evidence support, and ΔSE, the change in uncertainty once evidence is shown. The policy then accepts,
> answers from verified evidence, or abstains.”

**Fallback:** `python scripts/run_demo.py --scene 2`, `--scene 3`, `--scene 4`.

**Honesty line (say it):**
> “These traces come from a calibrated *simulator* that encodes these behaviours by construction. It validates
> the pipeline end to end; the hypothesis itself is tested on a real model, which I show next.”

---

## 5. UG-RAV results (1.5 min) · *📈 UG-RAV Results*

**Do:** with **Simulation · sim-7b** selected, step through the tabs:

1. **🎯 Detection.** Fused with evidence: AUROC **0.921** [0.903–0.940]. Semantic entropy: **0.696**.
2. **⚖️ Policies.** UG-RAV hallucination rate **6.9 %** vs **12.4 %** for always-retrieve and **30.6 %** for
   never-retrieve. Coverage is 87.3 %, at a retrieval rate of 0.62. Show the risk–coverage curve: UG-RAV has
   the highest AURAC, 0.957.
3. **🌐 Popularity.** Long-tail (low-popularity) questions benefit most from retrieval: it fixes 50 % of
   low-tier questions, compared with 4 % of high-tier ones.

Then switch the run selector to **Real model · PopQA · qwen2.5-1.5b-instruct**: a pilot with 60 stratified
PopQA questions, Qwen2.5-1.5B-Instruct on CPU, Wikipedia retrieval and DeBERTa-large-MNLI.

**Say:**
> “On a real model the pattern holds qualitatively: retrieval fixes long-tail errors, and evidence support adds
> to semantic entropy. With N = 60 the confidence intervals are wide, so I treat it as a pilot. The full 7–8B
> experiments on TriviaQA, NQ-Open and PopQA run on the Kaggle GPU with the provided notebook.”

---

## 6. VGHD — verse-grounded verification (2.5 min) · *📜 Gita Verse Verifier*

Use NLI back-end **Lexical (offline)**. The support threshold defaults to the dev-tuned value (0.10).

| Preset (pills) | Expected verdict | Talking point |
|---|---|---|
| **Motivating example · BG 2.20** | 🔀 *misattributed citation*. The claim is true, but BG 2.20 does not support it; **repair → BG 2.23**, which re-verifies as faithful | “Answer-level checks call this answer correct. Verse-level checking catches the wrong verse and fixes it.” |
| **Fabricated · BG 2.80** | 👻 *fabricated citation*. Chapter 2 has only 72 verses | “The model invented a verse that does not exist.” |
| **Wrong speaker** | 🗣️ *wrong speaker*. Arjuna → **Krishna** | “Who said it matters in scripture.” |
| **Misquotation · BG 18.66** | ✂️ *misquotation*. Similarity 0.55; the quotation is restored from the verse | “Quotes are checked character by character against the source.” |
| **False premise · “Time am I”** | ⚠️ *false premise accepted*. The system **abstains** and explains that Krishna, not Arjuna, speaks BG 11.32 | “The right answer to a wrong question is a correction.” |
| **Out of scope · smartphones** | 🚫 *out of scope answered* → **abstain** | “The Gita says nothing about smartphones. We decline instead of inventing.” |
| **Correct quotation · BG 18.66** | ✅ faithful | “No false alarm on a correct answer.” |

**Fallback:** `python scripts/run_demo.py --scene 5` and `--scene 6`.

---

## 7. Gita RAG assistant (1.5 min) · *🪔 Gita RAG Assistant*

**Do:**

1. Generator: **Offline answer simulator**. Set **Inject a hallucination → Fabricated citation**.
2. Click the English example **“What does Krishna say about action and its fruits?”** and press **Ask**. The
   timeline shows retrieved verses (BG 2.47 first), then the generated answer with a fabricated verse number,
   then **detected**, **repaired** and **re-verified** ✓.
3. Switch to the **हिन्दी · Hindi** tab and ask **“आत्मा को शस्त्र क्यों नहीं काट सकते?”**. BG 2.23 is
   retrieved through the Hindi glossary and the Devanagari index.

**Say:**
> “The same verifier can sit behind any generator. With Ollama running, choose ‘OpenAI-compatible server’ and
> a local Qwen or Llama model. The pipeline is unchanged and adds verifier-feedback regeneration.”

**Fallback:** `python scripts/run_demo.py --scene 7`.

---

## 8. VGHD results and corpus (1.5 min) · *🏅 VGHD Results* → *📖 Corpus Browser*

**Do:**

1. **🏁 Detectors.** VGHD reaches F1 **0.988** with **1.9 %** false alarms, against **0.726 / 49.1 %** for an
   answer-level NLI check and **0.325** for a citation-existence check.
2. **🧪 Ablations.** Removing the speaker check drops F1 to **0.883**. Removing the premise check raises false
   alarms to **17.0 %**.
3. **🛠️ Repair & abstain.** **97 %** of test hallucinations (203 of 209) are repaired and re-verified, or
   correctly abstained.
4. **⚖️ Lexical vs DeBERTa** (if present). The off-the-shelf DeBERTa-large-MNLI was not tuned to this test set,
   so it gives the conservative estimate.
5. **📖 Corpus Browser → Read.** Open **BG 2.23** to show the Devanagari, IAST, English and speaker.

**Honesty line:**
> “The test set consists of controlled perturbations of gold answers, and I refined the lexical verifier's
> vocabulary while looking at it. That is why the lexical numbers are optimistic and I also report DeBERTa.”

---

## 9. Command line and reproducibility (1 min, optional) · terminal

```powershell
halludetect search "weapons cannot cut the soul" --k 3
halludetect verify -q "Who says that the soul cannot be cut by weapons?" -a "Arjuna says the soul cannot be cut by weapons, as stated in BG 2.23."
halludetect ugrav-ask "Who wrote Godan?"
halludetect ugrav-ask "What is the capital of Mars?"        # → abstain
python scripts/run_demo.py --scene 8                          # headline numbers from results/*.json
```

**Say:**
> “Every number in the reports is regenerated by one command: `halludetect bench-ugrav`, `bench-gita`, or
> `scripts/run_real_experiment.py`. CI runs the tests on Python 3.10–3.12, and `scripts/e2e_check.py`
> re-runs the simulation benchmark and confirms it reproduces bit-for-bit.”

---

## 10. Close (30 s)

**Say:**
> “To summarise: uncertainty tells us *when to look*, and evidence tells us *whether the answer holds*. Fusing
> them, and being willing to abstain, lowers the hallucination rate below always retrieving, at a lower
> retrieval cost. For scripture, checking every verse, speaker, quotation and premise catches errors that
> answer-level checks miss. Next comes the full 7–8B evaluation on GPU, claim-level verification for long
> answers, and conformal guarantees on the error rate. Thank you.”

---

## Terminal-only version (no browser)

`python scripts/run_demo.py --pause` walks through 8 scenes, waiting for **Enter** between them:

| Scene | Content | Maps to |
|---|---|---|
| 1 | Godan samples → meaning clusters, semantic vs lexical entropy | §3 |
| 2 | UG-RAV corrects a high-uncertainty error with evidence (Godan) | §4a |
| 3 | UG-RAV catches a confident, SE = 0 error (crescograph), plus an honest miss (Wings of Fire) | §4b |
| 4 | Accept without retrieval (Gitanjali); abstain on an unanswerable question (Mars) | §4c |
| 5 | VGHD repairs BG 2.20 → BG 2.23 | §6 |
| 6 | Wrong speaker, misquotation, false premise, out of scope | §6 |
| 7 | Gita RAG with an injected fabricated citation → repaired → re-verified | §7 |
| 8 | Headline results read from `results/*.json` | §5, §8 |

---

## Q&A crib (likely examiner questions)

| Question | Crisp answer |
|---|---|
| *Why not just always retrieve?* | Retrieval costs latency and can mislead (external hallucination). On the simulation test split, always-retrieve leaves a 12.4 % hallucination rate. UG-RAV reaches 6.9 % while retrieving for 62 % of questions, because it verifies the evidence and can abstain. |
| *Isn't semantic entropy enough?* | No. It cannot see confident errors: on low-SE answers its AUROC is 0.51, versus 0.93 for the fused detector. Evidence support is the signal that catches them. |
| *What is the only trained component?* | A logistic regression over about 14 scalar signals. It is fitted out-of-fold, and the thresholds are tuned on the dev split only. No LLM weights are changed. |
| *Are the simulation numbers real results?* | They validate the pipeline. The simulator encodes the phenomena by construction. The real-model evidence is the PopQA pilot (Qwen2.5-1.5B, N = 60), and the full 7–8B study is the next step. |
| *How do you label correctness?* | Alias containment after normalisation. It is noisy for paraphrases such as “Christian” vs “Christianity”; the pilot records are audited manually in report 03 §6. |
| *Why is the lexical VGHD so high?* | Its vocabulary was refined on this test set, so it is optimistic. DeBERTa-large-MNLI, untouched, is the conservative estimate, shown side by side. |
| *What does VGHD add over answer-level NLI?* | Verse-level grounding. It checks that each claim is entailed *by the verse it cites*, and checks speakers, quotations and the question's premise. False alarms fall from 49 % to 1.9 %. |
| *Can it run on a real LLM?* | Yes. Any OpenAI-compatible endpoint (Ollama, vLLM) plugs into both apps and the CLI with `--llm qwen2.5:7b`, and the Kaggle notebook runs the 7–8B experiments. |
