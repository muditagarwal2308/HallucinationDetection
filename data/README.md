# Data

| Path | Content | Source and licence |
|---|---|---|
| `gita/verses.json` | 701 verses of the Bhagavad Gita: Devanagari, IAST transliteration, English translation, speaker, Besant and 700-verse numbering | English: Annie Besant, *The Bhagavad-Gita, or The Lord's Song*, 4th ed., G. A. Natesan, Madras, 1922 — **public domain** (published before 1931; translator died 1933). Sanskrit text public domain. Transcription: [English Wikisource](https://en.wikisource.org/wiki/Bhagavad-Gita_(Besant_4th)). IAST generated with `indic-transliteration`. Built by `scripts/build_gita_corpus.py`. |
| `gita/chapters.json` | Chapter names (Besant's titles) and verse counts in both numberings | Same as above |
| `gita/qa.jsonl` | 106 questions: 70 English answerable (concept, verse lookup, quotation, speaker), 8 Hindi, 12 false-premise, 6 non-existent-reference, 10 out-of-scope; gold verses and reference answers | Written for this project (MIT). Built by `scripts/build_gita_datasets.py`. |
| `gita/hallucination_testset.jsonl` | 524 labelled answers: 78 correct + 78 each of fabricated citation, misattributed citation, misquotation, wrong speaker and unsupported claim; 28 correct refusals; 18 accepted false premises; 10 answered out-of-scope questions | Generated from `qa.jsonl` by controlled perturbation (`halludetect.gita.perturb`, seed 2026). MIT. |
| `odqa/world.jsonl` | 197 factual questions (Indian and world literature, monuments, capitals, science, people, history, geography; 8 ambiguous) with aliases, popularity tier, answer type, common misconception, distractors | Written for this project (MIT). Built by `scripts/build_odqa_world.py`. Popularity is a judgement-based tier, not measured page views. |
| `odqa/passages.jsonl` | 198 evidence passages; 20 gold passages are deliberately withheld to simulate retrieval failure | Written for this project (MIT). |
| `real/wikipedia_passages.json` | Cache of the Wikipedia search results and lead-section excerpts retrieved during the real-model experiment | English Wikipedia, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); article titles are kept for attribution. |

PopQA (Mallen et al., ACL 2023) is downloaded at run time from the Hugging Face Hub
(`akariasai/PopQA`) into `data/raw/` (git-ignored) and is not redistributed.

## Verse numbering

Besant numbers Arjuna's opening question of chapter 13 ("Matter and Spirit, even the Field
and the Knower of the Field …") as 13.1, so his edition has 35 verses in chapter 13 and 701
in total. Many popular editions leave that verse unnumbered (34 verses in chapter 13, 700 in
total). Every verse therefore carries both `ref` (Besant) and `ref_700`, and the citation
checker accepts a reference if it exists under either convention.

## Rebuilding

```bash
python scripts/build_gita_corpus.py          # needs network on first run (Wikisource API, rate-limited)
python scripts/build_gita_datasets.py
python scripts/build_odqa_world.py
```
