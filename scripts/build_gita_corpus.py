"""Build the verse-aligned Bhagavad Gita corpus used by the VGHD framework.

Source
------
Annie Besant, *The Bhagavad-Gita, or The Lord's Song* (4th ed., G. A. Natesan,
Madras, 1922). The translation is in the public domain (published before 1931,
translator died 1933). The transcription is taken from English Wikisource:
https://en.wikisource.org/wiki/Bhagavad-Gita_(Besant_4th)

The Wikisource edition prints the Devanagari verse followed by the English
rendering. Every Devanagari verse ends with its number (e.g. ``॥ २३ ॥``), whereas
the English verse numbers in the transcription contain a handful of OCR slips
(``(42`` instead of ``(42)``, duplicated ``(20)``, ...). We therefore anchor every
English paragraph to the number of the Devanagari verse that precedes it, which
yields a complete, gap-free alignment (701 verses; see ``EDITION NOTE`` below).

Output: ``data/gita/verses.json`` and ``data/gita/chapters.json``.

EDITION NOTE
------------
Besant's edition numbers Arjuna's opening question in chapter 13
("Matter and Spirit ... these I fain would learn, O Keshava") as 13.1, giving 35
verses in chapter 13 and 701 verses in total. Many popular editions leave that
verse unnumbered, giving the canonical count of 700 (34 verses in chapter 13).
The corpus keeps Besant's numbering and records the 700-verse equivalent in the
``ref_700`` field so that citations can be checked against either convention.

Usage
-----
    python scripts/build_gita_corpus.py                 # fetch from Wikisource (rate-limited)
    python scripts/build_gita_corpus.py --cache data/raw/besant_wikisource
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

API = (
    "https://en.wikisource.org/w/api.php?action=parse&page=Bhagavad-Gita_(Besant_4th)/"
    "Discourse_{ch}&prop=text&format=json&formatversion=2"
)
USER_AGENT = "HallucinationDetection-research/1.0 (https://github.com/muditagarwal2308/HallucinationDetection)"
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
EXPECTED = {1: 47, 2: 72, 3: 43, 4: 42, 5: 29, 6: 47, 7: 30, 8: 28, 9: 34, 10: 42,
            11: 55, 12: 20, 13: 35, 14: 27, 15: 20, 16: 24, 17: 28, 18: 78}
CHAPTER_NAMES = {
    1: "The Yoga of the Despondency of Arjuna",
    2: "The Yoga of the Sankhya",
    3: "The Yoga of Action",
    4: "The Yoga of Wisdom",
    5: "The Yoga of Renunciation of Action",
    6: "The Yoga of Self-Subdual",
    7: "The Yoga of Discrimination",
    8: "The Yoga of the Indestructible Supreme Eternal",
    9: "The Yoga of the Kingly Science and the Kingly Secret",
    10: "The Yoga of Sovereignty",
    11: "The Yoga of the Vision of the Universal Form",
    12: "The Yoga of Devotion",
    13: "The Yoga of Distinction between the Field and the Knower of the Field",
    14: "The Yoga of Separation from the Three Qualities",
    15: "The Yoga of Attaining the Supreme Spirit",
    16: "The Yoga of Division between the Divine and the Demoniacal Properties",
    17: "The Yoga of Division of the Threefold Faith",
    18: "The Yoga of Liberation by Renunciation",
}


def fetch(ch: int, cache: Path | None) -> str:
    if cache is not None and (cache / f"d{ch}.json").exists():
        payload = json.loads((cache / f"d{ch}.json").read_text(encoding="utf-8"))
    else:
        import requests

        resp = requests.get(API.format(ch=ch), headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            (cache / f"d{ch}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        time.sleep(2.0)  # be polite to the Wikimedia API
    return payload["parse"]["text"]


def html_to_lines(raw: str) -> list[str]:
    text = re.sub(r"<style.*?</style>", "", raw, flags=re.S)
    text = re.sub(r'<sup[^>]*class="reference"[^>]*>.*?</sup>', "", text, flags=re.S)
    text = re.sub(r'<ol class="references">.*?</ol>', "", text, flags=re.S)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</p>|</div>|</li>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\u200b", "").replace("\xa0", " ")
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def clean_english(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s*\(?\s*\d+\s*\)?\s*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_sanskrit(text: str) -> str:
    text = re.sub(r"॥\s*[०-९]+\s*॥", "", text)
    text = re.sub(r"^ॐ\s*।?\s*", "", text.strip())
    text = re.sub(r"।(\s*।)+", "।", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip("।").strip()


def to_iast(devanagari: str) -> str:
    from indic_transliteration import sanscript

    return sanscript.transliterate(devanagari, sanscript.DEVANAGARI, sanscript.IAST)


SA_SPEAKERS = {"श्रीभगवान": "Krishna", "भगवान": "Krishna", "अर्जुन": "Arjuna", "सञ्जय": "Sanjaya",
               "संजय": "Sanjaya", "धृतराष्ट्र": "Dhritarashtra"}


def sanskrit_speaker(line: str) -> str | None:
    """Map a Devanagari ``... उवाच`` line to a canonical speaker."""
    if not re.search(r"उवाच|ुवाच", line) or len(line) > 40:
        return None
    for key, name in SA_SPEAKERS.items():
        if key in line:
            return name
    return None


def parse_chapter(ch: int, lines: list[str]) -> dict[int, dict]:
    speaker: str | None = None
    current: int | None = None
    pending_sa: list[str] = []
    verses: dict[int, dict] = {}
    for line in lines:
        if re.search(r"[\u0900-\u097F]", line):
            spk = sanskrit_speaker(line)
            if spk:
                speaker = spk
                continue
            pending_sa.append(line)
            m = re.search(r"॥\s*([०-९]+)\s*॥", line)
            if m:
                current = int(m.group(1).translate(DEV_DIGITS))
                verses.setdefault(current, {"speaker": speaker, "en": [], "sa": ""})
                verses[current]["sa"] = clean_sanskrit(" ".join(pending_sa))
                pending_sa = []
            continue
        if line.startswith("Thus in the glorious"):
            current = None
            continue
        m = re.match(r"^(.{3,30}) said:?$", line)
        if m:
            continue  # the Sanskrit "uvāca" line already fixed the speaker
        if current is None:
            continue
        english = clean_english(line)
        if english:
            verses[current]["en"].append(english)
    return verses


def build(cache: Path | None, out_dir: Path) -> None:
    corpus = []
    for ch in range(1, 19):
        parsed = parse_chapter(ch, html_to_lines(fetch(ch, cache)))
        missing = sorted(set(range(1, EXPECTED[ch] + 1)) - set(parsed))
        if missing or len(parsed) != EXPECTED[ch]:
            raise SystemExit(f"Chapter {ch}: alignment incomplete, missing {missing}")
        for v in sorted(parsed):
            rec = parsed[v]
            if ch == 13:
                ref_700 = None if v == 1 else f"13.{v - 1}"
            else:
                ref_700 = f"{ch}.{v}"
            corpus.append({
                "ref": f"{ch}.{v}",
                "chapter": ch,
                "verse": v,
                "ref_700": ref_700,
                "speaker": rec["speaker"] or "Sanjaya",
                "sanskrit": rec["sa"],
                "iast": to_iast(rec["sa"]),
                "english": " ".join(rec["en"]),
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verses.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    chapters = [{"chapter": ch, "name": CHAPTER_NAMES[ch], "verses": EXPECTED[ch],
                 "verses_700": EXPECTED[ch] - (1 if ch == 13 else 0)} for ch in range(1, 19)]
    (out_dir / "chapters.json").write_text(json.dumps(chapters, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(corpus)} verses to {out_dir / 'verses.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data/raw/besant_wikisource"))
    ap.add_argument("--out", type=Path, default=Path("data/gita"))
    args = ap.parse_args()
    build(args.cache, args.out)
