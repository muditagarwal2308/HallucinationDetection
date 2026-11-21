"""Extraction of verse citations such as ``BG 2.47``, ``Gita 2:47``, ``Chapter 2, verse 47``."""

from __future__ import annotations

import re
from dataclasses import dataclass

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11,
         "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17, "xviii": 18, "xix": 19, "xx": 20}

_NUM = r"(\d{1,3}|[ivxlIVXL]{1,6})"
_SEP = r"\s*[.:]\s*"
PATTERNS = [
    # BG 2.47 | B.G. 2.47 | Bhagavad Gita 2:47 | Gita 2.47-48 | BG 2.47–2.50 | Gītā 18.66
    re.compile(
        r"(?:\bB\.?\s?G\.?|\bBhagavad[\s-]*G[iī]t[aā]|\bG[iī]t[aā])\s*,?\s*(?:ch(?:apter|\.)?\s*)?"
        rf"(\d{{1,2}}){_SEP}(\d{{1,3}})(?:\s*[-–—]\s*(?:(\d{{1,2}}){_SEP})?(\d{{1,3}}))?",
        re.IGNORECASE,
    ),
    # Chapter 2, verse 47 | chapter II verses 22-25
    re.compile(rf"\bchapter\s+{_NUM}\s*,?\s*(?:verses?|v\.|śloka|shloka|sloka)\s+(\d{{1,3}})"
               r"(?:\s*(?:[-–—]|to|and)\s*(\d{1,3}))?", re.IGNORECASE),
    # verse 47 of chapter 2
    re.compile(rf"\b(?:verses?|śloka|shloka|sloka)\s+(\d{{1,3}})(?:\s*(?:[-–—]|to|and)\s*(\d{{1,3}}))?"
               rf"\s+(?:of|in)\s+(?:the\s+)?(?:chapter|adhyaya)\s+{_NUM}", re.IGNORECASE),
    # bare parenthesised reference: (2.47) or (2.47-48)
    re.compile(r"\((\d{1,2})\.(\d{1,3})(?:\s*[-–—]\s*(\d{1,3}))?\)"),
]


@dataclass(frozen=True)
class Citation:
    chapter: int
    start: int
    end: int
    raw: str
    span: tuple[int, int]

    @property
    def refs(self) -> list[tuple[int, int]]:
        if self.end < self.start or self.end - self.start > 20:
            return [(self.chapter, self.start)]
        return [(self.chapter, v) for v in range(self.start, self.end + 1)]

    @property
    def label(self) -> str:
        return f"BG {self.chapter}.{self.start}" + (f"-{self.end}" if self.end != self.start else "")


def _to_int(tok: str) -> int:
    return int(tok) if tok.isdigit() else ROMAN.get(tok.lower(), 0)


def extract_citations(text: str) -> list[Citation]:
    """Return non-overlapping citations in order of appearance."""
    found: list[Citation] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in taken)

    for idx, pat in enumerate(PATTERNS):
        for m in pat.finditer(text):
            if not free(*m.span()):
                continue
            g = m.groups()
            if idx == 0:
                ch, v1 = int(g[0]), int(g[1])
                ch2 = int(g[2]) if g[2] else ch
                v2 = int(g[3]) if g[3] else v1
                if ch2 != ch:
                    v2 = v1  # cross-chapter range: keep the first reference only
            elif idx == 1:
                ch, v1 = _to_int(g[0]), int(g[1])
                v2 = int(g[2]) if g[2] else v1
            elif idx == 2:
                v1 = int(g[0])
                v2 = int(g[1]) if g[1] else v1
                ch = _to_int(g[2])
            else:
                ch, v1 = int(g[0]), int(g[1])
                v2 = int(g[2]) if g[2] else v1
            if ch <= 0 or v1 <= 0:
                continue
            found.append(Citation(ch, v1, max(v1, v2), m.group(0), m.span()))
            taken.append(m.span())
    return sorted(found, key=lambda c: c.span[0])


# Pointer phrases that only introduce a citation ("..., as stated in BG 2.23"); they are dropped with it.
_POINTER_TAIL = re.compile(
    r"[,;]?\s*\b(?:as\s+(?:is\s+)?(?:stated|described|said|mentioned|given|taught|explained|declared|found|"
    r"expressed|noted|seen|told|written)\s+(?:in|at|by)|according\s+to|see|cf\.?|refer\s+to|in|from|at)\s*$",
    re.IGNORECASE,
)


def strip_citations(text: str) -> str:
    """Remove citations, the brackets around them and phrases that only introduce them."""
    out = text
    for c in sorted(extract_citations(text), key=lambda c: -c.span[0]):
        head = out[: c.span[0]]
        trimmed = _POINTER_TAIL.sub("", head)
        if trimmed.strip():
            head = trimmed
        out = head + out[c.span[1]:]
    out = re.sub(r"\(\s*(?:see|cf\.?)?\s*[,;]?\s*\)", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return " ".join(out.split())
