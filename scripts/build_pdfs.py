"""Render the Markdown reports to print-quality PDFs (``reports/pdf/``).

Pipeline: Markdown → (Mermaid blocks pre-rendered to PNG by mermaid-cli) → HTML (python-markdown +
pymdown-extensions; maths typeset by MathJax) → PDF with headless Microsoft Edge or Chrome.

Usage: python scripts/build_pdfs.py [report.md ...]
Requires: pip install markdown pymdown-extensions; Node.js (npx) for mermaid-cli; Edge or Chrome; network
access for the MathJax CDN at render time.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "pdf"

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
:root { --ink:#1f2937; --muted:#6b7280; --accent:#3730a3; --saffron:#d97706; --rule:#e5e7eb; }
html { font-size: 10.4pt; }
body { font-family: "Segoe UI", "Noto Sans", "Nirmala UI", "Helvetica Neue", Arial, sans-serif; color: var(--ink);
       line-height: 1.5; max-width: 100%; margin: 0; }
h1 { color: var(--accent); font-size: 21pt; margin: 0 0 6pt; border-bottom: 3px solid var(--saffron);
     padding-bottom: 6pt; }
h2 { color: var(--accent); font-size: 14.5pt; margin-top: 18pt; border-bottom: 1px solid var(--rule);
     padding-bottom: 3pt;
     page-break-after: avoid; }
h3 { color: #312e81; font-size: 12pt; margin-top: 12pt; page-break-after: avoid; }
h4 { font-size: 10.8pt; margin-top: 10pt; page-break-after: avoid; }
p, li { text-align: justify; hyphens: auto; }
a { color: var(--accent); text-decoration: none; }
code { font-family: "Cascadia Code", Consolas, "DejaVu Sans Mono", monospace; font-size: 8.8pt; background: #f3f4f6;
       padding: 0 3px; border-radius: 3px; }
pre { background: #f8fafc; border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px; overflow-x: auto;
      font-size: 8.4pt; line-height: 1.35; page-break-inside: avoid; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 8.8pt; page-break-inside: auto; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th { background: #eef2ff; color: #1e1b4b; text-align: left; font-weight: 600; }
th, td { border: 1px solid #d1d5db; padding: 4px 6px; vertical-align: top; }
tbody tr:nth-child(even) td { background: #fafafa; }
blockquote { border-left: 4px solid var(--saffron); background: #fffbeb; margin: 8pt 0; padding: 6pt 10pt;
             color: #78350f; }
blockquote p { margin: 3pt 0; }
img { max-width: 100%; display: block; margin: 8pt auto; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid var(--rule); margin: 14pt 0; }
.footer-note { color: var(--muted); font-size: 8pt; margin-top: 18pt; border-top: 1px solid var(--rule);
               padding-top: 4pt; }
mjx-container[display="true"] { margin: 6pt 0 !important; }
"""

HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style>
<script>
window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
  options: {{ processHtmlClass: 'arithmatex' }}, svg: {{ fontCache: 'global' }} }};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head><body>
"""


def normalise_markdown(text: str) -> str:
    """Bridge GitHub-flavoured Markdown and python-markdown."""
    text = re.sub(r"(?<!\\)\\\n", "  \n", text)  # trailing backslash = hard line break on GitHub
    out = []
    for line in text.split("\n"):
        m = re.match(r"^( +)([-*+] |\d+\. )", line)
        if m:  # GitHub nests list items indented by 2-3 spaces; python-markdown needs 4
            level = (len(m.group(1)) + 1) // 3
            line = " " * (4 * level) + line[len(m.group(1)):]
        out.append(line)
    return "\n".join(out)


def prerender_mermaid(md_path: Path, text: str, tmpdir: Path) -> str:
    """Render Mermaid blocks to PNG with mermaid-cli (npx) and link them as images."""
    if "```mermaid" not in text:
        return text
    src = tmpdir / f"{md_path.stem}.src.md"
    dst = tmpdir / f"{md_path.stem}.out.md"
    src.write_text(text, encoding="utf-8")
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        print("npx not found: Mermaid diagrams are left as code blocks")
        return text
    subprocess.run([npx, "-y", "@mermaid-js/mermaid-cli", "-i", str(src), "-o", str(dst), "-e", "png", "-s", "2",
                    "-b", "white"], check=True, capture_output=True, timeout=600)
    out = dst.read_text(encoding="utf-8")
    def link(m: re.Match) -> str:
        return f"![{m.group(1)}]({(tmpdir / m.group(2)).as_uri()})"

    return re.sub(r"!\[([^\]]*)\]\(\./([^)]+\.png)\)", link, out)


def to_html(md_path: Path, tmpdir: Path) -> str:
    text = normalise_markdown(md_path.read_text(encoding="utf-8"))
    text = prerender_mermaid(md_path, text, tmpdir)
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists", "md_in_html",
                    "pymdownx.arithmatex", "pymdownx.tilde", "pymdownx.superfences"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}, "toc": {"permalink": False}},
    )
    title = re.search(r"^#\s+(.+)$", text, re.M)
    footer = ('<div class="footer-note">Generated from <code>reports/' + md_path.name +
              '</code> · https://github.com/muditagarwal2308/HallucinationDetection</div>')
    return HEAD.format(title=html.escape(title.group(1) if title else md_path.stem), css=CSS) + body + footer + \
        "</body></html>"


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists():
            return b
    found = shutil.which("msedge") or shutil.which("google-chrome") or shutil.which("chromium")
    if not found:
        raise SystemExit("No Edge/Chrome found for PDF rendering.")
    return found


def render(md_path: Path) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    html_path = md_path.with_suffix(".print.html")
    pdf_path = OUT / (md_path.stem + ".pdf")
    with tempfile.TemporaryDirectory() as profile, tempfile.TemporaryDirectory() as work:
        # written next to the .md so that relative image paths (figures/...) resolve
        html_path.write_text(to_html(md_path, Path(work)), encoding="utf-8")
        cmd = [browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
               "--run-all-compositor-stages-before-draw",
               f"--user-data-dir={profile}", "--virtual-time-budget=30000", f"--print-to-pdf={pdf_path}",
               html_path.resolve().as_uri()]
        subprocess.run(cmd, check=True, capture_output=True, timeout=240)
        html_path.unlink(missing_ok=True)
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")
    return pdf_path


def main(argv: list[str]) -> None:
    files = [Path(a) for a in argv] or sorted(p for p in REPORTS.glob("*.md") if p.name != "README.md")
    for f in files:
        render(f.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
