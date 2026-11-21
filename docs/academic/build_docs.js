#!/usr/bin/env node
// Builds the academic documents (dissertation, research paper, progress reports, work plan) as DOCX
// from the Markdown sources in docs/academic/src/ using docx-js.
//
//   node docs/academic/build_docs.js                 # figures + all documents
//   node docs/academic/build_docs.js --only paper    # documents whose source name contains "paper"
//   node docs/academic/build_docs.js --no-figures    # skip make_doc_figures.py
//
// Markdown subset: front matter; # / ## / ### headings ({.unnumbered}, {.notoc}, {#sec:label});
// paragraphs with **bold**, *italic*, `code`, links, $inline math$, citations [@key] and cross
// references @fig:x / @tbl:x / @eq:x / @sec:x / @alg:x; bullet and numbered lists (two levels);
// pipe tables preceded by "Table: caption {#tbl:x}"; figures ![caption](path){#fig:x width=80%};
// display equations $$ ... $$ {#eq:x}; algorithms ("Algorithm: caption" + fenced block);
// blockquotes (shaded callouts); fenced code; ```deflist and ```signature blocks; templates
// {{source:path|format}} (see facts.js); directives <!-- pagebreak -->, <!-- toc -->, <!-- lof -->,
// <!-- lot -->, <!-- mainmatter -->, <!-- appendix -->, <!-- references -->, <!-- style: Name -->,
// <!-- columns -->, <!-- if:source --> ... <!-- else --> ... <!-- endif -->.

'use strict';

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { marked } = require('marked');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, Header, Footer,
  AlignmentType, LevelFormat, ExternalHyperlink, TableOfContents, StyleLevel, HeadingLevel,
  BorderStyle, WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak, TabStopType,
  SectionType, NumberFormat, Tab, LineRuleType, HighlightColor, TableLayoutType,
} = require('docx');
const BIB = require('./bib');
const { loadSources, resolveTemplates } = require('./facts');

const HERE = __dirname;
const SRC_DIR = path.join(HERE, 'src');
const ROOT = path.resolve(HERE, '..', '..');

// ------------------------------------------------------------------------------------------------
// Layout presets
// ------------------------------------------------------------------------------------------------

const A4 = { width: 11906, height: 16838 };

function layoutFor(docType) {
  if (docType === 'paper') {
    const margin = { top: 1080, bottom: 1300, left: 850, right: 850, header: 600, footer: 600, gutter: 0 };
    const content = A4.width - margin.left - margin.right;
    const colSpace = 340;
    return {
      paper: true, margin, content, colSpace, col: Math.floor((content - colSpace) / 2),
      body: 20, line: 240, after: 0, indent: 240, table: 15, caption: 16, code: 14, ref: 16, eq: 20,
      maxImgH: 5.5,
    };
  }
  const margin = { top: 1440, bottom: 1440, left: 2160, right: 1440, header: 720, footer: 720, gutter: 0 };
  const content = A4.width - margin.left - margin.right;
  return {
    paper: false, margin, content, col: content,
    body: 24, line: 360, after: 120, indent: 0, table: 20, caption: 22, code: 18, ref: 22, eq: 24,
    maxImgH: 7.9,
  };
}

const FONT = 'Times New Roman';
const MATH_FONT = 'Cambria Math';
const CODE_FONT = 'Consolas';
const DEVANAGARI_FONT = 'Nirmala UI';

// ------------------------------------------------------------------------------------------------
// Small utilities
// ------------------------------------------------------------------------------------------------

function decodeEntities(s) {
  return s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'").replace(/&#x27;/g, "'").replace(/&nbsp;/g, '\u00a0');
}

function smartQuotes(s, prev) {
  let out = '';
  let p = prev || ' ';
  for (const ch of s) {
    if (ch === '"') out += /[\s([{\u2014\u2013/]/.test(p) ? '\u201c' : '\u201d';
    else if (ch === "'") out += /[\s([{\u2014\u2013/]/.test(p) ? '\u2018' : '\u2019';
    else out += ch;
    p = ch;
  }
  return out;
}

function roman(n) {
  const map = [[1000, 'M'], [900, 'CM'], [500, 'D'], [400, 'CD'], [100, 'C'], [90, 'XC'], [50, 'L'],
    [40, 'XL'], [10, 'X'], [9, 'IX'], [5, 'V'], [4, 'IV'], [1, 'I']];
  let r = '';
  for (const [v, s] of map) while (n >= v) { r += s; n -= v; }
  return r;
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  if (b.readUInt32BE(0) !== 0x89504e47) throw new Error(`not a PNG: ${file}`);
  return { width: b.readUInt32BE(16), height: b.readUInt32BE(20), data: b };
}

function parseAttrs(s) {
  // "{#fig:x .unnumbered width=80%}" -> { id, classes:[], kv:{} }
  const out = { id: null, classes: [], kv: {} };
  if (!s) return out;
  const body = s.trim().replace(/^\{/, '').replace(/\}$/, '');
  for (const part of body.split(/\s+/).filter(Boolean)) {
    if (part.startsWith('#')) out.id = part.slice(1);
    else if (part.startsWith('.')) out.classes.push(part.slice(1));
    else if (part.includes('=')) {
      const [k, v] = part.split('=');
      out.kv[k] = v.replace(/^"|"$/g, '');
    }
  }
  return out;
}

function splitTrailingAttrs(text) {
  const m = text.match(/^(.*?)\s*(\{[^{}]*\})\s*$/s);
  if (m && /[#.=]/.test(m[2])) return [m[1], parseAttrs(m[2])];
  return [text, parseAttrs(null)];
}

// Characters that Times New Roman lacks are set in Cambria Math (symbols) or Nirmala UI (Devanagari)
function scriptOf(ch) {
  const c = ch.codePointAt(0);
  if (c >= 0x0900 && c <= 0x097f) return 'deva';
  if (c >= 0xa8e0 && c <= 0xa8ff) return 'deva';
  if ((c >= 0x2190 && c <= 0x22ff) || (c >= 0x27c0 && c <= 0x27ef) || (c >= 0x2a00 && c <= 0x2aff) ||
      (c >= 0x1d400 && c <= 0x1d7ff) || (c >= 0x2300 && c <= 0x23ff) || c === 0x2016 || c === 0x2032 ||
      (c >= 0x2070 && c <= 0x209f) || c === 0x22a4 || c === 0x2225) return 'math';
  return 'latin';
}

// ------------------------------------------------------------------------------------------------
// Front matter and pre-processing
// ------------------------------------------------------------------------------------------------

function parseFrontMatter(text) {
  const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
  if (!m) throw new Error('missing front matter');
  const meta = {};
  for (const line of m[1].split(/\r?\n/)) {
    const mm = line.match(/^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$/);
    if (!mm) continue;
    let v = mm[2].trim();
    if (v === 'true') v = true;
    else if (v === 'false') v = false;
    meta[mm[1]] = v;
  }
  return [meta, text.slice(m[0].length)];
}

function applyConditionals(text, sources) {
  return text.replace(/<!--\s*if:(\w+)\s*-->([\s\S]*?)(?:<!--\s*else\s*-->([\s\S]*?))?<!--\s*endif\s*-->/g,
    (m, key, yes, no) => (sources[key] ? yes : (no || '')));
}

function applyIncludes(text, depth = 0) {
  if (depth > 3) return text;
  return text.replace(/<!--\s*include:\s*([^\s>]+)\s*-->/g, (m, f) =>
    applyIncludes(fs.readFileSync(path.join(SRC_DIR, f), 'utf8'), depth + 1));
}

// Protect math, citations and cross references from the Markdown lexer
function protect(text, store) {
  const parts = text.split(/(^```[\s\S]*?^```)/m);
  for (let i = 0; i < parts.length; i += 2) {
    let s = parts[i];
    s = s.replace(/\$\$([\s\S]+?)\$\$/g, (m, body) => {
      store.math.push(body.trim().replace(/\s*\n\s*/g, ' '));
      return `\u0002EQ${store.math.length - 1}\u0002`;
    });
    s = s.replace(/(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$])?)\$/g, (m, pre, body) => {
      store.math.push(body);
      return `${pre}\u0002IM${store.math.length - 1}\u0002`;
    });
    s = s.replace(/\[(@[A-Za-z0-9_:\-]+(?:\s*;\s*@[A-Za-z0-9_:\-]+)*)\]/g, (m, body) => {
      store.cites.push(body.split(';').map((k) => k.trim().replace(/^@/, '')));
      return `\u0002CI${store.cites.length - 1}\u0002`;
    });
    s = s.replace(/(^|[^\w@])@((?:fig|tbl|eq|sec|alg|ch|app):[A-Za-z0-9_\-]+)/g, (m, pre, lab) => {
      store.xrefs.push(lab);
      return `${pre}\u0002XR${store.xrefs.length - 1}\u0002`;
    });
    parts[i] = s;
  }
  return parts.join('');
}

// ------------------------------------------------------------------------------------------------
// The renderer
// ------------------------------------------------------------------------------------------------

class Renderer {
  constructor(meta, layout, store, labels, citeOrder, docName) {
    this.meta = meta;
    this.L = layout;
    this.store = store;
    this.labels = labels; // label -> {kind, num}
    this.citeOrder = citeOrder; // key -> number
    this.docName = docName;
    this.type = meta.doc_type;
    this.sections = [];
    this.children = [];
    this.sectionKind = 'title';
    this.chapter = 0;
    this.appendix = false;
    this.unnumberedChapter = false;
    this.h2 = 0;
    this.h3 = 0;
    this.counters = { fig: 0, tbl: 0, eq: 0, alg: 0 };
    this.globalCounters = { fig: 0, tbl: 0, eq: 0, alg: 0 };
    this.listInstance = 0;
    this.pendingStyle = null;
    this.pendingCaption = null;
    this.prevChar = ' ';
    this.missing = [];
  }

  // ---------------------------------------------------------------------------- numbering helpers
  chapterTag() {
    return this.appendix ? String.fromCharCode(64 + this.chapter) : String(this.chapter);
  }

  nextNumber(kind) {
    this.globalCounters[kind] += 1;
    if (this.L.paper || kind === 'alg') return String(this.globalCounters[kind]);
    this.counters[kind] += 1;
    if (this.chapter === 0 || this.unnumberedChapter) return String(this.globalCounters[kind]);
    return `${this.chapterTag()}.${this.counters[kind]}`;
  }

  refText(label) {
    const l = this.labels[label];
    if (!l) {
      this.missing.push(label);
      return '??';
    }
    const P = this.L.paper;
    switch (l.kind) {
      case 'fig': return P ? `Fig. ${l.num}` : `Figure ${l.num}`;
      case 'tbl': return P ? `Table ${roman(Number(l.num))}` : `Table ${l.num}`;
      case 'eq': return `(${l.num})`;
      case 'alg': return `Algorithm ${l.num}`;
      case 'chapter': return l.appendix ? `Appendix ${l.num}` : (P ? `Section ${roman(Number(l.num))}` : (this.type === 'plan' ? `Section ${l.num}` : `Chapter ${l.num}`));
      case 'sec': return P ? `Section ${l.num}` : `Section ${l.num}`;
      default: return l.num;
    }
  }

  citeText(keys) {
    const nums = [];
    for (const k of keys) {
      if (!BIB[k]) throw new Error(`${this.docName}: unknown citation key '${k}'`);
      if (!(k in this.citeOrder)) this.citeOrder[k] = Object.keys(this.citeOrder).length + 1;
      nums.push(this.citeOrder[k]);
    }
    const u = [...new Set(nums)].sort((a, b) => a - b);
    const groups = [];
    for (const n of u) {
      const g = groups[groups.length - 1];
      if (g && n === g[1] + 1) g[1] = n;
      else groups.push([n, n]);
    }
    return groups.map(([a, b]) => (b - a >= 2 ? `[${a}]\u2013[${b}]` : a === b ? `[${a}]` : `[${a}], [${b}]`)).join(', ');
  }

  // ---------------------------------------------------------------------------- inline rendering
  runOpts(base, extra = {}) {
    const o = { size: base.size || this.L.body, font: base.font || FONT };
    if (base.bold) o.bold = true;
    if (base.italics) o.italics = true;
    if (base.smallCaps) o.smallCaps = true;
    if (base.allCaps) o.allCaps = true;
    if (base.color) o.color = base.color;
    if (base.strike) o.strike = true;
    if (base.subScript) o.subScript = true;
    if (base.superScript) o.superScript = true;
    if (base.style) o.style = base.style;
    return Object.assign(o, extra);
  }

  plainRuns(text, base) {
    // split by script so that missing glyphs fall back to a font that has them
    const runs = [];
    let cur = '';
    let curScript = null;
    const flush = () => {
      if (!cur) return;
      const extra = {};
      if (curScript === 'deva') extra.font = DEVANAGARI_FONT;
      if (curScript === 'math') extra.font = MATH_FONT;
      runs.push(new TextRun(this.runOpts(base, Object.assign({ text: cur }, extra))));
      cur = '';
    };
    for (const ch of text) {
      const sc = scriptOf(ch);
      if (sc !== curScript) {
        flush();
        curScript = sc;
      }
      cur += ch;
    }
    flush();
    if (text) this.prevChar = text[text.length - 1];
    return runs;
  }

  mathRuns(src, base, display) {
    const runs = [];
    const font = display ? MATH_FONT : null;
    const emitText = (text, script) => {
      const re = /([A-Za-z]+|[\u03b1-\u03c9\u03d1\u03d5\u03f5]|[^A-Za-z\u03b1-\u03c9\u03d1\u03d5\u03f5]+)/g;
      let m;
      while ((m = re.exec(text))) {
        const t = m[1];
        const italic = /^[A-Za-z]$/.test(t) || /^[\u03b1-\u03c9\u03d1\u03d5\u03f5]$/.test(t);
        const o = Object.assign({}, base, { italics: base.italics || italic });
        if (script === 'sub') o.subScript = true;
        if (script === 'sup') o.superScript = true;
        if (font) {
          o.font = font;
          runs.push(new TextRun(this.runOpts(o, { text: t, font })));
        } else {
          runs.push(...this.plainRuns(t, o));
        }
      }
    };
    let i = 0;
    let buf = '';
    while (i < src.length) {
      const ch = src[i];
      if (ch === '\\' && (src[i + 1] === '_' || src[i + 1] === '^')) {
        buf += src[i + 1];
        i += 2;
        continue;
      }
      if ((ch === '_' || ch === '^') && i + 1 < src.length) {
        if (buf) emitText(buf, null);
        buf = '';
        let grp;
        if (src[i + 1] === '{') {
          const j = src.indexOf('}', i + 2);
          grp = src.slice(i + 2, j);
          i = j + 1;
        } else {
          const cp = src.codePointAt(i + 1);
          grp = String.fromCodePoint(cp);
          i += 1 + grp.length;
        }
        emitText(grp, ch === '_' ? 'sub' : 'sup');
        continue;
      }
      buf += ch;
      i += 1;
    }
    if (buf) emitText(buf, null);
    this.prevChar = 'x';
    return runs;
  }

  textWithPlaceholders(text, base) {
    const out = [];
    const parts = decodeEntities(text).split(/(\u0002(?:IM|CI|XR|EQ)\d+\u0002|\[\[PENDING:[A-Z_]+\]\])/);
    for (const part of parts) {
      if (!part) continue;
      let m;
      if ((m = part.match(/^\u0002IM(\d+)\u0002$/))) {
        out.push(...this.mathRuns(this.store.math[Number(m[1])], base, false));
      } else if ((m = part.match(/^\u0002CI(\d+)\u0002$/))) {
        out.push(...this.plainRuns(this.citeText(this.store.cites[Number(m[1])]), Object.assign({}, base, { italics: false })));
      } else if ((m = part.match(/^\u0002XR(\d+)\u0002$/))) {
        out.push(...this.plainRuns(this.refText(this.store.xrefs[Number(m[1])]), base));
      } else if (/^\[\[PENDING:/.test(part)) {
        out.push(new TextRun(this.runOpts(Object.assign({}, base, { bold: true }), { text: part, highlight: HighlightColor ? 'yellow' : undefined })));
      } else {
        out.push(...this.plainRuns(smartQuotes(part, this.prevChar), base));
      }
    }
    return out;
  }

  inline(tokens, base = {}) {
    const out = [];
    for (let idx = 0; idx < (tokens || []).length; idx += 1) {
      const t = tokens[idx];
      switch (t.type) {
        case 'text':
        case 'escape':
          if (t.tokens && t.tokens.length && t.type === 'text') out.push(...this.inline(t.tokens, base));
          else out.push(...this.textWithPlaceholders(t.text, base));
          break;
        case 'strong':
          out.push(...this.inline(t.tokens, Object.assign({}, base, { bold: true })));
          break;
        case 'em':
          out.push(...this.inline(t.tokens, Object.assign({}, base, { italics: !base.italics })));
          break;
        case 'del':
          out.push(...this.inline(t.tokens, Object.assign({}, base, { strike: true })));
          break;
        case 'codespan':
          out.push(new TextRun(this.runOpts(base, { text: decodeEntities(t.text), font: CODE_FONT, size: Math.round((base.size || this.L.body) * 0.85) })));
          this.prevChar = 'x';
          break;
        case 'br':
          out.push(new TextRun({ break: 1 }));
          break;
        case 'link': {
          const children = this.inline(t.tokens, Object.assign({}, base, { color: '1F4E79' }));
          out.push(new ExternalHyperlink({ link: t.href, children }));
          break;
        }
        case 'html': {
          const tag = t.text.trim().toLowerCase();
          if (tag === '<br>' || tag === '<br/>' || tag === '<br />') out.push(new TextRun({ break: 1 }));
          else if (tag === '<sub>') base = Object.assign({}, base, { subScript: true });
          else if (tag === '</sub>') base = Object.assign({}, base, { subScript: false });
          else if (tag === '<sup>') base = Object.assign({}, base, { superScript: true });
          else if (tag === '</sup>') base = Object.assign({}, base, { superScript: false });
          break;
        }
        case 'image':
          break;
        default:
          if (t.text) out.push(...this.textWithPlaceholders(t.text, base));
      }
    }
    return out;
  }

  inlineFromString(s, base = {}) {
    return this.inline(marked.Lexer.lexInline(s), base);
  }

  // ---------------------------------------------------------------------------- block helpers
  para(children, opts = {}) {
    const style = opts.style || this.pendingStyle || undefined;
    if (!opts.style && this.pendingStyle) this.pendingStyle = null;
    const p = {
      children,
      style,
      alignment: opts.alignment !== undefined ? opts.alignment : (style ? undefined : AlignmentType.JUSTIFIED),
    };
    for (const k of ['numbering', 'indent', 'spacing', 'keepNext', 'keepLines', 'tabStops', 'border', 'shading', 'pageBreakBefore', 'heading', 'outlineLevel']) {
      if (opts[k] !== undefined) p[k] = opts[k];
    }
    return new Paragraph(p);
  }

  push(...els) {
    this.children.push(...els);
  }

  // ---------------------------------------------------------------------------- sections
  sectionProps(kind) {
    const L = this.L;
    const page = { size: { width: A4.width, height: A4.height }, margin: L.margin };
    if (kind === 'front') page.pageNumbers = { start: 2, formatType: NumberFormat.LOWER_ROMAN };
    if (kind === 'body') page.pageNumbers = { start: 1, formatType: NumberFormat.DECIMAL };
    const props = { page };
    if (kind === 'cols') {
      props.type = SectionType.CONTINUOUS;
      props.column = { count: 2, space: L.colSpace, equalWidth: true };
    } else if (this.sections.length) {
      props.type = SectionType.NEXT_PAGE;
    }
    return props;
  }

  pageFooter(fmtRoman) {
    return new Footer({
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: this.L.paper ? 16 : 22, font: FONT })],
      })],
    });
  }

  bodyHeader() {
    const short = this.meta.short_title || this.meta.title;
    return new Header({
      children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: '7F7F7F', space: 2 } },
        children: this.plainRuns(short, { size: 18, italics: true }),
      })],
    });
  }

  emptyHeader() {
    return new Header({ children: [new Paragraph({ children: [] })] });
  }

  closeSection() {
    if (!this.children.length && !this.sections.length) return;
    const kind = this.sectionKind;
    const s = { properties: this.sectionProps(kind), children: this.children.length ? this.children : [new Paragraph({ children: [] })] };
    if (!this.L.paper) {
      if (kind === 'title') {
        s.headers = { default: this.emptyHeader() };
        s.footers = { default: new Footer({ children: [new Paragraph({ children: [] })] }) };
      } else if (kind === 'front') {
        s.headers = { default: this.emptyHeader() };
        s.footers = { default: this.pageFooter(true) };
      } else if (kind === 'body') {
        s.headers = { default: this.bodyHeader() };
        s.footers = { default: this.pageFooter(false) };
      }
    }
    this.sections.push(s);
    this.children = [];
  }

  startSection(kind) {
    this.closeSection();
    this.sectionKind = kind;
  }

  // ---------------------------------------------------------------------------- title pages
  titlePage() {
    const m = this.meta;
    if (this.L.paper) return this.paperTitle();
    const C = AlignmentType.CENTER;
    const line = (text, o = {}) => new Paragraph({
      alignment: C,
      spacing: { before: o.before || 0, after: o.after || 0, line: 276 },
      children: this.plainRuns(text, Object.assign({ size: 24 }, o)),
    });
    const els = [];
    els.push(line(m.title.toUpperCase(), { size: 32, bold: true, before: 600, after: 360 }));
    if (m.subtitle) els.push(line(m.subtitle.toUpperCase(), { size: 26, bold: true, after: 480 }));
    els.push(line(m.submission || 'A dissertation submitted in partial fulfilment of the requirements for the award of the degree of', { italics: true, after: 240 }));
    els.push(line('MASTER OF TECHNOLOGY', { size: 28, bold: true, after: 60 }));
    els.push(line('IN', { size: 24, bold: true, after: 60 }));
    els.push(line('COMPUTER SCIENCE AND ENGINEERING', { size: 28, bold: true, after: 60 }));
    els.push(line(`(SPECIALISATION: ${(m.specialisation || 'Artificial Intelligence and Robotics').toUpperCase()})`, { size: 24, bold: true, after: 900 }));
    const tabs = [{ type: TabStopType.RIGHT, position: this.L.content }];
    const two = (l, r, o = {}) => new Paragraph({
      tabStops: tabs,
      spacing: { after: 40, line: 276 },
      children: [...this.plainRuns(l, Object.assign({ size: 24 }, o)), new TextRun({ children: [new Tab()] }), ...this.plainRuns(r, Object.assign({ size: 24 }, o))],
    });
    els.push(two('SUBMITTED BY:', 'SUPERVISED BY:', { bold: true }));
    els.push(two(m.author.toUpperCase(), m.supervisor.toUpperCase(), { bold: true }));
    els.push(two(m.author_line2 || '', m.supervisor_line2 || ''));
    els.push(two(m.author_line3 || '', m.supervisor_line3 || ''));
    els.push(new Paragraph({ spacing: { after: 900 }, children: [] }));
    els.push(line(m.institution.toUpperCase(), { size: 26, bold: true, after: 60 }));
    els.push(line(m.university.toUpperCase(), { size: 26, bold: true, after: 60 }));
    els.push(line(m.city.toUpperCase(), { size: 24, bold: true, after: 360 }));
    els.push(line(m.date.toUpperCase(), { size: 24, bold: true }));
    this.push(...els);
    this.startSection('front');
  }

  paperTitle() {
    const m = this.meta;
    const C = AlignmentType.CENTER;
    this.push(new Paragraph({ alignment: C, spacing: { after: 240, line: 240 }, children: this.plainRuns(m.title, { size: 46 }) }));
    this.push(new Paragraph({ alignment: C, spacing: { after: 40 }, children: this.plainRuns(m.authors, { size: 22 }) }));
    for (const k of ['affiliation1', 'affiliation2', 'affiliation3']) {
      if (m[k]) this.push(new Paragraph({ alignment: C, spacing: { after: 0 }, children: this.plainRuns(m[k], { size: 19, italics: true }) }));
    }
    this.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
    this.startSection('cols');
  }

  // ---------------------------------------------------------------------------- headings
  heading(t) {
    let [text, attrs] = splitTrailingAttrs(t.text);
    const unnumbered = attrs.classes.includes('unnumbered');
    const notoc = attrs.classes.includes('notoc');
    const depth = t.depth;
    const P = this.L.paper;
    const base = {};
    if (depth === 1) {
      if (notoc) {
        this.unnumberedChapter = true;
        this.push(new Paragraph({ style: 'FrontTitle', pageBreakBefore: !P, children: this.inlineFromString(text.toUpperCase()) }));
        return;
      }
      this.h2 = 0;
      this.h3 = 0;
      this.counters = { fig: 0, tbl: 0, eq: 0, alg: this.counters.alg };
      if (unnumbered) {
        this.unnumberedChapter = true;
        const runs = this.inlineFromString(P ? text : text.toUpperCase(), base);
        this.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: !P && this.type !== 'plan', children: runs }));
        if (attrs.id) this.labels[attrs.id] = this.labels[attrs.id] || { kind: 'sec', num: text };
        return;
      }
      this.unnumberedChapter = false;
      this.chapter += 1;
      const tag = this.chapterTag();
      if (attrs.id) this.labels[attrs.id] = { kind: 'chapter', num: tag, appendix: this.appendix };
      if (P) {
        const runs = [new TextRun({ text: `${roman(this.chapter)}. ` }), ...this.inlineFromString(text, base)];
        this.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs }));
      } else if (this.type === 'plan') {
        const runs = [new TextRun({ text: `${tag}` }), new TextRun({ children: [new Tab()] }), ...this.inlineFromString(text, base)];
        this.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs }));
      } else {
        const label = this.appendix ? `APPENDIX ${tag}` : `CHAPTER ${tag}`;
        const runs = [new TextRun({ text: label, size: 28 }), new TextRun({ break: 1 }), ...this.inlineFromString(text.toUpperCase(), base)];
        this.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: runs }));
      }
      return;
    }
    if (depth === 2) {
      this.h2 += 1;
      this.h3 = 0;
      let num = '';
      if (!unnumbered && this.chapter > 0 && !this.unnumberedChapter) num = P ? `${String.fromCharCode(64 + this.h2)}.` : `${this.chapterTag()}.${this.h2}`;
      if (attrs.id) this.labels[attrs.id] = { kind: 'sec', num: P ? `${roman(this.chapter)}-${String.fromCharCode(64 + this.h2)}` : num };
      const runs = num ? [new TextRun({ text: num }), new TextRun({ children: [new Tab()] }), ...this.inlineFromString(text, base)] : this.inlineFromString(text, base);
      this.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs }));
      return;
    }
    if (depth === 3) {
      this.h3 += 1;
      let num = '';
      if (!unnumbered && this.chapter > 0 && !this.unnumberedChapter) num = P ? `${this.h3})` : `${this.chapterTag()}.${this.h2}.${this.h3}`;
      if (attrs.id) this.labels[attrs.id] = { kind: 'sec', num };
      const runs = num ? [new TextRun({ text: num }), new TextRun({ children: [new Tab()] }), ...this.inlineFromString(text, base)] : this.inlineFromString(text, base);
      this.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: runs }));
      return;
    }
    this.push(new Paragraph({ style: 'Heading4', children: this.inlineFromString(text, base) }));
  }

  // ---------------------------------------------------------------------------- figures, tables, equations
  figure(raw) {
    const m = raw.trim().match(/^!\[([\s\S]*?)\]\(([^)\s]+)\)\s*(\{[^}]*\})?\s*$/);
    const caption = m[1];
    const attrs = parseAttrs(m[3]);
    const file = path.resolve(SRC_DIR, m[2]);
    if (!fs.existsSync(file)) throw new Error(`${this.docName}: image not found ${file}`);
    const png = pngSize(file);
    const L = this.L;
    const pct = attrs.kv.width ? parseFloat(attrs.kv.width) / 100 : 1.0;
    let wIn = (L.col / 1440) * pct;
    let hIn = wIn * png.height / png.width;
    if (hIn > L.maxImgH) {
      wIn *= L.maxImgH / hIn;
      hIn = L.maxImgH;
    }
    const num = this.nextNumber('fig');
    if (attrs.id) this.labels[attrs.id] = { kind: 'fig', num };
    const plainCaption = caption.replace(/[*_`]/g, '').replace(/\u0002[A-Z]+\d+\u0002/g, '').trim();
    this.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: L.paper ? 80 : 120, after: 60, line: 240 },
      children: [new ImageRun({
        type: 'png',
        data: png.data,
        transformation: { width: Math.round(wIn * 96), height: Math.round(hIn * 96) },
        altText: { title: `Figure ${num}`, description: plainCaption, name: path.basename(file) },
      })],
    }));
    const label = L.paper ? `Fig. ${num}.` : `Figure ${num}:`;
    this.push(new Paragraph({
      style: 'FigureCaption',
      children: [new TextRun({ text: label, bold: !L.paper }), new TextRun({ text: ' ' }), ...this.inlineFromString(caption, { size: L.caption })],
    }));
  }

  tableCaption(text) {
    const [cap, attrs] = splitTrailingAttrs(text);
    this.pendingCaption = { cap, attrs };
  }

  table(t) {
    const L = this.L;
    const capInfo = this.pendingCaption;
    this.pendingCaption = null;
    const pct = capInfo && capInfo.attrs.kv.width ? parseFloat(capInfo.attrs.kv.width) / 100 : 1.0;
    const total = Math.floor(L.col * pct);
    if (capInfo) {
      const num = this.nextNumber('tbl');
      if (capInfo.attrs.id) this.labels[capInfo.attrs.id] = { kind: 'tbl', num };
      if (L.paper) {
        this.push(new Paragraph({ style: 'TableCaption', children: [new TextRun({ text: `TABLE ${roman(Number(num))}` }), new TextRun({ break: 1 }), ...this.inlineFromString(capInfo.cap, { smallCaps: true, size: L.caption })] }));
      } else {
        this.push(new Paragraph({ style: 'TableCaption', children: [new TextRun({ text: `Table ${num}:`, bold: true }), new TextRun({ text: ' ' }), ...this.inlineFromString(capInfo.cap, { size: L.caption })] }));
      }
    }
    const ncol = t.header.length;
    const textOf = (cell) => decodeEntities(cell.text || '').replace(/\u0002[A-Z]+\d+\u0002/g, 'xxxx').replace(/[*`]/g, '');
    const scores = [];
    for (let j = 0; j < ncol; j += 1) {
      const cells = [t.header[j], ...t.rows.map((r) => r[j])].map(textOf);
      const maxLen = Math.max(...cells.map((c) => c.length));
      const longestWord = Math.max(...cells.map((c) => Math.max(0, ...c.split(/\s+/).map((w) => w.length))));
      const avg = cells.reduce((a, c) => a + c.length, 0) / cells.length;
      scores.push(Math.max(longestWord * 1.15, Math.min(maxLen, 42) * 0.55 + Math.min(avg, 60) * 0.45, 3.5));
    }
    const sum = scores.reduce((a, b) => a + b, 0);
    let widths = scores.map((s) => Math.max(Math.round(total * s / sum), L.paper ? 420 : 700));
    const diff = total - widths.reduce((a, b) => a + b, 0);
    const wi = widths.indexOf(Math.max(...widths));
    widths[wi] += diff;
    const numericCol = [];
    for (let j = 0; j < ncol; j += 1) {
      const vals = t.rows.map((r) => textOf(r[j]).trim()).filter(Boolean);
      numericCol.push(vals.length > 0 && vals.every((v) => /^[\s\u2212\-+~\u2248<>\u2264\u2265()\d.,%\u2013\u00d7/a-zA-Z]{0,4}[\d.,]+[\s\d.,%()\u2013\-\u2212×/a-zA-Z]*$/.test(v) && /\d/.test(v) && v.length < 26));
    }
    const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' };
    const borders = { top: border, bottom: border, left: border, right: border };
    const alignOf = (j) => {
      const a = t.align[j];
      if (a === 'center') return AlignmentType.CENTER;
      if (a === 'right') return AlignmentType.RIGHT;
      if (a === 'left') return AlignmentType.LEFT;
      return numericCol[j] ? AlignmentType.CENTER : AlignmentType.LEFT;
    };
    const cellPara = (cell, j, header) => new Paragraph({
      alignment: header ? (numericCol[j] || t.align[j] === 'center' ? AlignmentType.CENTER : AlignmentType.LEFT) : alignOf(j),
      spacing: { before: 0, after: 0, line: 240 },
      children: this.inline(cell.tokens, { size: L.table, bold: header }),
    });
    const mkRow = (cells, header) => new TableRow({
      tableHeader: header,
      cantSplit: true,
      children: cells.map((cell, j) => new TableCell({
        borders,
        width: { size: widths[j], type: WidthType.DXA },
        shading: header ? { fill: 'DCE6F1', type: ShadingType.CLEAR, color: 'auto' } : undefined,
        margins: { top: 50, bottom: 50, left: L.paper ? 70 : 100, right: L.paper ? 70 : 100 },
        verticalAlign: VerticalAlign.CENTER,
        children: [cellPara(cell, j, header)],
      })),
    });
    this.push(new Table({
      width: { size: total, type: WidthType.DXA },
      columnWidths: widths,
      alignment: AlignmentType.CENTER,
      layout: TableLayoutType.FIXED,
      rows: [mkRow(t.header, true), ...t.rows.map((r) => mkRow(r, false))],
    }));
    this.push(new Paragraph({ spacing: { before: 0, after: L.paper ? 60 : 120, line: 240 }, children: [] }));
  }

  equation(idx, attrs) {
    const L = this.L;
    const num = this.nextNumber('eq');
    if (attrs.id) this.labels[attrs.id] = { kind: 'eq', num };
    const runs = this.mathRuns(this.store.math[idx], { size: L.eq }, true);
    this.push(new Paragraph({
      tabStops: [{ type: TabStopType.CENTER, position: Math.round(L.col / 2) }, { type: TabStopType.RIGHT, position: L.col }],
      spacing: { before: L.paper ? 60 : 120, after: L.paper ? 60 : 120, line: 276 },
      keepLines: true,
      children: [new TextRun({ children: [new Tab()] }), ...runs, new TextRun({ children: [new Tab()] }), new TextRun({ text: `(${num})`, size: L.paper ? 20 : 24 })],
    }));
  }

  // ---------------------------------------------------------------------------- lists, quotes, code
  list(t, level = 0, quote = false) {
    const ref = t.ordered ? 'numbers' : 'bullets';
    if (t.ordered && level === 0) this.listInstance += 1;
    const instance = t.ordered ? this.listInstance : 0;
    for (const item of t.items) {
      let first = true;
      for (const tok of item.tokens) {
        if (tok.type === 'list') {
          this.list(tok, Math.min(level + 1, 1), quote);
          continue;
        }
        if (tok.type === 'space') continue;
        const runs = this.inline(tok.tokens || [{ type: 'text', text: tok.text }], {});
        const opts = {
          spacing: { before: 0, after: this.L.paper ? 20 : 60 },
          shading: quote ? { fill: 'EEF3FA', type: ShadingType.CLEAR, color: 'auto' } : undefined,
        };
        if (first) opts.numbering = { reference: ref, level, instance };
        else opts.indent = { left: (level + 1) * (this.L.paper ? 360 : 720) };
        this.push(this.para(runs, opts));
        first = false;
      }
    }
  }

  blockquote(t) {
    for (const tok of t.tokens) {
      if (tok.type === 'paragraph') {
        this.push(this.para(this.inline(tok.tokens), {
          style: 'Callout',
          alignment: AlignmentType.JUSTIFIED,
        }));
      } else if (tok.type === 'list') {
        this.list(tok, 0, true);
      }
    }
  }

  codeBlock(t) {
    const L = this.L;
    const lines = t.text.replace(/\t/g, '    ').split('\n');
    if (t.lang === 'deflist') return this.deflist(lines);
    if (t.lang === 'signature') return this.signature(lines);
    const alg = this.pendingCaption && this.pendingCaption.alg ? this.pendingCaption : null;
    this.pendingCaption = null;
    const rule = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: '000000', space: 2 });
    if (alg) {
      const num = this.nextNumber('alg');
      if (alg.attrs.id) this.labels[alg.attrs.id] = { kind: 'alg', num };
      this.push(new Paragraph({
        keepNext: true,
        spacing: { before: 160, after: 40, line: 240 },
        border: { top: rule(12), bottom: rule(6) },
        children: [new TextRun({ text: `Algorithm ${num}:`, bold: true, size: L.body - 2 }), new TextRun({ text: ' ', size: L.body - 2 }), ...this.inlineFromString(alg.cap, { size: L.body - 2 })],
      }));
    }
    lines.forEach((ln, i) => {
      const last = i === lines.length - 1;
      const p = {
        keepNext: !last,
        keepLines: true,
        spacing: { before: 0, after: last ? (alg ? 200 : 160) : 0, line: 240 },
        children: [new TextRun({ text: ln.length ? ln : ' ', font: CODE_FONT, size: L.code })],
      };
      if (alg) {
        if (last) p.border = { bottom: rule(12) };
      } else {
        p.shading = { fill: 'F3F4F6', type: ShadingType.CLEAR, color: 'auto' };
        p.indent = { left: 120, right: 120 };
      }
      this.push(new Paragraph(p));
    });
  }

  deflist(lines) {
    const tab = this.L.paper ? 1100 : 1900;
    for (const ln of lines.filter((l) => l.trim())) {
      const [term, ...rest] = ln.split('|');
      this.push(new Paragraph({
        tabStops: [{ type: TabStopType.LEFT, position: tab }],
        indent: { left: tab, hanging: tab },
        spacing: { before: 0, after: 60, line: 300 },
        children: [...this.inlineFromString(term.trim(), { bold: true }), new TextRun({ children: [new Tab()] }), ...this.inlineFromString(rest.join('|').trim())],
      }));
    }
  }

  signature(lines) {
    const tabs = [{ type: TabStopType.RIGHT, position: this.L.content }];
    this.push(new Paragraph({ spacing: { before: 720, after: 0 }, tabStops: tabs, children: [new TextRun({ text: '______________________________' }), new TextRun({ children: [new Tab()] }), new TextRun({ text: '______________________________' })] }));
    for (const ln of lines.filter((l) => l.trim())) {
      const [l, r] = ln.split('|').map((x) => (x || '').trim());
      this.push(new Paragraph({ spacing: { before: 0, after: 0, line: 276 }, tabStops: tabs, children: [...this.inlineFromString(l), new TextRun({ children: [new Tab()] }), ...this.inlineFromString(r)] }));
    }
    this.push(new Paragraph({ spacing: { after: 240 }, children: [] }));
  }

  references() {
    const keys = Object.keys(this.citeOrder).sort((a, b) => this.citeOrder[a] - this.citeOrder[b]);
    const L = this.L;
    const hang = L.paper ? 400 : 640;
    for (const k of keys) {
      this.push(new Paragraph({
        style: 'Reference',
        tabStops: [{ type: TabStopType.LEFT, position: hang }],
        indent: { left: hang, hanging: hang },
        children: [new TextRun({ text: `[${this.citeOrder[k]}]` }), new TextRun({ children: [new Tab()] }), ...this.inlineFromString(BIB[k], { size: L.ref })],
      }));
    }
  }

  tocBlock(kind) {
    const L = this.L;
    if (kind === 'toc') {
      this.push(new TableOfContents('Table of Contents', { hyperlink: true, headingStyleRange: '1-3' }));
    } else {
      const style = kind === 'lof' ? 'FigureCaption' : 'TableCaption';
      this.push(new TableOfContents(kind === 'lof' ? 'List of Figures' : 'List of Tables', { hyperlink: true, stylesWithLevels: [new StyleLevel(style, 4)] }));
    }
    void L;
  }

  // ---------------------------------------------------------------------------- directives
  directive(text) {
    const d = text.replace(/^<!--\s*/, '').replace(/\s*-->\s*$/, '').trim();
    if (d === 'pagebreak') return this.push(new Paragraph({ children: [new PageBreak()] }));
    if (d === 'toc' || d === 'lof' || d === 'lot') return this.tocBlock(d);
    if (d === 'mainmatter') {
      this.startSection('body');
      this.chapter = 0;
      return null;
    }
    if (d === 'appendix') {
      this.appendix = true;
      this.chapter = 0;
      return null;
    }
    if (d === 'references') return this.references();
    if (d === 'columns') return this.startSection('cols');
    let m;
    if ((m = d.match(/^style:\s*(\w+)$/))) {
      this.pendingStyle = m[1];
      return null;
    }
    if ((m = d.match(/^vspace:\s*(\d+)$/))) return this.push(new Paragraph({ spacing: { before: Number(m[1]), after: 0 }, children: [] }));
    return null;
  }

  // ---------------------------------------------------------------------------- main loop
  blocks(tokens) {
    for (const t of tokens) {
      switch (t.type) {
        case 'heading':
          this.heading(t);
          break;
        case 'paragraph': {
          const raw = t.raw.trim();
          let m;
          if (/^!\[[\s\S]*\]\([^)]+\)\s*(\{[^}]*\})?$/.test(raw)) {
            this.figure(raw);
          } else if ((m = raw.match(/^\u0002EQ(\d+)\u0002\s*(\{[^}]*\})?$/))) {
            this.equation(Number(m[1]), parseAttrs(m[2]));
          } else if ((m = raw.match(/^Table:\s*([\s\S]+)$/))) {
            this.tableCaption(m[1].replace(/\s*\n\s*/g, ' '));
          } else if ((m = raw.match(/^Algorithm:\s*([\s\S]+)$/))) {
            const [cap, attrs] = splitTrailingAttrs(m[1].replace(/\s*\n\s*/g, ' '));
            this.pendingCaption = { cap, attrs, alg: true };
          } else {
            const style = this.pendingStyle;
            this.pendingStyle = null;
            const opts = { spacing: undefined };
            if (style) opts.style = style;
            else if (this.L.paper) opts.indent = { firstLine: this.L.indent };
            this.push(this.para(this.inline(t.tokens), opts));
          }
          break;
        }
        case 'list':
          this.list(t);
          break;
        case 'table':
          this.table(t);
          break;
        case 'blockquote':
          this.blockquote(t);
          break;
        case 'code':
          this.codeBlock(t);
          break;
        case 'html':
          if (/^<!--/.test(t.text.trim())) this.directive(t.text.trim());
          break;
        case 'hr':
          this.push(new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '999999', space: 1 } }, children: [] }));
          break;
        default:
          break;
      }
    }
  }

  render(tokens) {
    this.titlePage();
    this.blocks(tokens);
    this.closeSection();
    return this.sections;
  }
}

// ------------------------------------------------------------------------------------------------
// Styles and numbering
// ------------------------------------------------------------------------------------------------

function stylesFor(L, docType) {
  const P = L.paper;
  const body = { font: FONT, size: L.body };
  const para = (id, name, run, paragraph, extra = {}) => Object.assign({ id, name, basedOn: 'Normal', next: 'Normal', quickFormat: true, run, paragraph }, extra);
  const heading1 = P
    ? para('Heading1', 'Heading 1', { font: FONT, size: 20, smallCaps: true, bold: false }, { alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100, line: 240 }, keepNext: true, keepLines: true, outlineLevel: 0 })
    : docType === 'plan'
      ? para('Heading1', 'Heading 1', { font: FONT, size: 30, bold: true }, { spacing: { before: 360, after: 180, line: 300 }, keepNext: true, keepLines: true, outlineLevel: 0, indent: { left: 567, hanging: 567 }, tabStops: [{ type: TabStopType.LEFT, position: 567 }] })
      : para('Heading1', 'Heading 1', { font: FONT, size: 32, bold: true }, { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 480, line: 360 }, keepNext: true, keepLines: true, outlineLevel: 0 });
  const heading2 = P
    ? para('Heading2', 'Heading 2', { font: FONT, size: 20, italics: true }, { spacing: { before: 120, after: 60, line: 240 }, keepNext: true, keepLines: true, outlineLevel: 1, indent: { left: 280, hanging: 280 }, tabStops: [{ type: TabStopType.LEFT, position: 280 }] })
    : para('Heading2', 'Heading 2', { font: FONT, size: 28, bold: true }, { spacing: { before: 300, after: 120, line: 300 }, keepNext: true, keepLines: true, outlineLevel: 1, indent: { left: 720, hanging: 720 }, tabStops: [{ type: TabStopType.LEFT, position: 720 }] });
  const heading3 = P
    ? para('Heading3', 'Heading 3', { font: FONT, size: 20, italics: true }, { spacing: { before: 80, after: 40, line: 240 }, keepNext: true, keepLines: true, outlineLevel: 2, indent: { left: 400, hanging: 400 }, tabStops: [{ type: TabStopType.LEFT, position: 400 }] })
    : para('Heading3', 'Heading 3', { font: FONT, size: 24, bold: true }, { spacing: { before: 240, after: 100, line: 300 }, keepNext: true, keepLines: true, outlineLevel: 2, indent: { left: 900, hanging: 900 }, tabStops: [{ type: TabStopType.LEFT, position: 900 }] });
  const tocEntry = (id, name, size, left, bold, caps) => para(id, name, { font: FONT, size, bold, allCaps: caps }, { spacing: { before: bold ? 120 : 20, after: 20, line: 276 }, indent: { left, right: 500 }, tabStops: [{ type: TabStopType.RIGHT, position: L.content, leader: 'dot' }] }, { basedOn: 'Normal', next: 'Normal', quickFormat: false });
  return {
    default: { document: { run: { font: { ascii: FONT, hAnsi: FONT, cs: DEVANAGARI_FONT, eastAsia: FONT }, size: L.body } } },
    characterStyles: [
      { id: 'Hyperlink', name: 'Hyperlink', basedOn: 'DefaultParagraphFont', run: { color: '1F4E79', underline: { type: 'single' } } },
    ],
    paragraphStyles: [
      { id: 'Normal', name: 'Normal', run: body, paragraph: { spacing: { before: 0, after: L.after, line: L.line, lineRule: LineRuleType.AUTO }, alignment: AlignmentType.JUSTIFIED } },
      heading1, heading2, heading3,
      para('Heading4', 'Heading 4', { font: FONT, size: L.body, bold: true, italics: true }, { spacing: { before: 160, after: 60 }, keepNext: true }),
      para('FrontTitle', 'Front Title', { font: FONT, size: 32, bold: true }, { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 480, line: 360 }, keepNext: true }),
      para('FigureCaption', 'FigureCaption', { font: FONT, size: L.caption }, { alignment: P ? AlignmentType.JUSTIFIED : AlignmentType.CENTER, spacing: { before: 60, after: P ? 160 : 240, line: P ? 220 : 276 } }),
      para('TableCaption', 'TableCaption', { font: FONT, size: L.caption }, { alignment: AlignmentType.CENTER, spacing: { before: P ? 120 : 200, after: P ? 60 : 100, line: P ? 220 : 276 }, keepNext: true, keepLines: true }),
      para('Callout', 'Callout', { font: FONT, size: P ? 18 : 22 }, { spacing: { before: 60, after: 60, line: P ? 230 : 300 }, indent: { left: 200, right: 200 }, shading: { fill: 'EEF3FA', type: ShadingType.CLEAR, color: 'auto' }, border: { left: { style: BorderStyle.SINGLE, size: 18, color: '2F5F98', space: 6 } } }),
      para('Reference', 'Reference', { font: FONT, size: L.ref }, { alignment: AlignmentType.LEFT, spacing: { before: 0, after: P ? 30 : 100, line: P ? 220 : 276 } }),
      para('Abstract', 'Abstract', { font: FONT, size: 18, bold: true }, { alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 100, line: 230 } }),
      para('IndexTerms', 'Index Terms', { font: FONT, size: 18, bold: true }, { alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 160, line: 230 } }),
      para('Centered', 'Centered', body, { alignment: AlignmentType.CENTER }),
      tocEntry('TOC1', 'toc 1', P ? 18 : 22, 0, true, true),
      tocEntry('TOC2', 'toc 2', P ? 18 : 22, 360, false, false),
      tocEntry('TOC3', 'toc 3', P ? 18 : 22, 720, false, false),
      tocEntry('TOC4', 'toc 4', P ? 18 : 22, 0, false, false),
    ],
  };
}

function numberingFor(L) {
  const step = L.paper ? 300 : 720;
  const hang = L.paper ? 240 : 360;
  return {
    config: [
      {
        reference: 'bullets',
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: '\u2022', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: step, hanging: hang } } } },
          { level: 1, format: LevelFormat.BULLET, text: '\u2013', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 2 * step, hanging: hang } } } },
        ],
      },
      {
        reference: 'numbers',
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: step, hanging: hang } } } },
          { level: 1, format: LevelFormat.LOWER_LETTER, text: '(%2)', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 2 * step, hanging: hang + 60 } } } },
        ],
      },
    ],
  };
}

// ------------------------------------------------------------------------------------------------
// Build one document
// ------------------------------------------------------------------------------------------------

function renderOnce(meta, body, sources, labels, citeOrder, docName) {
  const store = { math: [], cites: [], xrefs: [] };
  const protectedText = protect(body, store);
  const tokens = marked.lexer(protectedText, { gfm: true });
  const r = new Renderer(meta, layoutFor(meta.doc_type), store, labels, citeOrder, docName);
  const sections = r.render(tokens);
  return { sections, renderer: r };
}

async function buildDoc(file, sources) {
  const docName = path.basename(file);
  const text = fs.readFileSync(file, 'utf8');
  const [meta, rawBody] = parseFrontMatter(text);
  let body = applyIncludes(rawBody);
  body = applyConditionals(body, sources);
  body = resolveTemplates(body, sources, docName);
  for (const k of Object.keys(meta)) if (typeof meta[k] === 'string') meta[k] = resolveTemplates(meta[k], sources, docName);
  // pass 1 collects labels and the citation order; pass 2 renders with them
  const labels = {};
  const citeOrder = {};
  renderOnce(meta, body, sources, labels, citeOrder, docName);
  const { sections, renderer } = renderOnce(meta, body, sources, labels, citeOrder, docName);
  if (renderer.missing.length) throw new Error(`${docName}: unresolved cross references: ${[...new Set(renderer.missing)].join(', ')}`);
  const L = layoutFor(meta.doc_type);
  const doc = new Document({
    creator: meta.author || 'Manju Pal',
    title: meta.title,
    subject: meta.subtitle || meta.doc_type,
    description: `${meta.subtitle || ''} \u2014 built from docs/academic/src/${docName}`,
    styles: stylesFor(L, meta.doc_type),
    numbering: numberingFor(L),
    sections,
  });
  const out = path.join(HERE, meta.output);
  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(out, buf);
  const pending = (body.match(/\[\[PENDING:[A-Z_]+\]\]/g) || []);
  return { out, cites: Object.keys(citeOrder).length, labels: Object.keys(labels).length, pending };
}

function findPython() {
  const candidates = [process.env.PYTHON, path.resolve(ROOT, '..', '.venv', 'Scripts', 'python.exe'), path.resolve(ROOT, '.venv', 'Scripts', 'python.exe'), path.resolve(ROOT, '.venv', 'bin', 'python'), 'python'];
  for (const c of candidates) {
    if (!c) continue;
    if (c === 'python' || fs.existsSync(c)) return c;
  }
  return 'python';
}

async function main() {
  const args = process.argv.slice(2);
  const only = args.includes('--only') ? args[args.indexOf('--only') + 1] : null;
  if (!args.includes('--no-figures')) {
    const py = findPython();
    console.log(`figures: ${py} make_doc_figures.py`);
    execFileSync(py, [path.join(HERE, 'make_doc_figures.py')], { stdio: ['ignore', 'ignore', 'inherit'], env: Object.assign({}, process.env, { PYTHONIOENCODING: 'utf-8' }) });
  }
  const sources = loadSources();
  console.log(`sources: ${Object.entries(sources).filter(([k]) => k !== 'calc').map(([k, v]) => `${k}=${v ? 'yes' : 'missing'}`).join(', ')}`);
  const files = fs.readdirSync(SRC_DIR).filter((f) => f.endsWith('.md') && !f.startsWith('_')).sort()
    .filter((f) => !only || f.includes(only));
  for (const f of files) {
    const res = await buildDoc(path.join(SRC_DIR, f), sources);
    console.log(`built ${path.relative(ROOT, res.out)}  (${res.cites} references, ${res.labels} labels${res.pending.length ? `, PENDING: ${[...new Set(res.pending)].join(' ')}` : ''})`);
  }
}

main().catch((e) => {
  console.error(e.stack || e.message);
  process.exit(1);
});
