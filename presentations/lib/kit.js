// Shared theme ("Indigo & Saffron") and slide components for the four milestone decks.
// Palette: deep indigo dominates (title/closing/divider slides, footer band, headings),
// saffron is the single sharp accent (proposed method, current section, highlights),
// teal / slate / rose are reserved for data (evidence, baselines, errors).
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const icons = require("./icons");

const W = 10;
const H = 5.625;

const C = {
  ink: "1B1745",
  indigo: "312E81",
  indigoMid: "4F46E5",
  indigoSoft: "E0E7FF",
  indigoPale: "F1F0FA",
  saffron: "F08C1A",
  saffronDark: "B45309",
  saffronSoft: "FDEBD3",
  teal: "0F8B8D",
  tealDark: "0B6466",
  tealSoft: "D5F0EF",
  rose: "BE123C",
  roseSoft: "FCE7EC",
  green: "15803D",
  greenSoft: "DCFCE7",
  slate: "64748B",
  slateLight: "CBD5E1",
  slatePale: "EEF1F5",
  text: "1F2937",
  muted: "4B5563",
  bg: "FAF7F2",
  white: "FFFFFF",
  rule: "E6E0D4",
  dimText: "C7D2FE",
};
const F = { head: "Georgia", body: "Calibri", mono: "Consolas" };

// Colours the icons are pre-rendered in.
const ICON_COLORS = [C.white, C.ink, C.indigo, C.saffron, C.saffronDark, C.teal, C.rose, C.slate, C.green];
const ICON_NAMES = (
  "FaBrain FaSearch FaBalanceScale FaCheckCircle FaTimesCircle FaQuestionCircle FaBookOpen FaQuoteLeft " +
  "FaUserCheck FaExclamationTriangle FaDatabase FaCogs FaChartLine FaChartBar FaLayerGroup FaShieldAlt FaFlask " +
  "FaLanguage FaGraduationCap FaUniversity FaLightbulb FaRoute FaCode FaTerminal FaDesktop FaHourglassHalf " +
  "FaFileAlt FaGlobe FaCubes FaMicrochip FaRandom FaFilter FaPenNib FaWrench FaBan FaLink FaListOl FaGavel " +
  "FaDice FaProjectDiagram FaClipboardCheck FaLaptopCode FaServer FaBullseye FaSitemap FaTools FaEye " +
  "FaHandPaper FaRedo FaFeatherAlt FaUsers FaCheck FaArrowRight FaPaperPlane FaBook FaUserTie FaCalendarAlt " +
  "FaMapSigns FaPlayCircle FaTasks FaBug FaSlidersH FaInfoCircle FaUserGraduate FaComments FaGithub FaPython " +
  "FaStream FaExchangeAlt FaMagic FaSyncAlt FaTag FaKey FaVial FaHistory FaListUl FaTable FaTrophy FaFlagCheckered"
).split(" ");

async function preloadIcons() {
  await icons.preload(ICON_NAMES, ICON_COLORS);
}

const shadow = () => ({ type: "outer", color: "000000", blur: 6, offset: 1.5, angle: 90, opacity: 0.1 });

// ------------------------------------------------------------------ text
// Minimal markup: **bold**, _{subscript}, ^{superscript}, and "\n" for a line break.
function runs(text, base = {}) {
  const s = String(text);
  const out = [];
  let bold = false;
  let buf = "";
  const push = (t, extra = {}) => {
    if (t) out.push({ text: t, options: { ...base, ...(bold ? { bold: true } : {}), ...extra } });
  };
  for (let i = 0; i < s.length; ) {
    if (s.startsWith("**", i)) {
      push(buf); buf = ""; bold = !bold; i += 2; continue;
    }
    if (s.startsWith("_{", i) || s.startsWith("^{", i)) {
      const j = s.indexOf("}", i);
      push(buf); buf = "";
      push(s.slice(i + 2, j), s[i] === "_" ? { subscript: true } : { superscript: true });
      i = j + 1; continue;
    }
    if (s[i] === "\n") {
      push(buf); buf = "";
      if (out.length) out[out.length - 1].options.breakLine = true;
      else out.push({ text: "", options: { ...base, breakLine: true } });
      i += 1; continue;
    }
    buf += s[i]; i += 1;
  }
  push(buf);
  return out.length ? out : [{ text: "", options: { ...base } }];
}

function txt(slide, text, o = {}) {
  const opts = {
    fontFace: F.body, fontSize: 12, color: C.text, margin: 0, valign: "top",
    ...o,
  };
  const content = typeof text === "string" ? runs(text, {}) : text;
  slide.addText(content, opts);
}

// items: strings (with **bold**) or { t, sub: true, color }
function bullets(slide, items, o = {}) {
  const fontSize = o.fontSize || 12;
  const arr = [];
  items.forEach((it, i) => {
    const item = typeof it === "string" ? { t: it } : it;
    const last = i === items.length - 1;
    const para = {
      bullet: item.noBullet ? false : { indent: o.indent || 13 },
      indentLevel: item.sub ? 1 : 0,
      paraSpaceAfter: o.gap === undefined ? 5 : o.gap,
    };
    const rr = runs(item.t, { color: item.color || o.color || C.text, fontSize: item.sub ? fontSize - 1 : fontSize });
    rr[0].options = { ...rr[0].options, ...para };
    rr[rr.length - 1].options.breakLine = !last;
    arr.push(...rr);
  });
  slide.addText(arr, {
    fontFace: F.body, margin: 0, valign: "top", ...o, fontSize,
  });
}

// ------------------------------------------------------------------ shapes
function rect(slide, pres, x, y, w, h, fill, extra = {}) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, ...extra });
}

function card(slide, pres, { x, y, w, h, accent = C.saffron, fill = C.white, border = C.rule, top = false, noShadow = false }) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: fill }, line: { color: border, width: 0.75 },
    ...(noShadow ? {} : { shadow: shadow() }),
  });
  if (accent) {
    if (top) slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 0.06, fill: { color: accent } });
    else slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h, fill: { color: accent } });
  }
}

function iconCircle(slide, pres, name, { x, y, d = 0.46, bg = C.indigo, fg = C.white, scale = 0.5 }) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg } });
  const s = d * scale;
  slide.addImage({ data: icons.get(name, fg), x: x + (d - s) / 2, y: y + (d - s) / 2, w: s, h: s });
}

function icon(slide, name, { x, y, s = 0.3, color = C.indigo }) {
  slide.addImage({ data: icons.get(name, color), x, y, w: s, h: s });
}

function numCircle(slide, pres, label, { x, y, d = 0.42, bg = C.saffron, fg = C.white, fontSize = 13 }) {
  slide.addText(String(label), {
    shape: pres.shapes.OVAL, x, y, w: d, h: d, fill: { color: bg }, color: fg,
    fontFace: F.head, fontSize, bold: true, align: "center", valign: "middle", margin: 0,
  });
}

// Boxes with centred text (flow diagrams).
function box(slide, pres, text, { x, y, w, h, fill = C.white, color = C.text, border = C.indigo, fontSize = 10, bold = false,
  radius = 0.06, align = "center", dash, shape = "round", italic = false, borderW = 1, fontFace = F.body, valign = "middle" }) {
  const shp = shape === "round" ? pres.shapes.ROUNDED_RECTANGLE : shape === "diamond" ? pres.shapes.DIAMOND
    : shape === "oval" ? pres.shapes.OVAL : pres.shapes.RECTANGLE;
  const opts = {
    shape: shp, x, y, w, h, fill: { color: fill }, color, fontFace, fontSize, bold, italic, align, valign,
    margin: 3, line: border ? { color: border, width: borderW, ...(dash ? { dashType: dash } : {}) } : undefined,
  };
  if (shape === "round") opts.rectRadius = Math.min(radius, Math.min(w, h) / 2);
  slide.addText(typeof text === "string" ? runs(text, {}) : text, opts);
}

// Straight connector with an arrow head at (x2, y2).
function arrow(slide, pres, x1, y1, x2, y2, { color = C.slate, width = 1.25, head = "triangle", dash } = {}) {
  const x = Math.min(x1, x2);
  const y = Math.min(y1, y2);
  const w = Math.abs(x2 - x1);
  const h = Math.abs(y2 - y1);
  slide.addShape(pres.shapes.LINE, {
    x, y, w, h, flipH: x2 < x1, flipV: y2 < y1,
    line: { color, width, endArrowType: head, ...(dash ? { dashType: dash } : {}) },
  });
}

function line(slide, pres, x1, y1, x2, y2, { color = C.rule, width = 0.75, dash } = {}) {
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width, ...(dash ? { dashType: dash } : {}) },
  });
}

function pill(slide, pres, text, { x, y, w, h = 0.26, fill = C.indigoSoft, color = C.indigo, fontSize = 8.5, border, bold = true }) {
  slide.addText(text, {
    shape: pres.shapes.ROUNDED_RECTANGLE, x, y, w, h, rectRadius: h / 2, fill: { color: fill },
    line: border ? { color: border, width: 0.75 } : undefined,
    fontFace: F.body, fontSize, bold, color, align: "center", valign: "middle", margin: 0,
  });
}

// Status pill used in progress tables.
const STATUS = {
  Completed: [C.greenSoft, C.green],
  "In progress": [C.saffronSoft, C.saffronDark],
  Planned: [C.slatePale, C.slate],
  Running: [C.saffronSoft, C.saffronDark],
};

// ------------------------------------------------------------------ stat callout
function stat(slide, pres, { x, y, w, h, value, label, sub, color = C.saffronDark, valueSize = 28, accent = C.saffron, fill = C.white }) {
  card(slide, pres, { x, y, w, h, accent, fill });
  txt(slide, value, { x: x + 0.2, y: y + 0.1, w: w - 0.3, h: h * 0.45, fontFace: F.head, fontSize: valueSize, bold: true, color, valign: "middle" });
  txt(slide, label.toUpperCase(), { x: x + 0.2, y: y + 0.1 + h * 0.45, w: w - 0.3, h: 0.24, fontSize: 8.5, bold: true, color: C.indigo, charSpacing: 1, valign: "top" });
  if (sub) txt(slide, sub, { x: x + 0.2, y: y + 0.36 + h * 0.45, w: w - 0.3, h: h * 0.55 - 0.44, fontSize: 9.5, color: C.muted, valign: "top" });
}

// ------------------------------------------------------------------ tables
// rows[0] is the header. Cells: string (with **bold**) or { text, fill, color, bold, align, pill }.
function table(slide, pres, rows, { x, y, w, colW, fontSize = 10, rowH = 0.3, headFill = C.ink, headColor = C.white,
  zebra = true, align = [], highlightRows = [], highlightFill = C.saffronSoft, valign = "middle", headFontSize, firstColBold = false }) {
  const border = (last) => [
    { type: "none" }, { type: "none" },
    { type: last ? "none" : "solid", pt: 0.5, color: "E2DCCF" }, { type: "none" },
  ];
  const out = rows.map((row, ri) =>
    row.map((cell, ci) => {
      const c = typeof cell === "object" && cell !== null && !Array.isArray(cell) ? cell : { text: cell };
      const isHead = ri === 0;
      const fill = isHead ? headFill
        : c.fill || (highlightRows.includes(ri) ? highlightFill : zebra && ri % 2 === 0 ? "F6F2EA" : C.white);
      const base = {
        fontFace: F.body, fontSize: isHead ? (headFontSize || fontSize) : fontSize,
        color: isHead ? headColor : c.color || C.text,
        bold: isHead || c.bold || (firstColBold && ci === 0),
        align: c.align || align[ci] || "left", valign,
        fill: { color: fill }, border: border(ri === rows.length - 1),
        margin: [0.03, 0.07, 0.03, 0.07],
      };
      const t = c.text === undefined ? "" : String(c.text);
      return { text: isHead ? t : runs(t, { color: base.color, fontSize: base.fontSize }), options: base };
    })
  );
  slide.addTable(out, { x, y, w, colW, rowH, autoPage: false });
}

// ------------------------------------------------------------------ dot-and-whisker (AUROC with CI)
function forest(slide, pres, rows, { x, y, w, h, min = 0.5, max = 1.0, step = 0.1, labelW = 2.3, valueW = 1.25,
  fontSize = 9.5, axisLabel = "AUROC (95% bootstrap CI)", chance = 0.5 }) {
  const axisH = 0.42;
  const rowH = (h - axisH) / rows.length;
  const px = x + labelW;
  const pw = w - labelW - valueW;
  const X = (v) => px + ((Math.min(Math.max(v, min), max) - min) / (max - min)) * pw;
  const bottom = y + rows.length * rowH;
  for (let t = min; t <= max + 1e-9; t += step) {
    const tx = X(t);
    line(slide, pres, tx, y, tx, bottom, { color: Math.abs(t - chance) < 1e-9 ? "B8B2A6" : "E6E0D4", width: 0.75, dash: Math.abs(t - chance) < 1e-9 ? "dash" : undefined });
    txt(slide, t.toFixed(1), { x: tx - 0.3, y: bottom + 0.03, w: 0.6, h: 0.18, fontSize: 8.5, color: C.muted, align: "center" });
  }
  txt(slide, axisLabel, { x: px, y: bottom + 0.2, w: pw, h: 0.2, fontSize: 8.5, color: C.muted, align: "center", italic: true });
  rows.forEach((r, i) => {
    const cy = y + i * rowH + rowH / 2;
    const col = r.color || C.indigo;
    if (r.band) slide.addShape(pres.shapes.RECTANGLE, { x, y: y + i * rowH, w, h: rowH, fill: { color: r.band } });
    txt(slide, r.label, { x, y: cy - rowH / 2, w: labelW - 0.12, h: rowH, fontSize, color: C.text, bold: !!r.bold, align: "right", valign: "middle" });
    if (r.lo !== undefined && r.hi !== undefined) line(slide, pres, X(r.lo), cy, X(r.hi), cy, { color: col, width: 2 });
    const d = Math.min(0.15, rowH * 0.7);
    slide.addShape(pres.shapes.OVAL, { x: X(r.value) - d / 2, y: cy - d / 2, w: d, h: d, fill: { color: col }, line: { color: C.white, width: 0.75 } });
    const ci = r.lo !== undefined ? ` (${r.lo.toFixed(3)}–${r.hi.toFixed(3)})` : "";
    slide.addText([
      { text: r.value.toFixed(3), options: { bold: true, color: col === C.slate ? C.text : col } },
      { text: ci, options: { color: C.muted, fontSize: fontSize - 1.5 } },
    ], { x: px + pw + 0.08, y: cy - rowH / 2, w: valueW - 0.08, h: rowH, fontFace: F.body, fontSize, margin: 0, valign: "middle" });
  });
}

// ------------------------------------------------------------------ diverging bars (fusion weights)
function diverging(slide, pres, rows, { x, y, w, h, labelW = 2.0, maxAbs, posColor = C.rose, negColor = C.teal, fontSize = 9.5 }) {
  const m = maxAbs || Math.max(...rows.map((r) => Math.abs(r.value))) * 1.15;
  const rowH = h / rows.length;
  const px = x + labelW;
  const pw = w - labelW;
  const zero = px + pw / 2;
  line(slide, pres, zero, y, zero, y + h, { color: "9CA3AF", width: 1 });
  rows.forEach((r, i) => {
    const cy = y + i * rowH + rowH / 2;
    const bw = (Math.abs(r.value) / m) * (pw / 2);
    const bh = Math.min(0.2, rowH * 0.62);
    const col = r.value >= 0 ? posColor : negColor;
    txt(slide, r.label, { x, y: cy - rowH / 2, w: labelW - 0.1, h: rowH, fontSize, color: C.text, align: "right", valign: "middle", bold: !!r.bold });
    slide.addShape(pres.shapes.RECTANGLE, { x: r.value >= 0 ? zero : zero - bw, y: cy - bh / 2, w: Math.max(bw, 0.01), h: bh, fill: { color: col } });
    const vx = r.value >= 0 ? zero + bw + 0.05 : zero - bw - 0.6;
    txt(slide, (r.value >= 0 ? "+" : "−") + Math.abs(r.value).toFixed(2), { x: vx, y: cy - rowH / 2, w: 0.55, h: rowH, fontSize: fontSize - 1, color: C.muted, align: r.value >= 0 ? "left" : "right", valign: "middle" });
  });
}

// ------------------------------------------------------------------ pending result panel
// Text always contains the token PENDING so that unfinished slides can be grepped.
function pending(slide, pres, { x, y, w, h, title, body, file, cmd }) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08, fill: { color: "F3F1FB" }, line: { color: "8B87C9", width: 1, dashType: "dash" },
  });
  iconCircle(slide, pres, "FaHourglassHalf", { x: x + 0.25, y: y + 0.25, d: 0.5, bg: C.saffron });
  txt(slide, "PENDING · RESULT NOT YET AVAILABLE", { x: x + 0.9, y: y + 0.24, w: w - 1.1, h: 0.22, fontSize: 9, bold: true, color: C.saffronDark, charSpacing: 1 });
  txt(slide, title, { x: x + 0.9, y: y + 0.46, w: w - 1.1, h: 0.32, fontFace: F.head, fontSize: 13, bold: true, color: C.ink });
  txt(slide, body || "", { x: x + 0.3, y: y + 0.95, w: w - 0.6, h: Math.max(0.3, h - 1.6), fontSize: 10.5, color: C.muted });
  if (file) {
    txt(slide, [
      { text: "Appears automatically when this file exists:", options: { color: C.muted, fontSize: 8.5, breakLine: true } },
      { text: file, options: { fontFace: F.mono, color: C.indigo, fontSize: 8.5, breakLine: !!cmd } },
      ...(cmd ? [{ text: `Rebuild: ${cmd}`, options: { fontFace: F.mono, color: C.indigo, fontSize: 8.5 } }] : []),
    ], { x: x + 0.3, y: y + h - 0.62, w: w - 0.6, h: 0.52, valign: "bottom" });
  }
}

// ------------------------------------------------------------------ gantt / work plan
// tasks: { label, start, end, status }  (month indices relative to months[0], fractional allowed)
function gantt(slide, pres, tasks, { x, y, w, h, months, years, today, todayLabel = "this review", labelW = 2.55, fontSize = 9.5 }) {
  const headH = 0.42;
  const rowH = (h - headH) / tasks.length;
  const gx = x + labelW;
  const gw = w - labelW;
  const mw = gw / months.length;
  // year band
  if (years) {
    years.forEach(([label, from, to]) => {
      txt(slide, label, { x: gx + from * mw, y, w: (to - from) * mw, h: 0.18, fontSize: 8, bold: true, color: C.indigo, align: "center" });
    });
  }
  months.forEach((m, i) => {
    txt(slide, m, { x: gx + i * mw, y: y + 0.18, w: mw, h: 0.2, fontSize: 8.5, color: C.muted, align: "center" });
    if (i > 0) line(slide, pres, gx + i * mw, y + headH, gx + i * mw, y + h, { color: "ECE6DA", width: 0.5 });
  });
  const colors = { Completed: [C.indigo, C.white], "In progress": [C.saffron, C.white], Planned: ["D9DEE7", C.text] };
  tasks.forEach((t, i) => {
    const ry = y + headH + i * rowH;
    if (i % 2 === 0) slide.addShape(pres.shapes.RECTANGLE, { x, y: ry, w, h: rowH, fill: { color: "F4EFE6" } });
    txt(slide, t.label, { x: x + 0.05, y: ry, w: labelW - 0.1, h: rowH, fontSize, color: C.text, valign: "middle" });
    const [fill] = colors[t.status] || colors.Planned;
    const bh = Math.min(0.2, rowH * 0.6);
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: gx + t.start * mw + 0.03, y: ry + (rowH - bh) / 2, w: Math.max((t.end - t.start) * mw - 0.06, 0.08), h: bh,
      rectRadius: bh / 2, fill: { color: fill },
    });
  });
  if (today !== undefined) {
    const tx = gx + today * mw;
    line(slide, pres, tx, y + headH - 0.04, tx, y + h, { color: C.rose, width: 1.5, dash: "dash" });
    pill(slide, pres, todayLabel, { x: tx - 0.5, y: y + h + 0.03, w: 1.0, h: 0.2, fill: C.roseSoft, color: C.rose, fontSize: 7.5 });
  }
}

function legendRow(slide, pres, items, { x, y, fontSize = 8.5, gap = 0.18 }) {
  let cx = x;
  items.forEach(([label, color]) => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: y + 0.04, w: 0.22, h: 0.12, rectRadius: 0.06, fill: { color } });
    const tw = 0.065 * label.length + 0.1;
    txt(slide, label, { x: cx + 0.27, y, w: tw, h: 0.2, fontSize, color: C.muted, valign: "middle" });
    cx += 0.27 + tw + gap;
  });
}

// ------------------------------------------------------------------ images
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function imageFit(slide, { path, data, size, x, y, w, h, align = "center", valign = "middle", frame = true, pres }) {
  const s = size || pngSize(path);
  const r = Math.min(w / s.w, h / s.h);
  const iw = s.w * r;
  const ih = s.h * r;
  const ix = align === "left" ? x : align === "right" ? x + w - iw : x + (w - iw) / 2;
  const iy = valign === "top" ? y : valign === "bottom" ? y + h - ih : y + (h - ih) / 2;
  if (frame && pres) {
    slide.addShape(pres.shapes.RECTANGLE, { x: ix - 0.04, y: iy - 0.04, w: iw + 0.08, h: ih + 0.08, fill: { color: C.white }, line: { color: C.rule, width: 0.75 }, shadow: shadow() });
  }
  slide.addImage({ ...(path ? { path } : { data }), x: ix, y: iy, w: iw, h: ih });
  return { x: ix, y: iy, w: iw, h: ih };
}

// ------------------------------------------------------------------ charts
function chartBase(o = {}) {
  return {
    fontFace: F.body,
    catAxisLabelColor: C.muted, valAxisLabelColor: C.muted,
    catAxisLabelFontSize: 9, valAxisLabelFontSize: 8.5, catAxisLabelFontFace: F.body, valAxisLabelFontFace: F.body,
    valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
    catAxisLineColor: "9CA3AF", valAxisLineShow: false,
    showLegend: true, legendPos: "b", legendFontSize: 9, legendColor: C.text, legendFontFace: F.body,
    chartArea: { fill: { color: C.white } }, plotArea: { fill: { color: C.white } },
    ...o,
  };
}

function barChart(slide, pres, data, o) {
  slide.addChart(pres.charts.BAR, data, chartBase({
    barDir: "col", barGapWidthPct: 60,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.text, dataLabelFontSize: 8, dataLabelFormatCode: "0.00",
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0",
    ...o,
  }));
}

function lineChart(slide, pres, data, o) {
  slide.addChart(pres.charts.LINE, data, chartBase({
    lineSize: 2, lineDataSymbol: "none", valAxisLabelFormatCode: "0.00", ...o,
  }));
}

// ------------------------------------------------------------------ deck and slide templates
class Deck {
  constructor(meta) {
    this.meta = meta;
    this.items = [];
  }

  add(section, build) {
    this.items.push({ section, build });
    return this;
  }

  build() {
    const pres = new pptxgen();
    pres.layout = "LAYOUT_16x9";
    pres.author = "Manju Pal";
    pres.company = "University School of Information and Communication Technology, Gautam Buddha University";
    pres.subject = this.meta.milestone;
    pres.title = this.meta.title;
    const N = this.items.length;
    this.items.forEach((it, i) => {
      const slide = pres.addSlide();
      it.build({ pres, slide, n: i + 1, N, section: it.section, meta: this.meta });
    });
    return pres;
  }
}

function footer(ctx) {
  const { slide, pres, n, N, meta, section } = ctx;
  const y = H - 0.34;
  rect(slide, pres, 0, y, W, 0.34, C.ink);
  rect(slide, pres, 0, y, 0.18, 0.34, C.saffron);
  txt(slide, "USICT · Gautam Buddha University", { x: 0.5, y, w: 2.5, h: 0.34, fontSize: 8, color: C.dimText, valign: "middle" });
  const secs = meta.sections;
  const idx = secs.indexOf(section);
  let x = 3.0;
  secs.forEach((s, i) => {
    const w = 0.058 * s.length + 0.2;
    const cur = i === idx;
    txt(slide, s, {
      x, y, w, h: 0.34, fontSize: 8, bold: cur, align: "center", valign: "middle",
      color: cur ? C.saffron : idx >= 0 && i < idx ? "A5B4FC" : "7D80B3",
    });
    x += w + 0.1;
  });
  txt(slide, `${n}/${N}`, { x: 8.7, y, w: 0.8, h: 0.34, fontSize: 9, bold: true, color: C.white, align: "right", valign: "middle" });
}

const BADGES = {
  sim: [C.saffronSoft, C.saffronDark],
  real: [C.tealSoft, C.tealDark],
  synth: [C.roseSoft, C.rose],
  info: [C.indigoSoft, C.indigo],
  pending: ["F3F1FB", C.indigo],
};

function content(ctx, { kicker, title, badge, notes, titleSize = 24 }) {
  const { slide, pres } = ctx;
  slide.background = { color: C.bg };
  txt(slide, kicker.toUpperCase(), { x: 0.5, y: 0.22, w: 6.5, h: 0.24, fontSize: 9.5, bold: true, color: C.saffronDark, charSpacing: 2, valign: "middle" });
  let titleW = 9.0;
  if (badge) {
    const [text, kind] = Array.isArray(badge) ? badge : [badge, "sim"];
    const [bg, fg] = BADGES[kind] || BADGES.info;
    const bw = 0.068 * text.length + 0.4;
    pill(slide, pres, text, { x: W - 0.5 - bw, y: 0.2, w: bw, h: 0.28, fill: bg, color: fg, border: fg, fontSize: 8.5 });
  }
  txt(slide, title, { x: 0.5, y: 0.47, w: titleW, h: 0.62, fontFace: F.head, fontSize: titleSize, bold: true, color: C.ink, valign: "middle" });
  footer(ctx);
  if (notes) slide.addNotes(notes);
}

// Concentric rings (motif on dark slides): a quiet nod to a chakra / rising sun.
function rings(slide, pres, cx, cy, radii, color = C.saffron) {
  radii.forEach((r, i) => {
    slide.addShape(pres.shapes.OVAL, {
      x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { type: "none" },
      line: { color, width: 1.25, transparency: 35 + i * 15 },
    });
  });
}

function darkBase(ctx) {
  const { slide, pres } = ctx;
  slide.background = { color: C.ink };
  rings(slide, pres, 9.35, 4.95, [0.75, 1.35, 1.95, 2.55]);
  slide.addShape(pres.shapes.OVAL, { x: 9.35 - 0.32, y: 4.95 - 0.32, w: 0.64, h: 0.64, fill: { color: C.saffron } });
}

function titleSlide(ctx, { title, titleSize = 26, subtitle, milestone, date, notes }) {
  const { slide, pres, n, N } = ctx;
  darkBase(ctx);
  txt(slide, "UNIVERSITY SCHOOL OF INFORMATION AND COMMUNICATION TECHNOLOGY", { x: 0.6, y: 0.38, w: 8, h: 0.22, fontSize: 9, bold: true, color: C.saffron, charSpacing: 2 });
  txt(slide, "Gautam Buddha University, Greater Noida", { x: 0.6, y: 0.6, w: 8, h: 0.22, fontSize: 10.5, color: C.dimText });
  const mw = 0.075 * milestone.length + 0.5;
  pill(slide, pres, milestone.toUpperCase(), { x: 0.6, y: 1.02, w: mw, h: 0.3, fill: C.ink, color: C.saffron, border: C.saffron, fontSize: 8.5 });
  txt(slide, title, { x: 0.6, y: 1.45, w: 7.9, h: 1.65, fontFace: F.head, fontSize: titleSize, bold: true, color: C.white, valign: "middle", paraSpaceAfter: 0 });
  if (subtitle) txt(slide, subtitle, { x: 0.6, y: 3.12, w: 7.9, h: 0.4, fontFace: F.head, fontSize: 13.5, italic: true, color: "F6C27A", valign: "top" });
  const cols = [
    ["PRESENTED BY", "Manju Pal", "M.Tech CSE (Artificial Intelligence and Robotics)"],
    ["SUPERVISOR", "Dr. Raju Pal", "Assistant Professor, Department of CSE"],
    ["DATE", date, ""],
  ];
  const xs = [0.6, 3.75, 6.75];
  const ws = [3.0, 2.85, 1.3];
  slide.addShape(pres.shapes.LINE, { x: 0.6, y: 3.85, w: 7.5, h: 0, line: { color: "3B3680", width: 0.75 } });
  cols.forEach(([l, v, s], i) => {
    txt(slide, l, { x: xs[i], y: 3.98, w: ws[i], h: 0.2, fontSize: 8, bold: true, color: C.saffron, charSpacing: 2 });
    txt(slide, v, { x: xs[i], y: 4.2, w: ws[i], h: 0.3, fontFace: F.head, fontSize: 13.5, bold: true, color: C.white });
    if (s) txt(slide, s, { x: xs[i], y: 4.5, w: ws[i], h: 0.4, fontSize: 9.5, color: C.dimText });
  });
  txt(slide, `${n}/${N}`, { x: 0.6, y: 5.18, w: 1, h: 0.25, fontSize: 9, color: C.dimText });
  if (notes) slide.addNotes(notes);
}

function divider(ctx, { num, title, items, notes }) {
  const { slide, pres, n, N } = ctx;
  darkBase(ctx);
  txt(slide, String(num).padStart(2, "0"), { x: 0.6, y: 1.05, w: 2.0, h: 1.2, fontFace: F.head, fontSize: 72, bold: true, color: C.saffron, valign: "middle" });
  txt(slide, title, { x: 0.6, y: 2.25, w: 7.5, h: 0.8, fontFace: F.head, fontSize: 36, bold: true, color: C.white, valign: "middle" });
  let x = 0.6;
  items.forEach((it) => {
    const w = 0.085 * it.length + 0.45;
    pill(slide, pres, it, { x, y: 3.3, w, h: 0.36, fill: "2A2563", color: C.white, border: "4B4596", fontSize: 11, bold: false });
    x += w + 0.15;
  });
  txt(slide, "USICT · Gautam Buddha University", { x: 0.6, y: 5.18, w: 4, h: 0.25, fontSize: 8.5, color: C.dimText });
  txt(slide, `${n}/${N}`, { x: 7.3, y: 5.18, w: 0.8, h: 0.25, fontSize: 9, bold: true, color: C.white, align: "right" });
  if (notes) slide.addNotes(notes);
}

function outline(ctx, { sections, notes, kicker = "Agenda" }) {
  const { slide, pres } = ctx;
  content(ctx, { kicker, title: "Outline of the Presentation", notes });
  const top = 1.25;
  const rowH = 0.76;
  line(slide, pres, 0.88, top + 0.25, 0.88, top + (sections.length - 1) * rowH + 0.25, { color: C.saffron, width: 1.5 });
  sections.forEach(([title, sub, iconName], i) => {
    const y = top + i * rowH;
    numCircle(slide, pres, i + 1, { x: 0.67, y: y + 0.04, d: 0.42 });
    txt(slide, title, { x: 1.3, y: y, w: 3.1, h: 0.5, fontFace: F.head, fontSize: 16, bold: true, color: C.ink, valign: "middle" });
    txt(slide, sub, { x: 4.4, y: y, w: 4.5, h: 0.5, fontSize: 11.5, color: C.muted, valign: "middle" });
    if (iconName) icon(slide, iconName, { x: 9.0, y: y + 0.1, s: 0.3, color: C.indigo });
    if (i < sections.length - 1) line(slide, pres, 1.3, y + rowH - 0.1, 9.5, y + rowH - 0.1, { color: "EAE4D8", width: 0.75 });
  });
}

function references(ctx, { refs, notes, title = "References", startAt = 1, fontSize = 8.5 }) {
  const { slide, pres } = ctx;
  content(ctx, { kicker: "Bibliography", title, notes });
  const half = Math.ceil(refs.length / 2);
  [refs.slice(0, half), refs.slice(half)].forEach((col, ci) => {
    const arr = [];
    col.forEach((r, i) => {
      const num = startAt + i + (ci ? half : 0);
      arr.push({ text: `[${num}] `, options: { bold: true, color: C.indigo } });
      arr.push({ text: r, options: { color: C.text, breakLine: i < col.length - 1 } });
    });
    slide.addText(arr, { x: 0.5 + ci * 4.6, y: 1.22, w: 4.4, h: 3.88, fontFace: F.body, fontSize, margin: 0, valign: "top", paraSpaceAfter: 3.5 });
  });
  iconCircle(slide, pres, "FaBook", { x: 9.0, y: 0.52, d: 0.5, bg: C.indigo });
}

function thanks(ctx, { lines = [], notes }) {
  const { slide, pres, n, N } = ctx;
  darkBase(ctx);
  txt(slide, "Thank You", { x: 0.6, y: 1.35, w: 7.5, h: 1.0, fontFace: F.head, fontSize: 48, bold: true, color: C.white, valign: "middle" });
  txt(slide, "Questions and discussion", { x: 0.6, y: 2.35, w: 7.5, h: 0.45, fontFace: F.head, fontSize: 18, italic: true, color: "F6C27A" });
  lines.forEach((l, i) => {
    iconCircle(slide, pres, l.icon, { x: 0.6, y: 3.15 + i * 0.5, d: 0.36, bg: "2A2563", fg: C.saffron });
    txt(slide, l.text, { x: 1.1, y: 3.15 + i * 0.5, w: 6.8, h: 0.36, fontSize: 11.5, color: C.dimText, valign: "middle" });
  });
  txt(slide, `${n}/${N}`, { x: 0.6, y: 5.18, w: 1, h: 0.25, fontSize: 9, color: C.dimText });
  if (notes) slide.addNotes(notes);
}

module.exports = {
  W, H, C, F, preloadIcons, runs, txt, bullets, rect, card, iconCircle, icon, numCircle, box, arrow, line, pill,
  STATUS, stat, table, forest, diverging, pending, gantt, legendRow, imageFit, pngSize, chartBase, barChart, lineChart,
  Deck, content, titleSlide, divider, outline, references, thanks, shadow,
};
