// Icon rasterisation: react-icons (SVG) -> PNG (sharp) -> base64.
// All icons are rendered once, up front, so that slide builders can stay synchronous.
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const cache = new Map();

async function render(name, color) {
  const Comp = fa[name];
  if (!Comp) throw new Error(`Unknown icon: ${name}`);
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Comp, { color: `#${color}`, size: "256" })
  );
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

async function preload(names, colors) {
  for (const n of names) {
    for (const c of colors) {
      const key = `${n}|${c}`;
      if (!cache.has(key)) cache.set(key, await render(n, c));
    }
  }
}

function get(name, color) {
  const key = `${name}|${color}`;
  if (!cache.has(key)) throw new Error(`Icon not preloaded: ${key}`);
  return cache.get(key);
}

module.exports = { preload, get };
