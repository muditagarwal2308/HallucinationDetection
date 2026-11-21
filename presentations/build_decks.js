// Builds the four milestone decks (M1–M4) from the repository's result files.
//
//   node presentations/build_decks.js            # all decks
//   node presentations/build_decks.js M2 M4      # selected decks
//
// Then export PDFs with:  powershell -File presentations/export_pdf.ps1
// Result files that do not exist yet are rendered as "PENDING" panels (grep for PENDING).
const path = require("path");
const sharp = require("sharp");
const K = require("./lib/kit");
const data = require("./lib/data");

const DECKS = {
  M1: "./decks/m1_vghd",
  M2: "./decks/m2_ugrav",
  M3: "./decks/m3_implementation",
  M4: "./decks/m4_defence",
};

async function crop(file, region) {
  const buf = await sharp(file).extract(region).png().toBuffer();
  return { data: "image/png;base64," + buf.toString("base64"), size: { w: region.width, h: region.height } };
}

async function main() {
  const wanted = process.argv.slice(2).map((s) => s.toUpperCase());
  const D = data.load();
  await K.preloadIcons();
  const shots = path.join(data.ROOT, "docs", "screenshots");
  const assets = {
    verifier: await crop(path.join(shots, "verifier.png"), { left: 512, top: 52, width: 1180, height: 745 }),
    ugravLive: await crop(path.join(shots, "ugrav_live.png"), { left: 505, top: 30, width: 1190, height: 640 }),
    home: await crop(path.join(shots, "home.png"), { left: 505, top: 30, width: 1190, height: 640 }),
  };
  console.log(`Real-model pilot result: ${D.real ? "found" : "PENDING"} (${D.files.real})`);
  console.log(`VGHD DeBERTa result:     ${D.gitaDeb ? "found" : "PENDING"} (${D.files.gitaDeb})`);
  for (const [key, modPath] of Object.entries(DECKS)) {
    if (wanted.length && !wanted.includes(key)) continue;
    const mod = require(modPath);
    const deck = mod.build(D, assets);
    const pres = deck.build();
    const out = path.join(__dirname, `${mod.file}.pptx`);
    await pres.writeFile({ fileName: out });
    console.log(`${key}: ${deck.items.length} slides -> ${path.relative(data.ROOT, out)}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
