// Reference sprites from the web demo's own code, headless: engine.js (the sampler), mix.js (caption mixing) and
// text.js (the text encoder) of release/github/docs/demo are imported as they are; the run logic of app.js
// (prepareMix, strategyFor, the modes, the row budget) is copied below line by line, because app.js touches the page.
//
//   node demo_parity.mjs --cases cases.json --out ref_web.json [--backend web|node] [--only name,name]
//
// --backend web : onnxruntime-web 1.31.1, WebAssembly EP, 1 thread (the demo's CPU path in a browser that is not
//                 cross-origin isolated, as on GitHub Pages);
// --backend node: onnxruntime-node 1.31.1, CPU EP (faster; the app's native ONNX Runtime build is the same version).
// --encoder node: text.js on onnxruntime-node too (the default is WebAssembly, as in the demo).
// The text encoder always runs on onnxruntime-web (WebAssembly), as in the demo.
//
// Packages: `npm install onnxruntime-web@1.31.1 onnxruntime-node@1.31.1 @huggingface/tokenizers@0.2.0` in this
// folder, or point PARITY_MODULES at a node_modules folder that has them. Files: the demo from
// ../../release/github/docs/demo, the model from ../../release/hf (decoder.f16 expanded in memory, as the demo does)
// or, for --backend node, models/v2 (the app's float32 decoder.data, the same numbers), the encoder from
// models/encoder. Output: palette indices per sprite (-1 = transparent), the variant labels and timings.

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = path.resolve(HERE, "..");
const ROOT = path.resolve(APP, "..");
const DEMO = path.join(ROOT, "release", "github", "docs", "demo");
const MODEL = path.join(ROOT, "release", "hf");

const args = process.argv.slice(2);
const arg = (k, d = null) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const backend = arg("--backend", "web");
const casesPath = path.resolve(arg("--cases", path.join(HERE, "cases.json")));
const outPath = path.resolve(arg("--out", path.join(HERE, `ref_${backend}.json`)));
const only = arg("--only") ? new Set(arg("--only").split(",")) : null;
const fixedCache = arg("--cache", "fixed") === "fixed";  // the demo: fixed (decoder_static.onnx); growing = decoder.onnx
// --encoder node: the demo's text.js on onnxruntime-node (native CPU) instead of WebAssembly, to tell the encoder's
// float differences from the sampler's (needs --backend node)
const encoderNative = arg("--encoder", "web") === "node";

const modules = process.env.PARITY_MODULES || path.join(HERE, "node_modules");
const req = createRequire(path.join(modules, "_.js"));
const load = async (name) => import(pathToFileURL(req.resolve(name)).href);
// onnxruntime-node first: both packages register a "cpu" backend in the shared onnxruntime-common, and the node one
// (priority 100) must win over the WebAssembly one (priority 10) for --backend node.
const ortNode = backend === "node" ? await load("onnxruntime-node") : null;
const ortWeb = await load("onnxruntime-web");
const tokMod = await load("@huggingface/tokenizers");
const Tokenizer = tokMod.Tokenizer || tokMod.default.Tokenizer;
const engine = await import(pathToFileURL(path.join(DEMO, "engine.js")).href);
const MX = await import(pathToFileURL(path.join(DEMO, "mix.js")).href);
const textjs = await import(pathToFileURL(path.join(DEMO, "text.js")).href);

ortWeb.env.wasm.numThreads = 1;
const fetchBytes = async (p) => new Uint8Array(fs.readFileSync(p));

// ------------------------------------------------------------------------------------------------- the model
let model;
const t0 = performance.now();
if (backend === "web") {
  model = await engine.loadModel({ ort: ortWeb, files: engine.modelFiles(MODEL.replace(/\\/g, "/")), ep: "wasm", fetchBytes, fixedCache });
} else {
  const files = engine.modelFiles(MODEL.replace(/\\/g, "/"));
  const [info, pal, emb] = [JSON.parse(fs.readFileSync(files.embJson)), JSON.parse(fs.readFileSync(files.palette)), fs.readFileSync(files.embBin)];
  const session = await ortNode.InferenceSession.create(path.join(APP, "models", "v2", fixedCache ? "decoder_static.onnx" : "decoder.onnx"), { executionProviders: ["cpu"], graphOptimizationLevel: "all" });
  model = new engine.SpriteModel(ortNode, session, info, new Uint8Array(emb), pal.colors, "cpu");
  model.fixedCache = fixedCache;
}
console.log(`model on ${backend} in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

// ---------------------------------------------------------------------------------- app.js state and helpers
const presetsJson = JSON.parse(fs.readFileSync(path.join(DEMO, "presets.json"), "utf8"));
function b64floats(s) {
  const u8 = Uint8Array.from(Buffer.from(s, "base64"));
  return new Float32Array(u8.buffer);
}
const state = { presets: new Map(), lex: null, kindOverride: new Map(), learnedKind: new Map(), variants: new Set(MX.ALL_VARIANTS),
                encoder: null, encoderReady: false };
for (const it of presetsJson.items) state.presets.set(it.text, b64floats(it.vector));
state.lex = await MX.loadLexicon(DEMO.replace(/\\/g, "/"), fetchBytes);

async function getEncoder() {
  if (!state.encoder) {
    const ortEnc = encoderNative
      ? { ...ortNode, InferenceSession: { create: (b, o) => ortNode.InferenceSession.create(b, { ...o, executionProviders: ["cpu"] }) } }
      : ortWeb;
    state.encoder = textjs.loadTextEncoder({ ort: ortEnc, Tokenizer, files: textjs.encoderFiles(path.join(APP, "models", "encoder").replace(/\\/g, "/")), fetchBytes });
  }
  const enc = await state.encoder;
  state.encoderReady = true;
  return enc;
}
const encodeText = async (text) => (await getEncoder()).encode(text);
async function vectorFor(text) {
  if (!text) return null;
  if (state.presets.has(text)) return state.presets.get(text);
  return encodeText(text);
}
const kindsFor = () => new Map([...state.learnedKind, ...state.kindOverride]);
const presetNeedsEncoder = (text, concepts) =>
  state.presets.has(text) && !state.encoderReady && concepts.some((c) => !state.lex.index.has(c.text));

// app.js prepareMix (verbatim but for renderWords)
async function prepareMix(text) {
  if (!state.lex || !text) return null;
  let { concepts } = MX.parse(text, state.lex, kindsFor());
  if (concepts.length < 2 || presetNeedsEncoder(text, concepts)) return null;
  const unknown = concepts.filter((c) => !c.known);
  if (unknown.length) {
    const enc = await getEncoder();
    for (const c of unknown) {
      c.vec = await enc.encode(c.text);
      if (!state.learnedKind.has(c.text)) state.learnedKind.set(c.text, (await MX.nearestKind(c.vec, state.lex)).kind);
      if (!state.kindOverride.has(c.text)) c.kind = state.learnedKind.get(c.text);
    }
  }
  const pl = MX.plan(concepts, [...state.variants]);
  if (!pl) return null;
  const vectors = await Promise.all(pl.concepts.map((c) => c.vec || MX.conceptVector(c, state.lex, encodeText)));
  let caption = null, captionExact = false;
  if (pl.kinds.includes("caption")) {
    if (state.presets.has(text)) { caption = state.presets.get(text); captionExact = true; }
    else if (state.encoderReady) { caption = await encodeText(text); captionExact = true; }
    else {
      caption = new Float32Array(vectors[0].length);
      for (const v of vectors) for (let j = 0; j < v.length; j++) caption[j] += v[j];
      const norm = Math.hypot(...caption) || 1;
      for (let j = 0; j < caption.length; j++) caption[j] /= norm;
    }
  }
  return { pl, vectors, caption, captionExact };
}

// app.js rowsFor on a desktop: WebGPU 24 rows at 16x16, else 2 * chunkFor (WebGPU 4 / 8 sprites, CPU 2 / 8).
const rowsFor = (size, ep) => (ep === "webgpu" && size === 16 ? 24 : 2 * (ep === "webgpu" ? (size !== 16 ? 4 : 8) : (size !== 16 ? 2 : 8)));

function maskOf(rows, size) {
  if (!rows) return null;
  const m = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) m[y * size + x] = rows[y] && rows[y][x] === "#" ? 1 : 0;
  return m;
}

// app.js run(st) without the page: returns the images
async function run(st) {
  const tStart = performance.now();
  const mixed = st.mode === "new" ? await prepareMix(st.text) : null;
  const vector = mixed ? null : await vectorFor(st.text);
  const common = { vector, size: st.size, gShape: st.gShape, gColour: st.gColour, temperature: st.temperature, topP: st.topP,
                   rows: st.rows };
  let variants = null;
  if (mixed) {
    const rngV = engine.makeRng((st.seed ^ 0x5bd1e995) >>> 0), order = mixed.pl.concepts;
    variants = Array.from({ length: st.n }, () => MX.drawVariant(mixed.pl, rngV));
    const asCaption = mixed.caption ? engine.Strategy.split(model, mixed.caption, st.gShape, st.gColour) : null;
    common.strategyFor = (id, view) => {
      if (variants[id].kind === "caption") return asCaption;
      const { concepts, maps } = MX.variantMaps(variants[id], mixed.pl, view.S, view.o, view.w);
      return engine.Strategy.mix(model, mixed.vectors, order.map((c) => maps[concepts.indexOf(c)] || new Float32Array(view.S * view.S)));
    };
  }
  let mask = null, source = null, seed = st.seed;
  if (st.mode === "draw") mask = maskOf(st.mask, st.size);
  if (st.mode === "same") {
    const r = await model.generate({ ...common, n: 1, seed, filter: st.filter, chunk: 1 });
    source = r.images[0];
    mask = Uint8Array.from(source, (v) => (v >= 0 ? 1 : 0));
    seed += 1;
  }
  const res = await model.generate({ ...common, n: st.n, seed, mask, filter: st.filter });
  return {
    name: st.name, images: res.images.map((a) => Array.from(a)), source: source ? Array.from(source) : null,
    variants: variants ? variants.map((v) => `${v.kind}: ${MX.label(v, st.text)}`) : null,
    concepts: mixed ? mixed.pl.concepts.map((c) => `${c.text}:${c.kind}:${c.p.toFixed(4)}`) : null,
    captionExact: mixed ? mixed.captionExact : null, stats: res.stats, secs: (performance.now() - tStart) / 1000,
  };
}

// ------------------------------------------------------------------------------------------------- the cases
const cases = JSON.parse(fs.readFileSync(casesPath, "utf8"));
const out = { backend, cache: fixedCache ? "fixed" : "growing",
              ort: (backend === "web" ? "onnxruntime-web 1.31.1 (wasm, 1 thread)" : "onnxruntime-node 1.31.1 (cpu)") +
                   (encoderNative ? ", encoder on onnxruntime-node" : ", encoder on onnxruntime-web (wasm)"), cases: [] };
for (const c of cases) {
  if (only && !only.has(c.name)) continue;
  const st = { mode: "new", gShape: 1, gColour: 2, temperature: 1, topP: 1, filter: true, ...c };
  st.rows = c.rows || rowsFor(st.size, c.demoEp || "webgpu");
  // per-case page state: a fresh page unless the case says otherwise
  state.kindOverride = new Map(Object.entries(c.overrides || {}));
  state.learnedKind = new Map();
  state.variants = new Set(c.variants || MX.ALL_VARIANTS);
  if (c.encoderReady) await getEncoder(); else state.encoderReady = false;
  const r = await run(st);
  r.rows = st.rows;
  out.cases.push(r);
  console.log(`${c.name}: ${r.images.length} sprites in ${r.secs.toFixed(1)} s, rows ${st.rows}, redrawn ${r.stats.rejected}` +
              (r.concepts ? ` · concepts ${r.concepts.join(" ")}` : "") + (r.variants ? ` · ${r.variants.join(" | ")}` : ""));
  fs.writeFileSync(outPath, JSON.stringify(out));
}
console.log(`wrote ${outPath}`);
