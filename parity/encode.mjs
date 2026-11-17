// Caption vectors from the demo's text.js (onnxruntime-web, WebAssembly), for comparing the app's encoder:
//   node encode.mjs "череп|бургер|череп бургер" [node] > vectors_web.json
// With "node" as the second argument the same text.js runs on onnxruntime-node (native CPU, the app's ORT version).
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = path.resolve(HERE, "..");
const DEMO = path.join(APP, "..", "release", "github", "docs", "demo");
const req = createRequire(path.join(process.env.PARITY_MODULES || path.join(HERE, "node_modules"), "_.js"));
const load = async (name) => import(pathToFileURL(req.resolve(name)).href);
const native = process.argv[3] === "node";
const ort = native ? await load("onnxruntime-node") : await load("onnxruntime-web");
const tok = await load("@huggingface/tokenizers");
const textjs = await import(pathToFileURL(path.join(DEMO, "text.js")).href);
if (!native) ort.env.wasm.numThreads = 1;
// text.js asks for the "wasm" provider; onnxruntime-node only has "cpu"
const ortFor = native ? { ...ort, InferenceSession: { create: (b, o) => ort.InferenceSession.create(b, { ...o, executionProviders: ["cpu"] }) } } : ort;
const enc = await textjs.loadTextEncoder({ ort: ortFor, Tokenizer: tok.Tokenizer || tok.default.Tokenizer,
  files: textjs.encoderFiles(path.join(APP, "models", "encoder").replace(/\\/g, "/")), fetchBytes: async (p) => new Uint8Array(fs.readFileSync(p)) });
const out = {};
for (const t of (process.argv[2] || "").split("|").filter(Boolean)) {
  const v = await enc.encode(t);
  out[t] = { ids: enc.tokenizer.encode(t).ids, vector: Buffer.from(v.buffer, v.byteOffset, v.byteLength).toString("base64") };
}
console.log(JSON.stringify(out));
