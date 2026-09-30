// Measures the memory and speed of the WebAssembly package, in a fresh process, on Bun or Node.js.
//
// Usage (after `pnpm run build`; run a few times, each run is a fresh process):
//   bun scripts/measure-wasm.mjs [pkg/web]
//   node --expose-gc scripts/measure-wasm.mjs [pkg/web]
//
// Scores the samples in tests/fixtures/features.i32.bin and prints one JSON line. Memory is the resident set size
// (RSS) after a full garbage collection, in MiB: compiling the module, instantiating it, the first call, and the
// total from before compiling to after 61 calls. Times are in milliseconds; `medianMs` is over calls 2 to 61.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const dir = resolve(process.argv[2] ?? "pkg/web");
const root = new URL("../", import.meta.url);

if (!globalThis.Bun && !globalThis.gc) {
  console.error("Run Node.js with --expose-gc");
  process.exit(2);
}
const gc = globalThis.Bun ? () => Bun.gc(true) : () => globalThis.gc();
const rss = () => (gc(), process.memoryUsage().rss / 1048576);

const bytes = readFileSync(new URL("tests/fixtures/features.i32.bin", root));
const features = new Int32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const samples = features.length / 2048;
const sample = (i) => features.subarray((i % samples) * 2048, ((i % samples) + 1) * 2048);

const glue = await import(pathToFileURL(`${dir}/magika.js`).href);
const wasm = readFileSync(`${dir}/magika_bg.wasm`);

const before = rss();
let start = performance.now();
const module = new WebAssembly.Module(wasm);
const compileMs = performance.now() - start;
const compiled = rss();
glue.initSync({ module });
const model = new glue.MagikaModel();
const ready = rss();

start = performance.now();
model.scores(sample(0));
const firstMs = performance.now() - start;
const afterFirst = rss();

const times = [];
for (let i = 1; i <= 60; i++) {
  const t = performance.now();
  model.scores(sample(i));
  times.push(performance.now() - t);
}
times.sort((a, b) => a - b);
const after = rss();

const round = (n) => Math.round(n * 10) / 10;
console.log(
  JSON.stringify({
    runtime: globalThis.Bun ? `bun ${Bun.version}` : `node ${process.versions.node}`,
    wasmMiB: round(wasm.length / 1048576),
    compileMs: round(compileMs),
    compileMiB: round(compiled - before),
    instanceMiB: round(ready - compiled),
    firstMs: round(firstMs),
    firstCallMiB: round(afterFirst - ready),
    medianMs: round(times[30]),
    totalMiB: round(after - before),
  }),
);
