// Node.js entry point: loads the WebAssembly module synchronously, so that
// `MagikaModel` can be used right away. Browsers and bundlers use
// pkg/web/magika.js instead, which needs `await init()`.
import { readFileSync } from "node:fs";
import { initSync } from "./pkg/web/magika.js";

initSync({ module: readFileSync(new URL("./pkg/web/magika_bg.wasm", import.meta.url)) });

export { MagikaModel } from "./pkg/web/magika.js";
