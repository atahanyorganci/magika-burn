// Checks that the browser build can be loaded from Node.js through the `./web`
// export, as Bun executables do (see tests/bun/compile.sh).
//
// Usage: pnpm run build && pnpm test

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import init, { initSync, MagikaModel } from "@yorganci/magika-burn/web";

test("loads the web build with initSync", () => {
  assert.equal(typeof init, "function");
  const wasm = fileURLToPath(import.meta.resolve("@yorganci/magika-burn/magika_bg.wasm"));
  initSync({ module: readFileSync(wasm) });
  const model = new MagikaModel();
  assert.equal(model.identifyBytes(new Uint8Array()).output.label, "empty");
  assert.equal(model.identifyBytes(new TextEncoder().encode("hello\n")).output.label, "txt");
});
