// Checks the WebAssembly build against ONNX Runtime outputs of the upstream model.
//
// Usage: scripts/build-wasm.sh nodejs && node --test 'tests/wasm/*.test.mjs'

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const { MagikaModel } = require("../../pkg/nodejs/magika.js");

const INPUT_SIZE = 2048;
const NUM_LABELS = 214;
const PADDING_TOKEN = 256;

function fixture(name, Type) {
  const { buffer, byteOffset, byteLength } = readFileSync(
    new URL(`../fixtures/${name}`, import.meta.url),
  );
  return new Type(buffer.slice(byteOffset, byteOffset + byteLength));
}

const model = new MagikaModel();

test("matches onnxruntime", () => {
  const features = fixture("features.i32.bin", Int32Array);
  const expected = fixture("scores.f32.bin", Float32Array);
  assert.equal(features.length / INPUT_SIZE, expected.length / NUM_LABELS);

  const scores = model.scores(features);

  assert.ok(scores instanceof Float32Array);
  assert.equal(scores.length, expected.length);
  let maxError = 0;
  for (let i = 0; i < scores.length; i++) {
    maxError = Math.max(maxError, Math.abs(scores[i] - expected[i]));
  }
  assert.ok(maxError < 1e-5, `max abs error ${maxError}`);
});

test("throws on invalid length", () => {
  assert.throws(() => model.scores(new Int32Array(0)), /multiple of 2048/);
  assert.throws(
    () => model.scores(new Int32Array(INPUT_SIZE + 1)),
    /multiple of 2048/,
  );
});

test("throws on invalid token", () => {
  const features = new Int32Array(INPUT_SIZE).fill(PADDING_TOKEN);
  features[7] = PADDING_TOKEN + 1;
  assert.throws(() => model.scores(features), /token 257 at index 7/);
});
