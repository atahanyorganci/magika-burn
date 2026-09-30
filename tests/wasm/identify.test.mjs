// Checks content type identification in the WebAssembly build, including
// upstream Magika's reference predictions.
//
// Usage: pnpm run build && pnpm test

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { gunzipSync } from "node:zlib";
import { MagikaModel } from "@yorganci/magika-burn";

const MODES = ["high_confidence", "medium_confidence", "best_guess"];

test("matches upstream predictions", () => {
  const examples = JSON.parse(
    gunzipSync(
      readFileSync(
        new URL(
          "../fixtures/upstream/standard_v3_3-inference_examples_by_content.json.gz",
          import.meta.url,
        ),
      ),
    ),
  );
  assert.ok(examples.length > 0);
  const models = Object.fromEntries(
    MODES.map((predictionMode) => [predictionMode, new MagikaModel({ predictionMode })]),
  );

  for (const example of examples) {
    const content = Buffer.from(example.content_base64, "base64");
    const expected = example.prediction;
    const context = `${example.prediction_mode}: ${JSON.stringify(content.toString("latin1").slice(0, 40))}`;

    const actual = models[example.prediction_mode].identifyBytes(content);

    assert.equal(actual.dl.label, expected.dl, context);
    assert.equal(actual.output.label, expected.output, context);
    assert.equal(actual.overwriteReason, expected.overwrite_reason, context);
    assert.ok(Math.abs(actual.score - expected.score) < 1e-4, context);
  }
});

test("returns content type information", () => {
  const model = new MagikaModel();
  assert.deepEqual(model.identifyBytes(new Uint8Array()), {
    dl: MagikaModel.contentTypeInfo("undefined"),
    output: {
      label: "empty",
      mimeType: "inode/x-empty",
      group: "inode",
      description: "Empty file",
      extensions: [],
      isText: false,
    },
    score: 1,
    overwriteReason: "none",
  });
});

test("looks up content types", () => {
  assert.deepEqual(MagikaModel.contentTypeInfo("pdf"), {
    label: "pdf",
    mimeType: "application/pdf",
    group: "document",
    description: "PDF document",
    extensions: ["pdf"],
    isText: false,
  });
  assert.equal(MagikaModel.contentTypeInfo("not-a-content-type"), undefined);

  const all = MagikaModel.contentTypes();
  assert.equal(all.length, 415);
  assert.deepEqual(
    all.map((info) => info.label),
    all.map((info) => info.label).toSorted(),
  );
});

test("prediction mode option", () => {
  assert.equal(new MagikaModel().predictionMode, "high_confidence");
  assert.equal(new MagikaModel({}).predictionMode, "high_confidence");
  assert.equal(
    new MagikaModel({ predictionMode: "best_guess" }).predictionMode,
    "best_guess",
  );
  assert.throws(
    () => new MagikaModel({ predictionMode: "nope" }),
    /invalid predictionMode "nope": expected "high_confidence", "medium_confidence" or "best_guess"/,
  );
  assert.throws(() => new MagikaModel({ predictionMode: 1 }), /invalid predictionMode: expected/);
  assert.equal(new MagikaModel({ predictionMode: undefined }).predictionMode, "high_confidence");
  assert.equal(new MagikaModel(null).predictionMode, "high_confidence");
});
