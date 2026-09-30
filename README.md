# magika-burn

[Magika](https://github.com/google/magika) is Google's deep-learning model for
detecting file content types. **magika-burn** runs it in pure Rust with
[Burn](https://burn.dev): no ONNX Runtime, no C++ and no Python. The same code
runs natively and as WebAssembly in browsers and Node.js.

This is an unofficial port and is not affiliated with Google. For the official
implementations, see [google/magika](https://github.com/google/magika).

## Installation

Rust (1.92 or later):

```sh
cargo add magika-burn
```

JavaScript:

```sh
npm install @yorganci/magika-burn
```

## Usage

### Rust

```rust no_run
use std::fs::File;

use magika_burn::Magika;

fn main() -> std::io::Result<()> {
    let magika = Magika::new();
    let prediction = magika.identify_reader(File::open("report.pdf")?)?;
    let info = prediction.info();
    println!("{} ({}), score {:.3}", info.label, info.mime_type, prediction.score);
    Ok(())
}
```

`identify_bytes` works on in-memory content. The prediction mode sets how
confident the model must be (see [Results](#results)), and content types can be
looked up without running the model:

```rust
use magika_burn::{ContentType, Magika, PredictionMode};

let magika = Magika::new().with_prediction_mode(PredictionMode::BestGuess);
let prediction = magika.identify_bytes(b"#!/bin/sh\necho 'Hello, world!'\n");
println!("{}: {}", prediction.output, prediction.info().description);

assert_eq!(magika.identify_bytes(b"").output, ContentType::Empty);

let pdf = ContentType::from_label("pdf").unwrap();
assert_eq!(pdf, ContentType::Pdf);
assert_eq!(pdf.info().mime_type, "application/pdf");
assert_eq!(pdf.info().extensions, ["pdf"]);
```

The `serde` feature implements `Serialize` and `Deserialize` for the public
types, with content types as their labels. A serialized `Prediction` has the same
shape as upstream's JSON output.

The crate also builds for `wasm32-unknown-unknown`. Enable WebAssembly SIMD with
`RUSTFLAGS="-C target-feature=+simd128"` for faster inference.

### Node.js

```js
import { readFileSync } from "node:fs";
import { MagikaModel } from "@yorganci/magika-burn";

const magika = new MagikaModel();
const { output, score } = magika.identifyBytes(readFileSync("report.pdf"));
console.log(output.label, output.mimeType, score);
```

### Browsers and bundlers

The browser build has to be initialized before use. `init()` fetches
`magika_bg.wasm` from next to the JavaScript module:

```js
import init, { MagikaModel } from "@yorganci/magika-burn";

await init();
const magika = new MagikaModel({ predictionMode: "high_confidence" });

input.addEventListener("change", async () => {
  const content = new Uint8Array(await input.files[0].arrayBuffer());
  const { output } = magika.identifyBytes(content);
  console.log(`${output.description} (${output.mimeType})`);
});
```

If your bundler doesn't handle `new URL("…", import.meta.url)`, pass the URL or
the bytes of `@yorganci/magika-burn/magika_bg.wasm` to
`init({ module_or_path })` instead.

Identifying a file takes tens of milliseconds, so run the model in a Web Worker
if you identify many files. `MagikaModel` holds WebAssembly memory: call
`magika.free()` when you are done with it, or declare it with `using`.

## Results

A prediction contains two content types:

| Rust               | JavaScript        | Meaning                                                                           |
| ------------------ | ----------------- | --------------------------------------------------------------------------------- |
| `dl`               | `dl`              | The model's prediction, or the `undefined` content type if the model was not run. |
| `output`           | `output`          | The final content type, after the overwrite map and the confidence threshold.     |
| `score`            | `score`           | The model's score for `dl`, or 1 if the model was not run.                        |
| `overwrite_reason` | `overwriteReason` | Why `output` differs from `dl`.                                                   |

In Rust `dl` and `output` are `ContentType`s, and `.info()` returns their
`ContentTypeInfo`. In JavaScript they are `ContentTypeInfo` objects:
`{ label, mimeType, group, description, extensions, isText }`.

The prediction mode sets the confidence threshold. Below it, `output` becomes
`txt` for text content types and `unknown` for binary ones:

| Mode                                             | Threshold                                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `HighConfidence` / `"high_confidence"` (default) | Per content type, e.g. 0.75 for `markdown` and 0.95 for `latex`, and 0.5 for the rest |
| `MediumConfidence` / `"medium_confidence"`       | 0.5                                                                                   |
| `BestGuess` / `"best_guess"`                     | None                                                                                  |

The overwrite reason is one of:

- `None`: `output` is `dl`.
- `LowConfidence`: the score is below the threshold.
- `OverwriteMap`: the model's label is replaced by a more generic one. The model
  recognizes random bytes and random text, which are reported as `unknown` and
  `txt`.

The model is not run for empty content (`empty`), or for content with fewer than
8 bytes besides leading and trailing whitespace. That content is `txt` if it is
valid UTF-8 and `unknown` otherwise.

The Rust crate also exposes the individual steps: `Features::extract`,
`Magika::scores` and `Prediction::from_scores`.

## How it works

1. **Features:** Magika reads up to 4 KiB from each end of the content. It strips
   whitespace and keeps the first and last 1024 bytes, padding shorter content.
2. **Model:** at build time, [`burn-onnx`](https://github.com/tracel-ai/burn-onnx)
   converts Magika's `standard_v3_3` ONNX model to Rust code that runs on Burn's
   Flex CPU backend. The weights are embedded in the binary.
3. **Content types:** the scores are resolved with the thresholds and the
   overwrite map from the model's configuration. MIME types and other information
   come from upstream's content types knowledge base.

The upstream files are vendored in [`assets/`](assets); see
[`assets/README.md`](assets/README.md). Compared to upstream Magika, this port
does not include the optional magic-byte rules. It identifies bytes and readers,
not paths, so there is no handling of directories or symbolic links.

## Performance

Identifying one file at a time on an Apple Silicon Mac:

| Implementation                      | Time per file |
| ----------------------------------- | ------------- |
| Upstream Magika (ONNX Runtime)      | ~2.5 ms       |
| magika-burn, native                 | ~16 ms        |
| magika-burn, WebAssembly in Node.js | ~32 ms        |

Most of the time is spent in element-wise operations, so there is room for
improvement. The WebAssembly module is about 5 MB (3.3 MB gzipped), mostly model
weights.

## Development

`nix develop` provides the Rust toolchain with the WebAssembly target,
`wasm-bindgen-cli`, Node.js and pnpm.

```sh
cargo test --all-features        # Rust tests, including upstream reference data
pnpm run build                   # build pkg/web and pkg/nodejs
pnpm test                        # test the WebAssembly build with Node.js
uv run scripts/prepare_model.py  # re-vendor upstream assets and regenerate fixtures
```

## License

Copyright 2026 Atahan Yorgancı. Licensed under the [Apache License 2.0](LICENSE).

The Magika model, its configuration, the content types knowledge base and the
upstream test data are © Google LLC and also licensed under the Apache License
2.0 ([`assets/LICENSE`](assets/LICENSE)). The model is modified as described in
[`assets/README.md`](assets/README.md).
