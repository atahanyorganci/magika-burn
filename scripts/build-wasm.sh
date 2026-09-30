#!/usr/bin/env bash
# Build the WebAssembly package into pkg/web. Browsers and bundlers load it with
# `init()`, Node.js through node.js, which calls `initSync()`.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo build --release --lib --target wasm32-unknown-unknown
wasm-bindgen \
  --target web \
  --out-dir pkg/web \
  --out-name magika \
  target/wasm32-unknown-unknown/release/magika_burn.wasm
