#!/usr/bin/env bash
# Build the WebAssembly package into pkg/<target>.
#
# Usage: scripts/build-wasm.sh [web|nodejs|bundler|deno]  (default: web)
set -euo pipefail

cd "$(dirname "$0")/.."
target="${1:-web}"

cargo build --release --lib --target wasm32-unknown-unknown
wasm-bindgen \
  --target "$target" \
  --out-dir "pkg/$target" \
  --out-name magika \
  target/wasm32-unknown-unknown/release/magika_burn.wasm
