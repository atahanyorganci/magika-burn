#!/usr/bin/env bash
# Build the WebAssembly packages into pkg/<target>.
#
# Usage: scripts/build-wasm.sh [target...]
#   targets: web, nodejs, bundler, deno (default: web nodejs, which the npm package uses)
set -euo pipefail

cd "$(dirname "$0")/.."
targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=(web nodejs)
fi

cargo build --release --lib --target wasm32-unknown-unknown
for target in "${targets[@]}"; do
  wasm-bindgen \
    --target "$target" \
    --out-dir "pkg/$target" \
    --out-name magika \
    target/wasm32-unknown-unknown/release/magika_burn.wasm
  if [[ $target == nodejs ]]; then
    # wasm-bindgen emits CommonJS for Node.js, but package.json has "type": "module".
    echo '{ "type": "commonjs" }' >"pkg/$target/package.json"
  fi
done
