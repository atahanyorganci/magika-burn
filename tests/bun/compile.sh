#!/usr/bin/env bash
# Checks that a compiled Bun executable works without the package's files:
# builds tests/bun/compiled.mjs with `bun build --compile`, moves the executable
# to an empty directory and runs it there. Run `pnpm run build` first.
set -euo pipefail

cd "$(dirname "$0")/../.."
dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT

bun build --compile tests/bun/compiled.mjs --outfile "$dir/build/compiled" >/dev/null
mkdir "$dir/empty"
mv "$dir/build/compiled" "$dir/empty/"
rm -rf "$dir/build"

output="$(cd "$dir/empty" && ./compiled)"
if [[ $output != rust ]]; then
  echo "expected the compiled executable to print rust, got: $output" >&2
  exit 1
fi
echo "compiled Bun executable: ok"
