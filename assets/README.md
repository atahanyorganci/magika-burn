# Assets

These files come from Google's [Magika](https://github.com/google/magika) at
commit [`a95a7a4`](https://github.com/google/magika/tree/a95a7a4e8fc5f9061a7adfec1b8d5f2ebbe42fe6)
and are licensed under the Apache License 2.0; see [`LICENSE`](./LICENSE).

| File | Upstream path |
| --- | --- |
| `models/standard_v3_3/model.onnx` | `assets/models/standard_v3_3/model.onnx` (modified, see below) |
| `models/standard_v3_3/config.min.json` | `assets/models/standard_v3_3/config.min.json` |
| `content_types_kb.min.json` | `assets/content_types_kb.min.json` |
| `../tests/fixtures/upstream/*.json.gz` | `tests_data/reference/*.json.gz` |

`build.rs` generates the Burn model from `model.onnx`, and the content types,
label list, thresholds and overwrite map from the two JSON files.

## Modifications

The single `GlobalMaxPool` node in `model.onnx` is replaced by the equivalent
`ReduceMax(axes=[2], keepdims=1)` because `burn-onnx` does not support
`GlobalMaxPool`. Nothing else is changed.

Regenerate the assets and the test fixtures with:

```sh
uv run scripts/prepare_model.py
```

The script verifies the SHA-256 checksum of every downloaded file.
