# Assets

These files come from Google's [Magika](https://github.com/google/magika) at
commit [`a95a7a4`](https://github.com/google/magika/tree/a95a7a4e8fc5f9061a7adfec1b8d5f2ebbe42fe6)
and are licensed under the Apache License 2.0; see [`LICENSE`](./LICENSE).

| File | Upstream path |
| --- | --- |
| `models/standard_v3_3/model.onnx` | `assets/models/standard_v3_3/model.onnx` |
| `models/standard_v3_3/weights.f32` | derived from `model.onnx`, see below |
| `models/standard_v3_3/config.min.json` | `assets/models/standard_v3_3/config.min.json` |
| `content_types_kb.min.json` | `assets/content_types_kb.min.json` |
| `../tests/fixtures/upstream/*.json.gz` | `tests_data/reference/*.json.gz` |

`src/model.rs` runs the model with the weights in `weights.f32`, and `build.rs`
generates the content types, label list, thresholds and overwrite map from the
two JSON files. `model.onnx` is only read by `scripts/prepare_model.py`, which
derives the weights and the test fixtures from it.

## Weights

`models/standard_v3_3/weights.f32` holds the model's ten weight tensors,
little-endian f32, in the order `scripts/prepare_model.py` lists them
(`WEIGHTS`): 3,136,856 bytes, SHA-256
`195256fffa2760df8024df8acabd047948a59b3a4618afd23b87dd26c870ea47`. The script
checks a NumPy forward pass over these weights against ONNX Runtime on the
upstream model before writing them.

Regenerate the assets and the test fixtures with:

```sh
uv run scripts/prepare_model.py
```

The script verifies the SHA-256 checksum of every downloaded file.
