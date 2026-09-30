# Models

`standard_v3_3/model.onnx` is derived from Google's
[Magika](https://github.com/google/magika) model at commit
[`3f2cb85`](https://github.com/google/magika/tree/3f2cb8537dce123587bfb3535a402145adcba2c7/assets/models/standard_v3_3)
(SHA-256 of the original: `fe2d2eb49c5f88a9e0a6c048e15d6ffdf86235519c2afc535044de433169ec8c`).
It is licensed under the Apache License 2.0; see [`LICENSE`](./LICENSE).

## Modifications

The single `GlobalMaxPool` node is replaced by the equivalent
`ReduceMax(axes=[2], keepdims=1)` because `burn-onnx` does not support
`GlobalMaxPool`. Nothing else is changed.

Regenerate the model and the test fixtures with:

```sh
uv run scripts/prepare_model.py
```
