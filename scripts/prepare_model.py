# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "onnx", "onnxruntime"]
# ///
"""Vendor the upstream Magika model and generate reference test fixtures.

Usage: uv run scripts/prepare_model.py

1. Downloads `standard_v3_3/model.onnx` from google/magika at a pinned commit and
   verifies its SHA-256.
2. Replaces `GlobalMaxPool` (not supported by burn-onnx) with the equivalent
   `ReduceMax(axes=[2], keepdims=1)` and writes the result to `assets/models/`.
3. Runs the *original* model with ONNX Runtime on deterministic inputs and writes
   the inputs/outputs to `tests/fixtures/`, so the Rust and WASM tests check the
   Burn port (including the patch) against the upstream model.
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

COMMIT = "3f2cb8537dce123587bfb3535a402145adcba2c7"
MODEL = "standard_v3_3"
BASE_URL = f"https://raw.githubusercontent.com/google/magika/{COMMIT}"
SHA256 = "fe2d2eb49c5f88a9e0a6c048e15d6ffdf86235519c2afc535044de433169ec8c"

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "assets" / "models" / MODEL
FIXTURES_DIR = ROOT / "tests" / "fixtures"

INPUT_SIZE = 2048  # beg_size (1024) + mid_size (0) + end_size (1024)
PADDING_TOKEN = 256


def download(path: str) -> bytes:
    with urllib.request.urlopen(f"{BASE_URL}/{path}") as response:
        return response.read()


def patch(model: onnx.ModelProto) -> onnx.ModelProto:
    patched = onnx.ModelProto()
    patched.CopyFrom(model)
    nodes = patched.graph.node
    for index, node in enumerate(nodes):
        if node.op_type != "GlobalMaxPool":
            continue
        # Input is [N, C, L]; GlobalMaxPool -> [N, C, 1] == ReduceMax over axis 2.
        # Opset 15 still takes `axes` as an attribute.
        replacement = helper.make_node(
            "ReduceMax",
            inputs=list(node.input),
            outputs=list(node.output),
            name=node.name or f"global_max_pool_{index}",
            axes=[2],
            keepdims=1,
        )
        nodes.remove(node)
        nodes.insert(index, replacement)
    assert not any(n.op_type == "GlobalMaxPool" for n in patched.graph.node)
    onnx.checker.check_model(patched)
    return patched


def fixtures(model_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    features = rng.integers(0, PADDING_TOKEN + 1, size=(4, INPUT_SIZE), dtype=np.int32)
    features[1, 1500:] = PADDING_TOKEN  # short file: padded tail of `beg`, head of `end`
    features[2, :] = PADDING_TOKEN  # all padding
    features[3, :] = rng.integers(0x20, 0x7F, size=INPUT_SIZE)  # printable ASCII
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    (scores,) = session.run(None, {"bytes": features})
    return features, scores.astype(np.float32)


def main() -> None:
    original = download(f"assets/models/{MODEL}/model.onnx")
    digest = hashlib.sha256(original).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch: expected {SHA256}, got {digest}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx.save(patch(onnx.load_from_string(original)), MODEL_DIR / "model.onnx")
    (MODEL_DIR.parent / "LICENSE").write_bytes(download("LICENSE"))

    features, scores = fixtures(original)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    features.astype("<i4").tofile(FIXTURES_DIR / "features.i32.bin")
    scores.astype("<f4").tofile(FIXTURES_DIR / "scores.f32.bin")
    print(f"wrote {MODEL_DIR / 'model.onnx'} and {len(features)} fixtures")


if __name__ == "__main__":
    main()
