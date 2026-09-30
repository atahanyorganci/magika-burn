# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "onnx", "onnxruntime"]
# ///
"""Vendor the upstream Magika model assets and generate test fixtures.

Usage: uv run scripts/prepare_model.py

1. Downloads the model, its config, the content types knowledge base, the license
   and upstream reference test data from google/magika at a pinned commit, and
   verifies their SHA-256 checksums.
2. Replaces `GlobalMaxPool` (not supported by burn-onnx) with the equivalent
   `ReduceMax(axes=[2], keepdims=1)` in the model.
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

COMMIT = "a95a7a4e8fc5f9061a7adfec1b8d5f2ebbe42fe6"
BASE_URL = f"https://raw.githubusercontent.com/google/magika/{COMMIT}"
SHA256 = {
    "assets/models/standard_v3_3/model.onnx": "fe2d2eb49c5f88a9e0a6c048e15d6ffdf86235519c2afc535044de433169ec8c",
    "assets/models/standard_v3_3/config.min.json": "ae24c742205358f6ff6dfd5facb6743fb69743dbba8373e73da58ff0cbd695db",
    "assets/content_types_kb.min.json": "b10630d9e7303fc5c87f8f6f6d28a8a66e5d5b843e44b9996cddd6583fb7419c",
    "LICENSE": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
    "tests_data/reference/standard_v3_3-inference_examples_by_content.json.gz": "3eda361e3d7290457bc859c02d491a5d94fcda3df75f1c53147adda57c0f756f",
    "tests_data/reference/features_extraction_examples.json.gz": "f8c78b07f3070089799f97d530658744604970688a15e25479cf9c3afbe49e41",
}
MODEL = "assets/models/standard_v3_3/model.onnx"
# Upstream path -> vendored path (relative to the repository root). The model is
# written separately because it is patched.
VERBATIM = {
    "assets/models/standard_v3_3/config.min.json": "assets/models/standard_v3_3/config.min.json",
    "assets/content_types_kb.min.json": "assets/content_types_kb.min.json",
    "LICENSE": "assets/LICENSE",
    "tests_data/reference/standard_v3_3-inference_examples_by_content.json.gz": "tests/fixtures/upstream/standard_v3_3-inference_examples_by_content.json.gz",
    "tests_data/reference/features_extraction_examples.json.gz": "tests/fixtures/upstream/features_extraction_examples.json.gz",
}

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "tests" / "fixtures"

INPUT_SIZE = 2048  # beg_size (1024) + mid_size (0) + end_size (1024)
PADDING_TOKEN = 256


def download(path: str) -> bytes:
    with urllib.request.urlopen(f"{BASE_URL}/{path}") as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256[path]:
        raise SystemExit(f"{path}: expected SHA-256 {SHA256[path]}, got {digest}")
    return data


def write(path: str, data: bytes) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


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
    original = download(MODEL)
    (ROOT / MODEL).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(patch(onnx.load_from_string(original)), ROOT / MODEL)

    for upstream, vendored in VERBATIM.items():
        write(vendored, download(upstream))

    features, scores = fixtures(original)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    features.astype("<i4").tofile(FIXTURES_DIR / "features.i32.bin")
    scores.astype("<f4").tofile(FIXTURES_DIR / "scores.f32.bin")
    print(f"vendored {len(VERBATIM) + 1} files from google/magika@{COMMIT[:7]}")


if __name__ == "__main__":
    main()
