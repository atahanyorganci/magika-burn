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
4. Writes the model's weights, little-endian f32 in the order `WEIGHTS` lists them,
   to `weights.f32`, after checking the forward pass they are meant for (`forward`,
   in NumPy) against the ONNX Runtime outputs.
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

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
WEIGHTS_FILE = "assets/models/standard_v3_3/weights.f32"

INPUT_SIZE = 2048  # beg_size (1024) + mid_size (0) + end_size (1024)
PADDING_TOKEN = 256

P = "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/"
# Name, ONNX initializer and the shape the forward pass reads it in, in file order.
WEIGHTS = [
    ("emb", "jax2tf_get_logits_/Const:0", (257, 64)),  # the one-hot matmul's matrix
    ("b0", P + "Dense_0/Reshape:0", (64,)),
    ("ln0_s", P + "LayerNorm_0/Reshape_2:0", (512,)),  # scale, per position
    ("ln0_b", P + "LayerNorm_0/Reshape_3:0", (512,)),  # bias, per position
    ("conv_w", P + "Conv_0/transpose_3:0", (512, 256, 5)),  # [out][in][width]
    ("conv_b", "const_fold_opt__209", (512,)),
    ("ln1_s", P + "LayerNorm_1/Reshape_2:0", (512,)),
    ("ln1_b", P + "LayerNorm_1/Reshape_3:0", (512,)),
    ("w1", "jax2tf_get_logits_/Const_24:0", (512, 214)),
    ("b1", P + "Dense_1/Reshape:0", (214,)),
]


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


def weights(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    initializers = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    return {
        name: initializers[onnx_name].reshape(shape).astype("<f4")
        for name, onnx_name, shape in WEIGHTS
    }


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU, tanh approximation, in the ONNX graph's order of operations."""
    return x * (0.5 * (1 + np.tanh(0.7978846 * (x + 0.044715 * x * x * x))))


def forward(w: dict[str, np.ndarray], tokens: np.ndarray) -> np.ndarray:
    """One sample: 2048 tokens (0-255, 256 for padding) to 214 softmax scores."""
    x = gelu(w["emb"][tokens] + w["b0"])  # embedding: [2048, 64]
    x = x.reshape(512, 256)  # [positions, channels]
    mean = x.mean(0)
    var = np.maximum(0, (x * x).mean(0) - mean * mean)
    x = (x - mean) * (1 / np.sqrt(var + 1e-6))[None, :] * w["ln0_s"][:, None] + w["ln0_b"][:, None]
    xc = x.T  # [channels, positions]
    y = np.zeros((512, 508), np.float32)
    for k in range(5):  # convolution, width 5, no padding
        y += w["conv_w"][:, :, k] @ xc[:, k : k + 508]
    y = gelu(y + w["conv_b"][:, None]).max(1)  # [512]
    mean = y.mean()
    var = max(0, (y * y).mean() - mean * mean)
    y = (y - mean) * (1 / np.sqrt(var + 1e-6)) * w["ln1_s"] + w["ln1_b"]
    z = y @ w["w1"] + w["b1"]
    z = np.exp(z - z.max())
    return z / z.sum()


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

    w = weights(onnx.load_from_string(original))
    error = np.abs(np.stack([forward(w, f) for f in features]) - scores).max()
    print(f"NumPy forward pass vs ONNX Runtime: max abs error {error:.1e}")
    if not error < 1e-5:
        raise SystemExit("the NumPy forward pass does not match ONNX Runtime")
    blob = b"".join(np.ascontiguousarray(w[name]).tobytes() for name, _, _ in WEIGHTS)
    write(WEIGHTS_FILE, blob)
    print(f"wrote {WEIGHTS_FILE}: {len(blob)} bytes, SHA-256 {hashlib.sha256(blob).hexdigest()}")
    print(f"vendored {len(VERBATIM) + 1} files from google/magika@{COMMIT[:7]}")


if __name__ == "__main__":
    main()
