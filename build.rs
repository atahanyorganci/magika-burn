use burn_onnx::{LoadStrategy, ModelGen};

const MODEL: &str = "assets/models/standard_v3_3/model.onnx";

fn main() {
    println!("cargo:rerun-if-changed={MODEL}");

    ModelGen::new()
        .input(MODEL)
        .out_dir("model/")
        // Browsers have no filesystem, so the weights (~3 MB) are baked into the binary.
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();
}
