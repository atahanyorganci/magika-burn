//! JavaScript bindings, generated with `wasm-bindgen`.

use wasm_bindgen::prelude::*;

use crate::Magika;

/// The Magika model with its weights loaded.
#[wasm_bindgen]
pub struct MagikaModel(Magika);

#[wasm_bindgen]
impl MagikaModel {
    /// Loads the embedded model weights.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self(Magika::new())
    }

    /// Scores a batch of samples.
    ///
    /// `features` holds `n` samples of 2048 tokens each, back to back. A token is
    /// a byte value (0-255) or the padding token 256. Returns `n` rows of 214
    /// softmax scores, also back to back.
    ///
    /// Throws if `features.length` is not a positive multiple of 2048 or if a
    /// token is out of range.
    pub fn scores(&self, features: &[i32]) -> Result<Vec<f32>, JsError> {
        Ok(self.0.scores(features)?)
    }
}

impl Default for MagikaModel {
    fn default() -> Self {
        Self::new()
    }
}
