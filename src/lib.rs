//! Google's [Magika](https://github.com/google/magika) content-type detection
//! model, compiled to native Rust with `burn-onnx` and run on Burn's Flex CPU
//! backend. Works natively and on `wasm32-unknown-unknown`, where the `wasm`
//! module provides JavaScript bindings.

mod config;
mod content_type;
#[allow(clippy::all, clippy::pedantic, dead_code, unused)]
mod model;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod wasm;

use std::fmt;

use burn::{
    backend::Flex,
    tensor::{Device, Int, Tensor, TensorData},
};

pub use crate::{
    config::{INPUT_SIZE, NUM_LABELS, PADDING_TOKEN},
    content_type::{ContentType, ContentTypeInfo},
};

type Backend = Flex;

/// Invalid model input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The number of tokens is not a positive multiple of [`INPUT_SIZE`].
    InvalidLength(usize),
    /// A token is outside `0..=PADDING_TOKEN`.
    InvalidToken { index: usize, token: i32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength(len) => write!(
                f,
                "expected a positive multiple of {INPUT_SIZE} tokens, got {len}"
            ),
            Self::InvalidToken { index, token } => write!(
                f,
                "token {token} at index {index} is outside 0..={PADDING_TOKEN}"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// The Magika model with its weights loaded.
pub struct Magika {
    model: model::Model<Backend>,
    device: Device<Backend>,
}

impl Magika {
    /// Loads the embedded model weights.
    pub fn new() -> Self {
        let device = Device::<Backend>::default();
        Self {
            model: model::Model::from_embedded(&device),
            device,
        }
    }

    /// Scores a batch of samples.
    ///
    /// `features` holds `n` samples of [`INPUT_SIZE`] tokens each, back to back.
    /// A token is a byte value (`0..=255`) or [`PADDING_TOKEN`]. Returns `n` rows
    /// of [`NUM_LABELS`] softmax scores, also back to back.
    pub fn scores(&self, features: &[i32]) -> Result<Vec<f32>, Error> {
        if features.is_empty() || !features.len().is_multiple_of(INPUT_SIZE) {
            return Err(Error::InvalidLength(features.len()));
        }
        if let Some((index, &token)) = features
            .iter()
            .enumerate()
            .find(|(_, token)| !(0..=PADDING_TOKEN).contains(*token))
        {
            return Err(Error::InvalidToken { index, token });
        }

        let batch = features.len() / INPUT_SIZE;
        let input = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(features.to_vec(), [batch, INPUT_SIZE]),
            &self.device,
        );
        let scores = self
            .model
            .forward(input)
            .into_data()
            .into_vec::<f32>()
            .expect("model output is f32");
        Ok(scores)
    }
}

impl Default for Magika {
    fn default() -> Self {
        Self::new()
    }
}
