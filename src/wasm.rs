//! JavaScript bindings, generated with `wasm-bindgen`.
//!
//! Predictions and content type information are returned as plain JavaScript
//! objects with camelCase fields; their TypeScript types are declared below.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::{ContentType, ContentTypeInfo, Magika, Prediction, PredictionMode};

#[wasm_bindgen(typescript_custom_section)]
const TYPES: &str = r#"
/** How confident the model must be for its prediction to be used. */
export type PredictionMode = "high_confidence" | "medium_confidence" | "best_guess";

/** Why `Prediction.output` differs from `Prediction.dl`. */
export type OverwriteReason = "none" | "low_confidence" | "overwrite_map";

export interface MagikaOptions {
    /** Defaults to `"high_confidence"`. */
    predictionMode?: PredictionMode;
}

/** Information about a content type. */
export interface ContentTypeInfo {
    /** Unique label, e.g. `"python"`. */
    label: string;
    /** MIME type, e.g. `"text/x-python"`. */
    mimeType: string;
    /** Group, e.g. `"code"`. */
    group: string;
    /** Human-readable description, e.g. `"Python source"`. */
    description: string;
    /** Common file extensions without the dot, e.g. `["py", "pyi"]`. */
    extensions: string[];
    /** Whether content of this type is text. */
    isText: boolean;
}

/** The identified content type. */
export interface Prediction {
    /** The model's content type; its label is `"undefined"` if the model was not run. */
    dl: ContentTypeInfo;
    /** The final content type, after the overwrite map and the confidence threshold. */
    output: ContentTypeInfo;
    /** The model's score for `dl`, or 1 if the model was not run. */
    score: number;
    /** Why `output` differs from `dl`. */
    overwriteReason: OverwriteReason;
}
"#;

#[wasm_bindgen]
extern "C" {
    /// `MagikaOptions` from the TypeScript declarations above.
    #[wasm_bindgen(typescript_type = "MagikaOptions")]
    pub type MagikaOptions;
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsContentTypeInfo {
    label: &'static str,
    mime_type: &'static str,
    group: &'static str,
    description: &'static str,
    extensions: &'static [&'static str],
    is_text: bool,
}

impl From<&'static ContentTypeInfo> for JsContentTypeInfo {
    fn from(info: &'static ContentTypeInfo) -> Self {
        Self {
            label: info.label,
            mime_type: info.mime_type,
            group: info.group,
            description: info.description,
            extensions: info.extensions,
            is_text: info.is_text,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsPrediction {
    dl: JsContentTypeInfo,
    output: JsContentTypeInfo,
    score: f32,
    overwrite_reason: &'static str,
}

impl From<Prediction> for JsPrediction {
    fn from(prediction: Prediction) -> Self {
        Self {
            dl: prediction.dl.info().into(),
            output: prediction.output.info().into(),
            score: prediction.score,
            overwrite_reason: prediction.overwrite_reason.as_str(),
        }
    }
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct JsOptions {
    prediction_mode: Option<JsPredictionMode>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum JsPredictionMode {
    HighConfidence,
    MediumConfidence,
    BestGuess,
}

impl From<JsPredictionMode> for PredictionMode {
    fn from(mode: JsPredictionMode) -> Self {
        match mode {
            JsPredictionMode::HighConfidence => Self::HighConfidence,
            JsPredictionMode::MediumConfidence => Self::MediumConfidence,
            JsPredictionMode::BestGuess => Self::BestGuess,
        }
    }
}

fn to_js(value: &impl Serialize) -> JsValue {
    serde_wasm_bindgen::to_value(value).expect("plain data serializes to JavaScript")
}

/// The Magika model with its weights loaded.
#[wasm_bindgen]
pub struct MagikaModel(Magika);

#[wasm_bindgen]
impl MagikaModel {
    /// Loads the embedded model weights.
    ///
    /// Throws if `options` is invalid.
    #[wasm_bindgen(constructor)]
    pub fn new(options: Option<MagikaOptions>) -> Result<MagikaModel, JsError> {
        let options: JsOptions = match options {
            Some(options) => serde_wasm_bindgen::from_value(options.into())?,
            None => JsOptions::default(),
        };
        let mode = options.prediction_mode.map(Into::into).unwrap_or_default();
        Ok(Self(Magika::new().with_prediction_mode(mode)))
    }

    /// The prediction mode used by `identifyBytes`.
    #[wasm_bindgen(getter = predictionMode, unchecked_return_type = "PredictionMode")]
    pub fn prediction_mode(&self) -> String {
        self.0.prediction_mode().as_str().to_owned()
    }

    /// Identifies the content type of `content`.
    #[wasm_bindgen(js_name = identifyBytes, unchecked_return_type = "Prediction")]
    pub fn identify_bytes(&self, content: &[u8]) -> JsValue {
        to_js(&JsPrediction::from(self.0.identify_bytes(content)))
    }

    /// Scores a batch of samples (low-level).
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

    /// Returns information about the content type with the given label, e.g.
    /// `"python"`, or `undefined` if there is none.
    #[wasm_bindgen(js_name = contentTypeInfo, unchecked_return_type = "ContentTypeInfo | undefined")]
    pub fn content_type_info(label: &str) -> JsValue {
        match ContentType::from_label(label) {
            Some(content_type) => to_js(&JsContentTypeInfo::from(content_type.info())),
            None => JsValue::UNDEFINED,
        }
    }

    /// Returns information about all content types, sorted by label.
    #[wasm_bindgen(js_name = contentTypes, unchecked_return_type = "ContentTypeInfo[]")]
    pub fn content_types() -> JsValue {
        let infos: Vec<JsContentTypeInfo> = ContentType::ALL
            .iter()
            .map(|content_type| content_type.info().into())
            .collect();
        to_js(&infos)
    }
}
