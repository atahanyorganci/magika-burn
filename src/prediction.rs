//! Resolution of model scores to content types.

use std::fmt;

use crate::{
    ContentType, ContentTypeInfo,
    config::{LABELS, MEDIUM_CONFIDENCE_THRESHOLD, NUM_LABELS, OVERWRITE_MAP, THRESHOLDS},
};

/// How confident the model must be for its prediction to be used.
///
/// Below the threshold, [`Prediction::output`] falls back to [`ContentType::Txt`]
/// or [`ContentType::Unknown`], depending on whether the predicted type is text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum PredictionMode {
    /// Per-content-type thresholds, e.g. 0.75 for `markdown` and 0.95 for `latex`,
    /// and 0.5 for the rest.
    #[default]
    HighConfidence,
    /// A threshold of 0.5 for every content type.
    MediumConfidence,
    /// No threshold: always use the model's prediction.
    BestGuess,
}

impl PredictionMode {
    /// Returns the name used by upstream Magika, e.g. `"high_confidence"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HighConfidence => "high_confidence",
            Self::MediumConfidence => "medium_confidence",
            Self::BestGuess => "best_guess",
        }
    }
}

impl fmt::Display for PredictionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why [`Prediction::output`] differs from [`Prediction::dl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum OverwriteReason {
    /// Nothing was overwritten: `output` is `dl`, or the model was not run.
    None,
    /// The score is below the threshold of the [`PredictionMode`], so `output` is
    /// [`ContentType::Txt`] or [`ContentType::Unknown`]. Takes precedence over
    /// [`OverwriteReason::OverwriteMap`].
    LowConfidence,
    /// The model label is replaced by a more generic content type, e.g.
    /// `randombytes` by `unknown`.
    OverwriteMap,
}

impl OverwriteReason {
    /// Returns the name used by upstream Magika, e.g. `"low_confidence"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LowConfidence => "low_confidence",
            Self::OverwriteMap => "overwrite_map",
        }
    }
}

impl fmt::Display for OverwriteReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The identified content type.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Prediction {
    /// The model's label, or [`ContentType::Undefined`] if the model was not run
    /// (empty or tiny content).
    pub dl: ContentType,
    /// The final content type, after the overwrite map and the confidence threshold.
    pub output: ContentType,
    /// The model's score for `dl`, or 1.0 if the model was not run.
    pub score: f32,
    /// Why `output` differs from `dl`.
    pub overwrite_reason: OverwriteReason,
}

impl Prediction {
    /// Resolves one sample's model scores, e.g. a row of [`Magika::scores`].
    ///
    /// Takes the highest-scoring label, applies the overwrite map, then falls back
    /// to `txt` or `unknown` if the score is below the threshold of `mode`.
    ///
    /// [`Magika::scores`]: crate::Magika::scores
    pub fn from_scores(scores: &[f32; NUM_LABELS], mode: PredictionMode) -> Self {
        // Ties go to the last label, like upstream.
        let mut best = 0;
        for (index, &score) in scores.iter().enumerate() {
            if scores[best].max(score) == score {
                best = index;
            }
        }
        let score = scores[best];
        let dl = LABELS[best];

        let mut output = OVERWRITE_MAP[best];
        let mut overwrite_reason = if output == dl {
            OverwriteReason::None
        } else {
            OverwriteReason::OverwriteMap
        };

        let threshold = match mode {
            PredictionMode::HighConfidence => Some(THRESHOLDS[best]),
            PredictionMode::MediumConfidence => Some(MEDIUM_CONFIDENCE_THRESHOLD),
            PredictionMode::BestGuess => None,
        };
        // Compare in f64 like upstream, which compares with Python floats.
        if threshold.is_some_and(|threshold| f64::from(score) < threshold) {
            output = if output.info().is_text {
                ContentType::Txt
            } else {
                ContentType::Unknown
            };
            overwrite_reason = if output == dl {
                OverwriteReason::None
            } else {
                OverwriteReason::LowConfidence
            };
        }

        Self {
            dl,
            output,
            score,
            overwrite_reason,
        }
    }

    /// A prediction decided without the model, e.g. for empty content or a
    /// directory: `dl` is [`ContentType::Undefined`], the score is 1 and nothing
    /// is overwritten.
    ///
    /// ```
    /// use magika_burn::{ContentType, OverwriteReason, Prediction};
    ///
    /// let prediction = Prediction::ruled(ContentType::Directory);
    /// assert_eq!(prediction.dl, ContentType::Undefined);
    /// assert_eq!(prediction.output, ContentType::Directory);
    /// assert_eq!(prediction.score, 1.0);
    /// assert_eq!(prediction.overwrite_reason, OverwriteReason::None);
    /// ```
    pub fn ruled(output: ContentType) -> Self {
        Self {
            dl: ContentType::Undefined,
            output,
            score: 1.0,
            overwrite_reason: OverwriteReason::None,
        }
    }

    /// Shorthand for `self.output.info()`.
    pub fn info(&self) -> &'static ContentTypeInfo {
        self.output.info()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODES: [PredictionMode; 3] = [
        PredictionMode::HighConfidence,
        PredictionMode::MediumConfidence,
        PredictionMode::BestGuess,
    ];

    /// Scores where `label` wins with `score`, and the rest share the remainder.
    fn scores(label: ContentType, score: f32) -> [f32; NUM_LABELS] {
        let index = LABELS.iter().position(|&l| l == label).unwrap();
        let mut scores = [(1.0 - score) / (NUM_LABELS - 1) as f32; NUM_LABELS];
        scores[index] = score;
        scores
    }

    fn resolve(
        label: ContentType,
        score: f32,
        mode: PredictionMode,
    ) -> (ContentType, OverwriteReason) {
        let prediction = Prediction::from_scores(&scores(label, score), mode);
        assert_eq!(prediction.dl, label);
        assert_eq!(prediction.score, score);
        (prediction.output, prediction.overwrite_reason)
    }

    #[test]
    fn confident_predictions_are_kept() {
        for mode in MODES {
            assert_eq!(
                resolve(ContentType::Python, 0.99, mode),
                (ContentType::Python, OverwriteReason::None)
            );
        }
    }

    #[test]
    fn thresholds_depend_on_the_mode() {
        // Markdown needs 0.75 in HighConfidence mode, and is text.
        let (dl, score) = (ContentType::Markdown, 0.7);
        assert_eq!(
            resolve(dl, score, PredictionMode::HighConfidence),
            (ContentType::Txt, OverwriteReason::LowConfidence)
        );
        assert_eq!(
            resolve(dl, score, PredictionMode::MediumConfidence),
            (dl, OverwriteReason::None)
        );
        assert_eq!(
            resolve(dl, score, PredictionMode::BestGuess),
            (dl, OverwriteReason::None)
        );
    }

    #[test]
    fn low_confidence_binary_is_unknown() {
        assert_eq!(
            resolve(ContentType::Pdf, 0.3, PredictionMode::MediumConfidence),
            (ContentType::Unknown, OverwriteReason::LowConfidence)
        );
    }

    #[test]
    fn overwrite_map() {
        for mode in MODES {
            assert_eq!(
                resolve(ContentType::Randombytes, 0.9, mode),
                (ContentType::Unknown, OverwriteReason::OverwriteMap)
            );
            assert_eq!(
                resolve(ContentType::Randomtxt, 0.9, mode),
                (ContentType::Txt, OverwriteReason::OverwriteMap)
            );
        }
    }

    #[test]
    fn low_confidence_takes_precedence_over_overwrite_map() {
        assert_eq!(
            resolve(
                ContentType::Randombytes,
                0.3,
                PredictionMode::HighConfidence
            ),
            (ContentType::Unknown, OverwriteReason::LowConfidence)
        );
    }

    #[test]
    fn low_confidence_fallback_to_the_same_type_is_not_an_overwrite() {
        assert_eq!(
            resolve(ContentType::Txt, 0.3, PredictionMode::HighConfidence),
            (ContentType::Txt, OverwriteReason::None)
        );
    }

    #[test]
    fn names_match_upstream() {
        let modes = MODES.map(PredictionMode::as_str);
        assert_eq!(
            modes,
            ["high_confidence", "medium_confidence", "best_guess"]
        );
        assert_eq!(PredictionMode::default(), PredictionMode::HighConfidence);
        assert_eq!(OverwriteReason::LowConfidence.to_string(), "low_confidence");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_matches_upstream_reference_format() {
        let prediction = Prediction::from_scores(
            &scores(ContentType::Randombytes, 0.5),
            PredictionMode::HighConfidence,
        );
        let json = serde_json::json!({
            "dl": "randombytes",
            "output": "unknown",
            "score": 0.5,
            "overwrite_reason": "overwrite_map",
        });
        assert_eq!(serde_json::to_value(prediction).unwrap(), json);
        assert_eq!(
            serde_json::from_value::<Prediction>(json).unwrap(),
            prediction
        );
        assert_eq!(
            serde_json::from_str::<PredictionMode>(r#""best_guess""#).unwrap(),
            PredictionMode::BestGuess
        );
    }
}
