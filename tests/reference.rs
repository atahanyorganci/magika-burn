//! Checks identification end to end against upstream Magika's reference
//! predictions (`tests_data/reference` in google/magika).

use std::io::Cursor;

use base64::prelude::*;
use flate2::read::GzDecoder;
use magika_burn::{Magika, PredictionMode};

const EXAMPLES: &[u8] =
    include_bytes!("fixtures/upstream/standard_v3_3-inference_examples_by_content.json.gz");

#[test]
fn matches_upstream_predictions() {
    let examples: Vec<serde_json::Value> =
        serde_json::from_reader(GzDecoder::new(EXAMPLES)).unwrap();
    assert!(!examples.is_empty());
    let magikas = [
        PredictionMode::HighConfidence,
        PredictionMode::MediumConfidence,
        PredictionMode::BestGuess,
    ]
    .map(|mode| Magika::new().with_prediction_mode(mode));

    for example in &examples {
        assert_eq!(example["status"], "ok");
        let mode = example["prediction_mode"].as_str().unwrap();
        let magika = magikas
            .iter()
            .find(|magika| magika.prediction_mode().as_str() == mode)
            .unwrap();
        let content = BASE64_STANDARD
            .decode(example["content_base64"].as_str().unwrap())
            .unwrap();
        let expected = &example["prediction"];

        let actual = magika.identify_bytes(&content);

        let context = format!("{mode}: {:?}", content.escape_ascii().to_string());
        assert_eq!(actual.dl.label(), expected["dl"], "{context}");
        assert_eq!(actual.output.label(), expected["output"], "{context}");
        assert_eq!(
            actual.overwrite_reason.as_str(),
            expected["overwrite_reason"],
            "{context}"
        );
        let expected_score = expected["score"].as_f64().unwrap();
        assert!(
            (f64::from(actual.score) - expected_score).abs() < 1e-4,
            "{context}: score {} != {expected_score}",
            actual.score
        );

        let from_reader = magika.identify_reader(Cursor::new(&content)).unwrap();
        assert_eq!(from_reader, actual, "{context}");
    }
}
