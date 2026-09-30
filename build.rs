use std::{collections::BTreeMap, env, fmt::Write as _, fs, path::PathBuf};

use serde::Deserialize;

const CONFIG: &str = "assets/models/standard_v3_3/config.min.json";
const KNOWLEDGE_BASE: &str = "assets/content_types_kb.min.json";

/// Content types the crate refers to by name.
const REQUIRED_CONTENT_TYPES: [&str; 4] = ["empty", "txt", "undefined", "unknown"];

/// An entry of `content_types_kb.min.json`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KnowledgeBaseEntry {
    mime_type: Option<String>,
    group: Option<String>,
    description: Option<String>,
    extensions: Vec<String>,
    is_text: bool,
}

/// `config.min.json`. Unknown fields are rejected so that format changes are noticed.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    beg_size: usize,
    mid_size: usize,
    end_size: usize,
    use_inputs_at_offsets: bool,
    medium_confidence_threshold: f64,
    min_file_size_for_dl: usize,
    padding_token: i32,
    block_size: usize,
    target_labels_space: Vec<String>,
    thresholds: BTreeMap<String, f64>,
    overwrite_map: BTreeMap<String, String>,
    protection: String,
    aes_key_hex: String,
    version_major: u32,
}

fn main() {
    for path in [CONFIG, KNOWLEDGE_BASE] {
        println!("cargo:rerun-if-changed={path}");
    }

    let knowledge_base: BTreeMap<String, KnowledgeBaseEntry> = read_json(KNOWLEDGE_BASE);
    let config: ModelConfig = read_json(CONFIG);
    validate(&config, &knowledge_base);

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::write(
        out_dir.join("content_types.rs"),
        content_types(&knowledge_base),
    )
    .unwrap();
    fs::write(out_dir.join("config.rs"), model_config(&config)).unwrap();
}

fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> T {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_reader(file).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn validate(config: &ModelConfig, knowledge_base: &BTreeMap<String, KnowledgeBaseEntry>) {
    let known = |label: &str| knowledge_base.contains_key(label);
    let is_model_label = |label: &str| config.target_labels_space.iter().any(|l| l == label);

    assert_eq!(config.version_major, 3, "unsupported config version");
    assert_eq!(config.mid_size, 0, "mid_size is not supported");
    assert!(
        !config.use_inputs_at_offsets,
        "use_inputs_at_offsets is not supported"
    );
    assert_eq!(
        config.protection, "none",
        "protected models are not supported"
    );
    assert!(
        config.aes_key_hex.is_empty(),
        "protected models are not supported"
    );
    assert!(config.beg_size <= config.block_size && config.end_size <= config.block_size);
    assert!((1..=config.beg_size).contains(&config.min_file_size_for_dl));
    assert!(config.padding_token > i32::from(u8::MAX));

    for label in REQUIRED_CONTENT_TYPES {
        assert!(known(label), "{label:?} is missing from the knowledge base");
    }
    for label in &config.target_labels_space {
        assert!(
            known(label),
            "model label {label:?} is missing from the knowledge base"
        );
    }
    let thresholds = config.thresholds.values();
    for threshold in thresholds.chain([&config.medium_confidence_threshold]) {
        assert!(
            (0.0..=1.0).contains(threshold),
            "threshold {threshold} is out of range"
        );
    }
    for label in config.thresholds.keys() {
        assert!(
            is_model_label(label),
            "threshold for unknown model label {label:?}"
        );
    }
    for (from, to) in &config.overwrite_map {
        assert!(
            is_model_label(from),
            "overwrite of unknown model label {from:?}"
        );
        assert!(
            known(to),
            "overwrite to {to:?}, which is missing from the knowledge base"
        );
    }
}

/// Rust identifier of a content type: `python` -> `Python`,
/// `llvm_bitcode` -> `LlvmBitcode`, `3gp` -> `_3gp`.
fn variant(label: &str) -> String {
    assert!(
        !label.is_empty()
            && label.split('_').all(|part| !part.is_empty()
                && part
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())),
        "unsupported label {label:?}"
    );
    let prefix = if label.starts_with(|c: char| c.is_ascii_digit()) {
        "_"
    } else {
        ""
    };
    let camel: String = label
        .split('_')
        .map(|part| part[..1].to_ascii_uppercase() + &part[1..])
        .collect();
    prefix.to_owned() + &camel
}

fn content_types(knowledge_base: &BTreeMap<String, KnowledgeBaseEntry>) -> String {
    let mut names = std::collections::BTreeSet::new();
    let mut variants = String::new();
    let mut all = String::new();
    let mut from_label = String::new();
    let mut infos = String::new();
    for (label, entry) in knowledge_base {
        let variant = variant(label);
        assert!(names.insert(variant.clone()), "duplicate variant {variant}");
        // Same defaults as the upstream Python package.
        let mime_type = entry.mime_type.as_deref().unwrap_or(if entry.is_text {
            "text/plain"
        } else {
            "application/octet-stream"
        });
        let group = entry.group.as_deref().unwrap_or("unknown");
        let description = entry.description.as_deref().unwrap_or(label);

        let doc = format!("{description} (`{label}`, `{mime_type}`).");
        writeln!(variants, "    #[doc = {doc:?}]\n    {variant},").unwrap();
        writeln!(all, "        ContentType::{variant},").unwrap();
        writeln!(
            from_label,
            "            {label:?} => Some(ContentType::{variant}),"
        )
        .unwrap();
        writeln!(
            infos,
            "    ContentTypeInfo {{ label: {label:?}, mime_type: {mime_type:?}, group: {group:?}, \
             description: {description:?}, extensions: &{:?}, is_text: {} }},",
            entry.extensions, entry.is_text,
        )
        .unwrap();
    }
    let count = knowledge_base.len();

    format!(
        r#"// Generated by build.rs from {KNOWLEDGE_BASE}.

/// A content type from Magika's knowledge base.
///
/// Variants are named after their label in UpperCamelCase, with a leading `_` when
/// the label starts with a digit: `python` is [`ContentType::Python`],
/// `llvm_bitcode` is [`ContentType::LlvmBitcode`] and `3gp` is [`ContentType::_3gp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum ContentType {{
{variants}}}

impl ContentType {{
    /// All content types, sorted by label.
    pub const ALL: &'static [ContentType] = &[
{all}    ];

    /// Returns the content type with the given label, e.g. `"python"`.
    pub fn from_label(label: &str) -> Option<Self> {{
        match label {{
{from_label}            _ => None,
        }}
    }}
}}

/// Information about each content type, indexed by `ContentType as usize`.
static INFOS: [ContentTypeInfo; {count}] = [
{infos}];
"#
    )
}

fn model_config(config: &ModelConfig) -> String {
    let input_size = config.beg_size + config.mid_size + config.end_size;
    let num_labels = config.target_labels_space.len();
    let padding_token = config.padding_token;
    let medium_confidence_threshold = config.medium_confidence_threshold;
    let block_size = config.block_size;
    let min_file_size_for_dl = config.min_file_size_for_dl;

    let mut labels = String::new();
    let mut thresholds = String::new();
    let mut overwrite_map = String::new();
    for label in &config.target_labels_space {
        let threshold = config
            .thresholds
            .get(label)
            .unwrap_or(&config.medium_confidence_threshold);
        let overwrite = config.overwrite_map.get(label).unwrap_or(label);
        writeln!(labels, "    ContentType::{},", variant(label)).unwrap();
        writeln!(thresholds, "    {threshold:?},").unwrap();
        writeln!(overwrite_map, "    ContentType::{},", variant(overwrite)).unwrap();
    }

    format!(
        r#"// Generated by build.rs from {CONFIG}.

/// Number of tokens in one sample: the first {beg} and the last {end} bytes of a file.
pub const INPUT_SIZE: usize = {input_size};
/// Token used to pad samples of files shorter than [`INPUT_SIZE`] bytes.
pub const PADDING_TOKEN: i32 = {padding_token};
/// Number of scores (one per model label) the model returns per sample.
pub const NUM_LABELS: usize = {num_labels};

/// Number of tokens taken from the beginning of the content.
pub(crate) const BEG_SIZE: usize = {beg};
/// Number of tokens taken from the end of the content.
pub(crate) const END_SIZE: usize = {end};
/// Number of bytes read from each end of the content before stripping whitespace.
pub(crate) const BLOCK_SIZE: usize = {block_size};
/// Minimum content size, without surrounding whitespace, for the model to be run.
pub(crate) const MIN_FILE_SIZE_FOR_DL: usize = {min_file_size_for_dl};

/// Threshold of `PredictionMode::MediumConfidence`, and the default threshold of
/// `PredictionMode::HighConfidence`.
pub(crate) const MEDIUM_CONFIDENCE_THRESHOLD: f64 = {medium_confidence_threshold:?};

/// Content type of each model output, by index.
pub(crate) const LABELS: [ContentType; NUM_LABELS] = [
{labels}];

/// `PredictionMode::HighConfidence` threshold of each model output, by index.
pub(crate) const THRESHOLDS: [f64; NUM_LABELS] = [
{thresholds}];

/// Content type that replaces each model output (usually itself), by index.
pub(crate) const OVERWRITE_MAP: [ContentType; NUM_LABELS] = [
{overwrite_map}];
"#,
        beg = config.beg_size,
        end = config.end_size,
    )
}
