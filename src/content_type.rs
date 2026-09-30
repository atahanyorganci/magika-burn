//! Content types from Magika's knowledge base (`assets/content_types_kb.min.json`).

use std::fmt;

include!(concat!(env!("OUT_DIR"), "/content_types.rs"));

/// Information about a [`ContentType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[non_exhaustive]
pub struct ContentTypeInfo {
    /// Unique label, e.g. `"python"`.
    pub label: &'static str,
    /// MIME type, e.g. `"text/x-python"`.
    pub mime_type: &'static str,
    /// Group, e.g. `"code"`.
    pub group: &'static str,
    /// Human-readable description, e.g. `"Python source"`.
    pub description: &'static str,
    /// Common file extensions without the dot, e.g. `["py", "pyi"]`.
    pub extensions: &'static [&'static str],
    /// Whether content of this type is text.
    pub is_text: bool,
}

impl ContentType {
    /// Returns the unique label, e.g. `"python"`.
    pub fn label(self) -> &'static str {
        self.info().label
    }

    /// Returns the MIME type, description and other information.
    pub fn info(self) -> &'static ContentTypeInfo {
        &INFOS[self as usize]
    }
}

impl fmt::Display for ContentType {
    /// Writes the label.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Serializes as the label.
#[cfg(feature = "serde")]
impl serde::Serialize for ContentType {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.label())
    }
}

/// Deserializes from the label.
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ContentType {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let label = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        Self::from_label(&label)
            .ok_or_else(|| serde::de::Error::custom(format_args!("unknown content type {label:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_round_trip_in_sorted_order() {
        assert!(ContentType::ALL.is_sorted_by_key(|ct| ct.label()));
        for (index, &ct) in ContentType::ALL.iter().enumerate() {
            assert_eq!(ct as usize, index);
            assert_eq!(ContentType::from_label(ct.label()), Some(ct));
            assert_eq!(ct.to_string(), ct.label());
        }
        assert_eq!(ContentType::from_label("not-a-content-type"), None);
    }

    #[test]
    fn info() {
        assert_eq!(
            *ContentType::Python.info(),
            ContentTypeInfo {
                label: "python",
                mime_type: "text/x-python",
                group: "code",
                description: "Python source",
                extensions: &["py", "pyi"],
                is_text: true,
            }
        );
        assert_eq!(ContentType::_3gp.label(), "3gp");
        assert_eq!(ContentType::Rar.info().mime_type, "application/vnd.rar");
    }

    #[test]
    fn null_fields_use_upstream_defaults() {
        // `mime_type: null` depends on `is_text`.
        assert_eq!(ContentType::Solidity.info().mime_type, "text/plain");
        assert_eq!(
            ContentType::Sqlite.info().mime_type,
            "application/octet-stream"
        );
        assert_eq!(ContentType::Aidl.info().group, "unknown");
        assert_eq!(ContentType::Algol68.info().description, "algol68");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_uses_labels() {
        assert_eq!(
            serde_json::to_string(&ContentType::_3gp).unwrap(),
            r#""3gp""#
        );
        assert_eq!(
            serde_json::from_str::<ContentType>(r#""python""#).unwrap(),
            ContentType::Python
        );
        assert!(serde_json::from_str::<ContentType>(r#""nope""#).is_err());
        assert_eq!(
            serde_json::to_value(ContentType::Pdf.info()).unwrap(),
            serde_json::json!({
                "label": "pdf",
                "mime_type": "application/pdf",
                "group": "document",
                "description": "PDF document",
                "extensions": ["pdf"],
                "is_text": false,
            })
        );
    }
}
