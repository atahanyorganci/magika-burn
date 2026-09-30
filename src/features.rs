//! Feature extraction: turns content into model input.

use std::io::{self, Read, Seek, SeekFrom};

use crate::{
    ContentType,
    config::{BEG_SIZE, BLOCK_SIZE, END_SIZE, MIN_FILE_SIZE_FOR_DL, PADDING_TOKEN},
};

/// Model input extracted from content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Features {
    /// [`INPUT_SIZE`] tokens for [`Magika::scores`]: the first and the last bytes
    /// of the content without surrounding whitespace, padded with
    /// [`PADDING_TOKEN`].
    ///
    /// [`INPUT_SIZE`]: crate::INPUT_SIZE
    /// [`Magika::scores`]: crate::Magika::scores
    Tokens(Vec<i32>),
    /// The content is too small for the model, so its type is decided directly:
    /// [`ContentType::Empty`], or [`ContentType::Txt`] if it is valid UTF-8 and
    /// [`ContentType::Unknown`] otherwise.
    Ruled(ContentType),
}

/// Parameters of feature extraction; only the reference tests use other values.
struct Sizes {
    beg: usize,
    end: usize,
    block: usize,
    padding_token: i32,
    min_file_size_for_dl: usize,
}

const SIZES: Sizes = Sizes {
    beg: BEG_SIZE,
    end: END_SIZE,
    block: BLOCK_SIZE,
    padding_token: PADDING_TOKEN,
    min_file_size_for_dl: MIN_FILE_SIZE_FOR_DL,
};

impl Features {
    /// Extracts the features of in-memory content.
    pub fn extract(content: &[u8]) -> Self {
        let read_at = slice_reader(content);
        Self::extract_with(&SIZES, content.len() as u64, read_at)
            .expect("reading from a slice cannot fail")
    }

    /// Extracts the features of seekable content, e.g. a [`std::fs::File`].
    ///
    /// Reads at most the first and the last 4 KiB, so this is cheap for large
    /// files. The reader is read from its start regardless of its current position,
    /// and its position afterwards is unspecified.
    pub fn extract_reader(mut reader: impl Read + Seek) -> io::Result<Self> {
        let len = reader.seek(SeekFrom::End(0))?;
        Self::extract_with(&SIZES, len, |buf, offset| {
            reader.seek(SeekFrom::Start(offset))?;
            reader.read_exact(buf)
        })
    }

    /// Extracts the features of content of length `len`, reading at most its first
    /// and last 4 KiB with `read_at(buffer, offset)`.
    #[cfg_attr(
        not(all(target_arch = "wasm32", target_os = "unknown")),
        expect(dead_code, reason = "used by the WebAssembly bindings")
    )]
    pub(crate) fn extract_at(
        len: u64,
        read_at: impl FnMut(&mut [u8], u64) -> io::Result<()>,
    ) -> io::Result<Self> {
        Self::extract_with(&SIZES, len, read_at)
    }

    fn extract_with(
        sizes: &Sizes,
        len: u64,
        read_at: impl FnMut(&mut [u8], u64) -> io::Result<()>,
    ) -> io::Result<Self> {
        if len == 0 {
            return Ok(Self::Ruled(ContentType::Empty));
        }
        let (first_block, tokens) = read_tokens(sizes, len, read_at)?;
        if tokens[sizes.min_file_size_for_dl - 1] != sizes.padding_token {
            return Ok(Self::Tokens(tokens));
        }
        // Too little content besides whitespace for the model. Like upstream, only
        // the first block is checked for UTF-8.
        let content_type = match std::str::from_utf8(&first_block) {
            Ok(_) => ContentType::Txt,
            Err(_) => ContentType::Unknown,
        };
        Ok(Self::Ruled(content_type))
    }
}

fn slice_reader(content: &[u8]) -> impl FnMut(&mut [u8], u64) -> io::Result<()> + '_ {
    |buf, offset| {
        let offset = usize::try_from(offset).expect("offset is within the slice");
        buf.copy_from_slice(&content[offset..][..buf.len()]);
        Ok(())
    }
}

/// Reads the first and last blocks of the content, and returns the first block and
/// the tokens: the beginning left-aligned and the end right-aligned, both without
/// surrounding whitespace, and padding in between.
fn read_tokens(
    sizes: &Sizes,
    len: u64,
    mut read_at: impl FnMut(&mut [u8], u64) -> io::Result<()>,
) -> io::Result<(Vec<u8>, Vec<i32>)> {
    let block_len = usize::try_from(len.min(sizes.block as u64)).unwrap();
    let mut first_block = vec![0; block_len];
    read_at(&mut first_block, 0)?;
    let last_block = if len == block_len as u64 {
        None
    } else {
        let mut last_block = vec![0; block_len];
        read_at(&mut last_block, len - block_len as u64)?;
        Some(last_block)
    };

    let beg = strip_start(&first_block);
    let end = strip_end(last_block.as_deref().unwrap_or(&first_block));
    let mut tokens = vec![sizes.padding_token; sizes.beg + sizes.end];
    let (beg_tokens, end_tokens) = tokens.split_at_mut(sizes.beg);
    for (token, &byte) in beg_tokens.iter_mut().zip(beg) {
        *token = byte.into();
    }
    for (token, &byte) in end_tokens.iter_mut().rev().zip(end.iter().rev()) {
        *token = byte.into();
    }
    Ok((first_block, tokens))
}

/// Whitespace as in Python's `bytes.strip()`: ASCII whitespace and vertical tab.
fn is_whitespace(byte: u8) -> bool {
    byte.is_ascii_whitespace() || byte == 0x0b
}

fn strip_start(bytes: &[u8]) -> &[u8] {
    let whitespace = bytes.iter().take_while(|&&b| is_whitespace(b)).count();
    &bytes[whitespace..]
}

fn strip_end(bytes: &[u8]) -> &[u8] {
    let whitespace = bytes
        .iter()
        .rev()
        .take_while(|&&b| is_whitespace(b))
        .count();
    &bytes[..bytes.len() - whitespace]
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use base64::prelude::*;
    use flate2::read::GzDecoder;

    use super::*;
    use crate::INPUT_SIZE;

    #[test]
    fn matches_upstream_reference() {
        const EXAMPLES: &[u8] =
            include_bytes!("../tests/fixtures/upstream/features_extraction_examples.json.gz");
        let examples: Vec<serde_json::Value> =
            serde_json::from_reader(GzDecoder::new(EXAMPLES)).unwrap();
        assert!(!examples.is_empty());

        let size = |value: &serde_json::Value| value.as_u64().unwrap() as usize;
        let tokens = |value: &serde_json::Value| -> Vec<i32> {
            let values = value.as_array().unwrap();
            values.iter().map(|v| v.as_i64().unwrap() as i32).collect()
        };
        for example in &examples {
            let (args, features) = (&example["args"], &example["features"]);
            assert_eq!(args["mid_size"], 0);
            assert_eq!(args["use_inputs_at_offsets"], false);
            let sizes = Sizes {
                beg: size(&args["beg_size"]),
                end: size(&args["end_size"]),
                block: size(&args["block_size"]),
                padding_token: args["padding_token"].as_i64().unwrap() as i32,
                min_file_size_for_dl: 1,
            };
            let content = BASE64_STANDARD
                .decode(example["content_base64"].as_str().unwrap())
                .unwrap();
            let expected = [tokens(&features["beg"]), tokens(&features["end"])].concat();

            let (_, actual) =
                read_tokens(&sizes, content.len() as u64, slice_reader(&content)).unwrap();

            assert_eq!(actual, expected, "{example}");
        }
    }

    #[test]
    fn tiny_content_is_ruled() {
        assert_eq!(Features::extract(b""), Features::Ruled(ContentType::Empty));
        assert_eq!(
            Features::extract(b"0123456"),
            Features::Ruled(ContentType::Txt)
        );
        // Control characters are valid UTF-8.
        assert_eq!(
            Features::extract(&[0, 1, 2, 3, 4, 5, 6]),
            Features::Ruled(ContentType::Txt)
        );
        assert_eq!(
            Features::extract(&[0xff; 7]),
            Features::Ruled(ContentType::Unknown)
        );
        // Surrounding whitespace does not count.
        assert_eq!(
            Features::extract(b" \t\n\x0b\x0c\rabc\r\n  "),
            Features::Ruled(ContentType::Txt)
        );
    }

    #[test]
    fn tokens() {
        let Features::Tokens(tokens) = Features::extract(b"  01234567\n") else {
            panic!("expected tokens");
        };
        let bytes = |s: &[u8]| -> Vec<i32> { s.iter().map(|&b| b.into()).collect() };
        assert_eq!(tokens.len(), INPUT_SIZE);
        // The beginning loses leading whitespace, the end loses trailing whitespace.
        assert_eq!(tokens[..9], bytes(b"01234567\n"));
        let padding = &tokens[9..INPUT_SIZE - 10];
        assert!(padding.iter().all(|&t| t == PADDING_TOKEN));
        assert_eq!(tokens[INPUT_SIZE - 10..], bytes(b"  01234567"));
    }

    #[test]
    fn reader_matches_slice() {
        for len in [
            0,
            7,
            8,
            100,
            BLOCK_SIZE,
            BLOCK_SIZE + 1,
            3 * BLOCK_SIZE + 17,
        ] {
            let content: Vec<u8> = (0..len).map(|i| (i * 7 % 251) as u8).collect();
            let mut reader = Cursor::new(&content);
            reader.set_position(len as u64 / 2);
            assert_eq!(
                Features::extract_reader(reader).unwrap(),
                Features::extract(&content),
                "len {len}"
            );
        }
    }
}
