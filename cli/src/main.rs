//! `magika`: detect the content type of files with Google's Magika model.

use std::{
    borrow::Cow,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::Result;
use clap::{CommandFactory, Parser, ValueEnum, error::ErrorKind};
use magika_burn::{
    ContentType, ContentTypeInfo, Magika, OverwriteReason, Prediction, PredictionMode,
};
use serde::Serialize;
use walkdir::WalkDir;

/// Detect the content type of files with Google's Magika model.
#[derive(Parser)]
#[command(name = "magika", version, arg_required_else_help = true)]
struct Args {
    /// Files or directories to identify. `-` reads standard input.
    #[arg(required = true, value_name = "PATH")]
    paths: Vec<PathBuf>,

    /// Identify the files inside directories instead of the directories themselves.
    #[arg(short, long)]
    recursive: bool,

    /// Identify symbolic links as such instead of following them.
    #[arg(long)]
    no_dereference: bool,

    /// How confident the model must be for its prediction to be used.
    #[arg(short = 'm', long, value_name = "MODE", value_enum, default_value_t = Mode::HighConfidence)]
    prediction_mode: Mode,

    /// Print the label (e.g. `rust`) instead of the description.
    #[arg(short, long, conflicts_with = "mime_type")]
    label: bool,

    /// Print the MIME type (e.g. `application/x-rust`) instead of the description.
    #[arg(short = 'i', long)]
    mime_type: bool,

    /// Append the score of the prediction.
    #[arg(short = 's', long)]
    output_score: bool,

    /// Print a JSON array with the prediction and content type of each path.
    #[arg(long, conflicts_with_all = ["label", "mime_type", "output_score"])]
    json: bool,
}

/// Command-line names of [`PredictionMode`].
#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    /// Per-content-type thresholds, e.g. 0.75 for Markdown, and 0.5 for the rest.
    HighConfidence,
    /// A threshold of 0.5 for every content type.
    MediumConfidence,
    /// No threshold: always use the model's prediction.
    BestGuess,
}

impl From<Mode> for PredictionMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::HighConfidence => Self::HighConfidence,
            Mode::MediumConfidence => Self::MediumConfidence,
            Mode::BestGuess => Self::BestGuess,
        }
    }
}

/// The result of identifying one path.
struct Entry {
    path: PathBuf,
    result: io::Result<Prediction>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    if args.paths.iter().filter(|path| is_stdin(path)).count() > 1 {
        Args::command()
            .error(
                ErrorKind::ArgumentConflict,
                "standard input (`-`) can only be read once",
            )
            .exit();
    }

    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        // The reader went away, e.g. `magika … | head`.
        Err(error) if is_broken_pipe(&error) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("magika: {error:#}");
            ExitCode::FAILURE
        }
    }
}

/// Identifies and prints every path. Returns whether all of them were identified.
fn run(args: &Args) -> Result<bool> {
    let magika = Magika::new().with_prediction_mode(args.prediction_mode.into());
    let mut output = Output::new(args);
    for path in &args.paths {
        if is_stdin(path) {
            let result = read_stdin().map(|content| magika.identify_bytes(&content));
            output.print(Entry {
                path: path.clone(),
                result,
            })?;
        } else if args.recursive {
            for entry in walk(&magika, path, args.no_dereference) {
                output.print(entry)?;
            }
        } else {
            let result = identify_path(&magika, path, args.no_dereference);
            output.print(Entry {
                path: path.clone(),
                result,
            })?;
        }
    }
    output.finish()?;
    Ok(output.all_identified)
}

/// Prints entries as they are identified: as text lines, or as the elements of
/// a JSON array.
struct Output<'a> {
    args: &'a Args,
    stdout: io::StdoutLock<'static>,
    printed: usize,
    all_identified: bool,
}

impl<'a> Output<'a> {
    fn new(args: &'a Args) -> Self {
        Self {
            args,
            stdout: io::stdout().lock(),
            printed: 0,
            all_identified: true,
        }
    }

    fn print(&mut self, entry: Entry) -> io::Result<()> {
        self.all_identified &= entry.result.is_ok();
        if self.args.json {
            let json = serde_json::to_string_pretty(&JsonEntry::from(&entry))
                .expect("entries serialize to JSON");
            let separator = if self.printed == 0 { "[\n" } else { ",\n" };
            // Indent the element; newlines inside JSON strings are escaped.
            write!(self.stdout, "{separator}  {}", json.replace('\n', "\n  "))?;
        } else {
            writeln!(self.stdout, "{}", text(&entry, self.args))?;
        }
        self.printed += 1;
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        match (self.args.json, self.printed) {
            (false, _) => Ok(()),
            (true, 0) => writeln!(self.stdout, "[]"),
            (true, _) => writeln!(self.stdout, "\n]"),
        }
    }
}

/// An element of the `--json` output.
#[derive(Serialize)]
#[serde(untagged)]
enum JsonEntry<'a> {
    Identified {
        path: Cow<'a, str>,
        prediction: Prediction,
        /// Information about `prediction.output`.
        info: &'static ContentTypeInfo,
    },
    Failed {
        path: Cow<'a, str>,
        error: String,
    },
}

impl<'a> From<&'a Entry> for JsonEntry<'a> {
    fn from(entry: &'a Entry) -> Self {
        let path = entry.path.to_string_lossy();
        match &entry.result {
            Ok(prediction) => Self::Identified {
                path,
                prediction: *prediction,
                info: prediction.info(),
            },
            Err(error) => Self::Failed {
                path,
                error: error.to_string(),
            },
        }
    }
}

fn is_stdin(path: &Path) -> bool {
    path.as_os_str() == "-"
}

fn is_broken_pipe(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<io::Error>()
        .is_some_and(|error| error.kind() == io::ErrorKind::BrokenPipe)
}

fn read_stdin() -> io::Result<Vec<u8>> {
    let mut content = Vec::new();
    io::stdin().lock().read_to_end(&mut content)?;
    Ok(content)
}

/// Identifies the files under `path` (or `path` itself if it is not a directory),
/// depth-first in name order.
fn walk<'a>(
    magika: &'a Magika,
    path: &'a Path,
    no_dereference: bool,
) -> impl Iterator<Item = Entry> + 'a {
    WalkDir::new(path)
        .follow_links(!no_dereference)
        .follow_root_links(!no_dereference)
        .sort_by_file_name()
        .into_iter()
        .filter_map(move |entry| match entry {
            // Directories are walked, not identified.
            Ok(entry) if entry.file_type().is_dir() => None,
            Ok(entry) => Some(Entry {
                result: identify_path(magika, entry.path(), no_dereference),
                path: entry.into_path(),
            }),
            Err(error) => Some(walk_error(error, path)),
        })
}

/// Converts a walk error, whose message would repeat the path, to an entry.
fn walk_error(error: walkdir::Error, root: &Path) -> Entry {
    let path = error.path().unwrap_or(root).to_owned();
    let result = Err(match error.loop_ancestor() {
        Some(ancestor) => io::Error::other(format!(
            "directory cycle: links to its ancestor {}",
            ancestor.display()
        )),
        None => error
            .into_io_error()
            .unwrap_or_else(|| io::Error::other("unknown error")),
    });
    Entry { path, result }
}

fn identify_path(magika: &Magika, path: &Path, no_dereference: bool) -> io::Result<Prediction> {
    if no_dereference && fs::symlink_metadata(path)?.is_symlink() {
        return Ok(Prediction::ruled(ContentType::Symlink));
    }
    let metadata = fs::metadata(path)?;
    if metadata.is_dir() {
        return Ok(Prediction::ruled(ContentType::Directory));
    }
    if !metadata.is_file() {
        return Err(io::Error::other("not a regular file"));
    }
    magika.identify_reader(File::open(path)?)
}

/// Formats an entry as `<path>: <content type>`, followed by the score if requested.
fn text(entry: &Entry, args: &Args) -> String {
    let path = entry.path.display();
    let prediction = match &entry.result {
        Ok(prediction) => prediction,
        Err(error) => return format!("{path}: error: {error}"),
    };
    let content_type = describe(prediction.output, args);
    if prediction.overwrite_reason == OverwriteReason::LowConfidence {
        // The score is the model's score for its guess, so it goes with the guess.
        let guess = describe(prediction.dl, args);
        let score = percent(prediction.score);
        format!("{path}: {content_type} [low confidence: {guess}, {score}]")
    } else if args.output_score {
        format!("{path}: {content_type} {}", percent(prediction.score))
    } else {
        format!("{path}: {content_type}")
    }
}

/// The label, the MIME type, or the description and group of a content type.
fn describe(content_type: ContentType, args: &Args) -> String {
    let info = content_type.info();
    if args.label {
        info.label.to_owned()
    } else if args.mime_type {
        info.mime_type.to_owned()
    } else {
        format!("{} ({})", info.description, info.group)
    }
}

/// Formats a score as a whole percentage, rounded down like upstream Magika.
fn percent(score: f32) -> String {
    format!("{}%", (f64::from(score) * 100.0).floor())
}
