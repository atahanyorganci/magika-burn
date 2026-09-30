//! `magika`: detect the content type of files with Google's Magika model.

use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::Result;
use clap::{CommandFactory, Parser, ValueEnum, error::ErrorKind};
use magika_burn::{ContentType, Magika, OverwriteReason, Prediction, PredictionMode};

/// Detect the content type of files with Google's Magika model.
#[derive(Parser)]
#[command(name = "magika", version, arg_required_else_help = true)]
struct Args {
    /// Files or directories to identify. `-` reads standard input.
    #[arg(required = true, value_name = "PATH")]
    paths: Vec<PathBuf>,

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
    let mut stdout = io::stdout().lock();
    let mut all_identified = true;
    for path in &args.paths {
        let entry = identify(&magika, path);
        all_identified &= entry.result.is_ok();
        writeln!(stdout, "{}", text(&entry, args))?;
    }
    Ok(all_identified)
}

fn is_stdin(path: &Path) -> bool {
    path.as_os_str() == "-"
}

fn is_broken_pipe(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<io::Error>()
        .is_some_and(|error| error.kind() == io::ErrorKind::BrokenPipe)
}

fn identify(magika: &Magika, path: &Path) -> Entry {
    let result = if is_stdin(path) {
        read_stdin().map(|content| magika.identify_bytes(&content))
    } else {
        identify_path(magika, path)
    };
    Entry {
        path: path.to_owned(),
        result,
    }
}

fn read_stdin() -> io::Result<Vec<u8>> {
    let mut content = Vec::new();
    io::stdin().lock().read_to_end(&mut content)?;
    Ok(content)
}

fn identify_path(magika: &Magika, path: &Path) -> io::Result<Prediction> {
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
