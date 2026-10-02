mod audio;
mod features;
mod looper;
mod tags;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::Serialize;

/// Finds seamless loop points in music, and reads/writes loop tags. Prints JSON.
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Find the best loop points (in samples at the file's sample rate)
    Analyze {
        path: PathBuf,
        /// Include intermediate values for comparing against PyMusicLooper
        #[arg(long)]
        debug: bool,
        /// Only list the best N candidates (default: all)
        #[arg(long, value_name = "N", value_parser = parse_max_candidates)]
        max_candidates: Option<usize>,
    },
    /// Read loop tags (names auto-detected)
    ReadTags { path: PathBuf },
    /// Write loop tags in place, replacing any existing loop tags
    WriteTags {
        path: PathBuf,
        #[arg(long)]
        start: u64,
        #[arg(long)]
        end: u64,
        #[arg(long, default_value = "LOOP_START")]
        start_tag: String,
        /// Stored as a length if the name contains LEN or OFFSET
        #[arg(long, default_value = "LOOP_END")]
        end_tag: String,
    },
}

fn parse_max_candidates(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        _ => Err("must be a whole number of at least 1".into()),
    }
}

fn print_json(value: &impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Analyze {
            path,
            debug,
            max_candidates,
        } => {
            let audio = audio::decode(&path)?;
            let mut analysis = looper::find_loop_points(&audio, debug)?;
            if let Some(max) = max_candidates {
                analysis.candidates.truncate(max);
            }
            print_json(&analysis)
        }
        Command::ReadTags { path } => print_json(&tags::read(&path)?),
        Command::WriteTags {
            path,
            start,
            end,
            start_tag,
            end_tag,
        } => print_json(&tags::write(&path, start, end, &start_tag, &end_tag)?),
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}
