//! Loop tags in Vorbis comments, following PyMusicLooper's `read_tags`/`export_tags` (core.py).

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use lofty::config::{ParseOptions, WriteOptions};
use lofty::file::{AudioFile, FileType};
use lofty::flac::FlacFile;
use lofty::ogg::tag::VorbisComments;
use lofty::probe::Probe;
use serde::Serialize;

/// From SDL_mixer and vgmstream, in PyMusicLooper's order of preference
const START_TAGS: &[&str] = &[
    "COMMENT=LOOPPOINT",
    "LOOP",
    "LOOP_BEGIN",
    "LOOPPOINT",
    "LOOPS",
    "LOOPSTART",
    "LOOP_START",
    "LOOP-START",
    "UM3.STREAM.LOOPPOINT.START",
    "XIPH_CUE_LOOPSTART",
];
const END_TAGS: &[&str] = &[
    "LOOPE",
    "LOOPEND",
    "LOOP_END",
    "LOOP-END",
    "LOOPLENGTH",
    "LOOP_LENGTH",
    "LOOP-LENGTH",
    "XIPH_CUE_LOOPEND",
];

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoopTags {
    pub sample_rate: u32,
    pub start: Option<u64>,
    pub end: Option<u64>,
    pub start_tag: Option<String>,
    pub end_tag: Option<String>,
}

/// The end tag holds a length instead of a position
fn is_length_tag(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    upper.contains("LEN") || upper.contains("OFFSET")
}

fn read_flac(path: &Path) -> Result<FlacFile> {
    let file_type = Probe::open(path)?.guess_file_type()?.file_type();
    if file_type != Some(FileType::Flac) {
        bail!("Loop tags are only supported for FLAC files");
    }
    let mut file = File::open(path).with_context(|| format!("Unable to open {}", path.display()))?;
    Ok(FlacFile::read_from(&mut file, ParseOptions::new())?)
}

fn find_tag<'a>(comments: &'a VorbisComments, names: &[&str]) -> Option<(String, &'a str)> {
    names
        .iter()
        .find_map(|name| comments.get(name).map(|value| (name.to_string(), value)))
}

pub fn read(path: &Path) -> Result<LoopTags> {
    let file = read_flac(path)?;
    let sample_rate = file.properties().sample_rate();
    let none = LoopTags {
        sample_rate,
        start: None,
        end: None,
        start_tag: None,
        end_tag: None,
    };
    let Some(comments) = file.vorbis_comments() else {
        return Ok(none);
    };
    let (Some((start_tag, start)), Some((end_tag, end))) = (find_tag(comments, START_TAGS), find_tag(comments, END_TAGS))
    else {
        return Ok(none);
    };
    let start: u64 = start.trim().parse().with_context(|| format!("{start_tag} is not an integer"))?;
    let end: u64 = end.trim().parse().with_context(|| format!("{end_tag} is not an integer"))?;
    let (loop_start, mut loop_end) = (start.min(end), start.max(end));
    if is_length_tag(&end_tag) {
        loop_end += loop_start;
    }
    Ok(LoopTags {
        sample_rate,
        start: Some(loop_start),
        end: Some(loop_end),
        start_tag: Some(start_tag),
        end_tag: Some(end_tag),
    })
}

/// Writes the loop tags in place, replacing any other known loop tags so they can't shadow the new values
pub fn write(path: &Path, start: u64, end: u64, start_tag: &str, end_tag: &str) -> Result<LoopTags> {
    if end <= start {
        bail!("Loop end must be after loop start");
    }
    let mut file = read_flac(path)?;
    let sample_rate = file.properties().sample_rate();
    if file.vorbis_comments().is_none() {
        file.set_vorbis_comments(VorbisComments::default());
    }
    let comments = file.vorbis_comments_mut().expect("vorbis comments were just set");
    for name in START_TAGS.iter().chain(END_TAGS) {
        comments.remove(name).for_each(drop);
    }
    let end_value = if is_length_tag(end_tag) { end - start } else { end };
    comments.insert(start_tag.to_string(), start.to_string());
    comments.insert(end_tag.to_string(), end_value.to_string());
    file.save_to_path(path, WriteOptions::default())
        .with_context(|| format!("Unable to write tags to {}", path.display()))?;

    Ok(LoopTags {
        sample_rate,
        start: Some(start),
        end: Some(end),
        start_tag: Some(start_tag.to_string()),
        end_tag: Some(end_tag.to_string()),
    })
}
