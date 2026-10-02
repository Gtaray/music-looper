use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Decoded audio at its native sample rate, interleaved
pub struct Audio {
    pub rate: u32,
    pub channels: usize,
    pub samples: Vec<f32>,
}

impl Audio {
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels
    }

    /// Mean of all channels, like `librosa.to_mono`
    pub fn mono(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.samples.clone();
        }
        let scale = self.channels as f32;
        self.samples
            .chunks_exact(self.channels)
            .map(|frame| frame.iter().sum::<f32>() / scale)
            .collect()
    }
}

pub fn decode(path: &Path) -> Result<Audio> {
    let file = File::open(path).with_context(|| format!("Unable to open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }

    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .context("Unsupported or unreadable audio file")?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow!("No audio track found"))?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| anyhow!("Audio track has no codec parameters"))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .context("Unsupported audio codec")?;

    let mut rate = params.sample_rate.unwrap_or(0);
    let mut channels = params.channels.as_ref().map(|c| c.count()).unwrap_or(0);
    let mut samples: Vec<f32> = Vec::new();
    let mut packet_samples: Vec<f32> = Vec::new();

    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buffer) => {
                rate = buffer.spec().rate();
                channels = buffer.spec().channels().count();
                packet_samples.resize(buffer.samples_interleaved(), 0.0);
                buffer.copy_to_slice_interleaved(&mut packet_samples);
                samples.extend_from_slice(&packet_samples);
            }
            Err(Error::DecodeError(_)) => continue,
            Err(error) => return Err(error.into()),
        }
    }

    if rate == 0 || channels == 0 || samples.is_empty() {
        bail!("No audio data could be decoded");
    }
    Ok(Audio {
        rate,
        channels,
        samples,
    })
}
