//! Port of PyMusicLooper 3.6.0 `find_best_loop_points` (analysis.py), with onset peaks instead of beats.

use anyhow::{Result, bail};
use rayon::prelude::*;
use serde::Serialize;

use crate::audio::Audio;
use crate::features::{self, HOP};

const ACCEPTABLE_NOTE_DEVIATION: f32 = 0.0875;
const ACCEPTABLE_LOUDNESS_DIFFERENCE: f64 = 0.5;
const NUM_TEST_BEATS: usize = 12;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoopPair {
    #[serde(skip)]
    start_frame: usize,
    #[serde(skip)]
    end_frame: usize,
    pub start: usize,
    pub end: usize,
    pub score: f64,
    pub note_distance: f32,
    pub loudness_difference: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Analysis {
    pub sample_rate: u32,
    pub start: usize,
    pub end: usize,
    pub candidates: Vec<LoopPair>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<Debug>,
}

/// Intermediate values
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Debug {
    pub trim_offset: usize,
    pub mono_len: usize,
    pub frames: usize,
    pub chroma_sum: f64,
    pub chroma_col100: Vec<f32>,
    pub bpm: f64,
    pub tuning: f64,
    pub beats: usize,
    pub candidate_pairs: usize,
}

/// `librosa.time_to_frames(seconds, sr=rate)`
fn seconds_to_frames(seconds: f64, rate: u32) -> usize {
    ((seconds * rate as f64) as usize) / HOP
}

fn norm(a: &[f32; 12]) -> f32 {
    a.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// `_find_candidate_pairs`, in the same order (by loop end, then loop start)
fn find_candidate_pairs(
    chroma: &[[f32; 12]],
    power_db_max: &[f64],
    beats: &[usize],
    min_loop: usize,
    max_loop: usize,
) -> Vec<LoopPair> {
    beats
        .par_iter()
        .map(|&loop_end| {
            let deviation = norm(&chroma[loop_end].map(|v| v * ACCEPTABLE_NOTE_DEVIATION));
            let mut pairs = Vec::new();
            for &loop_start in beats {
                let length = loop_end as isize - loop_start as isize;
                if length < min_loop as isize {
                    break;
                }
                if length > max_loop as isize {
                    continue;
                }
                let mut difference = [0.0f32; 12];
                for (c, d) in difference.iter_mut().enumerate() {
                    *d = chroma[loop_end][c] - chroma[loop_start][c];
                }
                let note_distance = norm(&difference);
                if note_distance <= deviation {
                    let loudness_difference = (power_db_max[loop_end] - power_db_max[loop_start]).abs();
                    if loudness_difference <= ACCEPTABLE_LOUDNESS_DIFFERENCE {
                        pairs.push(LoopPair {
                            start_frame: loop_start,
                            end_frame: loop_end,
                            start: 0,
                            end: 0,
                            score: 0.0,
                            note_distance,
                            loudness_difference,
                        });
                    }
                }
            }
            pairs
        })
        .flatten()
        .collect()
}

/// `np.percentile(values, q)` with linear interpolation
fn percentile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * q / 100.0;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

/// `_prune_candidates`
fn prune_candidates(pairs: Vec<LoopPair>) -> Vec<LoopPair> {
    let epsilon = 1e-3;
    let db: Vec<f64> = pairs.iter().map(|p| p.loudness_difference).collect();
    let notes: Vec<f64> = pairs.iter().map(|p| p.note_distance as f64).collect();
    let threshold = |values: &[f64], q: f64| {
        let adjusted: Vec<f64> = values.iter().cloned().filter(|v| *v > epsilon).collect();
        if adjusted.len() > 3 {
            percentile(&adjusted, q)
        } else {
            values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }
    };
    let db_threshold = threshold(&db, 50.0).max(0.25);
    let note_threshold = threshold(&notes, 75.0);
    pairs
        .into_iter()
        .filter(|p| p.loudness_difference <= db_threshold && (p.note_distance as f64) <= note_threshold)
        .collect()
}

/// `np.geomspace(start, stop, num)`
fn geomspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let (log_start, log_stop) = (start.log10(), stop.log10());
    let step = (log_stop - log_start) / (num - 1) as f64;
    let mut values: Vec<f64> = (0..num).map(|i| 10f64.powf(log_start + i as f64 * step)).collect();
    values[0] = start;
    values[num - 1] = stop;
    values
}

/// `_calculate_subseq_beat_similarity`
fn subsequence_similarity(chroma: &[[f32; 12]], b1: usize, b2: usize, offset: isize, weights: &[f64]) -> f64 {
    let chroma_len = chroma.len() as isize;
    let test_length = offset.unsigned_abs();
    let (b1, b2) = (b1 as isize, b2 as isize);
    let (b1_start, b2_start, max_offset) = if offset < 0 {
        let max_negative = offset.max(-b1).max(-b2);
        (b1 + max_negative, b2 + max_negative, max_negative.unsigned_abs())
    } else {
        let b1_end = (b1 + offset).min(chroma_len);
        let b2_end = (b2 + offset).min(chroma_len);
        (b1, b2, (b1_end - b1).min(b2_end - b2).max(0) as usize)
    };

    let mut weighted = 0.0f64;
    for i in 0..max_offset {
        let a = &chroma[b1_start as usize + i];
        let b = &chroma[b2_start as usize + i];
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let similarity = dot / (norm(a) * norm(b)).max(1e-10);
        weighted += similarity as f64 * weights[i];
    }
    // Frames past the clipped window count as zero similarity
    weighted / weights[..test_length].iter().sum::<f64>()
}

/// `_assess_and_filter_loop_pairs`
fn assess_and_filter(chroma: &[[f32; 12]], bpm: f64, rate: u32, pairs: Vec<LoopPair>) -> Vec<LoopPair> {
    let seconds_to_test = NUM_TEST_BEATS as f64 / (bpm / 60.0);
    let mut test_offset = ((seconds_to_test * rate as f64) as usize) / HOP;
    if test_offset > chroma.len() {
        test_offset = chroma.len() / 4;
    }

    let mut pairs = if pairs.len() >= 100 { prune_candidates(pairs) } else { pairs };

    let weights = geomspace((test_offset / NUM_TEST_BEATS).max(2) as f64, 1.0, test_offset);
    let reversed: Vec<f64> = weights.iter().rev().cloned().collect();
    let scores: Vec<f64> = pairs
        .par_iter()
        .map(|p| {
            let lookahead = subsequence_similarity(chroma, p.start_frame, p.end_frame, test_offset as isize, &weights);
            let lookbehind =
                subsequence_similarity(chroma, p.start_frame, p.end_frame, -(test_offset as isize), &reversed);
            lookahead.max(lookbehind)
        })
        .collect();
    for (pair, score) in pairs.iter_mut().zip(scores) {
        pair.score = score;
    }
    // Stable, like Python's sorted
    pairs.sort_by(|a, b| b.score.total_cmp(&a.score));
    pairs
}

/// `nearest_zero_crossing` (Audacity's "At Zero Crossings") on the original interleaved audio
fn nearest_zero_crossing(audio: &Audio, sample_idx: usize) -> usize {
    let channels = audio.channels;
    let length = audio.frames() as isize;
    let window_size = ((audio.rate as f64 / 100.0).max(1.0)) as isize;
    let offset = window_size / 2;
    let sample_idx = sample_idx as isize;
    let neg_offset = (sample_idx - offset).max(0);
    let pos_offset = length.min(sample_idx + offset);
    let window_length = (pos_offset - neg_offset).max(0) as usize;
    let offset_correction = if sample_idx - offset < 0 { (sample_idx - offset).abs() } else { 0 };

    let mut dist = vec![0.0f64; window_length];
    for channel in 0..channels {
        let mut prev = 2.0f64;
        let mut one_dist = vec![0.0f32; window_length];
        for (i, d) in one_dist.iter_mut().enumerate() {
            let sample = audio.samples[(neg_offset as usize + i) * channels + channel];
            let mut fdist = sample.abs() as f64;
            if prev * sample as f64 > 0.0 {
                fdist += 0.4;
            } else if prev > 0.0 {
                fdist += 0.1;
            }
            prev = sample as f64;
            *d = fdist as f32;
        }
        for (i, d) in dist.iter_mut().enumerate() {
            *d += one_dist[i] as f64;
            *d += 0.1 * (i as isize - offset + offset_correction).abs() as f64 / (window_size as f64 / 2.0);
        }
    }

    let Some((argmin, minimum)) = dist
        .iter()
        .enumerate()
        .fold(None, |best: Option<(usize, f64)>, (i, &d)| match best {
            Some((_, m)) if m <= d => best,
            _ => Some((i, d)),
        })
    else {
        return sample_idx as usize;
    };
    let limit = if channels == 1 { 0.2 } else { 0.6 * channels as f64 };
    if minimum > limit {
        return sample_idx as usize;
    }
    (sample_idx + argmin as isize - offset + offset_correction) as usize
}

/// `find_best_loop_points` with PyMusicLooper's defaults (min loop 35% of the track, no max)
pub fn find_loop_points(audio: &Audio, include_debug: bool) -> Result<Analysis> {
    let mut mono = audio.mono();
    let peak = mono.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak == 0.0 {
        bail!("The audio only contains silence and cannot be analyzed");
    }
    for v in mono.iter_mut() {
        *v /= peak;
    }
    let (trim_start, trim_end) = features::trim(&mono, 40.0);
    let trimmed = &mono[trim_start..trim_end];
    if trimmed.is_empty() {
        bail!("The audio only contains silence and cannot be analyzed");
    }

    let total_duration = audio.frames() as f64 / audio.rate as f64;
    let min_loop = seconds_to_frames((0.35 * total_duration).trunc(), audio.rate).max(1);
    let max_loop = seconds_to_frames(total_duration, audio.rate);

    let features = features::analyze(trimmed, audio.rate);
    let beats = features::local_maxima(&features.onset_env);
    let pairs = find_candidate_pairs(&features.chroma, &features.power_db_max, &beats, min_loop, max_loop);
    let candidate_pairs = pairs.len();
    if pairs.is_empty() {
        bail!("No loop points found");
    }

    let mut pairs = assess_and_filter(&features.chroma, features.bpm, audio.rate, pairs);
    // Upstream's _prioritize_duration reads loop lengths before they are set, so it never reorders; skipped to match
    for pair in pairs.iter_mut() {
        let (mut start_frame, mut end_frame) = (pair.start_frame, pair.end_frame);
        if trim_start > 0 {
            start_frame = (start_frame * HOP + trim_start) / HOP;
            end_frame = (end_frame * HOP + trim_start) / HOP;
        }
        pair.start = nearest_zero_crossing(audio, start_frame * HOP);
        pair.end = nearest_zero_crossing(audio, end_frame * HOP);
    }
    let best = pairs.first().cloned().expect("pairs is not empty");

    let column = 100.min(features.chroma.len() - 1);
    Ok(Analysis {
        sample_rate: audio.rate,
        start: best.start,
        end: best.end,
        debug: include_debug.then(|| Debug {
            trim_offset: trim_start,
            mono_len: trimmed.len(),
            frames: features.chroma.len(),
            chroma_sum: features.chroma.iter().flatten().map(|v| *v as f64).sum(),
            chroma_col100: features.chroma[column].to_vec(),
            bpm: features.bpm,
            tuning: features.tuning,
            beats: beats.len(),
            candidate_pairs,
        }),
        candidates: pairs,
    })
}
