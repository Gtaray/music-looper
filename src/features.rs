//! Port of the librosa 0.11 feature extraction used by PyMusicLooper 3.6.0.
//! Float widths follow librosa (float32 spectra and chroma, float64 weighting) so results match.

use std::sync::Arc;

use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

pub const N_FFT: usize = 2048;
pub const HOP: usize = 512;
pub const N_BINS: usize = N_FFT / 2 + 1;
const N_MELS: usize = 128;
const N_CHROMA: usize = 12;
/// PyMusicLooper calls chroma_stft, beat_track and tempo without `sr`, so librosa assumes 22050 Hz
pub const LIBROSA_DEFAULT_SR: f64 = 22050.0;

pub struct Features {
    /// Chroma per frame, 12 values each
    pub chroma: Vec<[f32; N_CHROMA]>,
    /// Per frame max of `power_to_db(S_weighed, ref=np.median)`, without the median reference
    /// (it cancels out in loudness differences)
    pub power_db_max: Vec<f64>,
    pub onset_env: Vec<f64>,
    pub bpm: f64,
    pub tuning: f64,
}

/// `scipy.signal.get_window('hann', n, fftbins=True)`
fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

/// `numpy.linspace`
fn linspace(start: f64, stop: f64, num: usize, endpoint: bool) -> Vec<f64> {
    let div = if endpoint { num - 1 } else { num } as f64;
    let step = (stop - start) / div;
    let mut values: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    if endpoint && num > 1 {
        values[num - 1] = stop;
    }
    values
}

/// `numpy.fft.rfftfreq(n_fft, 1 / sr)`
fn fft_frequencies(sr: f64) -> Vec<f64> {
    let val = 1.0 / (N_FFT as f64 * (1.0 / sr));
    (0..N_BINS).map(|k| k as f64 * val).collect()
}

/// `librosa.effects.trim(y, top_db)`, returns the kept sample range
pub fn trim(y: &[f32], top_db: f64) -> (usize, usize) {
    let frame_length = N_FFT;
    let n_frames = 1 + y.len() / HOP;
    let half = (frame_length / 2) as isize;
    let rms: Vec<f64> = (0..n_frames)
        .into_par_iter()
        .map(|t| {
            let start = (t * HOP) as isize - half;
            let mut sum = 0.0f64;
            for j in 0..frame_length as isize {
                let idx = start + j;
                if idx >= 0 && (idx as usize) < y.len() {
                    let v = y[idx as usize] as f64;
                    sum += v * v;
                }
            }
            ((sum / frame_length as f64) as f32).sqrt() as f64
        })
        .collect();
    let max = rms.iter().cloned().fold(0.0f64, f64::max);
    let reference = 10.0 * (max * max).max(1e-10).log10();
    let non_silent: Vec<usize> = rms
        .iter()
        .enumerate()
        .filter(|(_, r)| 10.0 * (*r * *r).max(1e-10).log10() - reference > -top_db)
        .map(|(t, _)| t)
        .collect();
    match (non_silent.first(), non_silent.last()) {
        (Some(&first), Some(&last)) => (first * HOP, y.len().min((last + 1) * HOP)),
        _ => (0, 0),
    }
}

/// `librosa.A_weighting(frequencies, min_db=-80)`
fn a_weighting(frequencies: &[f64]) -> Vec<f64> {
    let c = [12194.217f64, 20.598997, 107.65265, 737.86223].map(|v| v * v);
    frequencies
        .iter()
        .map(|f| {
            let f_sq = f * f;
            let weight = 2.0
                + 20.0
                    * (c[0].log10() + 2.0 * f_sq.log10()
                        - (f_sq + c[0]).log10()
                        - (f_sq + c[1]).log10()
                        - 0.5 * (f_sq + c[2]).log10()
                        - 0.5 * (f_sq + c[3]).log10());
            if weight.is_nan() { -80.0 } else { weight.max(-80.0) }
        })
        .collect()
}

fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// Non-zero part of a filter: first bin and weights
struct SparseFilter {
    start: usize,
    weights: Vec<f32>,
}

/// `librosa.filters.mel(sr, n_fft=2048, n_mels=128, fmin=0, fmax=8000, norm='slaney')`, float32
fn mel_filters(sr: f64) -> Vec<SparseFilter> {
    let fftfreqs = fft_frequencies(sr);
    let mel_f: Vec<f64> = linspace(hz_to_mel(0.0), hz_to_mel(8000.0), N_MELS + 2, true)
        .into_iter()
        .map(mel_to_hz)
        .collect();
    (0..N_MELS)
        .map(|i| {
            let fdiff_lower = mel_f[i + 1] - mel_f[i];
            let fdiff_upper = mel_f[i + 2] - mel_f[i + 1];
            let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
            let row: Vec<f32> = fftfreqs
                .iter()
                .map(|f| {
                    let lower = -(mel_f[i] - f) / fdiff_lower;
                    let upper = (mel_f[i + 2] - f) / fdiff_upper;
                    let w = 0.0f64.max(lower.min(upper)) as f32;
                    (w as f64 * enorm) as f32
                })
                .collect();
            let start = row.iter().position(|w| *w != 0.0).unwrap_or(0);
            let end = row.iter().rposition(|w| *w != 0.0).map_or(start, |e| e + 1);
            SparseFilter {
                start,
                weights: row[start..end].to_vec(),
            }
        })
        .collect()
}

/// `librosa.filters.chroma(sr, n_fft=2048, tuning)` with defaults (ctroct 5, octwidth 2, norm 2, base_c), float32
fn chroma_filters(sr: f64, tuning: f64) -> Vec<[f32; N_CHROMA]> {
    let n_chroma = N_CHROMA as f64;
    let a440 = 440.0 * 2.0f64.powf(tuning / n_chroma);
    let frequencies = &linspace(0.0, sr, N_FFT, false)[1..];
    let mut frqbins: Vec<f64> = Vec::with_capacity(N_FFT);
    let octs: Vec<f64> = frequencies
        .iter()
        .map(|f| n_chroma * (f / (a440 / 16.0)).log2())
        .collect();
    frqbins.push(octs[0] - 1.5 * n_chroma);
    frqbins.extend_from_slice(&octs);
    let mut binwidthbins: Vec<f64> = frqbins.windows(2).map(|w| (w[1] - w[0]).max(1.0)).collect();
    binwidthbins.push(1.0);

    let n_chroma2 = (n_chroma / 2.0).round();
    let mut columns = Vec::with_capacity(N_BINS);
    for k in 0..N_BINS {
        let mut wts = [0.0f64; N_CHROMA];
        for (c, w) in wts.iter_mut().enumerate() {
            let d = (frqbins[k] - c as f64 + n_chroma2 + 10.0 * n_chroma).rem_euclid(n_chroma) - n_chroma2;
            *w = (-0.5 * (2.0 * d / binwidthbins[k]).powi(2)).exp();
        }
        let norm = wts.iter().map(|w| w * w).sum::<f64>().sqrt();
        let norm = if norm < f64::MIN_POSITIVE { 1.0 } else { norm };
        let octave_weight = (-0.5 * ((frqbins[k] / n_chroma - 5.0) / 2.0).powi(2)).exp();
        let mut column = [0.0f32; N_CHROMA];
        for (c, value) in column.iter_mut().enumerate() {
            // base_c rolls the rows by -3 so the bank starts at C
            *value = (wts[(c + 3) % N_CHROMA] / norm * octave_weight) as f32;
        }
        columns.push(column);
    }
    columns
}

struct Stft {
    fft: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
}

impl Stft {
    fn new() -> Self {
        Self {
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
            window: hann_periodic(N_FFT),
        }
    }

    /// `np.abs(librosa.stft(y)[:, t]) ** 2` as float32, centered with zero padding
    fn power(&self, y: &[f32], t: usize, buffer: &mut Vec<Complex<f64>>, out: &mut [f32; N_BINS]) {
        buffer.clear();
        let start = (t * HOP) as isize - (N_FFT / 2) as isize;
        for j in 0..N_FFT {
            let idx = start + j as isize;
            let v = if idx >= 0 && (idx as usize) < y.len() { y[idx as usize] as f64 } else { 0.0 };
            buffer.push(Complex::new(v * self.window[j], 0.0));
        }
        self.fft.process(buffer);
        for (k, value) in out.iter_mut().enumerate() {
            let (re, im) = (buffer[k].re as f32, buffer[k].im as f32);
            let magnitude = re.hypot(im);
            *value = magnitude * magnitude;
        }
    }
}

/// `librosa.piptrack(S, sr=22050)` peaks of one frame, as (pitch, magnitude)
fn piptrack_frame(s: &[f32; N_BINS], freq_mask: &[bool], peaks: &mut Vec<(f32, f32)>) {
    let reference = 0.1f32 * s.iter().cloned().fold(f32::MIN, f32::max);
    let z = |k: usize| if s[k] > reference { s[k] } else { 0.0 };
    for k in 1..N_BINS {
        if !freq_mask[k] {
            continue;
        }
        let is_max = if k == N_BINS - 1 {
            z(k) > z(k - 1)
        } else {
            z(k) > z(k - 1) && z(k) >= z(k + 1)
        };
        if !is_max {
            continue;
        }
        let shift = if k == N_BINS - 1 {
            0.0f32
        } else {
            let a = s[k + 1] + s[k - 1] - 2.0 * s[k];
            let b = (s[k + 1] - s[k - 1]) / 2.0;
            if b.abs() >= a.abs() { 0.0 } else { -b / a }
        };
        let avg = if k == N_BINS - 1 {
            s[k] - s[k - 1]
        } else {
            (s[k + 1] - s[k - 1]) / 2.0
        };
        let pitch = ((k as f64 + shift as f64) * LIBROSA_DEFAULT_SR / N_FFT as f64) as f32;
        let magnitude = s[k] + 0.5 * avg * shift;
        peaks.push((pitch, magnitude));
    }
}

/// `librosa.estimate_tuning(S, sr=22050)` from the collected piptrack peaks
fn estimate_tuning(mut peaks: Vec<(f32, f32)>) -> f64 {
    peaks.retain(|(pitch, _)| *pitch > 0.0);
    if peaks.is_empty() {
        return 0.0;
    }
    let mut magnitudes: Vec<f32> = peaks.iter().map(|(_, m)| *m).collect();
    let mid = magnitudes.len() / 2;
    magnitudes.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let upper = magnitudes[mid];
    let threshold = if magnitudes.len() % 2 == 1 {
        upper
    } else {
        let lower = magnitudes[..mid].iter().cloned().fold(f32::MIN, f32::max);
        (lower + upper) / 2.0
    };

    let edges = linspace(-0.5, 0.5, 101, true);
    let mut counts = [0usize; 100];
    for (pitch, magnitude) in &peaks {
        if *magnitude < threshold {
            continue;
        }
        let octs = (*pitch / (440.0f32 / 16.0)).log2();
        let mut residual = (12.0f32 * octs).rem_euclid(1.0);
        if residual >= 0.5 {
            residual -= 1.0;
        }
        let r = residual as f64;
        // np.histogram: bins are half-open except the last, which includes 0.5
        let bin = edges.partition_point(|e| *e <= r).saturating_sub(1).min(99);
        counts[bin] += 1;
    }
    let best = counts
        .iter()
        .enumerate()
        .fold((0, 0), |best, (i, &c)| if c > best.1 { (i, c) } else { best })
        .0;
    edges[best]
}

/// `librosa.feature.tempo(onset_envelope, sr=22050)` with its defaults (start_bpm 120, ac_size 8, max_tempo 320)
fn tempo(onset_env: &[f64]) -> f64 {
    let sr = LIBROSA_DEFAULT_SR;
    let win_length = ((8.0 * sr) as usize) / HOP;
    let pad = win_length / 2;
    let n = onset_env.len();
    if n == 0 || onset_env.iter().all(|v| *v == 0.0) {
        return 0.0;
    }

    let mut padded = Vec::with_capacity(n + 2 * pad);
    padded.extend(linspace(0.0, onset_env[0], pad, false));
    padded.extend_from_slice(onset_env);
    padded.extend(linspace(0.0, onset_env[n - 1], pad, false).into_iter().rev());

    let window = hann_periodic(win_length);
    let fft_size = (2 * win_length - 1).next_power_of_two();
    let mut planner = FftPlanner::new();
    let forward = planner.plan_fft_forward(fft_size);
    let inverse = planner.plan_fft_inverse(fft_size);

    let frames: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map_init(
            || vec![Complex::new(0.0, 0.0); fft_size],
            |buffer, t| {
                for (i, value) in buffer.iter_mut().enumerate() {
                    *value = if i < win_length {
                        Complex::new(padded[t + i] * window[i], 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                }
                forward.process(buffer);
                for value in buffer.iter_mut() {
                    *value = Complex::new(value.norm_sqr(), 0.0);
                }
                inverse.process(buffer);
                let ac: Vec<f64> = buffer[..win_length].iter().map(|c| c.re / fft_size as f64).collect();
                let max = ac.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let max = if max < f64::MIN_POSITIVE { 1.0 } else { max };
                ac.into_iter().map(|v| v / max).collect()
            },
        )
        .collect();

    let mut best = (0usize, f64::NEG_INFINITY);
    for lag in 0..win_length {
        let tg = frames.iter().map(|f| f[lag]).sum::<f64>() / n as f64;
        let bpm = if lag == 0 { f64::INFINITY } else { 60.0 * sr / (HOP as f64 * lag as f64) };
        if bpm >= 320.0 {
            continue;
        }
        let logprior = -0.5 * (bpm.log2() - 120.0f64.log2()).powi(2);
        let score = (1e6 * tg).ln_1p() + logprior;
        if score > best.1 {
            best = (lag, score);
        }
    }
    60.0 * sr / (HOP as f64 * best.0 as f64)
}

/// `librosa.util.localmax` indices
pub fn local_maxima(x: &[f64]) -> Vec<usize> {
    let n = x.len();
    (0..n)
        .filter(|&i| match i {
            0 => false,
            _ if i == n - 1 => x[i] > x[i - 1],
            _ => x[i] > x[i - 1] && x[i] >= x[i + 1],
        })
        .collect()
}

/// `_analyze_audio` from PyMusicLooper, on the trimmed mono signal
pub fn analyze(y: &[f32], rate: u32) -> Features {
    let sr = rate as f64;
    let n_frames = 1 + y.len() / HOP;
    let stft = Stft::new();

    // Pass 1: global maximum for power_to_db's top_db clip, and pitch peaks for the tuning estimate
    let librosa_freqs = fft_frequencies(LIBROSA_DEFAULT_SR);
    let freq_mask: Vec<bool> = librosa_freqs.iter().map(|f| *f >= 150.0 && *f < 4000.0).collect();
    let (max_power, peaks) = (0..n_frames)
        .into_par_iter()
        .fold(
            || (Vec::with_capacity(N_FFT), [0.0f32; N_BINS], 0.0f32, Vec::new()),
            |(mut buffer, mut power, max, mut peaks), t| {
                stft.power(y, t, &mut buffer, &mut power);
                piptrack_frame(&power, &freq_mask, &mut peaks);
                let frame_max = power.iter().cloned().fold(max, f32::max);
                (buffer, power, frame_max, peaks)
            },
        )
        .map(|(_, _, max, peaks)| (max, peaks))
        .reduce(
            || (0.0f32, Vec::new()),
            |(a_max, mut a_peaks), (b_max, b_peaks)| {
                a_peaks.extend(b_peaks);
                (a_max.max(b_max), a_peaks)
            },
        );
    let tuning = estimate_tuning(peaks);
    let chroma_fb = chroma_filters(LIBROSA_DEFAULT_SR, tuning);
    let mel_fb = mel_filters(sr);
    let weighting = a_weighting(&fft_frequencies(sr));
    let db_floor = 10.0f32 * max_power.max(1e-10).log10() - 80.0;

    // Pass 2: chroma, perceptually weighted dB spectrum (its mel projection and column maximum)
    let frames: Vec<([f32; N_CHROMA], Vec<f64>, f64)> = (0..n_frames)
        .into_par_iter()
        .map_init(
            || (Vec::with_capacity(N_FFT), [0.0f32; N_BINS]),
            |(buffer, power), t| {
                stft.power(y, t, buffer, power);

                let mut raw = [0.0f64; N_CHROMA];
                for (k, p) in power.iter().enumerate() {
                    for (c, r) in raw.iter_mut().enumerate() {
                        *r += chroma_fb[k][c] as f64 * *p as f64;
                    }
                }
                let raw = raw.map(|r| r as f32);
                let length = raw.iter().fold(0.0f64, |m, r| m.max(r.abs() as f64));
                let length = if length < f32::MIN_POSITIVE as f64 { 1.0 } else { length };
                let chroma = raw.map(|r| (r as f64 / length) as f32);

                let weighed: Vec<f64> = power
                    .iter()
                    .zip(&weighting)
                    .map(|(p, w)| w + (10.0f32 * p.max(1e-10).log10()).max(db_floor) as f64)
                    .collect();
                let column_max = weighed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mel: Vec<f64> = mel_fb
                    .iter()
                    .map(|filter| {
                        filter
                            .weights
                            .iter()
                            .zip(&weighed[filter.start..])
                            .map(|(w, s)| *w as f64 * s)
                            .sum()
                    })
                    .collect();
                (chroma, mel, column_max)
            },
        )
        .collect();

    let chroma: Vec<[f32; N_CHROMA]> = frames.iter().map(|f| f.0).collect();

    // librosa.onset.onset_strength(S=mel): lag 1, mean over bands, padded by lag + n_fft // (2 * hop)
    let pad = 1 + N_FFT / (2 * HOP);
    let mut onset_env = vec![0.0f64; n_frames];
    for t in pad..n_frames {
        let (current, previous) = (&frames[t - pad + 1].1, &frames[t - pad].1);
        onset_env[t] = current
            .iter()
            .zip(previous)
            .map(|(c, p)| (c - p).max(0.0))
            .sum::<f64>()
            / N_MELS as f64;
    }

    let db: Vec<f64> = frames.iter().map(|f| 10.0 * f.2.max(1e-10).log10()).collect();
    let db_max = db.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let power_db_max = db.into_iter().map(|v| v.max(db_max - 80.0)).collect();

    let bpm = tempo(&onset_env);
    Features {
        chroma,
        power_db_max,
        onset_env,
        bpm,
        tuning,
    }
}
