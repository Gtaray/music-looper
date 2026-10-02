# AGENTS.md

Notes for AI coding agents working in this repo.

## Project

`music-looper`: a small Rust command-line tool that finds seamless loop points in audio files and reads/writes loop tags. It replaces PyMusicLooper for the owner's Kenku FM fork (`~/repos/kenku-fm`), which will call it as a subprocess and parse its JSON output. Goal: no Python in the app.

The algorithm is a port of upstream **PyMusicLooper 3.6.0** (`arkrow/PyMusicLooper`, commit `78313f8`). A checkout of Soteyl's fork is at `~/repos/PyMusicLooper` (HEAD `711aca6`); use `git -C ~/repos/PyMusicLooper show 78313f8:pymusiclooper/analysis.py` (and `audio.py`, `core.py`) for the upstream reference. The fork's `analysis.py`/`audio.py` changes are leftover "fast mode" plumbing; don't port them.

## Decisions (agreed with the owner)

- **Beats:** instead of librosa `beat_track` + PLP, candidate points are **every local maximum of the onset strength envelope** (`librosa.util.localmax(onset_env)`). Tested against upstream on 6 FLACs (see Reference results): matches or lands within a few hundred samples of upstream's loop lengths. `librosa.onset.onset_detect` peaks were worse. Port the full beat tracker only if needed later.
- **Keep upstream quirks so results match** (maybe revisit once there's a solid baseline):
  - `chroma_stft`, `beat_track`/`tempo`, `plp` are called without `sr`, so librosa assumes **sr = 22050** even for 44.1/48/192 kHz audio (chroma filterbank, tempo estimate in BPM, tempo prior).
  - `perceptual_weighting` returns **dB**, and the mel spectrogram, onset strength and `power_db = power_to_db(S_weighed, ref=np.median)` are all computed on those dB values.
  - `_prioritize_duration` reads `loop_end - loop_start` before they are set (both 0), so it never reorders anything. **Leave it out** to match.
- **Tags:** read and write (write only on an explicit request from Kenku, never automatically).
- **Formats:** FLAC first (the owner's library is all FLAC). No Opus yet (symphonia 0.6.1 has no Opus decoder).
- **Comparison env must be fully transient:** venv lives in `/tmp/pml-compare` (see below), never in a repo.

## Planned CLI (JSON on stdout, errors on stderr, non-zero exit on failure)

- `analyze <path> [--max-candidates N]` → `{ start, end, sampleRate, candidates: [{ start, end, score, noteDistance, loudnessDifference }] }` (samples, at the file's native rate). Candidates are all scored pairs, best first (can be MBs of JSON); `--max-candidates` (≥ 1, default all) keeps the best N. Kenku passes `--max-candidates 1`.
- `read-tags <path>` → loop tags found (auto-detect names, see below)
- `write-tags <path> --start N --end N [--start-tag LOOP_START --end-tag LOOP_END]`

## Pipeline to port (upstream behaviour, n_fft = 2048, hop = 512 everywhere)

1. **Load** (`librosa.load(sr=None, mono=False)`, float32): native rate, all channels. `total_duration = samples / rate`.
2. **Mono** = mean of channels; error if all zero; normalise by max |x|.
3. **Trim** (`effects.trim(top_db=40)`): RMS per frame (frame 2048, hop 512, centered with zero pad 1024 each side), `amplitude_to_db(rms, ref=max)` (amin 1e-5, ref^2, power_to_db), non-silent = db > -40; start = first_nonsilent*512, end = min(len, (last+1)*512). `trim_offset` = start.
4. **STFT** of trimmed mono: periodic Hann (`get_window('hann', 2048, fftbins=True)`), center=True with **zero** padding of 1024 on both sides, rfft → 1025 bins. `S_power = |S|^2`.
5. **perceptual_weighting**: `A_weighting(fft_freqs(rate))` (min_db -80; f=0 → -inf → -80) + `power_to_db(S_power)` (ref 1, amin 1e-10, top_db 80 → clip to max-80). Result `S_weighed` (dB, float64).
6. **Mel** = `filters.mel(sr=rate, n_fft=2048, n_mels=128, fmin=0, fmax=8000, htk=False, norm='slaney')` @ `S_weighed`. Slaney mel scale (linear below 1 kHz, f_sp=200/3, logstep=ln(6.4)/27).
7. **Chroma** = `chroma_stft(S=S_power)` with **sr=22050**: `tuning = estimate_tuning(S=S_power, sr=22050, bins_per_octave=12)` (piptrack fmin 150, fmax 4000 clipped to sr/2, threshold 0.1·max per frame, localmax over freq, parabolic interpolation; keep mags ≥ median of positive-pitch mags; `pitch_tuning` histogram, resolution 0.01, bins linspace(-0.5,0.5,101), take left edge of argmax bin). Then `filters.chroma(sr=22050, n_fft=2048, tuning, ctroct=5, octwidth=2, norm=2, base_c=True)` @ S_power, then normalise each frame by its max (norm=inf; frames with max < float32 tiny left unscaled).
8. **power_db** = `power_to_db(S_weighed, ref=np.median)`: 10·log10(max(1e-10, x)) − 10·log10(max(1e-10, median(S_weighed over all bins and frames))), then clip to max−80.
9. **Onset strength** = `onset_strength(S=mel)`: diff along time with lag 1, `max(0, ·)`, mean over the 128 mel bands, left-pad 3 zeros (lag 1 + 2048//(2·512)), truncate to n_frames.
10. **BPM** (still needed for scoring) = `feature.tempo(onset_envelope, sr=22050, hop 512, start_bpm 120, std_bpm 1, ac_size 8, max_tempo 320)`: tempogram win_length = time_to_frames(8, sr=22050) = 344; onset env padded both sides by 172 with `linear_ramp` to 0; frames of 344 (hop 1), × periodic Hann(344), autocorrelate (FFT, keep first 344 lags), normalise each frame by max (inf norm), mean over time; bpms[k] = 60·22050/(512·k) (k=0 → inf); logprior = −0.5·(log2(bpm) − log2(120))²; set logprior[:argmax(bpms < 320)] = −inf; best = argmax(log1p(1e6·tg) + logprior).
11. **Candidate points** = `localmax(onset_env)`: x[i] > x[i−1] and x[i] ≥ x[i+1]; first element false; last = x[−1] > x[−2].
12. **Loop length limits** (frames): min = time_to_frames(int(0.35·total_duration)) (int truncation of seconds!), max = time_to_frames(total_duration); min = max(1, min). time_to_frames(t) = floor(int(t·rate) / 512).
13. **_find_candidate_pairs**: deviation[idx] = ‖chroma[:, beat]·0.0875‖₂; for each end beat (ascending), for each start beat ascending: len = end−start; if len < min → break; if len > max → continue; note_distance = ‖chroma[:,end]−chroma[:,start]‖₂ ≤ deviation[idx_of_end]; loudness_difference = |max(power_db[:,end]) − max(power_db[:,start])| ≤ 0.5 → keep (start, end, note_distance, loudness_difference). Error "no loop points" if none.
14. **Scoring** (`_assess_and_filter_loop_pairs`): test_offset = floor(int((12 / (bpm/60))·rate) / 512) frames; if > n_frames → n_frames // 4. If ≥ 100 candidates → prune: among values > 1e-3, db threshold = 50th percentile (if > 3 such values, else max of all), note threshold = 75th percentile (same rule); keep db ≤ max(0.25, db_thr) and note ≤ note_thr (numpy percentile = linear interpolation). weights = geomspace(max(2, test_offset // 12), 1, test_offset). score = max(lookahead, lookbehind) where each is the weighted average of per-frame cosine similarity of chroma columns (denominator max(‖a‖‖b‖, 1e-10)), lookbehind uses reversed weights; windows clipped at the edges and zero-padded to test_length (see `_calculate_subseq_beat_similarity`). Sort by score descending (Python `sorted` is stable).
15. **Back to samples**: if trim_offset > 0: frame = floor((frame·512 + trim_offset) / 512). sample = frame·512. Then `nearest_zero_crossing` (Audacity algorithm) on the **original multi-channel** audio at the native rate: window = int(max(1, rate/100)), see upstream code.
16. Best loop = first candidate.

## Tags (PyMusicLooper `core.py`)

- Start tag names, in order: `COMMENT=LOOPPOINT`, `LOOP`, `LOOP_BEGIN`, `LOOPPOINT`, `LOOPS`, `LOOPSTART`, `LOOP_START`, `LOOP-START`, `UM3.STREAM.LOOPPOINT.START`, `XIPH_CUE_LOOPSTART`.
- End tag names: `LOOPE`, `LOOPEND`, `LOOP_END`, `LOOP-END`, `LOOPLENGTH`, `LOOP_LENGTH`, `LOOP-LENGTH`, `XIPH_CUE_LOOPEND`.
- End tag is a **length** (start + value) if its name contains `LEN` or `OFFSET` (case-insensitive). Values are integers in samples; swap if start > end.
- lofty 0.25.4's generic `ItemKey` has no custom-key variant, so custom loop tags need per-format APIs: Vorbis comments (FLAC/Ogg) take arbitrary keys; ID3v2 needs `TXXX`; MP4 needs freeform atoms.

## Crates

- symphonia 0.6.1 (default features cover FLAC/WAV/Ogg Vorbis; add `mp3` etc. later). 0.6 API differs from 0.5: see `~/.cargo/registry/src/*/symphonia-0.6.1/examples/basic-interleaved.rs` (`get_probe().probe(&hint, mss, fmt_opts, meta_opts)`, `format.default_track(TrackType::Audio)`, `get_codecs().make_audio_decoder(track.codec_params.as_ref().unwrap().audio().unwrap(), &dec_opts)`, `format.next_packet()` returns `Result<Option<Packet>>`, `audio_buf.spec().rate()`, `audio_buf.spec().channels().count()`, `audio_buf.copy_to_slice_interleaved(&mut Vec<f32>)`).
- rustfft, rayon, lofty 0.25.4, clap (derive), serde/serde_json, anyhow.

## Reference / comparison

- `reference/` holds the comparison scripts, `tracks.txt` (6 test FLACs from `~/Music`, a symlink to the NAS `/mnt/unraid/media/music`), `results/` (upstream output: `original`, `onset` = onset_detect peaks, `localmax` = chosen variant), and `librosa-excerpts/` (librosa 0.11.0 sources the pipeline uses, docstrings stripped, dumped with `show.py`).
- Recreate the transient env (needs network):
  ```
  set T /tmp/pml-compare; mkdir -p $T/src; and git -C ~/repos/PyMusicLooper archive 78313f8 | tar -x -C $T/src; and python3 -m venv $T/venv; and $T/venv/bin/pip install --quiet $T/src
  $T/venv/bin/python reference/reference.py [--variant localmax] (cat reference/tracks.txt) > out.jsonl
  $T/venv/bin/python reference/summary.py a.jsonl b.jsonl ...
  ```
  Works on the system Python 3.14 (librosa 0.11.0, numba 0.68.0). Delete `/tmp/pml-compare` when done.
- `reference.py` records intermediate stats (`chroma_sum`, `chroma_col100`, `power_db_sum`, `bpm`, beat count, `trim_offset`) so the Rust port can be checked stage by stage.

## Status

- Done and verified: `analyze`, `read-tags`, `write-tags`.
  - `analyze --debug` matches PyMusicLooper (localmax variant) **exactly** on all 6 reference tracks: trim offset, frame count, BPM, peak count, candidate count, and top-3 loop points/scores. 6–15× faster (0.1–0.7 s vs 0.6–10.8 s).
  - Second, randomly picked set (`reference/tracks-2.txt`, results in `reference/results/*-2.jsonl`: 44.1/48/192 kHz, 16/24-bit, 39 s to 9 min) also matches exactly; top-5 loop points identical on all 11 tracks, scores within 5e-7.
  - Tags tested on temp copies: reads `LOOPSTART`/`LOOPLENGTH` (length → end), replaces all known loop tags on write (so stale ones can't shadow the new values), other Vorbis comments untouched, picture bytes identical (block order may change), `flac -t` passes. FLAC only for now.
- Code: `src/audio.rs` (symphonia decode), `src/features.rs` (librosa port), `src/looper.rs` (analysis.py port), `src/tags.rs`, `src/main.rs` (clap CLI).
- Power dB: only per-frame maxima are kept and the `ref=np.median` offset is dropped, since it cancels out in loudness differences (avoids holding the whole spectrogram).
- Re-run the comparison: `fish reference/run-rust.fish > /tmp/pml-compare/rust.jsonl; /tmp/pml-compare/venv/bin/python reference/compare.py reference/results/ref-localmax.jsonl /tmp/pml-compare/rust.jsonl`.
- Next: Kenku integration (main-process bridge, read tags before analysing, no automatic tag writes, analysis state kept out of persisted playlists). The owner wants music-looper kept as a **separate package**; don't change Kenku from this repo.

## Packaging

- MIT (`LICENSE` also carries PyMusicLooper's MIT and librosa's ISC notices).
- Arch: `packaging/arch/PKGBUILD` (package `music-looper`, installs `/usr/bin/music-looper`) builds the committed `main` of this repo via `git+file`, with `cargo fetch --locked` / `cargo build --frozen --release`. Use `packaging/arch/install_or_update.sh` / `uninstall.sh`; the owner runs `sudo` himself. makepkg rewrites `pkgver=` on every build; its output is git-ignored.

## Agent environment

- Terminal is **fish**: no heredocs; an unmatched glob aborts the whole command line (don't use `**`); never put `$|` or `$"` inside double quotes; use `command cp/mv`, `/usr/bin/rm`.
- The owner commits and pushes himself; explain plans before changes and keep scope to what was asked.
