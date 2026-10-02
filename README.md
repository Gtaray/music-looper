# music-looper

A small command-line tool that finds seamless loop points in music and reads/writes loop tags. It is a Rust port of [PyMusicLooper](https://github.com/arkrow/PyMusicLooper)'s loop detection, with no Python required, and prints JSON so other programs can use it.

## Install

On Arch Linux, commit your changes and run:

```
packaging/arch/install_or_update.sh
```

This builds the committed `main` branch into a `music-looper` package and installs `/usr/bin/music-looper`. Remove it with `packaging/arch/uninstall.sh`.

Elsewhere, build with Cargo:

```
cargo build --release
```

The binary is `target/release/music-looper`.

## Usage

All commands print JSON on stdout. Errors go to stderr with a non-zero exit code. Positions are in samples at the file's own sample rate.

### Find loop points

```
music-looper analyze track.flac
```

```json
{"sampleRate":44100,"start":62513,"end":2479792,"candidates":[{"start":62513,"end":2479792,"score":0.999991670364779,"noteDistance":0.0006679611,"loudnessDifference":0.019816760379832488}, …]}
```

`start`/`end` is the best loop; `candidates` lists the other loops considered, best first. That list can run to several MB; `--max-candidates N` keeps only the best N (N ≥ 1, default all).

### Read loop tags

```
music-looper read-tags track.flac
```

```json
{"sampleRate":44100,"start":62513,"end":2479792,"startTag":"LOOP_START","endTag":"LOOP_END"}
```

Tag names are detected automatically (`LOOPSTART`, `LOOP_START`, `LOOPLENGTH`, `LOOP_END`, …). An end tag with `LEN` or `OFFSET` in its name is read as a length. `start` and `end` are `null` when the file has no loop tags.

### Write loop tags

```
music-looper write-tags track.flac --start 62513 --end 2479792
```

Writes `LOOP_START`/`LOOP_END` into the file, replacing any existing loop tags. Use `--start-tag` and `--end-tag` for other names; an end tag name containing `LEN` or `OFFSET` is written as a length.

## Supported formats

- Analysis: FLAC, WAV, Ogg Vorbis, MP3, and Matroska/WebM files containing those codecs. Not supported: Opus (symphonia has no Opus decoder) and M4A/AAC (symphonia ignores the MP4 edit list, so the encoder delay isn't trimmed and loop points would be about 1024 samples late).
- Tags: FLAC (Vorbis comments).

## How it works

The analysis follows PyMusicLooper 3.6.0 and the librosa functions it uses, and gives the same loop points. The one difference is how candidate loop points are chosen: instead of librosa's beat tracker, it uses every peak of the onset strength.

## License

MIT. See [LICENSE](LICENSE), which also includes the PyMusicLooper (MIT) and librosa (ISC) notices.
