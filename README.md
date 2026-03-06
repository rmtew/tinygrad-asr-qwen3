# tinygrad-asr-qwen3

Qwen3-ASR speech recognition via [tinygrad](https://github.com/tinygrad/tinygrad). Single-file implementation with OpenAI-compatible API and browser-based microphone transcription.

## Features

- Loads GGUF models (F16/F32) via tinygrad's built-in `gguf_load`
- Reuses tinygrad's `Transformer` for the decoder — no duplicated code
- OpenAI-compatible `/v1/audio/transcriptions` endpoint
- Web UI with live microphone transcription and file drag-and-drop
- Streaming mode for long audio (2s chunks, sliding window, KV cache reuse)
- GPU accelerated (CUDA, etc.)

## Install

```bash
pip install tinygrad
```

For audio formats other than WAV, [ffmpeg](https://ffmpeg.org/) must be on PATH.

## Quick Start

```bash
# Transcribe a file (local GGUF)
CUDA=1 python asr.py --model path/to/qwen3-asr-0.6b-f16.gguf audio.wav

# Or download a known model
CUDA=1 python asr.py --model qwen3-asr:0.6b audio.wav

# Start server with web UI
CUDA=1 JITBEAM=2 python asr.py --model path/to/qwen3-asr-0.6b-f16.gguf --serve
# Open http://localhost:8090 in your browser
```

## Usage

`--model` is required. Pass a path to a local GGUF file, or a known model name to download.

### CLI

```bash
# Transcribe a WAV file
python asr.py --model model.gguf audio.wav

# Interactive mode (type paths, get transcriptions)
python asr.py --model model.gguf
```

### Server + Web UI

```bash
# Start on default port 8090
python asr.py --model model.gguf --serve

# Custom port
python asr.py --model model.gguf --serve 9000
```

Open `http://localhost:8090` in your browser to get the transcription UI:
- **Record** — click to start/stop microphone recording. Transcription updates live every 2 seconds while recording.
- **Upload file** — or drag-and-drop an audio file onto the page.

Audio longer than 32 seconds automatically uses streaming mode (2s chunks with encoder window caching and decoder KV cache reuse).

### API

```bash
curl -X POST http://localhost:8090/v1/audio/transcriptions \
  -F "file=@audio.wav"
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/v1", api_key="x")
result = client.audio.transcriptions.create(model="qwen3-asr", file=open("audio.wav", "rb"))
print(result.text)
```

## Performance

With `JITBEAM=2` on an RTX 3070 Ti Laptop (warm, JIT cached):

| Mode | Audio | RTF | Notes |
|------|-------|-----|-------|
| Per-file | 11s (JFK) | **0.05** | Encoder 109ms, total 591ms |
| Per-file | LibriSpeech 30-utt | **0.07** | WER 0.65% |
| Streaming | 11s (JFK, 6 chunks) | **0.16** | KV cache reuse: 245 tokens |
| Streaming | LibriSpeech 30-utt | **0.20** | WER 2.10% |
| Streaming | 119s (movie clip) | **0.16** | Sequential encoder, 19 windows |

### JITBEAM and Disk Cache

`JITBEAM=2` runs beam search to find optimal GPU kernels. Results are cached to disk (`~/.cache/tinygrad/cache.db`) so the cost is paid only once:

| Run | Time | What happens |
|-----|------|-------------|
| First ever | ~11 min | Beam search for all kernel variants → written to disk cache |
| Subsequent | ~65 sec | JIT capture + cache hits (no beam search) |
| Without JITBEAM | ~15 sec | JIT with default heuristic kernels (2-3× slower inference) |

The disk cache is ~1.5 GB (28K PTX kernels). It persists across runs and is versioned by tinygrad — a tinygrad upgrade will trigger a one-time re-warmup.

**Recommendation:** Always use `JITBEAM=2` for the server. The first run is slow but all subsequent runs are fast. Without it, inference still works but RTF is roughly 2-3× worse.

```bash
# First run: slow (beam search), but results are cached
CUDA=1 JITBEAM=2 python asr.py --serve

# All future runs: fast (~65s warmup, then serving)
CUDA=1 JITBEAM=2 python asr.py --serve
```

## Testing

```bash
# Quick tests (~30s): JFK exact match, streaming, 5 diverse LibriSpeech files
CUDA=1 JITBEAM=2 python test.py

# With performance regression gates
CUDA=1 JITBEAM=2 python test.py --perf

# Full benchmark: 30-file LibriSpeech WER + RTF (per-file and streaming)
CUDA=1 JITBEAM=2 python test.py --full
```

## Models

| Model | Size | Source |
|-------|------|--------|
| `qwen3-asr:0.6b` (default) | 1.88 GB | [FlippyDora/qwen3-asr-0.6b-GGUF](https://huggingface.co/FlippyDora/qwen3-asr-0.6b-GGUF) |

## Architecture

The ASR model has two parts:

1. **Audio Encoder** — Conv2D stem (3× stride-2) + 18-layer transformer with windowed attention + linear projection. Two-path JIT: batched for known bucket sizes (≤32s), sequential single-window fallback for any length.

2. **Text Decoder** — Standard Qwen3 decoder (28 layers, GQA, RoPE, SwiGLU). Loaded directly into tinygrad's `Transformer` class from `tinygrad.apps.llm` — architecturally identical, zero code duplication.

The GGUF contains both encoder and decoder weights. The state dict is split by prefix (`audio.encoder.*` vs `blk.*`), each part loaded into its respective model.

### Streaming

For long audio, `transcribe_stream` processes 2-second chunks:
- Encoder windows cached per chunk (no re-encoding earlier chunks)
- Decoder KV cache reused across chunks (row-by-row embedding comparison finds longest matching prefix)
- Sliding window eviction keeps max 4 encoder windows (32s context)
- Each chunk produces a complete transcription of all audio heard so far

## Requirements

- Python 3.10+
- tinygrad
- numpy
- ffmpeg on PATH (for non-WAV audio: webm, ogg, mp3, flac, etc.)
