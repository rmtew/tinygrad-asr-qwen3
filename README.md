# tinygrad-asr-qwen3

Speech recognition and LLM chat via [tinygrad](https://github.com/tinygrad/tinygrad). Single-file implementation with OpenAI-compatible API and browser-based UIs.

## Features

- **ASR** — Qwen3-ASR 0.6B with live microphone streaming, file transcription
- **LLM** — Qwen3.5 chat (0.8B–9B) with streaming SSE responses
- **Voice + text chat UI** — record voice → ASR transcription → LLM response
- Loads GGUF models (F16/F32) via tinygrad's built-in `gguf_load`
- Reuses tinygrad's `Transformer` for ASR decoder and LLM — no duplicated code
- OpenAI-compatible `/v1/audio/transcriptions` and `/v1/chat/completions`
- WebSocket streaming with committed/pending confidence display
- Dispatch queue for thread-safe multi-model GPU inference
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
# ASR only
python asr.py --model model.gguf --serve

# ASR + LLM chat
python asr.py --model model.gguf --llm-model Qwen3.5-0.8B-Q4_K_M.gguf --serve

# Custom port
python asr.py --model model.gguf --serve 9000
```

Open `http://localhost:8090` for the ASR UI, or `http://localhost:8090/chat` for the chat UI:

**ASR page (`/`):**
- **Record** — click to start/stop microphone recording. Audio streams to the server over WebSocket. Transcription updates live every 2 seconds. Committed text appears in white; the provisional rollback tail appears in dim italic.
- **Upload file** — or drag-and-drop an audio file onto the page for one-shot transcription.
- **Stats panel** — shows RTF, latency breakdown (encoder/prefill/decode), encoder windows, KV cache reuse, committed/pending token counts.

**Chat page (`/chat`):**
- **Text input** — type a message and press Enter (or click send).
- **Voice input** — click 🎤 to record, click again to stop. Live ASR preview shows committed/pending text. On stop, the transcript is sent to the LLM automatically.
- **Streaming responses** — LLM tokens appear in real-time. `<think>` tags are stripped from display.
- Mic and send buttons are disabled during LLM generation to prevent concurrent inference.

### API

One-shot file transcription (OpenAI-compatible):

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

Streaming via WebSocket (`/ws`):

```
Client → Server:
  Text frame:   {"type":"start"}           → create session
  Binary frame: Int16 LE PCM (16kHz mono)  → feed audio chunk
  Text frame:   {"type":"end"}             → finalize

Server → Client:
  Text frame:   {"committed":"...","pending":"...","stats":{...}}
```

The WebSocket server runs on HTTP port + 1 (default 8091).

## Performance

With `JITBEAM=2` on an RTX 3070 Ti Laptop (warm, JIT cached):

| Mode | Audio | RTF | WER | Notes |
|------|-------|-----|-----|-------|
| Per-file | 11s (JFK) | **0.05** | — | Encoder 109ms, total 591ms |
| Per-file | LibriSpeech 30-utt | **0.07** | **0.65%** | |
| Streaming | 11s (JFK, 6 chunks) | **0.16** | **0.0%** | KV cache reuse: 245 tokens |
| Streaming | LibriSpeech 30-utt | **0.20** | **2.10%** | |
| Streaming | 119s (movie clip) | **0.16** | ~9% | Sequential encoder, 19 windows |

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
CUDA=1 JITBEAM=2 python tests/test.py

# With performance regression gates
CUDA=1 JITBEAM=2 python tests/test.py --perf

# Full benchmark: 30-file LibriSpeech WER + RTF (per-file and streaming)
CUDA=1 JITBEAM=2 python tests/test.py --full

# Streaming session diagnostics (feed() with per-chunk logging)
python tests/test_session.py path/to/audio.wav

# Streaming vs per-file WER comparison
python tests/test_stream_quality.py

# WebSocket protocol tests (no model required)
python tests/test_ws.py

# Parameter sweep: grid-search chunk_sec × rollback
CUDA=1 BEAM=2 python sweep_params.py captures/*.wav
```

## Models

| Model | Size | Format | Source |
|-------|------|--------|--------|
| `qwen3-asr:0.6b` | 1.88 GB | GGUF (F16) | [FlippyDora/qwen3-asr-0.6b-GGUF](https://huggingface.co/FlippyDora/qwen3-asr-0.6b-GGUF) |
| Qwen3-TTS 0.6B CustomVoice | 1.73 GB | GGUF (F16) | Convert with `tools/convert_tts_gguf.py` from [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) |
| Qwen3-TTS Vocoder | 651 MB | safetensors (F32) | Ships with Qwen3-TTS as `Qwen3-TTS-Tokenizer-12Hz/` |

## TTS

Qwen3-TTS 0.6B text-to-speech with 9 built-in voices. Uses CustomVoice model (named voices via `spk_id`). Loads talker weights from F16 GGUF (converted from safetensors).

### Setup

```bash
# Download CustomVoice model + vocoder from HuggingFace, then convert to GGUF:
python tools/convert_tts_gguf.py path/to/qwen3-tts-12hz-0.6b-customvoice
# Produces qwen3-tts-0b6-custom_voice-f16.gguf (1.73 GB) in the model directory
# Vocoder (Qwen3-TTS-Tokenizer-12Hz/) must be a sibling directory
```

### Voices

`serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`

### Pipeline

```
Text → BPE tokenize → interleaved text+codec prefix embeddings (+voice embedding)
  → Talker LM (28-layer Qwen3, KV cache, top-k sampling) → codec tokens
  → Code Predictor (5-layer, 15 sub-codebook tokens per step)
  → Vocoder (BigVGAN: RVQ → pre-transformer → upsample → 24kHz audio)
```

### Performance (no JITBEAM, RTX 3070 Ti Laptop)

| Text | Steps | Audio | RTF | C ref RTF |
|------|-------|-------|-----|-----------|
| Short (2 words) | 13 | 1.0s | 8.6 | ~1.7 |
| Medium (9 words) | 44 | 3.5s | 4.1 | ~1.7 |
| Long (20 words) | 92 | 7.4s | 3.1 | ~1.7 |

JITBEAM=2 should close the RTF gap significantly.

## Architecture

The ASR model has two parts:

1. **Audio Encoder** — Conv2D stem (3× stride-2) + 18-layer transformer with windowed attention + linear projection. Two-path JIT: batched for known bucket sizes (≤32s), sequential single-window fallback for any length.

2. **Text Decoder** — Standard Qwen3 decoder (28 layers, GQA, RoPE, SwiGLU). Loaded directly into tinygrad's `Transformer` class from `tinygrad.apps.llm` — architecturally identical, zero code duplication.

The GGUF contains both encoder and decoder weights. The state dict is split by prefix (`audio.encoder.*` vs `blk.*`), each part loaded into its respective model.

### Streaming

`StreamingSession` implements C-style streaming (matching `qwen_asr.c stream_impl`):

- **2s chunks** — audio arrives as PCM deltas via WebSocket (or HTTP)
- **Encoder window cache** — completed 8s windows cached, only partial tail re-encoded each chunk. Max 4 windows (32s sliding context)
- **Decoder KV cache reuse** — row-by-row embedding comparison finds longest matching prefix (~245 tokens reused for 11s audio)
- **Text prefix feedback** — previous decoded tokens (minus rollback) fed back as decoder context, anchoring output after encoder windows are evicted
- **Monotonic commit** — LCP against previous stable tokens, overlap dedup, emit only new tokens. Committed text never removed
- **Recovery** — stagnation detection (4+ chunks with no advance), degenerate repeat detection, periodic reset every 45 chunks (~90s). Resets KV cache and text state but preserves encoder cache

## Requirements

- Python 3.10+
- tinygrad
- numpy
- ffmpeg on PATH (for non-WAV audio: webm, ogg, mp3, flac, etc.)
