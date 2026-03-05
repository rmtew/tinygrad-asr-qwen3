# tinygrad-asr-qwen3

Qwen3-ASR speech recognition via [tinygrad](https://github.com/tinygrad/tinygrad). Single-file implementation with OpenAI-compatible API.

## Features

- Loads GGUF models (F16/F32) via tinygrad's built-in `gguf_load`
- Reuses tinygrad's `Transformer` for the decoder — no duplicated code
- OpenAI-compatible `/v1/audio/transcriptions` endpoint
- Interactive CLI mode
- GPU accelerated (CUDA, etc.)

## Install

```bash
pip install tinygrad
```

## Usage

### CLI

```bash
# Transcribe a WAV file (auto-downloads model on first run)
python asr.py audio.wav

# Use a local GGUF file
python asr.py audio.wav --model path/to/qwen3-asr-0.6b-f16.gguf

# Interactive mode
python asr.py
```

### Server

```bash
# Start OpenAI-compatible server on port 8090
python asr.py --serve

# Custom port
python asr.py --serve 9000
```

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

## Models

| Model | Size | Source |
|-------|------|--------|
| `qwen3-asr:0.6b` (default) | 1.88 GB | [FlippyDora/qwen3-asr-0.6b-GGUF](https://huggingface.co/FlippyDora/qwen3-asr-0.6b-GGUF) |

## Architecture

The ASR model has two parts:

1. **Audio Encoder** — Conv2D stem (3× stride-2) + 18-layer transformer with windowed attention + linear projection. Implemented in `asr.py` as `AudioEncoder`.

2. **Text Decoder** — Standard Qwen3 decoder (28 layers, GQA, RoPE, SwiGLU). Loaded directly into tinygrad's `Transformer` class from `tinygrad.apps.llm` — architecturally identical, zero code duplication.

The GGUF contains both encoder and decoder weights. The state dict is split by prefix (`audio.encoder.*` vs `blk.*`), each part loaded into its respective model.

## Requirements

- Python 3.10+
- tinygrad
- numpy
- WAV input (16kHz mono recommended; auto-resamples other rates)
