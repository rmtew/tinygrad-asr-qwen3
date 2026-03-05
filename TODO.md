# TODO

## Performance

- **Decoder KV cache reuse across streaming chunks.** Currently each streaming chunk re-prefills from `start_pos=0`. The C implementation compares encoder embeddings to find a reuse point, only re-prefilling new tokens. Could save ~200ms/chunk.
- **Larger streaming chunks.** Test 4s or 8s `chunk_sec` to reduce encoder/prefill/decode cycles per file. Tradeoff: higher latency for first text output.
- **Encoder partial-tail caching.** The partial tail (audio beyond the last complete 8s window) is re-encoded from scratch every chunk. Could cache and extend instead.
- **Pre-warm more encoder buckets.** New bucket sizes (e.g. 2400 frames for ~20s audio) trigger JIT compilation on first encounter (~3-4s hit). Could pre-warm common sizes during startup.

## Testing

- **Regression tests.** Both single-file and streaming modes need tests that prevent quality regressions:
  - Quick smoke tests: JFK transcription exact match, a few short LibriSpeech files.
  - Longer benchmark: 30+ file LibriSpeech WER + RTF check against known baselines.
  - Streaming correctness: verify that each chunk produces the full transcription seen so far (not just the tail).
  - Performance gate: fail if RTF regresses beyond a threshold.

## Code Cleanup

- **Server refactor.** `asr.py` server code currently uses a basic HTTP handler adapted from tinygrad's viz server. Consider subclassing or reusing `llm.py`'s OpenAI-compatible server infrastructure instead — it already handles `/v1/` routing, JSON responses, and streaming.
- **Separate encoder/decoder modules.** `asr.py` is ~700 lines. The `AudioEncoder`, `ASR` model, server handler, and CLI could be split into separate files.

## Features

- **Microphone/live transcription mode.** `llm.py` runs an OpenAI chat server designed for LLM conversation. If we can optionally use the microphone, users could see their speech transcribed in real time — valuable for evaluating how the model responds in actual usage. The LLM response side may or may not be running; if no LLM is connected, just show transcriptions without errors. This would make it easy for users to test ASR quality interactively by speaking and watching the output.
