# TODO

## Streaming Quality

- **Long-form audio testing.** Need 10-30 minute test clips (podcast, lecture) to stress periodic reset and long-term drift. Current max is 119s.

## Performance

- **Encoder partial-tail caching.** The partial tail (audio beyond the last complete 8s window) is re-encoded from scratch every chunk. Could cache and extend.
- **Pre-warm more encoder buckets.** New bucket sizes trigger JIT compilation on first encounter (~3-4s). Could pre-warm common sizes during startup.
- **TTS GPU-resident decode loop.** Eliminate per-step Python dispatch overhead (~85ms/step vs ~7ms GPU compute). Unroll N steps per JIT call, check EOS every Nth step.

## Testing

- **Streaming session WER gate.** Currently only in `tests/test_stream_quality.py`, not in CI.
- **Recovery mechanism coverage.** Verify stagnation detection triggers, periodic reset works.
- **Long-form audio benchmarks.** Current test data maxes at 119s. Need varied speakers, background noise, topic changes.

## Code

- **Separate modules.** `asr.py` is ~1200 lines. Could split AudioEncoder, ASR model, StreamingSession if it keeps growing.

## Features

- **TTS `/v1/audio/speech` endpoint.** Wire into server with `--tts-model` flag.
- **Language switch detection.** Track pending token instability + decode confidence. On detection, `_reanchor()` with zero carry tokens.
- **Speaker diarization.** ECAPA-TDNN embeddings → agglomerative clustering → speaker-labeled segments.
- **Server-side noise reduction.** Spectral gating or high-pass filter for upload-file path.
