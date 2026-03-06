# TODO

## Done

- ~~**Decoder KV cache reuse across streaming chunks.**~~ 245 tokens reused for JFK 11s, streaming RTF 0.22 → 0.16.
- ~~**Server with web UI.**~~ OpenAI-compatible server, embedded HTML mic UI, file upload, auto-routing.
- ~~**Microphone/live transcription.**~~ Record button, live updates every 2s, drag-and-drop.
- ~~**C-style streaming architecture.**~~ Text prefix feedback, monotonic commit, overlap dedup, repeat suppression, stagnation recovery, periodic reset. Matches `qwen_asr.c stream_impl` design.
- ~~**Diagnostic logging.**~~ Per-chunk server log: decoded text, LCP/commit flow, emit delta, pending tail, recovery events. Correlates browser sessions to quality issues.
- ~~**Quality test infrastructure.**~~ `test_stream_quality.py`: streaming vs per-file WER on clean, noisy, real mic audio. `test_session.py`: diagnostic harness for StreamingSession.feed().
- ~~**Browser noise suppression.**~~ `getUserMedia` with `noiseSuppression`, `echoCancellation`, `autoGainControl`.
- ~~**WebSocket streaming.**~~ Replaces HTTP polling. Binary Int16 PCM frames, JSON responses. Lower overhead per chunk.
- ~~**Confidence display.**~~ Committed text (bright) vs pending rollback tail (dim italic) in web UI.

## Streaming Quality

Current streaming WER gap vs per-file (per-file is 0%):
- Clean 11s: 0% | Clean 119s: 9.3% | Real mic 47s: 23.3%

- **Investigate remaining 9% clean baseline gap.** Each 2s chunk generates ~5-13 tokens continuation with rollback=5, so only ~0-8 net new tokens committed per chunk. Fundamental to the architecture — model under-generates relative to speech rate. Options:
  - Test larger `chunk_sec` (4s, 8s) — more tokens per chunk, fewer rollback losses, higher latency.
  - Test `rollback=3` without recovery mechanisms (simpler, may work for clean audio).
  - Server-side noise reduction (`noisereduce` spectral gating or high-pass filter) for mic audio.
- **Long-form audio testing.** Need 10-30 minute test clips (podcast, lecture) to stress the periodic reset and long-term drift behavior. Current max is 119s.

## Performance

- **Larger streaming chunks.** Test 4s or 8s `chunk_sec` to reduce encoder/prefill/decode cycles. Tradeoff: higher latency for first text output.
- **Encoder partial-tail caching.** The partial tail (audio beyond the last complete 8s window) is re-encoded from scratch every chunk. Could cache and extend.
- **Pre-warm more encoder buckets.** New bucket sizes trigger JIT compilation on first encounter (~3-4s). Could pre-warm common sizes during startup.

## Testing

- **Regression tests.** `test.py` covers: JFK exact match (per-file + streaming), 5-file LibriSpeech WER, RTF gates. Could add:
  - Streaming session WER gate (currently only in `test_stream_quality.py`, not in CI).
  - Recovery mechanism coverage: verify stagnation detection triggers, periodic reset works.
- **Long-form audio benchmarks.** Current test data maxes at 119s. Need varied speakers, background noise, topic changes.

## Code Cleanup

- **Separate modules.** `asr.py` is ~1100 lines. AudioEncoder, ASR model, StreamingSession, server handler, and CLI could be split.

## Features

- **Server-side noise reduction.** `noisereduce` (spectral gating) or simple high-pass filter (~200-300Hz) to remove fan/ambient noise before ASR. Browser noiseSuppression helps but server-side would catch upload-file path too.
