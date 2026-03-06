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

- ~~**`--save-audio` flag.**~~ Server saves each session's raw PCM to timestamped WAV. `--save-audio DIR`.
- ~~**`sweep_params.py` tool.**~~ Grid-search `(chunk_sec, rollback)` on captured WAVs vs per-file reference. Used to validate defaults.
- ~~**Parameter sweep completed.**~~ Tested chunk_sec={2,4,6,8} × rollback={3,5,7} on real mic 51.7s + JFK 11s. Findings: rollback≥5 achieves 3.8% WER regardless of chunk size. Residual error is prefix feedback reinforcement (not chunk boundary). Current defaults (2s/rb=5) confirmed near-optimal. See ENGINEERING_LOG.md for full results.
- ~~**Silence auto-commit.**~~ Pending tokens stable for 3 chunks → auto-committed via pipeline (not display hack).
- ~~**Final commit on stop.**~~ `feed([], is_final=True)` commits all pending tokens immediately.
- **~~UI presets.~~** Not needed — sweep showed no meaningful quality tradeoff between chunk sizes once rollback≥5. Responsiveness (2s) wins over RTF savings (4-6s).
- **Long-form audio testing.** Need 10-30 minute test clips (podcast, lecture) to stress periodic reset and long-term drift. Current max is 119s.

## Performance

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
