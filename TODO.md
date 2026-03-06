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

## Streaming Quality — Parameter Sweep

Current streaming WER gap vs per-file (per-file is 0%):
- Clean 11s: 0% | Clean 119s: 9.3% | Real mic 47s: 23.3%

The two main knobs are `chunk_sec` (latency vs throughput) and `rollback` (safety vs waste). Need data to pick defaults.

- **`--save-audio` flag.** Server saves each session's raw PCM to a timestamped WAV file. Captures real mic audio for offline replay — zero cost when off.
- **`sweep_params.py` tool.** Takes a captured WAV + reference transcript, runs it through `StreamingSession` with a grid of `(chunk_sec, rollback)` values, reports WER and committed word count for each combo. Determines the best operating points for live vs batch use cases.
- **UI presets.** Dropdown or toggle in the web UI: e.g. "Live" (2s/rollback=5), "Balanced" (4s/rollback=4), "Accurate" (8s/rollback=3). Sent in the WebSocket `start` message, server creates session with those params. Defaults chosen from sweep results.
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
