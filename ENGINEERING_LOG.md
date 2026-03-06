# Engineering Log

Decisions, failed approaches, and solutions encountered during development. Intended as context for future sessions — both human and AI.

---

## GGUF format: Q8_0 vs F16

**Problem:** The most common Qwen3-ASR GGUF (Alkd) uses Q8_0 quantization. Tinygrad's built-in `gguf_load` doesn't handle Q8_0 dequantization. The Q8_0 GGUF also has 3 weights tagged as TQ1_0, but these are junk data — not a real issue once known.

**Solution:** Use the FlippyDora F16 GGUF instead. Tinygrad's `gguf_load` handles F16/F32 natively — zero custom loading code needed. 1.88 GB, no quality loss.

---

## Decoder architecture: custom vs reuse

**Problem:** The ASR decoder is a standard Qwen3 transformer. Could implement it from scratch or reuse tinygrad's existing `Transformer` from `llm.py`.

**Tried:** Initially explored custom implementation.

**Solution:** Direct reuse of `tinygrad.apps.llm.Transformer`. The ASR decoder's architecture (GQA, RoPE, SwiGLU) is identical to Qwen3's text LM. Load GGUF weights, remap key names, call `load_state_dict`. Zero duplicated transformer code.

---

## Prefill: chunked loop vs single-call JIT

**Problem:** Initial prefill used a 32-token chunked loop (1418ms for 158 tokens). Each chunk was a separate JIT call.

**Tried:** Chunked prefill with fixed chunk size. Functional but slow — many small JIT replays with Python loop overhead.

**Solution:** Single-call JIT prefill using `UOp.variable('asr_nt', 1, max_context)` for symbolic sequence length. One JIT compilation handles any prompt length. With JITBEAM=2: 1407ms → 213ms (6.6× speedup).

---

## BEAM=2 vs JITBEAM=2

**Problem:** Needed to optimize kernel quality for better GPU utilization. Default heuristic kernels achieved <0.3% of RTX 3070's theoretical FP16 throughput.

**Tried:** `BEAM=2` — optimizes ALL kernels at first compilation, including the encoder's 18-layer transformer. Took 15 minutes on startup. The encoder kernels showed zero improvement (already optimal for their small sizes). All the gain came from decoder kernels.

**Solution:** `JITBEAM=2` — only optimizes kernels captured inside TinyJit functions (decoder prefill + decode). 43s warmup instead of 15min. Same speedup for the decoder paths that matter. The encoder wasn't JIT'd at that point so JITBEAM skipped it entirely.

**Later update:** After JIT'ing the encoder (see below), JITBEAM=2 warmup increased to ~280s (optimizes encoder kernels too), but encoder went 725ms → 109ms.

---

## Symbolic variable sharing across methods

**Problem:** `transcribe()` and the streaming path both call the decoder's JIT'd prefill. Initially each method created its own `UOp.variable(...)` instances with the same name. This caused separate JIT compilations per method — the JIT saw different variable objects despite identical names.

**Solution:** Create shared symbolic variables once in `__init__`: `self.v_sp`, `self.v_nt`, `self.v_dec_pos`. Both methods use the same variable objects. Single JIT compilation serves all callers.

---

## Encoder JIT: Python loops prevent graph capture

**Problem:** The encoder had two Python loops that prevented TinyJit from capturing the full computation graph:
1. Conv stem: looped over 8 mel chunks, processing each through Conv2d separately
2. Windowed attention: looped over attention windows in `_windowed_sdpa`

**Solution:** Restructured both as batch-dimension operations:
- Conv stem: reshape `[128, 800]` → `[8, 1, 128, 100]`, process all chunks as a batch
- Windowed attention: all windows are the same size (104 tokens), so reshape `[N*104, d_model]` → `[N, 104, d_model]` and use the batch dimension for parallel attention

Numerically identical output (verified: max diff = 0.000000). One TinyJit per bucket size (800, 1600, ...), each compiled once.

**Result:** Encoder 725ms → 109ms (6.7×). Now 2.1× faster than the C implementation's 231ms.

---

## JIT buffer aliasing: `.contiguous()` doesn't copy

**Problem:** After JIT'ing the encoder, streaming mode produced garbage transcriptions (WER 46.6%). Short files (<8s) worked; longer files only transcribed the tail end.

**Root cause:** TinyJit reuses output buffers across calls. Streaming caches encoder outputs from completed 8s windows. When the encoder JIT ran again (for the partial tail), it overwrote the buffer that the cached windows pointed to. All cached entries became views of the latest output.

**Tried:** `.contiguous()` on the JIT output — doesn't work. In tinygrad, `.contiguous()` on a JIT output buffer is a no-op (already contiguous, same buffer). `.realize()` also doesn't copy — just materializes the existing buffer.

**Solution:** `(output + 0).realize()` — the `+ 0` creates a new computation node that forces allocation of a fresh buffer. The `.realize()` ensures the copy executes before the JIT buffer can be overwritten.

**Verified:** Isolated test confirmed `.contiguous()` fails, `(+0).realize()` succeeds:
```python
out1 = jit_fn(x1)
saved = out1[:,:2].contiguous().realize()  # ❌ stale after next call
saved = (out1[:,:2] + 0).realize()          # ✅ independent copy
```

---

## Streaming: delta vs full transcription per chunk

**Problem:** Each streaming chunk could return either a delta (new words only) or a complete transcription of all audio heard so far. The C implementation uses rollback for incremental display optimization.

**Tried:** Both approaches.

**Solution:** Full transcription per chunk. Each chunk's decoder starts from `start_pos=0` and generates the complete text for all accumulated audio. Simpler than rollback/dedup, and the final chunk's output is the correct complete result. Higher compute cost (redundant decode), but correct by construction. The C implementation's rollback is an optimization for incremental *display*, not correctness.

---

## realize() is async: timing gotchas

**Problem:** Encoder showed "10ms" in timing logs but total pipeline time didn't decrease proportionally. Prefill showed "1407ms" but `realize()` returned in 12ms.

**Root cause:** `realize()` submits GPU work asynchronously and returns immediately. `item()` or `numpy()` forces synchronization, revealing the true GPU compute time.

**Solution:** For accurate per-stage timing, call `.numpy()` or `.item()` to force sync. For end-to-end timing, just measure wall clock around the full pipeline (async submission + final sync). The total is accurate; per-stage splits are only accurate with forced sync points.

---

## Decoder KV cache reuse across streaming chunks

**Problem:** Streaming mode re-prefilled the entire prompt from `start_pos=0` every chunk. For a 16s file with 2 cached encoder windows, each chunk prefills ~200+ tokens even though only ~30 change (the partial tail + suffix).

**Solution:** After building each chunk's prompt embeddings, compare row-by-row with the previous chunk's embeddings (saved as numpy). Find the longest matching prefix (reuse point). Only prefill the delta tokens from `start_pos=reuse_point`. The decoder's KV cache already retains entries from the previous chunk — positions 0..reuse_point have valid cached values.

**Why embedding comparison instead of analytical reuse:** The reuse point could be computed analytically (prefix + cached windows are always identical). But embedding comparison is simpler, handles all edge cases (window eviction, bucket changes), and matches the C implementation's approach. The CPU readback cost (~1ms) is negligible vs prefill savings (~150ms).

**Result:** Streaming RTF 0.22 → 0.16 (JFK), 0.232 → 0.194 (30-utt). 245 tokens reused across 6 chunks for JFK 11s. WER unchanged.

---

## Two-path encoder: batched JIT + sequential fallback

**Problem:** The encoder used one-JIT-per-bucket-size. Unseen bucket sizes (e.g., 180,000 frames for 30min audio) triggered JITBEAM compilation that could take hours. Even within LibriSpeech, a 16.8s file hitting the 2400-frame bucket for the first time added a 4.5s outlier.

**Solution:** Two JIT paths:
- **Batched JIT** for pre-warmed bucket sizes (800/1600/2400/3200, covering ≤32s): processes all windows in one JIT call with batched attention. Fast, pre-compiled.
- **Sequential single-window JIT** for any other size: splits into 800-frame windows, encodes each through the pre-warmed single-window JIT. No compilation surprises. Scales to any length.

Both produce identical output because attention is windowed — no cross-window dependencies. Verified numerically (max diff = 0.000000).

**JITBEAM disk caching:** Beam search results are cached in `~/.cache/tinygrad/cache.db` (sqlite, keyed by kernel AST hash + device). First-ever run pays full beam search cost (~11min for all buckets). Subsequent runs hit cache (~65s warmup — just JIT graph capture). Cache is versioned by tinygrad version; no automatic eviction within a version.

**Result:** Per-file RTF improved 0.089 → 0.070 (no compilation outliers). 119s audio file: encoder 98ms via 19 sequential single-window calls, RTF=0.16.

---

## Performance summary (RTX 3070 Laptop, 0.6B model, JFK 11s)

| Stage | No JIT | + Decoder JIT | + JITBEAM=2 | + Encoder JIT | C impl |
|-------|--------|---------------|-------------|---------------|--------|
| Encoder | 725ms | 725ms | 725ms | **109ms** | 231ms |
| Prefill | 1407ms | 1407ms | **213ms** | 217ms | 275ms |
| Decode | 689ms | 689ms | **229ms** | 264ms | 552ms |
| **Total** | 2846ms | 2846ms | 1178ms | **591ms** | 783ms |
| RTF | 0.26 | 0.26 | 0.107 | **0.054** | 0.07 |

**Streaming (JFK 11s, 6 chunks):**

| Metric | Before KV reuse | After KV reuse |
|--------|----------------|---------------|
| RTF | 0.22 | **0.16** |
| Prefill total | ~1200ms | ~800ms |
| KV tokens reused | 0 | 245 |

---

## C-style streaming: text prefix feedback + monotonic commit

**Problem:** The original streaming approach re-transcribed the entire audio window each chunk. When encoder windows were evicted (>32s), old text disappeared. The transcript flickered — each chunk replaced the full display. For long audio (>60s), the output was essentially useless.

**Reference:** The C implementation (`qwen_asr.c stream_impl`) solves this with three mechanisms: text prefix feedback, rollback-based commit, and recovery resets.

**Solution:** Rewrote `StreamingSession` to match the C architecture:

1. **Text prefix feedback.** After initial unfixed chunks, embed previous decoded tokens (minus rollback=5) and append to the prompt after audio+suffix. The decoder sees what was said before, even after the audio is gone. `raw_tokens` tracks the full decoded history (prefix + continuation); only `max_prefix_tokens=150` are embedded in the prompt.

2. **Monotonic commit.** Extract text tokens from `raw_tokens`, compute candidate (minus rollback tail). LCP against previous `stable_text_tokens` finds the stable frontier. Only emit delta tokens beyond the LCP. `emitted_text` grows monotonically — old text never removed.

3. **Overlap dedup.** Before emitting from the LCP point, check if new tokens overlap with the tail of `emitted_text_tokens` (window: min=4, max=48 tokens). Prevents duplicate text when the model re-generates tokens already committed.

4. **Repeat suppression.** Filter `chunk_tokens` runs exceeding `MAX_REPEAT_TOKEN_RUN=12`, continuing the run count from the end of the prefix. Prevents degenerate repetition loops.

5. **Stagnation detection + recovery reset.** Three triggers:
   - Stagnant: `candidate_advance ≤ 1` AND hit `max_new_tokens` for 4+ consecutive chunks
   - Degenerate: tail repeat blocks (period ≤ 6, repeats ≥ 4)
   - Dropped repeats ≥ 8
   
   Recovery (`_reanchor`): rebuild `raw_tokens` from last 24 `emitted_text_tokens`, reset KV cache. **Critical difference from C:** preserve encoder cache and `_window_buf`. The C code clears the encoder cache because it retains the full audio buffer and re-encodes. Our session receives audio incrementally via `feed()` — clearing enc_cache loses all audio context.

6. **Periodic reset.** Every 45 chunks (~90s), force `_reanchor` to prevent slow drift accumulation.

**Bug: destructive reanchor.** Initial implementation cleared `enc_cache` and `_window_buf` on reset, matching the C code literally. This was catastrophic — after the periodic reset at chunk 45, the model had zero encoder windows and could barely produce text for the remaining 30 seconds. NOTLD 119s WER: 8.4% → 21.9%. Fix: preserve encoder cache in `_reanchor`. WER: 21.9% → 9.3%.

---

## Streaming quality baselines

**Test infrastructure:** `tests/test_stream_quality.py` compares streaming vs per-file WER. `tests/test_session.py` runs `StreamingSession.feed()` with per-chunk diagnostic output.

**Results (0.6B model, RTX 3070 Laptop):**

| Test | Stream WER | Per-file WER | Gap |
|------|-----------|-------------|-----|
| JFK 11s clean | 0.0% | 0.0% | 0% |
| NOTLD 119s clean | 9.3% | 0.0% | 9.3% |
| Real mic 47s | 23.3% | 0.0% | 23.3% |

**Root cause of baseline gap (~9% on clean audio):** Each 2s chunk generates ~5-13 continuation tokens. With rollback=5, only ~0-8 net new tokens are committed per chunk. Normal speech at 150 WPM produces ~7 tokens per 2s. The model's continuation sometimes under-generates, losing words. This compounds over 60 chunks.

**Noise amplifies the gap.** Noisy audio (phone speaker → mic, fan noise) causes the model to generate fewer/worse continuation tokens, widening the gap to ~23%+.

**Mitigations applied:** Browser-side `getUserMedia` constraints (`noiseSuppression`, `echoCancellation`, `autoGainControl`) for free noise reduction via Chrome's WebRTC pipeline.

---

## Parameter sweep: chunk_sec × rollback

**Problem:** Streaming WER (3.8% on real mic, 9.3% on 119s clean) has a gap vs per-file (0%). The two main knobs are `chunk_sec` (how much audio per decode) and `rollback` (how many tail tokens to withhold from commit). Need data to pick defaults.

**Tool:** `sweep_params.py` — grid-searches `(chunk_sec, rollback)` on captured WAV files, using per-file transcription as reference. Reports WER table per file + ranked summary. Uses `StreamingSession.verbose = False` to suppress per-chunk logging.

**Results (real mic 51.7s, 52 ref words):**

| | rb=3 | rb=5 | rb=7 |
|---|---|---|---|
| **2s** | 5.8% | 3.8% | 3.8% |
| **4s** | 7.7% | 3.8% | 3.8% |
| **6s** | 3.8% | 3.8% | 3.8% |
| **8s** | 7.7% | 3.8% | 3.8% |

JFK (11s, clean): 0% WER across all 12 combos.

**Key findings:**

1. **Rollback matters more than chunk size.** rb≥5 achieves 3.8% regardless of chunk size. rb=3 is worse at every chunk size — commits too aggressively.

2. **Chunk size barely affects WER** once rollback is sufficient. Even 8s chunks (near per-file context) produce the same 3.8% error.

3. **The residual error is prefix feedback reinforcement, not chunk boundary.** The "Thim" misdecoding at chunk 13 persists at all chunk sizes because once decoded, it's fed back as prefix context in every subsequent chunk. The model sees "Thim" in its input and keeps reproducing it. Rollback only delays commitment — it can't correct errors already in the prefix. Per-file mode avoids this entirely (no prefix feedback, full audio context).

4. **RTF sweet spot: 4-6s.** 2s chunks have highest per-chunk overhead (RTF 0.65). 6s is most efficient (RTF 0.33). But for live mic UX, 2s gives more responsive text updates.

**Decision:** Keep defaults at **2s chunks, rollback=5**. The sweep confirmed these are near-optimal for WER, and the responsiveness advantage of 2s updates outweighs the RTF savings of larger chunks. UI presets are not needed — there's no meaningful quality tradeoff to expose to users.

---

## WebSocket: BaseHTTPRequestHandler incompatibility

**Problem:** Hand-rolled WebSocket (RFC 6455) over Python's `BaseHTTPRequestHandler` worked with Python test clients but failed with browsers. Browser connection closed immediately with code 1006 (abnormal closure). Server's `rfile.read(2)` returned `b''` right after sending the 101 response.

**Root cause:** Two issues discovered:
1. `BaseHTTPRequestHandler.send_response()` uses `self.protocol_version` which defaults to `HTTP/1.0`. Browsers reject `HTTP/1.0 101 Switching Protocols` — the WebSocket spec requires HTTP/1.1.
2. `BufferedReader` (rfile) returns empty bytes after HTTP header parsing on Windows, even though the socket is still open. This occurs with both `self.request.sendall()` and `self.wfile.write()` approaches.

**Tried:** Setting `self.protocol_version = "HTTP/1.1"`, using `self.wfile` instead of raw socket, creating fresh `sock.makefile("rb")` wrappers. Python test clients passed with all approaches; browsers failed with all.

**Solution:** Replaced hand-rolled WebSocket with the `websockets` library (`pip install websockets`). Runs on a separate port (HTTP port + 1) in its own asyncio event loop. ~80 lines of RFC 6455 code removed. Browser connections work immediately.

**Threading constraint:** The websockets server must run in the main thread because tinygrad's SQLite disk cache is thread-local (created during `model.warmup()` in the main thread). HTTP server runs in a daemon thread instead.

**Ctrl+C on Windows:** `asyncio.run(serve_forever())` ignores `KeyboardInterrupt` on Windows. Fixed with `signal.SIGINT` handler that sets an `asyncio.Event`, cleanly stopping the server.

---

## Silence auto-commit: pipeline-level, not display-level

**Problem:** When the user stops speaking, the last few words stay as pending (dim italic) indefinitely. The rollback tail is never committed because no new text arrives to push it past the LCP.

**First attempt (broken):** In `feed()`'s display section, track how many chunks `committed` text is unchanged while `pending` exists. After 3 chunks, directly append `pending_text` to `emitted_text` and update `stable_text_tokens`. This caused **double-emission**: `_process_chunk`'s commit pipeline didn't know about the direct modification, so it later re-committed the same tokens through the normal LCP/emit path.

**Solution:** Move silence detection into `_process_chunk`'s commit logic. Track `prev_pending_tokens` and `pending_stable_chunks`. If the rollback tail tokens are identical for 3 consecutive chunks, set `candidate_len = n_text` (commit everything, including the tail). This works within the existing LCP/emit pipeline — tokens can only be emitted once.

**Also fixed:** `feed([], is_final=True)` with no remaining audio (stop button) now commits all pending tokens immediately, regardless of the silence counter.

---

## Diagnostic logging for streaming sessions

**Problem:** Browser streaming sessions showed quality issues (dropped words, garbled text) but the server log only showed timing stats — no visibility into what the model decoded or what was committed.

**Solution:** Rich per-chunk diagnostic log to stderr:

```
chunk 15: 30.0s  enc=14ms prefill=731ms(309kv) decode=125ms(12tok)  win=3/4 prefix=58 raw=70
  decoded : ' all? I mean, is Willer the nearest town?'
  commit  : lcp=55/62 emit=7tok emitted_total=62
  +emit   : ' all? I mean, is Will'
  pending : 'er the nearest town?'
```

Each line shows: what the model generated (`decoded`), how it compared to previous stable (`lcp`), what new text was committed (`+emit`), and the unfixed rollback tail (`pending`). Recovery events are flagged with `!RECOVERY` or `!PERIODIC_RESET`.

Stagnation is immediately visible as consecutive `emit=0tok` chunks. LCP breakage (model contradicting prefix) shows as `lcp < stable`. Enables direct correlation between browser user experience and server-side model behavior.
