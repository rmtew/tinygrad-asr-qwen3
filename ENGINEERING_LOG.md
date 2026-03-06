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

**Problem:** `transcribe()` and `transcribe_stream()` both call the decoder's JIT'd prefill. Initially each method created its own `UOp.variable(...)` instances with the same name. This caused separate JIT compilations per method — the JIT saw different variable objects despite identical names.

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
