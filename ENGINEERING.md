# Engineering Decisions

Key decisions, failed approaches, and solutions. Concise reference for future work.

---

## ASR

### GGUF format: Q8_0 vs F16

**Problem:** Common Qwen3-ASR GGUF uses Q8_0. Tinygrad's `gguf_load` doesn't handle Q8_0 dequantization.

**Solution:** Use FlippyDora F16 GGUF. Tinygrad handles F16/F32 natively — zero custom loading code. 1.88 GB, no quality loss.

---

### Decoder: reuse tinygrad's Transformer

**Problem:** ASR decoder is a standard Qwen3 transformer — custom impl vs reuse.

**Solution:** Direct reuse of `tinygrad.apps.llm.Transformer`. Same architecture (GQA, RoPE, SwiGLU). Load GGUF, remap keys, `load_state_dict`. Zero duplicated transformer code.

---

### JIT prefill: symbolic sequence length

**Problem:** Chunked prefill loop (1418ms for 158 tokens). Each chunk was a separate JIT call.

**Solution:** Single-call JIT using `UOp.variable('asr_nt', 1, max_context)` for symbolic length. One JIT compilation handles any prompt length. With JITBEAM=2: 1407ms → 213ms (6.6x).

---

### BEAM=2 vs JITBEAM=2

`BEAM=2` optimizes ALL kernels at first compilation — 15 min startup. Encoder kernels showed zero improvement. `JITBEAM=2` only optimizes JIT-captured kernels — 43s warmup, same decoder speedup. After JIT'ing the encoder, JITBEAM warmup ~280s but encoder went 725ms → 109ms.

---

### Encoder JIT: batching eliminates Python loops

**Problem:** Two Python loops (conv stem, windowed attention) prevented TinyJit capture.

**Solution:** Reshape to use batch dimension: mel chunks as batch for conv stem, attention windows as batch for SDPA. Numerically identical output. Encoder 725ms → 109ms (6.7x), 2.1x faster than C impl.

---

### JIT buffer aliasing: `(+0).realize()` for safe copies

TinyJit reuses output buffers across calls. `.contiguous()` on a JIT output is a no-op. `.realize()` doesn't copy.

**Solution:** `(output + 0).realize()` forces a new buffer allocation. Required when caching JIT outputs across calls (e.g., encoder windows in streaming).

---

### Two-path encoder: batched JIT + sequential fallback

Pre-warmed bucket sizes (800/1600/2400/3200, ≤32s) use batched JIT. Unknown sizes split into 800-frame windows, each through the single-window JIT. No compilation surprises, scales to any length. JITBEAM results cached in `~/.cache/tinygrad/cache.db`.

---

### Decoder KV cache reuse across streaming chunks

Compare embeddings row-by-row with previous chunk, find longest matching prefix. Only prefill the delta from `start_pos=reuse_point`. Streaming RTF 0.22 → 0.16 (JFK), 245 tokens reused across 6 chunks.

---

### C-style streaming architecture

Rewrote `StreamingSession` to match C implementation:

1. **Text prefix feedback** — embed previous decoded tokens (minus rollback) after audio+suffix
2. **Monotonic commit** — LCP against previous stable tokens, emit only delta
3. **Overlap dedup** — check new tokens against emitted tail (window 4-48 tokens)
4. **Repeat suppression** — filter runs exceeding MAX_REPEAT_TOKEN_RUN=12
5. **Stagnation recovery** — detect stagnant/degenerate patterns, `_reanchor` with last 24 emitted tokens
6. **Periodic reset** — every 45 chunks (~90s), force reanchor

**Critical bug:** Initial reanchor cleared encoder cache (matching C literally). Catastrophic — after reset, model had zero audio context. Fix: preserve encoder cache in `_reanchor`. WER: 21.9% → 9.3%.

---

### Parameter sweep: chunk_sec x rollback

Sweep tool: `benchmarks/sweep_params.py`. Tested chunk_sec={2,4,6,8} × rollback={3,5,7}.

**Findings:** rollback≥5 achieves 3.8% WER regardless of chunk size. Residual error is prefix feedback reinforcement, not chunk boundaries. Defaults: **2s chunks, rollback=5** (responsiveness wins over RTF savings).

---

### ASR performance summary (RTX 3070 Laptop, 0.6B, JFK 11s)

| Stage | No JIT | + JITBEAM=2 | + Encoder JIT | C impl |
|-------|--------|-------------|---------------|--------|
| Encoder | 725ms | 725ms | **109ms** | 231ms |
| Prefill | 1407ms | **213ms** | 217ms | 275ms |
| Decode | 689ms | **229ms** | 264ms | 552ms |
| **Total** | 2846ms | 1178ms | **591ms** | 783ms |
| RTF | 0.26 | 0.107 | **0.054** | 0.07 |

---

## TTS

### Safetensors BF16 on CUDA with PTX renderer

**Problem:** Qwen3-TTS weights are BF16 safetensors. PTX renderer has no `bf16` register type.

**Solution:** Convert BF16 safetensors → F16 GGUF via `tools/convert_tts_gguf.py`. Load with `gguf_load` + `.cast(float32)`. Vocoder safetensors is F32, loads directly via `safe_load`.

---

### Base vs CustomVoice model

CustomVoice (0.6B): 402 tensors, 9 built-in voices via `spk_id` codec embedding indices. No speaker encoder needed. Voice selection: look up codec embedding row, inject as extra token in prefill.

---

### EOS suppression bug

`CODEC_EOS = 2150` fell inside the suppression range `[2048, 3072)`. Fix: save EOS logit before blanket suppression, restore after.

---

### Unrolled CP: 16 → 2 syncs per step

Rewrote code predictor as single unrolled JIT: prefill(T=2) + 14x decode(T=1) in one `@TinyJit`. All argmax on GPU, single `.numpy()` sync. CP: 1550ms → 221ms/step (7x). Warm RTF: 28.5 → 8.7.

---

### GPU sampling merged into talker JIT

**Key insight:** GPU sampling as a *separate* JIT added 130-200ms inter-JIT overhead. Merged *into* the talker's existing JIT — same CUDA graph, just more kernels. Sort-based top-k (TOP_K=50), deterministic GPU RNG via captured counter buffer.

**Result:** RTF 5.45 → 4.61 (JITBEAM=2), 4.40 (no JITBEAM with warmup fix), **4.24** (GPU top-k).

---

### Vocoder optimization: JITBEAM + F16

JITBEAM=2 on vocoder (57 unique kernels, 33 exceed BEAM_UOPS_MAX): 5.0s → 1.4s (3.6x). F16 compute (GGUF F16 weights): 1.4s → 0.9s. **Total 6.3x vocoder speedup.**

---

### Sync reduction progression

| State | Syncs/step | Per-step ms | Long RTF |
|-------|-----------|-------------|----------|
| Split CP baseline | 16+ | ~1800 | 28.5 |
| Unrolled CP | 2 | ~430 | 8.7 |
| + GPU CP, remove cache reset | 1 | ~317 | 5.45 |
| + GPU sampling in JIT | 1 (.item()) | ~268 | 4.61 |
| + Warmup fix + GPU top-k | 1 (.item()) | ~140 | **4.24** |
| **All JITBEAM=2 + F16 vocoder** | 1 (.item()) | ~140 | **3.00** |
| C reference | 0 | ~5 | 1.7 |

**Gap analysis:** GPU computes each step in ~7ms. Wall clock ~87-97ms/step. Remaining gap is Python dispatch — JIT replay, CUDA graph submission, `.item()` sync.

---

### TTS optimization log

| Change | Talker ms/step | CP ms/step | RTF (long) |
|--------|---------------|-----------|------------|
| Split CP baseline | 212 | 1550 | 28.5 |
| Unrolled CP | 133 | 221 | 8.7 |
| CP on GPU (no `.numpy()`) | ~370 | **127** | 8.1 |
| + Remove CP cache reset | ~370 | **27** | 6.8 |
| + codec_sum in CP JIT | **289** | **27** | 5.45 |
| + GPU sampling in JIT | ~247 | ~20 | 4.61 |
| + warmup fix (no JITBEAM) | ~89 | ~28 | 4.40 |
| + GPU top-k (no JITBEAM) | ~92 | ~48 | **4.24** |
| + Vocoder JITBEAM=2 + F16 | ~92 | ~48 | **3.00** |
| C reference | ~4 | ~1 | 1.7 |

---

## Infrastructure

### WebSocket: use `websockets` library

Hand-rolled RFC 6455 over `BaseHTTPRequestHandler` failed with browsers (HTTP/1.0 rejection, BufferedReader issues on Windows). Replaced with `websockets` library on separate port. Threading constraint: WS server must run in main thread (tinygrad SQLite cache is thread-local).

---

### Dispatch queue: thread-safe inference

HTTP server runs in daemon thread. Inference must run on main thread (CUDA context + SQLite are thread-local). `dispatch(fn)` schedules work on main asyncio loop via `asyncio.run_coroutine_threadsafe()`. `dispatch_generator(fn)` bridges streaming responses via `queue.Queue`.

---

### write_after regression

Qwen3.5 commit introduced `write_after()` helper for KV cache. Semantically identical but changed UOp graph structure → different scheduler output → JITBEAM optimized wrong kernels. Fix: reverted to inline pattern. Lesson: don't refactor working code into helpers when graph structure matters.

---

### FP16 RoPE breaks ASR

`freqs_cis.cast(x.dtype)` caused garbage tokens. FP16×FP16 accumulates precision errors through 24 decoder layers. C impl uses FP16 weight storage + F32 compute. Fix: keep RoPE in F32.

---

### PTX vs cubin: NVRTC version mismatch

NVRTC 13.1 emits PTX 9.1, driver 13.0 only supports PTX 9.0. Fix: force cubin output in CUDA renderer. No functional impact.

---

## Dead Ends

### Fused linear via custom_kernel

Tested replacing `nn.Linear` with fused kernel using `Tensor.custom_kernel()` and UOp DSL. Two variants: naive (LOOP+REDUCE) and tiled (GLOBAL+GROUP_REDUCE).

**Result: dead end.** `custom_kernel` breaks JIT memory optimization — 1000x+ memory explosion (1.26 MB → 1504-8411 MB). Kernel count doubles (+89%) due to forced `.contiguous()` copies. The bottleneck is NOT individual kernel quality — it's that `custom_kernel` creates opaque barriers preventing buffer sharing.

| Model | Baseline JIT mem | Fused JIT mem | Ratio |
|-------|-----------------|---------------|-------|
| 0.8B | 1.26 MB | 1,504 MB | 1,193x |
| 2B | 1.27 MB | 3,764 MB | 2,964x |
| 4B | 2.39 MB | 8,411 MB | 3,519x |

**Lesson:** Tinygrad's JIT buffer sharing is the critical optimization. Any approach that breaks it (custom_kernel, opaque ops) will fail at scale regardless of kernel quality. The correct approach is expressing operations as standard tensor ops that the scheduler can fuse.

---

### GPU sampling as separate JIT

Tested moving top-k sampling to a separate `@TinyJit` function. Added 130-200ms inter-JIT orchestration overhead per step (buffer binding, CUDA graph submission, sync barriers between JITs). Standalone benchmark showed 5ms — the overhead only appears when interacting with talker and CP JITs.

**Resolution:** Merging sampling into the talker's existing JIT eliminated the overhead entirely.

---

### @function on TTS transformer blocks

`@function` creates UOp boundaries that JITBEAM optimizes independently. Works for ASR (30 steps). For TTS (19-92 steps × 28 layers × 2 boundaries = 56 dispatches/step), Python dispatch overhead eats the 21% kernel speed improvement.

---

### JITBEAM on unrolled CP

Compilation >1hr (full model GPU memory pressure slows NVRTC). First run: CUDA error 719. The unrolled CP graph (5140 kernels) is too large for BEAM search. Split CP + JITBEAM: kernels actually *slower* than heuristic (per-call overhead dominates).

---

### JITBEAM non-determinism

Different `cache.db` contents produce different kernel selections with measurably different performance. One run: 126ms/step, fresh cache: 277ms/step. BEAM search uses random restarts — results aren't reproducible.

---

### Quantized compute: keep_quantized

Tested keeping Q4_K weights on GPU and dequantizing at inference time using standard tensor ops. Unlike `custom_kernel`, tensor ops are transparent to the scheduler — zero kernel overhead, zero JIT memory overhead.

**Key result: quantized+JITBEAM=2 gives 3.44x speedup on 0.8B** (10.6 → 36.5 tok/s). On 2B, quantized+JITBEAM trails baseline+JITBEAM slightly (14.2 vs 14.8 tok/s) but uses 3.6x less VRAM and compiles faster.

This approach was upstreamed to tinygrad core as `QuantizedLinear` in `nn/__init__.py` and `keep_quantized` in `state.py`.

| Config (0.8B) | tok/s | Weight VRAM |
|---------------|-------|-------------|
| baseline | 10.6 | 600 MB (FP16) |
| quantized | 12.9 | 169 MB (Q4_K) |
| baseline+JITBEAM=2 | 17.4 | 600 MB |
| **quantized+JITBEAM=2** | **36.5** | **169 MB** |

---

## Key Lessons

1. **GPU sync is the dominant cost, not kernel quality.** GPU computes TTS step in ~7ms. The ~130ms remainder is Python dispatch and sync.

2. **Merging into existing JIT ≠ adding a new JIT.** Separate JITs add 130-200ms orchestration overhead. Same ops merged into one JIT are essentially free.

3. **JIT warmup must match production tensor types exactly.** `Tensor(scalar)` creates a const; `Tensor([scalar]).contiguous().realize()` creates a buffer. Mismatched types cause silent JIT hangs.

4. **TinyJit output buffers are reused across calls.** Store JIT results across iterations → must `.numpy()` or `(+0).realize()` immediately.

5. **`realize()` is async on CUDA.** For accurate per-stage timing, force sync with `.numpy()` or `.item()`.

6. **Standard tensor ops > custom_kernel.** Tensor ops join the lazy graph and get fused by the scheduler. `custom_kernel` creates opaque barriers that break buffer sharing.

7. **Vocoder JITBEAM: "hang" was just slow compilation (~15 min).** BEAM_UOPS_MAX exceeded by 33/57 kernels. Results cache to disk.

8. **`signal.alarm` is POSIX-only.** No compilation timeout on Windows — pathological kernels can hang BEAM search forever.
