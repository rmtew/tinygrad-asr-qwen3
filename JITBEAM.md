# JITBEAM Debugging Guide

Practical reference for diagnosing and fixing tinygrad BEAM search problems.

---

## Environment Variables

### BEAM Control

| Variable | Default | Description |
|----------|---------|-------------|
| `BEAM=N` | 0 | Beam search width for kernel optimization |
| `JITBEAM=N` | 0 | BEAM width applied during JIT capture |
| `IGNORE_BEAM_CACHE=1` | 0 | Force re-search, ignore cached results |
| `NOOPT=1` | 0 | Disable all kernel optimizations |

### BEAM Search Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `BEAM_DEBUG=N` | 0 | 1=print AST+result, 2=print every candidate (uop count, compile time, run time) |
| `BEAM_TIMEOUT_SEC=N` | 10 | Compile timeout per candidate (seconds) |
| `BEAM_UOPS_MAX=N` | 3000 | Skip candidates exceeding N uops |
| `BEAM_UPCAST_MAX=N` | 256 | Max upcast factor before skipping |
| `BEAM_LOCAL_MAX=N` | 1024 | Max local memory (elements) before skipping |
| `BEAM_MIN_PROGRESS=N` | 0.01 | Min microsecond improvement to continue search |
| `BEAM_LOG_SURPASS_MAX=1` | 0 | Print when candidates are skipped for exceeding limits |
| `BEAM_PADTO=1` | 0 | Enable PADTO optimization action |
| `BEAM_STRICT_MODE=1` | 0 | Raise exceptions instead of silently skipping |
| `PARALLEL=N` | cpu_count | Number of parallel compile workers |

### Debug Output Levels

| Level | Output |
|-------|--------|
| `DEBUG=1` | JIT capture info ("captured N kernels"), device info |
| `DEBUG=2` | Per-kernel execution: timing, FLOPS, memory bandwidth |
| `DEBUG=3` | Buffer shapes/dtypes, applied optimizations per kernel |
| `DEBUG=4` | Generated kernel source code |
| `DEBUG=6` | Full UOp list dump |
| `DEBUG=7` | GPU disassembly |

### Other Useful Variables

| Variable | Description |
|----------|-------------|
| `VIZ=1` | Interactive visualization server for kernel graphs |
| `PROFILE=1` | GPU kernel profiling (HCQ backends) |
| `SPLIT_REDUCEOP=0` | Disable reduce op splitting (default on) |
| `MAX_KERNEL_BUFFERS=N` | Force kernel split if >N buffers used |

---

## Diagnostic Workflow

### Step 1: Count kernels

```bash
DEBUG=1 python script.py
```

Look for "JIT captured N kernels". Rules of thumb:
- < 500 kernels: BEAM=2 finishes in seconds/minutes
- 500-1500 kernels: BEAM=2 may take minutes
- > 1500 kernels: BEAM=2 may hang — graph is too large

### Step 2: Find problem kernels

```bash
BEAM_DEBUG=2 BEAM_LOG_SURPASS_MAX=1 BEAM_TIMEOUT_SEC=5 JITBEAM=2 python script.py
```

Shows every BEAM candidate. Look for:
- Kernels with > 3000 uops — scheduler fused too aggressively
- Candidates taking > 5s to compile — search space too large
- Kernels being skipped — hitting uops/upcast/local limits

### Step 3: Measure dispatch overhead

```bash
DEBUG=2 python script.py
```

Per-kernel timing shows actual compute time. If you have 5000 kernels each taking 0.02ms, that's 100ms of dispatch overhead before any compute. Compare total dispatch time vs total compute time.

### Step 4: Compare optimized vs unoptimized

```bash
# Baseline: no optimization
NOOPT=1 python script.py

# With BEAM
JITBEAM=2 python script.py
```

If BEAM doesn't reduce kernel count or improve speed meaningfully, the problem is graph structure, not kernel optimization.

### Step 5: Visualize

```bash
VIZ=1 python script.py
```

Interactive server showing how operations are grouped into kernels. Reveals where the scheduler made bad fusion decisions.

---

## Problem Signals & Fixes

### BEAM hangs (stuck on one kernel)

**Signal:** BEAM cache stops growing. `BEAM_DEBUG=2` shows one kernel being searched for minutes.

**Causes:**
- Kernel has too many uops (scheduler fused too many operations)
- Search tree is too deep (many possible axis splits)

**Fixes:**
- Add `.contiguous()` barriers to break the graph into smaller kernels
- Use separate JITs for independent computation stages
- Set `BEAM_TIMEOUT_SEC=5 BEAM_UOPS_MAX=2000` to skip problem kernels

### Too many kernels (> 1500 per JIT)

**Signal:** `DEBUG=1` shows "captured 5000 kernels". BEAM takes forever.

**Causes:**
- Multiple transformer models in one JIT (e.g., 5-layer CP + 28-layer talker)
- Unrolled loops generating separate kernels per iteration
- Scheduler not finding fusion opportunities

**Fixes:**
- Split into separate JITs (each with tractable kernel count)
- The CPU round-trip between JITs (~1ms) is cheaper than BEAM failing

### Dispatch overhead dominates compute

**Signal:** `DEBUG=2` shows thousands of tiny kernels (< 0.01ms each). Total time is kernel_count * ~0.02ms dispatch overhead.

**Cause:** Graph decomposed into too many small kernels.

**Fix:** Restructure operations to enable fusion. Add intermediate `.contiguous()` calls strategically — they can both break and enable fusions depending on placement.

---

## Cache Behavior

### What gets cached

| Table | Contents | Key |
|-------|----------|-----|
| `compile_cuda_sm_XX_22` | Compiled GPU binaries (cubin) | Source code hash |
| `beam_search_22` | Optimal schedules per kernel | AST + BEAM width + device |

- Compile cache: content-addressed, grows monotonically, includes BEAM losers
- BEAM cache: keyed by (AST, BEAM width, device) — BEAM=2 and BEAM=3 don't share entries
- BEAM cache does NOT include `NOLOCALS`, `USE_TC`, or other env vars in key — use `IGNORE_BEAM_CACHE=1` after changing these

### Cache is device-specific

- Compiled binaries are architecture-specific (sm_86 won't run on sm_75)
- BEAM schedules are GPU-specific (optimal schedule depends on hardware)
- Cannot distribute pre-built caches across different GPU families

### Cache persistence

- Entries never expire, no LRU, no hit tracking
- Clearing: delete `~/.cache/tinygrad/cache.db` or drop specific tables
- BEAM progress survives interruption — completed kernels stay cached, search resumes from where it left off

---

## Our Project: Lessons Learned

### Separate JITs vs combined JIT

| Approach | Kernels | BEAM=2 | Per-step |
|----------|---------|--------|----------|
| Separate (talker + CP) | ~716 + ~400 | Finishes | ~135ms (2 dispatches) |
| Combined (talker + CP) | ~5000 | Hangs | ~143ms (1 dispatch) |

The combined JIT eliminates one CPU round-trip but produces too many kernels for BEAM to handle. The separate approach gives BEAM tractable search spaces.

### Buffer aliasing bug

TinyJit reuses the same output buffer on every replay. If you store JIT output tensors across calls (e.g., accumulating in a list), all entries point to the same buffer. Snapshot with `.numpy()` immediately before the next call overwrites it.

### PTX vs cubin

NVRTC version mismatches can produce PTX the driver can't load. Force cubin output (`ptx=bool(MOCKGPU)` in cstyle.py) to avoid driver JIT compilation entirely. No performance cost.
