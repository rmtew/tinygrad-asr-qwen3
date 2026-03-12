"""Test Transformer.from_gguf with QuantizedLinear — quantized weights stay as raw blocks on GPU."""
import os, time, pathlib, gc
os.environ.setdefault("DEBUG", "1")

from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters
from tinygrad.apps.llm import Transformer, SimpleTokenizer

MODEL = r"C:\Users\richa\AppData\Local\local-models\llm\Qwen3.5-0.8B-Q4_K_M.gguf"
N_TOKENS = 20
WARMUP = 3

print("Loading model...")
raw = Tensor(pathlib.Path(MODEL))
model, kv = Transformer.from_gguf(raw, max_context=4096)
tok = SimpleTokenizer.from_gguf_kv(kv)
del raw; gc.collect()

# generate
toks = tok.encode("Hello")
tok_out = model(Tensor([toks]), 0).realize()
start_pos = len(toks)
ids = [tok_out.item()]
for i in range(WARMUP):
  tok_out = model(Tensor([[ids[-1]]]), start_pos + i).realize()
  ids.append(tok_out.item())
start_pos += WARMUP

GlobalCounters.reset()
t0 = time.perf_counter()
for i in range(N_TOKENS):
  tok_out = model(Tensor([[ids[-1]]]), start_pos + i).realize()
  ids.append(tok_out.item())
dt = time.perf_counter() - t0
tps = N_TOKENS / dt

all_toks = toks + ids
try: text = tok.decode(all_toks)
except Exception: text = f"[{len(all_toks)} tokens]"

print(f"RESULT: {tps:.1f} tok/s | {GlobalCounters.kernel_count} kernels | {dt:.2f}s")
print(f"Output: {text[:100]}")

# verify tokens match expected baseline sequence
expected = [11, 353, 1044, 264]  # first 4 tokens after "Hello" from baseline
match = ids[:4] == expected
print(f"Token match vs baseline: {'PASS' if match else 'FAIL'} (got {ids[:4]})")
