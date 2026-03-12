"""Benchmark LLM decode throughput across models and JITBEAM settings.

Uses tinygrad's native QuantizedLinear (keep_quantized) and baseline (FP16) paths.

Usage:
  python benchmarks/bench_llm.py
  python benchmarks/bench_llm.py --models 0.8B
  python benchmarks/bench_llm.py --configs baseline quantized+JITBEAM=2
"""
import os, sys, subprocess, argparse

MODELS = {
  "0.8B": r"C:\Users\richa\AppData\Local\local-models\llm\Qwen3.5-0.8B-Q4_K_M.gguf",
  "2B":   r"C:\Users\richa\AppData\Local\local-models\llm\Qwen3.5-2B-Q4_K_M.gguf",
}

N_TOKENS = 20
WARMUP = 3

BENCH_SCRIPT = r'''
import os, time, pathlib, gc
os.environ["DEBUG"] = "1"
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters
from tinygrad.apps.llm import Transformer, SimpleTokenizer
from tinygrad.nn.state import keep_quantized

MODEL = r"{model}"
N = {n}
WARMUP = {warmup}
USE_QUANTIZED = {use_quantized}

raw = Tensor(pathlib.Path(MODEL))
with keep_quantized():
  model, kv = Transformer.from_gguf(raw, max_context=4096)
if not USE_QUANTIZED:
  model.load_state_dict(dict(model.state_dict()), consume=True)
tok = SimpleTokenizer.from_gguf_kv(kv)
del raw; gc.collect()

# Prefill + warmup
toks = tok.encode("Hello")
tok_out = model(Tensor([toks]), 0).realize()
start_pos = len(toks)
ids = [tok_out.item()]
for i in range(WARMUP):
  tok_out = model(Tensor([[ids[-1]]]), start_pos + i).realize()
  ids.append(tok_out.item())
start_pos += WARMUP

# Timed decode
GlobalCounters.reset()
t0 = time.perf_counter()
for i in range(N):
  tok_out = model(Tensor([[ids[-1]]]), start_pos + i).realize()
  ids.append(tok_out.item())
dt = time.perf_counter() - t0
tps = N / dt
kernels = GlobalCounters.kernel_count
print(f"RESULT: {{tps:.1f}} tok/s | {{kernels}} kernels | {{dt:.2f}}s")
'''

ALL_CONFIGS = {
  "baseline":             (False, {}),
  "quantized":            (True,  {}),
  "baseline+JITBEAM=2":   (False, {"JITBEAM": "2"}),
  "quantized+JITBEAM=2":  (True,  {"JITBEAM": "2"}),
}

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="LLM decode benchmark")
  parser.add_argument("--models", nargs="*", default=list(MODELS.keys()), choices=list(MODELS.keys()))
  parser.add_argument("--configs", nargs="*", default=list(ALL_CONFIGS.keys()), choices=list(ALL_CONFIGS.keys()))
  parser.add_argument("--n", type=int, default=N_TOKENS, help="Tokens to decode")
  args = parser.parse_args()

  for model_name in args.models:
    model_path = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*60}")

    for cfg_name in args.configs:
      use_q, extra_env = ALL_CONFIGS[cfg_name]
      print(f"\n  --- {cfg_name} ---", flush=True)
      env = os.environ.copy()
      env.update(extra_env)
      env["PYTHONIOENCODING"] = "utf-8"

      script = BENCH_SCRIPT.format(model=model_path, n=args.n, warmup=WARMUP, use_quantized=use_q)

      try:
        result = subprocess.run(
          [sys.executable, "-c", script],
          env=env, capture_output=True, text=True, timeout=600
        )
        for line in result.stdout.splitlines():
          if "RESULT:" in line:
            print(f"  {line.strip()}")
            break
        else:
          if result.returncode != 0:
            stderr_lines = result.stderr.strip().splitlines()
            for line in stderr_lines[-5:]:
              print(f"  [stderr] {line}")
            print(f"  FAILED (exit code {result.returncode})")
      except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>600s)")
