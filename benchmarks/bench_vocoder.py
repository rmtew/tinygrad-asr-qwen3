#!/usr/bin/env python3
"""Vocoder benchmark: measure decode_symbolic timing with F32 vs F16.

Usage:
  python benchmarks/bench_vocoder.py           # F32 baseline (GGUF)
  python benchmarks/bench_vocoder.py --fp16    # F16 compute
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor

def find_model_dir():
  for candidate in [
    os.path.join(os.environ.get('LOCALAPPDATA', ''), 'local-models', 'tts', 'Qwen3-TTS-Tokenizer-12Hz'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'Qwen3-TTS-Tokenizer-12Hz'),
  ]:
    if os.path.isdir(candidate): return candidate
  return None

def bench(model_dir: str, fp16: bool):
  label = "F16" if fp16 else "F32"
  print(f"\n{'='*50}")
  print(f"  Vocoder benchmark: {label} compute")
  print(f"{'='*50}")

  from tts_vocoder import Vocoder
  voc = Vocoder(model_dir, verbose=True, fp16=fp16)
  voc.load()
  voc.init_symbolic(max_T=200)
  voc.warmup_symbolic()
  print("Warmup done.\n")

  rng = np.random.default_rng(42)
  for T in [10, 20, 44, 92]:
    codes = rng.integers(0, 2048, size=(T, 16), dtype=np.int64)
    _ = voc.decode_symbolic(codes)  # warm JIT cache
    times = []
    for trial in range(3):
      t0 = time.time()
      audio = voc.decode_symbolic(codes)
      elapsed = (time.time() - t0) * 1000
      times.append(elapsed)
    avg = sum(times) / len(times)
    audio_sec = T * 0.08
    print(f"  T={T:3d} ({audio_sec:.1f}s): {avg:.0f}ms avg  [{', '.join(f'{t:.0f}' for t in times)}ms]  samples={len(audio)}")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fp16', action='store_true', help='Use F16 compute')
  parser.add_argument('--both', action='store_true', help='Run both F32 and F16')
  args = parser.parse_args()

  model_dir = find_model_dir()
  if not model_dir:
    print("Cannot find vocoder model dir"); sys.exit(1)
  print(f"Model: {model_dir}")

  if args.both:
    bench(model_dir, fp16=False)
    bench(model_dir, fp16=True)
  else:
    bench(model_dir, fp16=args.fp16)

  print("\nDone.")

if __name__ == '__main__':
  main()
