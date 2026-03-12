#!/usr/bin/env python3
"""JITBEAM diagnostic: test each TTS JIT path in isolation.

Run with JITBEAM=2 to measure beam search time per path.
Run without JITBEAM to count kernels and estimate.

Usage:
  CUDA=1 CUDA_PTX=1 python tools/jitbeam_diag.py                    # kernel counts only
  CUDA=1 CUDA_PTX=1 JITBEAM=2 python tools/jitbeam_diag.py          # actual beam search
  CUDA=1 CUDA_PTX=1 JITBEAM=2 python tools/jitbeam_diag.py --vocoder-only
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, UOp, getenv
from tinygrad.engine.jit import TinyJit
import tinygrad.nn as nn


def count_unique(jit: TinyJit) -> int:
  unique = set()
  for ji in jit.jit_cache:
    unique.add(ji.prg.name if hasattr(ji.prg, 'name') else id(ji.prg))
  return len(unique)


def test_talker(n_layers=28):
  from tts import Qwen3Block
  print(f'\n=== Talker decode ({n_layers} layers, T=1) ===')
  blks = [Qwen3Block(1024, 3072, 16, 8, 128) for _ in range(n_layers)]
  norm = nn.RMSNorm(1024, 1e-6)
  v = UOp.variable('tsp', 0, 4095)

  @TinyJit
  def fwd(x: Tensor, sp: UOp) -> Tensor:
    for b in blks: x = b(x, sp)
    return norm(x)

  x = Tensor.randn(1, 1, 1024).contiguous().realize()
  t0 = time.time()
  fwd(x, v.bind(0)).realize()
  t_first = time.time() - t0

  t0 = time.time()
  fwd(x, v.bind(1)).realize()
  t_jit = time.time() - t0

  total = len(fwd.jit_cache)
  unique = count_unique(fwd)
  print(f'  First call:  {t_first:.1f}s')
  print(f'  JIT replay:  {t_jit:.1f}s')
  print(f'  Kernels:     {total} total, {unique} unique')
  return total, unique, t_jit


def test_cp_prefill(n_layers=5):
  from tts import Qwen3Block
  print(f'\n=== Code predictor prefill ({n_layers} layers, T=2) ===')
  blks = [Qwen3Block(1024, 3072, 16, 8, 128, max_context=32) for _ in range(n_layers)]
  norm = nn.RMSNorm(1024, 1e-6)

  @TinyJit
  def fwd(buf: Tensor) -> Tensor:
    for b in blks: buf = b(buf, start_pos=0)
    return norm(buf)

  buf = Tensor.randn(1, 2, 1024).contiguous().realize()
  t0 = time.time()
  fwd(buf).realize()
  t_first = time.time() - t0

  for b in blks:
    if hasattr(b, 'cache_kv'):
      b.cache_kv.assign(Tensor.zeros_like(b.cache_kv).contiguous()).realize()
  buf = Tensor.randn(1, 2, 1024).contiguous().realize()
  t0 = time.time()
  fwd(buf).realize()
  t_jit = time.time() - t0

  total = len(fwd.jit_cache)
  unique = count_unique(fwd)
  print(f'  First call:  {t_first:.1f}s')
  print(f'  JIT replay:  {t_jit:.1f}s')
  print(f'  Kernels:     {total} total, {unique} unique')
  return total, unique, t_jit


def test_cp_decode(n_layers=5):
  from tts import Qwen3Block
  print(f'\n=== Code predictor decode ({n_layers} layers, T=1) ===')
  blks = [Qwen3Block(1024, 3072, 16, 8, 128, max_context=32) for _ in range(n_layers)]
  norm = nn.RMSNorm(1024, 1e-6)
  v = UOp.variable('cpd', 0, 31)

  @TinyJit
  def fwd(x: Tensor, sp: UOp) -> Tensor:
    for b in blks: x = b(x, sp)
    return norm(x)

  x = Tensor.randn(1, 1, 1024).contiguous().realize()
  t0 = time.time()
  fwd(x, v.bind(2)).realize()
  t_first = time.time() - t0

  t0 = time.time()
  fwd(x, v.bind(3)).realize()
  t_jit = time.time() - t0

  total = len(fwd.jit_cache)
  unique = count_unique(fwd)
  print(f'  First call:  {t_first:.1f}s')
  print(f'  JIT replay:  {t_jit:.1f}s')
  print(f'  Kernels:     {total} total, {unique} unique')
  return total, unique, t_jit


def test_vocoder(model_dir: str):
  print(f'\n=== Vocoder ===')
  from tts_vocoder import Vocoder
  voc_dir = os.path.join(os.path.dirname(model_dir), 'Qwen3-TTS-Tokenizer-12Hz')
  voc = Vocoder(voc_dir, verbose=True)
  voc.load()
  voc.init_symbolic(max_T=200)

  codes = np.zeros((1, 16), dtype=np.int64)
  t0 = time.time()
  voc.decode_symbolic(codes)
  t_first = time.time() - t0

  codes5 = np.zeros((5, 16), dtype=np.int64)
  t0 = time.time()
  voc.decode_symbolic(codes5)
  t_jit = time.time() - t0

  total = len(voc._vocoder_jit.jit_cache)
  unique = count_unique(voc._vocoder_jit)
  print(f'  First call (T=1):  {t_first:.1f}s')
  print(f'  Second call (T=5): {t_jit:.1f}s')
  print(f'  Kernels:           {total} total, {unique} unique')
  return total, unique, t_jit


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--tts-model', default='C:/Users/richa/AppData/Local/local-models/tts/qwen3-tts-12hz-0.6b-customvoice')
  parser.add_argument('--talker-only', action='store_true')
  parser.add_argument('--cp-only', action='store_true')
  parser.add_argument('--vocoder-only', action='store_true')
  args = parser.parse_args()

  jitbeam = getenv('JITBEAM', 0)
  print(f'JITBEAM={jitbeam}')

  results = {}
  run_all = not (args.talker_only or args.cp_only or args.vocoder_only)

  if run_all or args.talker_only:
    results['talker'] = test_talker()
  if run_all or args.cp_only:
    results['cp_prefill'] = test_cp_prefill()
    results['cp_decode'] = test_cp_decode()
  if run_all or args.vocoder_only:
    results['vocoder'] = test_vocoder(args.tts_model)

  print('\n=== Summary ===')
  total_kernels = 0
  total_unique = 0
  for name, (tot, uniq, t) in results.items():
    total_kernels += tot
    total_unique += uniq
    print(f'  {name:15s}: {tot:4d} kernels, {uniq:3d} unique, {t:.1f}s')
  print(f'  {"TOTAL":15s}: {total_kernels:4d} kernels, {total_unique:3d} unique')
  if not jitbeam:
    print(f'\n  Estimated JITBEAM time: ~{total_unique * 3}s ({total_unique * 3 / 60:.0f} min) at ~3s/kernel')
    print(f'  Run with JITBEAM=2 to measure actual time')
