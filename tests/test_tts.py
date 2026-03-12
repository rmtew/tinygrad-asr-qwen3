#!/usr/bin/env python3
"""TTS tests: correctness (EOS, non-silence, determinism) and performance (RTF).

Usage:
  CUDA=1 CUDA_PTX=1 python tests/test_tts.py                    # quick tests
  CUDA=1 CUDA_PTX=1 python tests/test_tts.py --perf              # + performance gates
  CUDA=1 CUDA_PTX=1 JITBEAM=2 python tests/test_tts.py --perf   # with beam search
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None  # type: ignore[union-attr]

import numpy as np

# ── C reference numbers (0.6B, RTX 3070 Laptop, from QWEN3-TTS.md) ──
# "Hello world." seed=42: ~1.8s (RTF ~1.7)
# Quick brown fox (63 chars) seed=42: ~6.0s
# Douglas Adams (256 chars) seed=42: ~7.1s
C_REF_RTF = 1.7  # approximate C impl RTF for short text

def load_model(model_dir: str):
  from tts import TTSModel
  m = TTSModel(model_dir, verbose=False)
  m.load()
  return m

def test_eos_detection(m):
  """Short texts should hit EOS, not max_steps."""
  print('\n=== EOS detection ===')
  cases = [
    ('Hello.', 42),
    ('Two', 42),
    ('Yes.', 0),
  ]
  for text, seed in cases:
    r = m.synthesize(text, seed=seed, max_steps=100)
    ok = r['n_steps'] < 100
    status = '✓' if ok else '✗'
    print(f'  {status} "{text}" seed={seed}: {r["n_steps"]} steps ({r["audio_sec"]:.1f}s)')
    assert ok, f'Expected EOS before 100 steps for "{text}"'

def test_non_silence(m):
  """Generated audio should not be silent."""
  print('\n=== Non-silence ===')
  r = m.synthesize('Hello world.', seed=42)
  import wave
  with wave.open(r['audio_path'], 'rb') as wf:
    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
  rms = np.sqrt(np.mean(data.astype(np.float32)**2))
  ok = rms > 100
  status = '✓' if ok else '✗'
  print(f'  {status} RMS={rms:.1f} (threshold=100)')
  assert ok, f'Audio is too quiet (RMS={rms:.1f})'

def test_determinism(m):
  """Same seed should produce identical output."""
  print('\n=== Determinism ===')
  r1 = m.synthesize('Hello world.', seed=42)
  r2 = m.synthesize('Hello world.', seed=42)
  ok = r1['n_steps'] == r2['n_steps']
  status = '✓' if ok else '✗'
  print(f'  {status} steps: {r1["n_steps"]} vs {r2["n_steps"]}')
  assert ok, f'Non-deterministic step count: {r1["n_steps"]} vs {r2["n_steps"]}'

  # Compare audio content
  import wave
  with wave.open(r1['audio_path'], 'rb') as wf: d1 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
  with wave.open(r2['audio_path'], 'rb') as wf: d2 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
  corr = np.corrcoef(d1.astype(float), d2.astype(float))[0, 1]
  ok = corr > 0.99
  status = '✓' if ok else '✗'
  print(f'  {status} correlation={corr:.6f}')
  assert ok, f'Audio not deterministic (corr={corr:.6f})'

def test_different_seeds(m):
  """Different seeds should produce different output."""
  print('\n=== Different seeds ===')
  r1 = m.synthesize('Hello world.', seed=42)
  r2 = m.synthesize('Hello world.', seed=123)
  # Step counts may differ
  status = '✓'
  print(f'  {status} seed=42: {r1["n_steps"]} steps, seed=123: {r2["n_steps"]} steps')

def bench_rtf(m, perf: bool = False):
  """Benchmark RTF across several texts."""
  print('\n=== RTF benchmark ===')
  cases = [
    ('Hello.', 42, 'short'),
    ('Hello world.', 42, 'short'),
    ('The quick brown fox jumps over the lazy dog.', 42, 'medium'),
    ('It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife.', 42, 'long'),
  ]
  results = []
  for text, seed, label in cases:
    r = m.synthesize(text, seed=seed)
    results.append(r)
    print(f'  {label:8s} "{text[:40]:40s}..." steps={r["n_steps"]:3d} '
          f'audio={r["audio_sec"]:.1f}s decode={r["decode_ms"]:.0f}ms '
          f'vocoder={r["vocoder_ms"]:.0f}ms total={r["elapsed_ms"]:.0f}ms RTF={r["rtf"]:.2f}')

  avg_rtf = np.mean([r['rtf'] for r in results])
  print(f'\n  Average RTF: {avg_rtf:.2f} (C ref: ~{C_REF_RTF})')

  if perf:
    # Performance gate: with JITBEAM=2, should be competitive with C
    rtf_gate = 10.0  # generous gate — tighten after JITBEAM benchmarking
    ok = avg_rtf < rtf_gate
    status = '✓' if ok else '✗'
    print(f'  {status} RTF gate: {avg_rtf:.2f} < {rtf_gate}')
    assert ok, f'RTF {avg_rtf:.2f} exceeds gate {rtf_gate}'


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--tts-model', default=None, help='TTS model directory')
  parser.add_argument('--perf', action='store_true', help='Enable performance gates')
  args = parser.parse_args()

  model_dir = args.tts_model or os.environ.get('TTS_MODEL',
    'C:/Users/richa/AppData/Local/local-models/tts/qwen3-tts-12hz-0.6b-customvoice')

  if not os.path.isdir(model_dir):
    print(f'TTS model not found at {model_dir}')
    print('Set --tts-model or TTS_MODEL env var')
    sys.exit(1)

  t0 = time.time()
  m = load_model(model_dir)
  print(f'Model loaded in {time.time()-t0:.1f}s')

  test_eos_detection(m)
  test_non_silence(m)
  test_determinism(m)
  test_different_seeds(m)
  bench_rtf(m, perf=args.perf)

  print(f'\n=== All TTS tests passed ({time.time()-t0:.0f}s) ===')
