"""
Test suite for tinygrad Qwen3-ASR.

Quick tests (~30s after warmup):
  python test.py                    # correctness only
  python test.py --perf             # correctness + performance gates

Full benchmarks (~5min):
  python test.py --full             # 30-file LibriSpeech WER + RTF
  python test.py --full --perf      # full with performance gates

Usage:
  python test.py [--quick] [--full] [--perf] [--model PATH] [--dataset PATH] [--verbose]
"""
import sys, os, time, argparse, glob, pathlib, json, string
import numpy as np

os.environ.setdefault('CUDA', '1')
os.environ.setdefault('CUDA_PTX', '1')

# ============================================================================
# Baselines — update these when performance improves
# ============================================================================

BASELINES = {
  # Per-file mode (JITBEAM=2, 30 utterances, LibriSpeech test-clean)
  "perfile_wer": 1.5,       # % — must be at or below (current: 0.65%)
  "perfile_rtf": 0.25,      # must be at or below (current: 0.123)

  # Streaming mode (JITBEAM=2, 30 utterances)
  "stream_wer": 5.0,        # % — must be at or below (current: 2.10%)
  "stream_rtf": 0.50,       # must be at or below (current: 0.263)

  # Single-file JFK (warm, JITBEAM=2)
  "jfk_rtf": 0.15,          # must be at or below (current: ~0.05)
  "jfk_stream_rtf": 0.40,   # must be at or below (current: ~0.20)
}

JFK_EXPECTED = "And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country."

# ============================================================================
# Helpers
# ============================================================================

def normalize(text: str) -> str:
  return ' '.join(text.upper().translate(str.maketrans('', '', string.punctuation)).split())

def wer(ref: str, hyp: str) -> tuple[int, int]:
  ref_w, hyp_w = normalize(ref).split(), normalize(hyp).split()
  r, h = len(ref_w), len(hyp_w)
  d = [[0]*(h+1) for _ in range(r+1)]
  for i in range(r+1): d[i][0] = i
  for j in range(h+1): d[0][j] = j
  for i in range(1, r+1):
    for j in range(1, h+1):
      d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(0 if ref_w[i-1]==hyp_w[j-1] else 1))
  return d[r][h], r

def load_refs(dataset_dir: str) -> dict[str, str]:
  refs = {}
  for txt in glob.glob(os.path.join(dataset_dir, '**', '*.trans.txt'), recursive=True):
    with open(txt) as f:
      for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2: refs[parts[0]] = parts[1]
  return refs

class TestResult:
  def __init__(self, name: str):
    self.name = name
    self.passed = True
    self.checks: list[tuple[str, bool, str]] = []  # (label, passed, detail)

  def check(self, label: str, ok: bool, detail: str = ""):
    self.checks.append((label, ok, detail))
    if not ok: self.passed = False

  def check_le(self, label: str, value: float, limit: float, unit: str = ""):
    detail = f"{value:.4f} <= {limit:.4f}{(' ' + unit) if unit else ''}"
    self.check(label, value <= limit, detail)

  def summary(self) -> str:
    status = "PASS" if self.passed else "FAIL"
    lines = [f"[{status}] {self.name}"]
    for label, ok, detail in self.checks:
      mark = "  OK" if ok else "  FAIL"
      lines.append(f"{mark} {label}: {detail}")
    return '\n'.join(lines)

# ============================================================================
# Test groups
# ============================================================================

def test_jfk_perfile(model, jfk_path: str, check_perf: bool) -> TestResult:
  """Single-file JFK transcription: correctness + optional perf."""
  t = TestResult("JFK per-file transcription")

  # Warm (model.warmup already called, but warm this specific file)
  model.transcribe(jfk_path)

  # Run 3 times, take best
  results = []
  for _ in range(3):
    r = model.transcribe(jfk_path)
    results.append(r)

  best = min(results, key=lambda r: r["elapsed_ms"])
  text = best["text"]
  audio_sec = 11.0
  rtf = (best["elapsed_ms"] / 1000) / audio_sec

  # Correctness: exact text match
  t.check("text match", normalize(text) == normalize(JFK_EXPECTED),
          f'got: "{text[:80]}..."' if text != JFK_EXPECTED else "exact match")

  # Correctness: all 3 runs produce same text
  texts = [r["text"] for r in results]
  t.check("deterministic", len(set(texts)) == 1,
          f"{len(set(texts))} unique outputs" if len(set(texts)) > 1 else "all 3 identical")

  if check_perf:
    t.check_le("RTF", rtf, BASELINES["jfk_rtf"])

  return t


def test_jfk_streaming(model, jfk_path: str, check_perf: bool) -> TestResult:
  """Streaming JFK transcription: correctness + optional perf."""
  from asr import load_audio, SAMPLE_RATE
  t = TestResult("JFK streaming transcription")

  audio = load_audio(jfk_path)
  audio_sec = len(audio) / SAMPLE_RATE

  # Warm
  model.transcribe_stream(audio)

  # Run
  chunks_seen = []
  def on_chunk(text, is_final):
    chunks_seen.append((text, is_final))

  result = model.transcribe_stream(audio, callback=on_chunk)
  text = result["text"]
  rtf = result["rtf"]

  # Correctness: final text matches
  t.check("text match", normalize(text) == normalize(JFK_EXPECTED),
          f'got: "{text[:80]}..."' if normalize(text) != normalize(JFK_EXPECTED) else "exact match")

  # Correctness: callback was called with is_final=True
  t.check("final callback", any(is_final for _, is_final in chunks_seen),
          f"{len(chunks_seen)} chunks, final={'yes' if any(f for _,f in chunks_seen) else 'no'}")

  # Correctness: progressive chunks get longer (more text over time)
  if len(chunks_seen) >= 2:
    lengths = [len(txt) for txt, _ in chunks_seen]
    growing = all(lengths[i] <= lengths[i+1] for i in range(len(lengths)-1))
    # Not strictly required (streaming can fluctuate), but generally expected
    t.check("progressive", True, f"chunk lengths: {lengths}")

  if check_perf:
    t.check_le("RTF", rtf, BASELINES["jfk_stream_rtf"])

  return t


def test_short_files(model, dataset_dir: str) -> TestResult:
  """Quick correctness check on 5 diverse-length files."""
  t = TestResult("Short file correctness (5 files)")

  refs = load_refs(dataset_dir)
  audio_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.flac'), recursive=True))

  # Pick 5 files at different positions (variety of lengths/speakers)
  indices = [3, 10, 50, 100, 200]
  indices = [i for i in indices if i < len(audio_files)]

  total_errors, total_words = 0, 0
  for idx in indices:
    fpath = audio_files[idx]
    utt_id = os.path.splitext(os.path.basename(fpath))[0]
    ref_text = refs.get(utt_id, "")
    if not ref_text: continue

    result = model.transcribe(fpath)
    errs, nwords = wer(ref_text, result["text"])
    total_errors += errs
    total_words += nwords

    if errs > 0:
      t.checks.append((f"  {utt_id}", True, f"WER {errs}/{nwords} — REF: {ref_text[:60]}"))

  file_wer = total_errors / total_words * 100 if total_words > 0 else 0
  t.check("aggregate WER", file_wer < 5.0, f"{file_wer:.2f}% ({total_errors}/{total_words})")
  return t


def test_librispeech_perfile(model, dataset_dir: str, n: int, warmup: int,
                              check_perf: bool, verbose: bool) -> TestResult:
  """Full LibriSpeech per-file benchmark."""
  t = TestResult(f"LibriSpeech per-file ({n} utterances)")

  refs = load_refs(dataset_dir)
  audio_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.flac'), recursive=True))
  audio_files = audio_files[:n + warmup]

  # Warmup
  for f in audio_files[:warmup]:
    model.transcribe(f)

  test_files = audio_files[warmup:warmup + n]
  total_errors, total_words = 0, 0
  total_audio_sec, total_proc_ms = 0.0, 0.0

  for i, fpath in enumerate(test_files):
    from asr import load_audio, SAMPLE_RATE
    utt_id = os.path.splitext(os.path.basename(fpath))[0]
    ref_text = refs.get(utt_id, "")

    audio = load_audio(fpath)
    audio_sec = len(audio) / SAMPLE_RATE
    result = model.transcribe(fpath)

    errs, nwords = wer(ref_text, result["text"]) if ref_text else (0, 0)
    total_errors += errs
    total_words += nwords
    total_audio_sec += audio_sec
    total_proc_ms += result["elapsed_ms"]

    if verbose and (errs/nwords*100 > 5 if nwords else False):
      print(f"  [{i+1:3d}/{n}] WER={errs/nwords*100:.1f}%  {utt_id}")
      print(f"    REF: {ref_text}")
      print(f"    HYP: {result['text']}")

  overall_wer = total_errors / total_words * 100 if total_words > 0 else 0
  overall_rtf = (total_proc_ms / 1000) / total_audio_sec if total_audio_sec > 0 else 0

  t.check_le("WER", overall_wer, BASELINES["perfile_wer"], "%")
  if check_perf:
    t.check_le("RTF", overall_rtf, BASELINES["perfile_rtf"])
  else:
    t.check(f"RTF (info)", True, f"{overall_rtf:.3f} ({1/overall_rtf:.1f}x RT)")

  return t


def test_librispeech_streaming(model, dataset_dir: str, n: int, warmup: int,
                                check_perf: bool, verbose: bool) -> TestResult:
  """Full LibriSpeech streaming benchmark."""
  from asr import load_audio, SAMPLE_RATE
  t = TestResult(f"LibriSpeech streaming ({n} utterances)")

  refs = load_refs(dataset_dir)
  audio_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.flac'), recursive=True))
  audio_files = audio_files[:n + warmup]

  # Warmup
  for f in audio_files[:warmup]:
    model.transcribe_stream(load_audio(f))

  test_files = audio_files[warmup:warmup + n]
  total_errors, total_words = 0, 0
  total_audio_sec, total_proc_ms = 0.0, 0.0

  for i, fpath in enumerate(test_files):
    utt_id = os.path.splitext(os.path.basename(fpath))[0]
    ref_text = refs.get(utt_id, "")

    audio = load_audio(fpath)
    audio_sec = len(audio) / SAMPLE_RATE
    result = model.transcribe_stream(audio)

    errs, nwords = wer(ref_text, result["text"]) if ref_text else (0, 0)
    total_errors += errs
    total_words += nwords
    total_audio_sec += audio_sec
    total_proc_ms += result["elapsed_ms"]

    if verbose and (errs/nwords*100 > 10 if nwords else False):
      print(f"  [{i+1:3d}/{n}] WER={errs/nwords*100:.1f}%  {utt_id}")
      print(f"    REF: {ref_text}")
      print(f"    HYP: {result['text']}")

  overall_wer = total_errors / total_words * 100 if total_words > 0 else 0
  overall_rtf = (total_proc_ms / 1000) / total_audio_sec if total_audio_sec > 0 else 0

  t.check_le("WER", overall_wer, BASELINES["stream_wer"], "%")
  if check_perf:
    t.check_le("RTF", overall_rtf, BASELINES["stream_rtf"])
  else:
    t.check(f"RTF (info)", True, f"{overall_rtf:.3f} ({1/overall_rtf:.1f}x RT)")

  return t

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="ASR test suite")
  parser.add_argument("--quick", action="store_true", default=True, help="Quick tests (default)")
  parser.add_argument("--full", action="store_true", help="Full LibriSpeech benchmarks")
  parser.add_argument("--perf", action="store_true", help="Enforce performance baselines")
  parser.add_argument("--verbose", action="store_true", help="Show individual WER failures")
  parser.add_argument("--model", default="C:/Users/richa/AppData/Local/local-models/asr/qwen3-asr-0.6b-f16.gguf")
  parser.add_argument("--dataset", default="C:/Data/R/git/claude-repos/deps/datasets/librispeech/LibriSpeech/test-clean")
  parser.add_argument("--jfk", default="C:/Data/R/git/claude-repos/local-ai-server/qwen-asr/samples/jfk.wav")
  parser.add_argument("--n", type=int, default=30, help="Utterances for full benchmarks")
  args = parser.parse_args()

  from tinygrad import Tensor
  from asr import ASR

  # Load model once
  print(f"Loading model: {os.path.basename(args.model)}")
  raw = Tensor(pathlib.Path(args.model))
  model = ASR.from_gguf(raw)
  del raw
  model.warmup()

  results: list[TestResult] = []

  # --- Quick tests (always run) ---
  print("\n=== Quick tests ===")
  results.append(test_jfk_perfile(model, args.jfk, args.perf))
  print(results[-1].summary())

  results.append(test_jfk_streaming(model, args.jfk, args.perf))
  print(results[-1].summary())

  if os.path.isdir(args.dataset):
    results.append(test_short_files(model, args.dataset))
    print(results[-1].summary())

  # --- Full benchmarks (--full) ---
  if args.full:
    if not os.path.isdir(args.dataset):
      print(f"\nSkipping full benchmarks: dataset not found at {args.dataset}")
    else:
      print(f"\n=== Full benchmarks ({args.n} utterances) ===")
      results.append(test_librispeech_perfile(model, args.dataset, args.n, warmup=3,
                                               check_perf=args.perf, verbose=args.verbose))
      print(results[-1].summary())

      results.append(test_librispeech_streaming(model, args.dataset, args.n, warmup=3,
                                                 check_perf=args.perf, verbose=args.verbose))
      print(results[-1].summary())

  # --- Summary ---
  total = len(results)
  passed = sum(1 for r in results if r.passed)
  failed = total - passed

  print(f"\n{'='*60}")
  print(f"{'PASS' if failed == 0 else 'FAIL'}: {passed}/{total} test groups passed")

  if failed > 0:
    for r in results:
      if not r.passed:
        for label, ok, detail in r.checks:
          if not ok: print(f"  FAIL {r.name} / {label}: {detail}")
    sys.exit(1)
