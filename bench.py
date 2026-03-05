"""
Benchmark ASR on LibriSpeech test-clean.

Usage:
  python bench.py [--n 50] [--model path/to/gguf] [--dataset path/to/LibriSpeech/test-clean]

Reports: WER, RTF, tok/s, and per-utterance timing.
"""
import sys, os, time, argparse, glob, pathlib
import numpy as np
os.environ.setdefault('CUDA', '1')
os.environ.setdefault('CUDA_PTX', '1')

from tinygrad import Tensor
from tinygrad.helpers import stderr_log, colored
from asr import ASR, load_audio, compute_mel, SAMPLE_RATE

def load_refs(dataset_dir: str) -> dict[str, str]:
  """Load reference transcriptions from LibriSpeech .trans.txt files."""
  refs = {}
  for txt in glob.glob(os.path.join(dataset_dir, '**', '*.trans.txt'), recursive=True):
    with open(txt) as f:
      for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
          refs[parts[0]] = parts[1]
  return refs

def normalize(text: str) -> str:
  """Normalize text for WER: uppercase, strip punctuation."""
  import string
  return ' '.join(text.upper().translate(str.maketrans('', '', string.punctuation)).split())

def wer(ref: str, hyp: str) -> tuple[int, int]:
  """Word error rate: returns (errors, total_ref_words)."""
  ref_words = normalize(ref).split()
  hyp_words = normalize(hyp).split()
  # Simple Levenshtein on words
  r, h = len(ref_words), len(hyp_words)
  d = [[0] * (h + 1) for _ in range(r + 1)]
  for i in range(r + 1): d[i][0] = i
  for j in range(h + 1): d[0][j] = j
  for i in range(1, r + 1):
    for j in range(1, h + 1):
      d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1,
                    d[i-1][j-1] + (0 if ref_words[i-1] == hyp_words[j-1] else 1))
  return d[r][h], r

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="C:/Users/richa/AppData/Local/local-models/asr/qwen3-asr-0.6b-f16.gguf")
  parser.add_argument("--dataset", default="C:/Data/R/git/claude-repos/deps/datasets/librispeech/LibriSpeech/test-clean")
  parser.add_argument("--n", type=int, default=50, help="Number of utterances")
  parser.add_argument("--warmup", type=int, default=3, help="Warmup runs before benchmarking")
  parser.add_argument("--stream", action="store_true", help="Use streaming mode (concatenated audio)")
  args = parser.parse_args()

  # Load model
  print(f"Loading model: {args.model}")
  raw = Tensor(pathlib.Path(args.model))
  model = ASR.from_gguf(raw)
  del raw
  import gc; gc.collect()

  # Load references
  refs = load_refs(args.dataset)
  print(f"Loaded {len(refs)} reference transcriptions")

  # Collect audio files
  audio_files = sorted(glob.glob(os.path.join(args.dataset, '**', '*.flac'), recursive=True))
  print(f"Found {len(audio_files)} audio files, using first {args.n}")

  if args.stream:
    # ---- Streaming mode: concatenate audio, process as continuous stream ----
    audio_files = audio_files[:args.n + args.warmup]

    # Warmup with a few files in streaming mode
    print(f"\nWarming up ({args.warmup} files in stream mode)...")
    warmup_audio = []
    for f in audio_files[:args.warmup]:
      warmup_audio.append(load_audio(f))
    warmup_concat = np.concatenate(warmup_audio)
    model.transcribe_stream(warmup_concat)
    print("Warmup done.\n")

    # Concatenate test audio
    test_files = audio_files[args.warmup:args.warmup + args.n]
    test_audio_parts = []
    test_refs = []
    total_audio_sec = 0.0
    for f in test_files:
      audio = load_audio(f)
      test_audio_parts.append(audio)
      total_audio_sec += len(audio) / SAMPLE_RATE
      utt_id = os.path.splitext(os.path.basename(f))[0]
      test_refs.append(refs.get(utt_id, ""))

    concat_audio = np.concatenate(test_audio_parts)
    concat_sec = len(concat_audio) / SAMPLE_RATE
    print(f"Concatenated {len(test_files)} utterances: {concat_sec:.1f}s total audio")

    # Run streaming transcription
    result = model.transcribe_stream(concat_audio)
    hyp_text = result["text"]
    ref_text = " ".join(test_refs)

    errs, nwords = wer(ref_text, hyp_text)
    overall_wer = errs / nwords * 100 if nwords > 0 else 0

    print(f"\n{'='*60}")
    print(f"Streaming results ({len(test_files)} utterances, {concat_sec:.1f}s audio):")
    print(f"  WER:     {overall_wer:.2f}% ({errs}/{nwords} words)")
    print(f"  RTF:     {result['rtf']:.3f} ({1/result['rtf']:.1f}x realtime)")
    print(f"  Total:   {result['elapsed_ms']/1000:.1f}s processing")
    print(f"\nFirst 200 chars of output:")
    print(f"  {hyp_text[:200]}")

  else:
    # ---- Per-file mode ----
    audio_files = audio_files[:args.n + args.warmup]

    print(f"\nWarming up ({args.warmup} files)...")
    for f in audio_files[:args.warmup]:
      model.transcribe(f)
    print("Warmup done.\n")

    total_errors, total_words = 0, 0
    total_audio_sec, total_proc_ms = 0.0, 0.0

    test_files = audio_files[args.warmup:args.warmup + args.n]
    for i, fpath in enumerate(test_files):
      utt_id = os.path.splitext(os.path.basename(fpath))[0]
      ref_text = refs.get(utt_id, "")

      audio = load_audio(fpath)
      audio_sec = len(audio) / SAMPLE_RATE

      result = model.transcribe(fpath)
      hyp_text = result["text"]
      proc_ms = result["elapsed_ms"]

      errs, nwords = wer(ref_text, hyp_text) if ref_text else (0, 0)
      total_errors += errs
      total_words += nwords
      total_audio_sec += audio_sec
      total_proc_ms += proc_ms

      rtf = (proc_ms / 1000) / audio_sec if audio_sec > 0 else 0
      utt_wer = errs / nwords * 100 if nwords > 0 else 0

      if utt_wer > 5 or i < 5:
        print(f"[{i+1:3d}/{args.n}] {audio_sec:5.1f}s  {proc_ms:6.0f}ms  RTF={rtf:.2f}  WER={utt_wer:5.1f}%  {utt_id}")
        if utt_wer > 5:
          print(f"    REF: {ref_text}")
          print(f"    HYP: {hyp_text}")

    overall_wer = total_errors / total_words * 100 if total_words > 0 else 0
    overall_rtf = (total_proc_ms / 1000) / total_audio_sec if total_audio_sec > 0 else 0
    avg_ms = total_proc_ms / len(test_files)

    print(f"\n{'='*60}")
    print(f"Results ({len(test_files)} utterances, {total_audio_sec:.1f}s audio):")
    print(f"  WER:     {overall_wer:.2f}% ({total_errors}/{total_words} words)")
    print(f"  RTF:     {overall_rtf:.3f} ({1/overall_rtf:.1f}x realtime)")
    print(f"  Avg:     {avg_ms:.0f}ms per utterance")
    print(f"  Total:   {total_proc_ms/1000:.1f}s processing")
