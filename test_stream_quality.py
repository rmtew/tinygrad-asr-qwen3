#!/usr/bin/env python3
"""Stream quality test: compare streaming vs per-file on various audio sources.

Measures word-level accuracy of streaming output against per-file reference.
Tests with clean files, real mic recordings, and synthesized noisy audio.

Usage:
  py test_stream_quality.py              # run all tests
  py test_stream_quality.py --noisy      # include noise-degraded tests
"""
import sys, os, time, glob, struct, wave
import numpy as np
os.environ.setdefault("CUDA", "1")
os.environ.setdefault("CUDA_PTX", "1")

from asr import ASR, StreamingSession, load_audio, SAMPLE_RATE

MODEL = os.environ.get("MODEL", "C:/Users/richa/AppData/Local/local-models/asr/qwen3-asr-0.6b-f16.gguf")

def word_error_rate(ref_words, hyp_words):
  """Simple WER via edit distance."""
  r, h = ref_words, hyp_words
  d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
  for i in range(len(r)+1): d[i][0] = i
  for j in range(len(h)+1): d[0][j] = j
  for i in range(1, len(r)+1):
    for j in range(1, len(h)+1):
      d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(0 if r[i-1]==h[j-1] else 1))
  return d[len(r)][len(h)], len(r)

def stream_transcribe(model, audio, chunk_sec=2.0):
  """Run StreamingSession on audio array, return final text."""
  sess = StreamingSession(model, chunk_sec=chunk_sec)
  chunk_samples = int(chunk_sec * SAMPLE_RATE)
  pos = 0
  while pos < len(audio):
    end = min(pos + chunk_samples, len(audio))
    result = sess.feed(audio[pos:end], is_final=(end >= len(audio)))
    pos = end
  return result["text"]

def add_noise(audio, snr_db=20):
  """Add Gaussian noise at given SNR."""
  sig_power = np.mean(audio ** 2)
  noise_power = sig_power / (10 ** (snr_db / 10))
  noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
  return audio + noise

def concat_audio_files(paths, gap_sec=0.3):
  """Concatenate audio files with small gaps between."""
  parts = []
  gap = np.zeros(int(gap_sec * SAMPLE_RATE), dtype=np.float32)
  for p in paths:
    parts.append(load_audio(p))
    parts.append(gap)
  return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

def save_wav(path, audio, sr=16000):
  """Save float32 audio as 16-bit WAV."""
  audio_i16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
  with wave.open(path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(audio_i16.tobytes())

def test_file(model, label, audio, ref_text=None):
  """Test streaming vs per-file on audio array. Returns (stream_wer, ref_wer, stream_text, ref_text)."""
  audio_sec = len(audio) / SAMPLE_RATE

  # Save temp file for per-file transcription
  tmp = f"_tmp_test_{label.replace(' ','_')}.wav"
  save_wav(tmp, audio)

  # Per-file reference
  ref_result = model.transcribe(tmp)
  ref = ref_text or ref_result["text"]

  # Streaming
  stream_text = stream_transcribe(model, audio)

  os.unlink(tmp)

  ref_words = ref.lower().split()
  stream_words = stream_text.lower().split()
  perfile_words = ref_result["text"].lower().split()

  errs_stream, n_ref = word_error_rate(ref_words, stream_words)
  errs_perfile, _ = word_error_rate(ref_words, perfile_words)

  wer_stream = errs_stream / n_ref * 100 if n_ref > 0 else 0
  wer_perfile = errs_perfile / n_ref * 100 if n_ref > 0 else 0
  word_coverage = len(stream_words) / len(ref_words) * 100 if ref_words else 0

  print(f"\n{'='*70}")
  print(f"  {label} ({audio_sec:.1f}s, {len(ref_words)} ref words)")
  print(f"  per-file WER: {wer_perfile:.1f}% ({errs_perfile}/{n_ref})")
  print(f"  stream  WER: {wer_stream:.1f}% ({errs_stream}/{n_ref}), coverage={word_coverage:.0f}%")
  print(f"  stream words: {len(stream_words)}, ref words: {len(ref_words)}")
  if wer_stream > 10:
    # Show the differences
    print(f"  STREAM: {stream_text[:200]}...")
    print(f"  REF:    {ref[:200]}...")
  print(f"{'='*70}")

  return wer_stream, wer_perfile, stream_text, ref

if __name__ == "__main__":
  from tinygrad import Tensor
  import gc

  noisy = "--noisy" in sys.argv

  print(f"Loading model: {MODEL}")
  raw = Tensor.empty(os.path.getsize(MODEL), dtype='uint8', device=f"disk:{MODEL}")
  model = ASR.from_gguf(raw)
  del raw; gc.collect()
  model.warmup()

  results = []

  # --- Test 1: JFK 11s (clean, short) ---
  jfk = "C:/Data/R/git/claude-repos/local-ai-server/qwen-asr/samples/jfk.wav"
  if os.path.exists(jfk):
    audio = load_audio(jfk)
    r = test_file(model, "JFK 11s clean", audio)
    results.append(("JFK 11s clean", *r[:2]))

  # --- Test 2: NOTLD 119s (clean, long, multi-speaker) ---
  notld = "C:/Data/R/git/claude-repos/local-ai-server/qwen-asr/samples/night_of_the_living_dead_1968/119s_theres_supposed_to_be_another_broadcast.wav"
  if os.path.exists(notld):
    audio = load_audio(notld)
    r = test_file(model, "NOTLD 119s clean", audio)
    results.append(("NOTLD 119s clean", *r[:2]))

  # --- Test 3: LibriSpeech concatenated (~60s continuous speech) ---
  ls_dir = "C:/Data/R/git/claude-repos/deps/datasets/librispeech/LibriSpeech/test-clean"
  if os.path.isdir(ls_dir):
    flacs = sorted(glob.glob(f"{ls_dir}/**/*.flac", recursive=True))[:15]
    if flacs:
      audio = concat_audio_files(flacs)
      r = test_file(model, f"LibriSpeech 15-utt concat ({len(audio)/SAMPLE_RATE:.0f}s)", audio)
      results.append(("LS concat", *r[:2]))

  # --- Test 4: Real mic recording ---
  mic_dir = "C:/Data/R/git/claude-repos/lifeapp/recordings"
  if os.path.isdir(mic_dir):
    mics = sorted(glob.glob(f"{mic_dir}/*.wav"), key=os.path.getsize, reverse=True)
    if mics:
      audio = load_audio(mics[0])
      if len(audio) / SAMPLE_RATE > 10:
        r = test_file(model, f"Real mic {len(audio)/SAMPLE_RATE:.0f}s", audio)
        results.append(("Real mic", *r[:2]))

  # --- Test 5: Noisy variants ---
  if noisy and os.path.exists(notld):
    for snr in [20, 10, 5]:
      audio = load_audio(notld)
      noisy_audio = add_noise(audio, snr_db=snr)
      r = test_file(model, f"NOTLD 119s SNR={snr}dB", noisy_audio)
      results.append((f"NOTLD SNR={snr}", *r[:2]))

  # --- Summary ---
  print(f"\n{'='*70}")
  print(f"  SUMMARY")
  print(f"  {'Test':<30s} {'Stream WER':>12s} {'Per-file WER':>14s} {'Gap':>8s}")
  for name, swer, pwer in results:
    print(f"  {name:<30s} {swer:>11.1f}% {pwer:>13.1f}% {swer-pwer:>+7.1f}%")
  print(f"{'='*70}")
