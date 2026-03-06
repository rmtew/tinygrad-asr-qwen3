#!/usr/bin/env python3
"""Diagnostic test: exercise StreamingSession.feed() directly, simulating browser mic sends.

Usage:
  py test_session.py [audio_file]                # default: JFK sample
  py test_session.py path/to/long_audio.wav      # any WAV/FLAC file

Output:
  - Server-style stderr diagnostics (chunk-by-chunk decode/commit/emit)
  - Per-chunk committed summary
  - Final text vs per-file reference comparison
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault("CUDA", "1")
os.environ.setdefault("CUDA_PTX", "1")

from asr import ASR, StreamingSession, load_audio, SAMPLE_RATE

MODEL = os.environ.get("MODEL", "C:/Users/richa/AppData/Local/local-models/asr/qwen3-asr-0.6b-f16.gguf")
JFK = "C:/Data/R/git/claude-repos/local-ai-server/qwen-asr/samples/jfk.wav"

def test_session(model, audio_path, chunk_sec=2.0):
  audio = load_audio(audio_path)
  audio_sec = len(audio) / SAMPLE_RATE
  print(f"\nAudio: {audio_path} ({audio_sec:.1f}s, {len(audio)} samples)")
  print(f"Chunk: {chunk_sec}s ({int(chunk_sec * SAMPLE_RATE)} samples)\n")

  sess = StreamingSession(model, chunk_sec=chunk_sec)
  chunk_samples = int(chunk_sec * SAMPLE_RATE)

  pos = 0
  t0 = time.time()
  while pos < len(audio):
    end = min(pos + chunk_samples, len(audio))
    is_final = (end >= len(audio))
    result = sess.feed(audio[pos:end], is_final=is_final)
    pos = end

  wall_sec = time.time() - t0
  print(f"\n{'='*70}")
  print(f"STREAMING SESSION RESULT ({wall_sec:.1f}s wall, {audio_sec:.1f}s audio, RTF={wall_sec/audio_sec:.3f})")
  print(f"  chunks={sess.chunk_idx}  committed={len(sess.stable_text_tokens)}tok  raw={len(sess.raw_tokens)}tok")
  print(f"{'='*70}")
  print(f"\n--- Streaming text ({len(result['text'])} chars) ---")
  print(result["text"])

  # Compare with per-file transcription
  print("\n--- Per-file reference ---")
  ref = model.transcribe(audio_path)
  print(ref["text"])

  # Quick diff
  s_words = result["text"].split()
  r_words = ref["text"].split()
  if s_words == r_words:
    print(f"\n[MATCH] Streaming output identical to per-file ({len(s_words)} words)")
  else:
    # simple word-level diff
    missing = [w for w in r_words if w not in s_words]
    extra = [w for w in s_words if w not in r_words]
    print(f"\n[DIFF] stream={len(s_words)} words, ref={len(r_words)} words")
    if missing: print(f"  missing from stream: {' '.join(missing[:20])}{'...' if len(missing)>20 else ''}")
    if extra: print(f"  extra in stream: {' '.join(extra[:20])}{'...' if len(extra)>20 else ''}")

if __name__ == "__main__":
  from tinygrad import Tensor
  audio_path = sys.argv[1] if len(sys.argv) > 1 else JFK
  print(f"Loading model: {MODEL}")
  raw = Tensor.empty(os.path.getsize(MODEL), dtype='uint8', device=f"disk:{MODEL}")
  model = ASR.from_gguf(raw)
  del raw
  import gc; gc.collect()
  model.warmup()
  test_session(model, audio_path)
