#!/usr/bin/env python3
"""Parameter sweep for streaming quality optimization.

Grid-searches (chunk_sec, rollback) pairs on captured audio,
reports WER for each combo vs per-file reference.

Usage:
  py sweep_params.py captures/session_*.wav
  py sweep_params.py --chunks 2,3,4,6,8 --rollbacks 3,4,5,6 captures/*.wav
  py sweep_params.py --all-audio   # use captures/ + known test files
"""
import sys, os, time, argparse, glob
os.environ.setdefault("CUDA", "1")
os.environ.setdefault("CUDA_PTX", "1")

from asr import ASR, StreamingSession, load_audio, SAMPLE_RATE
StreamingSession.verbose = False  # suppress per-chunk logging during sweep

MODEL = os.environ.get("MODEL", "C:/Users/richa/AppData/Local/local-models/asr/qwen3-asr-0.6b-f16.gguf")

def word_error_rate(ref_words, hyp_words):
  """WER via edit distance. Returns (errors, ref_count)."""
  r, h = ref_words, hyp_words
  d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
  for i in range(len(r)+1): d[i][0] = i
  for j in range(len(h)+1): d[0][j] = j
  for i in range(1, len(r)+1):
    for j in range(1, len(h)+1):
      d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(0 if r[i-1]==h[j-1] else 1))
  return d[len(r)][len(h)], len(r)

def stream_transcribe(model, audio, chunk_sec=2.0, rollback=5):
  """Run StreamingSession, return (text, stats)."""
  sess = StreamingSession(model, chunk_sec=chunk_sec)
  sess.rollback = rollback
  chunk_samples = int(chunk_sec * SAMPLE_RATE)
  result = None
  pos = 0
  n_chunks = 0
  t0 = time.time()
  while pos < len(audio):
    end = min(pos + chunk_samples, len(audio))
    result = sess.feed(audio[pos:end], is_final=(end >= len(audio)))
    pos = end
    n_chunks += 1
  elapsed = time.time() - t0
  text = result["text"] if result else ""
  return text, {"chunks": n_chunks, "elapsed_s": elapsed, "rtf": elapsed / (len(audio) / SAMPLE_RATE)}

def run_sweep(model, audio_files, chunk_secs, rollbacks, refs):
  """Run full grid sweep. Returns list of result dicts."""
  results = []
  total = len(audio_files) * len(chunk_secs) * len(rollbacks)
  done = 0

  for path, ref_text in zip(audio_files, refs):
    name = os.path.basename(path)
    audio = load_audio(path)
    audio_sec = len(audio) / SAMPLE_RATE
    ref_words = ref_text.lower().split()

    for cs in chunk_secs:
      for rb in rollbacks:
        done += 1
        sys.stderr.write(f"\r  [{done}/{total}] {name} chunk={cs}s rb={rb}  ")
        sys.stderr.flush()
        text, stats = stream_transcribe(model, audio, chunk_sec=cs, rollback=rb)
        hyp_words = text.lower().split()
        errs, n_ref = word_error_rate(ref_words, hyp_words)
        wer = errs / n_ref * 100 if n_ref > 0 else 0
        results.append({
          "file": name, "audio_sec": round(audio_sec, 1),
          "chunk_sec": cs, "rollback": rb,
          "wer": round(wer, 1), "errors": errs, "ref_words": n_ref,
          "hyp_words": len(hyp_words), "chunks": stats["chunks"],
          "rtf": round(stats["rtf"], 3), "elapsed_s": round(stats["elapsed_s"], 1),
        })
  sys.stderr.write("\r" + " "*60 + "\r")
  return results

def print_results(results, audio_files):
  """Print sweep results as table + summary."""
  # Per-file tables
  for path in audio_files:
    name = os.path.basename(path)
    file_results = [r for r in results if r["file"] == name]
    if not file_results: continue

    chunk_secs = sorted(set(r["chunk_sec"] for r in file_results))
    rollbacks = sorted(set(r["rollback"] for r in file_results))

    print(f"\n{'='*70}")
    print(f"  {name} ({file_results[0]['audio_sec']}s, {file_results[0]['ref_words']} ref words)")
    print(f"{'='*70}")

    # WER table
    header = f"  {'chunk\\rb':>10s}"
    for rb in rollbacks: header += f"  rb={rb:d}"
    print(header)
    print(f"  {'-'*10}" + "".join(f"  {'-----':>5s}" for _ in rollbacks))

    for cs in chunk_secs:
      row = f"  {cs:>8.0f}s  "
      for rb in rollbacks:
        r = next((x for x in file_results if x["chunk_sec"] == cs and x["rollback"] == rb), None)
        if r: row += f"  {r['wer']:4.1f}%"
        else: row += f"  {'---':>5s}"
      print(row)

    # RTF table
    print("\n  RTF (processing_time / audio_duration):")
    for cs in chunk_secs:
      row = f"  {cs:>8.0f}s  "
      for rb in rollbacks:
        r = next((x for x in file_results if x["chunk_sec"] == cs and x["rollback"] == rb), None)
        if r: row += f"  {r['rtf']:5.3f}"
        else: row += f"  {'---':>5s}"
      print(row)

  # Overall best combos
  print(f"\n{'='*70}")
  print("  BEST COMBINATIONS (averaged across files)")
  print(f"{'='*70}")

  combos = {}
  for r in results:
    key = (r["chunk_sec"], r["rollback"])
    if key not in combos: combos[key] = {"wers": [], "rtfs": []}
    combos[key]["wers"].append(r["wer"])
    combos[key]["rtfs"].append(r["rtf"])

  ranked = sorted(combos.items(), key=lambda x: sum(x[1]["wers"]) / len(x[1]["wers"]))
  print(f"  {'chunk_sec':>10s} {'rollback':>10s} {'avg WER':>10s} {'max WER':>10s} {'avg RTF':>10s}")
  for (cs, rb), v in ranked:
    avg_wer = sum(v["wers"]) / len(v["wers"])
    max_wer = max(v["wers"])
    avg_rtf = sum(v["rtfs"]) / len(v["rtfs"])
    print(f"  {cs:>9.0f}s {rb:>10d} {avg_wer:>9.1f}% {max_wer:>9.1f}% {avg_rtf:>10.3f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Streaming parameter sweep")
  parser.add_argument("audio", nargs="*", help="WAV files to test")
  parser.add_argument("--chunks", default="2,4,6,8", help="Comma-separated chunk sizes in seconds")
  parser.add_argument("--rollbacks", default="3,5,7", help="Comma-separated rollback values")
  parser.add_argument("--all-audio", action="store_true", help="Include captures/ + standard test files")
  args = parser.parse_args()

  chunk_secs = [float(x) for x in args.chunks.split(",")]
  rollbacks = [int(x) for x in args.rollbacks.split(",")]

  audio_files = list(args.audio)
  if args.all_audio or not audio_files:
    # Add captures
    caps = sorted(glob.glob("captures/*.wav"))
    audio_files.extend(caps)
    # Add standard test files
    extras = [
      "C:/Data/R/git/claude-repos/local-ai-server/qwen-asr/samples/jfk.wav",
      "C:/Data/R/git/claude-repos/local-ai-server/qwen-asr/samples/night_of_the_living_dead_1968/119s_theres_supposed_to_be_another_broadcast.wav",
    ]
    for e in extras:
      if os.path.exists(e) and e not in audio_files: audio_files.append(e)

  audio_files = [f for f in audio_files if os.path.exists(f)]
  if not audio_files:
    print("No audio files found. Pass WAV paths or use --all-audio")
    sys.exit(1)

  print(f"Loading model: {MODEL}")
  from tinygrad import Tensor
  import pathlib, gc
  raw = Tensor(pathlib.Path(MODEL))
  model = ASR.from_gguf(raw)
  del raw; gc.collect()
  model.warmup()

  # Generate per-file references
  print(f"\nGenerating per-file references for {len(audio_files)} files...")
  refs = []
  for path in audio_files:
    name = os.path.basename(path)
    sys.stderr.write(f"  ref: {name}...\r")
    result = model.transcribe(path)
    refs.append(result["text"])
    audio_sec = len(load_audio(path)) / SAMPLE_RATE
    print(f"  {name} ({audio_sec:.1f}s): {result['text'][:80]}...")

  # Run sweep
  n_combos = len(chunk_secs) * len(rollbacks) * len(audio_files)
  print(f"\nSweeping {len(chunk_secs)} chunk sizes × {len(rollbacks)} rollbacks × {len(audio_files)} files = {n_combos} runs")
  t0 = time.time()
  results = run_sweep(model, audio_files, chunk_secs, rollbacks, refs)
  print(f"Sweep completed in {time.time()-t0:.0f}s")

  print_results(results, audio_files)
