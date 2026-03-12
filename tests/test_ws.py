"""Test WebSocket server using the `websockets` library.

Starts a mock server (no model loading), exercises the WS protocol.
"""
import asyncio, json, os, sys, tempfile, shutil
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

class MockSession:
  def __init__(self):
    self.chunks = 0
    self.total_samples = 0
  def feed(self, audio, is_final=False):
    self.chunks += 1
    self.total_samples += len(audio)
    secs = self.total_samples / 16000
    return {
      "text": f"mock {self.chunks} chunks {secs:.1f}s",
      "committed": f"committed {self.chunks}",
      "pending": " pending",
      "stats": {"chunk": self.chunks, "audio_sec": round(secs, 1), "rtf": 0.1,
                "total_ms": 50, "enc_ms": 10, "prefill_ms": 20, "decode_ms": 20,
                "enc_windows": 1, "max_windows": 4, "reused": 0, "prompt_len": 100,
                "committed": 10, "pending": 3, "prefix_fed": 5},
    }

# Patch before importing server
import asr
asr.StreamingSession = lambda model, **kw: MockSession()
from server import start_ws_server

PORT = 18092

async def run_tests():
  from websockets.asyncio.client import connect

  # Start server
  ws_server = start_ws_server("mock", PORT)
  await asyncio.sleep(0.5)

  try:
    # Test 1: start, feed, end
    print("test_start_feed_end...")
    async with connect(f"ws://127.0.0.1:{PORT}") as ws:
      await ws.send(json.dumps({"type": "start"}))
      resp = json.loads(await ws.recv())
      assert resp["status"] == "started", resp
      print("  start: OK")

      for i in range(2):
        samples = np.zeros(16000, dtype=np.int16)
        await ws.send(samples.tobytes())
        resp = json.loads(await ws.recv())
        assert "committed" in resp and "pending" in resp
        print(f"  feed {i+1}: committed={resp['committed']!r}")

      await ws.send(json.dumps({"type": "end"}))
      resp = json.loads(await ws.recv())
      assert resp["status"] == "done"
      assert len(resp["committed"]) > 0
      print(f"  end: {resp['committed']!r}")
    print("  PASS")

    # Test 2: binary without start (dropped)
    print("test_binary_before_start...")
    async with connect(f"ws://127.0.0.1:{PORT}") as ws:
      samples = np.zeros(16000, dtype=np.int16)
      await ws.send(samples.tobytes())
      # No response expected — just close cleanly
    print("  PASS")

    # Test 3: save-audio
    print("test_save_audio...")
    tmpdir = tempfile.mkdtemp(prefix="asr_ws_test_")
    try:
      save_server = asr.start_ws_server("mock", PORT + 1, save_audio_dir=tmpdir)
      await asyncio.sleep(0.3)
      async with connect(f"ws://127.0.0.1:{PORT + 1}") as ws:
        await ws.send(json.dumps({"type": "start"}))
        await ws.recv()
        samples = np.random.randint(-1000, 1000, 8000, dtype=np.int16)
        await ws.send(samples.tobytes())
        await ws.recv()
        await ws.send(json.dumps({"type": "end"}))
        await ws.recv()
      await asyncio.sleep(0.5)  # let finally block run
      wavs = [f for f in os.listdir(tmpdir) if f.endswith('.wav')]
      assert len(wavs) == 1, f"Expected 1 WAV, got {wavs}"
      import wave
      with wave.open(os.path.join(tmpdir, wavs[0]), 'rb') as wf:
        assert wf.getframerate() == 16000
        print(f"  saved: {wavs[0]} ({wf.getnframes()} samples)")
      save_server.shutdown()
    finally:
      shutil.rmtree(tmpdir, ignore_errors=True)
    print("  PASS")

    print("\nAll tests passed!")
  finally:
    ws_server.shutdown()

if __name__ == "__main__":
  asyncio.run(run_tests())
