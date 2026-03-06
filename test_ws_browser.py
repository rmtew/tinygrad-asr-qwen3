"""Mock server for browser WebSocket testing. No model loading.

Run:  py test_ws_browser.py
Open: http://localhost:18091
Ctrl+C to stop.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class MockSession:
  def __init__(self):
    self.chunks = 0
    self.total_samples = 0
  def feed(self, audio, is_final=False):
    self.chunks += 1
    self.total_samples += len(audio)
    secs = self.total_samples / 16000
    print(f"  feed: chunk={self.chunks} samples={len(audio)} total={secs:.1f}s", flush=True)
    return {
      "text": f"chunk {self.chunks}: {secs:.1f}s of audio",
      "committed": f"chunk {self.chunks}: {secs:.1f}s",
      "pending": " (pending)",
      "stats": {"chunk": self.chunks, "audio_sec": round(secs, 1), "rtf": 0.05,
                "total_ms": 30, "enc_ms": 5, "prefill_ms": 10, "decode_ms": 15,
                "enc_windows": 1, "max_windows": 4, "reused": 0, "prompt_len": 50,
                "committed": self.chunks * 5, "pending": 3, "prefix_fed": 0},
    }

import asr
asr.StreamingSession = lambda model, **kw: MockSession()
from asr import ASRHandler, start_ws_server
from tinygrad.viz.serve import TCPServerWithReuse

ASRHandler.model = "mock"
HTTP_PORT = 18091
WS_PORT = HTTP_PORT + 1

ws_server = start_ws_server("mock", WS_PORT)
print(f"HTTP:      http://localhost:{HTTP_PORT}", flush=True)
print(f"WebSocket: ws://localhost:{WS_PORT}", flush=True)
print(f"Ctrl+C to stop\n", flush=True)

server = TCPServerWithReuse(("", HTTP_PORT), ASRHandler)
server.daemon_threads = True
try:
  server.serve_forever()
except KeyboardInterrupt:
  print("\nshutdown")
  server.server_close()
  ws_server.shutdown()
