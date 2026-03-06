"""Start mock server for browser WebSocket testing.
Open http://localhost:18091 in browser, click Record, check console."""
import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class MockSession:
  def __init__(self):
    self.chunks = 0
    self.total_samples = 0
  def feed(self, audio, is_final=False):
    self.chunks += 1
    self.total_samples += len(audio)
    secs = self.total_samples / 16000
    from tinygrad.helpers import stderr_log
    stderr_log(f"  MockSession.feed: chunk={self.chunks} samples={len(audio)} total={secs:.1f}s final={is_final}\n")
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
from asr import ASRHandler, _HTML_DIR
from tinygrad.viz.serve import TCPServerWithReuse

# Serve test_ws_page.html as the root page instead of index.html
_orig_do_GET = ASRHandler.do_GET
def _patched_do_GET(self):
  if self.path == '/':
    import os
    html_path = os.path.join(_HTML_DIR, 'test_ws_page.html')
    self.send_data(open(html_path, 'rb').read(), content_type="text/html")
  else:
    _orig_do_GET(self)
ASRHandler.do_GET = _patched_do_GET

ASRHandler.model = "mock"
PORT = 18091
print(f"Mock server on http://localhost:{PORT} — open in browser, click Record")
server = TCPServerWithReuse(("", PORT), ASRHandler)
server.daemon_threads = True
try:
  server.serve_forever()
except KeyboardInterrupt:
  print("shutdown")
  server.server_close()
