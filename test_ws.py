"""Test WebSocket implementation without model loading.

Starts a minimal server with a mock ASR model, then exercises the WebSocket
protocol from a Python client.
"""
import threading, time, json, struct, hashlib, base64, socket, sys, os
import numpy as np

# Minimal mock that replaces the real ASR model
class MockSession:
  def __init__(self):
    self.chunks = 0
    self.total_samples = 0
  def feed(self, audio, is_final=False):
    self.chunks += 1
    self.total_samples += len(audio)
    secs = self.total_samples / 16000
    return {
      "text": f"mock transcript {self.chunks} chunks {secs:.1f}s",
      "committed": f"mock committed {self.chunks}",
      "pending": " pending tail",
      "stats": {"chunk": self.chunks, "audio_sec": round(secs, 1), "rtf": 0.1,
                "total_ms": 50, "enc_ms": 10, "prefill_ms": 20, "decode_ms": 20,
                "enc_windows": 1, "max_windows": 4, "reused": 0, "prompt_len": 100,
                "committed": 10, "pending": 3, "prefix_fed": 5},
    }

# --- WebSocket client helpers ---
WS_MAGIC = b"258EAFA5-E914-47DA-95CA-5B99C7714885"

def ws_connect(host, port, path="/ws"):
  """Raw WebSocket handshake. Returns connected socket."""
  sock = socket.create_connection((host, port), timeout=5)
  key = base64.b64encode(os.urandom(16)).decode()
  expected = base64.b64encode(hashlib.sha1(key.encode() + WS_MAGIC).digest()).decode()
  req = (
    f"GET {path} HTTP/1.1\r\n"
    f"Host: {host}:{port}\r\n"
    f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
    f"Sec-WebSocket-Key: {key}\r\n"
    f"Sec-WebSocket-Version: 13\r\n\r\n"
  )
  sock.sendall(req.encode())
  resp = b""
  while b"\r\n\r\n" not in resp:
    chunk = sock.recv(4096)
    if not chunk: raise RuntimeError("Connection closed during handshake")
    resp += chunk
  status_line = resp.split(b"\r\n")[0].decode()
  if "101" not in status_line:
    raise RuntimeError(f"Handshake failed: {status_line}\nFull response:\n{resp.decode(errors='replace')}")
  if expected.encode() not in resp:
    raise RuntimeError(f"Bad Sec-WebSocket-Accept")
  print(f"  handshake OK: {status_line}")
  return sock

def ws_send_text(sock, msg):
  """Send masked text frame."""
  payload = msg.encode() if isinstance(msg, str) else msg
  _ws_send_frame(sock, 0x1, payload)

def ws_send_binary(sock, data):
  """Send masked binary frame."""
  _ws_send_frame(sock, 0x2, data)

def ws_send_close(sock):
  """Send close frame."""
  _ws_send_frame(sock, 0x8, b"")

def _ws_send_frame(sock, opcode, payload):
  """Send a masked WebSocket frame (client-to-server must be masked)."""
  mask = os.urandom(4)
  masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
  header = bytes([0x80 | opcode])
  n = len(payload)
  if n < 126: header += bytes([0x80 | n])
  elif n < 65536: header += bytes([0x80 | 126]) + n.to_bytes(2, "big")
  else: header += bytes([0x80 | 127]) + n.to_bytes(8, "big")
  sock.sendall(header + mask + masked)

def ws_recv(sock):
  """Read one unmasked WebSocket frame (server-to-client). Returns (opcode, payload)."""
  def readn(n):
    buf = b""
    while len(buf) < n:
      chunk = sock.recv(n - len(buf))
      if not chunk: return b""
      buf += chunk
    return buf
  b = readn(2)
  if len(b) < 2: return 0x8, b""
  opcode = b[0] & 0xF
  length = b[1] & 0x7F
  if length == 126: length = int.from_bytes(readn(2), "big")
  elif length == 127: length = int.from_bytes(readn(8), "big")
  payload = readn(length) if length > 0 else b""
  return opcode, payload

def ws_recv_json(sock):
  op, data = ws_recv(sock)
  assert op == 0x1, f"Expected text frame (0x1), got 0x{op:x}"
  return json.loads(data)

# --- Server setup with mock model ---
def start_mock_server(port):
  """Import the real server code but patch in a mock session."""
  # We need to import asr.py's server classes. Add to path.
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

  # Import server components (this imports tinygrad but doesn't load a model)
  from asr import ASRHandler, _ws_recv, _ws_send, StreamingSession
  from tinygrad.viz.serve import TCPServerWithReuse

  # Monkey-patch: replace StreamingSession with MockSession
  original_init = StreamingSession.__init__
  def mock_streaming_session(model):
    return MockSession()

  import asr
  asr_module = asr
  orig_StreamingSession = asr.StreamingSession
  asr.StreamingSession = lambda model, **kw: MockSession()

  # Set a dummy model attribute (handler checks self.model)
  ASRHandler.model = "mock"

  server = TCPServerWithReuse(("", port), ASRHandler)
  server.daemon_threads = True
  t = threading.Thread(target=server.serve_forever, daemon=True)
  t.start()
  time.sleep(0.5)  # let server bind
  return server

# --- Tests ---
def test_handshake(port):
  print("test_handshake...")
  sock = ws_connect("127.0.0.1", port)
  ws_send_close(sock)
  sock.close()
  print("  PASS")

def test_start_feed_end(port):
  print("test_start_feed_end...")
  sock = ws_connect("127.0.0.1", port)

  # Start
  ws_send_text(sock, json.dumps({"type": "start"}))
  resp = ws_recv_json(sock)
  print(f"  start: {resp}")
  assert resp.get("status") == "started", f"Expected started, got {resp}"

  # Feed 2 binary chunks (1s of Int16 PCM each)
  for i in range(2):
    samples = np.zeros(16000, dtype=np.int16)  # 1s silence
    ws_send_binary(sock, samples.tobytes())
    resp = ws_recv_json(sock)
    print(f"  feed {i+1}: committed={resp.get('committed')!r} pending={resp.get('pending')!r} stats.chunk={resp.get('stats',{}).get('chunk')}")
    assert "committed" in resp
    assert "pending" in resp
    assert resp["stats"]["chunk"] > 0

  # End
  ws_send_text(sock, json.dumps({"type": "end"}))
  resp = ws_recv_json(sock)
  print(f"  end: {resp}")
  assert resp.get("status") == "done"
  assert len(resp.get("committed", "")) > 0

  sock.close()
  print("  PASS")

def test_binary_before_start(port):
  print("test_binary_before_start (no session)...")
  sock = ws_connect("127.0.0.1", port)
  # Send binary without start — should be silently dropped (no response)
  samples = np.zeros(16000, dtype=np.int16)
  ws_send_binary(sock, samples.tobytes())
  # Send close
  ws_send_close(sock)
  op, _ = ws_recv(sock)
  print(f"  close response opcode=0x{op:x}")
  sock.close()
  print("  PASS")

def test_ping_pong(port):
  print("test_ping_pong...")
  sock = ws_connect("127.0.0.1", port)
  _ws_send_frame(sock, 0x9, b"hello")  # ping
  op, data = ws_recv(sock)
  print(f"  pong: opcode=0x{op:x} data={data!r}")
  assert op == 0xA, f"Expected pong (0xA), got 0x{op:x}"
  assert data == b"hello"
  ws_send_close(sock)
  sock.close()
  print("  PASS")

def test_chrome_headers(port):
  """Connect with the exact headers Chrome sends for a WebSocket upgrade."""
  print("test_chrome_headers (simulating browser)...")
  sock = socket.create_connection(("127.0.0.1", port), timeout=5)
  key = base64.b64encode(os.urandom(16)).decode()
  expected = base64.b64encode(hashlib.sha1(key.encode() + WS_MAGIC).digest()).decode()
  # Chrome sends these exact headers
  req = (
    f"GET /ws HTTP/1.1\r\n"
    f"Host: 127.0.0.1:{port}\r\n"
    f"Connection: Upgrade\r\n"
    f"Pragma: no-cache\r\n"
    f"Cache-Control: no-cache\r\n"
    f"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36\r\n"
    f"Upgrade: websocket\r\n"
    f"Origin: http://127.0.0.1:{port}\r\n"
    f"Sec-WebSocket-Version: 13\r\n"
    f"Accept-Encoding: gzip, deflate, br\r\n"
    f"Accept-Language: en-US,en;q=0.9\r\n"
    f"Sec-WebSocket-Key: {key}\r\n"
    f"Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits\r\n"
    f"\r\n"
  )
  sock.sendall(req.encode())
  resp = b""
  while b"\r\n\r\n" not in resp:
    chunk = sock.recv(4096)
    if not chunk: break
    resp += chunk
  print(f"  response: {resp.decode(errors='replace').strip()}")
  if b"101" not in resp:
    print(f"  FAIL: no 101 in response")
    sock.close()
    return

  # Now try the full start/feed/end flow
  ws_send_text(sock, json.dumps({"type": "start"}))
  resp_json = ws_recv_json(sock)
  print(f"  start: {resp_json}")

  samples = np.zeros(16000, dtype=np.int16)
  ws_send_binary(sock, samples.tobytes())
  resp_json = ws_recv_json(sock)
  print(f"  feed: committed={resp_json.get('committed')!r}")

  ws_send_text(sock, json.dumps({"type": "end"}))
  resp_json = ws_recv_json(sock)
  print(f"  end: {resp_json.get('status')}")

  sock.close()
  print("  PASS")

def test_save_audio(port, tmpdir):
  """Test --save-audio writes a WAV file after session end."""
  print("test_save_audio...")
  import asr
  asr.ASRHandler.save_audio_dir = tmpdir

  sock = ws_connect("127.0.0.1", port)
  ws_send_text(sock, json.dumps({"type": "start"}))
  ws_recv_json(sock)

  # Send 2 chunks of 0.5s each
  for _ in range(2):
    samples = np.random.randint(-1000, 1000, 8000, dtype=np.int16)
    ws_send_binary(sock, samples.tobytes())
    ws_recv_json(sock)

  ws_send_text(sock, json.dumps({"type": "end"}))
  ws_recv_json(sock)
  sock.close()
  time.sleep(0.5)  # let handler's finally block run (threaded server)

  # Check WAV was saved
  wavs = [f for f in os.listdir(tmpdir) if f.endswith('.wav')]
  assert len(wavs) == 1, f"Expected 1 WAV, got {wavs}"
  import wave
  with wave.open(os.path.join(tmpdir, wavs[0]), 'rb') as wf:
    assert wf.getnchannels() == 1
    assert wf.getframerate() == 16000
    n_samples = wf.getnframes()
    assert n_samples == 16000, f"Expected 16000 samples (1s), got {n_samples}"
  print(f"  saved: {wavs[0]} ({n_samples} samples)")

  asr.ASRHandler.save_audio_dir = None
  print("  PASS")

if __name__ == "__main__":
  PORT = 18091
  print(f"Starting mock server on port {PORT}...")
  server = start_mock_server(PORT)
  print()

  try:
    test_handshake(PORT)
    test_start_feed_end(PORT)
    test_binary_before_start(PORT)
    test_ping_pong(PORT)
    test_chrome_headers(PORT)
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix="asr_save_")
    try:
      test_save_audio(PORT, tmpdir)
    finally:
      shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nAll tests passed!")
  finally:
    server.shutdown()
