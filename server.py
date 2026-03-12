"""Combined ASR + TTS + LLM server with OpenAI-compatible API.

Serves:
  GET  /                        -> ASR microphone UI
  GET  /chat                    -> Chat UI
  GET  /health                  -> {"status":"ok"}
  GET  /v1/models               -> list loaded models
  POST /v1/audio/transcriptions -> one-shot ASR (OpenAI-compatible)
  POST /v1/audio/speech         -> TTS synthesis (OpenAI-compatible)
  POST /v1/chat/completions     -> LLM chat (OpenAI SSE)
  WebSocket (port+1)            -> live streaming ASR

Usage:
  python server.py --asr-model qwen3-asr:0.6b
  python server.py --llm-model qwen3.5:0.8b
  python server.py --asr-model path/to/asr.gguf --llm-model qwen3.5:0.8b --tts-model path/to/tts/dir
"""
import sys, os, argparse, json, time, wave, pathlib, tempfile, gc, ctypes, platform
import numpy as np

from asr import ASR, StreamingSession, SAMPLE_RATE, KNOWN_MODELS, KNOWN_LLM_MODELS  # noqa: F401

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.helpers import colored, stderr_log, GlobalCounters
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler
from tinygrad.apps.llm import Transformer, SimpleTokenizer

import asyncio, threading, queue

# ---- Dispatch queue: route all inference to the main (asyncio) thread ----
# tinygrad's CUDA context and SQLite disk cache are thread-local to the main
# thread. HTTP runs in a daemon thread, so all inference must be dispatched.
# The main thread runs asyncio for WS, but inference calls (ASR feed, LLM
# generate) are synchronous and block the event loop while running. This is
# intentional — GPU can only do one thing at a time, so serialization is correct.
_dispatch_loop: asyncio.AbstractEventLoop | None = None  # set by start_ws_server
_dispatch_ready = threading.Event()  # signals HTTP thread that dispatch is available

def dispatch(fn, *args, **kwargs):
  """Run fn(*args, **kwargs) on the main asyncio thread, blocking the caller."""
  _dispatch_ready.wait()  # block until event loop is running
  fut = asyncio.run_coroutine_threadsafe(_dispatch_call(fn, args, kwargs), _dispatch_loop)
  return fut.result()  # blocks HTTP thread until main thread completes

async def _dispatch_call(fn, args, kwargs):
  """Coroutine wrapper for dispatch — runs sync fn on event loop thread."""
  return fn(*args, **kwargs)

def dispatch_generator(fn, *args, **kwargs):
  """Run a generator fn on the main thread, yielding results to the caller thread.

  Uses a queue to bridge: main thread puts items, caller thread gets them.
  """
  _dispatch_ready.wait()
  q: queue.Queue = queue.Queue()
  _SENTINEL = object()
  async def _run():
    try:
      for item in fn(*args, **kwargs):
        q.put(item)
    except Exception as e:
      q.put(e)
    finally:
      q.put(_SENTINEL)
  asyncio.run_coroutine_threadsafe(_run(), _dispatch_loop)
  while True:
    item = q.get()
    if item is _SENTINEL: break
    if isinstance(item, Exception): raise item
    yield item

# HTML pages served from same directory as this file
_HTML_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_device_info() -> dict:
  """Get tinygrad backend, device name, and memory info."""
  dev_str = Device.DEFAULT
  backend = dev_str.split(":")[0]  # e.g. "CUDA", "AMD", "CPU"
  info: dict = {"backend": backend, "device": dev_str}

  # Try to get GPU name and total VRAM
  try:
    dev = Device[dev_str]
    info["arch"] = getattr(dev, "arch", None)
    if backend == "CUDA":
      from tinygrad.runtime.autogen.cuda import cuDeviceGetName, cuMemGetInfo_v2, cuCtxSetCurrent
      from tinygrad.runtime.autogen.cuda import size_t
      cuCtxSetCurrent(dev.context)
      name_buf = ctypes.create_string_buffer(256)
      cuDeviceGetName(name_buf, 256, dev.cu_device)
      info["gpu_name"] = name_buf.value.decode()
      free, total = size_t(), size_t()
      cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
      info["vram_total_mb"] = round(total.value / 1048576)
      info["vram_free_mb"] = round(free.value / 1048576)
  except Exception:
    pass

  # System RAM
  try:
    import psutil
    vm = psutil.virtual_memory()
    info["ram_total_mb"] = round(vm.total / 1048576)
    info["ram_used_mb"] = round(vm.used / 1048576)
  except ImportError:
    pass

  # tinygrad memory tracking
  info["tinygrad_mem_mb"] = round(GlobalCounters.mem_used / 1048576)
  per_device = {k: round(v / 1048576) for k, v in GlobalCounters.mem_used_per_device.items() if v > 0}
  if per_device:
    info["tinygrad_mem_per_device_mb"] = per_device

  info["platform"] = platform.platform()
  return info

class ServerHandler(HTTPRequestHandler):
  model: ASR | None = None  # ASR model, set if --asr-model provided
  tts_model = None  # TTSModel, set if --tts-model provided
  save_audio_dir: str | None = None  # --save-audio directory
  # LLM chat (optional, set if --llm-model provided)
  llm: Transformer | None = None
  llm_tok: SimpleTokenizer | None = None
  llm_bos_id: int | None = None
  llm_eos_id: int = 0

  def log_request(self, code='-', size='-'): pass

  # model labels set at startup (for stats display)
  _model_labels: dict = {}  # e.g. {"asr": "qwen3-asr:0.6b", "tts": "path/to/tts", "llm": "qwen3.5:0.8b"}

  def do_GET(self):
    if self.path == '/' or self.path == '/chat':
      name = 'chat.html' if self.path == '/chat' else 'index.html'
      html_path = os.path.join(_HTML_DIR, name)
      try: self.send_data(open(html_path, 'rb').read(), content_type="text/html")
      except FileNotFoundError: self.send_error(404, f"{name} not found")
    elif self.path == '/health': self.send_data(b'{"status":"ok"}')
    elif self.path == '/v1/models':
      data = [{"id": "qwen3-asr", "object": "model"}]
      if self.tts_model is not None: data.append({"id": "qwen3-tts", "object": "model"})
      if self.llm is not None: data.append({"id": "qwen3.5", "object": "model"})
      self.send_data(json.dumps({"data": data}).encode())
    elif self.path == '/v1/stats':
      stats = _get_device_info()
      stats["models"] = self._model_labels
      self.send_data(json.dumps(stats).encode())
    elif self.path == '/favicon.ico': self.send_data(b'', content_type="image/x-icon")
    else: self.send_error(404)

  def do_POST(self):
    if self.path == '/v1/audio/transcriptions':
      self._handle_transcribe()
    elif self.path == '/v1/audio/speech':
      self._handle_speech()
    elif self.path == '/v1/chat/completions':
      self._handle_chat()
    else:
      self.send_error(404)

  def _handle_transcribe(self):
    """One-shot file transcription (stateless, per-file mode)."""
    if self.model is None:
      self.send_data(json.dumps({"error": {"message": "No ASR model loaded (use --asr-model)"}}).encode(), status_code=503)
      return
    body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
    content_type = self.headers.get('Content-Type', '')
    audio_data, filename = self._extract_audio(body, content_type)
    if audio_data is None:
      self.send_error(400, "No audio file found"); return

    suffix = os.path.splitext(filename)[1] if filename else '.bin'
    if not suffix or suffix == '.': suffix = '.wav'

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
      f.write(audio_data)
      tmp_path = f.name
    try:
      result = dispatch(self.model.transcribe, tmp_path)
      self.send_data(json.dumps({"text": result["text"]}).encode())
    except Exception as e:
      self.send_data(json.dumps({"error": str(e)}).encode(), status_code=500)
    finally:
      os.unlink(tmp_path)

  def _extract_audio(self, body: bytes, content_type: str) -> tuple[bytes | None, str]:
    """Extract audio bytes and filename from multipart/form-data or raw body."""
    if 'multipart/form-data' not in content_type:
      return body, ''

    # Find boundary
    boundary = None
    for ct_part in content_type.split(';'):
      ct_part = ct_part.strip()
      if ct_part.startswith('boundary='):
        boundary = ct_part[9:].strip('"').encode()
    if boundary is None: return None, ''

    # Find file part
    for part in body.split(b'--' + boundary):
      if b'name="file"' not in part and b'name="audio"' not in part: continue
      # Extract filename from Content-Disposition
      filename = ''
      for line in part.split(b'\r\n'):
        if b'filename=' in line:
          fn = line.split(b'filename=')[1].split(b';')[0].strip(b' "\'')
          filename = fn.decode(errors='replace')
      # Extract body after blank line
      idx = part.find(b'\r\n\r\n')
      if idx < 0: continue
      data = part[idx + 4:]
      if data.endswith(b'\r\n'): data = data[:-2]
      if data.endswith(b'--'): data = data[:-2]
      if data.endswith(b'\r\n'): data = data[:-2]
      return data, filename
    return None, ''

  def _handle_speech(self):
    """OpenAI-compatible /v1/audio/speech endpoint."""
    if self.tts_model is None:
      self.send_data(json.dumps({"error": {"message": "No TTS model loaded (use --tts-model)"}}).encode(), status_code=503)
      return

    body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))).decode())
    text = body.get("input", "")
    if not text:
      self.send_data(json.dumps({"error": {"message": "input is required"}}).encode(), status_code=400)
      return

    voice = body.get("voice")
    try:
      result = dispatch(self.tts_model.synthesize, text, voice=voice)
      with open(result["audio_path"], "rb") as f:
        wav_data = f.read()
      os.unlink(result["audio_path"])
      self.send_data(wav_data, content_type="audio/wav")
    except Exception as e:
      self.send_data(json.dumps({"error": {"message": str(e)}}).encode(), status_code=500)

  def _handle_chat(self):
    """OpenAI-compatible /v1/chat/completions endpoint."""
    import uuid
    if self.llm is None or self.llm_tok is None:
      self.send_error(503, "No LLM model loaded (use --llm-model)"); return

    body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))).decode())
    stream = body.get("stream", False)
    model_name = body.get("model", "qwen3.5")

    # Build token sequence from messages
    tok, llm = self.llm_tok, self.llm
    ids: list[int] = [self.llm_bos_id] if self.llm_bos_id is not None else []
    for msg in body.get("messages", []):
      ids += tok.role(msg["role"])
      content = msg["content"]
      if isinstance(content, str): ids += tok.encode(content)
      elif isinstance(content, list):
        for c in content:
          if c["type"] == "text": ids += tok.encode(c["text"])
      ids += tok.end_turn(self.llm_eos_id)
    ids += tok.role("assistant")

    # Generate
    cache_start = llm.get_start_pos(ids)
    stderr_log(f"/v1/chat/completions  {colored('--', 'BLACK')}  in:{colored(f'{cache_start:5d}', 'green')} +{len(ids)-cache_start:5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model_name}
    t0 = time.perf_counter()
    out_tokens: list[int] = []
    pt = t0  # set to prefill-end time once first token arrives
    stop = threading.Event()  # signal GPU thread to stop on client disconnect

    def gen_chunks():
      nonlocal pt
      yield {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}], **tmpl}
      for next_id in llm.generate(ids):
        if stop.is_set(): break
        if not out_tokens:
          pt = time.perf_counter()
          stderr_log(f"prefill:{(len(ids)-cache_start)/(pt-t0):4.0f} tok/s  {colored('--', 'BLACK')}  ")
        if next_id == self.llm_eos_id: break
        out_tokens.append(next_id)
        yield {"choices": [{"index": 0, "delta": {"content": tok.decode([next_id])}, "finish_reason": None}], **tmpl}
      yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], **tmpl}

    if stream:
      try:
        self.stream_json(dispatch_generator(gen_chunks))
      except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
        stop.set()
        return
    else:
      chunks = list(dispatch_generator(gen_chunks))
      text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
      resp = {**tmpl, "object": "chat.completion",
              "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
      self.send_data(json.dumps(resp).encode())
    gen_ms = (time.perf_counter() - pt) * 1000
    if out_tokens and gen_ms > 0:
      stderr_log(f"gen:{len(out_tokens)/(gen_ms/1000):4.0f} tok/s  {colored('--', 'BLACK')}  out:{len(out_tokens):5d}\n")
    else:
      stderr_log("gen: 0 tokens\n")


# ============================================================================
# WebSocket server (via `websockets` library, separate port)
# ============================================================================

def _save_session_audio(chunks: list[np.ndarray], save_dir: str | None):
  """Save accumulated session audio to WAV if save_dir is set."""
  if not save_dir or not chunks: return
  try:
    audio = np.concatenate(chunks)
    if len(audio) == 0: return
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"session_{ts}.wav")
    int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
      wf.setnchannels(1)
      wf.setsampwidth(2)
      wf.setframerate(SAMPLE_RATE)
      wf.writeframes(int16.tobytes())
    stderr_log(f"saved {len(audio)/SAMPLE_RATE:.1f}s audio to {path}\n")
  except Exception as e:
    stderr_log(f"save-audio error: {e}\n")

def start_ws_server(model: 'ASR | None', port: int, save_audio_dir: str | None = None, run_in_background: bool = True):
  """Start WebSocket server (if ASR model loaded) and dispatch event loop.

  When model is None, only starts the asyncio dispatch loop (for LLM/TTS HTTP handlers).

  Protocol (when ASR enabled):
    Client -> Server:
      Text:   {"type":"start"}           -> create session
      Binary: Int16 LE PCM (16kHz mono)  -> feed audio chunk
      Text:   {"type":"end"}             -> finalize
    Server -> Client:
      Text:   {"committed":"...","pending":"...","stats":{...}}
  """
  from websockets.asyncio.server import serve

  async def ws_handler(ws):
    if model is None:
      await ws.close(1008, "No ASR model loaded")
      return
    session = None
    audio_chunks: list[np.ndarray] = []
    try:
      async for message in ws:
        if isinstance(message, str):
          msg = json.loads(message)
          if msg.get('type') == 'start':
            session = StreamingSession(model)
            audio_chunks = []
            stderr_log("ws: session started\n")
            await ws.send(json.dumps({"committed": "", "pending": "", "status": "started"}))
          elif msg.get('type') == 'end':
            if session:
              result = session.feed(np.array([], dtype=np.float32), is_final=True)
              await ws.send(json.dumps({
                "committed": result["text"], "pending": "",
                "stats": result.get("stats", {}), "status": "done",
              }))
            break
        elif isinstance(message, bytes) and session and len(message) >= 2:
          audio = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
          audio_chunks.append(audio)
          result = session.feed(audio)
          await ws.send(json.dumps({
            "committed": result["committed"], "pending": result["pending"],
            "stats": result.get("stats", {}),
          }))
    except Exception as e:
      stderr_log(f"ws: error: {type(e).__name__}: {e}\n")
    finally:
      _save_session_audio(audio_chunks, save_audio_dir)
      stderr_log("ws: connection closed\n")

  async def run_forever():
    async with serve(ws_handler, "0.0.0.0", port) as server:
      await server.serve_forever()

  async def run_until_sigint():
    global _dispatch_loop
    _dispatch_loop = asyncio.get_running_loop()
    _dispatch_ready.set()  # HTTP handlers can now dispatch inference
    stop = asyncio.Event()
    import signal
    def _shutdown(*_):
      stderr_log("shutting down\n")
      os._exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    async with serve(ws_handler, "0.0.0.0", port):
      await stop.wait()

  if run_in_background:
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_until_complete, args=(run_forever(),), daemon=True)
    t.start()
    class _Shutdown:
      def shutdown(self): loop.call_soon_threadsafe(loop.stop)
    return _Shutdown()
  else:
    asyncio.run(run_until_sigint())


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="ASR + TTS + LLM server")
  parser.add_argument("--asr-model", help="ASR GGUF path or model name (e.g. qwen3-asr:0.6b)")
  parser.add_argument("--tts-model", help="TTS model directory (containing GGUF)")
  parser.add_argument("--llm-model", help="LLM GGUF path or model name (e.g. qwen3.5:0.8b)")
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8090)
  parser.add_argument("--save-audio", metavar="DIR", help="Save streaming session audio to WAV files in DIR")
  args = parser.parse_args()

  if not args.asr_model and not args.tts_model and not args.llm_model:
    parser.error("at least one of --asr-model, --tts-model, or --llm-model is required")

  # Load ASR model (optional)
  model = None
  if args.asr_model:
    if os.path.exists(args.asr_model):
      stderr_log(f"loading ASR {args.asr_model}...\n")
      raw = Tensor(pathlib.Path(args.asr_model))
    elif args.asr_model in KNOWN_MODELS:
      url = KNOWN_MODELS[args.asr_model]
      stderr_log(f"downloading ASR {args.asr_model} from {url}...\n")
      raw = Tensor.from_url(url)
    else:
      print(f"ASR model not found: {args.asr_model}")
      print(f"  Pass a path to a GGUF file, or one of: {', '.join(KNOWN_MODELS.keys())}")
      sys.exit(1)
    model = ASR.from_gguf(raw)
    del raw; gc.collect()
    model.warmup()

  # Load TTS model (optional)
  tts_model = None
  if args.tts_model:
    from tts import TTSModel, log
    log(f'[TTS] Loading from {args.tts_model}')
    tts_model = TTSModel(args.tts_model, verbose=True)
    tts_model.load()
    log('[TTS] Ready')

  # Load LLM model (optional)
  llm_model = None
  if args.llm_model:
    if os.path.exists(args.llm_model):
      stderr_log(f"loading LLM {args.llm_model}...\n")
      llm_raw = Tensor(pathlib.Path(args.llm_model))
    elif args.llm_model in KNOWN_LLM_MODELS:
      url = KNOWN_LLM_MODELS[args.llm_model]
      stderr_log(f"downloading LLM {args.llm_model} from {url}...\n")
      llm_raw = Tensor.from_url(url)
    else:
      print(f"LLM model not found: {args.llm_model}")
      print(f"  Pass a path to a GGUF file, or one of: {', '.join(KNOWN_LLM_MODELS.keys())}")
      sys.exit(1)
    llm_model, llm_kv = Transformer.from_gguf(llm_raw, max_context=4096)
    llm_tok = SimpleTokenizer.from_gguf_kv(llm_kv)
    llm_bos_id = llm_kv.get('tokenizer.ggml.bos_token_id') if llm_kv.get('tokenizer.ggml.add_bos_token', True) else None
    llm_eos_id = llm_kv['tokenizer.ggml.eos_token_id']
    del llm_raw; gc.collect()
    # LLM warmup: 2 tokens through model twice to capture JIT
    stderr_log("warming up LLM...\n")
    for _ in range(2): list(zip(range(2), llm_model.generate([0])))
    stderr_log("LLM warmup done\n")

  # Configure handler
  ServerHandler.model = model
  ServerHandler.save_audio_dir = args.save_audio
  ServerHandler._model_labels = {}
  if model is not None:
    ServerHandler._model_labels["asr"] = args.asr_model
  if tts_model is not None:
    ServerHandler.tts_model = tts_model
    ServerHandler._model_labels["tts"] = args.tts_model
  if llm_model is not None:
    ServerHandler.llm = llm_model
    ServerHandler.llm_tok = llm_tok
    ServerHandler.llm_bos_id = llm_bos_id
    ServerHandler.llm_eos_id = llm_eos_id
    ServerHandler._model_labels["llm"] = args.llm_model

  # Start HTTP in background thread
  ws_port = args.port + 1
  http_server = TCPServerWithReuse(('', args.port), ServerHandler)
  http_server.daemon_threads = True
  http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
  http_thread.start()
  if model: stderr_log(f"ASR: http://localhost:{args.port}  ws://localhost:{ws_port}\n")
  if llm_model: stderr_log(f"Chat: http://localhost:{args.port}/chat\n")
  if tts_model: stderr_log(f"TTS: POST http://localhost:{args.port}/v1/audio/speech\n")

  # Main thread: WS server (if ASR) + dispatch queue for HTTP handlers
  # tinygrad CUDA + SQLite are thread-local to main thread, so all inference dispatches here
  start_ws_server(model, ws_port, args.save_audio, run_in_background=False)
