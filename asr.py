"""
Qwen3-ASR inference via tinygrad with OpenAI-compatible API.

Loads a Qwen3-ASR GGUF model and serves /v1/audio/transcriptions.
Reuses tinygrad's Transformer (llm.py) for the decoder — the ASR decoder
is architecturally identical to Qwen3's text LM.

Usage:
  python asr.py                          # interactive mode
  python asr.py --serve                  # OpenAI API server on port 8090
  python asr.py --serve 9000             # custom port
  python asr.py --model qwen3-asr:0.6b   # specific model

Requires: tinygrad (pip install tinygrad or local -e install)
"""
from __future__ import annotations
import sys, os, argparse, json, time, math, wave, struct, uuid, functools, pathlib, tempfile, hashlib, base64
import numpy as np

# Windows CUDA workarounds (must run before tinygrad import):
# 1. Auto-detect CUDA_PATH from versioned env vars (CUDA_PATH_V13_1 etc.)
#    so nvrtc can find cuda_fp16.h
# 2. Default to PTX compilation (CUDA_PTX=1) — avoids CUDA_ERROR_UNSUPPORTED_PTX_VERSION
#    when CUDA toolkit is newer than the driver (e.g. toolkit 13.1, driver supports 13.0)
if sys.platform == "win32":
  if not os.environ.get("CUDA_PATH"):
    for key, val in os.environ.items():
      if key.startswith("CUDA_PATH_V") and os.path.isdir(val):
        os.environ["CUDA_PATH"] = val
        break
  if not os.environ.get("CUDA_PTX"):
    os.environ["CUDA_PTX"] = "1"

from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function
from tinygrad.helpers import DEBUG, GlobalCounters, colored, stderr_log
from tinygrad.apps.llm import Transformer, SimpleTokenizer, precompute_freqs_cis, apply_rope

# ============================================================================
# Constants
# ============================================================================

SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
HOP_LENGTH = 160
WINDOW_SIZE = 400  # n_fft

# Special token IDs (Qwen3-ASR)
TOKEN_IM_START    = 151644
TOKEN_IM_END      = 151645
TOKEN_AUDIO_START = 151669
TOKEN_AUDIO_END   = 151670
TOKEN_AUDIO_PAD   = 151676
TOKEN_ENDOFTEXT   = 151643
TOKEN_ASR_TEXT    = 151704

EOS_TOKEN_IDS = {TOKEN_ENDOFTEXT, TOKEN_IM_END}

# Prompt template: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|startofaudio|>
PROMPT_PREFIX = [TOKEN_IM_START, 8948, 198, TOKEN_IM_END, 198,
                 TOKEN_IM_START, 872, 198, TOKEN_AUDIO_START]
# <|endofaudio|><|im_end|>\n<|im_start|>assistant\n
PROMPT_SUFFIX = [TOKEN_AUDIO_END, TOKEN_IM_END, 198,
                 TOKEN_IM_START, 77091, 198]

# ============================================================================
# Audio preprocessing (numpy)
# ============================================================================

@functools.cache
def _mel_filters() -> np.ndarray:
  """Slaney-style mel filter bank. Returns [201, 128]."""
  def hertz_to_mel(freq):
    min_log_hertz, min_log_mel, logstep = 1000.0, 15.0, 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
      log_region = freq >= min_log_hertz
      mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
      mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels

  num_freq = 1 + WINDOW_SIZE // 2  # 201
  fft_freqs = np.linspace(0, SAMPLE_RATE // 2, num_freq)
  mel_min, mel_max = hertz_to_mel(0.0), hertz_to_mel(8000.0)
  mel_freqs = np.linspace(mel_min, mel_max, NUM_MEL_BINS + 2)
  filter_freqs = np.array([1000.0 * np.exp(np.log(6.4) / 27.0 * (m - 15.0)) if m >= 15.0
                           else 200.0 * m / 3.0 for m in mel_freqs], dtype=np.float64)
  filter_diff = np.diff(filter_freqs)
  slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
  down_slopes = -slopes[:, :-2] / filter_diff[:-1]
  up_slopes = slopes[:, 2:] / filter_diff[1:]
  fb = np.maximum(0, np.minimum(down_slopes, up_slopes))
  enorm = 2.0 / (filter_freqs[2:NUM_MEL_BINS + 2] - filter_freqs[:NUM_MEL_BINS])
  fb *= np.expand_dims(enorm, 0)
  return fb.astype(np.float32)

def compute_mel(audio: np.ndarray) -> np.ndarray:
  """Compute log-mel spectrogram. audio: float32 1D. Returns [128, frames]."""
  window = (0.5 * (1 - np.cos(2 * np.pi * np.arange(WINDOW_SIZE) / WINDOW_SIZE))).astype(np.float32)
  pad_len = WINDOW_SIZE // 2
  audio_padded = np.pad(audio, (pad_len, pad_len), mode='reflect')
  n_frames = 1 + (len(audio_padded) - WINDOW_SIZE) // HOP_LENGTH
  frames = np.lib.stride_tricks.as_strided(
    audio_padded, shape=(n_frames, WINDOW_SIZE),
    strides=(audio_padded.strides[0] * HOP_LENGTH, audio_padded.strides[0])).copy()
  spectrum = np.fft.rfft(frames * window, n=WINDOW_SIZE)
  magnitudes = np.abs(spectrum[:-1]) ** 2  # drop last frame
  mel_spec = (magnitudes @ _mel_filters()).T  # [128, n_frames]
  log_spec = np.log10(np.maximum(mel_spec, 1e-10))
  log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
  return ((log_spec + 4.0) / 4.0).astype(np.float32)

def load_audio(path: str) -> np.ndarray:
  """Load audio file as float32 mono 16kHz. Supports WAV natively, other formats via ffmpeg."""
  if path.lower().endswith('.wav'):
    with wave.open(path, 'rb') as wf:
      n_channels, sample_width, framerate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
      raw = wf.readframes(wf.getnframes())
    if sample_width == 2: samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4: samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else: raise ValueError(f"Unsupported sample width: {sample_width}")
    if n_channels > 1: samples = samples.reshape(-1, n_channels).mean(axis=1)
    if framerate != SAMPLE_RATE:
      new_len = int(len(samples) * SAMPLE_RATE / framerate)
      samples = np.interp(np.linspace(0, len(samples) - 1, new_len), np.arange(len(samples)), samples).astype(np.float32)
    return samples
  else:
    # Use ffmpeg to decode any format to raw PCM
    import subprocess
    cmd = ['ffmpeg', '-i', path, '-f', 's16le', '-ac', '1', '-ar', str(SAMPLE_RATE), '-loglevel', 'error', '-']
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0: raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0

# Keep old name as alias
load_wav = load_audio

# ============================================================================
# Audio Encoder (Qwen3-ASR specific: conv stem + transformer + projection)
# ============================================================================

@functools.cache
def _sinusoidal_pos_emb(length: int, channels: int) -> Tensor:
  """Sinusoidal positional embeddings [length, channels], cached."""
  log_ts = math.log(10000.0) / (channels // 2 - 1)
  inv_ts = Tensor.exp(-log_ts * Tensor.arange(0, channels // 2))
  positions = Tensor.arange(0, length).unsqueeze(1)
  scaled = positions * inv_ts.unsqueeze(0)
  return scaled.sin().cat(scaled.cos(), dim=-1).contiguous()

def _windowed_sdpa(q: Tensor, k: Tensor, v: Tensor, n_heads: int, head_dim: int, cu_seqlens: list[int]) -> Tensor:
  """Windowed bidirectional attention. q/k/v: [seq, d_model]. Returns [seq, d_model]."""
  if len(cu_seqlens) <= 2:
    seq = q.shape[0]
    q4 = q.reshape(1, seq, n_heads, head_dim).permute(0, 2, 1, 3)
    k4 = k.reshape(1, seq, n_heads, head_dim).permute(0, 2, 1, 3)
    v4 = v.reshape(1, seq, n_heads, head_dim).permute(0, 2, 1, 3)
    return q4.scaled_dot_product_attention(k4, v4, is_causal=False).permute(0, 2, 1, 3).reshape(seq, n_heads * head_dim)
  outputs = []
  for i in range(len(cu_seqlens) - 1):
    s, e = cu_seqlens[i], cu_seqlens[i + 1]
    wlen = e - s
    wq = q[s:e].reshape(1, wlen, n_heads, head_dim).permute(0, 2, 1, 3)
    wk = k[s:e].reshape(1, wlen, n_heads, head_dim).permute(0, 2, 1, 3)
    wv = v[s:e].reshape(1, wlen, n_heads, head_dim).permute(0, 2, 1, 3)
    outputs.append(wq.scaled_dot_product_attention(wk, wv, is_causal=False).permute(0, 2, 1, 3).reshape(wlen, n_heads * head_dim))
  return Tensor.cat(*outputs, dim=0)

class AudioEncoderBlock:
  """Encoder transformer block: LayerNorm + bidirectional attention + GELU FFN."""
  def __init__(self, d_model: int, ffn_dim: int):
    self.attn_norm = nn.LayerNorm(d_model)
    self.attn_q    = nn.Linear(d_model, d_model)
    self.attn_k    = nn.Linear(d_model, d_model)
    self.attn_v    = nn.Linear(d_model, d_model)
    self.attn_out  = nn.Linear(d_model, d_model)
    self.ffn_norm  = nn.LayerNorm(d_model)
    self.ffn_up    = nn.Linear(d_model, ffn_dim)
    self.ffn_down  = nn.Linear(ffn_dim, d_model)

  def __call__(self, x: Tensor, n_heads: int, head_dim: int, cu_seqlens: list[int]) -> Tensor:
    h = self.attn_norm(x)
    attn = _windowed_sdpa(self.attn_q(h), self.attn_k(h), self.attn_v(h), n_heads, head_dim, cu_seqlens)
    x = x + self.attn_out(attn)
    h = self.ffn_norm(x)
    return x + self.ffn_down(self.ffn_up(h).gelu())

class AudioEncoder:
  """Qwen3-ASR audio encoder: conv2d stem → transformer → linear projection."""
  def __init__(self, d_model: int, n_layers: int, n_heads: int, ffn_dim: int, output_dim: int,
               conv_channels: int = 480, n_window: int = 50, n_window_infer: int = 800):
    self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads
    self.output_dim, self.n_window, self.n_window_infer = output_dim, n_window, n_window_infer
    self.chunk_size = n_window * 2  # 100 mel frames per chunk

    # Conv stem: 3 stride-2 conv layers (8x time reduction, mel_bins/8 freq reduction)
    self.conv1 = nn.Conv2d(1, conv_channels, 3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(conv_channels, conv_channels, 3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(conv_channels, conv_channels, 3, stride=2, padding=1)
    # Project flattened conv output to d_model
    self.conv_out = nn.Linear(conv_channels * (NUM_MEL_BINS // 8), d_model, bias=False)

    # Transformer layers
    self.blk = [AudioEncoderBlock(d_model, ffn_dim) for _ in range(n_layers)]

    # Post-norm + projection to decoder dimension
    self.ln_post = nn.LayerNorm(d_model)
    self.proj1 = nn.Linear(d_model, d_model)
    self.proj2 = nn.Linear(d_model, output_dim)

  def _conv_stem(self, mel: np.ndarray) -> tuple[Tensor, int]:
    """Conv stem + positional embeddings. Returns (tensor, seq_len). Python-loop fallback."""
    total_frames = mel.shape[1]
    chunk_outputs = []
    for start in range(0, total_frames, self.chunk_size):
      end = min(start + self.chunk_size, total_frames)
      chunk_mel = Tensor(mel[:, start:end]).reshape(1, 1, NUM_MEL_BINS, end - start)
      x = self.conv1(chunk_mel).gelu()
      x = self.conv2(x).gelu()
      x = self.conv3(x).gelu()
      b, c, f, t = x.shape
      chunk_outputs.append(x.permute(0, 3, 1, 2).reshape(1, t, c * f).squeeze(0))  # [time, conv_ch * freq]

    x = self.conv_out(Tensor.cat(*chunk_outputs, dim=0))  # [total_tokens, d_model]
    seq_len = x.shape[0]

    # Per-chunk sinusoidal position embeddings
    tokens_per_chunk = chunk_outputs[0].shape[0]
    pos_emb = _sinusoidal_pos_emb(tokens_per_chunk, self.d_model)
    chunks_with_pos = []
    offset = 0
    for co in chunk_outputs:
      cl = co.shape[0]
      chunks_with_pos.append(x[offset:offset + cl] + pos_emb[:cl])
      offset += cl
    return Tensor.cat(*chunks_with_pos, dim=0).realize(), seq_len

  def _encode_batched(self, mel_tensor: Tensor) -> Tensor:
    """Full encoder as single tensor graph (JIT-friendly, no Python loops).
    mel_tensor: [128, padded_frames] where padded_frames is multiple of chunk_size.
    Handles multi-window attention by treating windows as batch dimension.
    Returns [seq_len, output_dim]."""
    n_chunks = mel_tensor.shape[1] // self.chunk_size
    tokens_per_chunk = 13  # 100 mel frames through 3x stride-2 conv → 13 time steps
    tokens_per_window = tokens_per_chunk * (self.n_window_infer // self.chunk_size)  # 104
    n_windows = (n_chunks * tokens_per_chunk + tokens_per_window - 1) // tokens_per_window

    # Batch conv stem: reshape [128, N*100] → [N, 1, 128, 100], process all chunks at once
    mel_4d = mel_tensor.reshape(NUM_MEL_BINS, n_chunks, self.chunk_size).permute(1, 0, 2)
    mel_4d = mel_4d.reshape(n_chunks, 1, NUM_MEL_BINS, self.chunk_size)
    x = self.conv3(self.conv2(self.conv1(mel_4d).gelu()).gelu()).gelu()  # [N, 480, 16, 13]
    x = x.permute(0, 3, 1, 2).reshape(n_chunks * tokens_per_chunk, -1)  # [N*13, 480*16]
    x = self.conv_out(x)  # [N*13, d_model]

    # Sinusoidal position embeddings (same per chunk, broadcast add)
    pos = _sinusoidal_pos_emb(tokens_per_chunk, self.d_model)
    x = (x.reshape(n_chunks, tokens_per_chunk, self.d_model) + pos.unsqueeze(0)).reshape(-1, self.d_model)

    # Transformer with windowed attention: reshape to [n_windows, tokens_per_window, d_model]
    # All windows are the same size, so we use batch dimension for parallel attention
    seq = tokens_per_window  # 104 tokens per window
    x = x.reshape(n_windows, seq, self.d_model)
    for block in self.blk:
      h = block.attn_norm(x)
      q = block.attn_q(h).reshape(n_windows, seq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
      k = block.attn_k(h).reshape(n_windows, seq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
      v = block.attn_v(h).reshape(n_windows, seq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
      attn = q.scaled_dot_product_attention(k, v, is_causal=False).permute(0, 2, 1, 3).reshape(n_windows, seq, self.n_heads * self.head_dim)
      x = x + block.attn_out(attn)
      h = block.ffn_norm(x)
      x = x + block.ffn_down(block.ffn_up(h).gelu())

    x = x.reshape(n_windows * seq, self.d_model)
    x = self.ln_post(x)
    return self.proj2(self.proj1(x).gelu())

  def _encode_window(self, mel_tensor: Tensor) -> Tensor:
    """Encode a single 800-frame window. JIT-friendly (fixed shape).
    mel_tensor: [128, 800]. Returns [104, output_dim]."""
    return self._encode_batched(mel_tensor)

  def forward(self, mel: np.ndarray) -> Tensor:
    """Encode mel spectrogram [128, frames] → [n_tokens, output_dim].

    Two paths:
    - Batched JIT: for pre-warmed bucket sizes (800, 1600, etc.), processes
      all windows in a single JIT call with batched attention. Fast.
    - Sequential JIT: for any other size, splits into 800-frame windows and
      encodes each through the single-window JIT. No compilation surprises,
      scales to any audio length.

    Both produce identical output (attention is windowed, no cross-window deps).
    """
    actual_frames = mel.shape[1]
    bucket_frames = self.chunk_size * 8  # 800 frames per window
    padded_frames = ((actual_frames + bucket_frames - 1) // bucket_frames) * bucket_frames
    if padded_frames > actual_frames:
      mel = np.pad(mel, ((0, 0), (0, padded_frames - actual_frames)), mode='constant')

    tokens_per_chunk = 13  # 100 frames through 3x stride-2

    # Compute actual token count (before padding)
    actual_chunks = actual_frames // self.chunk_size
    actual_tokens = actual_chunks * tokens_per_chunk
    tail = actual_frames % self.chunk_size
    if tail > 0: actual_tokens += max(1, tail // 8)

    # (+0).realize() forces a fresh buffer copy — JIT reuses output buffers,
    # so callers caching encoder outputs (streaming) need independent copies

    # Path 1: Batched JIT for pre-warmed bucket sizes (fast, single JIT call)
    if hasattr(self, '_encode_jits') and padded_frames in self._encode_jits:
      mel_tensor = Tensor(mel).contiguous()
      out = self._encode_jits[padded_frames](mel_tensor)
      return (out[:actual_tokens] + 0).realize()

    # Path 2: Sequential single-window JIT (any length, no compilation surprise)
    if hasattr(self, '_encode_window_jit'):
      n_windows = padded_frames // bucket_frames
      window_outputs = []
      for i in range(n_windows):
        window_mel = Tensor(mel[:, i * bucket_frames:(i + 1) * bucket_frames]).contiguous()
        out = self._encode_window_jit(window_mel)
        window_outputs.append((out + 0).realize())
      full = Tensor.cat(*window_outputs, dim=0) if len(window_outputs) > 1 else window_outputs[0]
      return full[:actual_tokens]

    # Non-JIT fallback
    mel_tensor = Tensor(mel).contiguous()
    out = self._encode_batched(mel_tensor)
    return out[:actual_tokens]

# ============================================================================
# ASR Model: AudioEncoder + Transformer decoder from llm.py
# ============================================================================

class ASR:
  """Qwen3-ASR: audio encoder + Qwen3 text decoder."""

  def __init__(self, encoder: AudioEncoder, decoder: Transformer, tok: SimpleTokenizer):
    self.encoder, self.decoder, self.tok = encoder, decoder, tok
    # Shared symbolic variables for JIT stability (same names everywhere)
    # Full range allows single JIT call for any prompt length — no chunking needed
    self.v_sp = UOp.variable("asr_sp", 0, decoder.max_context - 1)
    self.v_nt = UOp.variable("asr_nt", 1, decoder.max_context)
    self.v_dec_pos = UOp.variable("asr_dec_pos", 1, decoder.max_context - 1)

  @staticmethod
  def from_gguf(gguf_tensor: Tensor) -> ASR:
    """Load ASR model from a GGUF tensor (FlippyDora F16 format)."""
    kv, state_dict = nn.state.gguf_load(gguf_tensor.to(None).realize())
    arch = kv['general.architecture']

    # Cast to float16 (matches llm.py convention)
    if getenv("HALF", 1): state_dict = {k: v.cast('float16') for k, v in state_dict.items()}

    # --- Encoder config ---
    enc_d = kv.get(f'{arch}.audio.encoder.embedding_length', 896)
    enc_layers = kv.get(f'{arch}.audio.encoder.layer_count', 18)
    enc_heads = kv.get(f'{arch}.audio.encoder.attention.head_count', 14)
    enc_ffn = kv.get(f'{arch}.audio.encoder.feed_forward_length', 3584)
    conv_ch = kv.get(f'{arch}.audio.conv_channels', 480)

    # --- Decoder config ---
    dec_dim = kv.get(f'{arch}.embedding_length', 1024)
    n_heads = kv.get(f'{arch}.attention.head_count', 16)
    n_kv_heads = kv.get(f'{arch}.attention.head_count_kv', 8)
    head_dim = kv.get(f'{arch}.attention.key_length', dec_dim // n_heads)
    num_blocks = kv.get(f'{arch}.block_count', 28)
    hidden_dim = kv.get(f'{arch}.feed_forward_length', 3072)
    norm_eps = kv.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-6)
    rope_theta = kv.get(f'{arch}.rope.freq_base', 1000000.0)
    vocab_size = kv.get(f'{arch}.vocab_size', 151936)

    # --- Build encoder ---
    encoder = AudioEncoder(enc_d, enc_layers, enc_heads, enc_ffn, dec_dim, conv_ch)

    # Split state dict: encoder gets audio.encoder.*, decoder gets the rest
    enc_sd = {k[len('audio.encoder.'):]: v for k, v in state_dict.items() if k.startswith('audio.encoder.')}
    dec_sd = {k: v for k, v in state_dict.items() if not k.startswith('audio.encoder.')}

    # Tie output weight if missing
    if 'output.weight' not in dec_sd: dec_sd['output.weight'] = dec_sd['token_embd.weight']

    nn.state.load_state_dict(encoder, enc_sd, strict=False, verbose=False, consume=True)

    # --- Build decoder (reuse llm.py Transformer) ---
    max_context = 2048
    decoder = Transformer(
      num_blocks=num_blocks, dim=dec_dim, hidden_dim=hidden_dim,
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=norm_eps,
      vocab_size=vocab_size, head_dim=head_dim, rope_theta=rope_theta,
      max_context=max_context,
      qk_norm=int(dec_sd['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in dec_sd else 0)
    nn.state.load_state_dict(decoder, dec_sd, strict=False, verbose=False, consume=True)

    # Realize all parameters
    params = nn.state.get_parameters(encoder) + nn.state.get_parameters(decoder)
    for p in params: p.replace(p.contiguous())
    Tensor.realize(*params)

    # Add JIT'd forward_embed for ASR prefill (bypasses tok_embeddings)
    # Two JITs like Transformer: one for chunked prefill (variable length), one for rollout
    def forward_embed(h: Tensor, start_pos: int|UOp) -> Tensor:
      for block in decoder.blk: h = block(h, start_pos)
      return decoder.output(decoder.output_norm(h))[:, -1, :].softmax(-1, dtype="float").argmax(-1, keepdim=True)
    decoder.forward_embed = forward_embed
    decoder.prefill_embed_jit = TinyJit(forward_embed)

    # Pre-allocate fixed embedding buffer for prefill (JIT requires stable shape)
    embed_buf = Tensor.zeros(1, max_context, dec_dim).contiguous().realize()

    # --- Tokenizer ---
    tok = SimpleTokenizer.from_gguf_kv(kv)

    model = ASR(encoder, decoder, tok)
    model._embed_buf = embed_buf

    # Pre-compute prompt token embeddings (fixed, reused every call)
    model._prefix_embeds = decoder.token_embd(Tensor(PROMPT_PREFIX)).realize()
    model._suffix_embeds = decoder.token_embd(Tensor(PROMPT_SUFFIX)).realize()

    # Encoder JIT setup:
    # - _encode_window_jit: single 800-frame window, universal fallback for any length
    # - _encode_jits: batched multi-window for pre-warmed bucket sizes (faster)
    encoder._encode_window_jit = TinyJit(encoder._encode_window)
    encoder._encode_jits = {}  # populated during warmup

    return model

  # Pre-warmed encoder bucket sizes: covers audio up to 32s in per-file mode.
  # Longer audio uses the sequential single-window path (no compilation needed).
  ENCODER_BUCKETS = [800, 1600, 2400, 3200]

  def warmup(self):
    """Exercise all JIT paths so subsequent calls are fast.

    JIT setup:
    - Encoder single-window JIT (800 frames): always available, any audio length
    - Encoder batched JITs (800-3200): pre-warmed for fast per-file mode (<=32s)
    - Decoder prefill JIT: symbolic variables, 3 sizes for @function compilation
    - Decoder single-token JIT: one call

    JITBEAM kernel optimizations are cached to disk (~/.cache/tinygrad/cache.db).
    First-ever run is slow (beam search). Subsequent runs hit cache.
    """
    stderr_log("warming up JIT...\n")
    dim = self._embed_buf.shape[2]
    dummy = np.zeros((128, 800), dtype=np.float32)

    # 1. Single-window encoder JIT (universal fallback, always needed)
    for _ in range(3):
      self.encoder._encode_window_jit(Tensor(dummy).contiguous()).realize()

    # 2. Batched multi-window JITs for pre-warmed bucket sizes
    for bucket in self.ENCODER_BUCKETS:
      self.encoder._encode_jits[bucket] = TinyJit(self.encoder._encode_batched)
      for _ in range(3):
        self.encoder.forward(np.zeros((128, bucket), dtype=np.float32)).realize()

    # 3. Prefill JIT: 3 different sizes to exercise @function compilation
    for nt in [200, 100, 150]:
      self._embed_buf[:, :nt].assign(Tensor.randn(1, nt, dim).contiguous()).realize()
      sp_b, nt_b = self.v_sp.bind(0), self.v_nt.bind(nt)
      self.decoder.prefill_embed_jit(self._embed_buf[:, sp_b:sp_b+nt_b], sp_b).realize()

    # 4. Decode JIT: single-token path
    self.decoder(Tensor([[0]]), self.v_dec_pos.bind(200))
    stderr_log("warmup done\n")

  def transcribe(self, audio_path: str) -> dict:
    """Transcribe a WAV file. Returns {"text": str, "elapsed_ms": float}."""
    t0 = time.time()

    # 1. Audio → mel
    audio = load_wav(audio_path)
    stderr_log(f"audio: {len(audio)/SAMPLE_RATE:.1f}s  {colored('--', 'BLACK')}  ")
    mel = compute_mel(audio)

    # 2. Encode
    t_enc = time.time()
    audio_embeds = self.encoder.forward(mel).realize()  # [n_tokens, decoder_dim] — realize to avoid deferred compute
    n_audio = audio_embeds.shape[0]
    stderr_log(f"enc: {n_audio} tokens in {(time.time()-t_enc)*1000:.0f}ms  {colored('--', 'BLACK')}  ")

    # 3. Build prompt embeddings
    #    prefix_tokens + audio_embeddings + suffix_tokens
    combined = Tensor.cat(self._prefix_embeds, audio_embeds, self._suffix_embeds, dim=0).reshape(1, -1, audio_embeds.shape[1])
    prompt_len = combined.shape[1]

    # 4. Prefill: single JIT call with full-range symbolic variable (handles any prompt length)
    t_prefill = time.time()
    assert prompt_len <= self._embed_buf.shape[1], f"prompt_len {prompt_len} exceeds max_context {self._embed_buf.shape[1]}"
    self._embed_buf[:, :prompt_len].assign(combined.contiguous()).realize()
    sp_b, nt_b = self.v_sp.bind(0), self.v_nt.bind(prompt_len)
    out = self.decoder.prefill_embed_jit(self._embed_buf[:, sp_b:sp_b+nt_b], sp_b).realize()
    token = int(out.item())
    stderr_log(f"prefill: {prompt_len} in {(time.time()-t_prefill)*1000:.0f}ms  {colored('--', 'BLACK')}  ")

    # 5. Autoregressive decode using Transformer's standard forward path
    generated = [token]
    t_dec = time.time()

    for step in range(1023):
      if token in EOS_TOKEN_IDS: break
      pos = prompt_len + step
      out = self.decoder(Tensor([[token]]), self.v_dec_pos.bind(pos))
      token = int(out.item())
      generated.append(token)

    n_dec = len(generated) - 1  # exclude first token from prefill
    dec_ms = (time.time() - t_dec) * 1000
    stderr_log(f"decode: {n_dec} tokens in {dec_ms:.0f}ms ({n_dec/(dec_ms/1000):.1f} tok/s)\n")

    # 6. Decode tokens to text
    # Strip EOS
    while generated and generated[-1] in EOS_TOKEN_IDS: generated.pop()
    # Strip prefix before <asr_text> marker
    if TOKEN_ASR_TEXT in generated:
      generated = generated[generated.index(TOKEN_ASR_TEXT) + 1:]
    text = self.tok.decode(generated).strip()

    return {"text": text, "elapsed_ms": (time.time() - t0) * 1000}

  def transcribe_stream(self, audio: np.ndarray, chunk_sec: float = 2.0,
                         max_new_tokens: int = 64, max_enc_windows: int = 4,
                         callback = None) -> dict:
    """Streaming transcription: process audio in fixed-size chunks.

    Each chunk encodes all audio heard so far (caching completed encoder windows)
    and produces a COMPLETE transcription. The last chunk's output is the final result.
    Intermediate results are delivered via callback for incremental display.

    Args:
      audio: float32 mono 16kHz samples
      chunk_sec: seconds of audio per processing step (default 2.0)
      max_new_tokens: max tokens decoded per chunk
      max_enc_windows: sliding window limit for encoder cache
      callback: optional fn(text_so_far, is_final) called after each chunk

    Returns: {"text": str, "elapsed_ms": float, "audio_sec": float, "rtf": float}
    """
    t0 = time.time()
    audio_sec = len(audio) / SAMPLE_RATE
    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    dim = self.encoder.output_dim
    enc_window_frames = 800  # 8s of audio per encoder window
    enc_window_samples = enc_window_frames * HOP_LENGTH

    # Encoder window cache: list of Tensor (realized encoder outputs for complete windows)
    enc_cache: list[Tensor] = []
    next_window_start = 0  # sample index of next full window boundary

    # KV cache reuse: save previous chunk's prompt embeddings for comparison
    prev_embeds: np.ndarray | None = None  # [prev_prompt_len, dim] numpy
    prev_prompt_len = 0

    audio_cursor = 0
    chunk_idx = 0
    last_text = ""
    total_enc_ms, total_prefill_ms, total_decode_ms = 0.0, 0.0, 0.0
    total_reused_tokens = 0

    while audio_cursor < len(audio):
      audio_cursor = min(audio_cursor + chunk_samples, len(audio))
      is_final = audio_cursor >= len(audio)

      # --- Encoder: cache completed windows, re-encode partial tail ---
      t_enc = time.time()
      full_end = (audio_cursor // enc_window_samples) * enc_window_samples

      # Encode any new complete windows
      while next_window_start < full_end:
        ws = next_window_start
        window_mel = compute_mel(audio[ws:ws + enc_window_samples])
        bucket = self.encoder.chunk_size * 8
        padded = ((window_mel.shape[1] + bucket - 1) // bucket) * bucket
        if padded > window_mel.shape[1]:
          window_mel = np.pad(window_mel, ((0, 0), (0, padded - window_mel.shape[1])))
        enc_out = self.encoder.forward(window_mel[:, :padded]).realize()
        actual_tokens = enc_window_frames // 8
        enc_cache.append(enc_out[:actual_tokens])
        next_window_start += enc_window_samples

      # Encode partial tail (from last full window boundary to audio_cursor)
      partial_enc = None
      if full_end < audio_cursor:
        partial_mel = compute_mel(audio[int(full_end):audio_cursor])
        bucket = self.encoder.chunk_size * 8
        padded = ((partial_mel.shape[1] + bucket - 1) // bucket) * bucket
        if padded > partial_mel.shape[1]:
          partial_mel = np.pad(partial_mel, ((0, 0), (0, padded - partial_mel.shape[1])))
        partial_out = self.encoder.forward(partial_mel[:, :padded]).realize()
        actual_partial = max(1, (audio_cursor - int(full_end)) // HOP_LENGTH // 8)
        partial_enc = partial_out[:actual_partial]

      # Evict old encoder windows (sliding window for long audio)
      while len(enc_cache) > max_enc_windows:
        enc_cache.pop(0)

      # Concatenate encoder outputs: cached windows + partial tail
      enc_parts = list(enc_cache) + ([partial_enc] if partial_enc is not None else [])
      if not enc_parts: chunk_idx += 1; continue
      audio_embeds = Tensor.cat(*enc_parts, dim=0) if len(enc_parts) > 1 else enc_parts[0]
      enc_ms = (time.time() - t_enc) * 1000
      total_enc_ms += enc_ms

      # --- Prefill with KV cache reuse ---
      # Compare current embeddings with previous chunk to find reuse point.
      # Positions 0..reuse_point have identical KV cache entries — skip them.
      t_pf = time.time()
      combined = Tensor.cat(self._prefix_embeds, audio_embeds, self._suffix_embeds, dim=0).reshape(1, -1, dim)
      prompt_len = combined.shape[1]
      assert prompt_len <= self._embed_buf.shape[1], f"prompt_len {prompt_len} > max_context"

      # Write full embeddings to buffer (needed for both reuse comparison and prefill)
      self._embed_buf[:, :prompt_len].assign(combined.contiguous()).realize()
      cur_embeds = self._embed_buf[0, :prompt_len].numpy()  # [prompt_len, dim]

      # Find longest matching prefix with previous chunk
      reuse_point = 0
      if prev_embeds is not None:
        cmp_len = min(prompt_len, prev_prompt_len)
        row_bytes = dim * 4  # float32
        for i in range(cmp_len):
          if not np.array_equal(cur_embeds[i], prev_embeds[i]): break
          reuse_point = i + 1

      # Prefill only the delta tokens (from reuse_point to prompt_len)
      delta = prompt_len - reuse_point
      if delta > 0:
        sp_b, nt_b = self.v_sp.bind(reuse_point), self.v_nt.bind(delta)
        out = self.decoder.prefill_embed_jit(self._embed_buf[:, sp_b:sp_b+nt_b], sp_b).realize()
        token = int(out.item())
      else:
        # Entire prompt is unchanged — just re-decode from the last prefill output
        # (This shouldn't happen in practice since at least the partial tail changes)
        sp_b, nt_b = self.v_sp.bind(0), self.v_nt.bind(prompt_len)
        out = self.decoder.prefill_embed_jit(self._embed_buf[:, sp_b:sp_b+nt_b], sp_b).realize()
        token = int(out.item())

      # Save embeddings for next chunk's comparison
      prev_embeds = cur_embeds.copy()
      prev_prompt_len = prompt_len
      total_reused_tokens += reuse_point

      prefill_ms = (time.time() - t_pf) * 1000
      total_prefill_ms += prefill_ms

      # --- Decode until EOS or max_new_tokens ---
      t_dec = time.time()
      chunk_tokens = [token]
      for step in range(max_new_tokens - 1):
        if token in EOS_TOKEN_IDS: break
        pos = prompt_len + step
        out = self.decoder(Tensor([[token]]), self.v_dec_pos.bind(pos))
        token = int(out.item())
        chunk_tokens.append(token)
      decode_ms = (time.time() - t_dec) * 1000
      total_decode_ms += decode_ms

      # Extract text from this chunk's full transcription
      while chunk_tokens and chunk_tokens[-1] in EOS_TOKEN_IDS: chunk_tokens.pop()
      if TOKEN_ASR_TEXT in chunk_tokens:
        chunk_tokens = chunk_tokens[chunk_tokens.index(TOKEN_ASR_TEXT) + 1:]
      last_text = self.tok.decode(chunk_tokens).strip()
      chunk_idx += 1

      if callback: callback(last_text, is_final)

    total_ms = (time.time() - t0) * 1000
    rtf = (total_ms / 1000) / audio_sec if audio_sec > 0 else 0
    stderr_log(f"stream: {audio_sec:.1f}s audio, {chunk_idx} chunks, "
               f"enc={total_enc_ms:.0f}ms, prefill={total_prefill_ms:.0f}ms, "
               f"decode={total_decode_ms:.0f}ms, total={total_ms:.0f}ms, RTF={rtf:.2f}, "
               f"kv_reused={total_reused_tokens}\n")

    return {"text": last_text, "elapsed_ms": total_ms, "audio_sec": audio_sec, "rtf": rtf}

# ============================================================================
# StreamingSession: server-side state for incremental live transcription
#
# Mirrors the C implementation's streaming architecture:
# - Client sends audio deltas (new PCM samples)
# - Server accumulates, caches complete encoder windows (bounded, sliding)
# - Re-encodes only the partial tail each chunk
# - Decoder KV cache reused across chunks via embedding comparison
# - Bounded: max 4 encoder windows (32s audio context)
# ============================================================================

class StreamingSession:
  """Server-side streaming session matching the C implementation's architecture.

  Key design (from qwen_asr.c stream_impl):
  - raw_tokens: full decoded history = prefix + continuation (unbounded, just IDs)
  - Text prefix feedback: embed raw_tokens[:-rollback] and append after audio+suffix
  - Candidate commit: text tokens minus rollback tail, LCP against previous stable
  - emitted_text grows monotonically — old text never removed
  - Encoder: sliding window of max 4 cached windows (32s audio context)
  """
  def __init__(self, model: 'ASR', chunk_sec: float = 2.0, max_enc_windows: int = 4, max_new_tokens: int = 64):
    self.model = model
    self.chunk_samples = int(chunk_sec * SAMPLE_RATE)
    self.max_enc_windows = max_enc_windows
    self.max_new_tokens = max_new_tokens
    self.enc_window_frames = 800  # 8s of audio per encoder window
    self.enc_window_samples = self.enc_window_frames * HOP_LENGTH

    # Audio: only the partial tail since last complete window boundary
    self.tail_audio = np.array([], dtype=np.float32)
    self._window_buf = np.array([], dtype=np.float32)
    self.total_samples = 0

    # Encoder window cache (bounded by max_enc_windows)
    self.enc_cache: list[Tensor] = []

    # Decoder KV reuse state
    self.prev_embeds: np.ndarray | None = None
    self.prev_prompt_len = 0

    # Text prefix feedback (C-style streaming)
    self.raw_tokens: list[int] = []       # full decoded tokens (lang + text_marker + text)
    self.stable_text_tokens: list[int] = []  # candidate tokens from last commit
    self.emitted_text_tokens: list[int] = [] # all emitted token IDs (for overlap dedup)
    self.emitted_text: str = ""           # accumulated committed text (grows monotonically)
    self.rollback = 5                     # unfixed tail tokens
    self.unfixed_chunks = 2               # first N chunks: no prefix feedback
    self.max_prefix_tokens = 150          # bound prefix length in prompt

    # Constants matching C implementation
    self.MAX_REPEAT_TOKEN_RUN = 12   # suppress runs longer than this
    self.OVERLAP_MAX_TOKENS = 48     # max overlap check window
    self.OVERLAP_MIN_TOKENS = 4      # min overlap to trigger dedup
    self.DEGEN_MAX_PERIOD = 6        # max repeat block period to check
    self.DEGEN_MIN_REPEATS = 4       # min repeats to trigger recovery
    self.STALE_CHUNKS = 4            # stagnant chunks before recovery
    self.RESET_INTERVAL_CHUNKS = 45  # periodic reset every N chunks
    self.RESET_CARRY_TOKENS = 24     # tokens to carry through reset

    # Stagnation tracking
    self.stagnant_chunks = 0
    self.hit_max_new = False  # did last decode hit max_new_tokens?

    # Stats
    self.chunk_idx = 0
    self.total_reused = 0
    self.last_stats: dict = {}
    self._is_final = False

  # --- Recovery helpers (matching C implementation) ---

  @staticmethod
  def _tail_repeat_blocks(tokens: list[int], max_period: int) -> tuple[int, int]:
    """Detect repeating block pattern at tail of token list.
    Returns (best_reps, best_period). E.g. [A,B,A,B,A,B] → (3, 2)."""
    n = len(tokens)
    if n < 2: return 1, 0
    best_reps, best_period = 1, 0
    period_cap = min(n // 2, max_period) if max_period > 0 else n // 2
    for p in range(1, period_cap + 1):
      reps = 1
      while (reps + 1) * p <= n:
        a = tokens[n - (reps + 1) * p : n - reps * p]
        b = tokens[n - reps * p : n]
        if len(a) != p or a != b[:p]: break
        reps += 1
      if reps > best_reps:
        best_reps, best_period = reps, p
    return best_reps, best_period

  def _suppress_repeats(self, chunk_tokens: list[int]) -> tuple[list[int], int]:
    """Filter out tokens repeating > MAX_REPEAT_TOKEN_RUN times, continuing
    the run count from the end of raw_tokens[:n_prefix_full]."""
    if not chunk_tokens: return chunk_tokens, 0
    # Seed run from end of prefix
    n_prefix_full = max(0, len(self.raw_tokens) - self.rollback) if self.chunk_idx >= self.unfixed_chunks else 0
    prev_tok, prev_run = -1, 0
    if n_prefix_full > 0:
      prev_tok = self.raw_tokens[n_prefix_full - 1]
      prev_run = 1
      for j in range(n_prefix_full - 2, -1, -1):
        if self.raw_tokens[j] != prev_tok: break
        prev_run += 1
        if prev_run >= self.MAX_REPEAT_TOKEN_RUN: break

    out, dropped = [], 0
    for tok in chunk_tokens:
      if tok == prev_tok:
        prev_run += 1
        if prev_run > self.MAX_REPEAT_TOKEN_RUN:
          dropped += 1; continue
      else:
        prev_tok, prev_run = tok, 1
      out.append(tok)
    return out, dropped

  def _reanchor(self):
    """Recovery reset: rebuild raw_tokens from emitted history, reset KV cache.

    Unlike the C implementation which clears the encoder cache (it retains
    the full audio buffer and re-encodes), we preserve encoder windows because
    audio arrives incrementally via feed() — clearing would lose all context.
    Only the decoder text state and KV cache are reset."""
    carry = min(len(self.emitted_text_tokens), self.RESET_CARRY_TOKENS)
    tail = self.emitted_text_tokens[-carry:] if carry > 0 else []

    # Rebuild raw_tokens: [<asr_text>] + last N emitted text tokens
    self.raw_tokens = [TOKEN_ASR_TEXT] + list(tail)
    self.stable_text_tokens = list(tail)

    # Reset KV cache (forces full re-prefill next chunk)
    # Preserve encoder cache and _window_buf — audio context is valuable
    self.prev_embeds = None
    self.prev_prompt_len = 0

    self.stagnant_chunks = 0

  def feed(self, new_audio: np.ndarray, is_final: bool = False) -> dict:
    """Feed new audio samples. Returns {"text": ..., "stats": ...}."""
    self._is_final = is_final
    if len(new_audio) > 0:
      self.tail_audio = np.append(self.tail_audio, new_audio)
      self.total_samples += len(new_audio)

    # Process complete chunks (not final yet)
    self._is_final = False
    while len(self.tail_audio) >= self.chunk_samples:
      remaining_after = len(self.tail_audio) - self.chunk_samples
      self._is_final = is_final and remaining_after < self.chunk_samples
      self._process_chunk()

    # On final: process any remaining audio
    if is_final and len(self.tail_audio) > 0:
      self._is_final = True
      self._process_chunk()

    # Build display: committed text + pending (unfixed rollback tail)
    committed = self.emitted_text
    pending_text = ""
    text_start = 0
    for i, t in enumerate(self.raw_tokens):
      if t == TOKEN_ASR_TEXT: text_start = i + 1; break
    text_tokens = self.raw_tokens[text_start:]
    if len(text_tokens) > len(self.stable_text_tokens):
      pending_toks = text_tokens[len(self.stable_text_tokens):]
      pending_text = self.model.tok.decode(pending_toks)

    return {
      "text": (committed + pending_text).strip(),
      "committed": committed.strip(),
      "pending": pending_text,
      "stats": self.last_stats,
    }

  def _process_chunk(self):
    """Process one chunk: encoder + text prefix + prefill + decode + commit."""
    model = self.model
    dim = model.encoder.output_dim
    t0 = time.time()

    # --- Encoder: cache complete windows, re-encode partial tail ---
    consume = min(self.chunk_samples, len(self.tail_audio))
    self._window_buf = np.append(self._window_buf, self.tail_audio[:consume])
    self.tail_audio = self.tail_audio[consume:]

    while len(self._window_buf) >= self.enc_window_samples:
      window_audio = self._window_buf[:self.enc_window_samples]
      self._window_buf = self._window_buf[self.enc_window_samples:]
      mel = compute_mel(window_audio)
      bucket = model.encoder.chunk_size * 8
      padded = ((mel.shape[1] + bucket - 1) // bucket) * bucket
      if padded > mel.shape[1]:
        mel = np.pad(mel, ((0, 0), (0, padded - mel.shape[1])))
      enc_out = model.encoder.forward(mel[:, :padded])
      actual_tokens = self.enc_window_frames // 8
      self.enc_cache.append((enc_out[:actual_tokens] + 0).realize())

    while len(self.enc_cache) > self.max_enc_windows:
      self.enc_cache.pop(0)

    partial_enc = None
    if len(self._window_buf) > 0:
      mel = compute_mel(self._window_buf)
      bucket = model.encoder.chunk_size * 8
      padded = ((mel.shape[1] + bucket - 1) // bucket) * bucket
      if padded > mel.shape[1]:
        mel = np.pad(mel, ((0, 0), (0, padded - mel.shape[1])))
      partial_out = model.encoder.forward(mel[:, :padded])
      actual_partial = max(1, len(self._window_buf) // HOP_LENGTH // 8)
      partial_enc = (partial_out[:actual_partial] + 0).realize()

    enc_parts = list(self.enc_cache) + ([partial_enc] if partial_enc is not None else [])
    if not enc_parts: return
    audio_embeds = Tensor.cat(*enc_parts, dim=0) if len(enc_parts) > 1 else enc_parts[0]
    enc_ms = (time.time() - t0) * 1000

    # --- Text prefix feedback ---
    # After initial chunks, feed previous decoded tokens (minus rollback) as context.
    # This anchors the decoder so old text survives encoder window eviction.
    n_prefix_full = 0  # uncapped prefix length (for raw_tokens tracking)
    prefix_toks = []    # capped prefix (for prompt embedding)
    if self.chunk_idx >= self.unfixed_chunks and len(self.raw_tokens) > 0:
      n_prefix_full = max(0, len(self.raw_tokens) - self.rollback)
      prefix_toks = self.raw_tokens[:n_prefix_full]
      if len(prefix_toks) > self.max_prefix_tokens:
        prefix_toks = prefix_toks[-self.max_prefix_tokens:]

    # --- Build prompt: [system prefix] [encoder output] [suffix] [text prefix tokens] ---
    t_pf = time.time()
    parts = [model._prefix_embeds, audio_embeds, model._suffix_embeds]
    if prefix_toks:
      tok_ids = Tensor([prefix_toks])
      tok_embeds = model.decoder.token_embd(tok_ids).squeeze(0).realize()
      parts.append(tok_embeds)
    combined = Tensor.cat(*parts, dim=0).reshape(1, -1, dim)
    prompt_len = combined.shape[1]
    assert prompt_len <= model._embed_buf.shape[1], f"prompt_len {prompt_len} > max_context"

    model._embed_buf[:, :prompt_len].assign(combined.contiguous()).realize()
    cur_embeds = model._embed_buf[0, :prompt_len].numpy()

    # KV cache reuse: find longest matching prefix with previous chunk
    reuse_point = 0
    if self.prev_embeds is not None:
      cmp_len = min(prompt_len, self.prev_prompt_len)
      for i in range(cmp_len):
        if not np.array_equal(cur_embeds[i], self.prev_embeds[i]): break
        reuse_point = i + 1

    delta = prompt_len - reuse_point
    if delta > 0:
      sp_b, nt_b = model.v_sp.bind(reuse_point), model.v_nt.bind(delta)
      out = model.decoder.prefill_embed_jit(model._embed_buf[:, sp_b:sp_b+nt_b], sp_b).realize()
      token = int(out.item())
    else:
      sp_b, nt_b = model.v_sp.bind(0), model.v_nt.bind(prompt_len)
      out = model.decoder.prefill_embed_jit(model._embed_buf[:, sp_b:sp_b+nt_b], sp_b).realize()
      token = int(out.item())

    self.prev_embeds = cur_embeds.copy()
    self.prev_prompt_len = prompt_len
    self.total_reused += reuse_point
    prefill_ms = (time.time() - t_pf) * 1000

    # --- Decode (continuation after prefix) ---
    t_dec = time.time()
    chunk_tokens = [token]
    self.hit_max_new = False
    for step in range(self.max_new_tokens - 1):
      if token in EOS_TOKEN_IDS: break
      pos = prompt_len + step
      out = model.decoder(Tensor([[token]]), model.v_dec_pos.bind(pos))
      token = int(out.item())
      chunk_tokens.append(token)
    else:
      if token not in EOS_TOKEN_IDS: self.hit_max_new = True
    while chunk_tokens and chunk_tokens[-1] in EOS_TOKEN_IDS: chunk_tokens.pop()
    decode_ms = (time.time() - t_dec) * 1000

    # --- Repeat suppression: filter runs > MAX_REPEAT_TOKEN_RUN ---
    chunk_tokens, dropped_repeats = self._suppress_repeats(chunk_tokens)

    # --- Update raw_tokens: prefix (uncapped) + newly decoded continuation ---
    self.raw_tokens = self.raw_tokens[:n_prefix_full] + chunk_tokens
    self.chunk_idx += 1

    # --- Commit logic: candidate, LCP, overlap dedup, stagnation detection ---
    text_start = 0
    for i, t in enumerate(self.raw_tokens):
      if t == TOKEN_ASR_TEXT: text_start = i + 1; break
    text_tokens = self.raw_tokens[text_start:]
    n_text = len(text_tokens)

    # Candidate: everything except the unfixed rollback tail
    if self._is_final:
      candidate_len = n_text
    elif self.chunk_idx > self.unfixed_chunks:
      candidate_len = max(0, n_text - self.rollback)
      if candidate_len <= 0 and n_text > 0: candidate_len = n_text - 1
    else:
      candidate_len = 0

    candidate = text_tokens[:candidate_len]
    did_recovery = False
    did_periodic = False

    # --- Stagnation / degenerate detection ---
    candidate_advance = candidate_len - len(self.stable_text_tokens)
    if not self._is_final and self.hit_max_new and candidate_advance <= 1:
      self.stagnant_chunks += 1
    else:
      self.stagnant_chunks = 0

    need_recovery = False
    # Degenerate tail repeats?
    tail_reps, tail_period = self._tail_repeat_blocks(candidate, self.DEGEN_MAX_PERIOD)
    if tail_period > 0 and tail_reps >= self.DEGEN_MIN_REPEATS:
      need_recovery = True
    # Stagnant too long?
    if self.stagnant_chunks >= self.STALE_CHUNKS:
      need_recovery = True
    # Too many repeats dropped?
    if dropped_repeats >= 8:
      need_recovery = True

    if need_recovery and not self._is_final:
      self._reanchor()
      did_recovery = True
      stderr_log(f"  ! recovery reset (stagnant={self.stagnant_chunks} tail_reps={tail_reps}x{tail_period} dropped={dropped_repeats})\n")
    else:
      # --- LCP: longest common prefix with previous stable tokens ---
      lcp = 0
      while lcp < len(self.stable_text_tokens) and lcp < candidate_len \
            and self.stable_text_tokens[lcp] == candidate[lcp]:
        lcp += 1

      # Update stable with new candidate
      self.stable_text_tokens = list(candidate)

      # --- Overlap dedup: skip tokens already emitted ---
      emit_start = lcp
      if emit_start < candidate_len and len(self.emitted_text_tokens) > 0:
        max_overlap = min(candidate_len - emit_start, len(self.emitted_text_tokens))
        max_overlap = min(max_overlap, self.OVERLAP_MAX_TOKENS)
        for k in range(max_overlap, self.OVERLAP_MIN_TOKENS - 1, -1):
          if self.emitted_text_tokens[-k:] == candidate[emit_start:emit_start + k]:
            emit_start += k
            break

      # Emit new tokens one by one (token-level tracking)
      n_emitted_before = len(self.emitted_text)
      for i in range(emit_start, candidate_len):
        tok = candidate[i]
        self.emitted_text += model.tok.decode([tok])
        self.emitted_text_tokens.append(tok)
      emit_delta = self.emitted_text[n_emitted_before:]

      # --- Periodic reset (every N chunks, prevents slow drift) ---
      if (not self._is_final and self.chunk_idx > self.unfixed_chunks
          and (self.chunk_idx % self.RESET_INTERVAL_CHUNKS == 0)):
        self._reanchor()
        did_periodic = True

    # --- Stats ---
    total_ms = enc_ms + prefill_ms + decode_ms
    audio_sec = self.total_samples / SAMPLE_RATE
    rtf = (total_ms / 1000) / audio_sec if audio_sec > 0 else 0
    self.last_stats = {
      "audio_sec": round(audio_sec, 1),
      "chunk": self.chunk_idx,
      "enc_ms": round(enc_ms),
      "prefill_ms": round(prefill_ms),
      "decode_ms": round(decode_ms),
      "total_ms": round(total_ms),
      "rtf": round(rtf, 3),
      "reused": reuse_point,
      "prompt_len": prompt_len,
      "enc_windows": len(self.enc_cache),
      "max_windows": self.max_enc_windows,
      "decode_tokens": len(chunk_tokens),
      "committed": len(self.stable_text_tokens),
      "pending": n_text - candidate_len if not did_recovery else 0,
      "prefix_fed": len(prefix_toks),
    }

    # --- Diagnostic log ---
    decode_text = model.tok.decode(chunk_tokens).replace('\n', ' ')
    pending_text = model.tok.decode(text_tokens[candidate_len:]) if candidate_len < n_text and not did_recovery else ""
    flags = ""
    if did_recovery: flags += " !RECOVERY"
    if did_periodic: flags += " !PERIODIC_RESET"
    if dropped_repeats: flags += f" dropped={dropped_repeats}"
    stderr_log(
      f"chunk {self.chunk_idx}: {audio_sec:.1f}s  "
      f"enc={enc_ms:.0f}ms prefill={prefill_ms:.0f}ms({reuse_point}kv) decode={decode_ms:.0f}ms({len(chunk_tokens)}tok)  "
      f"win={len(self.enc_cache)}/{self.max_enc_windows} prefix={len(prefix_toks)} raw={len(self.raw_tokens)}{flags}\n"
      f"  decoded : {decode_text!r}\n"
      + (f"  commit  : lcp={lcp}/{candidate_len} "
         f"emit={candidate_len - emit_start}tok emitted_total={len(self.emitted_text_tokens)}\n"
         f"  +emit   : {emit_delta!r}\n"
         f"  pending : {pending_text!r}\n"
         if not did_recovery else
         f"  commit  : reset (emitted_total={len(self.emitted_text_tokens)}, carry={self.RESET_CARRY_TOKENS})\n")
    )

# ============================================================================
# OpenAI-compatible server with live microphone transcription
# ============================================================================

from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

# Self-contained HTML page: microphone capture + live transcription display
_HTML_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- WebSocket protocol helpers (RFC 6455) ----
_WS_MAGIC = b"258EAFA5-E914-47DA-95CA-5B99C7714885"

def _ws_recv(rfile) -> tuple[int, bytes]:
  """Read a complete WebSocket message (handles fragmentation). Returns (opcode, payload)."""
  fragments = []
  msg_opcode = None
  while True:
    try:
      b = rfile.read(2)
    except (ConnectionError, OSError):
      return 0x8, b''
    if not b or len(b) < 2: return 0x8, b''
    fin = bool(b[0] & 0x80)
    opcode = b[0] & 0xF
    masked = bool(b[1] & 0x80)
    length = b[1] & 0x7F
    if length == 126: length = int.from_bytes(rfile.read(2), 'big')
    elif length == 127: length = int.from_bytes(rfile.read(8), 'big')
    mask = rfile.read(4) if masked else b''
    payload = rfile.read(length) if length > 0 else b''
    if masked and len(mask) == 4 and len(payload) > 0:
      p = np.frombuffer(payload, dtype=np.uint8)
      m = np.tile(np.frombuffer(mask, dtype=np.uint8), (len(p) + 3) // 4)[:len(p)]
      payload = bytes(p ^ m)
    # Control frames (close/ping/pong) are never fragmented
    if opcode >= 0x8: return opcode, payload
    if opcode != 0: msg_opcode = opcode  # text(1) or binary(2) start
    fragments.append(payload)
    if fin: return msg_opcode or opcode, b''.join(fragments)

def _ws_send(sock, opcode: int, payload: bytes):
  """Send one WebSocket frame (unmasked, server-to-client)."""
  header = bytes([0x80 | opcode])
  n = len(payload)
  if n < 126: header += bytes([n])
  elif n < 65536: header += bytes([126]) + n.to_bytes(2, 'big')
  else: header += bytes([127]) + n.to_bytes(8, 'big')
  sock.sendall(header + payload)

class ASRHandler(HTTPRequestHandler):
  model: ASR  # set before serving
  session: StreamingSession | None = None  # global streaming session (single-user server)

  def log_request(self, code='-', size='-'): pass

  def do_GET(self):
    if self.path == '/ws' and 'upgrade' in self.headers.get('Connection', '').lower():
      self._handle_ws()
    elif self.path == '/':
      html_path = os.path.join(_HTML_DIR, 'index.html')
      try: self.send_data(open(html_path, 'rb').read(), content_type="text/html")
      except FileNotFoundError: self.send_error(404, "index.html not found")
    elif self.path == '/health': self.send_data(b'{"status":"ok"}')
    elif self.path == '/v1/models':
      self.send_data(json.dumps({"data": [{"id": "qwen3-asr", "object": "model"}]}).encode())
    else: self.send_error(404)

  def _handle_ws(self):
    """WebSocket upgrade + message loop for streaming transcription.

    Protocol:
      Client → Server:
        Text:   {"type":"start"}           → create new session
        Binary: Int16 LE PCM (16kHz mono)  → feed audio chunk
        Text:   {"type":"end"}             → finalize and close
      Server → Client:
        Text:   {"committed":"...","pending":"...","stats":{...}}
    """
    key = self.headers.get('Sec-WebSocket-Key', '')
    if not key: self.send_error(400, "Missing Sec-WebSocket-Key"); return
    accept = base64.b64encode(hashlib.sha1(key.encode() + _WS_MAGIC).digest()).decode()
    self.request.sendall((
      "HTTP/1.1 101 Switching Protocols\r\n"
      "Upgrade: websocket\r\nConnection: Upgrade\r\n"
      f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    ).encode())
    self.close_connection = True

    session = None
    sock = self.request
    try:
      while True:
        opcode, payload = _ws_recv(self.rfile)
        if opcode == 0x8:  # close
          try: _ws_send(sock, 0x8, b'')
          except OSError: pass
          break
        if opcode == 0x9:  # ping → pong
          _ws_send(sock, 0xA, payload); continue

        if opcode == 0x1:  # text message (JSON control)
          msg = json.loads(payload)
          if msg.get('type') == 'start':
            session = StreamingSession(self.model)
            ASRHandler.session = session
            stderr_log("ws: session started\n")
            _ws_send(sock, 0x1, json.dumps({"committed":"","pending":"","status":"started"}).encode())
          elif msg.get('type') == 'end':
            if session:
              result = session.feed(np.array([], dtype=np.float32), is_final=True)
              _ws_send(sock, 0x1, json.dumps({
                "committed": result["committed"], "pending": "",
                "stats": result.get("stats", {}), "status": "done",
              }).encode())
            ASRHandler.session = None; session = None
            break

        elif opcode == 0x2:  # binary message (Int16 PCM audio)
          if session and len(payload) >= 2:
            audio = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
            result = session.feed(audio)
            _ws_send(sock, 0x1, json.dumps({
              "committed": result["committed"], "pending": result["pending"],
              "stats": result.get("stats", {}),
            }).encode())
    except (ConnectionError, OSError, BrokenPipeError):
      pass
    finally:
      if session: ASRHandler.session = None
      stderr_log("ws: connection closed\n")

  def do_POST(self):
    if self.path == '/v1/audio/transcriptions':
      self._handle_transcribe()
    elif self.path.startswith('/v1/audio/stream'):
      self._handle_stream()
    else:
      self.send_error(404)

  def _handle_transcribe(self):
    """One-shot file transcription (stateless, per-file mode)."""
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
      result = self.model.transcribe(tmp_path)
      self.send_data(json.dumps({"text": result["text"]}).encode())
    except Exception as e:
      self.send_data(json.dumps({"error": str(e)}).encode(), status_code=500)
    finally:
      os.unlink(tmp_path)

  def _handle_stream(self):
    """Streaming transcription: client sends audio deltas, server maintains session state.

    POST /v1/audio/stream?action=start  -> reset session
    POST /v1/audio/stream?action=feed   -> body: WAV delta, returns {"text": "..."}
    POST /v1/audio/stream?action=end    -> process remaining audio, clear session
    """
    from urllib.parse import urlparse, parse_qs
    params = parse_qs(urlparse(self.path).query)
    action = params.get('action', ['feed'])[0]

    if action == 'start':
      ASRHandler.session = StreamingSession(self.model)
      stderr_log("stream session started\n")
      self.send_data(json.dumps({"text": "", "status": "started"}).encode())
      return

    if ASRHandler.session is None:
      # Auto-start if no explicit start
      ASRHandler.session = StreamingSession(self.model)
      stderr_log("stream session auto-started\n")

    if action in ('feed', 'end'):
      body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
      content_type = self.headers.get('Content-Type', '')

      # Extract and decode audio delta
      audio = np.array([], dtype=np.float32)
      if len(body) > 0:
        audio_data, filename = self._extract_audio(body, content_type)
        if audio_data and len(audio_data) > 0:
          suffix = os.path.splitext(filename)[1] if filename else '.wav'
          if not suffix or suffix == '.': suffix = '.wav'
          with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name
          try:
            audio = load_audio(tmp_path)
          finally:
            os.unlink(tmp_path)

      is_final = action == 'end'
      result = ASRHandler.session.feed(audio, is_final=is_final)

      resp = {"text": result["text"], "committed": result["committed"], "pending": result["pending"], "stats": result.get("stats", {})}
      if is_final:
        ASRHandler.session = None
        resp["status"] = "done"
        stderr_log("stream session ended\n")

      self.send_data(json.dumps(resp).encode())
      return

    self.send_error(400, f"Unknown action: {action}")

  def _extract_audio(self, body: bytes, content_type: str) -> tuple[bytes | None, str]:
    """Extract audio bytes and filename from multipart/form-data or raw body."""
    if 'multipart/form-data' not in content_type:
      return (body, ''), ''

    # Find boundary
    boundary = None
    for part in content_type.split(';'):
      part = part.strip()
      if part.startswith('boundary='):
        boundary = part[9:].strip('"').encode()
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

# ============================================================================
# Model registry and CLI
# ============================================================================

KNOWN_MODELS = {
  "qwen3-asr:0.6b": "https://huggingface.co/FlippyDora/qwen3-asr-0.6b-GGUF/resolve/main/qwen3-asr-0.6b-f16.gguf",
}

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Qwen3-ASR inference via tinygrad")
  parser.add_argument("--model", "-m", required=True, help="Path to GGUF file, or model name to download (e.g. qwen3-asr:0.6b)")
  parser.add_argument("--serve", nargs='?', type=int, const=8090, metavar="PORT", help="Run OpenAI-compatible API server")
  parser.add_argument("audio", nargs='?', help="WAV file to transcribe (omit for interactive mode)")
  args = parser.parse_args()

  # Resolve model: local path > known model name (download)
  if os.path.exists(args.model):
    stderr_log(f"loading {args.model}...\n")
    raw = Tensor(pathlib.Path(args.model))
  elif args.model in KNOWN_MODELS:
    url = KNOWN_MODELS[args.model]
    stderr_log(f"downloading {args.model} from {url}...\n")
    raw = Tensor.from_url(url)
  else:
    print(f"Model not found: {args.model}")
    print(f"  Pass a path to a GGUF file, or one of: {', '.join(KNOWN_MODELS.keys())}")
    sys.exit(1)

  model = ASR.from_gguf(raw)
  del raw
  import gc; gc.collect()

  if args.serve:
    model.warmup()
    ASRHandler.model = model
    stderr_log(f"open http://localhost:{args.serve} for microphone transcription\n")
    server = TCPServerWithReuse(('', args.serve), ASRHandler)
    server.daemon_threads = True
    try: server.serve_forever()
    except KeyboardInterrupt: stderr_log("shutting down\n"); server.server_close()
  elif args.audio:
    result = model.transcribe(args.audio)
    print(result["text"])
  else:
    # Interactive mode
    while True:
      try:
        path = input("wav> ").strip()
      except (EOFError, KeyboardInterrupt):
        break
      if not path: continue
      if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
      result = model.transcribe(path)
      print(result["text"])
