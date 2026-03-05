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
import sys, os, argparse, json, time, math, wave, struct, uuid, functools, pathlib, tempfile
import numpy as np
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
    """Conv stem + positional embeddings. Returns (tensor, seq_len)."""
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

  def _transformer(self, x: Tensor, cu_seqlens: list[int]) -> Tensor:
    for block in self.blk: x = block(x, self.n_heads, self.head_dim, cu_seqlens)
    x = self.ln_post(x)
    return self.proj2(self.proj1(x).gelu()).realize()

  def forward(self, mel: np.ndarray) -> Tensor:
    """Encode mel spectrogram [128, frames] → [n_tokens, output_dim].

    Pads mel to fixed bucket boundaries (multiples of bucket_frames) so the JIT
    sees a small number of unique graph shapes and caches them effectively.
    """
    actual_frames = mel.shape[1]
    # Bucket: pad to next multiple of 800 frames (~8s). Gives ~4 buckets for typical audio.
    bucket_frames = self.chunk_size * 8  # 800 frames
    padded_frames = ((actual_frames + bucket_frames - 1) // bucket_frames) * bucket_frames
    if padded_frames > actual_frames:
      mel = np.pad(mel, ((0, 0), (0, padded_frames - actual_frames)), mode='constant')

    x, seq_len = self._conv_stem(mel)
    tokens_per_chunk = 13  # 100 frames through 3x stride-2

    # Compute actual token count (before padding)
    actual_chunks = actual_frames // self.chunk_size
    actual_tokens = actual_chunks * tokens_per_chunk
    tail = actual_frames % self.chunk_size
    if tail > 0: actual_tokens += max(1, tail // 8)

    tokens_per_window = tokens_per_chunk * (self.n_window_infer // self.chunk_size)
    cu_seqlens = list(range(0, seq_len, tokens_per_window)) + [seq_len]
    if cu_seqlens[-2] == cu_seqlens[-1]: cu_seqlens = cu_seqlens[:-1]
    out = self._transformer(x, cu_seqlens)
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

    return model

  def warmup(self):
    """Exercise all JIT paths so subsequent calls are fast.
    Needs 2-3 prefill calls with different sizes to compile @function variants."""
    stderr_log("warming up JIT...\n")
    dim = self._embed_buf.shape[2]
    # Encoder warmup (one bucket size)
    dummy_mel = np.zeros((128, 800), dtype=np.float32)
    self.encoder.forward(dummy_mel)
    # Prefill warmup: 3 different sizes to exercise @function compilation
    for nt in [200, 100, 150]:
      self._embed_buf[:, :nt].assign(Tensor.randn(1, nt, dim).contiguous()).realize()
      sp_b, nt_b = self.v_sp.bind(0), self.v_nt.bind(nt)
      self.decoder.prefill_embed_jit(self._embed_buf[:, sp_b:sp_b+nt_b], sp_b).realize()
    # Decode warmup: single-token path
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
                         rollback: int = 5, max_new_tokens: int = 32,
                         max_enc_windows: int = 4, max_prefix_tokens: int = 150,
                         callback = None) -> dict:
    """Streaming transcription: process audio in fixed-size chunks.

    Like antirez's C streaming: fixed encoder windows → cached, sliding decoder
    context with prefix rollback. All shapes are fixed → JIT compiles once.

    Args:
      audio: float32 mono 16kHz samples
      chunk_sec: seconds of audio per processing step (default 2.0)
      rollback: tokens to roll back between chunks for boundary stability
      max_new_tokens: max tokens decoded per chunk
      max_enc_windows: sliding window limit for encoder cache
      max_prefix_tokens: max previously-decoded tokens fed as prefix
      callback: optional fn(text_so_far, is_final) called after each chunk

    Returns: {"text": str, "elapsed_ms": float, "audio_sec": float, "rtf": float}
    """
    t0 = time.time()
    audio_sec = len(audio) / SAMPLE_RATE
    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    dim = self.encoder.output_dim
    enc_window_frames = 800  # 8s of audio per encoder window
    enc_window_samples = enc_window_frames * HOP_LENGTH

    # Reuse cached prompt token embeddings
    prefix_tok_embeds = self._prefix_embeds
    suffix_tok_embeds = self._suffix_embeds

    # Encoder window cache: list of (enc_output_tensor, n_tokens)
    enc_cache: list[tuple[Tensor, int]] = []
    next_window_start = 0  # sample index of next full window boundary

    # Decoded token history
    raw_tokens: list[int] = []
    audio_cursor = 0
    chunk_idx = 0

    # Reuse shared symbolic variables for JIT stability

    total_enc_ms, total_prefill_ms, total_decode_ms = 0.0, 0.0, 0.0

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
        # Pad to bucket so all complete windows have identical shape
        bucket = self.encoder.chunk_size * 8
        padded = ((window_mel.shape[1] + bucket - 1) // bucket) * bucket
        if padded > window_mel.shape[1]:
          window_mel = np.pad(window_mel, ((0, 0), (0, padded - window_mel.shape[1])))
        enc_out = self.encoder.forward(window_mel[:, :padded])
        # Trim padding tokens
        actual_tokens = enc_window_frames // 8  # ~100 tokens per 800 frames
        enc_cache.append((enc_out[:actual_tokens].realize(), actual_tokens))
        next_window_start += enc_window_samples

      # Encode partial tail (from last full window boundary to audio_cursor)
      partial_enc = None
      partial_tokens = 0
      if full_end < audio_cursor:
        partial_samples = audio_cursor - full_end
        partial_mel = compute_mel(audio[int(full_end):audio_cursor])
        # Pad to same bucket as full windows
        bucket = self.encoder.chunk_size * 8
        padded = ((partial_mel.shape[1] + bucket - 1) // bucket) * bucket
        if padded > partial_mel.shape[1]:
          partial_mel = np.pad(partial_mel, ((0, 0), (0, padded - partial_mel.shape[1])))
        partial_out = self.encoder.forward(partial_mel[:, :padded])
        actual_partial = max(1, partial_mel.shape[1] // 8)  # approximate
        partial_enc = partial_out[:actual_partial].realize()
        partial_tokens = actual_partial

      # Evict old encoder windows
      while len(enc_cache) > max_enc_windows:
        enc_cache.pop(0)

      # Concatenate encoder outputs
      enc_parts = [e for e, _ in enc_cache]
      if partial_enc is not None: enc_parts.append(partial_enc)
      if not enc_parts:
        chunk_idx += 1
        continue
      audio_embeds = Tensor.cat(*enc_parts, dim=0) if len(enc_parts) > 1 else enc_parts[0]
      n_audio_tokens = audio_embeds.shape[0]
      enc_ms = (time.time() - t_enc) * 1000
      total_enc_ms += enc_ms

      # --- Build prefill: [prompt_prefix] [audio] [prompt_suffix] [past_text_tokens] ---
      t_pf = time.time()
      n_text_prefix = min(max(len(raw_tokens) - rollback, 0), max_prefix_tokens)
      text_prefix_ids = raw_tokens[len(raw_tokens) - n_text_prefix:] if n_text_prefix > 0 else []

      parts = [prefix_tok_embeds, audio_embeds, suffix_tok_embeds]
      if text_prefix_ids:
        parts.append(self.decoder.token_embd(Tensor(text_prefix_ids)))
      combined = Tensor.cat(*parts, dim=0).reshape(1, -1, dim)
      prompt_len = combined.shape[1]

      # KV cache: start_pos=0 overwrites old positions; causal mask prevents reading stale data

      # Single-call prefill through JIT (shared buffer and variables with transcribe)
      assert prompt_len <= self._embed_buf.shape[1], f"prompt_len {prompt_len} > max_context"
      self._embed_buf[:, :prompt_len].assign(combined.contiguous()).realize()
      sp_b, nt_b = self.v_sp.bind(0), self.v_nt.bind(prompt_len)
      out = self.decoder.prefill_embed_jit(self._embed_buf[:, sp_b:sp_b+nt_b], sp_b).realize()
      token = int(out.item())
      prefill_ms = (time.time() - t_pf) * 1000
      total_prefill_ms += prefill_ms

      # --- Decode up to max_new_tokens ---
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

      # Strip EOS from this chunk's output
      while chunk_tokens and chunk_tokens[-1] in EOS_TOKEN_IDS: chunk_tokens.pop()
      # Strip language tag / <asr_text> marker if present
      if TOKEN_ASR_TEXT in chunk_tokens:
        chunk_tokens = chunk_tokens[chunk_tokens.index(TOKEN_ASR_TEXT) + 1:]

      raw_tokens.extend(chunk_tokens)
      chunk_idx += 1

      if callback:
        text_so_far = self.tok.decode(raw_tokens).strip()
        callback(text_so_far, is_final)

    total_ms = (time.time() - t0) * 1000
    # Strip EOS from final output
    while raw_tokens and raw_tokens[-1] in EOS_TOKEN_IDS: raw_tokens.pop()
    text = self.tok.decode(raw_tokens).strip()

    rtf = (total_ms / 1000) / audio_sec if audio_sec > 0 else 0
    stderr_log(f"stream: {audio_sec:.1f}s audio, {len(raw_tokens)} tokens, "
               f"enc={total_enc_ms:.0f}ms, prefill={total_prefill_ms:.0f}ms, "
               f"decode={total_decode_ms:.0f}ms, total={total_ms:.0f}ms, RTF={rtf:.2f}\n")

    return {"text": text, "elapsed_ms": total_ms, "audio_sec": audio_sec, "rtf": rtf}

# ============================================================================
# OpenAI-compatible server: /v1/audio/transcriptions
# ============================================================================

from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

class ASRHandler(HTTPRequestHandler):
  model: ASR  # set by main before serving

  def log_request(self, code='-', size='-'): pass

  def do_GET(self):
    if self.path == '/health':
      self.send_data(b'{"status":"ok"}')
    elif self.path == '/v1/models':
      self.send_data(json.dumps({"data": [{"id": "qwen3-asr", "object": "model"}]}).encode())
    else:
      self.send_error(404)

  def do_POST(self):
    if self.path != '/v1/audio/transcriptions':
      self.send_error(404)
      return

    content_type = self.headers.get('Content-Type', '')
    content_length = int(self.headers.get('Content-Length', 0))
    body = self.rfile.read(content_length)

    # Parse multipart form data to extract the audio file
    audio_data = self._extract_audio(body, content_type)
    if audio_data is None:
      self.send_error(400, "No audio file found in request")
      return

    # Write to temp file and transcribe
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
      f.write(audio_data)
      tmp_path = f.name

    try:
      result = self.model.transcribe(tmp_path)
      self.send_data(json.dumps({"text": result["text"]}).encode())
    finally:
      os.unlink(tmp_path)

  def _extract_audio(self, body: bytes, content_type: str) -> bytes | None:
    """Extract audio file bytes from multipart/form-data."""
    if 'multipart/form-data' not in content_type: return body  # assume raw audio
    # Find boundary
    for part in content_type.split(';'):
      part = part.strip()
      if part.startswith('boundary='):
        boundary = part[9:].strip('"').encode()
        break
    else:
      return None

    # Split on boundary and find the file part
    parts = body.split(b'--' + boundary)
    for part in parts:
      if b'name="file"' in part or b'name="audio"' in part:
        # Find the blank line separating headers from body
        idx = part.find(b'\r\n\r\n')
        if idx >= 0:
          data = part[idx + 4:]
          # Strip trailing \r\n-- if present
          if data.endswith(b'\r\n'): data = data[:-2]
          if data.endswith(b'--'): data = data[:-2]
          if data.endswith(b'\r\n'): data = data[:-2]
          return data
    return None

  def send_data(self, data: bytes, content_type: str = "application/json"):
    self.send_response(200)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

# ============================================================================
# Model registry and CLI
# ============================================================================

models = {
  "qwen3-asr:0.6b": "https://huggingface.co/FlippyDora/qwen3-asr-0.6b-GGUF/resolve/main/qwen3-asr-0.6b-f16.gguf",
}

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Qwen3-ASR inference via tinygrad")
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help="Model name or path to GGUF file")
  parser.add_argument("--serve", nargs='?', type=int, const=8090, metavar="PORT", help="Run OpenAI-compatible API server")
  parser.add_argument("audio", nargs='?', help="WAV file to transcribe (interactive mode)")
  args = parser.parse_args()

  # Load model
  if os.path.exists(args.model):
    gguf_path = args.model
  elif args.model in models:
    gguf_path = models[args.model]
  else:
    print(f"Unknown model: {args.model}. Available: {', '.join(models.keys())}")
    sys.exit(1)

  stderr_log(f"loading {gguf_path}...\n")
  raw = Tensor(pathlib.Path(gguf_path)) if os.path.exists(gguf_path) else Tensor.from_url(gguf_path)
  model = ASR.from_gguf(raw)
  del raw
  import gc; gc.collect()

  if args.serve:
    ASRHandler.model = model
    stderr_log(f"serving on http://localhost:{args.serve}/v1/audio/transcriptions\n")
    TCPServerWithReuse(('', args.serve), ASRHandler).serve_forever()
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
