"""Qwen3-TTS: text-to-speech via tinygrad.

Talker LM (28-layer Qwen3) + Code Predictor (5-layer) + Vocoder (BigVGAN).
Loads from safetensors, serves OpenAI-compatible /v1/audio/speech.
"""

import json
import os
import struct
import time

import numpy as np
from tinygrad import Tensor, dtypes, nn, UOp, TinyJit

import builtins
if not hasattr(builtins, '_tts_t0'): builtins._tts_t0 = time.time()
def log(msg: str): print(f'[{time.time()-builtins._tts_t0:7.2f}s] {msg}', flush=True)

# ── Constants ────────────────────────────────────────────────────────

TALKER_LAYERS = 28
CODEPRED_LAYERS = 5
TEXT_HIDDEN = 2048
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
TALKER_VOCAB = 3072     # codec head output (includes special tokens)
CODEC_VOCAB = 2048      # codebook token range
TEXT_VOCAB = 151936
NUM_CODE_GROUPS = 16
MAX_DECODE_STEPS = 200  # ~16s audio at 12.5 Hz
TOP_K = 50
ROPE_THETA = 1000000.0
RMS_NORM_EPS = 1e-6

# Special tokens
TOKEN_IM_START = 151644
TOKEN_IM_END = 151645
TOKEN_BOS = 151672
TOKEN_EOS = 151673
TOKEN_PAD = 151671

CODEC_BOS = 2149
CODEC_EOS = 2150
CODEC_PAD = 2148
CODEC_NOTHINK = 2155
CODEC_THINK_BOS = 2156
CODEC_THINK_EOS = 2157


# ── BPE Tokenizer ───────────────────────────────────────────────────

def _bytes_to_unicode():
  bs = list(range(ord('!'), ord('~')+1)) + list(range(0xA1, 0xAC+1)) + list(range(0xAE, 0xFF+1))
  cs = list(bs)
  n = 0
  for b in range(256):
    if b not in bs:
      bs.append(b)
      cs.append(256 + n)
      n += 1
  return dict(zip(bs, [chr(c) for c in cs]))

_BYTE_TO_UNICODE = _bytes_to_unicode()

class BPETokenizer:
  """Byte-level BPE tokenizer (Qwen3-compatible)."""

  def __init__(self, vocab_path: str, merges_path: str):
    with open(vocab_path, 'r', encoding='utf-8') as f:
      self.vocab: dict[str, int] = json.load(f)
    self.merges: dict[tuple[str, str], int] = {}
    with open(merges_path, 'r', encoding='utf-8') as f:
      for i, line in enumerate(f):
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split()
        if len(parts) == 2: self.merges[(parts[0], parts[1])] = i

  @classmethod
  def from_gguf(cls, tokens: list[str], merges: list[str]) -> 'BPETokenizer':
    """Build tokenizer from GGUF metadata (no files needed)."""
    tok = cls.__new__(cls)
    tok.vocab = {t: i for i, t in enumerate(tokens)}
    tok.merges = {}
    for i, m in enumerate(merges):
      parts = m.split(' ', 1)
      if len(parts) == 2: tok.merges[(parts[0], parts[1])] = i
    return tok

  def encode(self, text: str) -> list[int]:
    tokens = [_BYTE_TO_UNICODE[b] for b in text.encode('utf-8')]
    tokens = self._apply_bpe(tokens)
    return [self.vocab[t] for t in tokens if t in self.vocab]

  def _apply_bpe(self, tokens: list[str]) -> list[str]:
    if len(tokens) <= 1: return tokens
    while True:
      pairs = {(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)}
      if not pairs: break
      best = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
      if best not in self.merges: break
      new, i = [], 0
      while i < len(tokens):
        if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best:
          new.append(tokens[i] + tokens[i+1]); i += 2
        else:
          new.append(tokens[i]); i += 1
      tokens = new
    return tokens


# ── RoPE ─────────────────────────────────────────────────────────────

def _precompute_freqs(dim: int, end: int, theta: float) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(1) * freqs.unsqueeze(0)
  return Tensor.cat(freqs.cos(), freqs.sin(), dim=-1).contiguous()

def _apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
  cos, sin = freqs.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return Tensor.cat(x1 * cos - x2 * sin, x2 * cos + x1 * sin, dim=-1)


# ── Transformer block (shared by talker and code predictor) ──────────

class _Ns: pass  # namespace for weight loading

class Qwen3Block:
  """Transformer block matching llm.py's TransformerBlock graph structure.

  Uses @function on _attention and _feed_forward to create proper UOp graph
  boundaries. Without these, 28 layers inline into one massive graph that
  JITBEAM can't optimize (same issue as write_after regression).
  """

  def __init__(self, hidden: int, intermediate: int, n_heads: int, n_kv_heads: int,
               head_dim: int, max_context: int = 4096):
    self.n_heads, self.n_kv_heads, self.head_dim, self.max_context = n_heads, n_kv_heads, head_dim, max_context
    # Attention (named to match llm.py's TransformerBlock for weight loading)
    self.attn_q = nn.Linear(hidden, n_heads * head_dim, bias=False)
    self.attn_k = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
    self.attn_v = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
    self.attn_output = nn.Linear(n_heads * head_dim, hidden, bias=False)
    self.attn_q_norm = nn.RMSNorm(head_dim, RMS_NORM_EPS)
    self.attn_k_norm = nn.RMSNorm(head_dim, RMS_NORM_EPS)
    self.attn_norm = nn.RMSNorm(hidden, RMS_NORM_EPS)
    # Feed-forward
    self.ffn_gate = nn.Linear(hidden, intermediate, bias=False)
    self.ffn_up = nn.Linear(hidden, intermediate, bias=False)
    self.ffn_down = nn.Linear(intermediate, hidden, bias=False)
    self.ffn_norm = nn.RMSNorm(hidden, RMS_NORM_EPS)

  def __call__(self, x: Tensor, start_pos: int | UOp) -> Tensor:
    if not hasattr(self, 'cache_kv'):
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.n_kv_heads, self.max_context,
                                   self.head_dim, device=x.device).contiguous().realize()
    B, T, _ = x.shape
    # Attention
    x_n = self.attn_norm(x)
    q = self.attn_q_norm(self.attn_q(x_n).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2))
    k = self.attn_k_norm(self.attn_k(x_n).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2))
    v = self.attn_v(x_n).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
    freqs = _precompute_freqs(self.head_dim, self.max_context, ROPE_THETA)[start_pos:start_pos+T]
    q, k = _apply_rope(q, freqs), _apply_rope(k, freqs)

    assigned_kv = self.cache_kv.uop.after(
      self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.assign(Tensor.stack(k, v).contiguous().uop))
    tkv = Tensor(assigned_kv, device=assigned_kv.device)
    k, v = tkv[0, :, :, :start_pos+T, :], tkv[1, :, :, :start_pos+T, :]
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if T > 1 else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True).transpose(1, 2).reshape(B, T, -1)
    h = x + self.attn_output(attn)

    # Feed-forward
    h_n = self.ffn_norm(h)
    return (h + self.ffn_down(self.ffn_gate(h_n).silu() * self.ffn_up(h_n))).contiguous()


# ── Text Projection MLP ─────────────────────────────────────────────

class TextProjection:
  def __init__(self, text_hidden: int, talker_hidden: int, intermediate: int):
    self.linear_fc1 = nn.Linear(text_hidden, intermediate)
    self.linear_fc2 = nn.Linear(intermediate, talker_hidden)

  def __call__(self, x: Tensor) -> Tensor:
    return self.linear_fc2(self.linear_fc1(x).silu())


# ── Code Predictor ───────────────────────────────────────────────────

class CodePredictor:
  """5-layer transformer predicting 15 sub-codebook tokens per talker step.

  Unrolled JIT: prefill(T=2) + 14x decode(T=1) in one graph. All argmax on GPU,
  single CPU sync to return 15 codes. Eliminates 14 GPU sync round-trips per step.
  """
  def __init__(self, hidden: int, intermediate: int, n_heads: int, n_kv_heads: int,
               head_dim: int, n_layers: int = 5, max_context: int = 32):
    self.hidden, self.n_layers = hidden, n_layers
    self.model = _Ns()
    self.model.layers = [Qwen3Block(hidden, intermediate, n_heads, n_kv_heads, head_dim, max_context)
                         for _ in range(n_layers)]
    self.model.norm = nn.RMSNorm(hidden, RMS_NORM_EPS)
    self.model.codec_embedding = [nn.Embedding(CODEC_VOCAB, hidden) for _ in range(NUM_CODE_GROUPS - 1)]
    self.lm_head = [nn.Linear(hidden, CODEC_VOCAB, bias=False) for _ in range(NUM_CODE_GROUPS - 1)]

  def reset_cache(self):
    for layer in self.model.layers:
      if hasattr(layer, 'cache_kv'):
        layer.cache_kv.assign(Tensor.zeros_like(layer.cache_kv).contiguous()).realize()



# ── Sampling ─────────────────────────────────────────────────────────

def sample_topk(logits_np: np.ndarray, temperature: float = 0.9, top_k: int = 50,
                rng: np.random.Generator | None = None,
                history: list[int] | None = None, rep_penalty: float = 1.05) -> int:
  """Top-k sampling with repetition penalty (CPU/numpy). Kept for reference/testing."""
  logits_np = logits_np.copy()

  # Suppress special tokens [2048, 3072) except EOS
  eos_logit = logits_np[CODEC_EOS]
  logits_np[CODEC_VOCAB:TALKER_VOCAB] = -1e9
  logits_np[CODEC_EOS] = eos_logit

  if history and rep_penalty > 1.0:
    for tok in history:
      if 0 <= tok < len(logits_np):
        logits_np[tok] = logits_np[tok] / rep_penalty if logits_np[tok] > 0 else logits_np[tok] * rep_penalty

  if temperature <= 0: return int(np.argmax(logits_np))
  logits_np = logits_np / temperature
  top_idx = np.argpartition(logits_np, -top_k)[-top_k:]
  top_vals = logits_np[top_idx]
  top_vals -= top_vals.max()
  probs = np.exp(top_vals)
  probs /= probs.sum()
  idx = (rng or np.random).choice(top_k, p=probs)
  return int(top_idx[idx])



# ── WAV writing ──────────────────────────────────────────────────────

def write_wav(path: str, samples: np.ndarray, sample_rate: int = 24000):
  """Write float32 samples as 16-bit PCM WAV."""
  pcm = np.clip(samples, -1.0, 1.0)
  pcm16 = (pcm * 32767).astype(np.int16)
  n = len(pcm16)
  with open(path, 'wb') as f:
    f.write(b'RIFF')
    f.write(struct.pack('<I', 36 + n * 2))
    f.write(b'WAVEfmt ')
    f.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    f.write(b'data')
    f.write(struct.pack('<I', n * 2))
    f.write(pcm16.tobytes())





# ── TTSModel ─────────────────────────────────────────────────────────

class TTSModel:
  """Qwen3-TTS inference: talker LM + code predictor + vocoder."""

  def __init__(self, model_dir: str, verbose: bool = False):
    self.model_dir = model_dir
    self.verbose = verbose
    self.talker_layers: list[Qwen3Block] = []
    self.talker_norm: nn.RMSNorm | None = None
    self.text_embedding: nn.Embedding | None = None
    self.codec_embedding: nn.Embedding | None = None
    self.codec_head: nn.Linear | None = None
    self.text_projection: TextProjection | None = None
    self.code_pred: CodePredictor | None = None
    self.tokenizer: BPETokenizer | None = None
    self.vocoder = None
    self.voice_presets: dict[str, int] = {}
    self.hidden = 0

  def load(self):
    t0 = time.time()

    # Load GGUF (convert from safetensors first with tools/convert_tts_gguf.py)
    gguf_files = [f for f in os.listdir(self.model_dir) if f.endswith('.gguf')]
    if not gguf_files:
      raise FileNotFoundError(f'No .gguf file in {self.model_dir}. Run: python tools/convert_tts_gguf.py {self.model_dir}')
    self._load_gguf(os.path.join(self.model_dir, gguf_files[0]))
    log(f'[TTS] GGUF loaded in {time.time()-t0:.1f}s')

    # Init JIT (after weight loading so codec_head is available)
    self._v_start_pos = UOp.variable('tts_sp', 1, 4095)

    # Suppress mask: -inf for special tokens [2048, 3072) except EOS (2150)
    suppress = [0.0] * TALKER_VOCAB
    for i in range(CODEC_VOCAB, TALKER_VOCAB):
      if i != CODEC_EOS: suppress[i] = float('-inf')
    suppress_mask = Tensor(suppress).realize()
    self._suppress_mask = suppress_mask

    # Combined talker + CP JIT — one dispatch per decode step.
    # Inlines code predictor (15 sub-codes) + codec sum + talker decode + top-k sampling.
    # Eliminates one Python→GPU→Python round-trip vs separate JITs.
    talker_layers, talker_norm = self.talker_layers, self.talker_norm
    c_head = self.codec_head
    codec_emb_w = self.codec_embedding.weight
    cp_layers = self.code_pred.model.layers
    cp_norm = self.code_pred.model.norm
    cp_lm_heads = self.code_pred.lm_head
    cp_codec_embeds = self.code_pred.model.codec_embedding

    @TinyJit
    def _combined_step(talker_h: Tensor, cb0_e: Tensor, trail: Tensor,
                       sp: UOp, smask: Tensor, temp: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
      # --- Code Predictor: predict 15 sub-codes from talker hidden + cb0 embedding ---
      buf = Tensor.cat(talker_h, cb0_e, dim=1).contiguous()
      x = buf
      for layer in cp_layers: x = layer(x, start_pos=0)
      h = cp_norm(x)
      code_t = (h[:, 1:2, :] @ cp_lm_heads[0].weight.T).reshape(-1).argmax()
      codes_list = [code_t]
      csum = cp_codec_embeds[0].weight[code_t].reshape(1, 1, -1)
      for i in range(1, NUM_CODE_GROUPS - 1):
        embed = cp_codec_embeds[i - 1].weight[codes_list[-1]].reshape(1, 1, -1)
        x = embed
        for layer in cp_layers: x = layer(x, start_pos=1 + i)
        h = cp_norm(x)
        code_t = (h[:, 0:1, :] @ cp_lm_heads[i].weight.T).reshape(-1).argmax()
        codes_list.append(code_t)
        csum = csum + cp_codec_embeds[i].weight[code_t].reshape(1, 1, -1)
      sub_codes = Tensor.stack(*codes_list)

      # --- Codec sum + talker decode + sample ---
      codec_sum = cb0_e + csum + trail
      x = codec_sum
      for layer in talker_layers: x = layer(x, sp)
      h = talker_norm(x)
      logits = c_head(h[:, -1:, :]).reshape(-1) + smask
      logits = logits / temp
      sorted_logits, _ = logits.sort(descending=True)
      threshold = sorted_logits[TOP_K - 1]
      logits = (logits >= threshold).where(logits, float('-inf'))
      probs = (logits - logits.max()).exp()
      probs = probs / probs.sum()
      cb0 = probs.multinomial(1).reshape(())
      cb0_e_new = codec_emb_w[cb0].reshape(1, 1, -1)
      return h, cb0, cb0_e_new, sub_codes

    self._combined_step = _combined_step

    # Pre-allocate input buffers for combined JIT (assign pattern for stable UOps)
    H = self.hidden
    self._talker_h_buf = Tensor.randn(1, 1, H).contiguous().realize()
    self._cb0_e_buf = Tensor.randn(1, 1, H).contiguous().realize()
    self._trail_buf = Tensor.randn(1, 1, H).contiguous().realize()

    # Warm up combined JIT (3 calls: ignore, capture, first replay)
    log('[TTS] JIT warmup starting')
    temp_t = Tensor([0.9]).contiguous().realize()
    Tensor.manual_seed(0)  # ensure RNG counter exists before warmup
    for i in range(3):
      self._talker_h_buf.assign(Tensor.randn(1, 1, H).contiguous()).realize()
      self._cb0_e_buf.assign(Tensor.randn(1, 1, H).contiguous()).realize()
      self._trail_buf.assign(Tensor.randn(1, 1, H).contiguous()).realize()
      h_w, _, _, _ = self._combined_step(self._talker_h_buf, self._cb0_e_buf, self._trail_buf,
                                          self._v_start_pos.bind(10 + i), suppress_mask, temp_t)
      h_w.realize()
      log(f'[TTS] JIT warmup {i+1}/3 done')
    # Save reference to JIT's captured RNG counter (for deterministic seeding)
    self._jit_rng_counter = Tensor._device_rng_counters.get(temp_t.device)

    # Vocoder
    vocoder_dir = os.path.join(os.path.dirname(self.model_dir), 'Qwen3-TTS-Tokenizer-12Hz')
    if os.path.isdir(vocoder_dir):
      from tts_vocoder import Vocoder
      self.vocoder = Vocoder(vocoder_dir, verbose=self.verbose, fp16=True)
      self.vocoder.load()
      log(f'[TTS] Vocoder weights loaded')
      self.vocoder.init_symbolic(max_T=MAX_DECODE_STEPS)
      self.vocoder.warmup_symbolic()
      log(f'[TTS] Vocoder warmed up from {vocoder_dir}')

    log(f'[TTS] Loaded in {time.time()-t0:.1f}s')

  def _load_gguf(self, gguf_path: str):
    """Load from F16 GGUF (converted from safetensors). Config from GGUF metadata."""
    from tinygrad.nn.state import gguf_load
    log(f'[TTS] Loading GGUF: {gguf_path}')
    kv, weights = gguf_load(Tensor(open(gguf_path, 'rb').read()))
    arch = kv['general.architecture']

    H = kv[f'{arch}.embedding_length']
    intermediate = kv[f'{arch}.feed_forward_length']
    n_heads = kv[f'{arch}.attention.head_count']
    n_kv = kv[f'{arch}.attention.head_count_kv']
    head_dim = kv[f'{arch}.attention.key_length']
    n_layers = kv[f'{arch}.block_count']
    self.hidden = H

    log(f'[TTS] {arch} ({H}d, {n_layers}L, {n_heads}h)')

    self._build_model(H, intermediate, n_heads, n_kv, head_dim, n_layers,
                      kv[f'{arch}.code_predictor.hidden_size'],
                      kv[f'{arch}.code_predictor.intermediate_size'],
                      kv[f'{arch}.code_predictor.num_attention_heads'],
                      kv[f'{arch}.code_predictor.num_key_value_heads'],
                      kv[f'{arch}.code_predictor.head_dim'],
                      kv[f'{arch}.code_predictor.num_hidden_layers'])

    # Assign weights (F16 from GGUF, cast to F32 for compute)
    assigned = 0
    for key, tensor in weights.items():
      t = tensor.to(None).cast(dtypes.float32).realize()
      target, attr = self._resolve_target(key)
      if target is not None and attr is not None and hasattr(target, attr):
        getattr(target, attr).assign(t).realize()
        assigned += 1
    log(f'[TTS] Assigned {assigned}/{len(weights)} weights')

    # Tokenizer from GGUF metadata
    if 'tokenizer.ggml.tokens' in kv and 'tokenizer.ggml.merges' in kv:
      self.tokenizer = BPETokenizer.from_gguf(kv['tokenizer.ggml.tokens'], kv['tokenizer.ggml.merges'])

    # Voice presets from GGUF metadata
    prefix = f'{arch}.spk_id.'
    for k, v in kv.items():
      if k.startswith(prefix):
        self.voice_presets[k[len(prefix):]] = v
    if self.voice_presets:
      log(f'[TTS] Voice presets: {list(self.voice_presets.keys())}')

  def _build_model(self, H, intermediate, n_heads, n_kv, head_dim, n_layers,
                   cp_hidden, cp_intermediate, cp_n_heads, cp_n_kv, cp_head_dim, cp_n_layers):
    """Build model structures (shared by GGUF and safetensors paths)."""
    self.talker_layers = [Qwen3Block(H, intermediate, n_heads, n_kv, head_dim) for _ in range(n_layers)]
    self.talker_norm = nn.RMSNorm(H, RMS_NORM_EPS)
    self.text_embedding = nn.Embedding(TEXT_VOCAB, TEXT_HIDDEN)
    self.codec_embedding = nn.Embedding(TALKER_VOCAB, H)
    self.codec_head = nn.Linear(H, TALKER_VOCAB, bias=False)
    self.text_projection = TextProjection(TEXT_HIDDEN, H, TEXT_HIDDEN)
    self.code_pred = CodePredictor(cp_hidden, cp_intermediate, cp_n_heads, cp_n_kv, cp_head_dim, cp_n_layers)

  def _resolve_target(self, key: str):
    """Map safetensors key to (target_object, attribute_name)."""
    # talker.model.layers.N.* → talker_layers[N].*
    if key.startswith('talker.model.layers.'):
      parts = key[len('talker.model.layers.'):].split('.', 1)
      idx = int(parts[0])
      return self._walk(self.talker_layers[idx], parts[1])
    # talker.model.norm.weight → talker_norm.weight
    if key.startswith('talker.model.norm.'): return self.talker_norm, key.split('.')[-1]
    # talker.model.text_embedding.weight
    if key == 'talker.model.text_embedding.weight': return self.text_embedding, 'weight'
    # talker.model.codec_embedding.weight
    if key == 'talker.model.codec_embedding.weight': return self.codec_embedding, 'weight'
    # talker.codec_head.weight
    if key == 'talker.codec_head.weight': return self.codec_head, 'weight'
    # talker.text_projection.*
    if key.startswith('talker.text_projection.'):
      return self._walk(self.text_projection, key[len('talker.text_projection.'):])
    # talker.code_predictor.model.* → code_pred.model.*
    if key.startswith('talker.code_predictor.model.'):
      return self._walk(self.code_pred.model, key[len('talker.code_predictor.model.'):])
    # talker.code_predictor.lm_head.N.weight → code_pred.lm_head[N].weight
    if key.startswith('talker.code_predictor.lm_head.'):
      parts = key[len('talker.code_predictor.lm_head.'):].split('.')
      return self.code_pred.lm_head[int(parts[0])], parts[1]
    # speaker_encoder.* — skip (not needed for base TTS)
    return None, None

  # Map safetensors/GGUF weight names → llm.py-style attribute names
  _WEIGHT_MAP: dict[str, str] = {
    'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k',
    'self_attn.v_proj': 'attn_v', 'self_attn.o_proj': 'attn_output',
    'self_attn.q_norm': 'attn_q_norm', 'self_attn.k_norm': 'attn_k_norm',
    'input_layernorm': 'attn_norm', 'post_attention_layernorm': 'ffn_norm',
    'mlp.gate_proj': 'ffn_gate', 'mlp.up_proj': 'ffn_up', 'mlp.down_proj': 'ffn_down',
  }

  @staticmethod
  def _walk(obj, path: str):
    """Walk dotted path, handling list indices. Applies weight name mapping."""
    # Apply weight name mapping (matches anywhere in the path)
    for src, dst in TTSModel._WEIGHT_MAP.items():
      path = path.replace(src, dst)
    parts = path.split('.')
    for p in parts[:-1]:
      obj = obj[int(p)] if p.isdigit() else getattr(obj, p, None)
      if obj is None: return None, None
    return obj, parts[-1]

  def _text_embed_project(self, ids: list[int]) -> Tensor:
    """Embed text tokens and project to talker hidden dim."""
    t = Tensor([ids]).cast(dtypes.int)
    return self.text_projection(self.text_embedding(t))

  def _talker_forward(self, x: Tensor, start_pos: int | UOp) -> Tensor:
    """Forward through talker layers + norm."""
    for layer in self.talker_layers: x = layer(x, start_pos)
    return self.talker_norm(x)

  def _reset_talker_cache(self):
    for layer in self.talker_layers:
      if hasattr(layer, 'cache_kv'):
        layer.cache_kv.assign(Tensor.zeros_like(layer.cache_kv).contiguous()).realize()

  def synthesize(self, text: str, voice: str | None = None,
                 temperature: float = 0.9, top_k: int = 50,
                 seed: int | None = None, max_steps: int = MAX_DECODE_STEPS) -> dict:
    """Synthesize speech. Returns dict with audio_path, n_steps, n_samples, elapsed_ms, rtf."""
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # 1. Tokenize
    role_tokens = self.tokenizer.encode("assistant\n")
    text_tokens = self.tokenizer.encode(text)
    input_ids = [TOKEN_IM_START] + role_tokens + text_tokens + [TOKEN_EOS] + [TOKEN_IM_END] * 4
    role_len = 1 + len(role_tokens)

    # 2. Build interleaved prefill embeddings
    role_embed = self._text_embed_project(input_ids[:role_len])
    tts_bos_embed = self._text_embed_project([TOKEN_BOS])
    tts_pad_embed = self._text_embed_project([TOKEN_PAD])

    codec_prefix_ids = [CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS, CODEC_PAD, CODEC_BOS]
    codec_embeds = []
    insert_at = len(codec_prefix_ids) - 2
    for i, cid in enumerate(codec_prefix_ids):
      if i == insert_at and voice and voice in self.voice_presets:
        spk_embed = self.codec_embedding(Tensor([[self.voice_presets[voice]]]).cast(dtypes.int))
        codec_embeds.append(spk_embed)
      codec_embeds.append(self.codec_embedding(Tensor([[cid]]).cast(dtypes.int)))

    n_codec = len(codec_embeds)
    n_pad = n_codec - 2
    combined = [tts_pad_embed + codec_embeds[i] for i in range(n_pad)]
    combined.append(tts_bos_embed + codec_embeds[n_pad])
    text_first = self._text_embed_project([input_ids[role_len]])
    combined.append(text_first + codec_embeds[-1])

    prefill_embed = Tensor.cat(role_embed, *combined, dim=1)
    pf_len = prefill_embed.shape[1]

    # 3. Build trailing text embeddings (as GPU tensors)
    text_start = role_len + 1
    text_end = len(input_ids) - 5
    H = self.hidden
    trailing: list[Tensor] = []
    if text_end > text_start:
      t_embed = self._text_embed_project(input_ids[text_start:text_end])
      trailing = [t_embed[:, i:i+1, :] for i in range(t_embed.shape[1])]
    trailing.append(self._text_embed_project([TOKEN_EOS]))
    pad_tensor = tts_pad_embed

    # 4. Prefill
    self._reset_talker_cache()
    hidden = self._talker_forward(prefill_embed, 0)
    hidden.realize()
    logits_np = self.codec_head(hidden[:, -1:, :]).numpy()[0, 0]  # (TALKER_VOCAB,) to CPU

    if self.verbose:
      log(f'[TTS] Prefill: {pf_len} positions, {len(trailing)} trailing')

    # 5. Autoregressive decode — combined CP + talker JIT, one dispatch per step
    t_decode_start = time.time()
    all_codes: list[tuple[int, np.ndarray]] = []  # (cb0_int, sub_codes_np)
    pos = pf_len

    # Reset JIT RNG counter for deterministic seeding
    if seed is not None and self._jit_rng_counter is not None:
      self._jit_rng_counter.assign(Tensor([seed % (2**32)], dtype=dtypes.uint32).contiguous()).realize()

    # First token: CPU sampling from prefill logits (prefill isn't JIT'd)
    rng = np.random.default_rng(seed)
    cb0_int = sample_topk(logits_np, temperature, top_k, rng)
    temp_t = Tensor([temperature]).contiguous().realize()
    self._cb0_e_buf.assign(self.codec_embedding(Tensor([[cb0_int]]).cast(dtypes.int)).contiguous()).realize()

    step_times: list[float] = []
    for step in range(max_steps):
      if cb0_int == CODEC_EOS:
        if self.verbose: log(f'[TTS] EOS at step {step}')
        break
      t_step = time.time()

      self._talker_h_buf.assign(hidden[:, -1:, :].contiguous()).realize()
      trail = trailing[step] if step < len(trailing) else pad_tensor
      self._trail_buf.assign(trail.contiguous()).realize()

      # Combined CP + talker: one JIT dispatch per step
      hidden, cb0_t, cb0_embed_out, sub_codes_gpu = self._combined_step(
        self._talker_h_buf, self._cb0_e_buf, self._trail_buf,
        self._v_start_pos.bind(pos), self._suppress_mask, temp_t)

      # Snapshot sub-codes to CPU before next JIT call overwrites the output buffer
      all_codes.append((cb0_int, sub_codes_gpu.numpy()))
      self._cb0_e_buf.assign(cb0_embed_out.contiguous()).realize()
      cb0_int = cb0_t.item()  # single 4-byte sync for EOS check
      pos += 1
      step_times.append((time.time() - t_step) * 1000)

    if self.verbose and step_times:
      avg = sum(step_times) / len(step_times)
      log(f'[TTS] Per-step avg: {avg:.0f}ms ({len(step_times)} steps)')

    n_steps = len(all_codes)
    decode_ms = (time.time() - t_decode_start) * 1000

    # 6. Vocoder — codes already on CPU (snapshotted per step)
    t_voc = time.time()
    if self.vocoder and n_steps > 0:
      codes_np = np.array([[cb0] + sc.tolist() for cb0, sc in all_codes], dtype=np.int64)  # (n_steps, 16)
      samples = self.vocoder.decode_symbolic(codes_np)
    else:
      samples = np.zeros(n_steps * 1920, dtype=np.float32)
    voc_ms = (time.time() - t_voc) * 1000

    # 7. Write WAV
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), f'tts_{os.getpid()}_{int(time.time()*1000)}.wav')
    write_wav(tmp_path, samples)

    elapsed_ms = (time.time() - t0) * 1000
    audio_sec = n_steps * 0.08
    rtf = elapsed_ms / 1000.0 / audio_sec if audio_sec > 0 else 0.0

    if True:  # always print TTS summary
      log(f'[TTS] {n_steps} steps ({audio_sec:.1f}s audio), '
          f'decode={decode_ms:.0f}ms vocoder={voc_ms:.0f}ms total={elapsed_ms:.0f}ms RTF={rtf:.2f}')

    return {
      'audio_path': tmp_path,
      'n_steps': n_steps,
      'n_samples': len(samples),
      'elapsed_ms': elapsed_ms,
      'decode_ms': decode_ms,
      'vocoder_ms': voc_ms,
      'audio_sec': audio_sec,
      'rtf': rtf,
    }


if __name__ == '__main__':
  import argparse, shutil
  p = argparse.ArgumentParser(description='Qwen3-TTS synthesis')
  p.add_argument('texts', nargs='*', help='Text(s) to synthesize')
  p.add_argument('--model', default=os.path.join(os.environ.get('LOCALAPPDATA', '.'), 'local-models/tts/qwen3-tts-12hz-0.6b-customvoice'))
  p.add_argument('--seed', type=int, default=42)
  p.add_argument('--out', default='.', help='Output directory')
  p.add_argument('--voice', default=None)
  p.add_argument('--temperature', type=float, default=0.9)
  p.add_argument('--list-voices', action='store_true', help='List available voices and exit')
  args = p.parse_args()

  if args.list_voices:
    from tinygrad.nn.state import gguf_load
    gguf_files = [f for f in os.listdir(args.model) if f.endswith('.gguf')]
    if not gguf_files: raise SystemExit(f'No .gguf file in {args.model}')
    kv, _ = gguf_load(Tensor(open(os.path.join(args.model, gguf_files[0]), 'rb').read()))
    arch = kv['general.architecture']
    voices = sorted(k[len(f'{arch}.spk_id.'):] for k in kv if k.startswith(f'{arch}.spk_id.'))
    print(', '.join(voices) if voices else 'No voice presets found in model.')
    raise SystemExit(0)

  m = TTSModel(args.model, verbose=True)
  m.load()

  if args.voice and args.voice not in m.voice_presets:
    print(f'Unknown voice "{args.voice}". Available: {", ".join(sorted(m.voice_presets.keys())) or "none"}')
    raise SystemExit(1)

  if not args.texts:
    p.error('at least one text argument is required')

  for i, raw in enumerate(args.texts):
    if '::' in raw:
      text, seed_str = raw.rsplit('::', 1)
      seed = int(seed_str)
    else:
      text, seed = raw, args.seed
    log(f'[TTS] === Synthesizing {i+1}/{len(args.texts)}: "{text}" (seed={seed}) ===')
    r = m.synthesize(text, voice=args.voice, temperature=args.temperature, seed=seed)
    out_path = os.path.join(args.out, f'tts_{i+1}.wav')
    shutil.copy(r['audio_path'], out_path)
    log(f'[TTS] Wrote {out_path}')
