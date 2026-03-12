"""Vocoder for Qwen3-TTS: converts codec tokens to audio waveform.

Pipeline: codes [T, 16] -> RVQ decode -> pre_conv -> pre_transformer -> upsample -> BigVGAN -> audio
Total upsample: 2*2*8*5*4*3 = 1920x (12.5 Hz -> 24 kHz)
"""

import time
import numpy as np
from tinygrad import Tensor, dtypes, UOp
from tinygrad.engine.jit import TinyJit
from tts import log

NUM_CODEBOOKS = 16
CODEBOOK_DIM = 256
CODEBOOK_SIZE = 2048
RVQ_OUT_DIM = 512
PRE_XFMR_HIDDEN = 512
PRE_XFMR_LAYERS = 8
PRE_XFMR_HEADS = 16
PRE_XFMR_HEAD_DIM = 64
ROPE_THETA = 10000.0
BIGVGAN_RATES = [8, 5, 4, 3]


# ── helpers ──────────────────────────────────────────────────────────

def _causal_conv1d(x: Tensor, weight: Tensor, bias: Tensor | None,
                   dilation: int = 1, groups: int = 1) -> Tensor:
  """Causal 1D conv via conv2d. x:(1,C,T) → (1,C_out,T)."""
  K = weight.shape[-1]
  pad = dilation * (K - 1)
  x4 = x.reshape(x.shape[0], x.shape[1], 1, x.shape[2]).pad(((0,0),(0,0),(0,0),(pad,0)))
  w4 = weight.reshape(weight.shape[0], weight.shape[1], 1, K)
  out = x4.conv2d(w4, bias=bias, groups=groups, dilation=(1, dilation))
  return out.reshape(out.shape[0], out.shape[1], out.shape[3])

def _causal_conv_transpose1d(x: Tensor, weight: Tensor, bias: Tensor | None,
                             stride: int) -> Tensor:
  """Causal transposed 1D conv. x:(1,C,T) → (1,C_out,T_out)."""
  K = weight.shape[-1]
  x4 = x.reshape(x.shape[0], x.shape[1], 1, x.shape[2])
  w4 = weight.reshape(weight.shape[0], weight.shape[1], 1, K)
  out = x4.conv_transpose2d(w4, bias=bias, stride=(1, stride))
  out = out.reshape(out.shape[0], out.shape[1], out.shape[3])
  trim = K - stride
  return out[:, :, trim:-trim] if trim > 0 else out

def _snake_beta(x: Tensor, alpha: Tensor, inv_beta: Tensor) -> Tensor:
  """SnakeBeta: x + inv_beta * sin²(alpha * x)."""
  a = alpha.reshape(1, -1, 1)
  ib = inv_beta.reshape(1, -1, 1)
  s = (a * x).sin()
  return x + ib * s * s

def _rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
  """RMSNorm. x:(1,T,D), weight:(D,)."""
  return (x / (x * x).mean(axis=-1, keepdim=True).add(eps).sqrt()) * weight

def _layer_norm_channels(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-6) -> Tensor:
  """LayerNorm across channels per timestep. x:(1,C,T)."""
  mean = x.mean(axis=1, keepdim=True)
  var = ((x - mean) * (x - mean)).mean(axis=1, keepdim=True)
  return (x - mean) / (var + eps).sqrt() * weight.reshape(1, -1, 1) + bias.reshape(1, -1, 1)

def _precompute_rope(T: int, head_dim: int, theta: float = 10000.0):
  half = head_dim // 2
  freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32)[:half] / head_dim))
  angles = np.outer(np.arange(T, dtype=np.float32), freqs)
  return Tensor(np.cos(angles)), Tensor(np.sin(angles))

def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
  """Apply RoPE to (B,heads,T,head_dim)."""
  half = x.shape[-1] // 2
  c = cos.reshape(1, 1, -1, half)
  s = sin.reshape(1, 1, -1, half)
  return Tensor.cat(x[..., :half] * c - x[..., half:] * s,
                    x[..., half:] * c + x[..., :half] * s, dim=-1)


# ── Vocoder ──────────────────────────────────────────────────────────

class Vocoder:
  """Vocoder: RVQ decode → pre-conv → 8-layer pre-transformer → ConvNeXt upsample → BigVGAN."""

  def __init__(self, model_path: str, verbose: bool = False, fp16: bool = False):
    self.model_path = model_path
    self.verbose = verbose
    self.fp16 = fp16

  def load(self):
    """Load vocoder weights. Prefers GGUF (F16), falls back to safetensors (F32)."""
    import os
    gguf_path = os.path.join(self.model_path, 'vocoder-f16.gguf')
    if os.path.exists(gguf_path):
      self._load_gguf(gguf_path)
    else:
      self._load_safetensors()

  def _load_gguf(self, gguf_path: str):
    """Load from F16 GGUF with pre-normalized codebooks."""
    from tinygrad.nn.state import gguf_load
    if self.verbose: log(f'[Vocoder] Loading GGUF: {gguf_path}')
    _, w = gguf_load(Tensor(open(gguf_path, 'rb').read()))

    compute_dtype = dtypes.float16 if self.fp16 else dtypes.float32
    def cast(t: Tensor) -> Tensor: return t.to(None).cast(compute_dtype).realize()

    # 1. RVQ codebooks (pre-normalized in GGUF)
    self.codebooks: list[Tensor] = []
    self.codebook_projs: list[Tensor] = []
    self.codebooks.append(cast(w['decoder.quantizer.rvq_first.codebook']))
    self.codebook_projs.append(cast(w['decoder.quantizer.rvq_first.output_proj']))
    rest_proj = cast(w['decoder.quantizer.rvq_rest.output_proj'])
    for i in range(15):
      self.codebooks.append(cast(w[f'decoder.quantizer.rvq_rest.codebook.{i}']))
      self.codebook_projs.append(rest_proj)

    # 2-5: Load remaining weights
    self._load_weights(w, cast)
    if self.verbose: log(f'[Vocoder] Loaded GGUF ({compute_dtype.name}) from {self.model_path}')

  def _load_safetensors(self):
    """Load from safetensors (F32, legacy path)."""
    import os
    from tinygrad.nn.state import safe_load
    w = {k: v.to(None).realize() for k, v in safe_load(os.path.join(self.model_path, 'model.safetensors')).items()}

    compute_dtype = dtypes.float16 if self.fp16 else dtypes.float32
    def cast(t: Tensor) -> Tensor: return t.cast(compute_dtype).realize() if self.fp16 else t

    # 1. RVQ codebooks: normalize embedding_sum by cluster_usage
    self.codebooks: list[Tensor] = []
    self.codebook_projs: list[Tensor] = []
    usage_np = np.maximum(w['decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage'].numpy(), 1e-7)
    emb_np = w['decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum'].numpy() / usage_np[:, None]
    self.codebooks.append(Tensor(emb_np).cast(compute_dtype).realize())
    self.codebook_projs.append(cast(w['decoder.quantizer.rvq_first.output_proj.weight'])[:, :, 0])
    rest_proj = cast(w['decoder.quantizer.rvq_rest.output_proj.weight'])[:, :, 0]
    for i in range(15):
      usage_np = np.maximum(w[f'decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage'].numpy(), 1e-7)
      emb_np = w[f'decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum'].numpy() / usage_np[:, None]
      self.codebooks.append(Tensor(emb_np).cast(compute_dtype).realize())
      self.codebook_projs.append(rest_proj)

    # 2-5: Load remaining weights
    self._load_weights(w, cast)
    if self.verbose: log(f'[Vocoder] Loaded safetensors ({compute_dtype.name}) from {self.model_path}')

  def _load_weights(self, w: dict, cast):
    """Load pre-conv, pre-transformer, upsample, BigVGAN weights."""
    # 2. Pre-conv
    self.pre_conv_w = cast(w['decoder.pre_conv.conv.weight'])
    self.pre_conv_b = cast(w['decoder.pre_conv.conv.bias'])

    # 3. Pre-transformer
    self.pt_in_w = cast(w['decoder.pre_transformer.input_proj.weight'])
    self.pt_in_b = cast(w['decoder.pre_transformer.input_proj.bias'])
    self.pt_out_w = cast(w['decoder.pre_transformer.output_proj.weight'])
    self.pt_out_b = cast(w['decoder.pre_transformer.output_proj.bias'])
    self.pt_norm = cast(w['decoder.pre_transformer.norm.weight'])

    self.pt_layers: list[dict[str, Tensor]] = []
    for i in range(PRE_XFMR_LAYERS):
      p = f'decoder.pre_transformer.layers.{i}.'
      self.pt_layers.append({
        'in_norm': cast(w[p + 'input_layernorm.weight']),
        'q_w': cast(w[p + 'self_attn.q_proj.weight']), 'k_w': cast(w[p + 'self_attn.k_proj.weight']),
        'v_w': cast(w[p + 'self_attn.v_proj.weight']), 'o_w': cast(w[p + 'self_attn.o_proj.weight']),
        'attn_scale': cast(w[p + 'self_attn_layer_scale.scale']),
        'post_norm': cast(w[p + 'post_attention_layernorm.weight']),
        'gate_w': cast(w[p + 'mlp.gate_proj.weight']), 'up_w': cast(w[p + 'mlp.up_proj.weight']),
        'down_w': cast(w[p + 'mlp.down_proj.weight']),
        'mlp_scale': cast(w[p + 'mlp_layer_scale.scale']),
      })

    # 4. ConvNeXt upsample (2 stages × 2×)
    self.up_stages: list[dict[str, Tensor]] = []
    for i in range(2):
      p = f'decoder.upsample.{i}.'
      self.up_stages.append({
        'ct_w': cast(w[p + '0.conv.weight']), 'ct_b': cast(w[p + '0.conv.bias']),
        'dw_w': cast(w[p + '1.dwconv.conv.weight']), 'dw_b': cast(w[p + '1.dwconv.conv.bias']),
        'n_w': cast(w[p + '1.norm.weight']), 'n_b': cast(w[p + '1.norm.bias']),
        'gamma': cast(w[p + '1.gamma']),
        'pw1_w': cast(w[p + '1.pwconv1.weight']), 'pw1_b': cast(w[p + '1.pwconv1.bias']),
        'pw2_w': cast(w[p + '1.pwconv2.weight']), 'pw2_b': cast(w[p + '1.pwconv2.bias']),
      })

    # 5. BigVGAN
    self.bgv_init_w = cast(w['decoder.decoder.0.conv.weight'])
    self.bgv_init_b = cast(w['decoder.decoder.0.conv.bias'])

    self.bgv_blocks: list[dict] = []
    for bi in range(4):
      p = f'decoder.decoder.{bi + 1}.block.'
      block: dict = {
        'snake_a': cast(w[p + '0.alpha']).exp(), 'snake_ib': cast(w[p + '0.beta']).exp().reciprocal(),
        'up_w': cast(w[p + '1.conv.weight']), 'up_b': cast(w[p + '1.conv.bias']),
        'res': [],
      }
      for ri in range(3):
        block['res'].append({
          'a1_a': cast(w[p + f'{ri+2}.act1.alpha']).exp(), 'a1_ib': cast(w[p + f'{ri+2}.act1.beta']).exp().reciprocal(),
          'c1_w': cast(w[p + f'{ri+2}.conv1.conv.weight']), 'c1_b': cast(w[p + f'{ri+2}.conv1.conv.bias']),
          'a2_a': cast(w[p + f'{ri+2}.act2.alpha']).exp(), 'a2_ib': cast(w[p + f'{ri+2}.act2.beta']).exp().reciprocal(),
          'c2_w': cast(w[p + f'{ri+2}.conv2.conv.weight']), 'c2_b': cast(w[p + f'{ri+2}.conv2.conv.bias']),
        })
      self.bgv_blocks.append(block)

    self.bgv_final_a = cast(w['decoder.decoder.5.alpha']).exp()
    self.bgv_final_ib = cast(w['decoder.decoder.5.beta']).exp().reciprocal()
    self.bgv_final_w = cast(w['decoder.decoder.6.conv.weight'])
    self.bgv_final_b = cast(w['decoder.decoder.6.conv.bias'])

  def _rvq_decode(self, codes: np.ndarray) -> Tensor:
    """RVQ decode: codes (T,16) int64 → Tensor (1,512,T)."""
    T = codes.shape[0]
    out = None
    for cb in range(NUM_CODEBOOKS):
      emb = self.codebooks[cb][Tensor(codes[:, cb].tolist()).cast(dtypes.int)].contiguous().realize()
      proj = (emb @ self.codebook_projs[cb].T).contiguous().realize()
      out = proj if out is None else out + proj
    return out.reshape(1, T, RVQ_OUT_DIM).permute(0, 2, 1)

  def decode(self, codes: np.ndarray) -> np.ndarray:
    """Full pipeline: codes (T,16) → audio float32 ndarray."""
    t0 = time.time()
    x = self._rvq_decode(codes)
    x = _causal_conv1d(x, self.pre_conv_w, self.pre_conv_b)
    x = self._pre_transformer(x)
    x = self._upsample(x)
    x = self._bigvgan(x)
    x = x.clip(-1.0, 1.0)
    result = x[0, 0].realize().numpy()
    if self.verbose: log(f'[Vocoder] {codes.shape[0]} steps -> {len(result)} samples in {(time.time()-t0)*1000:.0f}ms')
    return result

  def _pre_transformer(self, x: Tensor) -> Tensor:
    T = x.shape[2]
    h = x.permute(0, 2, 1).linear(self.pt_in_w.T, self.pt_in_b)
    rope_cos, rope_sin = _precompute_rope(T, PRE_XFMR_HEAD_DIM, ROPE_THETA)
    for lw in self.pt_layers:
      residual = h
      h_n = _rms_norm(h, lw['in_norm'])
      q = h_n.linear(lw['q_w'].T).reshape(1, T, PRE_XFMR_HEADS, PRE_XFMR_HEAD_DIM).permute(0, 2, 1, 3)
      k = h_n.linear(lw['k_w'].T).reshape(1, T, PRE_XFMR_HEADS, PRE_XFMR_HEAD_DIM).permute(0, 2, 1, 3)
      v = h_n.linear(lw['v_w'].T).reshape(1, T, PRE_XFMR_HEADS, PRE_XFMR_HEAD_DIM).permute(0, 2, 1, 3)
      q, k = _apply_rope(q, rope_cos, rope_sin), _apply_rope(k, rope_cos, rope_sin)
      attn = q.scaled_dot_product_attention(k, v, is_causal=True).permute(0, 2, 1, 3).reshape(1, T, -1)
      h = residual + attn.linear(lw['o_w'].T) * lw['attn_scale']
      residual = h
      h_n = _rms_norm(h, lw['post_norm'])
      h = residual + (h_n.linear(lw['gate_w'].T).silu() * h_n.linear(lw['up_w'].T)).linear(lw['down_w'].T) * lw['mlp_scale']
    h = _rms_norm(h, self.pt_norm)
    return h.linear(self.pt_out_w.T, self.pt_out_b).permute(0, 2, 1)

  def _upsample(self, x: Tensor) -> Tensor:
    for s in self.up_stages:
      x = _causal_conv_transpose1d(x, s['ct_w'], s['ct_b'], stride=2)
      residual = x
      x = _causal_conv1d(x, s['dw_w'], s['dw_b'], groups=x.shape[1])
      x = _layer_norm_channels(x, s['n_w'], s['n_b'])
      h = x.permute(0, 2, 1).linear(s['pw1_w'].T, s['pw1_b']).gelu().linear(s['pw2_w'].T, s['pw2_b'])
      x = residual + h.permute(0, 2, 1) * s['gamma'].reshape(1, -1, 1)
    return x

  def _bigvgan(self, x: Tensor) -> Tensor:
    x = _causal_conv1d(x, self.bgv_init_w, self.bgv_init_b)
    for bi, blk in enumerate(self.bgv_blocks):
      x = _snake_beta(x, blk['snake_a'], blk['snake_ib'])
      x = _causal_conv_transpose1d(x, blk['up_w'], blk['up_b'], stride=BIGVGAN_RATES[bi])
      for ri, rb in enumerate(blk['res']):
        residual = x
        x = _snake_beta(x, rb['a1_a'], rb['a1_ib'])
        x = _causal_conv1d(x, rb['c1_w'], rb['c1_b'], dilation=[1, 3, 9][ri])
        x = _snake_beta(x, rb['a2_a'], rb['a2_ib'])
        x = _causal_conv1d(x, rb['c2_w'], rb['c2_b'])
        x = residual + x
    x = _snake_beta(x, self.bgv_final_a, self.bgv_final_ib)
    return _causal_conv1d(x, self.bgv_final_w, self.bgv_final_b)

  # ── Symbolic JIT: one compiled kernel for all step counts ──────────

  def init_symbolic(self, max_T: int = 200):
    """Compile full pipeline once with symbolic T. Handles any T in [1, max_T]."""
    self._voc_max_T = max_T
    self._voc_v_T = UOp.variable('voc_T', 1, max_T)

    rope_cos_full, rope_sin_full = _precompute_rope(max_T, PRE_XFMR_HEAD_DIM, ROPE_THETA)
    compute_dtype = dtypes.float16 if self.fp16 else dtypes.float32
    rope_cos_full = rope_cos_full.cast(compute_dtype).contiguous().realize()
    rope_sin_full = rope_sin_full.cast(compute_dtype).contiguous().realize()

    # Capture references for closure
    pcw, pcb = self.pre_conv_w, self.pre_conv_b
    piw, pib, pow_, pob, ptn = self.pt_in_w, self.pt_in_b, self.pt_out_w, self.pt_out_b, self.pt_norm
    ptl = self.pt_layers
    ups = self.up_stages
    biw, bib = self.bgv_init_w, self.bgv_init_b
    bbl = self.bgv_blocks
    bfa, bfi, bfw, bfb = self.bgv_final_a, self.bgv_final_ib, self.bgv_final_w, self.bgv_final_b
    nh, hd = PRE_XFMR_HEADS, PRE_XFMR_HEAD_DIM

    @TinyJit
    def _vocoder_jit(x: Tensor, T: UOp) -> Tensor:
      x = x[:, :, :T]
      x = _causal_conv1d(x, pcw, pcb)
      h = x.permute(0, 2, 1).linear(piw.T, pib)
      rc, rs = rope_cos_full[:T], rope_sin_full[:T]
      for lw in ptl:
        residual = h
        h_n = _rms_norm(h, lw['in_norm'])
        q = h_n.linear(lw['q_w'].T).reshape(1, T, nh, hd).permute(0, 2, 1, 3)
        k = h_n.linear(lw['k_w'].T).reshape(1, T, nh, hd).permute(0, 2, 1, 3)
        v = h_n.linear(lw['v_w'].T).reshape(1, T, nh, hd).permute(0, 2, 1, 3)
        q, k = _apply_rope(q, rc, rs), _apply_rope(k, rc, rs)
        attn = q.scaled_dot_product_attention(k, v, is_causal=True).permute(0, 2, 1, 3).reshape(1, T, -1)
        h = residual + attn.linear(lw['o_w'].T) * lw['attn_scale']
        residual = h
        h_n = _rms_norm(h, lw['post_norm'])
        h = residual + (h_n.linear(lw['gate_w'].T).silu() * h_n.linear(lw['up_w'].T)).linear(lw['down_w'].T) * lw['mlp_scale']
      h = _rms_norm(h, ptn)
      x = h.linear(pow_.T, pob).permute(0, 2, 1)
      for s in ups:
        x = _causal_conv_transpose1d(x, s['ct_w'], s['ct_b'], stride=2)
        residual = x
        x = _causal_conv1d(x, s['dw_w'], s['dw_b'], groups=x.shape[1])
        x = _layer_norm_channels(x, s['n_w'], s['n_b'])
        h2 = x.permute(0, 2, 1).linear(s['pw1_w'].T, s['pw1_b']).gelu().linear(s['pw2_w'].T, s['pw2_b'])
        x = residual + h2.permute(0, 2, 1) * s['gamma'].reshape(1, -1, 1)
      x = _causal_conv1d(x, biw, bib)
      for bi, blk in enumerate(bbl):
        x = _snake_beta(x, blk['snake_a'], blk['snake_ib'])
        x = _causal_conv_transpose1d(x, blk['up_w'], blk['up_b'], stride=BIGVGAN_RATES[bi])
        for ri, rb in enumerate(blk['res']):
          residual = x
          x = _snake_beta(x, rb['a1_a'], rb['a1_ib'])
          x = _causal_conv1d(x, rb['c1_w'], rb['c1_b'], dilation=[1, 3, 9][ri])
          x = _snake_beta(x, rb['a2_a'], rb['a2_ib'])
          x = _causal_conv1d(x, rb['c2_w'], rb['c2_b'])
          x = residual + x
      x = _snake_beta(x, bfa, bfi)
      x = _causal_conv1d(x, bfw, bfb)
      return x.clip(-1.0, 1.0).contiguous().realize()

    self._vocoder_jit = _vocoder_jit
    buf_dtype = dtypes.float16 if self.fp16 else dtypes.float32
    self._voc_buf = Tensor.zeros(1, RVQ_OUT_DIM, max_T, dtype=buf_dtype).contiguous().realize()

  def warmup_symbolic(self):
    """Warm up symbolic JIT (2 trace + 1 cached)."""
    self._voc_buf.assign(Tensor.randn(*self._voc_buf.shape, dtype=self._voc_buf.dtype).contiguous()).realize()
    for T in [10, 20, 30]:
      r = self._vocoder_jit(self._voc_buf, self._voc_v_T.bind(T))
      _ = r[:, :, :T*1920].numpy()

  def decode_symbolic(self, codes: np.ndarray) -> np.ndarray:
    """Decode via symbolic JIT — no recompilation for different step counts."""
    T = codes.shape[0]
    assert self._voc_max_T >= T, f"T={T} exceeds max_T={self._voc_max_T}"
    rvq_out = self._rvq_decode(codes)
    padded = rvq_out.pad(((0,0),(0,0),(0, self._voc_max_T - T))) if self._voc_max_T > T else rvq_out
    self._voc_buf.assign(padded.contiguous()).realize()
    result = self._vocoder_jit(self._voc_buf, self._voc_v_T.bind(T))
    return result[:, :, :T*1920].flatten().numpy()
