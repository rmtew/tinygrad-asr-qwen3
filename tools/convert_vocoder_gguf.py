#!/usr/bin/env python3
"""Convert Qwen3-TTS vocoder safetensors (F32) to F16 GGUF for tinygrad.

Usage:
  python tools/convert_vocoder_gguf.py path/to/Qwen3-TTS-Tokenizer-12Hz
  python tools/convert_vocoder_gguf.py path/to/Qwen3-TTS-Tokenizer-12Hz -o vocoder.gguf

Produces a single GGUF file with all vocoder weights as F16.
Dependencies: gguf (pip install gguf), numpy
"""

import argparse
import json
import os
import struct
import sys

import numpy as np


def load_safetensors_numpy(path: str) -> dict[str, np.ndarray]:
  """Load safetensors as F32 numpy arrays."""
  with open(path, 'rb') as f:
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_size))
    data_start = 8 + header_size
    f.seek(data_start)
    raw = f.read()
  arrays: dict[str, np.ndarray] = {}
  for name, info in header.items():
    if name == '__metadata__': continue
    start, end = info['data_offsets']
    shape = tuple(info['shape'])
    dt = info['dtype']
    chunk = raw[start:end]
    if dt == 'F32':
      arrays[name] = np.frombuffer(chunk, dtype=np.float32).reshape(shape).copy()
    elif dt == 'F16':
      arrays[name] = np.frombuffer(chunk, dtype=np.float16).reshape(shape).astype(np.float32)
    elif dt == 'BF16':
      u16 = np.frombuffer(chunk, dtype=np.uint16).reshape(shape)
      f32 = np.zeros(shape, dtype=np.float32)
      f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
      arrays[name] = f32
  return arrays


def convert(model_dir: str, output_path: str | None = None):
  import gguf

  safetensors_path = os.path.join(model_dir, 'model.safetensors')
  if not os.path.exists(safetensors_path):
    print(f'Error: {safetensors_path} not found')
    sys.exit(1)

  if output_path is None:
    output_path = os.path.join(model_dir, 'vocoder-f16.gguf')

  print(f'Loading safetensors from {safetensors_path}...')
  weights = load_safetensors_numpy(safetensors_path)
  print(f'  {len(weights)} tensors loaded')

  writer = gguf.GGUFWriter(output_path, arch='qwen3tts_vocoder')

  # Vocoder architecture metadata
  writer.add_uint32('qwen3tts_vocoder.num_codebooks', 16)
  writer.add_uint32('qwen3tts_vocoder.codebook_size', 2048)
  writer.add_uint32('qwen3tts_vocoder.codebook_dim', 256)
  writer.add_uint32('qwen3tts_vocoder.rvq_out_dim', 512)
  writer.add_uint32('qwen3tts_vocoder.pre_xfmr_layers', 8)
  writer.add_uint32('qwen3tts_vocoder.pre_xfmr_heads', 16)
  writer.add_uint32('qwen3tts_vocoder.pre_xfmr_head_dim', 64)
  writer.add_uint32('qwen3tts_vocoder.bigvgan_rates_0', 8)
  writer.add_uint32('qwen3tts_vocoder.bigvgan_rates_1', 5)
  writer.add_uint32('qwen3tts_vocoder.bigvgan_rates_2', 4)
  writer.add_uint32('qwen3tts_vocoder.bigvgan_rates_3', 3)

  # Pre-normalize RVQ codebooks: embedding_sum / cluster_usage → codebook
  # This avoids runtime normalization and removes cluster_usage/embedding_sum from the model
  normalized = {}

  # First codebook (rvq_first)
  usage = np.maximum(weights['decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage'], 1e-7)
  emb = weights['decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum'] / usage[:, None]
  normalized['decoder.quantizer.rvq_first.codebook'] = emb
  # Keep output_proj but squeeze the last dim (weight is [out, in, 1])
  proj = weights['decoder.quantizer.rvq_first.output_proj.weight'][:, :, 0]
  normalized['decoder.quantizer.rvq_first.output_proj'] = proj

  # Rest codebooks (rvq_rest, 15 layers)
  rest_proj = weights['decoder.quantizer.rvq_rest.output_proj.weight'][:, :, 0]
  normalized['decoder.quantizer.rvq_rest.output_proj'] = rest_proj
  for i in range(15):
    usage = np.maximum(weights[f'decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage'], 1e-7)
    emb = weights[f'decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum'] / usage[:, None]
    normalized[f'decoder.quantizer.rvq_rest.codebook.{i}'] = emb

  # Pre-compute exp() for snake alpha/beta (stored as log in safetensors)
  for name, data in weights.items():
    if '.alpha' in name or '.beta' in name:
      if 'decoder.decoder.' in name:  # BigVGAN snake params
        normalized[name] = data  # store raw, exp() at load time (F16 exp precision matters)

  # Copy all other weights (decoder.pre_conv, decoder.pre_transformer, decoder.upsample, decoder.decoder)
  skip_prefixes = ('decoder.quantizer.',)
  for name, data in sorted(weights.items()):
    if any(name.startswith(p) for p in skip_prefixes): continue
    if name not in normalized:
      normalized[name] = data

  # Write tensors as F16
  print(f'Writing {len(normalized)} tensors to {output_path} (from {len(weights)} original)...')
  for name, data in sorted(normalized.items()):
    f16 = data.astype(np.float16)
    writer.add_tensor(name, f16)

  writer.write_header_to_file()
  writer.write_kv_data_to_file()
  writer.write_tensors_to_file()
  writer.close()

  size_mb = os.path.getsize(output_path) / (1024 * 1024)
  print(f'Done: {output_path} ({size_mb:.1f} MB)')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert Qwen3-TTS vocoder safetensors to F16 GGUF')
  parser.add_argument('model_dir', help='Path to Qwen3-TTS-Tokenizer-12Hz directory')
  parser.add_argument('-o', '--output', help='Output GGUF path (default: model_dir/vocoder-f16.gguf)')
  args = parser.parse_args()
  convert(args.model_dir, args.output)
