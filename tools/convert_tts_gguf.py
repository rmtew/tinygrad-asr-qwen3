#!/usr/bin/env python3
"""Convert Qwen3-TTS safetensors (BF16) to F16 GGUF for tinygrad.

Usage:
  python tools/convert_tts_gguf.py path/to/qwen3-tts-12hz-0.6b-base
  python tools/convert_tts_gguf.py path/to/qwen3-tts-12hz-0.6b-base -o model.gguf

Produces a single GGUF file with:
  - All talker weights (F16)
  - Model config as GGUF metadata
  - Tokenizer vocab + merges embedded

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
    if dt == 'BF16':
      u16 = np.frombuffer(chunk, dtype=np.uint16).reshape(shape)
      f32 = np.zeros(shape, dtype=np.float32)
      f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
      arrays[name] = f32
    elif dt == 'F32':
      arrays[name] = np.frombuffer(chunk, dtype=np.float32).reshape(shape).copy()
    elif dt == 'F16':
      arrays[name] = np.frombuffer(chunk, dtype=np.float16).reshape(shape).astype(np.float32)
  return arrays


def convert(model_dir: str, output_path: str | None = None):
  import gguf

  config_path = os.path.join(model_dir, 'config.json')
  with open(config_path) as f:
    config = json.load(f)
  tc = config['talker_config']
  cpc = tc['code_predictor_config']

  if output_path is None:
    size = config.get('tts_model_size', '0b6')
    variant = config.get('tts_model_type', 'base')
    output_path = os.path.join(model_dir, f'qwen3-tts-{size}-{variant}-f16.gguf')

  print(f'Loading safetensors from {model_dir}...')
  weights = load_safetensors_numpy(os.path.join(model_dir, 'model.safetensors'))
  print(f'  {len(weights)} tensors loaded')

  # Filter to talker weights only (skip speaker_encoder)
  talker_weights = {k: v for k, v in weights.items() if k.startswith('talker.')}
  print(f'  {len(talker_weights)} talker tensors')

  writer = gguf.GGUFWriter(output_path, arch='qwen3tts')

  # Model config metadata
  writer.add_block_count(tc['num_hidden_layers'])
  writer.add_embedding_length(tc['hidden_size'])
  writer.add_feed_forward_length(tc['intermediate_size'])
  writer.add_head_count(tc['num_attention_heads'])
  writer.add_head_count_kv(tc['num_key_value_heads'])
  writer.add_layer_norm_rms_eps(tc['rms_norm_eps'])
  writer.add_rope_freq_base(tc['rope_theta'])
  writer.add_context_length(tc.get('max_position_embeddings', 32768))
  writer.add_key_length(tc['head_dim'])

  # TTS-specific metadata
  writer.add_uint32('qwen3tts.text_hidden_size', tc['text_hidden_size'])
  writer.add_uint32('qwen3tts.text_vocab_size', tc['text_vocab_size'])
  writer.add_uint32('qwen3tts.vocab_size', tc['vocab_size'])
  writer.add_uint32('qwen3tts.num_code_groups', tc['num_code_groups'])

  # Code predictor config
  writer.add_uint32('qwen3tts.code_predictor.num_hidden_layers', cpc['num_hidden_layers'])
  writer.add_uint32('qwen3tts.code_predictor.hidden_size', cpc['hidden_size'])
  writer.add_uint32('qwen3tts.code_predictor.intermediate_size', cpc['intermediate_size'])
  writer.add_uint32('qwen3tts.code_predictor.num_attention_heads', cpc['num_attention_heads'])
  writer.add_uint32('qwen3tts.code_predictor.num_key_value_heads', cpc['num_key_value_heads'])
  writer.add_uint32('qwen3tts.code_predictor.head_dim', cpc['head_dim'])
  writer.add_uint32('qwen3tts.code_predictor.vocab_size', cpc['vocab_size'])

  # Special tokens
  writer.add_uint32('qwen3tts.codec_bos_id', tc['codec_bos_id'])
  writer.add_uint32('qwen3tts.codec_eos_id', tc['codec_eos_token_id'])
  writer.add_uint32('qwen3tts.codec_pad_id', tc['codec_pad_id'])
  writer.add_uint32('qwen3tts.codec_nothink_id', tc['codec_nothink_id'])
  writer.add_uint32('qwen3tts.codec_think_bos_id', tc['codec_think_bos_id'])
  writer.add_uint32('qwen3tts.codec_think_eos_id', tc['codec_think_eos_id'])

  # Voice presets (spk_id)
  if tc.get('spk_id'):
    for name, idx in tc['spk_id'].items():
      writer.add_uint32(f'qwen3tts.spk_id.{name}', idx)

  # Tokenizer
  vocab_path = os.path.join(model_dir, 'vocab.json')
  merges_path = os.path.join(model_dir, 'merges.txt')
  if os.path.exists(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
      vocab = json.load(f)
    tokens = [''] * len(vocab)
    for tok, idx in vocab.items():
      if idx < len(tokens):
        tokens[idx] = tok
    writer.add_token_list(tokens)
    writer.add_string('tokenizer.ggml.model', 'gpt2')
  if os.path.exists(merges_path):
    with open(merges_path, 'r', encoding='utf-8') as f:
      merges = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    writer.add_token_merges(merges)

  # Write tensors as F16
  print(f'Writing {len(talker_weights)} tensors to {output_path}...')
  for name, data in sorted(talker_weights.items()):
    f16 = data.astype(np.float16)
    writer.add_tensor(name, f16)

  writer.write_header_to_file()
  writer.write_kv_data_to_file()
  writer.write_tensors_to_file()
  writer.close()

  size_mb = os.path.getsize(output_path) / (1024 * 1024)
  print(f'Done: {output_path} ({size_mb:.1f} MB)')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert Qwen3-TTS safetensors to F16 GGUF')
  parser.add_argument('model_dir', help='Path to Qwen3-TTS model directory')
  parser.add_argument('-o', '--output', help='Output GGUF path (default: model_dir/qwen3-tts-*.gguf)')
  args = parser.parse_args()
  convert(args.model_dir, args.output)
