# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM JAX backend mappings for Qwen3.6 text weights."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


_VLLM_PREFIX = 'vllm_model.language_model.'
_LINEAR_NUM_VALUE_HEADS = 48
_LINEAR_NUM_KEY_HEADS = 16
_LINEAR_KEY_HEAD_DIM = 128
_LINEAR_VALUE_HEAD_DIM = 128
_LINEAR_KEY_DIM = _LINEAR_NUM_KEY_HEADS * _LINEAR_KEY_HEAD_DIM
_LINEAR_VALUE_DIM = _LINEAR_NUM_VALUE_HEADS * _LINEAR_VALUE_HEAD_DIM


class _FlatState:
  """Minimal flat_state adapter used after lazy Qwen3.6 fusion."""

  def __init__(self, flat_state):
    self._flat_state = tuple(flat_state)

  def flat_state(self):
    return list(self._flat_state)

  def from_flat_path(self, flat_path):
    return _FlatState(flat_path)


class _LazyConcatParam:
  """Param-like wrapper that materializes a concatenation only when read."""

  def __init__(self, params, axis: int):
    self._params = tuple(params)
    self._axis = axis

  @property
  def value(self):
    return jnp.concatenate(
        [param.value if hasattr(param, 'value') else param for param in self._params],
        axis=self._axis,
    )


def _infer_output_shards(target_value: Any, dim: int = -1) -> int:
  """Infer the active output-axis shard count from a target JAX array."""
  sharding = getattr(target_value, 'sharding', None)
  mesh = getattr(sharding, 'mesh', None)
  spec = getattr(sharding, 'spec', None)
  if mesh is None or spec is None:
    return 1

  spec_tuple = tuple(spec)
  if dim < 0:
    dim += len(spec_tuple)
  if dim < 0 or dim >= len(spec_tuple):
    return 1

  axes = spec_tuple[dim]
  if axes is None:
    return 1
  if not isinstance(axes, tuple):
    axes = (axes,)

  n_shards = 1
  for axis in axes:
    if axis is None:
      continue
    try:
      n_shards *= int(mesh.shape[axis])
    except (KeyError, TypeError):
      continue
  return n_shards


def _reorder_concatenated_tensor_for_sharding(
    concatenated_tensor: jnp.ndarray,
    split_sizes: tuple[int, ...],
    n_shards: int,
    dim: int = -1,
) -> jnp.ndarray:
  """Match tpu-inference fused-linear layout for output-sharded weights."""
  if n_shards <= 1:
    return concatenated_tensor
  if dim < 0:
    dim += concatenated_tensor.ndim
  if sum(split_sizes) != concatenated_tensor.shape[dim]:
    raise ValueError(
        'Packed Qwen3.6 tensor shape does not match split sizes: '
        f'{concatenated_tensor.shape=} {split_sizes=} {dim=}'
    )
  for split_size in split_sizes:
    if split_size % n_shards:
      raise ValueError(
          'Packed Qwen3.6 split size must be divisible by output shards: '
          f'{split_size=} {n_shards=}'
      )

  old_shape = concatenated_tensor.shape
  new_shape = old_shape[:dim] + (n_shards, -1) + old_shape[dim + 1:]
  split_tensors = []
  start_offset = 0
  for split_size in split_sizes:
    index = [slice(None)] * concatenated_tensor.ndim
    index[dim] = slice(start_offset, start_offset + split_size)
    split_tensors.append(concatenated_tensor[tuple(index)].reshape(new_shape))
    start_offset += split_size
  return jnp.concatenate(split_tensors, axis=dim + 1).reshape(old_shape)


def _packed_output_hook(split_sizes: tuple[int, ...]):
  """Create a post-align hook for tpu-inference merged output projections."""

  def hook(val, *, target_value=None, tp_size=None, **_):
    n_shards = _infer_output_shards(target_value, dim=-1)
    if n_shards <= 1 and tp_size is not None:
      try:
        n_shards = max(1, int(tp_size))
      except (TypeError, ValueError):
        n_shards = 1
    if n_shards > 1:
      val = jax.device_put(val, jax.local_devices(backend='cpu')[0])
    return _reorder_concatenated_tensor_for_sharding(
        val, split_sizes, n_shards, dim=-1
    )

  hook.run_after_shape_align = True
  return hook


TO_HF_HOOK_FNS = {
    'layers.*.mlp.gate_up_proj.kernel': _packed_output_hook((17408, 17408)),
    # Full attention q_proj carries [q, output_gate] for Qwen3.5/3.6.
    'layers.*.attn.qkv_proj.kernel': _packed_output_hook((12288, 1024, 1024)),
    'layers.*.linear_attn.in_proj_qkvz.kernel': _packed_output_hook((
        _LINEAR_KEY_DIM,
        _LINEAR_KEY_DIM,
        _LINEAR_VALUE_DIM,
        _LINEAR_VALUE_DIM,
    )),
    'layers.*.linear_attn.in_proj_ba.kernel': _packed_output_hook((
        _LINEAR_NUM_VALUE_HEADS,
        _LINEAR_NUM_VALUE_HEADS,
    )),
}


TO_HF_MAPPINGS: Dict[str, MappingEntry] = {
    'embedder.input_embedding': (
        f'{_VLLM_PREFIX}model.embed_tokens.weight',
        ('model', None),
    ),
    'layers.*.input_layernorm.w': (
        f'{_VLLM_PREFIX}model.layers.*.input_layernorm.weight',
        (None,),
    ),
    'layers.*.post_attention_layernorm.w': (
        f'{_VLLM_PREFIX}model.layers.*.post_attention_layernorm.weight',
        (None,),
    ),
    'layers.*.mlp.gate_up_proj.kernel': (
        f'{_VLLM_PREFIX}model.layers.*.mlp.gate_up_proj.weight',
        (None, 'model'),
    ),
    'layers.*.mlp.down_proj.kernel': (
        f'{_VLLM_PREFIX}model.layers.*.mlp.down_proj.weight',
        ('model', None),
    ),
    'layers.*.attn.qkv_proj.kernel': (
        f'{_VLLM_PREFIX}model.layers.*.self_attn.qkv_proj.weight',
        (None, 'model'),
    ),
    'layers.*.attn.o_proj.kernel': (
        f'{_VLLM_PREFIX}model.layers.*.self_attn.o_proj.weight',
        ('model', None),
    ),
    'layers.*.attn.q_norm.w': (
        f'{_VLLM_PREFIX}model.layers.*.self_attn.q_norm.weight',
        (None,),
    ),
    'layers.*.attn.k_norm.w': (
        f'{_VLLM_PREFIX}model.layers.*.self_attn.k_norm.weight',
        (None,),
    ),
    'layers.*.linear_attn.in_proj_qkvz.kernel': (
        f'{_VLLM_PREFIX}model.layers.*.linear_attn.in_proj_qkvz.weight',
        (None, 'model'),
    ),
    'layers.*.linear_attn.in_proj_ba.kernel': (
        f'{_VLLM_PREFIX}model.layers.*.linear_attn.in_proj_ba.weight',
        (None, 'model'),
    ),
    'layers.*.linear_attn.out_proj.kernel': (
        f'{_VLLM_PREFIX}model.layers.*.linear_attn.out_proj.weight',
        (None, None),
    ),
    'layers.*.linear_attn.conv1d_weight': (
        f'{_VLLM_PREFIX}model.layers.*.linear_attn.conv1d.weight',
        (None, None, None),
    ),
    'layers.*.linear_attn.dt_bias': (
        f'{_VLLM_PREFIX}model.layers.*.linear_attn.dt_bias',
        (None,),
    ),
    'layers.*.linear_attn.a_log': (
        f'{_VLLM_PREFIX}model.layers.*.linear_attn.A_log',
        (None,),
    ),
    'layers.*.linear_attn.norm.w': (
        f'{_VLLM_PREFIX}model.layers.*.linear_attn.norm.weight',
        (None,),
    ),
    'final_norm.w': (f'{_VLLM_PREFIX}model.norm.weight', (None,)),
    'lm_head.kernel': (f'{_VLLM_PREFIX}lm_head.weight', (None, 'model')),
}


def _lora_mappings() -> Dict[str, MappingEntry]:
  base = {
      'layers.*.mlp.gate_proj.kernel': (
          'language_model.model.layers.*.mlp.gate_proj.weight',
          (None, 'model'),
      ),
      'layers.*.mlp.up_proj.kernel': (
          'language_model.model.layers.*.mlp.up_proj.weight',
          (None, 'model'),
      ),
      'layers.*.mlp.down_proj.kernel': (
          'language_model.model.layers.*.mlp.down_proj.weight',
          ('model', None),
      ),
      'layers.*.attn.q_proj.kernel': (
          'language_model.model.layers.*.self_attn.q_proj.weight',
          (None, 'model'),
      ),
      'layers.*.attn.k_proj.kernel': (
          'language_model.model.layers.*.self_attn.k_proj.weight',
          (None, 'model'),
      ),
      'layers.*.attn.v_proj.kernel': (
          'language_model.model.layers.*.self_attn.v_proj.weight',
          (None, 'model'),
      ),
      'layers.*.attn.o_proj.kernel': (
          'language_model.model.layers.*.self_attn.o_proj.weight',
          ('model', None),
      ),
      'layers.*.linear_attn.in_proj_qkv.kernel': (
          'language_model.model.layers.*.linear_attn.in_proj_qkv.weight',
          (None, None),
      ),
      'layers.*.linear_attn.in_proj_z.kernel': (
          'language_model.model.layers.*.linear_attn.in_proj_z.weight',
          (None, None),
      ),
      'layers.*.linear_attn.in_proj_b.kernel': (
          'language_model.model.layers.*.linear_attn.in_proj_b.weight',
          (None, None),
      ),
      'layers.*.linear_attn.in_proj_a.kernel': (
          'language_model.model.layers.*.linear_attn.in_proj_a.weight',
          (None, None),
      ),
      'layers.*.linear_attn.out_proj.kernel': (
          'language_model.model.layers.*.linear_attn.out_proj.weight',
          (None, None),
      ),
  }
  mappings = {}
  for key, (target, sharding) in base.items():
    mappings[f'{key}_lora_a'] = (f'{target}_lora_a', sharding)
    mappings[f'{key}_lora_b'] = (f'{target}_lora_b', sharding)
  return mappings


def preprocess_src_state(src_state: Any) -> Any:
  """Fuse Qwen3.6 projections to match the vLLM Qwen3.5 runtime layout."""
  if not hasattr(src_state, 'flat_state'):
    return src_state

  linear_projection_re = re.compile(
      r'^layers\.([0-9]+)\.linear_attn\.'
      r'(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)\.kernel$'
  )
  attn_projection_re = re.compile(
      r'^layers\.([0-9]+)\.attn\.(q_proj|k_proj|v_proj)\.kernel$'
  )
  mlp_projection_re = re.compile(
      r'^layers\.([0-9]+)\.mlp\.(gate_proj|up_proj)\.kernel$'
  )
  linear_buckets = {}
  attn_buckets = {}
  mlp_buckets = {}
  new_flat_state = []
  for keys, param in src_state.flat_state():
    src_key = '.'.join(str(k) for k in keys)
    if '_lora_' in src_key:
      continue
    match = linear_projection_re.match(src_key)
    if match:
      layer_idx, proj_name = match.groups()
      linear_buckets.setdefault(layer_idx, {})[proj_name] = (keys, param)
      continue
    match = attn_projection_re.match(src_key)
    if match:
      layer_idx, proj_name = match.groups()
      attn_buckets.setdefault(layer_idx, {})[proj_name] = (keys, param)
      continue
    match = mlp_projection_re.match(src_key)
    if match:
      layer_idx, proj_name = match.groups()
      mlp_buckets.setdefault(layer_idx, {})[proj_name] = (keys, param)
      continue
    new_flat_state.append((keys, param))

  for layer_idx in sorted(linear_buckets, key=int):
    bucket = linear_buckets[layer_idx]
    if 'in_proj_qkv' in bucket and 'in_proj_z' in bucket:
      qkv_keys, qkv_param = bucket['in_proj_qkv']
      z_param = bucket['in_proj_z'][1]
      fused_keys = qkv_keys[:-2] + ('in_proj_qkvz', 'kernel')
      new_flat_state.append((fused_keys, _LazyConcatParam((qkv_param, z_param), -1)))
    else:
      for proj_name in ('in_proj_qkv', 'in_proj_z'):
        if proj_name in bucket:
          new_flat_state.append(bucket[proj_name])

    if 'in_proj_b' in bucket and 'in_proj_a' in bucket:
      b_keys, b_param = bucket['in_proj_b']
      a_param = bucket['in_proj_a'][1]
      fused_keys = b_keys[:-2] + ('in_proj_ba', 'kernel')
      new_flat_state.append((fused_keys, _LazyConcatParam((b_param, a_param), -1)))
    else:
      for proj_name in ('in_proj_b', 'in_proj_a'):
        if proj_name in bucket:
          new_flat_state.append(bucket[proj_name])

  for layer_idx in sorted(attn_buckets, key=int):
    bucket = attn_buckets[layer_idx]
    if all(proj_name in bucket for proj_name in ('q_proj', 'k_proj', 'v_proj')):
      q_keys, q_param = bucket['q_proj']
      fused_keys = q_keys[:-2] + ('qkv_proj', 'kernel')
      new_flat_state.append((
          fused_keys,
          _LazyConcatParam((q_param, bucket['k_proj'][1], bucket['v_proj'][1]), -1),
      ))
    else:
      for proj_name in ('q_proj', 'k_proj', 'v_proj'):
        if proj_name in bucket:
          new_flat_state.append(bucket[proj_name])

  for layer_idx in sorted(mlp_buckets, key=int):
    bucket = mlp_buckets[layer_idx]
    if 'gate_proj' in bucket and 'up_proj' in bucket:
      gate_keys, gate_param = bucket['gate_proj']
      fused_keys = gate_keys[:-2] + ('gate_up_proj', 'kernel')
      new_flat_state.append((
          fused_keys,
          _LazyConcatParam((gate_param, bucket['up_proj'][1]), -1),
      ))
    else:
      for proj_name in ('gate_proj', 'up_proj'):
        if proj_name in bucket:
          new_flat_state.append(bucket[proj_name])

  return _FlatState(new_flat_state)


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': _lora_mappings(),
    'to_hf_transpose_keys': {
        'lm_head.kernel': (1, 0),
        'layers.*.mlp.down_proj.kernel': (1, 0),
        'layers.*.attn.o_proj.kernel': (1, 0),
        'layers.*.linear_attn.out_proj.kernel': (1, 0),
        'layers.*.linear_attn.conv1d_weight': (2, 1, 0),
    },
    'lora_to_hf_transpose_keys': {
        'layers.*.mlp.gate_proj.kernel_lora_a': (1, 0),
        'layers.*.mlp.gate_proj.kernel_lora_b': (1, 0),
        'layers.*.mlp.up_proj.kernel_lora_a': (1, 0),
        'layers.*.mlp.up_proj.kernel_lora_b': (1, 0),
        'layers.*.mlp.down_proj.kernel_lora_a': (1, 0),
        'layers.*.mlp.down_proj.kernel_lora_b': (1, 0),
        'layers.*.attn.q_proj.kernel_lora_a': (1, 0),
        'layers.*.attn.q_proj.kernel_lora_b': (1, 0),
        'layers.*.attn.k_proj.kernel_lora_a': (1, 0),
        'layers.*.attn.k_proj.kernel_lora_b': (1, 0),
        'layers.*.attn.v_proj.kernel_lora_a': (1, 0),
        'layers.*.attn.v_proj.kernel_lora_b': (1, 0),
        'layers.*.attn.o_proj.kernel_lora_a': (1, 0),
        'layers.*.attn.o_proj.kernel_lora_b': (1, 0),
        'layers.*.linear_attn.in_proj_qkv.kernel_lora_a': (1, 0),
        'layers.*.linear_attn.in_proj_qkv.kernel_lora_b': (1, 0),
        'layers.*.linear_attn.in_proj_z.kernel_lora_a': (1, 0),
        'layers.*.linear_attn.in_proj_z.kernel_lora_b': (1, 0),
        'layers.*.linear_attn.in_proj_b.kernel_lora_a': (1, 0),
        'layers.*.linear_attn.in_proj_b.kernel_lora_b': (1, 0),
        'layers.*.linear_attn.in_proj_a.kernel_lora_a': (1, 0),
        'layers.*.linear_attn.in_proj_a.kernel_lora_b': (1, 0),
        'layers.*.linear_attn.out_proj.kernel_lora_a': (1, 0),
        'layers.*.linear_attn.out_proj.kernel_lora_b': (1, 0),
    },
    'to_hf_hook_fns': TO_HF_HOOK_FNS,
    'preprocess_src_state': preprocess_src_state,
}


__all__ = ['VLLM_JAX_MAPPING']
