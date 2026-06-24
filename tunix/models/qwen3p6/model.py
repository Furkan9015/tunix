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

"""Qwen3.6 text-only model.

Qwen3.6 checkpoints use the Qwen3.5 hybrid language-model architecture:
Gated DeltaNet layers with every fourth layer using gated full attention.
This module follows Tunix's native model interface so RL code can call the
model as `(tokens, positions, cache, attention_mask) -> (logits, cache)`.
"""

from __future__ import annotations

import dataclasses
import enum
from functools import partial
import math
from typing import Tuple

import flax
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.shard_map import shard_map
from jax.interpreters import pxla
import jax.sharding as shd
from jax.sharding import PartitionSpec as P
import jaxtyping
from tunix.generate.mappings import BackendMappingMixin
from tunix.utils import compat
from tunix.utils import env_utils

env_utils.setup_sharding_environment()


K_MASK = -2.3819763e38

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class RematConfig(enum.Enum):
  NONE = enum.auto()
  BLOCK = enum.auto()
  DECODER = enum.auto()


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  """Sharding configuration for Qwen3.6."""

  emb_vd: Tuple[str | None, ...]
  emb_dv: Tuple[str | None, ...]
  dense_df: Tuple[str | None, ...]
  dense_fd: Tuple[str | None, ...]
  rms_norm_weight: Tuple[str | None, ...]
  act_btd: Tuple[str | None, ...]
  act_btf: Tuple[str | None, ...]
  act_btnh: Tuple[str | None, ...]
  conv_weight: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False, enable_sp: bool = False):
    fsdp = 'fsdp' if not is_sampling else None
    sp = 'sp' if (not is_sampling and enable_sp) else None
    fsdp = (fsdp, sp) if fsdp and sp else fsdp

    return ShardingConfig(
        emb_vd=('tp', fsdp),
        emb_dv=(fsdp, 'tp'),
        dense_df=(fsdp, 'tp'),
        dense_fd=('tp', fsdp),
        rms_norm_weight=(None,),
        act_btd=('fsdp', sp, None if is_sampling else 'tp'),
        act_btf=('fsdp', sp, 'tp'),
        act_btnh=('fsdp', sp, 'tp', None),
        conv_weight=(None, None, 'tp'),
    )


@dataclasses.dataclass(slots=True)
class ModelConfig:
  """Configuration for the Qwen3.6 language model."""

  num_layers: int
  vocab_size: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  rope_theta: int
  norm_eps: float
  layer_types: tuple[str, ...]
  linear_num_value_heads: int
  linear_num_key_heads: int
  linear_key_head_dim: int
  linear_value_head_dim: int
  linear_conv_kernel_dim: int = 4
  linear_attention_chunk_size: int = 64
  partial_rotary_factor: float = 0.25
  use_tied_embedding: bool = False
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE
  remat_policy: str | None = None
  logits_chunk_size: int = 0
  enable_sequence_parallel: bool = False
  use_flash_attention: bool = False
  flash_attention_block_size: int = 1024
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @classmethod
  def qwen3p6_27b(cls):
    layer_types = tuple(
        'linear_attention' if (i + 1) % 4 else 'full_attention'
        for i in range(64)
    )
    return cls(
        num_layers=64,
        vocab_size=248320,
        embed_dim=5120,
        hidden_dim=17408,
        num_heads=24,
        head_dim=256,
        num_kv_heads=4,
        rope_theta=10_000_000,
        norm_eps=1e-06,
        layer_types=layer_types,
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        use_tied_embedding=False,
    )


def _with_sequence_parallel_sharding(config: ModelConfig) -> ModelConfig:
  if not config.enable_sequence_parallel:
    return config
  return dataclasses.replace(
      config,
      shd_config=ShardingConfig.get_default_sharding(enable_sp=True),
  )


def shard(x: jnp.ndarray, s: Tuple[str | None, ...]):
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  if (
      s
      and isinstance(s[0], str)
      and s[0] in mesh.shape
      and x.shape[0] % mesh.shape[s[0]]
  ):
    # Actor/ref logprob paths often run with microbatch 1. Keep the batch
    # dimension unsharded in that case, while still sharding sequence/hidden.
    s = (None, *s[1:])
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )


class Embedder(nnx.Module):
  """Embedding module."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig,
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.input_embedding = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(
            rngs.params(), (vocab_size, embed_dim)
        ),
        sharding=shd_config.emb_vd,
    )
    self.shd_config = shd_config
    self.dtype = dtype

  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = self.input_embedding[(x,)]
    x = jnp.astype(x, self.dtype)
    return shard(x, self.shd_config.act_btd)

  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(self.input_embedding.value, self.dtype)
    return jnp.dot(x, w.T)


class Dense(nnx.Module):
  """Linear layer with explicit partitioned kernel."""

  def __init__(
      self,
      in_features: int,
      out_features: int,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
      use_bias: bool = False,
  ):
    self.kernel = nnx.Param(
        nnx.initializers.zeros_init()(
            rngs.params(), (in_features, out_features), param_dtype
        ),
        sharding=sharding,
    )
    self.bias = None
    if use_bias:
      self.bias = nnx.Param(
          nnx.initializers.zeros_init()(rngs.params(), (out_features,), param_dtype),
          sharding=(None,),
      )
    self.dtype = dtype

  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = jnp.astype(x, self.dtype)
    y = jnp.einsum('...d,df->...f', x, jnp.astype(self.kernel.value, self.dtype))
    if self.bias is not None:
      y = y + jnp.astype(self.bias.value, self.dtype)
    return y


def apply_rope(
    inputs: jaxtyping.Array,
    positions: jaxtyping.Array,
    head_dim: int,
    rope_theta: int,
) -> jaxtyping.Array:
  fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
  timescale = rope_theta**fraction
  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  sin = jnp.sin(sinusoid_inp).astype(inputs.dtype)
  cos = jnp.cos(sinusoid_inp).astype(inputs.dtype)
  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  return jnp.concatenate(
      [first_half * cos - second_half * sin, second_half * cos + first_half * sin],
      axis=-1,
  ).astype(inputs.dtype)


def apply_partial_rope(
    q: jaxtyping.Array,
    k: jaxtyping.Array,
    positions: jaxtyping.Array,
    rotary_dim: int,
    rope_theta: int,
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
  if rotary_dim <= 0:
    return q, k
  q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
  k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
  q_rot = apply_rope(q_rot, positions, rotary_dim, rope_theta)
  k_rot = apply_rope(k_rot, positions, rotary_dim, rope_theta)
  return jnp.concatenate([q_rot, q_pass], axis=-1), jnp.concatenate(
      [k_rot, k_pass], axis=-1
  )


class RMSNorm(nnx.Module):
  """Qwen3.5/3.6 RMSNorm with one-plus scale parameterization."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig,
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.w = nnx.Param(
        nnx.initializers.zeros_init()(rngs.params(), dim, param_dtype),
        sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps
    self.dtype = dtype

  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    x_float = jnp.astype(x, jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_float * x_float, axis=-1, keepdims=True) + self.norm_eps)
    scale = 1.0 + jnp.astype(self.w.value, jnp.float32)
    return jnp.astype(scale * (x_float / rms), self.dtype)


class RMSNormGated(nnx.Module):
  """RMSNorm followed by a SiLU gate."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig,
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.w = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), dim, param_dtype),
        sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps
    self.dtype = dtype

  def __call__(self, hidden_states: jaxtyping.Array, gate: jaxtyping.Array):
    hidden_float = jnp.astype(hidden_states, jnp.float32)
    rms = jnp.sqrt(jnp.mean(hidden_float * hidden_float, axis=-1, keepdims=True) + self.norm_eps)
    out = hidden_float / rms
    out = out.astype(self.dtype) * jnp.astype(self.w.value, self.dtype)
    return out * nnx.silu(gate)


def l2norm(x: jaxtyping.Array, axis: int = -1, eps: float = 1e-6):
  inv_norm = jax.lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
  return x * inv_norm


def apply_mask_to_padding_states(
    x: jaxtyping.Array, attention_mask: jaxtyping.Array | None
):
  if attention_mask is None:
    return x
  if attention_mask.ndim == 3:
    seq_len = x.shape[1]
    mask = jnp.diagonal(attention_mask[:, :, :seq_len], axis1=1, axis2=2)
  else:
    mask = attention_mask
  return x * mask[..., None].astype(x.dtype)


def recurrent_gated_delta_rule(
    query: jaxtyping.Array,
    key: jaxtyping.Array,
    value: jaxtyping.Array,
    g: jaxtyping.Array,
    beta: jaxtyping.Array,
    initial_state: jaxtyping.Array | None = None,
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
  dtype = query.dtype
  query = l2norm(query, axis=-1)
  key = l2norm(key, axis=-1)
  query = query * (1.0 / math.sqrt(query.shape[-1]))

  query = jnp.swapaxes(query, 0, 1)
  key = jnp.swapaxes(key, 0, 1)
  value = jnp.swapaxes(value, 0, 1)
  g = jnp.swapaxes(g, 0, 1)
  beta = jnp.swapaxes(beta, 0, 1)

  batch_size = query.shape[1]
  num_heads = query.shape[2]
  k_head_dim = query.shape[3]
  v_head_dim = value.shape[3]
  if initial_state is None:
    initial_state = jnp.zeros(
        (batch_size, num_heads, k_head_dim, v_head_dim), dtype=dtype
    )
  else:
    initial_state = initial_state.astype(dtype)

  def step_fn(state, inputs):
    q_t, k_t, v_t, g_t, beta_t = inputs
    decay = jnp.exp(g_t).astype(dtype)[..., None, None]
    state = state * decay
    kv_mem = jnp.sum(state * k_t[..., :, None], axis=-2)
    delta = (v_t - kv_mem) * beta_t[..., None]
    state = state + k_t[..., :, None] * delta[..., None, :]
    out_t = jnp.sum(state * q_t[..., :, None], axis=-2)
    return state, out_t

  # The recurrent path is differentiated in short-cache tests and generation
  # utilities; remat keeps scan-body residuals from being retained across time.
  remat_step_fn = jax.remat(step_fn, prevent_cse=False)
  final_state, outputs = jax.lax.scan(
      remat_step_fn, initial_state, (query, key, value, g, beta)
  )
  return jnp.swapaxes(outputs, 0, 1).astype(dtype), final_state.astype(dtype)


def chunk_gated_delta_rule(
    query: jaxtyping.Array,
    key: jaxtyping.Array,
    value: jaxtyping.Array,
    g: jaxtyping.Array,
    beta: jaxtyping.Array,
    chunk_size: int = 64,
    initial_state: jaxtyping.Array | None = None,
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
  dtype = query.dtype
  query = l2norm(query, axis=-1)
  key = l2norm(key, axis=-1)
  query = jnp.transpose(query, (0, 2, 1, 3))
  key = jnp.transpose(key, (0, 2, 1, 3))
  value = jnp.transpose(value, (0, 2, 1, 3))
  beta = jnp.transpose(beta, (0, 2, 1))
  g = jnp.transpose(g, (0, 2, 1))

  batch_size, num_heads, seq_len, k_head_dim = key.shape
  v_head_dim = value.shape[-1]
  query = query * (1.0 / math.sqrt(k_head_dim))

  pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
  if pad_size > 0:
    query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
    g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))
  total_seq_len = seq_len + pad_size
  num_chunks = total_seq_len // chunk_size

  query = query.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  key = key.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  value = value.reshape(batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
  beta = beta.reshape(batch_size, num_heads, num_chunks, chunk_size)
  g = g.reshape(batch_size, num_heads, num_chunks, chunk_size)

  k_beta = key * beta[..., None]
  v_beta = value * beta[..., None]
  g_cumsum = jnp.cumsum(g, axis=-1)
  gamma = jnp.exp(g_cumsum).astype(dtype)
  decay_mask = jnp.tril(
      jnp.exp(jnp.tril(g_cumsum[..., :, None] - g_cumsum[..., None, :]))
  ).astype(dtype)
  lower = jnp.tril((k_beta @ jnp.swapaxes(key, -1, -2)) * decay_mask, k=-1)
  rhs = jnp.concatenate([v_beta, k_beta * gamma[..., None]], axis=-1)
  solution = jax.lax.linalg.triangular_solve(
      lower, rhs, left_side=True, lower=True, unit_diagonal=True
  )
  u = solution[..., :v_head_dim]
  w_decay = solution[..., v_head_dim:]

  if initial_state is None:
    state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=dtype)
  else:
    state = initial_state.astype(dtype)

  def chunk_step(s, inputs):
    q_t, k_t, u_t, w_decay_t, gamma_t, decay_mask_t = inputs
    intra_attn = q_t @ jnp.swapaxes(k_t, -1, -2) * decay_mask_t
    u_minus_w_s = u_t - w_decay_t @ s
    inter_out = (q_t * gamma_t[..., None]) @ s
    out_t = inter_out + intra_attn @ u_minus_w_s
    gamma_c = gamma_t[..., -1, None, None]
    key_decay_t = decay_mask_t[..., -1, :][..., None]
    s = gamma_c * s + jnp.swapaxes(k_t * key_decay_t, -1, -2) @ u_minus_w_s
    return s, out_t

  scan_inputs = (
      jnp.transpose(query, (2, 0, 1, 3, 4)),
      jnp.transpose(key, (2, 0, 1, 3, 4)),
      jnp.transpose(u, (2, 0, 1, 3, 4)),
      jnp.transpose(w_decay, (2, 0, 1, 3, 4)),
      jnp.transpose(gamma, (2, 0, 1, 3)),
      jnp.transpose(decay_mask, (2, 0, 1, 3, 4)),
  )
  # Outer decoder remat removes residuals between layers, but the scan
  # transpose can still retain per-chunk body residuals for the long
  # GatedDeltaNet sequence path. Rematerialize the chunk body; CSE prevention is
  # unnecessary inside a rolled scan and can make backend optimization harder.
  remat_chunk_step = jax.remat(chunk_step, prevent_cse=False)
  final_state, outputs = jax.lax.scan(remat_chunk_step, state, scan_inputs)
  outputs = jnp.transpose(outputs, (1, 2, 0, 3, 4))
  outputs = outputs.reshape(batch_size, num_heads, total_seq_len, v_head_dim)
  outputs = jnp.transpose(outputs[:, :, :seq_len, :], (0, 2, 1, 3)).astype(dtype)
  return outputs, final_state.astype(dtype)


class Attention(nnx.Module):
  """Gated full attention layer used every fourth Qwen3.6 layer."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.shd_config = config.shd_config
    self.num_heads = config.num_heads
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    self.rotary_dim = min(config.head_dim, rotary_dim) // 2 * 2
    self.scale = self.head_dim**-0.5

    self.q_proj = Dense(
        config.embed_dim,
        2 * config.num_heads * config.head_dim,
        rngs=rngs,
        sharding=self.shd_config.dense_df,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.k_proj = Dense(
        config.embed_dim,
        config.num_kv_heads * config.head_dim,
        rngs=rngs,
        sharding=self.shd_config.dense_df,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.v_proj = Dense(
        config.embed_dim,
        config.num_kv_heads * config.head_dim,
        rngs=rngs,
        sharding=self.shd_config.dense_df,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.o_proj = Dense(
        config.num_heads * config.head_dim,
        config.embed_dim,
        rngs=rngs,
        sharding=self.shd_config.dense_fd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.q_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.k_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def __call__(
      self,
      x: jaxtyping.Array,
      positions: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    batch_size, seq_len, _ = x.shape
    q_raw = self.q_proj(x)
    q_all = q_raw.reshape(batch_size, seq_len, self.num_heads, self.head_dim * 2)
    query_proj, gate = jnp.split(q_all, 2, axis=-1)
    gate = gate.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

    key_proj = self.k_proj(x).reshape(
        batch_size, seq_len, self.num_kv_heads, self.head_dim
    )
    value_proj = self.v_proj(x).reshape(
        batch_size, seq_len, self.num_kv_heads, self.head_dim
    )
    query_proj = self.q_norm(query_proj)
    key_proj = self.k_norm(key_proj)
    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)

    query_proj, key_proj = apply_partial_rope(
        query_proj, key_proj, positions, self.rotary_dim, self.config.rope_theta
    )

    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(cache['v'], value_proj, slice_indices)
      key_proj = jax.lax.dynamic_update_slice(cache['k'], key_proj, slice_indices)
      cache_value_proj = value_proj
      cache_key_proj = key_proj
    else:
      cache_value_proj = value_proj
      cache_key_proj = key_proj

    b, t, qh, d = query_proj.shape
    _, s, kh, _ = key_proj.shape
    del s

    # The training path for Qwen3.6 can see long 32k-token sequences. The
    # dense fallback below materializes BHGTS attention logits, which is not
    # viable at that length. Use TPU Splash Attention when enabled; it supports
    # grouped-query attention as long as local Q heads are divisible by local KV
    # heads, so the recommended v6e-8 mesh is fsdp=2,tp=4 for Qwen3.6-27B.
    mesh = pxla.thread_resources.env.physical_mesh
    if (
        self.config.use_flash_attention
        and seq_len > 1
        and not mesh.empty
        and jax.devices()[0].platform == 'tpu'
    ):
      query_splash = query_proj.transpose(0, 2, 1, 3) * self.scale
      key_splash = key_proj.transpose(0, 2, 1, 3)
      value_splash = value_proj.transpose(0, 2, 1, 3)

      causal_mask = mask_lib.CausalMask((seq_len, seq_len))
      multi_head_mask = mask_lib.MultiHeadMask([causal_mask for _ in range(qh)])

      block = self.config.flash_attention_block_size
      block_sizes = splash.BlockSizes(
          block_q=block,
          block_kv=block,
          block_q_dkv=block,
          block_kv_dkv=block,
          block_kv_dkv_compute=block,
          block_q_dq=block,
          block_kv_dq=block,
      )

      shd_b, shd_t, shd_n, shd_h = self.shd_config.act_btnh
      head_shards = (
          mesh.shape[shd_n] if shd_n is not None and shd_n in mesh.shape else 1
      )
      q_seq_shards = (
          mesh.shape[shd_t] if shd_t is not None and shd_t in mesh.shape else 1
      )

      splash_attn_kernel = splash.make_splash_mha(
          multi_head_mask,
          block_sizes=block_sizes,
          head_shards=head_shards,
          q_seq_shards=q_seq_shards,
      )

      # The GRPO reference/actor logprob paths commonly run with microbatch 1
      # even when the mesh has an fsdp axis of size 2. shard_map requires every
      # sharded array axis to be evenly divisible by its mesh axis, so shard the
      # batch axis only when the current local batch can actually be split.
      splash_shd_b = shd_b
      if shd_b is not None and shd_b in mesh.shape and b % mesh.shape[shd_b]:
        splash_shd_b = None

      shd_spec = P(splash_shd_b, shd_n, shd_t, shd_h)
      unsharded_seq = P(splash_shd_b, shd_n, None, shd_h)
      kernel_spec = splash_attn_kernel.manual_sharding_spec(
          shd.NamedSharding(mesh, P(shd_n, shd_t))
      )

      @partial(
          shard_map,
          mesh=mesh,
          in_specs=(kernel_spec, shd_spec, unsharded_seq, unsharded_seq),
          out_specs=shd_spec,
          check_rep=False,
      )
      def sharded_splash_attn(kernel, q_block, k_block, v_block):
        return jax.vmap(kernel)(q_block, k_block, v_block)

      attn_output = sharded_splash_attn(
          splash_attn_kernel, query_splash, key_splash, value_splash
      )
      attn_output = attn_output.transpose(0, 2, 1, 3)
      attn_output = attn_output.reshape((b, t, qh * d))
    else:
      query_proj = query_proj.reshape((b, t, kh, qh // kh, d))
      attn = jnp.einsum('BTHGD,BSHD->BHGTS', query_proj, key_proj) * self.scale
      if attn_mask is not None:
        attn = jnp.where(attn_mask[:, None, None, :, :], attn, K_MASK)
      attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
          key_proj.dtype
      )
      attn_output = jnp.einsum('BHGTS,BSHD->BTHGD', attn, value_proj)
      attn_output = attn_output.reshape((b, t, qh * d))

    attn_output = attn_output * nnx.sigmoid(gate)
    outputs = self.o_proj(attn_output)
    outputs = shard(outputs, self.shd_config.act_btd)

    if cache is not None:
      new_cache = {
          'v': cache_value_proj,
          'k': cache_key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None
    return new_cache, outputs


class GatedDeltaNet(nnx.Module):
  """Gated DeltaNet linear-attention layer."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.shd_config = config.shd_config
    self.hidden_size = config.embed_dim
    self.num_v_heads = config.linear_num_value_heads
    self.num_k_heads = config.linear_num_key_heads
    self.head_k_dim = config.linear_key_head_dim
    self.head_v_dim = config.linear_value_head_dim
    self.key_dim = self.head_k_dim * self.num_k_heads
    self.value_dim = self.head_v_dim * self.num_v_heads
    self.conv_kernel_size = config.linear_conv_kernel_dim
    self.conv_dim = self.key_dim * 2 + self.value_dim

    self.in_proj_qkv = Dense(
        self.hidden_size,
        self.conv_dim,
        rngs=rngs,
        sharding=self.shd_config.dense_df,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.in_proj_z = Dense(
        self.hidden_size,
        self.value_dim,
        rngs=rngs,
        sharding=self.shd_config.dense_df,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.in_proj_b = Dense(
        self.hidden_size,
        self.num_v_heads,
        rngs=rngs,
        sharding=self.shd_config.dense_df,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.in_proj_a = Dense(
        self.hidden_size,
        self.num_v_heads,
        rngs=rngs,
        sharding=self.shd_config.dense_df,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.conv1d_weight = nnx.Param(
        nnx.initializers.zeros_init()(
            rngs.params(),
            (self.conv_kernel_size, 1, self.conv_dim),
            config.param_dtype,
        ),
        sharding=self.shd_config.conv_weight,
    )
    self.dt_bias = nnx.Param(
        nnx.initializers.zeros_init()(rngs.params(), (self.num_v_heads,), config.param_dtype),
        sharding=(self.shd_config.dense_df[1],),
    )
    self.a_log = nnx.Param(
        nnx.initializers.zeros_init()(rngs.params(), (self.num_v_heads,), config.param_dtype),
        sharding=(self.shd_config.dense_df[1],),
    )
    self.norm = RMSNormGated(
        self.head_v_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.out_proj = Dense(
        self.value_dim,
        self.hidden_size,
        rngs=rngs,
        sharding=self.shd_config.dense_fd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def _get_conv_kernel(self):
    return self.conv1d_weight.value.transpose((2, 1, 0))

  def _causal_conv(self, x, conv_state=None):
    kernel = jnp.astype(self._get_conv_kernel(), x.dtype)
    seq_len = x.shape[-1]
    if conv_state is None:
      x_full = jnp.pad(x, ((0, 0), (0, 0), (self.conv_kernel_size - 1, 0)))
    else:
      x_full = jnp.concatenate([conv_state, x], axis=-1)
    new_state = x_full[..., -self.conv_kernel_size :]
    out_full = jax.lax.conv_general_dilated(
        x_full,
        kernel,
        window_strides=(1,),
        padding='VALID',
        feature_group_count=self.conv_dim,
        dimension_numbers=('NCH', 'OIH', 'NCH'),
    )
    return nnx.silu(out_full[..., -seq_len:]), new_state

  def __call__(
      self,
      hidden_states: jaxtyping.Array,
      *,
      attention_mask: jaxtyping.Array | None,
      conv_state: jaxtyping.Array | None = None,
      recurrent_state: jaxtyping.Array | None = None,
  ) -> tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array]:
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    if conv_state is not None:
      assert recurrent_state is not None
      assert seq_len == 1, f'conv_state only supports seq_len == 1, got {seq_len}.'

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose((0, 2, 1))
    z = self.in_proj_z(hidden_states).reshape(
        batch_size, seq_len, -1, self.head_v_dim
    )
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    mixed_qkv, new_conv_state = self._causal_conv(mixed_qkv, conv_state)
    mixed_qkv = mixed_qkv.transpose((0, 2, 1))

    q_end = self.key_dim
    k_end = self.key_dim * 2
    query = mixed_qkv[..., :q_end].reshape(
        batch_size, seq_len, -1, self.head_k_dim
    )
    key = mixed_qkv[..., q_end:k_end].reshape(
        batch_size, seq_len, -1, self.head_k_dim
    )
    value = mixed_qkv[..., k_end:].reshape(
        batch_size, seq_len, -1, self.head_v_dim
    )

    beta = nnx.sigmoid(b)
    g = -jnp.exp(self.a_log.value.astype(jnp.float32)) * jax.nn.softplus(
        a.astype(jnp.float32) + self.dt_bias.value.astype(jnp.float32)
    )
    if self.num_v_heads // self.num_k_heads > 1:
      repeats = self.num_v_heads // self.num_k_heads
      query = jnp.repeat(query, repeats, axis=2)
      key = jnp.repeat(key, repeats, axis=2)

    if seq_len > 1:
      core_out, new_recurrent_state = chunk_gated_delta_rule(
          query,
          key,
          value,
          g,
          beta,
          chunk_size=self.config.linear_attention_chunk_size,
          initial_state=recurrent_state,
      )
    else:
      core_out, new_recurrent_state = recurrent_gated_delta_rule(
          query, key, value, g, beta, recurrent_state
      )
    core_out = self.norm(core_out, z).reshape(batch_size, seq_len, -1)
    out = self.out_proj(core_out)
    out = shard(out, self.shd_config.act_btd)
    return out, new_conv_state, new_recurrent_state


class MLP(nnx.Module):
  """Feed-forward block."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.shd_config = config.shd_config
    self.gate_proj = Dense(
        config.embed_dim,
        config.hidden_dim,
        rngs=rngs,
        sharding=self.shd_config.dense_df,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.up_proj = Dense(
        config.embed_dim,
        config.hidden_dim,
        rngs=rngs,
        sharding=self.shd_config.dense_df,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.down_proj = Dense(
        config.hidden_dim,
        config.embed_dim,
        rngs=rngs,
        sharding=self.shd_config.dense_fd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
    activations = shard(activations, self.shd_config.act_btf)
    return self.down_proj(activations)


class DecoderLayer(nnx.Module):
  """Hybrid Qwen3.6 decoder layer."""

  def __init__(self, config: ModelConfig, layer_idx: int, *, rngs: nnx.Rngs):
    self.config = config
    self.layer_type = config.layer_types[layer_idx]
    self.input_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.post_attention_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    if self.layer_type == 'linear_attention':
      self.linear_attn = GatedDeltaNet(config=config, rngs=rngs)
    else:
      self.attn = Attention(config=config, rngs=rngs)
    self.mlp = MLP(config=config, rngs=rngs)

  def block(
      self,
      x: jaxtyping.Array,
      positions: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if self.config.remat_policy == 'offload_decoder_input' and cache is None:
      x = checkpoint_name(x, 'decoder_block_input')
    residual = x
    hidden_states = self.input_layernorm(x)
    if self.layer_type == 'linear_attention':
      use_recurrent_cache = cache is not None and hidden_states.shape[1] == 1
      conv_state = cache['conv'] if use_recurrent_cache else None
      recurrent_state = cache['recurrent'] if use_recurrent_cache else None
      hidden_states, new_conv_state, new_recurrent_state = self.linear_attn(
          hidden_states,
          attention_mask=None if use_recurrent_cache else attn_mask,
          conv_state=conv_state,
          recurrent_state=recurrent_state,
      )
      new_cache = (
          {'conv': new_conv_state, 'recurrent': new_recurrent_state}
          if cache is not None
          else None
      )
    else:
      new_cache, hidden_states = self.attn(hidden_states, positions, cache, attn_mask)

    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    return new_cache, residual + hidden_states

  def __call__(
      self,
      x: jaxtyping.Array,
      positions: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if (
        self.config.remat_config == RematConfig.DECODER
        or self.config.remat_config == RematConfig.DECODER.value
    ):
      policy = None
      if self.config.remat_policy == 'offload_decoder_input':
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=(),
            names_which_can_be_offloaded=('decoder_block_input',),
            offload_src='device',
            offload_dst='pinned_host',
        )
      elif self.config.remat_policy == 'offload_dots':
        policy = jax.checkpoint_policies.offload_dot_with_no_batch_dims(
            'device',
            'pinned_host',
        )
      return nnx.remat(
          self.block.__func__, prevent_cse=False, policy=policy
      )(self, x, positions, cache, attn_mask)
    return self.block(x, positions, cache, attn_mask)


class Qwen3P6(BackendMappingMixin, nnx.Module):
  """Qwen3.6 text-only causal language model."""

  BACKEND_PACKAGE_PATH = __name__

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    config = _with_sequence_parallel_sharding(config)
    self.config = config
    self.embedder = Embedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.layers = compat.ModuleList([
        DecoderLayer(config=config, layer_idx=i, rngs=rngs)
        for i in range(config.num_layers)
    ])
    self.final_norm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    if not config.use_tied_embedding:
      self.lm_head = Dense(
          config.embed_dim,
          config.vocab_size,
          rngs=rngs,
          sharding=config.shd_config.emb_dv,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

  def init_cache(self, batch_size: int, cache_size: int, dtype: jnp.dtype) -> Cache:
    del dtype
    config = self.config
    full_shape = (batch_size, cache_size, config.num_kv_heads, config.head_dim)
    k = jnp.zeros(full_shape, dtype=config.dtype)
    v = jnp.zeros(full_shape, dtype=config.dtype)
    end_index = jnp.zeros((batch_size,), dtype=jnp.int32)
    linear_conv_dim = (
        2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim
    )
    conv = jnp.zeros(
        (batch_size, linear_conv_dim, config.linear_conv_kernel_dim),
        dtype=config.dtype,
    )
    recurrent = jnp.zeros(
        (
            batch_size,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        ),
        dtype=config.dtype,
    )
    caches = {}
    for i, layer_type in enumerate(config.layer_types):
      if layer_type == 'linear_attention':
        caches[f'layer_{i}'] = {'conv': conv, 'recurrent': recurrent}
      else:
        caches[f'layer_{i}'] = {'k': k, 'v': v, 'end_index': end_index}
    return caches

  def __call__(
      self,
      input_tokens: jaxtyping.Array,
      positions: jaxtyping.Array,
      cache: Cache | None,
      attention_mask: jaxtyping.Array,
      output_hidden_states: bool = False,
      return_hidden_states: bool = False,
      skip_lm_head: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    new_cache = None if cache is None else {}
    x = self.embedder.encode(input_tokens)
    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(x, positions, layer_cache, attention_mask)
      if cache is not None:
        new_cache[layer_name] = layer_cache

    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)
    if return_hidden_states or skip_lm_head:
      return x, new_cache
    logits = self.compute_final_logits(x)
    return jnp.astype(logits, jnp.float32), new_cache

  def compute_final_logits(
      self, hidden_states: jaxtyping.ArrayLike
  ) -> jaxtyping.Array:
    if self.config.use_tied_embedding:
      return self.embedder.decode(hidden_states)
    return self.lm_head(hidden_states)

  def selective_log_softmax_from_hidden(
      self,
      hidden_states: jaxtyping.Array,
      target_tokens: jaxtyping.Array,
      temperature: float = 1.0,
  ) -> jaxtyping.Array:
    """Computes selected-token logprobs without retaining full-sequence logits."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    chunk_size = self.config.logits_chunk_size
    if chunk_size <= 0:
      chunk_size = seq_len
    if seq_len % chunk_size:
      raise ValueError(
          'logits_chunk_size must divide the selected sequence length; got '
          f'{chunk_size=} for {seq_len=}.'
      )
    num_chunks = seq_len // chunk_size
    hidden_chunks = hidden_states.reshape(
        batch_size, num_chunks, chunk_size, hidden_dim
    )
    hidden_chunks = jnp.swapaxes(hidden_chunks, 0, 1)
    target_chunks = target_tokens.reshape(batch_size, num_chunks, chunk_size)
    target_chunks = jnp.swapaxes(target_chunks, 0, 1)

    def chunk_logps(args):
      hidden_chunk, target_chunk = args
      if self.config.use_tied_embedding:
        logits = self.embedder.decode(hidden_chunk)
      else:
        logits = self.lm_head(hidden_chunk)
      if temperature != 0.0 and temperature != 1.0:
        logits /= temperature
      target_logits = (
          jnp.take_along_axis(logits, target_chunk[..., None], axis=-1)
          .squeeze(-1)
          .astype(jnp.float32)
      )
      normalizer = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)
      return target_logits - normalizer

    # Rematerialize each chunk's vocab logits. Without this, the backward pass
    # through jax.lax.map (a scan) stores every chunk's [chunk, vocab] logits
    # simultaneously -- with vocab=248320/tp and ~252 chunks at 32k this was the
    # single largest live buffer (~7.46 GiB in f32, measured via XLA
    # buffer-assignment). nothing_saveable recomputes one chunk's logits at a
    # time in the backward, so peak drops to a single [chunk, vocab] tile.
    remat_chunk_logps = jax.checkpoint(
        chunk_logps, policy=jax.checkpoint_policies.nothing_saveable
    )
    logps = jax.lax.map(remat_chunk_logps, (hidden_chunks, target_chunks))
    return jnp.swapaxes(logps, 0, 1).reshape(batch_size, seq_len)

  def compute_per_token_logps(
      self,
      input_tokens: jaxtyping.Array,
      positions: jaxtyping.Array,
      attention_mask: jaxtyping.Array,
      logits_to_keep: int,
      temperature: float = 1.0,
  ) -> jaxtyping.Array:
    hidden_states, _ = self(
        input_tokens,
        positions=positions,
        cache=None,
        attention_mask=attention_mask,
        return_hidden_states=True,
    )
    hidden_states = hidden_states[:, -logits_to_keep - 1 : -1, :]
    target_tokens = input_tokens[:, -logits_to_keep:]
    return self.selective_log_softmax_from_hidden(
        hidden_states, target_tokens, temperature=temperature
    )

  def get_model_input(self):
    dummy_batch_size = 2
    dummy_seq_len = 1
    return {
        'input_tokens': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'positions': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'cache': None,
        'attention_mask': jnp.ones(
            (dummy_batch_size, 1, dummy_seq_len), dtype=jnp.bool
        ),
    }

  @property
  def num_embed(self) -> int:
    return self.config.vocab_size
