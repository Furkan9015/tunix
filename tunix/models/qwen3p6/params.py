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

"""Utils for loading Qwen3.6 safetensors weights."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models.qwen3p6 import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Mapping of Hugging Face Qwen3.6 keys to Tunix NNX keys."""
  return {
      r'model\.language_model\.embed_tokens\.weight': (
          'embedder.input_embedding',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.input_layernorm\.weight': (
          r'layers.\1.input_layernorm.w',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.post_attention_layernorm\.weight': (
          r'layers.\1.post_attention_layernorm.w',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight': (
          r'layers.\1.mlp.gate_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.mlp\.up_proj\.weight': (
          r'layers.\1.mlp.up_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.mlp\.down_proj\.weight': (
          r'layers.\1.mlp.down_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight': (
          r'layers.\1.attn.q_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight': (
          r'layers.\1.attn.k_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight': (
          r'layers.\1.attn.v_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight': (
          r'layers.\1.attn.o_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight': (
          r'layers.\1.attn.q_norm.w',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight': (
          r'layers.\1.attn.k_norm.w',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_qkv\.weight': (
          r'layers.\1.linear_attn.in_proj_qkv.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_z\.weight': (
          r'layers.\1.linear_attn.in_proj_z.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_b\.weight': (
          r'layers.\1.linear_attn.in_proj_b.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_a\.weight': (
          r'layers.\1.linear_attn.in_proj_a.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.out_proj\.weight': (
          r'layers.\1.linear_attn.out_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.conv1d\.weight': (
          r'layers.\1.linear_attn.conv1d_weight',
          ((2, 1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.dt_bias': (
          r'layers.\1.linear_attn.dt_bias',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.A_log': (
          r'layers.\1.linear_attn.a_log',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.norm\.weight': (
          r'layers.\1.linear_attn.norm.w',
          None,
      ),
      r'model\.language_model\.norm\.weight': ('final_norm.w', None),
      r'lm_head\.weight': ('lm_head.kernel', ((1, 0), None)),
  }


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = 'auto',
) -> model_lib.Qwen3P6:
  """Load tensors from safetensors and create a Qwen3.6 text model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Qwen3P6,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      dtype=dtype,
      mode=mode,
  )
