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

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.models.qwen3p6 import mapping_vllm_jax
from tunix.models.qwen3p6 import model as qwen3p6_model


class Qwen3P6ModelTest(absltest.TestCase):

  def test_skip_lm_head_and_compute_final_logits(self):
    config = qwen3p6_model.ModelConfig(
        num_layers=1,
        vocab_size=16,
        embed_dim=8,
        hidden_dim=16,
        num_heads=2,
        head_dim=4,
        num_kv_heads=1,
        rope_theta=10_000,
        norm_eps=1e-6,
        layer_types=("full_attention",),
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )
    model = qwen3p6_model.Qwen3P6(config, rngs=nnx.Rngs(0))
    tokens = jnp.array([[1, 2]], dtype=jnp.int32)
    positions = jnp.array([[0, 1]], dtype=jnp.int32)
    attention_mask = jnp.tril(jnp.ones((1, 2, 2), dtype=jnp.bool_))

    hidden_states, _ = model(
        tokens,
        positions,
        cache=None,
        attention_mask=attention_mask,
        skip_lm_head=True,
    )
    logits = model.compute_final_logits(hidden_states)

    self.assertEqual(hidden_states.shape, (1, 2, 8))
    self.assertEqual(logits.shape, (1, 2, 16))

  def test_packed_projection_reorder_matches_tpu_inference_layout(self):
    value = jnp.arange(24, dtype=jnp.float32).reshape(2, 12)

    result = mapping_vllm_jax._reorder_concatenated_tensor_for_sharding(
        value, split_sizes=(4, 4, 4), n_shards=2, dim=-1
    )

    np.testing.assert_array_equal(
        np.asarray(result),
        np.array([
            [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11],
            [12, 13, 16, 17, 20, 21, 14, 15, 18, 19, 22, 23],
        ], dtype=np.float32),
    )

  def test_gdn_qkvz_interleave_matches_qwen3_next_layout(self):
    num_key_heads = mapping_vllm_jax._LINEAR_NUM_KEY_HEADS
    key_head_dim = mapping_vllm_jax._LINEAR_KEY_HEAD_DIM
    value_head_dim = mapping_vllm_jax._LINEAR_VALUE_HEAD_DIM
    value_heads_per_key = mapping_vllm_jax._LINEAR_VALUE_HEADS_PER_KEY
    key_dim = num_key_heads * key_head_dim
    value_dim = mapping_vllm_jax._LINEAR_NUM_VALUE_HEADS * value_head_dim

    q = jnp.arange(key_dim, dtype=jnp.float32)
    k = 10_000 + jnp.arange(key_dim, dtype=jnp.float32)
    v = 20_000 + jnp.arange(value_dim, dtype=jnp.float32)
    z = 30_000 + jnp.arange(value_dim, dtype=jnp.float32)
    qkv = jnp.concatenate((q, k, v))[None, :]

    result = mapping_vllm_jax._qwen3_next_interleave_qkvz(qkv, z[None, :])
    grouped = np.asarray(result[0]).reshape(num_key_heads, -1)
    group_width = key_head_dim * 2 + value_heads_per_key * value_head_dim * 2

    self.assertEqual(result.shape, (1, key_dim * 2 + value_dim * 2))
    self.assertEqual(grouped.shape, (num_key_heads, group_width))
    np.testing.assert_array_equal(grouped[0, :key_head_dim], np.asarray(q[:128]))
    np.testing.assert_array_equal(
        grouped[0, key_head_dim : 2 * key_head_dim], np.asarray(k[:128])
    )
    np.testing.assert_array_equal(
        grouped[0, 2 * key_head_dim : 2 * key_head_dim + 384],
        np.asarray(v[:384]),
    )
    np.testing.assert_array_equal(grouped[1, :key_head_dim], np.asarray(q[128:256]))

  def test_gdn_ba_interleave_matches_qwen3_next_layout(self):
    num_key_heads = mapping_vllm_jax._LINEAR_NUM_KEY_HEADS
    value_heads_per_key = mapping_vllm_jax._LINEAR_VALUE_HEADS_PER_KEY
    num_value_heads = mapping_vllm_jax._LINEAR_NUM_VALUE_HEADS
    b = jnp.arange(num_value_heads, dtype=jnp.float32)
    a = 100 + jnp.arange(num_value_heads, dtype=jnp.float32)

    result = mapping_vllm_jax._qwen3_next_interleave_ba(b[None, :], a[None, :])
    grouped = np.asarray(result[0]).reshape(num_key_heads, -1)

    self.assertEqual(result.shape, (1, num_value_heads * 2))
    self.assertEqual(grouped.shape, (num_key_heads, value_heads_per_key * 2))
    np.testing.assert_array_equal(grouped[0], np.array([0, 1, 2, 100, 101, 102]))
    np.testing.assert_array_equal(grouped[1], np.array([3, 4, 5, 103, 104, 105]))

  def test_gdn_qwen3_next_fusions_are_not_qwen35_packed_hooks(self):
    self.assertNotIn(
        'layers.*.linear_attn.in_proj_qkvz.kernel',
        mapping_vllm_jax.TO_HF_HOOK_FNS,
    )
    self.assertNotIn(
        'layers.*.linear_attn.in_proj_ba.kernel',
        mapping_vllm_jax.TO_HF_HOOK_FNS,
    )


if __name__ == "__main__":
  absltest.main()
