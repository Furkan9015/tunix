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

  def test_gdn_qkvz_packed_reorder_matches_qwen35_runtime_layout(self):
    key_dim = mapping_vllm_jax._LINEAR_KEY_DIM
    value_dim = mapping_vllm_jax._LINEAR_VALUE_DIM
    q = jnp.arange(key_dim, dtype=jnp.float32)
    k = 10_000 + jnp.arange(key_dim, dtype=jnp.float32)
    v = 20_000 + jnp.arange(value_dim, dtype=jnp.float32)
    z = 30_000 + jnp.arange(value_dim, dtype=jnp.float32)
    qkvz = jnp.concatenate((q, k, v, z))[None, :]

    result = mapping_vllm_jax._reorder_concatenated_tensor_for_sharding(
        qkvz,
        split_sizes=(key_dim, key_dim, value_dim, value_dim),
        n_shards=4,
        dim=-1,
    )

    shard_width = result.shape[-1] // 4
    shard0 = np.asarray(result[0, :shard_width])
    np.testing.assert_array_equal(shard0[: key_dim // 4], np.asarray(q[:512]))
    np.testing.assert_array_equal(
        shard0[key_dim // 4 : key_dim // 2], np.asarray(k[:512])
    )
    np.testing.assert_array_equal(
        shard0[key_dim // 2 : key_dim // 2 + value_dim // 4],
        np.asarray(v[:1536]),
    )
    np.testing.assert_array_equal(shard0[-value_dim // 4 :], np.asarray(z[:1536]))

  def test_gdn_ba_packed_reorder_matches_qwen35_runtime_layout(self):
    num_value_heads = mapping_vllm_jax._LINEAR_NUM_VALUE_HEADS
    b = jnp.arange(num_value_heads, dtype=jnp.float32)
    a = 100 + jnp.arange(num_value_heads, dtype=jnp.float32)
    ba = jnp.concatenate((b, a))[None, :]

    result = mapping_vllm_jax._reorder_concatenated_tensor_for_sharding(
        ba,
        split_sizes=(num_value_heads, num_value_heads),
        n_shards=4,
        dim=-1,
    )

    shard_width = result.shape[-1] // 4
    np.testing.assert_array_equal(
        np.asarray(result[0, :shard_width]),
        np.concatenate((np.arange(12), 100 + np.arange(12))).astype(np.float32),
    )

  def test_gdn_qwen35_fusions_use_packed_hooks(self):
    self.assertIn(
        'layers.*.linear_attn.in_proj_qkvz.kernel',
        mapping_vllm_jax.TO_HF_HOOK_FNS,
    )
    self.assertIn(
        'layers.*.linear_attn.in_proj_ba.kernel',
        mapping_vllm_jax.TO_HF_HOOK_FNS,
    )

  def test_gdn_qwen35_fusions_use_column_parallel_sharding(self):
    self.assertEqual(
        mapping_vllm_jax.TO_HF_MAPPINGS[
            'layers.*.linear_attn.in_proj_qkvz.kernel'
        ][1],
        (None, 'model'),
    )
    self.assertEqual(
        mapping_vllm_jax.TO_HF_MAPPINGS[
            'layers.*.linear_attn.in_proj_ba.kernel'
        ][1],
        (None, 'model'),
    )


if __name__ == "__main__":
  absltest.main()
