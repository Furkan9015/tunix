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


if __name__ == "__main__":
  absltest.main()
