# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vllm_rollout


class _TinyModel(nnx.Module):

  def __init__(self):
    self.weight = nnx.Param(jnp.array([1.0], dtype=jnp.float32))


class VllmRolloutTest(absltest.TestCase):

  def test_kv_cache_reinitializes_lazily_after_weight_sync(self):
    sampler = mock.MagicMock()
    sampler.mesh = mock.MagicMock()
    sampler.return_value = SimpleNamespace(
        text=["answer"],
        tokens=[np.array([1], dtype=np.int32)],
        padded_prompt_tokens=np.array([[0]], dtype=np.int32),
        logprobs=None,
    )
    rollout_config = base_rollout.RolloutConfig(
        max_tokens_to_generate=8,
        max_prompt_length=4,
        rollout_vllm_model_version="dummy-model",
    )

    with mock.patch.object(
        vllm_rollout.mappings.MappingConfig, "build", return_value=mock.Mock()
    ), mock.patch.object(
        vllm_rollout.vllm_sampler, "VllmSampler", return_value=sampler
    ):
      rollout = vllm_rollout.VllmRollout(
          model=_TinyModel(),
          tokenizer=mock.MagicMock(),
          cache_config_or_size=32,
          mesh=mock.MagicMock(),
          rollout_config=rollout_config,
      )

    sampler.load_checkpoint.assert_called_once()
    self.assertFalse(sampler.load_checkpoint.call_args.kwargs["reinitialize_kv_cache"])
    sampler.reinitialize_kv_cache.assert_not_called()

    rollout.generate(["prompt"], rollout_config)
    sampler.reinitialize_kv_cache.assert_called_once()

    rollout.generate(["prompt"], rollout_config)
    sampler.reinitialize_kv_cache.assert_called_once()

    params = {"weight": jnp.array([2.0], dtype=jnp.float32)}
    rollout.update_params(params, filter_types=(nnx.Param,))
    sampler.update_params.assert_called_once_with(
        params, (nnx.Param,), reinitialize_kv_cache=False
    )

    rollout.generate(["prompt"], rollout_config)
    self.assertEqual(sampler.reinitialize_kv_cache.call_count, 2)


if __name__ == "__main__":
  absltest.main()
