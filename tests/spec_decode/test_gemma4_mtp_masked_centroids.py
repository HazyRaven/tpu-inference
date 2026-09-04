# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer


def _create_mock_proposer() -> Eagle3Proposer:
    vllm_config = mock.MagicMock()
    vllm_config.model_config.max_model_len = 8192
    vllm_config.model_config.seed = 42
    vllm_config.scheduler_config.max_num_seqs = 128
    vllm_config.scheduler_config.max_num_batched_tokens = 8192
    vllm_config.scheduler_config.is_encoder_decoder = False
    vllm_config.cache_config.block_size = 16

    speculative_config = mock.MagicMock()
    speculative_config.method = "mtp"
    speculative_config.num_speculative_tokens = 4
    speculative_config.use_gemma4_mtp.return_value = True

    draft_model_config = mock.MagicMock()
    draft_hf_config = mock.MagicMock()
    draft_hf_config.model_type = "gemma4_mtp"
    draft_hf_config.suppress_tokens = []
    draft_model_config.hf_config = draft_hf_config
    draft_model_config.try_get_generation_config.return_value = {}
    speculative_config.draft_model_config = draft_model_config

    vllm_config.speculative_config = speculative_config

    mock_runner = mock.MagicMock()
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mock_runner.mesh = jax.sharding.Mesh(devices, axis_names=('data', 'model'))
    mock_runner.max_num_tokens = 8192
    mock_runner.max_model_len = 8192
    mock_runner.kv_cache_config.kv_cache_groups = [mock.MagicMock()]
    mock_runner.input_batch = mock.MagicMock()

    proposer = Eagle3Proposer(vllm_config=vllm_config, runner=mock_runner)
    return proposer


def test_proposer_sparse_argmax_dispatch():
    """Verifies that proposer delegates to model.get_top_tokens when masked_embedding is present."""
    proposer = _create_mock_proposer()
    mock_model = mock.MagicMock()
    mock_model.masked_embedding = mock.MagicMock()
    mock_model.get_top_tokens.return_value = jnp.array([123, 456], dtype=jnp.int32)
    proposer.model = mock_model
    with jax.set_mesh(proposer.mesh):
        draft_tokens = proposer._get_draft_token_ids(None, jnp.zeros((2, 128)))
    mock_model.get_top_tokens.assert_called_once()
    assert jnp.array_equal(draft_tokens, jnp.array([123, 456], dtype=jnp.int32))
