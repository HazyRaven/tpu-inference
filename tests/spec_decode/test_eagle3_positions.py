# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer


def _create_mock_proposer(
    method: str = "mtp",
    model_type: str = "gemma4_mtp",
    architectures: list[str] = None,
) -> Eagle3Proposer:
    vllm_config = mock.MagicMock()
    vllm_config.model_config.max_model_len = 8192
    vllm_config.model_config.seed = 42
    vllm_config.scheduler_config.max_num_seqs = 128
    vllm_config.scheduler_config.max_num_batched_tokens = 8192
    vllm_config.scheduler_config.is_encoder_decoder = False
    vllm_config.cache_config.block_size = 16

    speculative_config = mock.MagicMock()
    speculative_config.method = method
    speculative_config.num_speculative_tokens = 4

    draft_model_config = mock.MagicMock()
    draft_hf_config = mock.MagicMock()
    draft_hf_config.model_type = model_type
    draft_model_config.hf_config = draft_hf_config
    draft_model_config.architectures = architectures or []
    speculative_config.draft_model_config = draft_model_config

    # use_gemma4_mtp logic
    speculative_config.use_gemma4_mtp.return_value = (
        (method in ("mtp", "eagle", "eagle3") or model_type in ("gemma4_mtp", "gemma4_assistant"))
        and (model_type in ("gemma4_mtp", "gemma4_assistant") or "Gemma4MTPModel" in draft_model_config.architectures)
    )

    vllm_config.speculative_config = speculative_config

    mock_runner = mock.MagicMock()
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mock_runner.mesh = jax.sharding.Mesh(devices, axis_names=('data', 'model'))
    mock_runner.max_num_tokens = 8192
    mock_runner.max_model_len = 8192
    mock_runner.kv_cache_config.kv_cache_groups = [mock.MagicMock()]
    mock_runner.input_batch = mock.MagicMock()

    return Eagle3Proposer(vllm_config=vllm_config, runner=mock_runner)


@pytest.mark.parametrize("method,model_type,archs,expected_constant", [
    ("mtp", "gemma4_mtp", ["Gemma4MTPModel"], True),
    ("eagle", "gemma4_mtp", ["Gemma4MTPForCausalLM"], True),
    ("eagle3", "gemma4_assistant", [], True),
    ("mtp", "llama", ["LlamaForCausalLM"], False),
    ("eagle3", "llama", [], False),
])
def test_constant_positions_detection_matrix(method, model_type, archs, expected_constant):
    """Tests all configuration permutations, asserting constant_draft_positions evaluates correctly."""
    proposer = _create_mock_proposer(method=method, model_type=model_type, architectures=archs)
    assert proposer.constant_draft_positions == expected_constant


def test_propose_loop_invariant_positions():
    """Executes _propose() loop for K=4 steps; asserts positions passed to draft forward are static."""
    num_speculative_tokens = 4
    proposer = _create_mock_proposer(method="mtp", model_type="gemma4_mtp")
    assert proposer.constant_draft_positions is True

    batch_size = 2
    seq_len = 5
    total_tokens = batch_size * seq_len
    hidden_size = 128
    vocab_size = 100

    observed_positions = []
    observed_seq_lens = []

    def mock_model_fn(state, kv_caches, input_ids, target_hidden_states,
                      attn_metadata, layer_name_to_kvcache_index,
                      spec_step_idx):
        observed_positions.append(attn_metadata.input_positions)
        observed_seq_lens.append(attn_metadata.seq_lens)
        num_tokens = input_ids.shape[0]
        hidden_states_out = jnp.zeros((num_tokens, hidden_size))
        residual_out = jnp.zeros((num_tokens, hidden_size))
        return kv_caches, hidden_states_out, (residual_out,), None

    proposer.model_fn = mock_model_fn
    proposer.compute_logits_fn = lambda s, h, l: jax.nn.one_hot(jnp.zeros(h.shape[0], dtype=jnp.int32), vocab_size)
    proposer.combine_hidden_states_fn = lambda s, h: h

    attn_metadata = AttentionMetadata(
        seq_lens=jnp.array([10, 10]),
        input_positions=jnp.arange(total_tokens),
        query_start_loc=jnp.array([0, 5, 10]),
        block_tables=jnp.zeros((2, 4), dtype=jnp.int32),
        request_distribution=None,
    )

    kv_caches = [jnp.zeros((1, 1, 1))]
    input_ids = jnp.arange(total_tokens)
    target_hidden_states = jnp.zeros((total_tokens, hidden_size))
    last_token_indices = jnp.array([4, 9])

    with jax.set_mesh(proposer.mesh):
        proposer._propose(
            state_leaves=None,
            kv_caches=kv_caches,
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            attn_metadata=attn_metadata,
            layer_name_to_kvcache_index=None,
            last_token_indices=last_token_indices,
            num_speculative_tokens=num_speculative_tokens,
        )

    # In a 4-step proposal, mock_model_fn is called 1 time for initial + 3 times in loop
    assert len(observed_positions) == 4
    # Loop steps (steps 1, 2, 3) must all receive static positions (identical object / values)
    assert observed_positions[1] is observed_positions[2]
    assert observed_positions[2] is observed_positions[3]
    # Sequence lengths must also remain identical across loop steps
    assert observed_seq_lens[1] is observed_seq_lens[2]
    assert observed_seq_lens[2] is observed_seq_lens[3]
