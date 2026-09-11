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
    """Verifies the proposer takes the sparse path when masked_embedding is present.

    Dispatch now goes through the jitted `get_top_tokens_fn` rather than
    reaching into the live nnx module; see
    test_proposer_sparse_argmax_uses_jitted_state_leaves_fn for why.
    """
    proposer = _create_mock_proposer()
    mock_model = mock.MagicMock()
    mock_model.masked_embedding = mock.MagicMock()
    proposer.model = mock_model
    proposer.get_top_tokens_fn = mock.MagicMock(
        return_value=jnp.array([123, 456], dtype=jnp.int32))
    with jax.set_mesh(proposer.mesh):
        draft_tokens = proposer._get_draft_token_ids(None, jnp.zeros((2, 128)))
    proposer.get_top_tokens_fn.assert_called_once()
    assert jnp.array_equal(draft_tokens, jnp.array([123, 456], dtype=jnp.int32))


def test_proposer_sparse_argmax_uses_jitted_state_leaves_fn():
    """NEW-5: get_top_tokens must go through a jitted state_leaves closure.

    `_propose` is decorated with `self` in static_argnums, so reaching into the
    live nnx.Module (`self.model.get_top_tokens(...)`) traces lm_head.weight,
    token_ordering and centroids as Python constants and lowers them to
    stablehlo.constant literals. lm_head.weight alone is [1024, 262144] bf16,
    roughly 512 MB baked into the executable and the compile cache, with the
    NamedSharding lost.

    Worse, it bypasses the state_leaves path every other model call uses, so
    any post-trace parameter mutation -- exactly what `load_model` does when it
    shares the target embedding -- is invisible here while `compute_logits_fn`
    picks it up.

    Route through `get_top_tokens_fn`, symmetric with `compute_logits_fn`.
    """
    proposer = _create_mock_proposer()
    mock_model = mock.MagicMock()
    mock_model.masked_embedding = mock.MagicMock()
    proposer.model = mock_model

    sentinel_state = ("state_leaf_0", "state_leaf_1")
    seen = {}

    def _fake_get_top_tokens_fn(state_leaves, hidden_states):
        seen["state_leaves"] = state_leaves
        return jnp.array([123, 456], dtype=jnp.int32)

    proposer.get_top_tokens_fn = _fake_get_top_tokens_fn

    with jax.set_mesh(proposer.mesh):
        draft_tokens = proposer._get_draft_token_ids(sentinel_state,
                                                     jnp.zeros((2, 128)))

    assert seen["state_leaves"] is sentinel_state, (
        "get_top_tokens must receive state_leaves as a jit argument")
    mock_model.get_top_tokens.assert_not_called()
    assert jnp.array_equal(draft_tokens,
                           jnp.array([123, 456], dtype=jnp.int32))


def test_proposer_falls_back_to_logits_without_masked_embedding():
    """No centroids -> dense argmax over compute_logits_fn, as before."""
    proposer = _create_mock_proposer()
    mock_model = mock.MagicMock()
    mock_model.masked_embedding = None
    proposer.model = mock_model
    proposer.get_top_tokens_fn = mock.MagicMock()

    logits = jnp.zeros((2, 8)).at[0, 3].set(1.0).at[1, 5].set(1.0)
    proposer.compute_logits_fn = lambda *a, **k: logits

    with jax.set_mesh(proposer.mesh):
        draft_tokens = proposer._get_draft_token_ids(None, jnp.zeros((2, 128)))

    proposer.get_top_tokens_fn.assert_not_called()
    assert jnp.array_equal(draft_tokens, jnp.array([3, 5], dtype=jnp.int32))


def test_model_loader_exposes_jitted_get_top_tokens_fn():
    """`ModelInterface` must carry a `get_top_tokens_fn` for the proposer."""
    import dataclasses

    from tpu_inference.models.common.interface import ModelInterface

    fields = {f.name for f in dataclasses.fields(ModelInterface)}
    assert "get_top_tokens_fn" in fields, (
        "ModelInterface must expose get_top_tokens_fn alongside "
        "compute_logits_fn")


def test_masked_embedding_get_top_tokens_honours_suppressed_tokens():
    """The sparse and dense paths must not disagree on suppressed tokens.

    `compute_logits` applies `_suppress_token_ids`, but
    `masked_embedding.get_top_tokens` applied neither that nor softcapping, so
    the two branches could return different tokens for the same hidden state --
    and the sparse one could emit a token the dense one masks to -inf.

    Calls the unbound function with a stand-in `self`, since the real embedder
    is an nnx module that cannot be built without full config.
    """
    from tpu_inference.models.jax.gemma4_mtp import Gemma4MTPMaskedEmbedder

    class _Stub:
        """Supplies only `_select_and_score`, all get_top_tokens depends on."""

        @staticmethod
        def _select_and_score(hidden_states, lm_head_weight):
            # 4 candidates; index 2 scores highest but is suppressed.
            return (jnp.array([[1.0, 2.0, 9.0, 3.0]]),
                    jnp.array([[10, 11, 12, 13]]))

    get_top_tokens = Gemma4MTPMaskedEmbedder.get_top_tokens
    stub = _Stub()

    top = get_top_tokens(stub,
                         jnp.zeros((1, 8)),
                         jnp.zeros((8, 4)),
                         suppress_token_ids=jnp.array([12], dtype=jnp.int32))
    assert int(top[0]) == 13, (
        "suppressed token 12 was selected by the sparse argmax path")

    unsuppressed = get_top_tokens(stub, jnp.zeros((1, 8)), jnp.zeros((8, 4)))
    assert int(unsuppressed[0]) == 12
