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

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import get_model_loader

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.pp_utils import PPMissingLayer
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.models.jax.gemma4_mtp import (Gemma4MTPDecoderLayer,
                                                 Gemma4MTPForCausalLM)


class DummyTextConfig:

    def __init__(self):
        self.hidden_size = 1024
        self.vocab_size = 262144
        self.num_hidden_layers = 4
        self.rms_norm_eps = 1e-6
        self.layer_types = [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention"
        ]
        self.rope_theta = 10000.0
        self.rope_local_base_freq = 10000.0
        self.rope_scaling = None
        self.head_dim = 256
        self.global_head_dim = 512
        self.num_attention_heads = 32
        self.num_key_value_heads = 16
        self.num_global_key_value_heads = 4
        self.attention_bias = False
        self.attention_k_eq_v = True
        self.intermediate_size = 8192
        self.final_logit_softcapping = None
        self.sliding_window = 1024


class DummyDraftConfig:

    def __init__(self, use_ordered_embeddings=True):
        self.text_config = DummyTextConfig()
        self.backbone_hidden_size = 5376
        self.tie_word_embeddings = True
        self.use_ordered_embeddings = use_ordered_embeddings
        self.num_centroids = 2048
        self.centroid_intermediate_top_k = 32


class TestGemma4MTPForCausalLM:

    @pytest.mark.parametrize("model_name", [
        "google/gemma-4-31B-it",
    ])
    @pytest.mark.parametrize("pp_rank,pp_world_size", [(0, 1), (0, 4), (1, 4),
                                                       (3, 4)])
    @pytest.mark.parametrize(
        "load_format", ["skip_layers_model_loader_for_test", "jax_dummy"])
    @pytest.mark.parametrize("use_ordered_embeddings", [True, False])
    def test_model_loading(
            self,
            model_name,
            pp_rank,
            pp_world_size,
            load_format,
            use_ordered_embeddings,
            # following are defined in conftest.py
            rng,
            mesh,
            mock_vllm_config):
        """Tests loading weights and running forward pass of the MTP model following test_gemma4.py"""
        kv_cache_type = "auto"
        vllm_config = mock_vllm_config(model_name, kv_cache_type)

        # Lightweight config for target/verifier layers
        vllm_config.model_config.hf_config.text_config.num_hidden_layers = 4
        vllm_config.load_config.load_format = load_format
        vllm_config.load_config.num_layers_to_load_for_test = 4
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.enable_expert_parallel = False

        # For HF loader testing, we redirect the model to point to the real assistant draft checkpoint
        if load_format == "skip_layers_model_loader_for_test":
            vllm_config.model_config.model = "google/gemma-4-31B-it-assistant"
            # The resolved revision belongs to the original repo; clear it so
            # the redirected repo resolves its own.
            vllm_config.model_config.revision = None

        # Construct Speculative Draft Config using solid, concrete Python classes to avoid MagicMock leakages
        vllm_config.speculative_config = MagicMock()
        draft_model_config = MagicMock()

        draft_hf_config = DummyDraftConfig(
            use_ordered_embeddings=use_ordered_embeddings)
        draft_hf_config.text_config.vocab_size = vllm_config.model_config.get_vocab_size(
        )
        draft_hf_config.backbone_hidden_size = vllm_config.model_config.get_hidden_size(
        )

        draft_model_config.hf_config = draft_hf_config
        draft_model_config.get_hidden_size = lambda: 1024
        vllm_config.speculative_config.draft_model_config = draft_model_config

        # Initialize Pipeline Parallel group
        init_pp_distributed_environment(
            ip="",
            rank=pp_rank,
            world_size=pp_world_size,
            device=jax.devices()[0],
            need_pp=False,
        )

        model_config = vllm_config.model_config
        kv_dtype = jnp.bfloat16

        vllm_config.quant_config = get_tpu_quantization_config(vllm_config)

        with jax.set_mesh(mesh), set_current_vllm_config(vllm_config):
            model = Gemma4MTPForCausalLM(vllm_config, rng, mesh)

        # Load weights
        with jax.set_mesh(mesh):
            loader = get_model_loader(vllm_config.load_config)
            with set_current_vllm_config(vllm_config):
                if use_ordered_embeddings and load_format == "skip_layers_model_loader_for_test":
                    with pytest.raises(
                            ValueError,
                            match="Ordered embeddings masking is enabled"):
                        loader.load_weights(model, model_config)
                    return
                else:
                    loader.load_weights(model, model_config)

        # Validate layer counts and partitioning
        assert model.model is not None
        assert len(model.model.layers) == 4

        # Fetch the active MTP layer index on this pipeline parallel rank
        start_layer_idx = model.model.start_layer
        end_layer_idx = model.model.end_layer

        if start_layer_idx < end_layer_idx:
            # Verify that the active layer is loaded
            layer_0: Gemma4MTPDecoderLayer = model.model.layers[
                start_layer_idx]
            assert not isinstance(layer_0, PPMissingLayer)

            num_key_value_heads = layer_0.self_attn.num_kv_heads
            qk_head_dim = layer_0.self_attn.head_dim_original

            # Run forward pass on active layer
            seq_len = 2
            input_tensor = jnp.ones(
                (seq_len, draft_hf_config.text_config.hidden_size),
                dtype=jnp.bfloat16)

            block_size = 16
            num_blocks = 8
            cache_shape = get_kv_cache_shape(num_blocks, block_size,
                                             num_key_value_heads, qk_head_dim,
                                             kv_dtype)

            # Populate centroids ordering if enabled to avoid sparse projection crashes
            if use_ordered_embeddings and model.masked_embedding is not None:
                model.masked_embedding.token_ordering.set_value(
                    jnp.arange(draft_hf_config.text_config.vocab_size,
                               dtype=jnp.int32))

            with jax.set_mesh(mesh):
                _, jax_output, _ = layer_0(
                    kv_cache=jnp.zeros(cache_shape, dtype=kv_dtype),
                    x=input_tensor,
                    attention_metadata=AttentionMetadata(
                        input_positions=jnp.arange(seq_len),
                        block_tables=jnp.array(list(range(1))),
                        seq_lens=jnp.array([seq_len]),
                        query_start_loc=jnp.array([0, seq_len]),
                        request_distribution=jnp.array([0, 0, 1]),
                    ),
                )
            assert jax_output is not None
        else:
            # Verify that all layers are missing on this rank (PPMissingLayer)
            for idx in range(4):
                assert isinstance(model.model.layers[idx], PPMissingLayer)


_MODEL_CACHE = {}


def _setup_test_model(rng, mesh, mock_vllm_config, use_ordered_embeddings=False):
    key = use_ordered_embeddings
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    init_pp_distributed_environment(
        ip="",
        rank=0,
        world_size=1,
        device=jax.devices()[0],
        need_pp=False,
    )
    vllm_config = mock_vllm_config("google/gemma-4-31B-it", "auto")
    vllm_config.speculative_config = MagicMock()
    draft_model_config = MagicMock()
    draft_hf_config = DummyDraftConfig(use_ordered_embeddings=use_ordered_embeddings)
    draft_hf_config.backbone_hidden_size = 5120
    draft_hf_config.text_config.hidden_size = 4096
    draft_hf_config.text_config.vocab_size = 256000
    draft_model_config.hf_config = draft_hf_config
    draft_model_config.get_hidden_size = lambda: 4096
    vllm_config.speculative_config.draft_model_config = draft_model_config
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config)

    with jax.set_mesh(mesh), set_current_vllm_config(vllm_config):
        model = Gemma4MTPForCausalLM(vllm_config, rng, mesh)
    _MODEL_CACHE[key] = (model, vllm_config)
    return model, vllm_config


def test_mtp_embed_init_features_equals_hidden_size(rng, mesh, mock_vllm_config):
    """Verifies Gemma4MultiTokenPredictor.embed_tokens.features == hidden_size (placeholder contract)."""
    model, _ = _setup_test_model(rng, mesh, mock_vllm_config)
    assert model.model.embed_tokens.features == 4096
    assert model.model.hidden_size == 4096
    assert model.model.backbone_hidden_size == 5120


def test_mtp_tied_lm_head_loading(rng, mesh, mock_vllm_config):
    """Verifies that load_weights populates lm_head.weight when tie_word_embeddings is True,
    even when backbone_hidden_size != hidden_size."""
    import torch
    model, _ = _setup_test_model(rng, mesh, mock_vllm_config)
    assert model.model.backbone_hidden_size != model.model.hidden_size

    fake_embed_tensor = torch.ones(
        (model.model.vocab_size, model.model.hidden_size), dtype=torch.bfloat16)
    weights_iterator = [("model.embed_tokens.weight", fake_embed_tensor)]

    loaded_keys = model.load_weights(weights_iterator)

    assert "model.embed_tokens.weight" in loaded_keys
    assert "lm_head.weight" in loaded_keys, "lm_head.weight must be loaded when tie_word_embeddings is True"
    assert model.lm_head.weight.shape == (model.model.hidden_size, model.model.vocab_size)


def test_mtp_o_proj_partition_spec(rng, mesh, mock_vllm_config):
    """Verifies that Gemma4MTPAttention o_proj kernel is sharded along Axis 0 (num_heads)
    with PartitionSpec('model', None, None) to avoid all-to-all collectives."""
    from flax.nnx import get_partition_spec
    from jax.sharding import PartitionSpec
    model, _ = _setup_test_model(rng, mesh, mock_vllm_config)
    attn = model.model.layers[0].self_attn
    spec = get_partition_spec(attn.o_proj.weight)
    assert spec == PartitionSpec("model", None, None)


def test_mtp_pp_unpartitioned_layers(rng, mesh, mock_vllm_config):
    """Verifies that all draft layers are fully instantiated as Gemma4MTPDecoderLayer
    even in a multi-rank pipeline parallel configuration (PP > 1), preventing PPMissingLayer stubs."""
    from tpu_inference.distributed.jax_parallel_state import init_pp_distributed_environment
    from tpu_inference.layers.jax.pp_utils import PPMissingLayer
    try:
        init_pp_distributed_environment(ip="", rank=1, world_size=2, device=jax.devices()[0], need_pp=False)
        vllm_config = mock_vllm_config("google/gemma-4-31B-it", "auto")
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.pipeline_parallel_size = 2
        vllm_config.parallel_config.rank = 1
        vllm_config.speculative_config = MagicMock()
        draft_model_config = MagicMock()
        draft_hf_config = DummyDraftConfig(use_ordered_embeddings=False)
        draft_model_config.hf_config = draft_hf_config
        draft_model_config.get_hidden_size = lambda: 4096
        vllm_config.speculative_config.draft_model_config = draft_model_config
        with jax.set_mesh(mesh), set_current_vllm_config(vllm_config):
            model = Gemma4MTPForCausalLM(vllm_config, rng, mesh)
        assert len(model.model.layers) == 4
        for idx, layer in enumerate(model.model.layers):
            assert not isinstance(layer, PPMissingLayer), f"Layer {idx} must not be PPMissingLayer"
            assert isinstance(layer, Gemma4MTPDecoderLayer)
    finally:
        init_pp_distributed_environment(ip="", rank=0, world_size=1, device=jax.devices()[0], need_pp=False)


def test_mtp_heterogeneous_swa_routing(rng, mesh, mock_vllm_config):
    """Verifies that Gemma4MultiTokenPredictor routes heterogeneous per-group block tables:
    draft layers 0..2 attend layer.58 (SWA Group 4) while draft layer 3 attends layer.59 (Full Group 5)."""
    from unittest.mock import MagicMock
    from tpu_inference.layers.common.attention_metadata import AttentionMetadata

    model, _ = _setup_test_model(rng, mesh, mock_vllm_config)
    model.model.config.layer_redirects = {
        "draft_layer.0": "layer.58",
        "draft_layer.1": "layer.58",
        "draft_layer.2": "layer.58",
        "draft_layer.3": "layer.59",
    }

    swa_metadata = MagicMock(spec=AttentionMetadata)
    swa_metadata.group_id = 4
    full_metadata = MagicMock(spec=AttentionMetadata)
    full_metadata.group_id = 5

    dict_attn_metadata = {
        "layer.58": swa_metadata,
        "layer.59": full_metadata,
    }

    passed_metadata = []
    for layer in model.model.layers:
        def make_spy(l):
            def spy_call(kv_cache, hidden_states, attention_metadata):
                passed_metadata.append(attention_metadata)
                return kv_cache, hidden_states, None
            return spy_call
        layer.__call__ = make_spy(layer)

    kv_caches = [jnp.zeros((1,)) for _ in range(4)]
    hidden_states = jnp.zeros((1, 5120), dtype=jnp.bfloat16)
    input_ids = jnp.array([42], dtype=jnp.int32)
    orig_weight = model.model.embed_tokens.weight.value
    try:
        model.model.embed_tokens.weight.value = jnp.zeros(
            (model.model.vocab_size, model.model.backbone_hidden_size), dtype=jnp.bfloat16)
        model.model(kv_caches, input_ids, hidden_states, dict_attn_metadata)
    finally:
        model.model.embed_tokens.weight.value = orig_weight

    assert len(passed_metadata) == 4
    assert passed_metadata[0] is swa_metadata
    assert passed_metadata[1] is swa_metadata
    assert passed_metadata[2] is swa_metadata
    assert passed_metadata[3] is full_metadata


def test_mtp_calibration_forward_unshared_embeddings_and_heterogeneous_kv(rng, mesh, mock_vllm_config):
    """Verifies forward pass succeeds during calibration when embed_tokens has draft hidden_size (1024)
    and layer_name_to_kv_cache is None, falling back to redirects for full-attention cache."""
    model, _ = _setup_test_model(rng, mesh, mock_vllm_config)
    # Unshared embedding: features == 4096 (mock) != backbone_hidden_size (5120)
    assert model.model.embed_tokens.features != model.model.backbone_hidden_size

    # Mock redirects mapping draft_layer.3 -> layer.59
    model.model.layer_redirects = {
        "draft_layer.0": "layer.58",
        "draft_layer.1": "layer.58",
        "draft_layer.2": "layer.58",
        "draft_layer.3": "layer.59",
    }

    # Spy on layer calls to capture accessed kv_cache and prevent mock rope failure
    accessed_caches = []
    for layer in model.model.layers:
        def make_spy(l):
            def spy_call(kv_cache, hidden_states, attention_metadata):
                accessed_caches.append(kv_cache)
                return kv_cache, hidden_states, None
            return spy_call
        layer.__call__ = make_spy(layer)

    kv_caches = [jnp.zeros((1, 16, 2, 4, 256)) for _ in range(59)] + [jnp.zeros((1, 16, 1, 4, 512))]
    input_ids = jnp.array([42], dtype=jnp.int32)
    hidden_states = jnp.zeros((1, 5120), dtype=jnp.bfloat16)
    attn_metadata = MagicMock()

    kv_caches_out, h_draft, h_backbone = model.model(
        kv_caches, input_ids, hidden_states, attn_metadata, layer_name_to_kv_cache=None
    )
    assert h_draft.shape == (1, 4096)
    assert h_backbone.shape == (1, 5120)
    assert len(accessed_caches) == 4
    # Verify Draft Layer 3 accessed kv_caches[59] (Full Attention 512-dim), not kv_caches[3]
    assert accessed_caches[3] is kv_caches[59]
