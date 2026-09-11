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

    # The zero-embedding fallback is calibration-only; at serving time a width
    # mismatch now raises rather than silently destroying token identity.
    model.model._calibrating = True
    try:
        kv_caches_out, h_draft, h_backbone = model.model(
            kv_caches, input_ids, hidden_states, attn_metadata, layer_name_to_kv_cache=None
        )
    finally:
        model.model._calibrating = False
    assert h_draft.shape == (1, 4096)
    assert h_backbone.shape == (1, 5120)
    assert len(accessed_caches) == 4
    # Verify Draft Layer 3 accessed kv_caches[59] (Full Attention 512-dim), not kv_caches[3]
    assert accessed_caches[3] is kv_caches[59]


def test_compute_logits_softcapping_masked_embedding(rng, mesh, mock_vllm_config):
    """Verifies that when masked_embedding is active, final_logit_softcapping is NOT applied
    to masked logits, ensuring unselected tokens remain -inf rather than collapsing to -30.0."""
    model, _ = _setup_test_model(rng, mesh, mock_vllm_config, use_ordered_embeddings=True)
    model.final_logit_softcapping = 30.0
    model.masked_embedding.token_ordering.set_value(jnp.arange(256000, dtype=jnp.int32))
    hidden_states = jnp.ones((1, 4096), dtype=jnp.bfloat16)
    logits = model.compute_logits(hidden_states)
    assert jnp.isneginf(logits[0, 250000]) or logits[0, 250000] == jnp.finfo(logits.dtype).min, f"Unselected token logit was {float(logits[0, 250000])}, expected -inf"


class _V5RopeTextConfig(DummyTextConfig):
    """A draft text config shaped like transformers v5 for Gemma 4.

    Two things matter here and neither is captured by `DummyTextConfig`:

    1. `rope_parameters` is a per-layer-type mapping, matching the shipped
       `google/gemma-4-31B-it-assistant` config.json.
    2. `rope_scaling` is a *property aliased to rope_parameters*, which is what
       `transformers>=5` does on `PreTrainedConfig`. It is therefore never
       None, so `getattr(config, "rope_scaling", None)` cannot be used as a
       "nothing configured" default.

    `DummyTextConfig.__init__` assigns `self.rope_scaling = None`, so the
    property is declared on a subclass with a no-op setter that swallows it.
    """

    def __init__(self):
        super().__init__()
        self.rope_parameters = {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        }

    @property
    def rope_scaling(self):
        # transformers v5 aliases this to rope_parameters.
        return self.rope_parameters

    @rope_scaling.setter
    def rope_scaling(self, value):
        # The base __init__ sets this to None; v5 configs do not allow that.
        pass


def _build_mtp_attention(layer_idx: int, rng, mesh):
    from flax import nnx

    from tpu_inference.models.jax.gemma4_mtp import Gemma4MTPAttention
    config = _V5RopeTextConfig()
    with jax.set_mesh(mesh):
        return Gemma4MTPAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=jnp.bfloat16,
            rng=nnx.Rngs(params=rng),
            mesh=mesh,
            kv_cache_dtype="auto",
            quant_config=None,
            prefix=f"draft_layer.{layer_idx}",
        )


@pytest.mark.parametrize("layer_idx,layer_type", [
    (0, "sliding_attention"),
    (1, "sliding_attention"),
    (2, "sliding_attention"),
])
def test_mtp_sliding_layers_do_not_get_spurious_rope_rescale(
        layer_idx, layer_type, rng, mesh):
    """NEW-1: the sliding draft layers must end up with rope_scaling None.

    The per-layer dict {"rope_theta": 10000.0, "rope_type": "default"} has no
    "rope_scaling" key, so `.get("rope_scaling", getattr(config,
    "rope_scaling", None))` fell through to the config-level attribute -- which
    in transformers v5 is an alias for the whole two-level rope_parameters
    dict. That dict is truthy and its `.get("rope_type")` is None (its keys are
    layer type names), so the guard in `apply_rope` passed and a llama3 NTK
    rescale fired with scale_factor=8.0 on 3 of the 4 draft layers.

    `normalize_rope_scaling` collapses rope_type == "default" with no factor to
    None, which is what `gemma4.py` has always done for the target model.
    """
    attn = _build_mtp_attention(layer_idx, rng, mesh)
    assert attn.layer_type == layer_type
    assert attn.rope_theta == 10000.0
    assert attn.rope_scaling is None, (
        f"draft layer {layer_idx} resolved rope_scaling to "
        f"{attn.rope_scaling!r}; a truthy value without rope_type "
        f"'proportional' triggers a spurious llama3 NTK rescale")


def test_mtp_full_attention_layer_rope_params(rng, mesh):
    """Layer 3 is full attention: partial rotary, high theta, no rescale."""
    attn = _build_mtp_attention(3, rng, mesh)
    assert attn.layer_type == "full_attention"
    assert attn.rope_theta == 1000000.0
    assert attn.rope_proportion == 0.25
    assert attn.rope_scaling is None


@pytest.mark.parametrize("layer_idx", [0, 1, 2, 3])
def test_mtp_rope_resolution_matches_target_model(layer_idx, rng, mesh):
    """The decisive check: draft and target must resolve rope identically.

    The draft reads K that the target already rotated and wrote to the cache,
    so any divergence in rope resolution between `Gemma4Attention` and
    `Gemma4MTPAttention` decorrelates draft queries from target keys. The
    target file was hardened with `normalize_rope_scaling`; the MTP file was a
    copy-paste that omitted it. Pin them together so they cannot drift again.
    """
    from flax import nnx

    from tpu_inference.models.jax.gemma4 import Gemma4Attention

    config = _V5RopeTextConfig()
    with jax.set_mesh(mesh):
        target = Gemma4Attention(
            config=config,
            layer_idx=layer_idx,
            dtype=jnp.bfloat16,
            rng=nnx.Rngs(params=rng),
            mesh=mesh,
            kv_cache_dtype="auto",
            quant_config=None,
            prefix=f"layer.{layer_idx}",
        )
    draft = _build_mtp_attention(layer_idx, rng, mesh)

    assert draft.rope_scaling == target.rope_scaling
    assert draft.rope_theta == target.rope_theta
    assert draft.rope_proportion == target.rope_proportion


def test_mtp_sliding_rope_frequencies_match_unscaled_reference(rng, mesh):
    """The numerical consequence: rotated queries must match the target's keys.

    The target rotates K at absolute positions before writing the cache and the
    draft reads those already-rotated keys, so a frequency error accumulates
    with the ABSOLUTE position, not the within-window offset. At 50k context
    the buggy rescale is worth ~7 full rotations of phase error.
    """
    from tpu_inference.layers.jax.rope_interface import apply_rope

    attn = _build_mtp_attention(0, rng, mesh)
    head_dim = attn.head_dim_original
    positions = jnp.array([0, 1024, 20000, 50000], dtype=jnp.int32)
    x = jnp.ones((positions.shape[0], 2, head_dim), dtype=jnp.float32)

    actual = apply_rope(x,
                        positions,
                        head_dim,
                        attn.rope_theta,
                        rope_scaling=attn.rope_scaling,
                        rope_proportion=attn.rope_proportion)
    expected = apply_rope(x,
                          positions,
                          head_dim,
                          attn.rope_theta,
                          rope_scaling=None,
                          rope_proportion=attn.rope_proportion)

    assert jnp.allclose(actual, expected, atol=1e-6), (
        "draft sliding-layer RoPE diverges from the unscaled reference the "
        "target model used when it rotated and cached K")


def _mtp_forward_args(model):
    """Minimal args for a Gemma4MultiTokenPredictor forward pass."""
    for layer in model.model.layers:

        def make_spy(l):

            def spy_call(kv_cache, hidden_states, attention_metadata):
                return kv_cache, hidden_states, None

            return spy_call

        layer.__call__ = make_spy(layer)

    kv_caches = ([jnp.zeros((1, 16, 2, 4, 256)) for _ in range(59)] +
                 [jnp.zeros((1, 16, 1, 4, 512))])
    return dict(
        kv_caches=kv_caches,
        input_ids=jnp.array([42], dtype=jnp.int32),
        hidden_states=jnp.zeros((1, 5120), dtype=jnp.bfloat16),
        attention_metadata=MagicMock(),
        layer_name_to_kv_cache=None,
    )


def test_mtp_embedding_width_mismatch_raises_at_serving(rng, mesh,
                                                        mock_vllm_config):
    """NEW-4: a width mismatch outside calibration must fail loud.

    At serving time `Eagle3Proposer.load_model` swaps the draft's embed_tokens
    for the target's [vocab, backbone_hidden_size] table. If that swap does not
    land -- params are None, a path name changed, load_model ordering shifted
    -- the previous code silently replaced the ENTIRE token embedding with
    zeros and continued. The drafter degenerates into a hidden-state-only
    predictor with no idea which token it was given, so acceptance collapses to
    near zero with no log line and no exception.
    """
    model, _ = _setup_test_model(rng, mesh, mock_vllm_config)
    assert model.model.embed_tokens.features != model.model.backbone_hidden_size
    assert not getattr(model.model, "_calibrating", False)

    with pytest.raises(ValueError, match="embedding"):
        model.model(**_mtp_forward_args(model))


def test_mtp_embedding_width_mismatch_allowed_during_calibration(
        rng, mesh, mock_vllm_config):
    """The same mismatch is legitimate during PTQ/QWIX calibration.

    Calibration runs BEFORE load_model shares the target embedding, so
    embed_tokens is still draft-width and pre_projection wants backbone-width.
    Substituting zeros is fine there: only the consumed weight shapes matter
    for tracing.
    """
    model, _ = _setup_test_model(rng, mesh, mock_vllm_config)
    model.model._calibrating = True
    try:
        _, h_draft, h_backbone = model.model(**_mtp_forward_args(model))
    finally:
        model.model._calibrating = False

    assert h_draft.shape == (1, 4096)
    assert h_backbone.shape == (1, 5120)


def test_qwix_calibration_sets_and_clears_calibrating_flag():
    """`qwix_quantize_nnx_model` must set `_calibrating` around quantize_model.

    Without this the hardened check above would break FP8/QWIX boot, which is
    the one caller that legitimately needs the zero-embedding fallback. The
    flag is set on the live module before tracing, so it is a plain Python
    attribute rather than anything a tracer sees.
    """
    from unittest.mock import patch

    from tpu_inference.models.jax.utils.qwix import qwix_utils

    observed = {}

    class _FakeModel:

        def __call__(self, kv_caches, input_ids, hidden_states,
                     attention_metadata):
            pass

    fake_model = _FakeModel()
    fake_model.backbone_hidden_size = 5120
    fake_model.vllm_config = MagicMock()
    fake_model.vllm_config.model_config.use_mla = False
    fake_model.vllm_config.sharding_config.total_dp_size = 1

    def _fake_quantize_model(model, provider, **model_input):
        observed["during"] = getattr(model, "_calibrating", False)
        return model

    # hbm_usage_gb reads device.memory_stats(), which is None on CPU.
    with patch.object(qwix_utils.qwix, "quantize_model",
                      _fake_quantize_model), \
         patch.object(qwix_utils, "create_kv_caches", return_value=[]), \
         patch.object(qwix_utils.utils, "hbm_usage_gb", return_value=0.0), \
         patch.object(qwix_utils, "device_array",
                      side_effect=lambda mesh, x, **kw: x):
        out = qwix_utils.qwix_quantize_nnx_model(
            model=fake_model,
            qwix_config=[],
            rng=jax.random.PRNGKey(0),
            mesh=MagicMock(),
            num_hidden_layers=0,
            kv_cache_block_size=16,
            kv_cache_num_kv_heads=4,
            kv_cache_head_size=256,
            kv_cache_dtype="auto",
        )

    assert observed["during"] is True, (
        "_calibrating must be set while qwix traces the model")
    assert getattr(out, "_calibrating", False) is False, (
        "_calibrating must be cleared once calibration finishes")


def test_mtp_layer_redirects_resolved_consistently(rng, mesh,
                                                   mock_vllm_config):
    """NEW-9: KV array and attention metadata must use the SAME redirect map.

    The same lookup was resolved with opposite precedence eleven lines apart in
    the same loop: the KV-cache-array selection preferred
    `self.config.layer_redirects` while the attention-metadata selection
    preferred `self.layer_redirects`. Both attributes are genuinely populated
    by different owners -- the runner sets the former on draft_hf_config, the
    model constructor computes the latter -- so if they ever diverge a draft
    layer reads one target layer's KV array while using another target layer's
    block tables. That is exactly the silent corruption the block-table fixes
    are about, and it would be very hard to trace.

    Give the two attributes deliberately different maps and assert both
    consumers agree.
    """
    model, _ = _setup_test_model(rng, mesh, mock_vllm_config)

    # Same key, different targets, on the two different owners.
    model.model.config.layer_redirects = {"draft_layer.0": "layer.58"}
    model.model.layer_redirects = {"draft_layer.0": "layer.59"}

    seen_caches = []
    seen_metadata = []
    for i, layer in enumerate(model.model.layers):

        def make_spy(idx):

            def spy_call(kv_cache, hidden_states, attention_metadata):
                if idx == 0:
                    seen_caches.append(kv_cache)
                    seen_metadata.append(attention_metadata)
                return kv_cache, hidden_states, None

            return spy_call

        layer.__call__ = make_spy(i)

    # Tag each cache so we can tell which target layer was picked.
    kv_caches = [
        jnp.full((1, 16, 2, 4, 256), float(i)) for i in range(59)
    ] + [jnp.full((1, 16, 1, 4, 512), 59.0)]
    # One metadata entry per candidate target layer, likewise tagged.
    attn_metadata = {
        "layer.58": MagicMock(name="md_58"),
        "layer.59": MagicMock(name="md_59"),
    }

    model.model._calibrating = True
    try:
        model.model(kv_caches,
                    jnp.array([42], dtype=jnp.int32),
                    jnp.zeros((1, 5120), dtype=jnp.bfloat16),
                    attn_metadata,
                    layer_name_to_kv_cache=None)
    finally:
        model.model._calibrating = False

    cache_target = int(seen_caches[0].flatten()[0])
    metadata_target = 58 if seen_metadata[0] is attn_metadata[
        "layer.58"] else 59
    assert cache_target == metadata_target, (
        f"draft_layer.0 read kv_caches[{cache_target}] but used layer."
        f"{metadata_target}'s attention metadata; the two redirect lookups "
        f"disagree")
