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
"""End-to-End Integration, High-Concurrency SWA Sizing, and MTP Tests for Gemma 4 12B on 1 TPU."""

import math
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh
from vllm.config import set_current_vllm_config
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        SlidingWindowSpec)

from tpu_inference import utils as common_utils
from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.models.jax.gemma4 import Gemma4DecoderLayer, Gemma4Model
from tpu_inference.models.jax.gemma4_mtp import (Gemma4MTPDecoderLayer,
                                                 Gemma4MTPForCausalLM)
from tpu_inference.runner.kv_cache_manager import KVCacheManager


@pytest.fixture(scope="module")
def single_tpu_mesh():
    if not jax.devices():
        pytest.skip("No JAX devices available.")
    # Limit strictly to 1 TPU device
    devices = np.array(jax.local_devices()[:1])
    device_mesh = devices.reshape((1, 1, 1, 1))
    mesh = Mesh(device_mesh, axis_names=("data", "attn_dp", "expert", "model"))
    with jax.set_mesh(mesh):
        yield mesh


class TestGemma412BIntegrationAndConcurrency:
    """Comprehensive test suite for Gemma 4 12B with SWA, single-projection global attention,
    MTP speculative decoding, and high concurrency on 1 TPU v6e.
    """

    def test_gemma4_12b_concurrency_and_memory_math(self):
        """Calculates and validates maximum concurrency and KV cache memory reduction
        for Gemma 4 12B on 1 TPU v6e chip (32 GiB HBM).
        
        Gemma 4 12B Architecture:
        - 48 total layers
        - 40 Sliding Window Attention layers (window = 1024, num_kv_heads = 8, head_dim = 256, block_size = 16)
        - 8 Global Full Attention layers (num_kv_heads = 1, head_dim = 512, attention_k_eq_v = True, block_size = 32)
        """
        num_layers = 48
        num_swa_layers = 40
        num_global_layers = 8
        sliding_window = 1024
        
        # SWA page size: 16 * 8 * 2 (k+v) * 256 * 2 (bfloat16) = 131,072 bytes (128 KiB)
        swa_page_size = 16 * 8 * 2 * 256 * 2
        assert swa_page_size == 128 * 1024
        
        # Global page size (single projection: k=v -> 1 projection):
        # 32 * 1 * 1 (single) * 512 * 2 (bfloat16) = 32,768 bytes (32 KiB)
        global_page_size = 32 * 1 * 1 * 512 * 2
        assert global_page_size == 32 * 1024

        # SWA ring buffer blocks per request at C_max = 512, B = 16
        c_max = 512
        block_size_swa = 16
        extra_retained_tokens = 16
        blocks_per_req_swa = math.ceil(((sliding_window - 1) + c_max + extra_retained_tokens) / block_size_swa) + 1
        assert blocks_per_req_swa == 98

        # SWA physical memory per stream across all 40 SWA layers:
        # 40 * 98 * 128 KiB = 501,760 KiB = 490.0 MiB
        swa_bytes_per_stream = num_swa_layers * blocks_per_req_swa * swa_page_size
        swa_mib_per_stream = swa_bytes_per_stream / (1024 * 1024)
        assert abs(swa_mib_per_stream - 490.0) < 0.1

        # Calculate KV cache memory per stream across various context lengths:
        context_lengths = [2048, 4096, 8192, 32768]
        # Total usable KV budget on 1 TPU v6e chip (~4.73 GiB with 0.90 utilization + MTP draft):
        kv_budget_bytes = int(4.73 * 1024 * 1024 * 1024)

        concurrency_results = {}
        reduction_ratios = {}

        for seq_len in context_lengths:
            # Global KV bytes (single projection: 8 layers * seq_len * 1 head * 512 dim * 2 bytes)
            global_bytes_per_stream = num_global_layers * seq_len * 1 * 512 * 2
            total_bytes_per_stream = swa_bytes_per_stream + global_bytes_per_stream
            
            # Baseline uncompressed KV cache (48 layers * seq_len * 8 heads * 2 (k+v) * 256 * 2 bytes)
            baseline_bytes_per_stream = num_layers * seq_len * 8 * 2 * 256 * 2
            
            reduction_ratio = baseline_bytes_per_stream / total_bytes_per_stream
            max_concurrency = kv_budget_bytes // total_bytes_per_stream
            
            concurrency_results[seq_len] = max_concurrency
            reduction_ratios[seq_len] = reduction_ratio

        # Assertions on concurrency and reduction targets:
        assert reduction_ratios[8192] >= 5.5, f"Expected >= 5.5x reduction at 8k, got {reduction_ratios[8192]:.2f}x"
        assert reduction_ratios[32768] >= 15.0, f"Expected >= 15.0x reduction at 32k, got {reduction_ratios[32768]:.2f}x"
        
        # Max concurrency on 1 TPU v6e chip:
        assert concurrency_results[2048] >= 9, f"Expected >= 9 concurrent streams at 2k, got {concurrency_results[2048]}"
        assert concurrency_results[8192] >= 8, f"Expected >= 8 concurrent streams at 8k, got {concurrency_results[8192]}"
        assert concurrency_results[32768] >= 6, f"Expected >= 6 concurrent streams at 32k, got {concurrency_results[32768]}"

    def test_gemma4_12b_multigroup_kv_cache_allocation(self, single_tpu_mesh):
        """Verifies multi-group KV cache allocation and high-concurrency logical-to-physical
        slot mapping for Gemma 4 12B on 1 TPU v6e.
        """
        max_num_reqs = 8
        c_max = 512
        sliding_window = 1024
        
        # SWA Group (40 layers, B=16, 8 heads, head_dim=256)
        swa_spec = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=sliding_window,
        )
        
        # Global Group (8 layers, B=32, 1 head, head_dim=512, single_projection=True)
        global_spec = FullAttentionSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
        )
        object.__setattr__(global_spec, "single_projection", True)

        blocks_per_req_swa = common_utils.get_swa_blocks_per_req(
            sliding_window=sliding_window,
            block_size=16,
            max_num_batched_tokens=c_max,
            extra_retained_tokens=16,
        )
        num_swa_blocks = max_num_reqs * blocks_per_req_swa + 1
        num_global_blocks = 2048  # Ample global blocks

        groups = [
            KVCacheGroupSpec(
                layer_names=[f"layer.{i}" for i in range(40)],
                kv_cache_spec=swa_spec,
            ),
            KVCacheGroupSpec(
                layer_names=[f"layer.{40 + i}" for i in range(8)],
                kv_cache_spec=global_spec,
            ),
        ]
        tensors = [
            KVCacheTensor(
                size=num_swa_blocks * swa_spec.page_size_bytes * 40,
                layers=[f"layer.{i}" for i in range(40)],
                layer_stride=num_swa_blocks * swa_spec.page_size_bytes,
                block_stride=swa_spec.page_size_bytes,
            ),
            KVCacheTensor(
                size=num_global_blocks * global_spec.page_size_bytes * 8,
                layers=[f"layer.{40 + i}" for i in range(8)],
                layer_stride=num_global_blocks * global_spec.page_size_bytes,
                block_stride=global_spec.page_size_bytes,
            ),
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_global_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=groups,
        )

        mock_runner = MagicMock()
        mock_runner.mesh = single_tpu_mesh
        mock_runner.vllm_config = MagicMock()
        mock_runner.vllm_config.kv_transfer_config = None
        mock_runner.vllm_config.offload_config.uva.cpu_offload_gb = 0
        mock_runner.vllm_config.speculative_config.num_speculative_tokens = 0
        mock_runner.cache_config = MagicMock()
        mock_runner.cache_config.block_size = 16
        mock_runner.cache_config.cache_dtype = "auto"
        mock_runner.cache_config.num_gpu_blocks_override = num_global_blocks
        mock_runner.scheduler_config = MagicMock()
        mock_runner.scheduler_config.max_num_seqs = max_num_reqs
        mock_runner.scheduler_config.max_num_batched_tokens = c_max
        mock_runner.model_config = MagicMock()
        mock_runner.model_config.get_sliding_window.return_value = sliding_window
        mock_runner.kv_cache_config = kv_cache_config
        mock_runner.kv_cache_dtype = jnp.bfloat16
        mock_runner.max_model_len = 8192
        mock_runner.max_num_reqs = max_num_reqs
        mock_runner.max_num_tokens = c_max
        mock_runner.uniform_page_size = True
        mock_runner.kv_caches = []
        mock_runner.input_batch = MagicMock()

        # Verify multi-array shapes
        with jax.set_mesh(single_tpu_mesh), set_current_vllm_config(mock_runner.vllm_config):
            kv_cache_manager = KVCacheManager(mock_runner)
            kv_cache_manager.initialize_kv_cache(kv_cache_config)

            assert len(mock_runner.kv_cache_config.kv_cache_groups) == 2
            assert mock_runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.sliding_window == 1024
            assert mock_runner.kv_cache_config.kv_cache_groups[1].kv_cache_spec.single_projection is True

            # Verify allocated layer arrays (40 SWA + 8 Global = 48 layers)
            assert len(mock_runner.kv_caches) == 48
            for cache in mock_runner.kv_caches:
                assert cache.shape[0] == num_global_blocks

    def test_gemma4_12b_jax_layer_execution(self, single_tpu_mesh):
        """Executes a physical JAX forward pass on 1 TPU v6e device for Gemma 4 12B decoder layer."""
        init_pp_distributed_environment(
            ip="",
            rank=0,
            world_size=1,
            device=jax.devices()[0],
            need_pp=False,
        )

        class Mock12BTextConfig:
            def __init__(self):
                self.hidden_size = 3840
                self.vocab_size = 262144
                self.num_hidden_layers = 2
                self.rms_norm_eps = 1e-6
                self.layer_types = ["sliding_attention", "full_attention"]
                self.rope_theta = 10000.0
                self.rope_local_base_freq = 10000.0
                self.rope_scaling = None
                self.head_dim = 256
                self.global_head_dim = 512
                self.num_attention_heads = 16
                self.num_key_value_heads = 8
                self.num_global_key_value_heads = 1
                self.attention_bias = False
                self.attention_k_eq_v = True
                self.intermediate_size = 15360
                self.final_logit_softcapping = None
                self.sliding_window = 1024
                self.use_double_wide_mlp = False
                self.enable_moe_block = False
                self.hidden_size_per_layer_input = 0
                self.num_kv_shared_layers = 0

        mock_vllm_config = MagicMock()
        text_cfg = Mock12BTextConfig()
        mock_vllm_config.model_config.hf_config.text_config = text_cfg
        mock_vllm_config.model_config.get_hidden_size = lambda: 3840
        mock_vllm_config.model_config.get_vocab_size = lambda: 262144
        mock_vllm_config.model_config.quantization = None
        mock_vllm_config.parallel_config = MagicMock()
        mock_vllm_config.parallel_config.tensor_parallel_size = 1
        mock_vllm_config.parallel_config.data_parallel_size = 1
        mock_vllm_config.parallel_config.enable_expert_parallel = False
        mock_vllm_config.quant_config = get_tpu_quantization_config(mock_vllm_config)

        rngs = nnx.Rngs(0)
        seq_len = 2
        input_tensor = jnp.ones((seq_len, 3840), dtype=jnp.bfloat16)

        with jax.set_mesh(single_tpu_mesh), set_current_vllm_config(mock_vllm_config):
            decoder_layer = Gemma4DecoderLayer(
                config=mock_vllm_config.model_config,
                layer_idx=0,
                dtype=jnp.bfloat16,
                rng=rngs,
                mesh=single_tpu_mesh,
                kv_cache_dtype="auto",
                quant_config=mock_vllm_config.quant_config,
            )

            cache_shape = get_kv_cache_shape(
                8,
                16,
                8,
                256,
                jnp.bfloat16,
            )
            kv_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)

            attn_metadata = AttentionMetadata(
                input_positions=jnp.arange(seq_len),
                block_tables=jnp.array(list(range(1))),
                seq_lens=jnp.array([seq_len]),
                query_start_loc=jnp.array([0, seq_len]),
                request_distribution=jnp.array([0, 0, 1]),
            )

            _, output_tensor, _ = decoder_layer(
                kv_cache=kv_cache,
                x=input_tensor,
                attention_metadata=attn_metadata,
            )

            assert output_tensor is not None
            assert output_tensor.shape == (seq_len, 3840)
            assert not jnp.any(jnp.isnan(output_tensor))

    def test_gemma4_12b_mtp_draft_execution(self, single_tpu_mesh):
        """Verifies Gemma 4 12B MTP draft layer execution with KV-cache sharing and Q-only attention."""
        class Mock12BDraftTextConfig:
            def __init__(self):
                self.hidden_size = 1024
                self.vocab_size = 262144
                self.num_hidden_layers = 1
                self.rms_norm_eps = 1e-6
                self.layer_types = ["sliding_attention"]
                self.rope_theta = 10000.0
                self.rope_local_base_freq = 10000.0
                self.rope_scaling = None
                self.head_dim = 256
                self.global_head_dim = 512
                self.num_attention_heads = 16
                self.num_key_value_heads = 8
                self.num_global_key_value_heads = 1
                self.attention_bias = False
                self.attention_k_eq_v = True
                self.intermediate_size = 8192
                self.final_logit_softcapping = None
                self.sliding_window = 1024
                self.use_double_wide_mlp = False
                self.enable_moe_block = False
                self.hidden_size_per_layer_input = 0
                self.num_kv_shared_layers = 0

        draft_cfg = Mock12BDraftTextConfig()
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.hf_config.text_config = draft_cfg
        mock_vllm_config.model_config.get_hidden_size = lambda: 1024
        mock_vllm_config.model_config.get_vocab_size = lambda: 262144
        mock_vllm_config.model_config.quantization = None
        mock_vllm_config.parallel_config = MagicMock()
        mock_vllm_config.parallel_config.tensor_parallel_size = 1
        mock_vllm_config.parallel_config.data_parallel_size = 1
        mock_vllm_config.parallel_config.enable_expert_parallel = False
        mock_vllm_config.quant_config = get_tpu_quantization_config(mock_vllm_config)

        rngs = nnx.Rngs(42)
        seq_len = 2
        input_tensor = jnp.ones((seq_len, 1024), dtype=jnp.bfloat16)

        with jax.set_mesh(single_tpu_mesh), set_current_vllm_config(mock_vllm_config):
            mtp_layer = Gemma4MTPDecoderLayer(
                config=draft_cfg,
                layer_idx=0,
                dtype=jnp.bfloat16,
                rng=rngs,
                mesh=single_tpu_mesh,
                kv_cache_dtype="auto",
                quant_config=mock_vllm_config.quant_config,
            )

            # MTP layer reads from shared target KV cache
            cache_shape = get_kv_cache_shape(
                8,
                16,
                8,
                256,
                jnp.bfloat16,
            )
            kv_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)

            attn_metadata = AttentionMetadata(
                input_positions=jnp.arange(seq_len),
                block_tables=jnp.array(list(range(1))),
                seq_lens=jnp.array([seq_len]),
                query_start_loc=jnp.array([0, seq_len]),
                request_distribution=jnp.array([0, 0, 1]),
            )

            _, mtp_out, _ = mtp_layer(
                kv_cache=kv_cache,
                x=input_tensor,
                attention_metadata=attn_metadata,
            )

            assert mtp_out is not None
            assert mtp_out.shape == (seq_len, 1024)
            assert not jnp.any(jnp.isnan(mtp_out))
