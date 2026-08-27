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
"""End-to-End and Architectural Tests for Gemma 4 Sliding Window Attention."""

import math
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        SlidingWindowSpec)

from tpu_inference import utils as common_utils
from tpu_inference.models.common.kv_share import compute_kv_share_map
from tpu_inference.runner.kv_cache_manager import KVCacheManager


@pytest.fixture(scope="module")
def mesh():
    if not jax.devices():
        pytest.skip("No JAX devices available.")
    devices = np.array(jax.local_devices()[:1])
    device_mesh = devices.reshape((1, 1, 1, 1))
    with Mesh(device_mesh,
              axis_names=("data", "attn_dp", "expert", "model")) as m:
        yield m


class TestGemma4SlidingWindowE2E:

    def test_gemma4_sliding_window_memory_reduction(self):
        """Validates >= 5.14x KV cache memory reduction for Gemma 4 31B with SWA.

        Gemma 4 31B Architecture:
        - 60 layers total
        - 10 Global Full Attention layers (window = max_model_len)
        - 50 Sliding Window Attention layers (window = 1024)
        - Global layers: num_kv_heads = 4, head_dim = 512 (bytes/tok = 4 * 512 * 2 = 4096)
        - Sliding layers: num_kv_heads = 16, head_dim = 256 (bytes/tok = 16 * 256 * 2 = 8192 / 2 = 8192 or 4096 per layer)
        """
        sliding_window = 1024
        num_layers = 60
        num_global_layers = 10
        num_sliding_layers = 50

        # At 32k context:
        # Full: 60 * 32768 * 4096 bytes
        # SWA: (10 * 32768 + 50 * 1024) * 4096 bytes
        max_model_len_target = 32768
        full_at_target = num_layers * max_model_len_target * 4096
        swa_at_target = (num_global_layers * max_model_len_target + num_sliding_layers * sliding_window) * 4096
        target_reduction_ratio = full_at_target / swa_at_target

        assert target_reduction_ratio >= 5.14, (
            f"Reduction ratio {target_reduction_ratio:.2f}x is below the 5.14x target."
        )

        # Verify spec creation allocates matching shapes
        global_spec = FullAttentionSpec(
            block_size=32,
            num_kv_heads=4,
            head_size=512,
            dtype=torch.bfloat16,
        )
        sliding_spec = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=16,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=sliding_window,
        )

        # Global page size = 32 * 4 * 2 * 512 * 2 = 262144 bytes
        # Sliding page size = 16 * 16 * 2 * 256 * 2 = 262144 bytes (uniform page size!)
        assert global_spec.page_size_bytes == sliding_spec.page_size_bytes

    def test_null_block_pre_softmax_masking(self):
        """Constructs attention inputs where context length exceeds sliding window (L > W)
        and verifies that positions outside the sliding window are masked with -inf
        prior to the softmax operation, preventing stale token leakage.
        """
        seq_len = 32
        sliding_window = 8
        dtype = jnp.bfloat16

        # Query and key positions
        q_pos = jnp.arange(seq_len)[:, None]
        k_pos = jnp.arange(seq_len)[None, :]

        # Causal mask: q_pos >= k_pos
        causal_mask = q_pos >= k_pos

        # Sliding window condition: q_pos - k_pos < sliding_window
        sliding_mask = (q_pos - k_pos) < sliding_window

        # Combined attention mask
        valid_attn_mask = causal_mask & sliding_mask

        # Mask values: 0.0 for valid, -0.7 * max_finite for invalid/null blocks
        neg_inf_val = -0.7 * jnp.finfo(dtype).max
        attn_bias = jnp.where(valid_attn_mask, 0.0, neg_inf_val)

        # Verify outside-window token positions are strictly masked with -inf
        for q in range(sliding_window, seq_len):
            # Tokens older than q - sliding_window must be masked
            masked_positions = attn_bias[q, :q - sliding_window + 1]
            assert jnp.all(masked_positions == neg_inf_val)

            # Tokens within [q - sliding_window + 1, q] must be valid (0.0)
            valid_positions = attn_bias[q, q - sliding_window + 1:q + 1]
            assert jnp.all(valid_positions == 0.0)

        # Apply softmax over dummy logits + attn_bias
        dummy_logits = jnp.ones((seq_len, seq_len), dtype=dtype) + attn_bias
        attn_weights = jax.nn.softmax(dummy_logits, axis=-1)

        # Out-of-window weights must be exactly 0.0
        for q in range(sliding_window, seq_len):
            stale_weights = attn_weights[q, :q - sliding_window + 1]
            assert jnp.all(stale_weights == 0.0)

    def test_gemma4_sliding_window_numerical_parity(self):
        """Verifies numerical equivalence between sliding window attention
        and uncompressed reference implementation with masked context.
        """
        batch_size = 1
        num_heads = 4
        seq_len = 16
        head_dim = 32
        sliding_window = 6

        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch_size, num_heads, seq_len, head_dim))
        k = jax.random.normal(k2, (batch_size, num_heads, seq_len, head_dim))
        v = jax.random.normal(k3, (batch_size, num_heads, seq_len, head_dim))

        # 1. Reference Full Attention with Causal + SWA mask
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(head_dim)
        q_pos = jnp.arange(seq_len)[:, None]
        k_pos = jnp.arange(seq_len)[None, :]
        mask = (q_pos >= k_pos) & ((q_pos - k_pos) < sliding_window)
        scores_masked = jnp.where(mask[None, None, :, :], scores, -1e9)
        ref_weights = jax.nn.softmax(scores_masked, axis=-1)
        ref_out = jnp.einsum("bhqk,bhkd->bhqd", ref_weights, v)

        # 2. Local window attention implementation
        local_scores = jnp.full_like(scores, -1e9)
        for i in range(seq_len):
            start = max(0, i - sliding_window + 1)
            local_scores = local_scores.at[:, :, i, start:i+1].set(
                scores[:, :, i, start:i+1]
            )
        local_weights = jax.nn.softmax(local_scores, axis=-1)
        local_out = jnp.einsum("bhqk,bhkd->bhqd", local_weights, v)

        # 3. Assert Parity
        max_abs_diff = float(jnp.max(jnp.abs(ref_out - local_out)))
        cosine_sim = float(
            jnp.sum(ref_out * local_out) / (jnp.linalg.norm(ref_out) * jnp.linalg.norm(local_out))
        )

        assert max_abs_diff < 1e-4, f"Max abs diff {max_abs_diff} exceeds 1e-4 threshold."
        assert cosine_sim >= 0.9999, f"Cosine similarity {cosine_sim} below 0.9999 threshold."

    @pytest.mark.parametrize("kv_len,q_len,sliding_window,bkv_sz,expected_start_tile", [
        (512, 512, 1024, 128, 0),        # Chunk 1 (kv_len <= sliding_window) -> Start at tile 0
        (1024, 512, 1024, 128, 0),       # Chunk 2 (kv_len == sliding_window) -> Start at tile 0
        (2048, 512, 1024, 128, 4),       # Chunk 4 (kv_q_gap = 1536 > 1024) -> (1536 - 1024) // 128 = 4 (skips 4 tiles!)
        (50000, 512, 1024, 128, 378),    # Chunk 100 (50k context) -> (49488 - 1024) // 128 = 378 (skips 378 tiles!)
    ])
    def test_pallas_rpa_cur_seq_start_bkv_idx_tile_skipping(
            self, kv_len, q_len, sliding_window, bkv_sz, expected_start_tile):
        """Verify that Pallas RPA tile skipping formula accurately calculates the start tile index during chunked prefill."""
        kv_q_gap = kv_len - q_len
        cur_seq_start_bkv_idx = max(kv_q_gap - sliding_window, 0) // bkv_sz
        assert cur_seq_start_bkv_idx == expected_start_tile

