from unittest.mock import patch
import jax
import jax.numpy as jnp
from tests.models.jax.test_gemma4_mtp import _setup_test_model
from tpu_inference.models.jax.utils.qwix.qwix_utils import qwix_quantize_nnx_model


def test_qwix_quantize_nnx_model_mtp_hidden_states(rng, mesh, mock_vllm_config):
    """Verifies that qwix_quantize_nnx_model inspects signature and supplies dummy hidden_states for MTP."""
    model, vllm_config = _setup_test_model(rng, mesh, mock_vllm_config)
    with patch("tpu_inference.models.jax.utils.qwix.qwix_utils.qwix.quantize_model") as mock_quantize, \
         patch("tpu_inference.utils.hbm_usage_gb", return_value=[(0.0, 0.0)]):
        mock_quantize.return_value = model
        quantized_model = qwix_quantize_nnx_model(
            model=model,
            qwix_config=[],
            rng=rng,
            mesh=mesh,
            num_hidden_layers=4,
            kv_cache_block_size=16,
            kv_cache_num_kv_heads=16,
            kv_cache_head_size=256,
            kv_cache_dtype="auto",
        )
        assert quantized_model is not None
        assert mock_quantize.call_count == 1
        _, kwargs = mock_quantize.call_args
        assert "hidden_states" in kwargs
        assert kwargs["hidden_states"].shape[-1] == 5120
