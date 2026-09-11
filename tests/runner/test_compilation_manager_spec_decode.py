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
"""Unit tests for the speculative-decoding precompilation helpers.

These tests drive `CompilationManager._precompile_*_helpers` against a mocked
runner on CPU. `_run_compilation` is replaced with a capture hook so no XLA
lowering happens; we only assert on the *structure* of the traced inputs, which
is what determines whether the precompiled executable matches the one the
runtime asks for later.
"""

from unittest import mock

import jax
import numpy as np
import pytest

from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata,
    GroupedAttentionMetadata,
)
from tpu_inference.runner.compilation_manager import CompilationManager


class _FakeBlockTable:
    """Stands in for `runner.input_batch.block_table[gid]`.

    Each KV cache group has its own `max_num_blocks_per_req`, which is exactly
    why a single group's block table cannot be broadcast to all groups.
    """

    def __init__(self, max_num_blocks_per_req: int, max_num_reqs: int):
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self._max_num_reqs = max_num_reqs

    def get_cpu_tensor(self):
        return np.zeros(
            (self._max_num_reqs, self.max_num_blocks_per_req), dtype=np.int32
        )


def _make_runner(blocks_per_req_per_group: list[int], max_num_reqs: int = 4):
    """Builds a mocked TPUModelRunner with `len(...)` KV cache groups."""
    devices = np.array(jax.devices()[:1]).reshape(1, 1, 1, 1)
    mesh = jax.make_mesh(devices.shape, ("data", "attn_dp", "expert", "model"))

    runner = mock.MagicMock()
    runner.mesh = mesh
    runner.vllm_config.parallel_config.tensor_parallel_size = 1
    runner.max_num_reqs = max_num_reqs
    runner.dp_size = 1
    runner.num_tokens_paddings = [8]
    runner.attn_num_reqs_paddings = [max_num_reqs]
    runner.uses_mrope = False
    runner.model_config.get_hidden_size.return_value = 16
    runner.model_config.dtype = "bfloat16"
    runner.speculative_config.draft_model_config.get_hidden_size.return_value = 8
    runner.kv_cache_config.has_mamba_layers = False
    runner.kv_caches = []

    groups = []
    for gid in range(len(blocks_per_req_per_group)):
        group = mock.MagicMock()
        group.layer_names = [f"layer.{58 + gid}"]
        groups.append(group)
    runner.kv_cache_config.kv_cache_groups = groups
    runner.input_batch.block_table = [
        _FakeBlockTable(n, max_num_reqs) for n in blocks_per_req_per_group
    ]
    return runner


def _capture_precompile(runner, method_name: str):
    """Runs a precompile helper, returning {name: (args, kwargs)} captures."""
    manager = CompilationManager(runner)
    captured = {}

    def _fake_run_compilation(name, fn, *args, **kwargs):
        captured.setdefault(name, []).append(args)

    manager._run_compilation = _fake_run_compilation
    getattr(manager, method_name)()
    return captured


def _sole_attn_metadata(args):
    """Extracts the single (Grouped)AttentionMetadata from a captured arglist."""
    found = [
        a for a in args if isinstance(a, (AttentionMetadata, GroupedAttentionMetadata))
    ]
    assert len(found) == 1, f"expected exactly one metadata arg, got {found}"
    return found[0]


def test_precompile_mtp_helpers_builds_grouped_metadata_for_multi_group():
    """NEW-7: `_precompile_mtp_helpers` must mirror the runtime metadata TYPE.

    `tpu_runner._prepare_inputs` builds a `GroupedAttentionMetadata` whenever
    there is more than one KV cache group (the SWA / hybrid layout Gemma 4 MTP
    serves under). `GroupedAttentionMetadata` is registered as its own pytree
    node and flattens to one leaf set per group, so tracing a flat
    `AttentionMetadata` here guarantees a jit cache miss at serving time ->
    `ForbidCompile` or a mid-serving recompile.

    `_precompile_backbone_helper` already branches correctly; this asserts the
    MTP helper does the same.
    """
    # 6 groups mirrors the real Gemma 4 31B SWA layout (5 sliding + 1 full).
    # The differing blocks-per-request per group is the crux: a single
    # broadcast table would have the wrong shape for five of the six groups.
    blocks_per_req = [3, 3, 3, 3, 3, 7]
    runner = _make_runner(blocks_per_req)
    captured = _capture_precompile(runner, "_precompile_mtp_helpers")

    assert "drafter_propose" in captured, "drafter_propose was never precompiled"

    for name, arglists in captured.items():
        for args in arglists:
            metadata = _sole_attn_metadata(args)
            assert isinstance(metadata, GroupedAttentionMetadata), (
                f"{name} traced a {type(metadata).__name__}; the runtime "
                f"builds a GroupedAttentionMetadata for "
                f"{len(blocks_per_req)} KV cache groups"
            )
            assert len(metadata.groups) == len(blocks_per_req)
            # Each group must carry ITS OWN block table, flattened to 1-D.
            for gid, expected_blocks in enumerate(blocks_per_req):
                assert metadata.groups[gid].block_tables.shape == (
                    runner.max_num_reqs * expected_blocks,
                ), f"{name} group {gid} has the wrong block table shape"
            assert metadata.layer_names_per_group == tuple(
                (f"layer.{58 + gid}",) for gid in range(len(blocks_per_req))
            )


def test_precompile_mtp_helpers_keeps_flat_metadata_for_single_group():
    """The 1-group (unified pool) path must stay a plain AttentionMetadata."""
    runner = _make_runner([5])
    captured = _capture_precompile(runner, "_precompile_mtp_helpers")

    assert captured, "no compilations were captured"
    for name, arglists in captured.items():
        for args in arglists:
            metadata = _sole_attn_metadata(args)
            assert isinstance(metadata, AttentionMetadata), (
                f"{name} traced a {type(metadata).__name__}; a single KV "
                f"cache group must produce a flat AttentionMetadata"
            )
            assert metadata.block_tables.shape == (runner.max_num_reqs * 5,)


def test_precompile_mtp_helpers_does_not_read_draft_group_n_minus_1():
    """NEW-8: the same wrong-KV-group construct, on the precompile side.

    `_precompile_mtp_helpers` selected `kv_cache_groups[-1]` to build one block
    table, exactly as the proposer did (NEW-3). Gemma 4 MTP allocates no draft
    KV group, so group N-1 is a *target* group. With per-group tables now built
    for the multi-group case, that extra read is both redundant and a source of
    shape disagreement with the runtime.

    Assert each group's table is materialised exactly once: an extra read of
    group N-1 means the stale single-table selection is still there.

    Must land together with the proposer fix, or the precompiled executable
    expects a different `block_tables` shape than the runtime supplies.
    """
    blocks_per_req = [3, 3, 3, 3, 3, 7]
    runner = _make_runner(blocks_per_req)
    reads: list[int] = []
    real_tables = runner.input_batch.block_table

    class _RecordingTables:

        def __getitem__(self, gid):
            reads.append(gid)
            return real_tables[gid]

    runner.input_batch.block_table = _RecordingTables()

    _capture_precompile(runner, "_precompile_mtp_helpers")

    assert reads, "no block tables were built at all"
    last_gid = len(blocks_per_req) - 1
    assert reads.count(last_gid) == reads.count(0), (
        f"group {last_gid} was read {reads.count(last_gid)} times vs "
        f"{reads.count(0)} for group 0; the draft-group N-1 selection is "
        f"still present. reads={reads}")


def test_precompile_eagle3_helpers_still_reads_draft_group_n_minus_1():
    """Negative control: genuine eagle3 keeps its trailing draft group."""
    blocks_per_req = [3, 5]
    runner = _make_runner(blocks_per_req)
    reads: list[int] = []
    real_tables = runner.input_batch.block_table

    class _RecordingTables:

        def __getitem__(self, gid):
            reads.append(gid)
            return real_tables[gid]

    runner.input_batch.block_table = _RecordingTables()
    runner.model_config.get_hidden_size.return_value = 16

    _capture_precompile(runner, "_precompile_eagle3_helpers")

    assert reads == [len(blocks_per_req) - 1] * len(reads), (
        f"eagle3 must read only the trailing draft group, got {reads}")


def _dispatch_target(method: str, model_type: str) -> str:
    """Runs `_precompile_speculative_decoding` and reports which helper ran."""
    runner = _make_runner([5])
    runner.speculative_config.method = method

    draft_hf_config = mock.MagicMock()
    draft_hf_config.model_type = model_type
    draft_hf_config.architectures = []
    runner.speculative_config.draft_model_config.hf_config = draft_hf_config
    runner.speculative_config.draft_model_config.architectures = []
    # Mirror the upstream circular definition: use_gemma4_mtp() only returns
    # True once `method` has already been rewritten to "mtp".
    runner.speculative_config.use_gemma4_mtp.return_value = (
        method == "mtp" and model_type == "gemma4_mtp")

    manager = CompilationManager(runner)
    called = []
    for helper in ("_precompile_eagle3_helpers", "_precompile_dflash_helpers",
                   "_precompile_mtp_helpers"):
        setattr(manager, helper, lambda h=helper: called.append(h))
    for noop in ("_precompile_rejection_sampler",
                 "_precompile_extract_last_sampled_tokens",
                 "_precompile_extract_draft_token_ids",
                 "_precompile_process_and_extend_logits",
                 "_precompile_extend_logits_simple",
                 "_precompile_select_from_array_spec_decode"):
        setattr(manager, noop, lambda: None)

    manager._precompile_speculative_decoding()
    assert len(called) == 1, f"expected exactly one helper, got {called}"
    return called[0]


@pytest.mark.parametrize("method,model_type,expected", [
    # Happy path: hf_config_override rewrote gemma4_assistant -> gemma4_mtp.
    ("mtp", "gemma4_mtp", "_precompile_mtp_helpers"),
    # An explicit --speculative-method eagle3 short-circuits the override, so
    # `method` stays "eagle3" on a genuine Gemma 4 MTP model.
    ("eagle3", "gemma4_mtp", "_precompile_mtp_helpers"),
    # The raw checkpoint ships model_type gemma4_assistant, which upstream
    # does not recognise at all.
    ("eagle3", "gemma4_assistant", "_precompile_mtp_helpers"),
    ("mtp", "gemma4_assistant", "_precompile_mtp_helpers"),
    # Non-Gemma models must keep their existing routing.
    ("eagle3", "llama", "_precompile_eagle3_helpers"),
    ("dflash", "llama", "_precompile_dflash_helpers"),
    ("mtp", "deepseek_v3", "_precompile_mtp_helpers"),
])
def test_precompile_dispatch_honours_is_gemma4_mtp(method, model_type,
                                                   expected):
    """NEW-6: precompile dispatch must use `is_gemma4_mtp`, not just `method`.

    `is_gemma4_mtp` appears in eagle3.py, speculative_decoding_manager.py and
    kv_cache_manager.py, but nowhere in compilation_manager.py -- exactly the
    inconsistency the helper was introduced to eliminate. A Gemma 4 MTP model
    reaching `_precompile_eagle3_helpers` is traced with a 3-tuple of
    aux_hidden_states (runtime supplies 1), MLP-sharded draft hidden states
    (runtime uses ATTN_DATA) and 1-D positions only. All three are jit cache
    key participants, so the result is a ForbidCompile error or a silent
    mid-serving recompile.

    Safe only now that `_precompile_mtp_helpers` builds the correct metadata
    type for multi-group models; widening this dispatch beforehand would have
    increased exposure to that bug instead of reducing it.
    """
    assert _dispatch_target(method, model_type) == expected
