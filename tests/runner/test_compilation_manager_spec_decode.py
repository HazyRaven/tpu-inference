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
