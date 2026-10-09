# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import pytest
import torch
from test_gluon_dsv4_indexer_gfx1250 import _inputs
from utils import is_cdna5

if not is_cdna5():
    pytest.skip("GFX1250 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsv4._triton.indexer import _indexer_logits
from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv4.indexer import (
    _dsv4_mxfp4_logits,
)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize(
    "offset,padding,large_cache", [(1, 15, False), (128, 128, False), (0, 0, True)]
)
def test_global_key_layout_graph_refresh(heads, prefill, offset, padding, large_cache):
    index_q, weights, original, _, _ = _inputs(2, heads, 8)
    stride = 4352 + padding
    pages = (2**31 // stride + 9) if large_cache else 8
    storage = torch.empty(pages * stride + offset, device="cuda", dtype=torch.uint8)
    cache = storage[offset:].reshape(pages, stride)[:, :4352]
    page_base = pages - 8
    live_cache = cache[page_base:]
    live_cache.copy_(original)
    table = torch.tensor([[3, 1, 0, 2], [7, 5, 4, 6]], device="cuda", dtype=torch.int32)
    table.add_(page_base)
    lengths = torch.tensor([193, 149], device="cuda", dtype=torch.int32)
    cu = (
        torch.tensor([0, 256, 512], device="cuda", dtype=torch.int32)
        if prefill
        else None
    )
    starts = (
        torch.tensor([0, 256], device="cuda", dtype=torch.int32) if prefill else None
    )

    def invoke():
        return _dsv4_mxfp4_logits(
            index_q,
            weights,
            cache,
            lengths,
            table,
            page_size=64,
            max_candidates=257,
            cu_seq_lens=cu,
            cu_seqlen_k_start=starts,
        )

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = invoke()
    for changed in (False, True):
        if changed:
            index_q[0].bitwise_xor_(85)
            weights.mul_(-0.75)
            live_cache[:, :4096].bitwise_xor_(19)
            table.copy_(table.flip(1))
            lengths.copy_(torch.tensor([131, 0], device="cuda", dtype=torch.int32))
        actual.fill_(float("nan"))
        graph.replay()
        expected, _ = _indexer_logits(
            index_q,
            weights,
            cache,
            lengths,
            table,
            page_size=64,
            max_candidates=257,
            cu_seq_lens=cu,
            starts=starts,
        )
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-5)
        torch.testing.assert_close(actual, invoke(), atol=0, rtol=0)
