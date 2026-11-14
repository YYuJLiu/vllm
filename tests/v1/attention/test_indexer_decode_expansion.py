# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU test: the indexer's per-token decode expansion is sized from the device lens.

``supports_device_cpu_query_lens_mismatch`` lets adaptive verification rewrite the
device ``query_start_loc`` below the host plan. The expansion repeats a *device*
tensor (``decode_lens``) and used to size it from ``decode_lens_cpu.sum()``; when the
two disagree ``torch.repeat_interleave`` raises -- a device-side assert on CUDA (the
CUDA context is poisoned and the engine dies), a plain RuntimeError on CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder


class _FakeSpecConfig:
    enable_adaptive_verification = True


class _FakeVllmConfig:
    speculative_config = _FakeSpecConfig()


def _make_builder(num_tokens: int) -> SimpleNamespace:
    """The attributes ``_prepare_decode_tensors`` touches, on CPU tensors."""
    return SimpleNamespace(
        vllm_config=_FakeVllmConfig(),
        supports_varlen=True,  # take the per-token expansion path
        decode_seq_lens_buffer=torch.zeros(num_tokens, dtype=torch.int32),
        expanded_block_table_buffer=torch.zeros((num_tokens, 1), dtype=torch.int32),
        arange_buffer=torch.arange(num_tokens, dtype=torch.int32),
        decode_lens_buffer=torch.zeros(num_tokens, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "device_lens, host_lens, expected_seq_lens",
    [
        # Adaptive verification admitted fewer drafts than the host plan scheduled:
        # the device total (4) is below the host total (5), and the former must win.
        pytest.param([2, 2], [3, 2], [9, 10, 5, 6, 0], id="device-trimmed"),
        # Control: device and host agree.
        pytest.param([3, 2], [3, 2], [8, 9, 10, 4, 5], id="no-trim"),
    ],
)
def test_expansion_is_sized_by_the_device_lens(device_lens, host_lens, expected_seq_lens):
    num_decodes = len(device_lens)
    num_decode_tokens = sum(host_lens)  # the host plan sizes the buffers
    seq_lens = torch.tensor([10, 7], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    decode_lens = torch.tensor(device_lens, dtype=torch.int32)
    decode_lens_cpu = torch.tensor(host_lens, dtype=torch.int32)
    block_table = torch.zeros((num_decodes, 1), dtype=torch.int32)

    builder = _make_builder(num_decode_tokens)
    builder.decode_lens_buffer[:num_decodes] = decode_lens

    seq_out, block_out, lens_out, n_tokens, native = (
        DeepseekV32IndexerMetadataBuilder._prepare_decode_tensors(
            builder,
            seq_lens=seq_lens,
            block_table=block_table,
            decode_lens=decode_lens,
            decode_lens_cpu=decode_lens_cpu,
            query_start_loc=query_start_loc,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            use_native=False,
            next_n=1,
            max_decode_len=max(device_lens),
        )
    )

    # The expansion covers exactly the device total; slots past it stay zeroed instead
    # of holding a stale position from a previous batch.
    assert seq_out.tolist() == expected_seq_lens
    # All requests become single-token rows.
    assert lens_out.tolist() == [1] * num_decode_tokens
    assert block_out.shape[0] == num_decode_tokens
    assert n_tokens == num_decode_tokens
    assert native is False
