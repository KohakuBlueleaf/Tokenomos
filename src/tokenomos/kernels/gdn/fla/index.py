# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch
import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def chunk_indices_host(cu_host: list[int], chunk_size: int) -> list[list[int]]:
    """``[sequence, chunk]`` per chunk of every sequence, from host offsets (no sync)."""
    return [
        [s, c]
        for s in range(len(cu_host) - 1)
        for c in range(triton.cdiv(cu_host[s + 1] - cu_host[s], chunk_size))
    ]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """The device chunk table read back from ``cu_seqlens`` (syncs; callers on the engine's
    path pass ``chunk_indices`` from ``chunk_indices_host``)."""
    rows = chunk_indices_host(cu_seqlens.tolist(), chunk_size)
    return torch.tensor(rows, device=cu_seqlens.device, dtype=cu_seqlens.dtype).view(-1, 2)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_zeros(1), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
