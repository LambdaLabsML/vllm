# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused RoPE + paged KV-cache write for unquantized (BF16/FP16) KV caches.

Replaces, at decode batch sizes, the separate rotary-embedding kernels for q and
k and ``reshape_and_cache_flash``: each of them is a latency-bound launch of a
few microseconds per attention layer. Used by RopeKVCacheFusionPass through
``AttentionImpl.do_rope_and_kv_cache_update``.

Numerics match the compiled native rotary embedding: inputs and cos/sin are
upcast to FP32 and each output is rounded once.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rope_reshape_and_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    positions_ptr,
    cos_sin_ptr,
    slot_mapping_ptr,
    key_cache_ptr,
    value_cache_ptr,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    v_stride_t,
    v_stride_h,
    cos_sin_stride,
    kc_stride_b,
    kc_stride_n,
    kc_stride_h,
    vc_stride_b,
    vc_stride_n,
    vc_stride_h,
    block_size,
    num_cache_tokens,
    num_q_heads: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    half: tl.constexpr = head_size // 2

    pos = tl.load(positions_ptr + token)
    offs = tl.arange(0, half)
    cos = tl.load(cos_sin_ptr + pos * cos_sin_stride + offs).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * cos_sin_stride + half + offs).to(tl.float32)
    if is_neox:
        idx1 = offs
        idx2 = offs + half
    else:
        idx1 = 2 * offs
        idx2 = 2 * offs + 1

    if head < num_q_heads:
        base = q_ptr + token * q_stride_t + head * q_stride_h
        x1 = tl.load(base + idx1).to(tl.float32)
        x2 = tl.load(base + idx2).to(tl.float32)
        tl.store(base + idx1, (x1 * cos - x2 * sin).to(q_ptr.dtype.element_ty))
        tl.store(base + idx2, (x2 * cos + x1 * sin).to(q_ptr.dtype.element_ty))
        return

    kv_head = head - num_q_heads
    base = k_ptr + token * k_stride_t + kv_head * k_stride_h
    x1 = tl.load(base + idx1).to(tl.float32)
    x2 = tl.load(base + idx2).to(tl.float32)
    o1 = (x1 * cos - x2 * sin).to(k_ptr.dtype.element_ty)
    o2 = (x2 * cos + x1 * sin).to(k_ptr.dtype.element_ty)
    # the op contract mutates key in place, like the unfused rotary embedding
    tl.store(base + idx1, o1)
    tl.store(base + idx2, o2)

    if token >= num_cache_tokens:  # CUDA-graph padding beyond slot_mapping
        return
    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
    if slot < 0:  # padding token
        return
    block = slot // block_size
    offset = slot % block_size
    kc = key_cache_ptr + block * kc_stride_b + offset * kc_stride_n + kv_head * kc_stride_h
    tl.store(kc + idx1, o1)
    tl.store(kc + idx2, o2)
    offs_v = tl.arange(0, head_size_v)
    v = tl.load(v_ptr + token * v_stride_t + kv_head * v_stride_h + offs_v)
    vc = value_cache_ptr + block * vc_stride_b + offset * vc_stride_n + kv_head * vc_stride_h
    tl.store(vc + offs_v, v)


def can_rope_reshape_and_cache(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
) -> bool:
    """Shapes the kernel handles: full-head rotary dim, power-of-2 head sizes,
    unit-stride head dims, cache dtype equal to the activations' dtype."""
    head_size = query.shape[-1]
    head_size_v = value.shape[-1]
    return (
        cos_sin_cache.shape[-1] == head_size
        and key.shape[-1] == head_size
        and head_size >= 2
        and head_size & (head_size - 1) == 0
        and head_size_v & (head_size_v - 1) == 0
        and key_cache.dtype == key.dtype
        and query.stride(-1) == 1
        and key.stride(-1) == 1
        and value.stride(-1) == 1
        and cos_sin_cache.stride(-1) == 1
        and key_cache.stride(-1) == 1
    )


def rope_reshape_and_cache(
    query: torch.Tensor,  # [num_tokens, num_heads, head_size], rotated in place
    key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size], rotated in place
    value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size_v]
    positions: torch.Tensor,  # [num_tokens]
    cos_sin_cache: torch.Tensor,  # [max_position, head_size]: cos | sin
    is_neox: bool,
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size_v]
    slot_mapping: torch.Tensor,  # [num_actual_tokens]; < 0 for padding
) -> None:
    # q/k/v may be padded (CUDA graphs) while slot_mapping is not: rotate every
    # row, like the unfused rotary embedding, but cache-write only the rows
    # slot_mapping covers, like reshape_and_cache_flash.
    num_tokens = query.shape[0]
    if num_tokens == 0:
        return
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    grid = (num_tokens, num_q_heads + num_kv_heads)
    _rope_reshape_and_cache_kernel[grid](
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        slot_mapping,
        key_cache,
        value_cache,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        cos_sin_cache.stride(0),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        key_cache.shape[1],
        slot_mapping.shape[0],
        num_q_heads=num_q_heads,
        head_size=query.shape[-1],
        head_size_v=value.shape[-1],
        is_neox=is_neox,
    )
