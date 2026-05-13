# vLLM BFLA (Block-Filtered Long-Context Attention) Triton Backend

# BFLA modifications:
#   Author: Chong Wu <imroxaswc@gmail.com>

# This file is based on the vLLM Triton Unified Attention implementation.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Original Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>



import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
is_batch_invariant = envs.VLLM_BATCH_INVARIANT
float8_info = torch.finfo(current_platform.fp8_dtype())

# Experimental only: reuse the first BFLA block mask across layers for the
# same batch shape/metadata. Disabled by default because it is approximate.
_BFLA_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _find_prefill_like_suffix(
    query_start_loc_cpu: torch.Tensor | None,
    seq_lens_cpu: torch.Tensor | None,
    num_actual_tokens: int,
    *,
    min_prefill_tokens: int,
) -> tuple[int, int, int, int] | None:
    """Find a reordered prefill/extend suffix.

    vLLM usually schedules decode rows before prefill-like rows. For BFLA we
    can sparsify rows with query_len > 1, including chunked prefill/extend
    where context_len = seq_len - query_len > 0. Decode rows (query_len == 1)
    and malformed rows stay dense.

    Returns:
        (first_prefill_req, prefix_tokens, num_prefill_reqs, prefill_tokens)
    """
    if query_start_loc_cpu is None or seq_lens_cpu is None:
        return None
    if query_start_loc_cpu.numel() <= 1:
        return None

    query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    seq_lens_cpu = seq_lens_cpu[: query_lens.numel()]
    is_prefill_like = (query_lens > 1) & (seq_lens_cpu >= query_lens)
    if not torch.any(is_prefill_like):
        return None

    first_prefill = int(is_prefill_like.int().argmax().item())
    if not torch.all(is_prefill_like[first_prefill:]).item():
        return None

    prefix_tokens = int(query_start_loc_cpu[first_prefill].item())
    prefill_tokens = int(num_actual_tokens - prefix_tokens)
    num_prefill_reqs = int(query_lens.numel() - first_prefill)
    if prefill_tokens < min_prefill_tokens or num_prefill_reqs <= 0:
        return None
    return first_prefill, prefix_tokens, num_prefill_reqs, prefill_tokens

def _build_bfla_block_mask(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cu_seqlens_q_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    *,
    first_prefill_req: int,
    num_prefill_reqs: int,
    block_size: int,
    block_n: int,
    attn_block_n: int,
    threshold: float,
    min_keep_blocks: int,
    keep_ratio: float,
    local_blocks: int,
    keep_mass: float,
    softmax_scale: float,
    pool_mode: str,
    spec_stride: int,
    spec_prob: float,
    spec_seed: int,
) -> torch.Tensor | None:
    """Build [num_seqs, num_kv_heads, max_q_blocks, max_kv_tiles] mask.

    Non-prefill requests are initialized to 1 so decode work remains
    dense. Prefill-like rows, including chunked prefill/extend, receive a
    coarse block-importance mask computed from pooled Q and paged K cache.
    """
    device = q.device
    num_seqs = int(seq_lens_cpu.numel())
    num_query_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[2])
    if num_query_heads % num_kv_heads != 0:
        return None
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_dim = int(q.shape[2])

    query_lens = cu_seqlens_q_cpu[1:] - cu_seqlens_q_cpu[:-1]
    max_query_len = int(query_lens.max().item())
    max_seq_len = int(seq_lens_cpu.max().item())
    max_q_blocks = triton.cdiv(max_query_len, block_n)
    max_kv_tiles = triton.cdiv(max_seq_len, block_n)
    max_attn_q_blocks = triton.cdiv(max_query_len, attn_block_n)
    max_attn_kv_tiles = triton.cdiv(max_seq_len, attn_block_n)
    expand_ratio = block_n // attn_block_n if block_n >= attn_block_n else 1
    if block_n % attn_block_n != 0:
        return None

    sparse_mask = torch.ones(
        (num_seqs, num_kv_heads, max_attn_q_blocks, max_attn_kv_tiles),
        device=device,
        dtype=torch.int32,
    )

    flat_group_tokens = 64
    use_flat64 = pool_mode == "flat64"
    if use_flat64 and block_n % flat_group_tokens != 0:
        return None

    def _pool_block_tensor(x: torch.Tensor) -> torch.Tensor:
        if use_flat64:
            # [B, block_n, H, D] -> [H, B, G, 64*D]. This keeps token
            # order inside each 64-token group and scores all 4x4 group pairs.
            groups = block_n // flat_group_tokens
            return x.view(x.shape[0], groups, flat_group_tokens, x.shape[2], x.shape[3]).permute(3, 0, 1, 2, 4).reshape(x.shape[2], x.shape[0], groups, flat_group_tokens * x.shape[3])
        if pool_mode == "center":
            idx = min(block_n // 2, x.shape[1] - 1)
            return x[:, idx]
        if pool_mode == "maxabs":
            idx = torch.argmax(x.abs(), dim=1, keepdim=True)
            return torch.gather(x, 1, idx).squeeze(1)
        return x.mean(dim=1)

    for req in range(first_prefill_req, first_prefill_req + num_prefill_reqs):
        q_start = int(cu_seqlens_q_cpu[req].item())
        q_end = int(cu_seqlens_q_cpu[req + 1].item())
        query_len = q_end - q_start
        seq_len = int(seq_lens_cpu[req].item())
        context_len = seq_len - query_len
        if query_len <= 1 or context_len < 0:
            continue

        q_req = q[q_start:q_end]
        q_blocks = triton.cdiv(query_len, block_n)
        kv_tiles = triton.cdiv(seq_len, block_n)
        if q_blocks == 0 or kv_tiles == 0:
            continue

        q_pad = torch.zeros(
            (q_blocks * block_n, num_query_heads, head_dim),
            device=device,
            dtype=q.dtype,
        )
        q_pad[: q_req.shape[0]].copy_(q_req)
        q_low = _pool_block_tensor(q_pad.view(q_blocks, block_n, num_query_heads, head_dim))
        if not use_flat64:
            q_low = q_low.permute(1, 0, 2)  # [Hq, QB, D]

        pages = block_table[req, : triton.cdiv(seq_len, block_size)].to(torch.long)
        # Expected old Triton layout: [num_blocks, block_size, num_kv_heads, D].
        k_req = k_cache.index_select(0, pages).reshape(-1, num_kv_heads, head_dim)
        k_req = k_req[:seq_len]

        k_pad = torch.zeros(
            (kv_tiles * block_n, num_kv_heads, head_dim),
            device=device,
            dtype=k_cache.dtype,
        )
        k_pad[: k_req.shape[0]].copy_(k_req)
        k_low = _pool_block_tensor(k_pad.view(kv_tiles, block_n, num_kv_heads, head_dim))
        if not use_flat64:
            k_low = k_low.permute(1, 0, 2)  # [Hkv, KB, D]

        keep_per_kv = torch.zeros(
            (num_kv_heads, q_blocks, kv_tiles),
            device=device,
            dtype=torch.bool,
        )
        for kv_h in range(num_kv_heads):
            q_h0 = kv_h * num_queries_per_kv
            q_h1 = q_h0 + num_queries_per_kv
            if use_flat64:
                group_scores = torch.einsum(
                    "hqgf,krf->hqkgr", q_low[q_h0:q_h1], k_low[kv_h])
                scores = group_scores.amax(dim=(-1, -2))
            else:
                scores = torch.einsum("qbd,kd->qbk", q_low[q_h0:q_h1], k_low[kv_h])
            q_block_end = context_len + (torch.arange(q_blocks, device=device) + 1) * block_n - 1
            q_block_end = torch.clamp(q_block_end, max=seq_len - 1)
            k_block_start = torch.arange(kv_tiles, device=device) * block_n
            causal = k_block_start[None, :] <= q_block_end[:, None]
            scores = scores.masked_fill(~causal[None, :, :], float("-inf"))
            probs = torch.softmax(scores.float() * softmax_scale, dim=-1)
            keep = (probs > threshold).any(dim=0)

            if keep_mass >= 1.0:
                keep |= causal[None, :, :].expand_as(probs).any(dim=0)
            elif keep_mass > 0:
                sorted_probs, sorted_idx = torch.sort(probs.float(), dim=-1, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mass_keep_sorted = cumsum <= keep_mass
                mass_keep_sorted[..., 0] = True
                first_over = torch.argmax((cumsum >= keep_mass).to(torch.int32), dim=-1, keepdim=True)
                mass_keep_sorted.scatter_(-1, first_over, True)
                mass_keep = torch.zeros_like(probs, dtype=torch.bool)
                mass_keep.scatter_(-1, sorted_idx, mass_keep_sorted)
                keep |= mass_keep.any(dim=0)

            if keep_ratio > 0 or min_keep_blocks > 0:
                topk = max(min_keep_blocks, int(kv_tiles * keep_ratio))
                topk = max(1, min(topk, kv_tiles))
                _, topk_idx = torch.topk(scores.float(), k=topk, dim=-1)
                topk_keep = torch.zeros_like(scores, dtype=torch.bool)
                topk_keep.scatter_(-1, topk_idx, True)
                keep |= topk_keep.any(dim=0)

            keep_per_kv[kv_h] = keep

        q_block_end = context_len + (torch.arange(q_blocks, device=device) + 1) * block_n - 1
        q_block_end = torch.clamp(q_block_end, max=seq_len - 1)
        k_block_start = torch.arange(kv_tiles, device=device) * block_n
        causal = k_block_start[None, :] <= q_block_end[:, None]
        keep_per_kv &= causal[None, :, :]

        # Keep a local band around each chunk query block and attention sink to
        # avoid fully empty rows. The local band is expressed in absolute KV
        # tile coordinates, so it works for chunked prefill/extend too.
        q_tile_abs = (context_len + torch.arange(q_blocks, device=device) * block_n) // block_n
        q_tile_abs = q_tile_abs[:, None]
        k_idx = torch.arange(kv_tiles, device=device)[None, :]
        local_blocks_mask = max(1, triton.cdiv(local_blocks * attn_block_n, block_n))
        local = (k_idx <= q_tile_abs) & (k_idx >= q_tile_abs - local_blocks_mask)
        keep_per_kv |= local[None, :, :]
        keep_per_kv[:, :, 0] = True

        # Optional speculative rescue for tiles that the sparse policy would
        # otherwise drop. This trades some speed for robustness while staying
        # deterministic across runs and tensor-parallel ranks.
        dropped = causal[None, :, :] & ~keep_per_kv
        if spec_stride > 0:
            q_idx = torch.arange(q_blocks, device=device, dtype=torch.int64)[:, None]
            k_idx_i64 = torch.arange(kv_tiles, device=device, dtype=torch.int64)[None, :]
            stride_keep = ((q_idx * 131 + k_idx_i64 * 17 + spec_seed) % spec_stride) == 0
            keep_per_kv |= dropped & stride_keep[None, :, :]
            dropped = causal[None, :, :] & ~keep_per_kv
        if spec_prob > 0:
            prob = max(0.0, min(float(spec_prob), 1.0))
            if prob >= 1.0:
                keep_per_kv |= dropped
            else:
                q_idx = torch.arange(q_blocks, device=device, dtype=torch.int64)[None, :, None]
                k_idx_i64 = torch.arange(kv_tiles, device=device, dtype=torch.int64)[None, None, :]
                h_idx = torch.arange(num_kv_heads, device=device, dtype=torch.int64)[:, None, None]
                hashed = (
                    (q_idx + 1) * 1103515245
                    + (k_idx_i64 + 1) * 12345
                    + (h_idx + 1) * 2654435761
                    + int(spec_seed)
                ) & 0x7FFFFFFF
                random_keep = (hashed % 1000000) < int(prob * 1000000)
                keep_per_kv |= dropped & random_keep

        keep_i32 = keep_per_kv.to(torch.int32)
        if expand_ratio != 1:
            keep_i32 = keep_i32.repeat_interleave(expand_ratio, dim=1)
            keep_i32 = keep_i32.repeat_interleave(expand_ratio, dim=2)
        attn_q_blocks = triton.cdiv(query_len, attn_block_n)
        attn_kv_tiles = triton.cdiv(seq_len, attn_block_n)
        sparse_mask[req, :, :attn_q_blocks, :attn_kv_tiles] = keep_i32[:, :attn_q_blocks, :attn_kv_tiles]

    return sparse_mask


@triton.jit
def _kernel_bfla_sample_mask(
    mask_ptr,
    query_ptr,
    key_cache_ptr,
    block_tables_ptr,
    query_start_len_ptr,
    seq_lens_ptr,
    scale,
    sample_threshold,
    num_queries_per_kv: tl.int32,
    block_table_stride: tl.int64,
    mask_stride_b: tl.int64,
    mask_stride_h: tl.int64,
    mask_stride_q: tl.int64,
    mask_stride_k: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    first_prefill_req: tl.int32,
    last_prefill_req: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    D_SAMPLE: tl.constexpr,
    MAX_Q_BLOCKS: tl.constexpr,
    MAX_KV_TILES: tl.constexpr,
    KV_TILE_GROUPS: tl.constexpr,
    BLOCK_KV_TILES: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    PREFIX_KEEP_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    kv_group = pid % KV_TILE_GROUPS
    tmp = pid // KV_TILE_GROUPS
    q_block_idx = tmp % MAX_Q_BLOCKS
    seq_idx = tmp // MAX_Q_BLOCKS

    if seq_idx < first_prefill_req or seq_idx >= last_prefill_req:
        return

    q_start = tl.load(query_start_len_ptr + seq_idx)
    q_end = tl.load(query_start_len_ptr + seq_idx + 1)
    query_len = q_end - q_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - query_len

    if query_len <= 1 or context_len < 0:
        return

    q_blocks = cdiv_fn(query_len, TILE_SIZE)
    kv_tiles = cdiv_fn(seq_len, TILE_SIZE)
    if q_block_idx >= q_blocks:
        return

    kv_offsets = kv_group * BLOCK_KV_TILES + tl.arange(0, BLOCK_KV_TILES)
    kv_valid = kv_offsets < kv_tiles

    q_sample_local = q_block_idx * TILE_SIZE + (TILE_SIZE // 2)
    q_sample_local = tl.minimum(q_sample_local, query_len - 1)
    q_token_idx = q_start + q_sample_local
    q_head_idx = kv_head_idx * num_queries_per_kv
    d = tl.arange(0, D_SAMPLE)
    q_vec = tl.load(
        query_ptr + q_token_idx * query_stride_0 + q_head_idx * query_stride_1 + d,
        mask=d < HEAD_SIZE,
        other=0.0,
    )

    k_sample_abs = kv_offsets * TILE_SIZE + (TILE_SIZE // 2)
    k_sample_abs = tl.minimum(k_sample_abs, seq_len - 1)
    kv_block = k_sample_abs // BLOCK_SIZE
    kv_block_offset = k_sample_abs - kv_block * BLOCK_SIZE
    physical_block = tl.load(
        block_tables_ptr + seq_idx * block_table_stride + kv_block,
        mask=kv_valid,
        other=0,
    )
    k_vec = tl.load(
        key_cache_ptr
        + physical_block[:, None] * stride_k_cache_0
        + kv_block_offset[:, None] * stride_k_cache_1
        + kv_head_idx * stride_k_cache_2
        + d[None, :] * stride_k_cache_3,
        mask=kv_valid[:, None] & (d[None, :] < HEAD_SIZE),
        other=0.0,
    )
    score = tl.sum(k_vec * q_vec[None, :], axis=1) * scale

    q_abs_end = context_len + (q_block_idx + 1) * TILE_SIZE - 1
    q_abs_end = tl.minimum(q_abs_end, seq_len - 1)
    k_tile_start = kv_offsets * TILE_SIZE
    causal = k_tile_start <= q_abs_end

    q_abs_tile = (context_len + q_block_idx * TILE_SIZE) // TILE_SIZE
    local = (kv_offsets <= q_abs_tile) & (kv_offsets + LOCAL_BLOCKS >= q_abs_tile)
    prefix = kv_offsets < PREFIX_KEEP_BLOCKS
    keep = causal & (prefix | local | (score > sample_threshold))

    tl.store(
        mask_ptr
        + seq_idx * mask_stride_b
        + kv_head_idx * mask_stride_h
        + q_block_idx * mask_stride_q
        + kv_offsets * mask_stride_k,
        keep.to(tl.int32),
        mask=kv_valid,
    )


def _build_bfla_block_mask_triton_sample(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cu_seqlens_q_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    *,
    first_prefill_req: int,
    num_prefill_reqs: int,
    block_size: int,
    block_n: int,
    sample_threshold: float,
    prefix_keep_blocks: int,
    local_blocks: int,
    softmax_scale: float,
) -> torch.Tensor | None:
    """Cheap sampled Triton mask builder for experimental BFLA prefill.

    This intentionally avoids PyTorch softmax/sort/topk. Each program samples
    one representative Q vector and one K vector per KV tile, keeps a local band
    plus prefix tiles, and writes the coarse block mask directly on GPU.
    """
    if not q.is_cuda or not k_cache.is_cuda or not block_table.is_cuda:
        return None
    device = q.device
    num_seqs = int(seq_lens_cpu.numel())
    num_query_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[2])
    if num_query_heads % num_kv_heads != 0:
        return None

    query_lens = cu_seqlens_q_cpu[1:] - cu_seqlens_q_cpu[:-1]
    max_query_len = int(query_lens.max().item())
    max_seq_len = int(seq_lens_cpu.max().item())
    max_q_blocks = triton.cdiv(max_query_len, block_n)
    max_kv_tiles = triton.cdiv(max_seq_len, block_n)
    if max_q_blocks <= 0 or max_kv_tiles <= 0:
        return None

    sparse_mask = torch.ones(
        (num_seqs, num_kv_heads, max_q_blocks, max_kv_tiles),
        device=device,
        dtype=torch.int32,
    )
    cu_q_gpu = cu_seqlens_q_cpu.to(device=device, non_blocking=True)
    seq_lens_gpu = seq_lens_cpu.to(device=device, non_blocking=True)

    block_kv_tiles = int(os.environ.get("VLLM_BFLA_SAMPLE_KV_GROUP", "64"))
    d_sample = min(int(os.environ.get("VLLM_BFLA_SAMPLE_D", "64")), int(q.shape[2]))
    d_sample = triton.next_power_of_2(max(1, d_sample))
    kv_tile_groups = triton.cdiv(max_kv_tiles, block_kv_tiles)

    _kernel_bfla_sample_mask[(num_seqs * max_q_blocks * kv_tile_groups, num_kv_heads)](
        sparse_mask,
        q,
        k_cache,
        block_table,
        cu_q_gpu,
        seq_lens_gpu,
        softmax_scale,
        sample_threshold,
        num_query_heads // num_kv_heads,
        block_table.stride(0),
        sparse_mask.stride(0),
        sparse_mask.stride(1),
        sparse_mask.stride(2),
        sparse_mask.stride(3),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        first_prefill_req,
        first_prefill_req + num_prefill_reqs,
        BLOCK_SIZE=block_size,
        TILE_SIZE=block_n,
        HEAD_SIZE=int(q.shape[2]),
        D_SAMPLE=d_sample,
        MAX_Q_BLOCKS=max_q_blocks,
        MAX_KV_TILES=max_kv_tiles,
        KV_TILE_GROUPS=kv_tile_groups,
        BLOCK_KV_TILES=block_kv_tiles,
        LOCAL_BLOCKS=max(0, int(local_blocks)),
        PREFIX_KEEP_BLOCKS=max(1, int(prefix_keep_blocks)),
    )
    return sparse_mask


@triton.jit
def _kernel_bfla_q_centroids(
    q_summary_ptr,
    query_ptr,
    query_start_len_ptr,
    seq_lens_ptr,
    num_queries_per_kv: tl.int32,
    q_summary_stride_b: tl.int64,
    q_summary_stride_h: tl.int64,
    q_summary_stride_q: tl.int64,
    q_summary_stride_c: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    first_prefill_req: tl.int32,
    last_prefill_req: tl.int32,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    D_SAMPLE: tl.constexpr,
    CENTROIDS: tl.constexpr,
    SEG_SIZE: tl.constexpr,
    MAX_Q_BLOCKS: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    if seq_idx < first_prefill_req or seq_idx >= last_prefill_req:
        return

    q_start = tl.load(query_start_len_ptr + seq_idx)
    q_end = tl.load(query_start_len_ptr + seq_idx + 1)
    query_len = q_end - q_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - query_len
    if query_len <= 1 or context_len < 0:
        return

    q_blocks = cdiv_fn(query_len, TILE_SIZE)
    if q_block_idx >= q_blocks:
        return

    d = tl.arange(0, D_SAMPLE)
    offs = tl.arange(0, SEG_SIZE)
    q_head_idx = kv_head_idx * num_queries_per_kv

    for c in tl.static_range(0, CENTROIDS):
        token_local = q_block_idx * TILE_SIZE + c * SEG_SIZE + offs
        valid = token_local < query_len
        vals = tl.load(
            query_ptr
            + (q_start + token_local)[:, None] * query_stride_0
            + q_head_idx * query_stride_1
            + d[None, :],
            mask=valid[:, None] & (d[None, :] < HEAD_SIZE),
            other=0.0,
        )
        denom = tl.maximum(tl.sum(valid.to(tl.float32), axis=0), 1.0)
        mean = tl.sum(vals, axis=0) / denom
        tl.store(
            q_summary_ptr
            + seq_idx * q_summary_stride_b
            + kv_head_idx * q_summary_stride_h
            + q_block_idx * q_summary_stride_q
            + c * q_summary_stride_c
            + d,
            mean,
            mask=d < HEAD_SIZE,
        )


@triton.jit
def _kernel_bfla_k_centroids(
    k_summary_ptr,
    key_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    block_table_stride: tl.int64,
    k_summary_stride_b: tl.int64,
    k_summary_stride_h: tl.int64,
    k_summary_stride_k: tl.int64,
    k_summary_stride_c: tl.int64,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    first_prefill_req: tl.int32,
    last_prefill_req: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    D_SAMPLE: tl.constexpr,
    CENTROIDS: tl.constexpr,
    SEG_SIZE: tl.constexpr,
):
    kv_tile_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    if seq_idx < first_prefill_req or seq_idx >= last_prefill_req:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    kv_tiles = cdiv_fn(seq_len, TILE_SIZE)
    if kv_tile_idx >= kv_tiles:
        return

    d = tl.arange(0, D_SAMPLE)
    offs = tl.arange(0, SEG_SIZE)

    for c in tl.static_range(0, CENTROIDS):
        token_abs = kv_tile_idx * TILE_SIZE + c * SEG_SIZE + offs
        valid = token_abs < seq_len
        kv_block = token_abs // BLOCK_SIZE
        kv_block_offset = token_abs - kv_block * BLOCK_SIZE
        physical_block = tl.load(
            block_tables_ptr + seq_idx * block_table_stride + kv_block,
            mask=valid,
            other=0,
        )
        vals = tl.load(
            key_cache_ptr
            + physical_block[:, None] * stride_k_cache_0
            + kv_block_offset[:, None] * stride_k_cache_1
            + kv_head_idx * stride_k_cache_2
            + d[None, :] * stride_k_cache_3,
            mask=valid[:, None] & (d[None, :] < HEAD_SIZE),
            other=0.0,
        )
        denom = tl.maximum(tl.sum(valid.to(tl.float32), axis=0), 1.0)
        mean = tl.sum(vals, axis=0) / denom
        tl.store(
            k_summary_ptr
            + seq_idx * k_summary_stride_b
            + kv_head_idx * k_summary_stride_h
            + kv_tile_idx * k_summary_stride_k
            + c * k_summary_stride_c
            + d,
            mean,
            mask=d < HEAD_SIZE,
        )


@triton.jit
def _kernel_bfla_centroid_mask(
    mask_ptr,
    q_summary_ptr,
    k_summary_ptr,
    query_start_len_ptr,
    seq_lens_ptr,
    scale,
    threshold,
    mask_stride_b: tl.int64,
    mask_stride_h: tl.int64,
    mask_stride_q: tl.int64,
    mask_stride_k: tl.int64,
    q_summary_stride_b: tl.int64,
    q_summary_stride_h: tl.int64,
    q_summary_stride_q: tl.int64,
    q_summary_stride_c: tl.int64,
    k_summary_stride_b: tl.int64,
    k_summary_stride_h: tl.int64,
    k_summary_stride_k: tl.int64,
    k_summary_stride_c: tl.int64,
    first_prefill_req: tl.int32,
    last_prefill_req: tl.int32,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    D_SAMPLE: tl.constexpr,
    CENTROIDS: tl.constexpr,
    MAX_Q_BLOCKS: tl.constexpr,
    KV_TILE_GROUPS: tl.constexpr,
    BLOCK_KV_TILES: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    PREFIX_KEEP_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    kv_group = pid % KV_TILE_GROUPS
    tmp = pid // KV_TILE_GROUPS
    q_block_idx = tmp % MAX_Q_BLOCKS
    seq_idx = tmp // MAX_Q_BLOCKS

    if seq_idx < first_prefill_req or seq_idx >= last_prefill_req:
        return

    q_start = tl.load(query_start_len_ptr + seq_idx)
    q_end = tl.load(query_start_len_ptr + seq_idx + 1)
    query_len = q_end - q_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - query_len
    if query_len <= 1 or context_len < 0:
        return

    q_blocks = cdiv_fn(query_len, TILE_SIZE)
    kv_tiles = cdiv_fn(seq_len, TILE_SIZE)
    if q_block_idx >= q_blocks:
        return

    kv_offsets = kv_group * BLOCK_KV_TILES + tl.arange(0, BLOCK_KV_TILES)
    kv_valid = kv_offsets < kv_tiles
    d = tl.arange(0, D_SAMPLE)
    best = tl.full((BLOCK_KV_TILES,), -3.402823e38, tl.float32)

    for cq in tl.static_range(0, CENTROIDS):
        qv = tl.load(
            q_summary_ptr
            + seq_idx * q_summary_stride_b
            + kv_head_idx * q_summary_stride_h
            + q_block_idx * q_summary_stride_q
            + cq * q_summary_stride_c
            + d,
            mask=d < HEAD_SIZE,
            other=0.0,
        )
        for ck in tl.static_range(0, CENTROIDS):
            kv = tl.load(
                k_summary_ptr
                + seq_idx * k_summary_stride_b
                + kv_head_idx * k_summary_stride_h
                + kv_offsets[:, None] * k_summary_stride_k
                + ck * k_summary_stride_c
                + d[None, :],
                mask=kv_valid[:, None] & (d[None, :] < HEAD_SIZE),
                other=0.0,
            )
            score = tl.sum(kv * qv[None, :], axis=1) * scale
            best = tl.maximum(best, score)

    q_abs_end = context_len + (q_block_idx + 1) * TILE_SIZE - 1
    q_abs_end = tl.minimum(q_abs_end, seq_len - 1)
    causal = kv_offsets * TILE_SIZE <= q_abs_end
    q_abs_tile = (context_len + q_block_idx * TILE_SIZE) // TILE_SIZE
    local = (kv_offsets <= q_abs_tile) & (kv_offsets + LOCAL_BLOCKS >= q_abs_tile)
    prefix = kv_offsets < PREFIX_KEEP_BLOCKS
    keep = causal & (prefix | local | (best > threshold))

    tl.store(
        mask_ptr
        + seq_idx * mask_stride_b
        + kv_head_idx * mask_stride_h
        + q_block_idx * mask_stride_q
        + kv_offsets * mask_stride_k,
        keep.to(tl.int32),
        mask=kv_valid,
    )


def _build_bfla_block_mask_triton_centroid(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cu_seqlens_q_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    *,
    first_prefill_req: int,
    num_prefill_reqs: int,
    block_size: int,
    block_n: int,
    sample_threshold: float,
    prefix_keep_blocks: int,
    local_blocks: int,
    softmax_scale: float,
) -> torch.Tensor | None:
    if not q.is_cuda or not k_cache.is_cuda or not block_table.is_cuda:
        return None
    device = q.device
    num_seqs = int(seq_lens_cpu.numel())
    num_query_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[2])
    if num_query_heads % num_kv_heads != 0:
        return None

    query_lens = cu_seqlens_q_cpu[1:] - cu_seqlens_q_cpu[:-1]
    max_query_len = int(query_lens.max().item())
    max_seq_len = int(seq_lens_cpu.max().item())
    max_q_blocks = triton.cdiv(max_query_len, block_n)
    max_kv_tiles = triton.cdiv(max_seq_len, block_n)
    if max_q_blocks <= 0 or max_kv_tiles <= 0:
        return None

    centroids = int(os.environ.get("VLLM_BFLA_CENTROIDS", "4"))
    if centroids not in (1, 2, 4, 8) or block_n % centroids != 0:
        return None
    seg_size = block_n // centroids
    d_sample = min(int(os.environ.get("VLLM_BFLA_SAMPLE_D", "64")), int(q.shape[2]))
    d_sample = triton.next_power_of_2(max(1, d_sample))
    block_kv_tiles = int(os.environ.get("VLLM_BFLA_SAMPLE_KV_GROUP", "64"))
    kv_tile_groups = triton.cdiv(max_kv_tiles, block_kv_tiles)

    sparse_mask = torch.ones(
        (num_seqs, num_kv_heads, max_q_blocks, max_kv_tiles),
        device=device,
        dtype=torch.int32,
    )
    q_summary = torch.empty(
        (num_seqs, num_kv_heads, max_q_blocks, centroids, d_sample),
        device=device,
        dtype=q.dtype,
    )
    k_summary = torch.empty(
        (num_seqs, num_kv_heads, max_kv_tiles, centroids, d_sample),
        device=device,
        dtype=k_cache.dtype,
    )
    cu_q_gpu = cu_seqlens_q_cpu.to(device=device, non_blocking=True)
    seq_lens_gpu = seq_lens_cpu.to(device=device, non_blocking=True)

    grid_q = (max_q_blocks, num_kv_heads, num_seqs)
    _kernel_bfla_q_centroids[grid_q](
        q_summary,
        q,
        cu_q_gpu,
        seq_lens_gpu,
        num_query_heads // num_kv_heads,
        q_summary.stride(0),
        q_summary.stride(1),
        q_summary.stride(2),
        q_summary.stride(3),
        q.stride(0),
        q.stride(1),
        first_prefill_req,
        first_prefill_req + num_prefill_reqs,
        TILE_SIZE=block_n,
        HEAD_SIZE=int(q.shape[2]),
        D_SAMPLE=d_sample,
        CENTROIDS=centroids,
        SEG_SIZE=seg_size,
        MAX_Q_BLOCKS=max_q_blocks,
    )

    grid_k = (max_kv_tiles, num_kv_heads, num_seqs)
    _kernel_bfla_k_centroids[grid_k](
        k_summary,
        k_cache,
        block_table,
        seq_lens_gpu,
        block_table.stride(0),
        k_summary.stride(0),
        k_summary.stride(1),
        k_summary.stride(2),
        k_summary.stride(3),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        first_prefill_req,
        first_prefill_req + num_prefill_reqs,
        BLOCK_SIZE=block_size,
        TILE_SIZE=block_n,
        HEAD_SIZE=int(q.shape[2]),
        D_SAMPLE=d_sample,
        CENTROIDS=centroids,
        SEG_SIZE=seg_size,
    )

    _kernel_bfla_centroid_mask[(num_seqs * max_q_blocks * kv_tile_groups, num_kv_heads)](
        sparse_mask,
        q_summary,
        k_summary,
        cu_q_gpu,
        seq_lens_gpu,
        softmax_scale,
        sample_threshold,
        sparse_mask.stride(0),
        sparse_mask.stride(1),
        sparse_mask.stride(2),
        sparse_mask.stride(3),
        q_summary.stride(0),
        q_summary.stride(1),
        q_summary.stride(2),
        q_summary.stride(3),
        k_summary.stride(0),
        k_summary.stride(1),
        k_summary.stride(2),
        k_summary.stride(3),
        first_prefill_req,
        first_prefill_req + num_prefill_reqs,
        TILE_SIZE=block_n,
        HEAD_SIZE=int(q.shape[2]),
        D_SAMPLE=d_sample,
        CENTROIDS=centroids,
        MAX_Q_BLOCKS=max_q_blocks,
        KV_TILE_GROUPS=kv_tile_groups,
        BLOCK_KV_TILES=block_kv_tiles,
        LOCAL_BLOCKS=max(0, int(local_blocks)),
        PREFIX_KEEP_BLOCKS=max(1, int(prefix_keep_blocks)),
    )
    return sparse_mask


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    bfla_block_mask_ptr,  # [num_seqs, num_kv_heads, max_q_blocks, max_kv_tiles]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    bfla_mask_stride_b: tl.int64,
    bfla_mask_stride_h: tl.int64,
    bfla_mask_stride_q: tl.int64,
    bfla_mask_stride_k: tl.int64,
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_BFLA_MASK: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    if USE_MM_PREFIX:
        # image bidirectional attention ranges require a full range
        # including q_block padding to make sure doc mask is correct
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        # adjust for potential padding in the last q_block by considering the
        # actual sequence length
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        bfla_keep = 1
        if USE_BFLA_MASK:
            # q_block_local_idx is in BLOCK_Q units, while the BFLA mask is
            # built in TILE_SIZE-sized query blocks. Convert before indexing;
            # using q_block_local_idx directly can read wrong/OOB mask rows.
            bfla_q_block_idx = (q_block_local_idx * BLOCK_Q) // TILE_SIZE
            bfla_keep = tl.load(
                bfla_block_mask_ptr
                + seq_idx * bfla_mask_stride_b
                + kv_head_idx * bfla_mask_stride_h
                + bfla_q_block_idx * bfla_mask_stride_q
                + j * bfla_mask_stride_k
            )

        if bfla_keep != 0:
            seq_offset = j * TILE_SIZE + offs_t
            tile_mask = seq_offset < max_seq_prefix_len

            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )

            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )

            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )

            if K_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    K = K_load
                else:
                    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
            else:
                K = K_load

            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )

            if V_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    V = V_load
                else:
                    V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
            else:
                V = V_load

            # Compute attention mask: causal by default (key <= query)
            query_abs_pos = context_len + query_pos[:, None]
            seq_mask = seq_offset[None, :] <= query_abs_pos

            # Apply sliding window to base mask BEFORE mm_prefix OR.
            # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
            if SLIDING_WINDOW > 0:
                seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

            # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
            # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
            if USE_MM_PREFIX:
                for i in range(MAX_MM_RANGES):
                    range_start = tl.load(
                        mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                    )
                    range_end = tl.load(
                        mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                    )

                    is_valid = range_start < range_end
                    q_in_range = (
                        (query_abs_pos >= range_start)
                        & (query_abs_pos <= range_end)
                        & is_valid
                    )
                    k_in_range = (
                        (seq_offset[None, :] >= range_start)
                        & (seq_offset[None, :] <= range_end)
                        & is_valid
                    )
                    seq_mask |= q_in_range & k_in_range

            # S : (BLOCK_M, TILE_SIZE)
            S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

            S += scale * tl.dot(Q, K)

            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)

            S = tl.where(
                query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
            )

            if USE_ALIBI_SLOPES:
                if USE_ALIBI_SQRT:
                    relative_pos = seq_offset - (context_len + query_pos[:, None])
                    alibi_offset = tl.where(
                        relative_pos <= 0,
                        -tl.sqrt((-relative_pos).to(tl.float32)),
                        0.0,
                    )
                else:
                    alibi_offset = seq_offset - context_len
                S += alibi_slope[:, None] * alibi_offset

            if USE_QQ_BIAS:
                # compute key positions relative to query section
                key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
                # load bias only for keys that correspond to queries
                is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
                qq_bias = tl.load(
                    qq_bias_row_ptrs + key_rel_pos[None, :],
                    mask=is_query_key[None, :],  # avoid OOB for context keys
                    other=0.0,
                )
                S += qq_bias

            # compute running maximum
            # m_j : (BLOCK_M,)
            m_j = tl.maximum(M, tl.max(S, axis=1))

            # For sliding window there's a chance the max is -inf due to masking of
            # the entire row. In this case we need to set m_j 0 to avoid NaN
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

            # P : (BLOCK_M, TILE_SIZE)
            P = tl.exp(S - m_j[:, None])

            # l_j : (BLOCK_M,)
            l_j = tl.sum(P, axis=1)

            # alpha : (BLOCK_M, )
            alpha = tl.exp(M - m_j)

            # acc : (BLOCK_M, HEAD_SIZE_PADDED)
            acc = acc * alpha[:, None]

            # update constants
            L = L * alpha + l_j
            M = m_j

            if SLIDING_WINDOW:
                qpos_lo = q_block_local_idx * BLOCK_Q
                V = tl.where(
                    (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
                )

            # acc : (BLOCK_M, HEAD_SIZE_PADDED)
            acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size_padded]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(
        max(segm_idx * tiles_per_segment, tile_start),
        min((segm_idx + 1) * tiles_per_segment, tile_end),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization.

    For Gemma3, use 32 for both prefill and decode to better utilize
    the larger head dimension (128/256). For other models, use
    the default vLLM behavior.
    """
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
    # Optional CPU metadata for BFLA sparse pure-prefill masking. If not
    # provided, behavior is identical to the original dense implementation.
    cu_seqlens_q_cpu: torch.Tensor | None = None,
    seq_lens_cpu: torch.Tensor | None = None,
    bfla_threshold: float = 0.005,
    bfla_min_keep_blocks: int = 256,
    bfla_keep_ratio: float = 0.75,
    bfla_local_blocks: int = 256,
    bfla_keep_mass: float = 0.999,
    bfla_min_prefill_tokens: int = 32768,
    bfla_common_prefix_len: int = 0,
    bfla_allow_sparse_prefill: bool = False,
):
    bfla_threshold = float(os.environ.get("VLLM_BFLA_THRESHOLD", bfla_threshold))
    bfla_min_keep_blocks = int(os.environ.get("VLLM_BFLA_MIN_KEEP_BLOCKS", bfla_min_keep_blocks))
    bfla_keep_ratio = float(os.environ.get("VLLM_BFLA_KEEP_RATIO", bfla_keep_ratio))
    bfla_local_blocks = int(os.environ.get("VLLM_BFLA_LOCAL_BLOCKS", bfla_local_blocks))
    bfla_keep_mass = float(os.environ.get("VLLM_BFLA_KEEP_MASS", bfla_keep_mass))
    bfla_min_prefill_tokens = int(os.environ.get("VLLM_BFLA_MIN_PREFILL_TOKENS", bfla_min_prefill_tokens))
    bfla_mask_impl = os.environ.get("VLLM_BFLA_MASK_IMPL", "torch").lower()
    bfla_sample_threshold = float(os.environ.get("VLLM_BFLA_SAMPLE_THRESHOLD", bfla_threshold))
    bfla_torch_mask_block_n = int(os.environ.get("VLLM_BFLA_TORCH_MASK_BLOCK_N", "64"))
    bfla_torch_pool_mode = os.environ.get("VLLM_BFLA_TORCH_POOL", "mean").lower()
    bfla_spec_stride = int(os.environ.get("VLLM_BFLA_SPEC_STRIDE", "0"))
    bfla_spec_prob = float(os.environ.get("VLLM_BFLA_SPEC_PROB", "0"))
    bfla_spec_seed = int(os.environ.get("VLLM_BFLA_SPEC_SEED", "1"))

    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Tile sizes for prefill and decode. Gemma3 models use optimized values.
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    TILE_SIZE_PREFILL = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=True,
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=False,
    )

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    if (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    ):
        bfla_block_mask = None
        if (
            bfla_allow_sparse_prefill
            and bfla_keep_mass < 1.0
            and bfla_common_prefix_len == 0
            and cu_seqlens_q_cpu is not None
            and seq_lens_cpu is not None
            and not use_alibi_slopes
            and not use_qq_bias
            and not use_mm_prefix
            and sinks is None
            and softcap <= 0
            and sliding_window_val == 0
            and not str(k.dtype).startswith("torch.float8")
        ):
            suffix = _find_prefill_like_suffix(
                cu_seqlens_q_cpu,
                seq_lens_cpu,
                int(q.shape[0]),
                min_prefill_tokens=bfla_min_prefill_tokens,
            )
            if suffix is not None:
                first_prefill_req, _prefix_tokens, num_prefill_reqs, _ = suffix
                reuse_bfla_mask = os.environ.get("VLLM_BFLA_REUSE_MASK", "0") == "1"
                bfla_cache_key = None
                if reuse_bfla_mask:
                    bfla_cache_key = (
                        int(q.shape[0]),
                        int(q.shape[1]),
                        int(q.shape[2]),
                        int(k.shape[2]),
                        int(num_seqs),
                        int(block_size),
                        int(TILE_SIZE_PREFILL),
                        int(first_prefill_req),
                        int(num_prefill_reqs),
                        float(bfla_threshold),
                        int(bfla_min_keep_blocks),
                        float(bfla_keep_ratio),
                        int(bfla_local_blocks),
                        float(bfla_keep_mass),
                        str(bfla_mask_impl),
                        int(bfla_torch_mask_block_n),
                        str(bfla_torch_pool_mode),
                        int(bfla_sample_d),
                        int(bfla_sample_kv_group),
                        int(bfla_centroids),
                        int(bfla_spec_stride),
                        float(bfla_spec_prob),
                        int(bfla_spec_seed),
                        tuple(int(x) for x in cu_seqlens_q_cpu.tolist()),
                        tuple(int(x) for x in seq_lens_cpu.tolist()),
                    )
                    cached_mask = _BFLA_MASK_CACHE.get(bfla_cache_key)
                    if cached_mask is not None and cached_mask.device == q.device:
                        bfla_block_mask = cached_mask

                if bfla_block_mask is None:
                    if bfla_mask_impl == "triton_sample":
                        bfla_block_mask = _build_bfla_block_mask_triton_sample(
                            q,
                            k,
                            cu_seqlens_q_cpu,
                            seq_lens_cpu,
                            block_table,
                            first_prefill_req=first_prefill_req,
                            num_prefill_reqs=num_prefill_reqs,
                            block_size=block_size,
                            block_n=TILE_SIZE_PREFILL,
                            sample_threshold=bfla_sample_threshold,
                            prefix_keep_blocks=bfla_min_keep_blocks,
                            local_blocks=bfla_local_blocks,
                            softmax_scale=softmax_scale,
                        )
                    elif bfla_mask_impl == "triton_centroid":
                        bfla_block_mask = _build_bfla_block_mask_triton_centroid(
                            q,
                            k,
                            cu_seqlens_q_cpu,
                            seq_lens_cpu,
                            block_table,
                            first_prefill_req=first_prefill_req,
                            num_prefill_reqs=num_prefill_reqs,
                            block_size=block_size,
                            block_n=TILE_SIZE_PREFILL,
                            sample_threshold=bfla_sample_threshold,
                            prefix_keep_blocks=bfla_min_keep_blocks,
                            local_blocks=bfla_local_blocks,
                            softmax_scale=softmax_scale,
                        )
                    if bfla_block_mask is None:
                        bfla_block_mask = _build_bfla_block_mask(
                            q,
                            k,
                            cu_seqlens_q_cpu,
                            seq_lens_cpu,
                            block_table,
                            first_prefill_req=first_prefill_req,
                            num_prefill_reqs=num_prefill_reqs,
                            block_size=block_size,
                            block_n=bfla_torch_mask_block_n,
                            attn_block_n=TILE_SIZE_PREFILL,
                            threshold=bfla_threshold,
                            min_keep_blocks=bfla_min_keep_blocks,
                            keep_ratio=bfla_keep_ratio,
                            local_blocks=bfla_local_blocks,
                            keep_mass=bfla_keep_mass,
                            softmax_scale=softmax_scale,
                            pool_mode=bfla_torch_pool_mode,
                            spec_stride=bfla_spec_stride,
                            spec_prob=bfla_spec_prob,
                            spec_seed=bfla_spec_seed,
                        )
                    if reuse_bfla_mask and bfla_cache_key is not None and bfla_block_mask is not None:
                        _BFLA_MASK_CACHE.clear()
                        _BFLA_MASK_CACHE[bfla_cache_key] = bfla_block_mask

        use_bfla_mask = bfla_block_mask is not None
        if use_bfla_mask:
            bfla_mask_ptr = bfla_block_mask
            bfla_mask_stride_b = bfla_block_mask.stride(0)
            bfla_mask_stride_h = bfla_block_mask.stride(1)
            bfla_mask_stride_q = bfla_block_mask.stride(2)
            bfla_mask_stride_k = bfla_block_mask.stride(3)
        else:
            # Placeholder pointer; never dereferenced when USE_BFLA_MASK=False.
            bfla_mask_ptr = block_table
            bfla_mask_stride_b = 0
            bfla_mask_stride_h = 0
            bfla_mask_stride_q = 0
            bfla_mask_stride_k = 0

        kernel_unified_attention_2d[
            (
                total_num_q_blocks,
                num_kv_heads,
            )
        ](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            bfla_block_mask_ptr=bfla_mask_ptr,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            bfla_mask_stride_b=bfla_mask_stride_b,
            bfla_mask_stride_h=bfla_mask_stride_h,
            bfla_mask_stride_q=bfla_mask_stride_q,
            bfla_mask_stride_k=bfla_mask_stride_k,
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_BFLA_MASK=use_bfla_mask,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
        )
    else:
        kernel_unified_attention_3d[
            (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        ](
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
        )
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )
