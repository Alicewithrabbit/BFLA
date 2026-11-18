"""BFLA prefill maintains native QSA K state for decoding.

The pre-indexer launch below reuses vLLM's Apache-2.0
``qwen4_exp/nvidia/ops/qsa_pre_indexer.py`` kernel unchanged. Its K programs
precede its independent Q programs. Only the launch extent is restricted.
Pure decode stays on the original model forward, including its CUDA graph.
"""
from typing import NamedTuple
import torch
from .protocol import prefill_context, uses_sparse_prefill


class IndexInputs(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor


def supported(indexer):
    from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
    from vllm.model_executor.layers.utils import default_unquantized_gemm
    from vllm import envs
    layer = indexer.index_qk_proj
    # A wrapped projection (e.g. LoRA) can expose the base weight and quant
    # method while changing forward(). Only slice a bare native linear.
    return (type(layer) is ReplicatedLinear and indexer.use_fused_pre_indexer
            and indexer.index_kv_heads == 1 and type(layer.quant_method) is UnquantizedLinearMethod
            and layer.quant_method._gemm_impl is default_unquantized_gemm
            and layer.bias is None and not envs.VLLM_BATCH_INVARIANT
            and layer.weight.dtype in (torch.bfloat16, torch.float16))


def project(indexer, hidden, decode_rows):
    if not supported(indexer):
        raise ValueError('K-only projection requires the native unquantized GEMM path')
    q_width = indexer.index_n_heads * indexer.index_head_dim
    k = torch.nn.functional.linear(hidden, indexer.index_qk_proj.weight[q_width:])
    # Only decode rows need index queries. All rows update the index keys.
    q = (indexer.index_qk_proj(hidden[:decode_rows])[0][:, :q_width]
         if decode_rows else hidden.new_empty((0, q_width)))
    return IndexInputs(q, k)


def prepare(indexer, inputs, positions, raw, compressed):
    from vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer import _qsa_pre_indexer_kernel
    from vllm.triton_utils import triton
    q, k = inputs
    n = k.shape[0]; d = q.shape[0]
    hq = indexer.index_n_heads; dim = indexer.index_head_dim
    # Keep upstream TILE_T_Q=2: a different tile changes MRoPE BF16 rounding.
    # The sole padding lane is never selected/consumed and has no Q projection.
    if d % 2:
        q = torch.nn.functional.pad(q, (0, 0, 0, 1))
    q_out = q.new_empty((q.shape[0], hq, dim), dtype=indexer.indexer_dtype)
    state = indexer.raw_key_cache.kv_cache
    cache = indexer.compressed_key_cache.kv_cache
    # K arithmetic and all cache ownership stay inside the upstream kernel.
    tile_t = 2
    tile_h = 2 if n <= 4096 else 4
    k_work = compressed.k_work_metadata.shape[0]
    q_work = triton.cdiv(d, tile_t) * triton.cdiv(hq, tile_h)
    if n == 0:
        return q_out
    is_2d = positions.ndim == 2
    pos_axis, pos_token = positions.stride() if is_2d else (0, positions.stride(0))
    section = getattr(indexer.rotary_emb, 'mrope_section', None)
    sections = section or (0, 0, 0)
    _qsa_pre_indexer_kernel[(k_work + q_work,)](
        q, q.stride(0), k, k.stride(0), positions, pos_axis, pos_token,
        indexer.rotary_emb.cos_sin_cache,
        indexer.q_layernorm.weight, indexer.k_layernorm.weight,
        indexer.q_layernorm.variance_epsilon,
        q_out, q_out.stride(0), q_out.stride(1),
        state, state.stride(0), state.stride(1), raw.slot_mapping,
        raw.block_table, raw.block_table.stride(0), raw.query_start_loc,
        raw.logical_positions, compressed.slot_mapping, compressed.k_work_metadata,
        cache, cache.stride(0), cache.stride(1), n, state.shape[0], cache.shape[0], k_work,
        HQ=hq, D=dim, TILE_T_Q=tile_t, TILE_H_Q=tile_h,
        COMPRESS_RATIO=indexer.compress_ratio, STATE_SIZE=state.shape[1],
        COMP_PAGE_SIZE=cache.shape[1], IS_2D_POSITIONS=is_2d,
        IS_K_MROPE=bool(section),
        CACHE_HAS_ROPE_POS=indexer.raw_key_cache.rope_position_cache is not None,
        MROPE_H=sections[1], MROPE_W=sections[2], num_warps=1)
    return q_out[:d]


def update_and_select(indexer, inputs, positions, out):
    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        qsa_select_paged_decode, expand_qsa_block_indices)
    raw, compressed = indexer._metadata()
    n = raw.num_actual_tokens
    q = prepare(indexer, inputs, positions[..., :n], raw, compressed)
    d = compressed.num_decode_tokens
    if not d or indexer.skip_topk:
        return out
    blocks = torch.empty((d, indexer.token_topk // indexer.compress_ratio),
                         device=q.device, dtype=torch.int32)
    qsa_select_paged_decode(q, indexer.compressed_key_cache.kv_cache,
                           compressed.block_table[:compressed.num_decodes],
                           compressed.visible_blocks[:d], indexer.token_topk,
                           indexer.compress_ratio, compressed.decode_query_len, blocks)
    expand_qsa_block_indices(blocks, compressed.logical_positions[:d],
                            compressed.visible_blocks[:d], indexer.compress_ratio,
                            indexer.token_topk, out[:d])
    return out


def install(attention_class):
    """Use K-only index projection for the prefill rows in supported batches."""
    original = attention_class.forward

    def forward(self, positions, hidden_states):
        from vllm.forward_context import get_forward_context
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        main = metadata.get(self.layer_name) if isinstance(metadata, dict) else None
        if main is None or main.max_query_len <= 1:
            return original(self, positions, hidden_states)
        side = metadata[self.indexer.raw_key_cache.prefix]
        # A piecewise bucket can replay with different P/D ownership. Capture
        # the full Q/K projection and pass no capture-time prefill context.
        if (not uses_sparse_prefill(main, side)
                or (hidden_states.is_cuda and torch.cuda.is_current_stream_capturing())
                or not supported(self.indexer)):
            return original(self, positions, hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        n = hidden_states.shape[0]
        query = q.view(n, self.num_heads, self.head_dim)
        key = k.view(n, self.num_kv_heads, self.head_dim)
        value = v.view(n, self.num_kv_heads, self.head_dim)
        output = torch.empty_like(query)
        prefill = prefill_context(self, query, main, side)
        if prefill is None:
            inputs, _ = self.indexer.index_qk_proj(hidden_states)
        else:
            inputs = project(self.indexer, hidden_states[:main.num_actual_tokens], side.num_decode_tokens)
        self._run_qsa(inputs, positions, query, key, value, output, gate, prefill=prefill)
        return self.o_proj(output.view(n, -1))[0]

    attention_class.forward = forward
