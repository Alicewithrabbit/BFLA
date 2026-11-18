"""PP-wide row ordering and graph shapes for vLLM 0.31."""
from functools import wraps


def cascade_policies(runner):
    """Describe the local decision using only immutable CPU-side parameters."""
    if not getattr(runner, 'cascade_attn_enabled', False) or runner.parallel_config.use_ubatching:
        return frozenset()
    from vllm.v1.attention.backend import AttentionMetadataBuilder
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import (
        ChunkedLocalAttentionSpec, EncoderOnlyAttentionSpec, FullAttentionSpec, SlidingWindowSpec,
    )

    policies = set()
    for gid, groups in enumerate(runner.attn_groups):
        for group in groups:
            spec = group.kv_cache_spec
            builder = group.get_metadata_builder()
            method = type(builder).use_cascade_attention
            if isinstance(spec, EncoderOnlyAttentionSpec) or method is AttentionMetadataBuilder.use_cascade_attention:
                continue
            if method is not FlashAttentionMetadataBuilder.use_cascade_attention:
                # A third-party heuristic may depend on opaque stage-local state.
                # Any physical shared prefix in its group can require mixed dispatch.
                policies.add((gid,))
                continue
            windowed = (isinstance(spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec))
                        or isinstance(spec, FullAttentionSpec) and
                        (spec.sliding_window is not None or spec.attention_chunk_size is not None))
            if not (builder.fa4_hd256 or runner.use_alibi or windowed or runner.dcp_world_size > 1):
                policies.add((gid, spec.block_size, runner.num_query_heads, spec.num_kv_heads, runner.num_sms))
    return frozenset(policies)


def align_v1(runner, thresholds, policies):
    threshold = min((t for t in thresholds if t is not None), default=None)
    runner.reorder_batch_threshold = threshold
    if threshold is not None and not runner.kv_cache_config.kv_cache_groups:
        # Native skips a completely KV-free stage before checking its threshold.
        from vllm.v1.attention.backends.utils import reorder_batch_to_split_decodes_and_prefills

        def reorder(scheduler_output):
            reorder_batch_to_split_decodes_and_prefills(
                runner.input_batch, scheduler_output, decode_threshold=threshold)

        runner._may_reorder_batch = reorder

    shared = frozenset().union(*policies)
    if len(set(policies)) == 1 and all(len(policy) > 1 for policy in shared):
        return
    from vllm.v1.attention.backends.flash_attn import use_cascade_attention

    original = runner._compute_cascade_attn_prefix_lens

    @wraps(original)
    def cascade(num_scheduled_tokens, num_computed_tokens, num_common_prefix_blocks):
        local = original(num_scheduled_tokens, num_computed_tokens, num_common_prefix_blocks)
        if local is not None:
            return local
        for gid, *flash in shared:
            blocks = num_common_prefix_blocks[gid]
            if not blocks:
                continue
            if flash:
                block, q_heads, kv_heads, sms = flash
                prefix = min(blocks, int(num_computed_tokens.min()) // block) * block
                if not use_cascade_attention(prefix, num_scheduled_tokens, q_heads, kv_heads,
                                             False, False, False, sms, 1):
                    continue
            # vLLM uses non-None for graph dispatch and the individual lengths
            # for actual attention. A zero on this stage keeps its native kernel.
            return [[0] * len(groups) for groups in runner.attn_groups]
        return None

    runner._compute_cascade_attn_prefix_lens = cascade


def common_decode_limit(stage_graphs, runner, width):
    """For unequal graph policies, find the shared range of ordinary FULL decode buckets."""
    from vllm.config import CUDAGraphMode

    if len(set(stage_graphs)) == 1:
        return None
    modes, buckets = zip(*stage_graphs)
    # A separated FULL decode graph only covers buckets up to max_num_seqs.
    # DP can change the agreed shape (and whether the batch is uniform).
    if (width != 1 or runner.parallel_config.data_parallel_size != 1
            or runner.compilation_config.pass_config.enable_sp or len(set(buckets)) != 1
            or any(mode.decode_mode() != CUDAGraphMode.FULL for mode in modes)):
        return 0
    max_reqs = runner.vllm_config.scheduler_config.max_num_seqs
    return max((size for size in buckets[0] if size <= max_reqs), default=0)


def align_v2_graphs(manager, force_mixed_eager, decode_limit):
    """Give incompatible graph sets the native unpadded descriptor before DP voting."""
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    original = manager.dispatch

    @wraps(original)
    def dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras,
                 max_query_len=None, num_ubatches=1):
        # The V2 runner drops the query bound on every prefill; BFLA clears
        # uniform_token_count for sparse single-token extensions on every stage.
        mixed_prefill = uniform_token_count is None and max_query_len is None
        incompatible_decode = (decode_limit is not None
                               and (uniform_token_count != 1 or num_tokens > decode_limit))
        if (mixed_prefill and force_mixed_eager) or (not mixed_prefill and incompatible_decode):
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs,
                num_active_loras=manager._resolve_effective_loras(num_active_loras), num_ubatches=num_ubatches)
        return original(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len, num_ubatches)

    manager.dispatch = dispatch
