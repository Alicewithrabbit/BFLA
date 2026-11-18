"""CPU metadata and native cache layout contracts for vLLM prefill."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillPlan:
    starts: tuple = ()
    lengths: tuple = ()
    reason: str = 'native'
    kv_lengths: tuple = ()
    prompt_lengths: tuple = ()
    request_keys: tuple = ()
    rows: tuple = ()
    native_rows: tuple = ()


def has_prefix_mask(metadata):
    """Mask fields published by the native Flash or Triton metadata builder."""
    return any(getattr(metadata, field, None) is not None for field in
               ('mm_prefix_range', 'mm_prefix_range_tensor', 'mm_prefix_query_range_tensor',
                'rswa_prefix_lens', 'rswa_window', 'rswa_window_tensor'))


def attention_contract_reason(impl, metadata, *, output_scaling=False, allow_cascade=False):
    """Return why native semantics are required, or None for plain causal D/P."""
    if getattr(impl, 'attn_type', None) != 'decoder':
        return 'unsupported_attention_type'
    if getattr(impl, 'sliding_window', None) != (-1, -1):
        return 'sliding_window'
    if (getattr(impl, 'logits_soft_cap', 0) or getattr(impl, 'alibi_slopes', None) is not None
            or getattr(impl, 'sinks', None) is not None or output_scaling
            or (getattr(metadata, 'use_cascade', False) and not allow_cascade) or has_prefix_mask(metadata)
            or getattr(impl, 'kv_cache_dtype', None) not in ('auto', 'bfloat16')):
        return 'unsupported_attention_contract'
    return None


def classify_prefill(starts, sequence_upper_bounds, causal, *, chunking_enabled=False,
                     prompt_lengths=None, request_keys=None, is_prefilling=None):
    """Use explicit P ownership for extend; otherwise prove only a fresh row.

    Query/bound equality alone cannot locate the full-prompt tail under chunking.
    Owned P rows supply that boundary and exact KV lengths. Mixed batches retain
    native D while P rows use their own packed sparse metadata.
    """
    if causal is not True:
        return PrefillPlan(reason='noncausal_or_unknown')
    owned = prompt_lengths is not None and request_keys is not None and is_prefilling is not None
    if chunking_enabled and not owned:
        return PrefillPlan(reason='chunking_enabled')
    if sequence_upper_bounds is None:
        return PrefillPlan(reason='missing_cpu_bound')
    starts, bounds = tuple(starts), tuple(sequence_upper_bounds)
    if not starts or starts[0] != 0 or len(starts) != len(bounds) + 1:
        raise ValueError('inconsistent request boundaries')
    lengths = tuple(b-a for a, b in zip(starts, starts[1:]))
    if any(q < 0 or n < q for q, n in zip(lengths, bounds)):
        raise ValueError('inconsistent query lengths / sequence upper bounds')
    if owned:
        prompts, keys, flags = tuple(prompt_lengths), tuple(request_keys), tuple(is_prefilling)
        if len(prompts) != len(lengths) or len(keys) != len(lengths) or len(flags) != len(lengths):
            return PrefillPlan(reason='request_metadata_mismatch')
        from .partition import prefill_rows
        rows = prefill_rows(starts,bounds,flags)
        if not rows:
            return PrefillPlan(reason='decode_mixed_or_padding')
        native = tuple((i,starts[i],starts[i+1],bounds[i]) for i,q in enumerate(lengths)
                       if q > 0 and not flags[i])
        if any(b-a != 1 for _,a,b,_ in native):
            return PrefillPlan(reason='mixed_spec_decode')
        if any(prompts[i] < 1 for i,*_ in rows):
            return PrefillPlan(reason='request_metadata_mismatch')
        if any(n > prompts[i] for i,_,_,n in rows):
            # A resumed request may recompute generated tokens as well as its
            # prompt. Preserve native semantics rather than moving the P tail.
            return PrefillPlan(reason='recompute_beyond_prompt')
        if any(key is None for key in keys) or len(set(keys)) != len(keys):
            return PrefillPlan(reason='request_metadata_mismatch')
        if native:
            return PrefillPlan(starts,tuple(b-a for _,a,b,_ in rows),'mixed_prefill',
                tuple(n for *_,n in rows),tuple(prompts[i] for i,*_ in rows),
                tuple(keys[i] for i,*_ in rows),rows,native)
        if len(rows) != len(lengths):
            return PrefillPlan(reason='decode_mixed_or_padding')
        reason = 'fresh_prefill' if lengths == bounds == prompts else 'extend_prefill'
        return PrefillPlan(starts, lengths, reason, bounds, prompts, keys,rows)
    if not lengths or any(q <= 1 for q in lengths):
        return PrefillPlan(reason='decode_mixed_or_padding')
    if lengths != bounds:
        return PrefillPlan(reason='prefix_or_chunk')
    return PrefillPlan(starts, lengths, 'fresh_prefill')


def _field(source, name, default=None):
    return source.get(name, default) if isinstance(source, dict) else getattr(source, name, default)


def resolve_head_layout(impl, layer, model_config, *, tp_size=1, tp_rank=0):
    """Use actual layer totals, then configuration, validating local TP geometry.

    A single local KV head is ambiguous under replication, so absent global
    metadata is rejected instead of guessing the global query/KV ratio.
    No model identifier participates in this decision.
    """
    from bfla import HeadLayout
    text = _field(model_config, 'hf_text_config', model_config)
    arch = _field(model_config, 'model_arch_config')
    hq = _field(layer, 'total_num_heads', _field(impl, 'total_num_heads'))
    hkv = _field(layer, 'total_num_kv_heads', _field(impl, 'total_num_kv_heads'))
    if hq is None:
        hq = _field(text, 'num_attention_heads', _field(arch, 'total_num_attention_heads'))
    if hkv is None:
        hkv = _field(text, 'num_key_value_heads', _field(arch, 'total_num_kv_heads'))
        full_attention = tuple(getattr(impl, 'sliding_window', (-1, -1))) == (-1, -1)
        if full_attention and _field(text, 'attention_k_eq_v', False):
            hkv = _field(text, 'num_global_key_value_heads', hkv)
    if hq is None:
        hq = impl.num_heads * tp_size
    if hkv is None:
        if tp_size > 1 and impl.num_kv_heads == 1:
            raise ValueError('global KV head metadata required to resolve TP replication')
        hkv = impl.num_kv_heads * tp_size
    layout = HeadLayout.create(int(hq), int(hkv), int(impl.head_size), tp_size, tp_rank)
    if (layout.local_hq, layout.local_hkv, layout.head_dim) != (
            impl.num_heads, impl.num_kv_heads, impl.head_size):
        raise ValueError('global and TP-local attention geometry disagree')
    return layout


def materialize_alias_kv(cache, block_table, length, head_dim):
    """Gather the owner's packed K/V in request-local logical order."""
    import torch
    if cache.ndim != 4 or cache.shape[-1] != 2 * head_dim:
        raise ValueError('expected native [page, head, token, 2D] cache')
    page_size = cache.shape[2]
    if length < 0 or block_table.numel() * page_size < length:
        raise ValueError('alias context exceeds block table capacity')
    logical = torch.arange(length, device=cache.device)
    pages, offsets = block_table[logical // page_size].long(), logical % page_size
    packed = cache[pages, :, offsets, :]
    return packed[..., :head_dim].contiguous(), packed[..., head_dim:].contiguous()
