"""Map Qwen3.8-Flash-Next QSA metadata to prefill request rows."""
from typing import NamedTuple

from ..collectives import has_feature_reducer
from ..partition import prefill_rows


def owned_prefills(starts, lengths, context):
    rows = prefill_rows(starts, lengths)
    if not rows:
        return (), (), ()
    owners = len(context.keys)
    if owners != len(context.prompt_lengths) or any(i >= owners for i, *_ in rows):
        return (), (), ()
    keys = tuple(context.keys[i] for i, *_ in rows)
    prompts = tuple(context.prompt_lengths[i] for i, *_ in rows)
    if (None in keys or len(set(keys)) != len(keys)
            or any(end > prompt for (*_, end), prompt in zip(rows, prompts))):
        return (), (), ()
    return rows, prompts, keys


def uses_sparse_prefill(main, side):
    """Both the projection and consumer must agree on native decode ownership."""
    return getattr(main, 'bfla_partition', None) == (side.num_decode_tokens, side.num_prefill_tokens)


class PrefillContext(NamedTuple):
    main: object
    side: object
    heads: object
    group: object
    settings: object


def prefill_context(layer, query, main, side):
    """Decide ownership before K-only projection, index selection or KV writes."""
    # MTP compacts the draft's native Top-K rows for later proposal steps.
    if getattr(layer, 'is_draft_layer', False) or not uses_sparse_prefill(main, side):
        return None
    import torch
    cache = layer.kv_cache
    if (not query.is_cuda or query.dtype != torch.bfloat16
            or cache.dtype != torch.bfloat16 or cache.device != query.device
            or cache.ndim != 4 or min(cache.shape) == 0
            or cache.shape[1] != layer.num_kv_heads or cache.shape[-1] != 2 * layer.head_dim
            or query.ndim != 3 or query.shape[1:] != (layer.num_heads, layer.head_dim)
            or main.num_actual_tokens != side.num_actual_tokens
            or main.num_actual_tokens > query.shape[0]
            or torch.cuda.is_current_stream_capturing()):
        return None
    from vllm.distributed import get_tp_group
    from bfla import HeadLayout, supports
    from ..settings import Settings
    group = get_tp_group()
    if not has_feature_reducer(group):
        return None
    try:
        heads = HeadLayout.create(layer.total_num_heads, layer.total_num_kv_heads,
                                  layer.head_dim, group.world_size, group.rank_in_group)
        heads.validate_local(layer.num_heads, layer.num_kv_heads, layer.head_dim)
    except ValueError:
        return None
    rows = main.bfla_rows
    settings = Settings.from_env()
    if not settings.enabled or not supports(
            heads, settings.config, device_capability=torch.cuda.get_device_capability(query.device),
            dtype=query.dtype, request_lengths=tuple(b - a for _, a, b, _ in rows),
            kv_lengths=tuple(n for *_, n in rows), prompt_lengths=main.bfla_prompts,
            kv_layout='paged').supported:
        return None
    return PrefillContext(main, side, heads, group, settings)
