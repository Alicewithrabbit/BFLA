"""Shared request-row geometry for native D plus BFLA P execution.

Rows are (request index, query begin, query end, visible KV upper bound).
Only P rows may use the host bound as an exact length. Native D retains its
device sequence lengths, including when its host bound is optimistic.
"""
from copy import copy


def prefill_rows(starts, lengths, is_prefilling=None):
    if len(starts) != len(lengths)+1 or not starts or starts[0] != 0:
        raise ValueError('one complete query interval per request required')
    if is_prefilling is not None and len(is_prefilling) != len(lengths):
        raise ValueError('one phase flag per request required')
    rows = []
    for i,(a,b,n) in enumerate(zip(starts,starts[1:],lengths)):
        if b < a or n < b-a:
            raise ValueError('visible KV must cover the query suffix')
        # Qwen3.8-Flash-Next QSA partitions by query length. Ordinary
        # backends supply phase flags so a one-token cached extend remains P.
        active = b-a > 1 if is_prefilling is None else is_prefilling[i] and b>a
        if active:
            rows.append((i,a,b,n))
    return tuple(rows)


def take_rows(tensor, rows):
    if tensor is None:
        return None
    if not rows:
        return tensor[:0]
    if all(left[2] == right[1] for left,right in zip(rows,rows[1:])):
        return tensor[rows[0][1]:rows[-1][2]]
    import torch
    return torch.cat([tensor[a:b] for _,a,b,_ in rows])


def write_rows(output, rows, packed):
    pos = 0
    for _,a,b,_ in rows:
        output[a:b].copy_(packed[pos:pos+b-a])
        pos += b-a


def take_requests(tensor, rows):
    """Select per-request entries, retaining a view for consecutive requests."""
    if tensor is None:
        return None
    if not rows:
        return tensor[:0]
    if all(left[0] + 1 == right[0] for left, right in zip(rows, rows[1:])):
        return tensor[rows[0][0]:rows[-1][0] + 1]
    from .metadata import integer_tensor
    indices = integer_tensor((row[0] for row in rows), device=tensor.device)
    return tensor.index_select(0, indices)


def native_metadata(metadata, rows):
    """Prepare a small native-only view once, outside layer forward/capture.

    A full-batch FA3 schedule or cascade cannot describe the selected D rows.
    Let the native kernel choose its schedule; Triton's group-sized workspace
    remains valid because the subset has no more rows than the original batch.
    """
    from .metadata import integer_tensor
    result = copy(metadata)
    starts = [0]
    for _,a,b,_ in rows:
        starts.append(starts[-1]+b-a)
    result.num_actual_tokens = starts[-1]
    result.max_query_len = max(b-a for _,a,b,_ in rows)
    result.max_seq_len = max(n for *_,n in rows)
    result.query_start_loc = integer_tensor(starts,device=metadata.query_start_loc.device,
                                            dtype=metadata.query_start_loc.dtype)
    result.seq_lens = take_requests(metadata.seq_lens, rows)
    result.block_table = take_requests(metadata.block_table, rows)
    result.slot_mapping = take_rows(metadata.slot_mapping,rows)
    result.use_cascade = False
    result.common_prefix_len = 0
    for name in ('scheduler_metadata','prefix_scheduler_metadata','cu_prefix_query_lens',
                 'prefix_kv_lens','suffix_kv_lens'):
        setattr(result,name,None)
    for name in ('num_decode_reqs','num_decode_tokens','num_prefill_reqs','num_prefill_tokens'):
        if hasattr(result,name):
            setattr(result,name, {'num_decode_reqs':len(rows),
                'num_decode_tokens':starts[-1]}.get(name,0))
    return result
