"""Native cache binding and bounded, exact-shape prefill reservations."""
from collections import OrderedDict, Counter
import torch
from bfla import IndexPoolState
from .protocol import materialize_alias_kv
from .runtime import PreparedExecution
from .metadata import integer_tensor

_MAX_RESERVATIONS = 2
RESERVATIONS = OrderedDict()
COUNTS = Counter()
POOLS = {}


def _reuse_reservation(identity):
    execution = RESERVATIONS.get(identity)
    if execution is not None:
        RESERVATIONS.move_to_end(identity)
    elif len(RESERVATIONS) >= _MAX_RESERVATIONS:
        _, retired = RESERVATIONS.popitem(last=False)
        retired.close()
    return execution


def forward(impl, query, key, value, kv_cache, meta, output, *, heads, group, settings,
            layer_owner=None):
    p=meta.bfla_plan
    owned_alias=(impl.kv_sharing_target_layer_name is not None and
                 p.reason=='fresh_prefill' and bool(p.request_keys) and
                 len(p.request_keys)==len(p.kv_lengths)==len(p.prompt_lengths)==len(p.lengths))
    if p.reason in ('extend_prefill','mixed_prefill') or owned_alias:
        return extend(impl, query, kv_cache, meta, output, heads=heads, group=group,
                      settings=settings, layer_owner=layer_owner)
    lengths = meta.bfla_plan.lengths
    n = sum(lengths)
    q = query[:n]
    if impl.kv_sharing_target_layer_name is not None:
        pairs = [materialize_alias_kv(kv_cache, meta.block_table[r], size,
                                     impl.head_size)
                 for r, size in enumerate(lengths)]
        k, v = pairs[0] if len(pairs) == 1 else (
            torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs]))
        COUNTS['alias_materializations'] += len(lengths)
    else:
        k, v = key[:n], value[:n]
    identity = (heads, lengths, impl.scale, q.dtype, q.device, settings)
    execution = _reuse_reservation(identity)
    if execution is None:
        execution = PreparedExecution(q, k, v, impl.scale,
            request_lengths=lengths, head_layout=heads, group=group, settings=settings)
        RESERVATIONS[identity] = execution
    if output is None:
        output = torch.empty_like(query)
    before = execution.captures
    execution(q, k, v, output[:n])
    COUNTS['captures'] += execution.captures - before
    COUNTS['sparse_prefill_requests'] += len(lengths)
    COUNTS['sparse_prefill_tokens'] += n
    return output


def extend(impl, query, kv_cache, meta, output, *, heads, group, settings, layer_owner):
    """Read owner KV pages after the framework's native write, then sparse P."""
    p = meta.bfla_plan
    rows = p.rows or tuple((i,a,b,n) for i,(a,b,n) in enumerate(
        zip(p.starts,p.starts[1:],p.kv_lengths)))
    if output is None:
        output = torch.empty_like(query)
    consume_prefills(query,kv_cache,meta.block_table,rows,heads,group,
        impl.scale,output,settings=settings,prompt_lengths=p.prompt_lengths,
        request_keys=p.request_keys,layer_owner=layer_owner)
    COUNTS['extend_prefill_requests'] += len(rows)
    if impl.kv_sharing_target_layer_name is not None:
        COUNTS['alias_native_page_bindings'] += len(rows)
    return output


def consume_prefills(query, cache, table, rows, heads, group, scale, output, *,
                     settings, prompt_lengths, request_keys, layer_owner,
                     live_requests=None, postprocess=None):
    """Run paged prefill for the Qwen3.8-Flash-Next and ordinary vLLM adapters."""
    from .paged import native_views
    from .partition import write_rows
    if (not rows or len(rows)!=len(prompt_lengths) or len(rows)!=len(request_keys)
            or layer_owner is None):
        raise ValueError('prefill requires original prompts and layer/request ownership')
    lengths = tuple(b-a for _,a,b,_ in rows)
    klens = tuple(n for *_,n in rows)
    prompts = tuple(prompt_lengths)
    n = sum(lengths)
    if len(set(request_keys)) != len(request_keys):
        raise ValueError('prefill request owners must be unique within a batch')
    if live_requests is not None:
        retain_requests(live_requests)
    k,v=native_views(cache,heads.head_dim)
    states = []
    for key, query_length, kv_length in zip(request_keys, lengths, klens):
        owner = (layer_owner, key, settings.config, heads, query.dtype, query.device)
        state = POOLS.get(owner)
        if state is None:
            state = POOLS[owner] = IndexPoolState(settings.config)
        elif kv_length - query_length < state.filled:
            # A retried external-KV load may repair a shared page even before
            # this request's restart position. Rebuild this owner's whole pool.
            state.reset()
        states.append(state)
    # Geometry-only selection graph shared across layers. Native KV cache
    # addresses are deliberately excluded: attention is submitted separately.
    identity = ('extend-native-paged',heads,lengths,klens,prompts,scale,query.dtype,query.device,settings,
                k.shape[1],tuple(k.stride()),tuple(v.stride()))
    execution = _reuse_reservation(identity)
    if execution is None:
        qbuf = torch.empty((n,heads.local_hq,heads.head_dim),device=query.device,dtype=query.dtype)
        width=(max(klens)+k.shape[1]-1)//k.shape[1]
        row_ids=integer_tensor((r for r,*_ in rows),device=query.device)
        request_table=torch.index_select(table[:,:width],0,row_ids)
        execution = PreparedExecution(qbuf,k,v,scale,
            request_lengths=lengths,kv_lengths=klens,prompt_lengths=prompts,
            head_layout=heads,group=group,settings=settings,copy_inputs=False,
            kv_page_table=request_table)
        RESERVATIONS[identity] = execution
    execution.bind_paged_kv(k,v,table,tuple(r for r,*_ in rows),states)
    qbuf=execution.inputs[0]
    qoffset = 0
    for _,a,b,size in rows:
        qbuf[qoffset:qoffset+b-a].copy_(query[a:b])
        qoffset += b-a
    before = execution.captures
    contiguous = all(a[2] == b[1] for a,b in zip(rows,rows[1:]))
    target = (output[rows[0][1]:rows[-1][2]] if contiguous and postprocess is None
              else execution.output)
    execution.replay(target)
    if postprocess is not None:
        pos = 0
        for _,a,b,_ in rows:
            postprocess(target[pos:pos+b-a],a,b,output[a:b])
            pos += b-a
    elif not contiguous:
        write_rows(output,rows,target)
    execution.record_use()
    for state in states:
        if state.pool is not None:
            current=torch.cuda.current_stream(query.device)
            state.pool.record_stream(current); state.scales.record_stream(current)
    execution.unbind_kv()
    completed = {key for key,prompt,size in zip(request_keys,prompts,klens) if size == prompt}
    release_requests(completed, layer_owner=layer_owner)
    COUNTS['captures'] += execution.captures-before
    COUNTS['sparse_prefill_requests'] += len(lengths)
    COUNTS['sparse_prefill_tokens'] += n
    COUNTS['prefix_tokens_visible'] += sum(k-q for k,q in zip(klens,lengths))
    COUNTS['key_feature_refresh_rows'] += sum(execution.plan.last_key_refresh_rows)
    COUNTS['index_key_gather_rows'] += sum(execution.plan.last_key_refresh_rows)
    COUNTS['native_paged_attention_calls'] += 1
    COUNTS['selection_graph_replays'] += int(settings.execution == 'localgraph')
    return execution


def release_request(request_id):
    release_requests({request_id})


def release_requests(request_ids, *, layer_owner=None):
    if not request_ids:
        return
    for key in tuple(POOLS):
        if key[1] in request_ids and (layer_owner is None or key[0] == layer_owner):
            del POOLS[key]


def retain_requests(request_ids):
    live = set(request_ids)
    for key in tuple(POOLS):
        if key[1] not in live:
            del POOLS[key]


def release():
    """Release adapter reservations before model/communicator teardown."""
    for execution in RESERVATIONS.values():
        execution.close()
    RESERVATIONS.clear()
    POOLS.clear()
    from bfla.prepared import METADATA as plan_metadata
    from .metadata import METADATA as request_metadata
    plan_metadata.clear()
    request_metadata.clear()
