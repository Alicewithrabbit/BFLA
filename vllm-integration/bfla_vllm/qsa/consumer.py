"""Apply the Qwen3.8-Flash-Next QSA output gate to paged BFLA prefill."""
from ..execution import consume_prefills as consume_rows, POOLS, RESERVATIONS, release


def consume_prefills(query, gate, cache, table, rows, heads, group, scale, output, *,
                     settings, prompt_lengths, request_keys, live_requests=None, layer_name=None):
    import triton as tr
    from .gate import apply_gate
    gates = gate.reshape(query.shape[0],heads.local_hq,heads.head_dim)
    def gated_write(packed,a,b,out):
        size = (b-a)*heads.local_hq*heads.head_dim
        apply_gate[(tr.cdiv(size,1024),)](packed,gates[a:b].contiguous(),out,size)
    return consume_rows(query,cache,table,rows,heads,group,scale,output,
        settings=settings,prompt_lengths=prompt_lengths,request_keys=request_keys,
        layer_owner=layer_name,live_requests=live_requests,postprocess=gated_write)
