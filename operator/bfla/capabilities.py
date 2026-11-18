"""Shape/semantic capability checks; no model names and no CUDA synchronization."""
from dataclasses import dataclass
from functools import lru_cache
from .config import BFLAConfig
from ._limits import QUERY_TILE_ROWS, MAX_KV_TOKENS, MAX_CANDIDATE_BLOCKS, MAX_SELECTED_BLOCKS


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    reasons: tuple[str, ...]

    def require(self):
        if not self.supported:
            raise ValueError('; '.join(self.reasons))


def supports(head_layout, config=BFLAConfig(), *, device_capability=(9, 0),
             dtype='bfloat16', request_lengths, kv_lengths=None, prompt_lengths=None,
             kv_layout='ragged', causal=True, softcap=0.0, custom_mask=False):
    q = tuple(request_lengths)
    k = q if kv_lengths is None else tuple(kv_lengths)
    prompt = k if prompt_lengths is None else tuple(prompt_lengths)
    return _supports(head_layout,config,tuple(device_capability),str(dtype),q,k,prompt,
                     kv_layout,causal,softcap,custom_mask)


@lru_cache(maxsize=128)
def _supports(head_layout,config,device_capability,dtype,request_lengths,kv_lengths,prompt_lengths,
              kv_layout,causal,softcap,custom_mask):
    reasons = []
    h, p = head_layout, config
    if tuple(device_capability) != (9, 0):
        reasons.append('device architecture: packaged TMA consumers require SM90')
    if str(dtype).removeprefix('torch.') not in ('bfloat16', 'bf16'):
        reasons.append('Q/K/V dtype: only BF16 consumers are built')
    if not causal or softcap != 0 or custom_mask:
        reasons.append('attention semantics: causal, no softcap and no custom mask required')
    if kv_layout not in ('ragged', 'padded', 'paged'):
        reasons.append('KV layout: packed ragged, padded block64 or native paged tensors required')
    if (h.head_dim, h.selection_ratio) not in ((256, 6), (256, 12), (512, 4)):
        reasons.append('head dimension/GQA ratio: available paths are D256/G6,G12 and D512/G4')
    if h.head_dim == 512 and (h.local_hq, h.local_hkv) not in ((8, 2), (4, 1), (2, 1)):
        reasons.append('D512 consumer supports local head shapes (8,2), (4,1), (2,1)')
    if h.consumer_ratio != h.selection_ratio and not (h.head_dim == 512 and h.selection_ratio == 4):
        reasons.append('replicated KV head mapping is only implemented for D512/G4')
    if h.head_dim % p.index_dim:
        reasons.append('index_dim must divide the original head dimension')
    if (p.sink_tokens, p.local_alignment, p.local_blocks) != (256, 128, 4):
        reasons.append('scorer supports sink=256 and local alignment/count=128/4')
    q = tuple(request_lengths)
    k = q if kv_lengths is None else tuple(kv_lengths)
    prompt = k if prompt_lengths is None else tuple(prompt_lengths)
    if (not q or len(q) != len(k) or len(q) != len(prompt) or
        any(type(x) is not int or x < 1 for x in (*q, *k, *prompt)) or
        any(qi > ki or ki > pi for qi, ki, pi in zip(q, k, prompt))):
        reasons.append('lengths must satisfy 1 <= query <= visible KV <= full prompt per request')
    else:
        if max(k) > MAX_KV_TOKENS:
            reasons.append(f'current request planner capacity is {MAX_KV_TOKENS} KV tokens')
        if max((n + p.pool_block_size - 1) // p.pool_block_size for n in k) > MAX_CANDIDATE_BLOCKS:
            reasons.append(f'CUB candidate capacity is {MAX_CANDIDATE_BLOCKS} blocks')
        if max(p.block_count(n) for n in k) > MAX_SELECTED_BLOCKS:
            reasons.append(f'CUB selected capacity is {MAX_SELECTED_BLOCKS} blocks')
        if not reasons:
            # Geometry alone cannot guarantee that mandatory support fits K.
            # Import at call time: planning itself uses this capability query.
            from .planning import _waves, mandatory_count
            share = p.query_tile_rows // QUERY_TILE_ROWS
            quota_ok = True
            for qi, ki, pi in zip(q,k,prompt):
                for wave in _waves(qi,ki-qi,pi,h.selection_ratio,h.head_dim,p):
                    for start in range(0,wave.sparse_tiles,share):
                        stop = min(start+share,wave.sparse_tiles)
                        first = ki-qi+wave.start+start*QUERY_TILE_ROWS//h.selection_ratio
                        last = ki-qi+wave.start+(min(stop*QUERY_TILE_ROWS,wave.sparse_rows*h.selection_ratio)-1)//h.selection_ratio
                        if mandatory_count(first,last,p) > p.block_count(last+1):
                            quota_ok = False
                            break
                    if not quota_ok: break
                if not quota_ok: break
            if not quota_ok:
                reasons.append('mandatory blocks exceed the configured selection quota')
    return SupportResult(not reasons, tuple(reasons))
