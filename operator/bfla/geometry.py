"""Global feature grouping and contiguous tensor-parallel head ownership."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadLayout:
    global_hq: int
    global_hkv: int
    head_dim: int
    tp_size: int
    tp_rank: int
    query_head_ids: tuple
    kv_head_ids: tuple
    kv_replication: int

    @classmethod
    def create(cls, hq, hkv, d, tp_size=1, tp_rank=0):
        if (any(type(x) is not int for x in (hq, hkv, d, tp_size, tp_rank)) or
            min(hq, hkv, d, tp_size) < 1 or hq % hkv or hq % tp_size or
            not 0 <= tp_rank < tp_size):
            raise ValueError('invalid global head geometry or TP rank')
        local_q = hq // tp_size
        qids = tuple(range(tp_rank * local_q, (tp_rank + 1) * local_q))
        if hkv >= tp_size:
            if hkv % tp_size:
                raise ValueError('KV heads must partition evenly across TP')
            local_k = hkv // tp_size
            kids, replicas = tuple(range(tp_rank * local_k, (tp_rank + 1) * local_k)), 1
        else:
            if tp_size % hkv:
                raise ValueError('replicated KV heads must map evenly to ranks')
            replicas = tp_size // hkv
            kids = (tp_rank // replicas,)
        if local_q % len(kids):
            raise ValueError('each local query-head group needs a complete KV owner')
        return cls(hq, hkv, d, tp_size, tp_rank, qids, kids, replicas)

    @property
    def local_hq(self): return len(self.query_head_ids)
    @property
    def local_hkv(self): return len(self.kv_head_ids)
    @property
    def selection_ratio(self): return self.global_hq // self.global_hkv
    @property
    def consumer_ratio(self): return self.local_hq // self.local_hkv
    @property
    def key_weight(self): return 1. / self.kv_replication

    def validate_local(self, hq, hkv, d):
        if (hq, hkv, d) != (self.local_hq, self.local_hkv, self.head_dim):
            raise ValueError('local head geometry differs from global TP layout')
