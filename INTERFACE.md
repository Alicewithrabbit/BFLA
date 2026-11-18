# Operator / framework boundary

```text
Framework: post-RoPE Q/K, original V, native cache, request and TP metadata
  → adapter: resolve layout/alias ownership; bind lengths and reducer
  → bfla: features → pooling → scoring → CUB → CSR → paged attention
  → adapter: output ownership and any model gate
  → native model and decoding
```

| Owner | State |
|---|---|
| Framework | Native raw paged KV, block tables, slot mapping and alias owner |
| Adapter | Request/layer generations, compact-state lifetimes, release, plan reuse, graph lifecycle, TP communicator |
| BFLAPrefillPlan | Features, pooled indices, scores, selection, CSR and scheduler buffers |
| Caller-owned IndexPoolState | Optional compact indices for one immutable growing prefix |
| Caller | Q/K/V and optional output buffer |

## Public API

`BFLAConfig`, `HeadLayout`, `SupportResult` and `supports` import without
PyTorch, Triton or vLLM. `HeadLayout.create` maps contiguous global Q/KV heads to
a TP rank, including replicated KV heads. Use `supports` to check execution
support for the resulting layout and configuration.

```python
BFLAPrefillPlan(q, k, v, scale, *, request_lengths, head_layout,
               all_reduce=None, kv_lengths=None, prompt_lengths=None,
               kv_padded=False, kv_page_offsets=None, kv_page_table=None, config=BFLAConfig())
plan.run(q, k, v, *, out=None, key_states=None)
# Optional incremental binding and separate execution boundaries:
plan.bind_kv(k, v, page_offsets)  # optional caller-owned block64 arena
plan.bind_paged_kv(k, v, table, request_rows=None)  # native physical pages
plan.prepare_keys(k, *, key_states=None)
plan.run_selection(q)
plan.run_attention(q, k, v, *, out=None)
```

`request_lengths` describes packed queries. `kv_lengths` defaults to query
lengths and includes visible prefix history. `prompt_lengths` defaults to KV
lengths and must retain the original prompt length across chunks. Each request
satisfies `0 < query_length <= kv_length <= prompt_length`. Causal query positions
start at `kv_length-query_length`.

Q: `[sum(query_lengths), local_Hq, D]`. K/V:
`[sum(kv_lengths), local_Hkv, D]`. With `kv_padded=True`, each request occupies
`ceil(kv_length/64)*64` rows, independently of pooling granularity. Q must be contiguous for TP1 or D512.
Packed and padded D512 also require contiguous K/V. Strided inputs require
unit stride in the last dimension and 16-byte alignment. Output must
be contiguous and disjoint from inputs.

With `kv_padded=True, kv_page_offsets=...`, K/V may be larger caller-owned
block64 arenas. Each offset gives the first physical block64 page for a request;
visible segments must be in bounds and disjoint, but need not follow query
order. Logical selected IDs and causal positions do not change. `bind_kv` is
host preparation outside capture; it updates physical addresses and backing
extent. The split API requires `prepare_keys` before `run_selection`, followed
by `run_attention`. The caller orders updates and consumers. Selection graphs
can be shared across arenas; captured attention uses fixed backing pointers.

With `kv_page_table`, K/V are views `[physical_page, token, local_Hkv, D]`,
with caller-owned BF16 storage and explicit strides. The pooling block size
determines selection granularity. Attention traverses selected blocks in
64-token tiles and resolves them through each request's native page table.
For D256, a complete tile within one physical page uses TMA. Partial or
cross-page tiles and all paged D512 loads use cp.async. Packed inputs use the
TMA consumer with plan-owned tail padding when needed. Caller-padded inputs
with a partial final tile use the paged consumer to mask inactive lanes.
Index building reads K in windows of at most 16K rows. Rebinding happens outside capture; each visible
logical row maps to a valid physical page.

For TP>1, `all_reduce(tensor)` returns the global **sum** on the current stream;
reduction precedes nonlinear normalization. Replicated KV heads are weighted to
avoid double counting. Raw KV and softmax outputs remain local to each rank.

`IndexPoolState(config)` retains pooled indices. Pass
one state per request via `key_states`; the caller owns request/layer identity.
Call `reset()` whenever a prefix is replaced or modified, including same-length
replacements. Append recomputes the incomplete last block from original keys;
shrinking lengths reset computation. Each request/layer owns its state;
simultaneous executions use separate plans.

## Plan lifetime and CUDA Graph capture

Compatible plans share immutable integer metadata through a bounded cache.
New metadata is packed into pinned host slabs and uploaded asynchronously;
cross-stream consumers wait on its completion event. Mutable selection and
scheduler buffers remain private to each plan.

Preparation reserves metadata and workspace for a fixed geometry. Reuse a plan
sequentially on one stream; simultaneous calls use separate plans. The caller
owns graph capture and completes preparation, JIT compilation and allocation
before capture. Incremental index-state growth runs outside capture. Captured
operations require stable buffer addresses and shapes.

The [vLLM integration guide](vllm-integration/README.md) describes request
classification, cache lifetimes, local-graph execution and version compatibility.
