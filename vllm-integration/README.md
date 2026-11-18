# bfla-vllm

An independent vLLM prefill adapter built on the public `bfla` API. It manages
request metadata, native KV caches, TP collectives, request lifetimes and
optional local CUDA graphs. Head geometry comes from the layer's global head
metadata and model configuration, together with TP-local heads.

## Installation and native-backend integration

Install both wheels in the target vLLM environment, or run from the repository root:

```sh
python -m pip install ./operator ./vllm-integration
```

Use an existing **vLLM 0.31.x** installation.

```sh
export VLLM_PLUGINS=bfla
export BFLA_NATIVE_BACKEND=flash_attn  # or triton, matching the native backend
export BFLA_EXECUTION=eager           # or localgraph
```

Set vLLM's `attention_config.backend` to `CUSTOM`. FlashAttention version,
KV cache dtype and other native options follow the deployment configuration.

Mixed prefill/decode (P/D) batches are partitioned once during metadata
construction: BFLA handles supported prefill requests, and the selected native
backend handles decode. Each writes disjoint output spans. Classification uses
framework phase flags, including single-token prefix hits; decode uses actual
device lengths. Request context supplies the original prompt length and
identity, while the framework supplies visible prefill lengths.

BFLA handles causal global prefill within the original prompt. Decoding,
sliding-window attention, custom attention semantics, context parallelism and
speculative decode use the selected native backend. Dual-batch overlap uses native
attention for both microbatches. TP prefill uses native attention when the
PyNccl communicator for BFLA feature reduction is unavailable.

To enable chunking and prefix reuse with the ordinary 0.31 adapter, use:

```python
enable_chunked_prefill=True
enable_prefix_caching=True
max_num_batched_tokens=16384
```

The final setting is the per-step token budget across the entire batch; the
native scheduler determines individual chunks. Prefill uses eager execution or
adapter-local graphs. Native decoding can use `FULL_DECODE_ONLY`.

`BFLA_ENABLED=0` keeps the CUSTOM backend registered while routing attention
through its native implementation. Select an integration through
`BFLA_NATIVE_BACKEND` or `BFLA_QSA_ENABLED=1`.

## KV ownership and index-state lifetime

vLLM stores packed K/V as `[page, head, token, 2D]`. The adapter transposes
the head and token axes and splits the last dimension into K and V. It passes
the resulting `[page, token, local_Hkv, D]` views, with their native strides,
and the request's block table to the paged operator.

For chunked or cached-prefix prefill, D256/D512 attention reads native paged KV
directly, including the owner's cache for alias layers. Both execution modes
maintain compact indices per request and layer and read keys in windows of
at most 16K rows. On a prefix-cache hit, a new request builds indices for
the visible prefix when selection needs them.
Subsequent chunks update the new keys and the last incomplete block.

vLLM manages prefix-cache hits. Prefill completion, request removal and
preemption release the associated compact indices. With a single KV cache
group and external-load recomputation enabled, a failed page load also
invalidates affected cached pages. The retry publishes recomputed prefixes
as the worker confirms them.

Fresh prefill accepts packed Q/K/V. A fresh alias layer reads its owner's KV
pages directly when request identities and prompt lengths are available.
Otherwise the adapter gathers the owner's visible K/V in request order.

## Configuration and execution modes

Each `bfla.BFLAConfig` field maps to `BFLA_<UPPERCASE_FIELD_NAME>`. For example:

```sh
export BFLA_K_MAX=8192
export BFLA_INDEX_DIM=64
export BFLA_INDEX_DTYPE=bf16
export BFLA_POOL_BLOCK_SIZE=128
export BFLA_QUERY_TILE_ROWS=256
```

`BFLA_K_MAX=none` removes the policy cap within the selector's capacity. Unset
fields use `BFLAConfig` defaults. The adapter passes GPU capability and
head geometry to `bfla.supports`. Tensor-parallel feature reduction uses
vLLM's PyNccl communicator.

`eager` is the default. With `localgraph`, packed fresh prefill captures the
complete operator pipeline. The paged path captures query features, scoring
and selection. It updates key indices before replay and reads native KV pages
during attention afterward. Layers with matching geometry can reuse the selection graph.
See the [capture contract](../INTERFACE.md#plan-lifetime-and-cuda-graph-capture)
for buffer and stream lifetimes.

Local graphs require `cuda-bindings`, available through `bfla-vllm[localgraph]`.
Each graph owns a capture stream; LRU eviction releases its graph and workspace
after replay and output copies complete. Ordinary and QSA adapters share the
paged-prefill consumer and cache up to two fixed-geometry execution plans.
Compatible layers reuse each plan's query, compact-index and bounded K scratch
workspaces sequentially. Concurrent calls use separate plans.

Before model teardown, call `bfla_vllm.execution.release()` or
`bfla_vllm.qsa.consumer.release()`.

## Qwen3.8-Flash-Next QSA integration

```sh
export VLLM_PLUGINS=bfla
export BFLA_QSA_ENABLED=1
export BFLA_EXECUTION=eager
```

This adapter runs BFLA prefill and retains native QSA decoding through
the Qwen3.8-Flash-Next implementation in `vllm.models.qwen4_exp.nvidia`.
Select either this integration or `BFLA_NATIVE_BACKEND`.
[`ABI_PINS.json`](bfla_vllm/qsa/ABI_PINS.json) lists the module fingerprints.

The adapter preserves native KV writes, index-K/compressed state and the
sigmoid output gate. Chunked prefill uses the original
prompt length and request/layer ownership to locate the protected tail. Prefill
uses `BFLAPrefillPlan` and `IndexPoolState` with the cache lifecycle described
above. In a vLLM breakable CUDA graph, attention dispatch runs eagerly at a
graph break and reads the current request metadata on replay. A single-token
final prefill fragment also releases compact indices when handled by native QSA.
The adapter uses native QSA for the whole batch when
original prompt ownership is unavailable or its prefill span differs from
the native decode/prefill split (for example, during speculative verification).

`BFLA_K_ONLY_INDEX=0` disables the native index-K-only projection optimization,
which is enabled by default. Use `BFLA_DEBUG_SYNC=1` to locate asynchronous GPU
errors during debugging.
