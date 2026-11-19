# BFLA: Block-Filtered Long-Context Attention Mechanism

Block-Filtered Long-Context Attention is a training-free sparse **prefill**
operator. Compact indexing, query-tile sharing and Ratio-K block selection feed
an FA3-derived attention consumer over the original Q/K/V. The vLLM adapter
preserves native decoding.

## Core design

1. Map post-RoPE Q/K to compact index features and pool key features into block
   indices.
2. Aggregate query features over packed query/head tiles to select shared block
   supports.
3. Use length-adaptive Ratio-K budgets, with sink/local blocks included in the
   quota and protected tail queries retaining full causal support.
4. Read the selected original K/V blocks and evaluate per-row softmax attention.

Logical selection blocks are independent of physical KV pages. During chunked
prefill, compact indices grow with each request while attention reads the
framework's native KV cache directly.

## Two independently installable packages

| Package | Responsibility | Entry point |
|---|---|---|
| [operator/](operator/) → `bfla` | Features, pooling, selection, planning and CUDA attention | `BFLAPrefillPlan` |
| [vllm-integration/](vllm-integration/) → `bfla-vllm` | Native KV binding, TP collectives, request state and optional graphs | `bfla_vllm.register` |

```sh
python -m pip install ./operator
# Optional vLLM integration:
python -m pip install ./vllm-integration
```

Use a compatible PyTorch/CUDA environment with Triton, NVCC, Ninja and a C++ compiler.
Optional dependencies: `pip install './operator[cuda]'`. CUDA extensions compile
on first use from packaged sources and headers into PyTorch's extension cache.

## Explicit plan/run API

```python
from bfla import BFLAConfig, HeadLayout, BFLAPrefillPlan, supports

# q: [sum(query_lengths), local_Hq, D]
# k/v: [sum(kv_lengths), local_Hkv, D], original BF16 tensors
heads = HeadLayout.create(hq=24, hkv=4, d=256)
config = BFLAConfig()  # ratio1/3, K2K..32K, DI128/FP8, block64, M128, tail8
supports(heads, config, device_capability=(9, 0),
         request_lengths=(4096, 8192)).require()
plan = BFLAPrefillPlan(q, k, v, 256**-0.5,
    request_lengths=(4096, 8192), head_layout=heads, config=config)
result = plan.run(q, k, v)  # or run(..., out=caller_output)
```

Native paged caches can be bound directly using `kv_page_table` and
`bind_paged_kv`. See [INTERFACE.md](INTERFACE.md) for tensor layouts, TP reduction,
plan reuse and CUDA Graph capture.

## Supported configurations

| Property | Support |
|---|---|
| Device / QKV dtype | SM90, BF16; SM90a compilation target |
| Head geometry | D256/global GQA ratio 6 or 12; D512/global ratio 4 |
| D512 local heads | (Hq,Hkv) = (8,2), (4,1), (2,1) |
| Requests | B ≥ 1, ragged lengths |
| Prefix/extend | D256/D512 explicit query/KV/full-prompt lengths |
| Native paged KV | BF16 strided K/V views with explicit page tables |
| Semantics | Causal, no soft cap/custom mask; original per-row softmax |
| Index configuration | DI64/128/256; FP8/BF16; block64/128/256; M128/256/512 |
| Per-request capacity | Up to 1,000,000 visible KV tokens, 16,384 candidate blocks and 2,048 selected blocks |

Default sink/local behavior is 256 tokens and four aligned 128-token local
windows. `short_full_threshold=2048` is independent of `k_min`; tail protection
uses eight logical M128 tiles independently of query-sharing size.
`k_max=None` removes the policy cap within the selector's capacity.
`supports` checks joint configuration/shape constraints. Tensor alignment/layout
are checked at preparation, and rebound buffers at run.

## vLLM integration

The vLLM 0.31 adapter supports fresh and chunked prefill, native prefix-cache
hits and mixed prefill/decode batches. The [integration guide](vllm-integration/README.md)
covers backend selection, alias KV and Qwen3.8-Flash-Next with native QSA decoding.

## LongBench-v2 with a 1M-token context window

On Qwen3.8-Flash-Next-FP8, BFLA achieved a **2.50× speedup in mean time to first token (TTFT)** over dense FA3. All three prefill methods used native QSA decoding.

| Prefill method | Official accuracy | Mean TTFT (s) | Mean E2E (s) |
|---|---:|---:|---:|
| Dense (FA3) | 57.85% (291/503) | 18.557 | 21.156 |
| Native QSA | **58.25% (293/503)** | 7.491 | 10.480 |
| BFLA | 58.05% (292/503) | **7.432** | **10.193** |

**Serving.** We ran vLLM 0.31 on two SM90 GPUs (TP2, batch size 1) with a 1,000,000-token context limit, 4× YaRN scaling, a 16,384-token chunked-prefill budget, and prefix caching disabled. BFLA used its default Ratio-K settings: a target ratio of 1/3, a 2K-token minimum, and a 32K-token maximum.

**Evaluation.** We used the official zero-shot, non-CoT prompt for all 503 questions, with greedy decoding, thinking disabled, and a 2,048-token output limit. For the 33 prompts exceeding 997,952 input tokens, we kept the beginning and end.

For RULER, InfiniteBench, Terminal Bench 2.1, and operator-level latency comparisons, see the [BFLA paper](https://arxiv.org/abs/2605.12193).

## Documentation

- [Operator API and state ownership](INTERFACE.md)
- [Build instructions](operator/BUILD.md)
- [vLLM integration](vllm-integration/README.md)
- [Upstream components and licenses](operator/UPSTREAM.md)

## License

BFLA's original code is copyright 2026 Chong WU and licensed under
[Apache-2.0](LICENSE.md). Third-party code retains the terms listed in the
[operator attributions](operator/UPSTREAM.md) and [adapter notice](vllm-integration/NOTICE.md).

## Citation

If you use BFLA in your research, please cite:

```bibtex
@misc{wu2026bflablockfilteredlongcontextattention,
      title={BFLA: Block-Filtered Long-Context Attention Mechanism},
      author={Chong Wu and Brian K Chen and Chuxu Song and Shezheng Song and Zhenan Feng and Renjie Xu and Houwang Zhang and Jiawang Cao and Maolin Che and Wenbo Zhu and Hong Yan},
      year={2026},
      eprint={2605.12193},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2605.12193},
}
```
