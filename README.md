# BFLA: Block-Filtered Long-Context Attention Mechanism (DuSAv2)

This repository provides the source code for **BFLA (Block-Filtered Long-Context Attention)**, a training-free sparse prefill attention mechanism that extends `DuSA (NeurIPS'25)`. BFLA is designed for vLLM-style paged-attention workloads and accelerates the prefilling stage of modern open-source LLMs, including the Gemma 4, Llama 3.1, and Qwen 3.5/3.6 series. In our evaluations, BFLA achieves up to **2-3x** prefill speedups on long-context workloads with minimal accuracy degradation compared with `FlashAttention-2`.

## Highlights

1. **Runtime block-level importance estimation.** BFLA compresses Q/K blocks into lightweight pooled representations and estimates causal block importance using low-cost block-level scores. In our measurements, this mask construction path can be up to **5x** faster than `XAttention (ICML'25)`-style block scoring. Important KV blocks are selected by mass/threshold-based criteria.

2. **Dynamic block/tile sparse prefill attention.** The Triton kernel computes attention only over selected KV tiles, while preserving local neighborhoods and applying optional speculative rescue to reduce information loss.

3. **Plug-and-play training-free acceleration.** BFLA requires no retraining or weight modification, supports runtime sparsity control, and provides near-dense accuracy with significant prefill speedups in vLLM-style paged attention workloads.

## Files and Target Paths

Copy each file into the same relative path under the target server's installed `vllm` (supported version: 0.19.1) package directory, for example:

`/path/to/conda/env/lib/python3.12/site-packages/vllm/...`

| Repository file | Target path under `site-packages/vllm` | Purpose |
|---|---|---|
| `vllm/v1/attention/ops/triton_unified_attention.py` | `vllm/v1/attention/ops/triton_unified_attention.py` | Main Triton unified attention implementation. Adds BFLA block-mask building, sparse tile skipping in the Triton prefill kernel, torch/triton mask builder options, prefill/decode gating, prefix/cascade safety checks, sliding-window filtering, and speculative rescue options. |
| `vllm/v1/attention/backends/triton_attn.py` | `vllm/v1/attention/backends/triton_attn.py` | Triton attention backend wrapper. Carries BFLA metadata into `triton_unified_attention`, handles common prefix/cascade metadata, and passes `bfla_allow_sparse_prefill`/`common_prefix_len`. |
| `vllm/v1/attention/backend.py` | `vllm/v1/attention/backend.py` | Shared attention metadata definitions. Adds `bfla_allow_sparse_prefill` and keeps compatibility with old/new vLLM attention metadata paths. |
| `vllm/v1/worker/gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | Builds runtime attention metadata. Computes whether the current batch is safe for BFLA sparse prefill, while avoiding decode, mixed unsafe prefix paths, and problematic cascade/prefix cases. |

Flat copies of the same files are also kept at the top level for quick inspection:

- `gpu_model_runner.py`
- `backend.py`
- `triton_attn.py`
- `triton_unified_attention.py`

## Migration Steps

1. Locate the target vLLM package directory:

```bash
python3 - <<'PY'
import vllm, pathlib
print(pathlib.Path(vllm.__file__).resolve().parent)
PY
```

2. Back up the target server's original files first:

```bash
VLLM_DIR=/path/to/site-packages/vllm
mkdir -p ~/vllm_original_backup
cp "$VLLM_DIR/v1/worker/gpu_model_runner.py" ~/vllm_original_backup/
cp "$VLLM_DIR/v1/attention/backend.py" ~/vllm_original_backup/
cp "$VLLM_DIR/v1/attention/backends/triton_attn.py" ~/vllm_original_backup/
cp "$VLLM_DIR/v1/attention/ops/triton_unified_attention.py" ~/vllm_original_backup/
```

3. Copy this backup's `vllm/` tree over the target package root:

```bash
VLLM_DIR=/path/to/site-packages/vllm
cp -r ./vllm/* "$VLLM_DIR/"
```

4. Clear stale Python bytecode if needed:

```bash
find "$VLLM_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} +
```

5. Run with `attention_backend=TRITON_ATTN`.

## Recommended Runtime Configuration

A conservative speed/accuracy setting that worked well across our Gemma 4, Llama 3.1, and Qwen 3.5/3.6 experiments is:

```bash
export VLLM_BFLA_MASK_IMPL=torch
export VLLM_BFLA_MIN_PREFILL_TOKENS=256
export VLLM_BFLA_TORCH_MASK_BLOCK_N=256
export VLLM_BFLA_TORCH_POOL=flat64
export VLLM_BFLA_THRESHOLD=999
export VLLM_BFLA_KEEP_MASS=0.99
export VLLM_BFLA_KEEP_RATIO=0
export VLLM_BFLA_MIN_KEEP_BLOCKS=0
export VLLM_BFLA_LOCAL_BLOCKS=8
export VLLM_BFLA_SPEC_STRIDE=0
export VLLM_BFLA_SPEC_PROB=0
export VLLM_BFLA_SPEC_SEED=1
```

## Core Parameters

| Variable | Meaning | Notes |
|---|---|---|
| `VLLM_BFLA_MASK_IMPL` | Mask builder implementation: `torch`, `triton_sample`, or `triton_centroid`. | `torch` is the most tested path. `triton_sample`/`triton_centroid` are experimental lightweight GPU-side mask builders. |
| `VLLM_BFLA_MIN_PREFILL_TOKENS` | Minimum prefill token count before enabling sparse BFLA. | Use `256` for broad testing. Raise it to avoid overhead on short prompts. |
| `VLLM_BFLA_TORCH_MASK_BLOCK_N` | Coarse mask block size for torch mask building. | Tested values: `128`, `256`, `512`. `256` is a stable default.|
| `VLLM_BFLA_TORCH_POOL` | How to compress each Q/K block before scoring. | `flat64` performed best for speed. `mean` and `maxabs` showed little speedup. |
| `VLLM_BFLA_THRESHOLD` | Score threshold for threshold-based keep. | Use `999` to effectively disable threshold keep and rely on mass/local/spec logic. `0` is dense-like sanity control because nearly all causal tiles are kept. |
| `VLLM_BFLA_KEEP_MASS` | Per-block cumulative mass target for keeping important KV tiles. | Tested `0.90`, `0.95`, `0.99`. `0.99` is a conservative default; despite the high mass target, it can still provide substantial sparsity because the block-level score distribution is often concentrated. Lower values are more aggressive. |
| `VLLM_BFLA_KEEP_RATIO` | Fixed top-ratio keep fallback. | `0` disables fixed-ratio keep. Use only when you want a hard top-ratio policy. |
| `VLLM_BFLA_MIN_KEEP_BLOCKS` | Always keep this many early/prefix KV blocks. | `0` for aggressive tests. Increase if prefix fidelity becomes important. |
| `VLLM_BFLA_LOCAL_BLOCKS` | Always keep recent local KV blocks per Q block. | Tested `4`, `8`, `16`. Local keep is important for causal/local continuity. |
| `VLLM_BFLA_SPEC_STRIDE` | Speculative rescue: keep every N-th otherwise-dropped KV tile. | `0` disables it. `16` adds safety but costs speed; in the latest AIME run it did not clearly help. |
| `VLLM_BFLA_SPEC_PROB` | Random speculative rescue probability for dropped tiles. | `0` disables random rescue. If enabled, use a fixed seed for reproducibility. |
| `VLLM_BFLA_SPEC_SEED` | Seed for random speculative rescue. | Used only when `SPEC_PROB > 0`. |
| `VLLM_BFLA_REUSE_MASK` | Reuse one built mask across compatible layers. | Experimental. Can reduce mask-building overhead, but validate accuracy carefully. |
| `VLLM_BFLA_SAMPLE_D` | Sampled dimension for Triton sample/centroid mask builders. | Applies to non-`torch` implementations. |
| `VLLM_BFLA_SAMPLE_KV_GROUP` | Number of KV tiles grouped in Triton sample/centroid builders. | Larger groups reduce mask overhead but may lose detail. |
| `VLLM_BFLA_CENTROIDS` | Number of centroids for `triton_centroid`. | Experimental. |
| `VLLM_BFLA_SAMPLE_THRESHOLD` | Threshold for Triton sample/centroid mask builders. | Defaults to `VLLM_BFLA_THRESHOLD` if unset. |

## BFLA Related Papers

If you find this repository useful, please consider citing the related papers.

```latex
@article{BFLA,
title={BFLA: Block-Filtered Long-Context Attention Mechanism},
author={Wu, Chong and Feng, Zhenan and Xu, Renjie and Zhang, Houwang and Cao, Jiawang and Che, Maolin and Zhu, Wenbo and Yan, Hong},
url={https://arxiv.org/abs/2605.12193},
year={2026}}

@inproceedings{DuSA,
 author = {Wu, Chong and Cao, Jiawang and Xu, Renjie and Ran, Zhuoheng and Che, Maolin and Zhu, Wenbo and Yan, Hong},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages = {41087--41113},
 publisher = {Curran Associates, Inc.},
 title = {DuSA: Fast and Accurate Dual-Stage Sparse Attention Mechanism Accelerating Both Training and Inference},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/3ab868033387edbdd775b8edd78ed056-Paper-Conference.pdf},
 volume = {38},
 year = {2025}
}
```

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations.
