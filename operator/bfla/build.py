"""Lazy SM90a extension builds using only installed package resources.

Build products use PyTorch's extension cache (including its configured cache
location), so an installed package may be read-only. Importing this module does
not import PyTorch, initialize CUDA, or compile an extension.
"""
from functools import lru_cache
from pathlib import Path
from ._limits import SELECTOR_VARIANTS, MAX_SELECTED_BLOCKS, MAX_CANDIDATE_BLOCKS


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "_csrc"
_VENDOR = _ROOT / "_vendor"
_ARCH = "-gencode=arch=compute_90a,code=sm_90a"


def _load(name, sources, *, include=(), cuda_flags=(), cflags=()):
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("BFLA CUDA extensions require an SM90 device")
    paths = [Path(source) for source in sources]
    missing = [str(path) for path in (*paths, *include) if not Path(path).exists()]
    if missing:
        raise RuntimeError("BFLA package resources are missing: " + ", ".join(missing))
    return load(
        name=name,
        sources=[str(path) for path in paths],
        extra_include_paths=[str(path) for path in include],
        extra_cflags=list(cflags),
        extra_cuda_cflags=[*cuda_flags, _ARCH],
        verbose=False,
    )


@lru_cache(None)
def attention(head_dim):
    """Return the SM90a attention module exposing the pointer-based ``forward``."""
    if type(head_dim) is not int or head_dim not in (256, 512):
        raise ValueError("attention head_dim must be 256 or 512")
    headers = _SOURCE / f"attention_d{head_dim}"
    include = [headers, _VENDOR / "cutlass/include"]
    if head_dim == 256:
        flags = [
            "-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
            "--expt-extended-lambda", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-DBFLA_FAST_INTERIOR=1", "-DNDEBUG", "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
            "-DBFLA_STRIDED_INPUT=1", "-DBFLA_TMA_KV=1",
        ]
        cflags = []
    else:
        flags = [
            "-O3", "-DNDEBUG", "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
            "-DBFLA_BATCHED_TMA=1", "-DBFLA_PERSISTENT=1", "-DBFLA_FAST_INTERIOR=0",
            "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
            "--expt-extended-lambda", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        ]
        cflags = ["-O3", "-DNDEBUG"]
    return _load(
        f"bfla_attention_d{head_dim}_sm90a_v1",
        [_SOURCE / f"attention_d{head_dim}.cu", headers / "flash_prepare_scheduler.cu"],
        include=include, cuda_flags=flags, cflags=cflags,
    )


@lru_cache(None)
def key_feature():
    """Return the D256, four-KV-head DI128 warp feature module."""
    return _load(
        "bfla_key_feature_warp_sm90a_v1", [_SOURCE / "key_feature_warp.cu"],
        cuda_flags=["-O3", "-std=c++17"],
    )


@lru_cache(None)
def feature_extension():
    """Return the tensor-based DI128 key-feature module."""
    feature = _SOURCE / "feature"
    return _load(
        "bfla_key_feature_sm90a_v1",
        [_SOURCE / "key_feature_bindings.cpp", feature / "indexer_key_feature.cu"],
        include=[feature], cflags=["-O3"], cuda_flags=["-O3", "-std=c++17"],
    )


def selector(capacity=512, *, candidates=None):
    """Return the smallest CUB implementation covering both capacities.

    The 1024-slot variant has a smaller candidate limit than the other variants.
    Preparation supplies the candidate count so callers need no special-case
    promotion. Omission retains the capacity-only dispatch for existing callers.
    """
    if type(capacity) is not int or not 1 <= capacity <= MAX_SELECTED_BLOCKS:
        raise ValueError(f"selector capacity must be an integer in [1, {MAX_SELECTED_BLOCKS}]")
    if candidates is not None and (type(candidates) is not int or not 1 <= candidates <= MAX_CANDIDATE_BLOCKS):
        raise ValueError(f"selector candidates must be an integer in [1, {MAX_CANDIDATE_BLOCKS}]")
    bound = next(selected for selected, limit in SELECTOR_VARIANTS
                 if capacity <= selected and (candidates is None or candidates <= limit))
    return _selector(bound)


@lru_cache(None)
def _selector(bound):
    cccl = _VENDOR / "cccl"
    return _load(
        f"bfla_selector_cub{bound}_sm90a_v1", [_SOURCE / f"selector{bound}.cu"],
        include=[cccl / "cub", cccl / "libcudacxx/include", cccl / "thrust"],
        cuda_flags=["-O3", "-std=c++17"],
    )


@lru_cache(None)
def paged_attention(head_dim):
    """Original BF16 KV pages, including pages crossing logical block64 tiles."""
    if type(head_dim) is not int or head_dim not in (256,512):
        raise ValueError('paged attention head_dim must be 256 or 512')
    headers=_SOURCE / f'attention_d{head_dim}'
    return _load(f'bfla_native_paged_d{head_dim}_sm90a_v1',
        [_SOURCE/'attention_paged.cu',headers/'flash_prepare_scheduler.cu'],
        include=[headers,_VENDOR/'cutlass/include'],cflags=['-O3','-DNDEBUG'],
        cuda_flags=['-O3','-std=c++17','--use_fast_math','--expt-relaxed-constexpr',
                    '--expt-extended-lambda','-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED',
                    '-DNDEBUG','-DCUTLASS_DEBUG_TRACE_LEVEL=0','-DBFLA_PERSISTENT=1',
                    f'-DBFLA_FAST_INTERIOR={int(head_dim==256)}',f'-DBFLA_HEAD_DIM={head_dim}'])
