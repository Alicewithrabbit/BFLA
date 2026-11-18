# Upstream components

BFLA's original operator code is copyright 2026 Chong WU and licensed under
[Apache-2.0](LICENSE.md).

BFLA adapts the FA3-derived block-sparse forward kernels from
[FlashPrefill V2](https://github.com/qhfan/FlashPrefillv2) for TMA loading,
batching and D256/D512 head dimensions. BFLA supplies its own block selection
and runs attention with mean correction disabled.

- [D256 kernels](bfla/_csrc/attention_d256/) and
  [D512 kernels](bfla/_csrc/attention_d512/) include code from FlashPrefill V2,
  which is distributed under [Apache-2.0](bfla/_vendor/FlashPrefillV2-LICENSE).
  For the FA3-derived code, BFLA preserves the original copyright headers and
  includes the [BSD-3-Clause license](bfla/_vendor/FlashAttention-LICENSE)
  from the root of the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE).
- `bfla/_vendor/cutlass/`: pinned CUTLASS/CuTe dependencies and licenses.
- `bfla/_vendor/cccl/`: pinned CUB, Thrust and libcu++ dependencies/licenses.
