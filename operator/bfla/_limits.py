"""Fixed execution contracts of the packaged kernels, not selection policy.

These values describe the compiled consumers/CUB variants. Changing them alone
does not add kernel support. In particular, a native KV page may have any size;
KV_BLOCK_TOKENS is the logical traversal tile, not the native page size.
"""

QUERY_TILE_ROWS = 128
KV_BLOCK_TOKENS = 64

# Bounded, block-aligned original-K scratch; no per-layer full KV copy.
KEY_READER_ROWS = 16384
# Query feature batching is independent of the key-reader window.
SELECTOR_WAVE_ROWS = 8192
SELECTOR_BATCH_ROWS = 16384
MAX_KV_TOKENS = 1_000_000

# (selected capacity, candidate capacity), matching selector*.cu templates.
SELECTOR_VARIANTS = ((512, 16384), (1024, 2048), (2048, 16384))
MAX_SELECTED_BLOCKS = max(selected for selected, _ in SELECTOR_VARIANTS)
MAX_CANDIDATE_BLOCKS = max(candidates for _, candidates in SELECTOR_VARIANTS)
