"""Block selection budgets, tile sizes, and protected context regions."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BFLAConfig:
    ratio_numerator: int = 1
    ratio_denominator: int = 3
    k_min: int = 2048
    k_max: int | None = 32768
    index_dim: int = 128
    index_dtype: str = 'fp8'
    pool_block_size: int = 64
    query_tile_rows: int = 128
    tail_tiles: int = 8
    short_full_threshold: int = 2048
    sink_tokens: int = 256
    local_alignment: int = 128
    local_blocks: int = 4

    def __post_init__(self):
        positive = ('ratio_numerator', 'ratio_denominator', 'k_min', 'index_dim',
                    'pool_block_size', 'query_tile_rows', 'local_alignment', 'local_blocks')
        for name in positive:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('tail_tiles', 'short_full_threshold', 'sink_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.ratio_numerator > self.ratio_denominator:
            raise ValueError('ratio must be in (0, 1]')
        if self.k_max is not None and (type(self.k_max) is not int or self.k_max < self.k_min):
            raise ValueError('k_max must be None or an integer >= k_min')
        if self.index_dim not in (64, 128, 256):
            raise ValueError('index_dim must be 64, 128 or 256')
        if self.index_dtype not in ('fp8', 'bf16'):
            raise ValueError('index_dtype must be fp8 or bf16')
        if self.pool_block_size not in (64, 128, 256):
            raise ValueError('pool_block_size must be 64, 128 or 256')
        if self.query_tile_rows not in (128, 256, 512):
            raise ValueError('query_tile_rows must be 128, 256 or 512')

    def budget(self, visible: int) -> int:
        if type(visible) is not int or visible < 1:
            raise ValueError('visible length must be a positive integer')
        target = (visible * self.ratio_numerator + self.ratio_denominator - 1) // self.ratio_denominator
        if self.k_max is not None:
            target = min(self.k_max, target)
        return min(visible, max(self.k_min, target))

    def block_count(self, visible: int) -> int:
        return (self.budget(visible) + self.pool_block_size - 1) // self.pool_block_size
