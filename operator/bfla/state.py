"""Incremental pooled indices for a caller-owned request/layer prefix."""
from .config import BFLAConfig
from ._limits import KEY_READER_ROWS


def refresh_start(filled, length, block):
    if length < filled: return 0
    if length == filled: return length
    return filled // block * block


def pool_features(features, pool, scales, n, config):
    if (config.index_dim, config.index_dtype, config.pool_block_size) == (128, 'fp8', 64):
        from ._kernels.pooling import index_pool
        index_pool[((n + 63) // 64,)](features, pool, scales, n)
    else:
        from ._kernels.features import pool_index
        pool_index[((n + config.pool_block_size - 1) // config.pool_block_size,)](
            features, pool, scales, n, config.index_dim, config.pool_block_size,
            BF16=config.index_dtype == 'bf16')


class IndexPoolState:
    """Reuse a growing, immutable key prefix for one request and layer.

    The caller MUST reset on request/generation replacement or modification of
    existing keys, even when the new input has the same length and shape. The
    operator cannot infer logical ownership from a reused tensor address.
    """
    def __init__(self, config=BFLAConfig(), *, _pooler=pool_features):
        self.config, self._pooler = config, _pooler
        self.reset()

    def reset(self):
        self.filled = 0
        self.pool = self.scales = self.identity = None
        self._event = self._stream = None

    @property
    def logical_bytes(self):
        c = self.config
        return ((self.filled + c.pool_block_size - 1) // c.pool_block_size *
                (c.index_dim * (2 if c.index_dtype == 'bf16' else 1) + 4))

    @property
    def allocated_bytes(self):
        return 0 if self.pool is None else (self.pool.numel() * self.pool.element_size() +
                                           self.scales.numel() * self.scales.element_size())

    def populate(self, features, key, scratch, out_pool, out_scales):
        return self.populate_from_reader(features, key.shape[0], lambda a,b: key[a:b],
            scratch, out_pool, out_scales, key_shape=tuple(key.shape[1:]),
            key_dtype=key.dtype, window=((key.shape[0]+self.config.pool_block_size-1)
                                        //self.config.pool_block_size*self.config.pool_block_size))

    def populate_from_reader(self, features, length, reader, scratch, out_pool, out_scales,
                             *, key_shape, key_dtype, window=KEY_READER_ROWS):
        """Refresh compact state using bounded original-key windows.

        ``reader(start, end)`` returns contiguous logical K rows on the caller's
        stream. It may reuse one scratch buffer: feature extraction and pooling
        finish reading each window before the next call on that same stream.
        Windows align to pooled blocks, so the numerical operation per block is
        identical to full-prefix construction, including its final partial block.
        """
        import torch
        c, n = self.config, length
        block = c.pool_block_size
        if type(window) is not int or window < block or window % block:
            raise ValueError('key reader window must be a positive multiple of pool block size')
        nb = (n + block - 1) // block
        device = out_pool.device
        dtype = torch.bfloat16 if c.index_dtype == 'bf16' else torch.float8_e4m3fn
        if (n < 1 or scratch.ndim != 2 or scratch.shape[1] != c.index_dim or
            scratch.shape[0] < min(window,n) or
            out_pool.shape != (nb, c.index_dim) or out_pool.dtype != dtype or
            out_scales.shape != (nb,) or out_scales.dtype != torch.float32 or
            any(x.device != device for x in (scratch,out_scales))):
            raise ValueError('pooled index dtype/dimension/capacity differs from configuration')
        identity = (c, key_dtype, device, tuple(key_shape), scratch.dtype,
                    getattr(features, 'layout', None))
        if self.identity != identity:
            self.reset()
            self.identity = identity
        current = torch.cuda.current_stream(device) if out_pool.is_cuda else None
        if current is not None and self._event is not None and self._stream != current.cuda_stream:
            current.wait_event(self._event)
        start = refresh_start(self.filled, n, block)
        if self.pool is None or self.pool.shape[0] < nb:
            capacity = 1 << (nb - 1).bit_length()
            pool = torch.empty((capacity, c.index_dim), device=device, dtype=dtype)
            scales = torch.empty(capacity, device=device, dtype=torch.float32)
            if self.pool is not None and start:
                keep = start // block
                pool[:keep].copy_(self.pool[:keep])
                scales[:keep].copy_(self.scales[:keep])
                if self.pool.is_cuda:
                    stream = torch.cuda.current_stream(device)
                    self.pool.record_stream(stream)
                    self.scales.record_stream(stream)
            self.pool, self.scales = pool, scales
        keep = nb if start == n else start // block
        if keep:
            out_pool[:keep].copy_(self.pool[:keep])
            out_scales[:keep].copy_(self.scales[:keep])
        for first in range(start,n,window):
            end = min(n,first+window)
            key = reader(first,end)
            if (key.shape != (end-first,*key_shape) or key.dtype != key_dtype or key.device != device):
                raise ValueError('key reader returned incompatible original key rows')
            features.key(key,scratch)
            self._pooler(scratch,out_pool[first//block:(end+block-1)//block],
                         out_scales[first//block:(end+block-1)//block],end-first,c)
        if start < n:
            self.pool[start//block:nb].copy_(out_pool[start//block:nb])
            self.scales[start//block:nb].copy_(out_scales[start//block:nb])
        self.filled = n
        if current is not None:
            self.pool.record_stream(current)
            self.scales.record_stream(current)
            if self._event is None: self._event = torch.cuda.Event()
            self._event.record(current)
            self._stream = current.cuda_stream
        return n - start
