"""Views of framework-owned BF16 KV; no raw-KV staging allocation or copy."""


def native_views(cache, head_dim):
    """Split the vLLM 0.31 BF16 cache, preserving native addresses."""
    if cache.ndim != 4 or cache.shape[-1] != 2 * head_dim:
        raise ValueError('expected native [page, head, token, 2D] cache')
    key, value = cache.transpose(1, 2).split(head_dim, dim=-1)
    shape = key.shape
    if 1 in shape:
        # Like native FlashAttention, give size-one axes a canonical stride
        # for the TMA descriptor. K and V have identical strides after split.
        strides = list(key.stride())
        next_stride = 1
        for axis in reversed(range(len(shape))):
            if shape[axis] == 1:
                strides[axis] = next_stride
            next_stride = shape[axis] * strides[axis]
        if tuple(strides) != key.stride():
            key, value = key.as_strided(shape, strides), value.as_strided(shape, strides)
    return key, value
