"""Small, immutable request geometry; request state stays in native tensors."""
import torch
from bfla.metadata import MetadataCache

METADATA = MetadataCache(max_entries=64, max_bytes=1 << 20)


def integer_tensor(values, *, device, dtype=torch.int64):
    values = tuple(values)
    return METADATA.get((values, dtype),
        lambda: {'value': torch.tensor(values, device='cpu', dtype=dtype)}, device)['value']
