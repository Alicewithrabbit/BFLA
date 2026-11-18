"""Bounded publication of immutable integer metadata on the caller's stream."""
from collections import OrderedDict
from threading import RLock
import torch


class MetadataCache:
    """Reuse read-only device views; never overwrite an in-flight publication.

    Keys identify the complete contents returned by ``factory``. The factory
    returns named CPU integer tensors. ``max_bytes`` counts retained host and
    device slabs together. Callers own the lifetime of returned views, including
    any subsequent raw-pointer launches on other streams.
    """
    def __init__(self, max_entries=64, max_bytes=32 << 20):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries = OrderedDict()
        self._bytes = self.hits = self.misses = 0
        self._lock = RLock()

    def get(self, key, factory, device):
        device = torch.device(device)
        if device.type == 'cpu':
            return factory()
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        stream = torch.cuda.current_stream(device)
        identity = (device, key)
        with self._lock:
            entry = self._entries.get(identity)
            if entry is None:
                self.misses += 1
                arrays = factory()
                groups = {}
                for name, tensor in arrays.items():
                    if tensor.device.type != 'cpu' or tensor.dtype not in (torch.int32, torch.int64):
                        raise ValueError('metadata must be CPU int32/int64 tensors')
                    groups.setdefault(tensor.dtype, []).append((name, tensor))
                views, hosts, slabs = {}, [], []
                size = 0
                for dtype, items in groups.items():
                    count = sum(x.numel() for _, x in items)
                    host = torch.empty(count, dtype=dtype, device='cpu', pin_memory=True)
                    slab = torch.empty(count, dtype=dtype, device=device)
                    offset = 0
                    for name, x in items:
                        n = x.numel()
                        host[offset:offset+n].copy_(x.reshape(-1))
                        views[name] = slab[offset:offset+n].view(x.shape)
                        offset += n
                    slab.copy_(host, non_blocking=True)
                    hosts.append(host); slabs.append(slab)
                    size += 2 * count * host.element_size()
                ready = torch.cuda.Event()
                ready.record(stream)
                entry = (views, hosts, slabs, ready, stream.cuda_stream, size)
                if self.max_entries > 0 and size <= self.max_bytes:
                    while self._entries and (len(self._entries) >= self.max_entries or
                                             self._bytes + size > self.max_bytes):
                        _, old = self._entries.popitem(last=False)
                        self._bytes -= old[-1]
                    self._entries[identity] = entry
                    self._bytes += size
            else:
                self.hits += 1
                self._entries.move_to_end(identity)
            views, _, slabs, ready, producer, _ = entry
            if producer != stream.cuda_stream:
                stream.wait_event(ready)
            for slab in slabs:
                slab.record_stream(stream)
            return views

    def clear(self):
        # Pinned nonblocking copies record their source allocation in PyTorch's
        # host allocator. Eviction never recycles or mutates live slab contents.
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def info(self):
        with self._lock:
            return dict(entries=len(self._entries), bytes=self._bytes,
                        hits=self.hits, misses=self.misses)
