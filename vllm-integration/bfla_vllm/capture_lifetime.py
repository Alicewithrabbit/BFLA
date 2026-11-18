"""Own one non-pooled capture stream for exactly one local CUDA graph."""
import warnings

import torch
from cuda.bindings import runtime


class CaptureLifetime:
    def __init__(self, device):
        self.device = torch.device(device)
        self.graph = None
        self.raw_stream = None
        self.stream = None
        self.last_use = None
        self.last_stream = None
        self.closed = False
        with torch.cuda.device(self.device):
            status, raw = runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
            if status != runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f'Cannot create isolated capture stream: {status}')
            self.raw_stream = raw
            self.stream = torch.cuda.ExternalStream(int(raw), device=self.device)
            self.last_use = torch.cuda.Event()

    def wait_for_last_use(self, current):
        # Calls on the same stream are already ordered. Different caller
        # streams must not overwrite the static inputs before the prior copy.
        if self.last_stream is not None and self.last_stream != current.cuda_stream:
            current.wait_event(self.last_use)

    def record_use(self, current):
        self.last_use.record(current)
        self.last_stream = current.cuda_stream

    def close(self):
        if self.closed:
            return
        with torch.cuda.device(self.device):
            if self.last_stream is not None:
                self.last_use.synchronize()
            if self.stream is not None:
                self.stream.synchronize()
            # PyTorch clears capture-stream BLAS workspaces during reset.
            # This stream is private, and all captured/replayed users are done.
            if self.graph is not None:
                self.graph.reset()
                self.graph = None
            if self.raw_stream is not None:
                status, = runtime.cudaStreamDestroy(self.raw_stream)
                if status != runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f'Cannot destroy isolated capture stream: {status}')
                self.raw_stream = None
            self.stream = None
            self.last_use = None
            self.closed = True

    def __del__(self):
        # Normal LRU retirement is explicit. This covers construction failure
        # and interpreter teardown without masking an earlier runtime error.
        try:
            self.close()
        except Exception as error:
            warnings.warn(f'Capture resource cleanup failed: {error}', ResourceWarning)
