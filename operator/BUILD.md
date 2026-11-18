# Build

Build a wheel with `python -m pip wheel --no-deps ./operator`. The wheel contains
CUDA sources and pinned CUTLASS/CCCL headers. First GPU preparation compiles the
needed SM90a extension from packaged sources into PyTorch's extension cache,
keeping build products separate from the installed package.

Use a Linux SM90 environment with NVCC and a compatible C++ compiler.
Install PyTorch, Triton and the Ninja build tool with `pip install './operator[cuda]'`.

The [operator interface](../INTERFACE.md) describes the packed and paged KV paths.
