# bfla operator package

Install with `pip install ./operator` from the repository root.
See the [root README](../README.md) and [interface contract](../INTERFACE.md).

| Module | Responsibility |
|---|---|
| config.py, capabilities.py | Explicit selection policy and joint capability query |
| geometry.py, planning.py | TP mapping and causal/request support planning |
| prepared.py | Workspace reservation and run orchestration |
| metadata.py | Bounded reuse and asynchronous upload of immutable plan metadata |
| features.py, state.py | Geometry-selected features and optional incremental pools |
| _kernels/ | Triton features, scores, pooling and publication |
| _csrc/ | CUDA consumers, feature projection and CUB selectors |
| _vendor/ | Required build dependencies and upstream licenses |

Use the public `bfla` exports for configuration, capability queries and plan/run
execution. Framework adapters live in [vllm-integration/](../vllm-integration/).
