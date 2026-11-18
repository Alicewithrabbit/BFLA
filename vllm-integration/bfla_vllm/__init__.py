"""Optional vLLM bindings for BFLA; importing this package does not load vLLM."""

__version__ = '0.3.0'


def _require_vllm031(version):
    if version.split('.')[:2] != ['0', '31']:
        raise ValueError(f'bfla-vllm requires vLLM 0.31.x, found {version}.')


def backend_target(native_backend, version):
    """Return the vLLM 0.31 backend for an explicit native implementation."""
    _require_vllm031(version)
    backends = {'flash_attn': 'FlashPrefillBackend', 'triton': 'TritonPrefillBackend'}
    if native_backend not in backends:
        raise ValueError('BFLA native backend must be flash_attn or triton')
    return 'bfla_vllm.v031.' + backends[native_backend]


def register():
    """Register the selected native backend or the Qwen3.8-Flash-Next QSA adapter."""
    import os
    qsa_enabled = os.environ.get('BFLA_QSA_ENABLED', '0') == '1'
    native_backend = os.environ.get('BFLA_NATIVE_BACKEND')
    if qsa_enabled and native_backend:
        raise ValueError('select either QSA integration or a normal native backend')
    from importlib.metadata import version
    if qsa_enabled and os.environ.get('BFLA_ENABLED', '1') != '0':
        _require_vllm031(version('vllm'))
        from .qsa.adapter import register as register_qsa
        register_qsa()
    elif native_backend:
        target = backend_target(native_backend, version('vllm'))
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
        register_backend(AttentionBackendEnum.CUSTOM, target)
        from .requests import install
        install()
    if (native_backend or qsa_enabled) and os.environ.get('BFLA_ENABLED', '1') != '0':
        from vllm.v1.core.sched.scheduler import Scheduler
        from .kv_repair import install_scheduler
        install_scheduler(Scheduler)
