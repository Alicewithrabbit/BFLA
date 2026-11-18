"""Narrow process-local QSA selection scope; native cache updates stay intact."""
from contextlib import contextmanager
from contextvars import ContextVar

_decode_rows=ContextVar('qsa_decode_selection_rows',default=None)


@contextmanager
def decode_selection_only(rows):
    token=_decode_rows.set(rows)
    try:yield
    finally:_decode_rows.reset(token)


def install(ops):
    if getattr(ops,'_bfla_decode_only_installed',False):return
    original_prefill=ops.qsa_select_paged_prefill
    original_expand=ops.expand_qsa_block_indices
    def prefill(*args,**kwargs):
        if _decode_rows.get() is None:return original_prefill(*args,**kwargs)
        # The native caller writes into its passed output; P rows are not used.
        return None
    def expand(blocks,positions,visible,ratio,budget,out):
        rows=_decode_rows.get()
        if rows is None:return original_expand(blocks,positions,visible,ratio,budget,out)
        if rows:
            return original_expand(blocks[:rows],positions[:rows],visible[:rows],
                                   ratio,budget,out[:rows])
        return out
    ops.qsa_select_paged_prefill=prefill
    ops.expand_qsa_block_indices=expand
    ops._bfla_decode_only_installed=True
