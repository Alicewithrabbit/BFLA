"""Process-local native-QSA-D / BFLA-P integration for vLLM 0.31."""
import logging
import os

from ..requests import current_requests
from .protocol import owned_prefills, prefill_context

logger = logging.getLogger(__name__)
DEBUG_SYNC = os.environ.get('BFLA_DEBUG_SYNC') == '1'


def debug_sync(label):
    if DEBUG_SYNC:
        import torch
        if not torch.cuda.is_current_stream_capturing():
            try:torch.cuda.synchronize()
            except Exception:
                print('BFLA_SYNC_FAILURE',label,flush=True)
                raise


def register():
    # Native model.py imports qsa.py, while qsa.py references model helpers.
    # Enter through the native model module to preserve that import order.
    from vllm.models.qwen4_exp.nvidia import model
    from vllm.models.qwen4_exp.nvidia import qsa
    if getattr(qsa.Qwen4ExpQSAAttention,'_bfla_installed',False):return
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture, eager_break_during_capture)
    original_build=qsa.Qwen4ExpQSAMetadataBuilder.build
    original_run=qsa.Qwen4ExpQSAAttention._run_qsa
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer
    # Share the ordinary admission/preemption/removal contract. This clears
    # compact pools and stale request bindings on readmission, even at equal length.
    from ..requests import install_v1, install_v2, install_v2_retirement, install_v2_ubatches
    install_v1(GPUModelRunner)
    install_v2(MambaHybridModelState)
    install_v2_retirement(GPUModelRunnerV2)
    install_v2_ubatches(UBatchRunner)
    from .decode_selection import install,decode_selection_only
    install(qsa_indexer)

    def build(self,common_prefix_len,common_attn_metadata,fast_build=False):
        result=original_build(self,common_prefix_len,common_attn_metadata,fast_build)
        common=common_attn_metadata
        context = current_requests.get()
        result.bfla_rows = ()
        upper = common.seq_lens_cpu_upper_bound
        result.bfla_partition = None
        if context is None or upper is None or common.max_query_len <= 1:
            return result
        starts = common.query_start_loc_cpu[:common.num_reqs + 1].tolist()
        lengths = upper[:common.num_reqs].tolist()
        result.bfla_rows, result.bfla_prompts, result.bfla_keys = owned_prefills(
            starts, lengths, context)
        rows = result.bfla_rows
        if rows:
            result.bfla_partition = (rows[0][1], sum(b - a for _, a, b, _ in rows))
        return result

    unchecked = object()

    def run(self,projected_qk,positions,query,key,value,output,output_gate,*,prefill=unchecked):
        if prefill is unchecked:
            from vllm.forward_context import get_forward_context
            metadata = get_forward_context().attn_metadata
            if isinstance(metadata, list):
                metadata = metadata[0]
            main = metadata[self.layer_name] if isinstance(metadata, dict) else None
            if main is None or main.max_query_len <= 1:
                result = original_run(self,projected_qk,positions,query,key,value,output,output_gate)
                if main is not None:
                    debug_sync((self.layer_name,'native_only'))
                return result
            side = metadata[self.indexer.raw_key_cache.prefix]
            prefill = prefill_context(self, query, main, side)
        if prefill is None:
            return original_run(self,projected_qk,positions,query,key,value,output,output_gate)

        main, side, heads, group, settings = prefill
        rows = main.bfla_rows
        from .consumer import consume_prefills
        n=main.num_actual_tokens
        previous=self.indexer.skip_topk
        if side.num_decode_tokens==0:self.indexer.skip_topk=True
        try:
            from .native_index import IndexInputs,update_and_select
            if isinstance(projected_qk,IndexInputs):
                update_and_select(self.indexer,projected_qk,positions,self.topk_indices_buffer[:n])
            else:
                with decode_selection_only(side.num_decode_tokens):
                    self.indexer(projected_qk,positions,self.topk_indices_buffer[:n])
        finally:
            self.indexer.skip_topk=previous
        debug_sync((self.layer_name,'indexer',rows))
        self.impl.do_kv_cache_update(self,key,value,self.kv_cache,main.slot_mapping)
        debug_sync((self.layer_name,'cache_update',rows))
        output.zero_()
        d=side.num_decode_tokens
        if d:
            from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
            kc,vc=self.kv_cache.transpose(1,2).split(self.head_dim,-1)
            qsa_sparse_paged_attention(query[:d],kc,vc,self.topk_indices_buffer[:d],
                main.block_table,side.token_to_req[:d],False,output[:d],output_gate=output_gate[:d])
            debug_sync((self.layer_name,'native_mixed_decode',rows))
        consume_prefills(query,output_gate,self.kv_cache,main.block_table,rows,
                               heads,group,self.scaling,output,
                               settings=settings,prompt_lengths=main.bfla_prompts,
                               request_keys=main.bfla_keys,
                               layer_name=self.layer_name)
        debug_sync((self.layer_name,'prefill',rows))
        self._bfla_calls = getattr(self, '_bfla_calls', 0) + 1

    def dispatch(*args, **kwargs):
        # vLLM can enable breakable graphs after plugin registration. Bind its
        # breakpoint during capture so replay re-reads the live request metadata.
        if BreakableCUDAGraphCapture.is_active():
            return eager_break_during_capture(run)(*args, **kwargs)
        return run(*args, **kwargs)

    qsa.Qwen4ExpQSAMetadataBuilder.build=build
    qsa.Qwen4ExpQSAAttention._run_qsa=dispatch
    if os.environ.get('BFLA_K_ONLY_INDEX','1')=='1':
        from .native_index import install as install_k_only
        install_k_only(qsa.Qwen4ExpQSAAttention)
    qsa.Qwen4ExpQSAAttention._bfla_installed=True
    logger.info('Enabled BFLA prefill with native QSA decode')
