"""vLLM 0.31 metadata and prefill dispatch for the native attention backends."""
from collections import Counter
import torch
from bfla import supports
from .collectives import has_feature_reducer
from .protocol import (PrefillPlan, classify_prefill, attention_contract_reason,
                       has_prefix_mask, resolve_head_layout)
from .settings import Settings
from .requests import current_requests, sparse_prefill_disabled

COUNTS = Counter()


class MetadataMixin:
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = super().build(common_prefix_len, common_attn_metadata, fast_build)
        common = common_attn_metadata
        result.bfla_model_config = self.vllm_config.model_config
        # Each microbatch restarts its row indices; vLLM can also copy the
        # first slice's metadata for a later slice with another KV table.
        if sparse_prefill_disabled.get():
            result.bfla_plan = PrefillPlan(reason='microbatch')
            return result
        # Use the native builder's mask. A runner can supply common prompt
        # lengths even when this backend has no active R-SWA window.
        if has_prefix_mask(result):
            result.bfla_plan = PrefillPlan(reason='unsupported_attention_contract')
            return result
        result.bfla_plan = PrefillPlan(reason='decode')
        context = current_requests.get()
        if common.max_query_len > 1 or context is not None:
            upper = getattr(common, 'seq_lens_cpu_upper_bound', None)
            flags = getattr(common, 'is_prefilling', None)
            ownership = {}
            if context is not None and flags is not None:
                if flags.device.type != 'cpu':
                    raise ValueError('prefill phase flags must be framework CPU metadata')
                ownership = dict(prompt_lengths=context.prompt_lengths,
                    request_keys=context.keys, is_prefilling=context.prefill_flags(flags[:common.num_reqs].tolist()))
            result.bfla_plan = classify_prefill(
                common.query_start_loc_cpu[:common.num_reqs + 1].tolist(),
                None if upper is None else upper[:common.num_reqs].tolist(),
                getattr(common, 'causal', None),
                chunking_enabled=self.vllm_config.scheduler_config.enable_chunked_prefill,
                **ownership)
            if result.bfla_plan.reason == 'mixed_prefill':
                from .partition import native_metadata
                result.bfla_native_metadata = native_metadata(result,result.bfla_plan.native_rows)
        return result


class PrefillMixin:
    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        meta = attn_metadata
        reason = getattr(meta, 'bfla_plan', PrefillPlan()).reason
        sparse_reasons = ('fresh_prefill','extend_prefill','mixed_prefill')
        if meta is None or (meta.max_query_len <= 1 and reason not in sparse_reasons):
            if meta is not None and not torch.cuda.is_current_stream_capturing():
                COUNTS['native_decode'] += 1
            return super().forward(layer, query, key, value, kv_cache, meta,
                                   output, output_scale, output_block_scale)
        settings = Settings.from_env()
        native_reason = attention_contract_reason(self, meta,
            output_scaling=output_scale is not None or output_block_scale is not None,
            allow_cascade=reason in ('extend_prefill','mixed_prefill'))
        if not settings.enabled:
            reason = 'disabled'
        elif native_reason:
            reason = native_reason
        elif query.dtype != torch.bfloat16 or not query.is_cuda:
            reason = 'unsupported_attention_contract'
        elif getattr(self,'dcp_world_size',1) != 1:
            reason = 'context_parallelism'
        if reason in sparse_reasons:
            from vllm.distributed import get_tp_group
            group = get_tp_group()
            try:
                heads = resolve_head_layout(self, layer, meta.bfla_model_config,
                    tp_size=group.world_size, tp_rank=group.rank_in_group)
            except ValueError:
                reason = 'unresolved_head_geometry'
            else:
                capability = supports(heads, settings.config,
                    device_capability=torch.cuda.get_device_capability(query.device),
                    dtype='bfloat16', request_lengths=meta.bfla_plan.lengths,
                    kv_lengths=meta.bfla_plan.kv_lengths or None,
                    prompt_lengths=meta.bfla_plan.prompt_lengths or None)
                if not capability.supported:
                    reason = 'unsupported_operator:' + ';'.join(capability.reasons)
                elif not has_feature_reducer(group):
                    reason = 'tp_feature_reducer_unavailable'
        if reason not in sparse_reasons:
            if not torch.cuda.is_current_stream_capturing():
                COUNTS['native_decode' if meta.max_query_len <= 1 else 'native_prefill:' + reason] += 1
            return super().forward(layer, query, key, value, kv_cache, meta,
                                   output, output_scale, output_block_scale)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('BFLA prefill preparation must run outside the model CUDA graph')
        from .execution import forward
        if reason == 'mixed_prefill':
            from .partition import take_rows, write_rows
            rows = meta.bfla_plan.native_rows
            if output is None:
                output = torch.empty_like(query)
            native_query = take_rows(query,rows)
            # Contiguous D rows write directly; interleaved requests use a
            # small packed output and scatter only those D rows afterward.
            contiguous = all(a[2] == b[1] for a,b in zip(rows,rows[1:]))
            native_out = (output[rows[0][1]:rows[-1][2]] if contiguous
                          else torch.empty_like(native_query))
            super().forward(layer,native_query,take_rows(key,rows),take_rows(value,rows),
                            kv_cache,meta.bfla_native_metadata,native_out,
                            output_scale,output_block_scale)
            if not contiguous:
                write_rows(output,rows,native_out)
            COUNTS['mixed_batches'] += 1
            COUNTS['mixed_native_decode_tokens'] += sum(b-a for _,a,b,_ in rows)
        return forward(self, query, key, value, kv_cache, meta, output,
                       heads=heads, group=group, settings=settings,
                       layer_owner=getattr(layer, 'layer_name', id(layer)))
