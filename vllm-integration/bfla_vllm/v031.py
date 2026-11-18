"""Native vLLM 0.31 backends with BFLA for supported fresh/extend prefills."""
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend, FlashAttentionImpl, FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend, TritonAttentionImpl, TritonAttentionMetadataBuilder,
)
from .backend import MetadataMixin, PrefillMixin
from .partition import take_requests, take_rows


class FlashMetadataBuilder(MetadataMixin, FlashAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def update_block_table(self, metadata, blk_table, slot_mapping):
        result = super().update_block_table(metadata, blk_table, slot_mapping)
        decode = getattr(metadata, 'bfla_native_metadata', None)
        if decode is not None:
            # The runner reuses this batch across cache groups. Native Flash
            # rebinds only the outer metadata, so rebind the decode subset too.
            rows = metadata.bfla_plan.native_rows
            result.bfla_native_metadata = super().update_block_table(
                decode, take_requests(blk_table, rows), take_rows(slot_mapping, rows))
        return result


class TritonMetadataBuilder(MetadataMixin, TritonAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Native decode stores group-sized vectors, which can be wider than
        # the model-wide head dimension in heterogeneous attention models.
        self.headdim = kv_cache_spec.head_size
        padded = 1 << (self.headdim - 1).bit_length()
        if self.softmax_segm_output.shape[-1] != padded:
            self.softmax_segm_output = self.softmax_segm_output.new_empty(
                (*self.softmax_segm_output.shape[:-1], padded))


class FlashPrefillImpl(PrefillMixin, FlashAttentionImpl):
    pass


class TritonPrefillImpl(PrefillMixin, TritonAttentionImpl):
    pass


class FlashPrefillBackend(FlashAttentionBackend):
    @staticmethod
    def get_name():
        return 'CUSTOM'

    @staticmethod
    def get_impl_cls():
        return FlashPrefillImpl

    @staticmethod
    def get_builder_cls():
        return FlashMetadataBuilder


class TritonPrefillBackend(TritonAttentionBackend):
    @staticmethod
    def get_name():
        return 'CUSTOM'

    @staticmethod
    def get_impl_cls():
        return TritonPrefillImpl

    @staticmethod
    def get_builder_cls():
        return TritonMetadataBuilder
