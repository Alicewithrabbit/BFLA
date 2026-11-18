"""V1 GDN phase and recycled-state parity across native microbatches."""
from importlib.util import find_spec
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch


def common_batch(device, first_history=0, flags=(True, True)):
    from vllm.v1.attention.backend import CommonAttentionMetadata
    starts = torch.tensor([0, 1, 512], dtype=torch.int32)
    lengths = torch.tensor([first_history + 1, 511], dtype=torch.int32)
    return CommonAttentionMetadata(query_start_loc=starts.to(device), query_start_loc_cpu=starts,
        seq_lens=lengths.to(device), seq_lens_cpu_upper_bound=lengths, num_reqs=2,
        num_actual_tokens=512, max_query_len=511, max_seq_len=int(lengths.max()),
        block_table_tensor=torch.tensor([[1], [2]], device=device, dtype=torch.int32),
        slot_mapping=torch.arange(512, device=device),
        is_prefilling=None if flags is None else torch.tensor(flags))


def slices():
    import numpy as np
    from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
    cuts, _ = maybe_create_ubatch_slices(True, np.array([1, 511]), 512, 2, 2)
    return cuts


def gdn_builder():
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    builder = object.__new__(GDNAttentionMetadataBuilder)
    builder.use_spec_decode = builder.use_full_cuda_graph = False
    builder.gdn_prefill_backend = 'triton'
    builder.kv_cache_spec = None
    builder.vllm_config = NS(cache_config=NS(mamba_cache_mode='none'))
    return builder


def aligned_batch(device, counts, histories, pages):
    from vllm.v1.attention.backend import CommonAttentionMetadata
    starts = torch.tensor([0, *counts]).cumsum(0, dtype=torch.int32)
    seq = torch.tensor([p + n for p, n in zip(histories, counts)], dtype=torch.int32)
    return CommonAttentionMetadata(query_start_loc=starts.to(device), query_start_loc_cpu=starts,
        seq_lens=seq.to(device), seq_lens_cpu_upper_bound=seq, num_reqs=len(counts),
        num_actual_tokens=sum(counts), max_query_len=max(counts), max_seq_len=int(seq.max()),
        block_table_tensor=torch.tensor(pages, device=device, dtype=torch.int32),
        slot_mapping=torch.arange(sum(counts), device=device), is_prefilling=torch.ones(len(counts), dtype=torch.bool))


def aligned_builder():
    from vllm.v1.kv_cache_interface import MambaSpec
    builder = gdn_builder()
    builder.kv_cache_spec = MambaSpec(block_size=256, shapes=((384, 3), (1, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32), mamba_cache_mode='align')
    builder.vllm_config.cache_config.mamba_cache_mode = 'align'
    return builder


def runner_vote(counts, histories, *, cap=1024, graph_cap=None, mode='align', dbo=True):
    import numpy as np
    from vllm.v1.worker import gpu_model_runner as v1
    from bfla_vllm.v1_masks import install
    class Runner:
        _allow_microbatching = v1.GPUModelRunner._allow_microbatching
    install(Runner)
    runner = Runner()
    runner.parallel_config = NS(use_ubatching=dbo, num_ubatches=2)
    runner.cache_config = NS(mamba_cache_mode=mode)
    runner.max_num_tokens = cap
    runner._pad_for_sequence_parallelism = lambda n: n
    runner.compilation_config = NS(max_cudagraph_capture_size=graph_cap)
    runner.is_mm_prefix_lm = False
    runner.reorder_batch_threshold = 1
    runner.input_batch = NS(num_computed_tokens_cpu=np.array(histories), block_table=NS(block_tables=[]))
    return runner._allow_microbatching(len(counts), np.array(counts))


def native_gdn_inputs(tokens, slots, *, align=False):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    layer = object.__new__(gdn.QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = 'test.gdn'
    layer.enable_packed_recurrent_decode = False
    layer.activation = 'silu'
    layer.tp_size = layer.num_k_heads = layer.num_v_heads = 1
    layer.head_k_dim = layer.head_v_dim = layer.key_dim = layer.value_dim = 128
    layer.num_spec, layer.conv_kernel_size = 0, 4
    layer.chunk_gated_delta_rule = gdn.fla_chunk_gated_delta_rule
    generator = torch.Generator(device='cuda').manual_seed(514)
    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device='cuda', generator=generator)
    layer.conv1d = NS(weight=rand(384, 1, 4) * 0.25, bias=None)
    layer.A_log = torch.zeros(1, device='cuda')
    layer.dt_bias = torch.zeros(1, device='cuda')
    qkv = rand(tokens, 384)
    b, a = rand(tokens, 1), rand(tokens, 1)
    if align:
        # Keep a long-lived recurrent signal across the second half.
        a.fill_(-15)
        b.fill_(-4)
    conv_shape, temporal_shape = layer.get_state_shape()
    stale = (rand(slots, *conv_shape), rand(slots, *temporal_shape, dtype=torch.float32))
    return layer, qkv, b, a, stale


def run_gdn(inputs, builder, parts, token_slices):
    from vllm.forward_context import ForwardContext, override_forward_context
    layer, qkv, b, a, stale = inputs
    layer.kv_cache = tuple(state.clone() for state in stale)
    outputs = torch.empty((len(qkv), 1, 128), device='cuda', dtype=qkv.dtype)
    for part, tokens in zip(parts, token_slices):
        meta = builder.build(0, part)
        ctx = ForwardContext(no_compile_layers={}, attn_metadata={layer.prefix: meta}, slot_mapping={})
        with override_forward_context(ctx):
            layer._forward_core(qkv[tokens], b[tokens], a[tokens], outputs[tokens])
    return outputs, *layer.kv_cache


@unittest.skipUnless(find_spec('vllm'), 'vLLM required')
class V1GdnPhase(unittest.TestCase):
    def test_fresh_single_token_and_history_and_capture_match_unsliced_builder(self):
        from vllm.v1.worker.ubatch_utils import split_attn_metadata
        from bfla_vllm.v1_masks import _masked_split
        # A CUDA build of vLLM enables pinned staging even for CPU destinations.
        # This metadata-only test runs on unpinned host tensors.
        for module in ('vllm.utils.torch_utils', 'vllm.v1.attention.backends.utils'):
            staging = patch(f'{module}.PIN_MEMORY', False)
            staging.start()
            self.addCleanup(staging.stop)
        build = gdn_builder().build
        for history, flags, decodes in ((0, (True, True), 0), (8, (True, True), 1),
                                        (8, (False, True), 1), (0, None, 1)):
            with self.subTest(history=history, flags=flags):
                cm = common_batch('cpu', history, flags)
                first, _ = _masked_split(split_attn_metadata)(slices(), cm)
                self.assertEqual(build(0, cm).num_decodes, decodes)
                self.assertEqual(build(0, first).num_decodes, decodes)
                if not history and flags is not None:
                    unpatched, _ = split_attn_metadata(slices(), cm)
                    self.assertEqual(build(0, unpatched).num_decodes, 1)

    def test_align_state_uses_whole_request_slot_before_dp_vote(self):
        import numpy as np
        from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices, split_attn_metadata
        for module in ('vllm.utils.torch_utils', 'vllm.v1.attention.backends.utils'):
            staging = patch(f'{module}.PIN_MEMORY', False)
            staging.start()
            self.addCleanup(staging.stop)
        build = aligned_builder().build
        for counts, histories, pages in (([1024], [0], [[0, 0, 0, 7]]),
                                         ([768], [256], [[5, 0, 0, 7]])):
            cm = aligned_batch('cpu', counts, histories, pages)
            cuts, _ = maybe_create_ubatch_slices(True, np.array(counts), sum(counts), 1, 2)
            old_slots = [build(0, part).prefill_state_indices.item()
                         for part in split_attn_metadata(cuts, cm)]
            self.assertEqual(build(0, cm).prefill_state_indices.tolist(), [7])
            self.assertEqual(old_slots, [0, 7])
            self.assertFalse(runner_vote(counts, histories))
        self.assertTrue(runner_vote([512, 512], [0, 0]))  # exact cut when local batch fills capacity
        self.assertFalse(runner_vote([512, 512], [0, 0], cap=2048))  # another DP rank can shift it
        self.assertFalse(runner_vote([512, 512], [0, 0], graph_cap=1536))
        self.assertTrue(runner_vote([1, 1], [512, 512]))
        self.assertTrue(runner_vote([1024], [0], mode='none'))
        self.assertTrue(runner_vote([1024], [0], dbo=False))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_native_gdn_outputs_and_recycled_cache_match_unsliced(self):
        from vllm.v1.worker.ubatch_utils import split_attn_metadata
        from bfla_vllm.v1_masks import _masked_split

        inputs = native_gdn_inputs(512, 3)
        cm = common_batch('cuda')
        cuts = slices()
        builder = gdn_builder()

        def run(parts, token_slices):
            out, conv, recurrent = run_gdn(inputs, builder, parts, token_slices)
            return out[0], conv[1], recurrent[1]

        expected = run([cm], [slice(0, 512)])
        token_slices = [cut.token_slice for cut in cuts]
        old = run(split_attn_metadata(cuts, cm), token_slices)
        self.assertFalse(torch.allclose(old[0], expected[0], rtol=0.01, atol=0.001))
        self.assertFalse(torch.allclose(old[2], expected[2], rtol=0.01, atol=0.001))
        corrected = run(_masked_split(split_attn_metadata)(cuts, cm), token_slices)
        if os.environ.get('BFLA_TEST_GDN_REPORT') == '1':
            for label, reference, control, candidate in zip(('attention', 'conv', 'recurrent'),
                                                           expected, old, corrected):
                print(f'{label} control_max_abs={(control-reference).abs().max().item():.9g} '
                      f'corrected_max_abs={(candidate-reference).abs().max().item():.9g}')
        for actual, reference in zip(corrected, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_aligned_native_gdn_output_and_checkpoint_ownership(self):
        import numpy as np
        from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices, split_attn_metadata
        from bfla_vllm.v1_masks import _masked_split
        split = _masked_split(split_attn_metadata)
        builder = aligned_builder()
        cases = (
            ('fresh', [1024], [0], [[0, 0, 0, 7]], False),
            ('history', [768], [256], [[5, 0, 0, 7]], False),
            ('row_boundary', [512, 512], [0, 0], [[0, 7], [0, 6]], True),
        )
        for label, counts, histories, pages, can_split in cases:
            with self.subTest(label=label):
                n = sum(counts)
                cm = aligned_batch('cuda', counts, histories, pages)
                inputs = native_gdn_inputs(n, 8, align=True)
                initial = inputs[-1]
                if histories[0]:
                    # Input to native attention after the once-per-step APC
                    # preprocessor has copied the prior checkpoint into slot 7.
                    for state in initial:
                        state[7].copy_(state[5])
                cuts, _ = maybe_create_ubatch_slices(True, np.array(counts), n, len(counts), 2)
                spans = [cut.token_slice for cut in cuts]
                whole = run_gdn(inputs, builder, [cm], [slice(n)])
                old = run_gdn(inputs, builder, split(cuts, cm), spans)
                allowed = runner_vote(counts, histories)
                self.assertEqual(allowed, can_split)
                selected, _ = maybe_create_ubatch_slices(allowed, np.array(counts), n, len(counts), 2)
                result = (run_gdn(inputs, builder, split(selected, cm), [cut.token_slice for cut in selected])
                          if selected else run_gdn(inputs, builder, [cm], [slice(n)]))
                if not can_split:
                    self.assertFalse(torch.allclose(old[0], whole[0], rtol=0.01, atol=0.001))
                    self.assertFalse(torch.allclose(old[2][7], whole[2][7], rtol=0.01, atol=0.001))
                for start, end in zip(initial, whole[1:]):
                    torch.testing.assert_close(start[0], end[0], rtol=0, atol=0)
                    self.assertFalse(torch.equal(start[7], end[7]))
                for actual, reference in zip(result, whole):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                if os.environ.get('BFLA_TEST_GDN_REPORT') == '1':
                    for name, reference, control, candidate in zip(
                            ('attention', 'conv_7', 'recurrent_7', 'null_conv', 'null_recurrent'),
                            (whole[0], whole[1][7], whole[2][7], whole[1][0], whole[2][0]),
                            (old[0], old[1][7], old[2][7], old[1][0], old[2][0]),
                            (result[0], result[1][7], result[2][7], result[1][0], result[2][0])):
                        print(f'align_{label} {name} control_max_abs={(control-reference).abs().max().item():.9g} '
                              f'corrected_max_abs={(candidate-reference).abs().max().item():.9g}')
