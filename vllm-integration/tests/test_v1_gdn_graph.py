"""V1 graph dispatch for the first token of a GDN state."""
from importlib.util import find_spec
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch


def graph_builder(device):
    from test_v1_gdn_phase import gdn_builder
    builder = gdn_builder()
    builder.use_full_cuda_graph = True
    builder.decode_cudagraph_max_bs = 2
    builder.non_spec_state_indices_tensor = torch.empty(2, dtype=torch.int32, device=device)
    builder.non_spec_query_start_loc = torch.empty(3, dtype=torch.int32, device=device)
    return builder


def one_token(device, history, slot, is_prompt):
    from vllm.v1.attention.backend import CommonAttentionMetadata
    starts = torch.tensor([0, 1], dtype=torch.int32)
    lengths = torch.tensor([history + 1], dtype=torch.int32)
    return CommonAttentionMetadata(query_start_loc=starts.to(device), query_start_loc_cpu=starts,
        seq_lens=lengths.to(device), seq_lens_cpu_upper_bound=lengths, num_reqs=1, num_actual_tokens=1,
        max_query_len=1, max_seq_len=history + 1, is_prefilling=torch.tensor([is_prompt]),
        block_table_tensor=torch.tensor([[slot]], dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(1, dtype=torch.int64, device=device))


def graph_runner(mode, *, gdn=True, tp=1, dp=1, speculative_tokens=0, capture_sizes=(1, 2), max_seqs=2):
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from bfla_vllm.requests import install_v1

    class Runner(GPUModelRunner):
        def initialize_metadata_builders(self):
            pass  # The native builder instances are supplied below.

    install_v1(Runner)
    runner = object.__new__(Runner)
    runner.compilation_config = NS(cudagraph_mode=mode, max_cudagraph_capture_size=max(capture_sizes),
        cudagraph_capture_sizes=list(capture_sizes), compile_sizes=[], pass_config=NS(enable_sp=False),
        static_forward_context={}, is_attention_compiled_piecewise=lambda: True)
    runner.parallel_config = NS(tensor_parallel_size=tp, data_parallel_size=dp,
        data_parallel_rank=0, num_ubatches=2, disable_nccl_for_dp_synchronization=True)
    runner.vllm_config = NS(compilation_config=runner.compilation_config, num_speculative_tokens=speculative_tokens,
        lora_config=None, scheduler_config=NS(max_num_seqs=max_seqs), parallel_config=runner.parallel_config,
        observability_config=NS(cudagraph_metrics=False))
    runner.model_config = NS(is_encoder_decoder=False)
    runner.uniform_decode_query_len = 1 + speculative_tokens
    runner.cudagraph_dispatcher = CudagraphDispatcher(runner.vllm_config)
    runner.cudagraph_dispatcher.initialize_cudagraph_keys(mode, runner.uniform_decode_query_len)
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
    other = object.__new__(Mamba2AttentionMetadataBuilder)
    runner.attn_groups = [[NS(metadata_builders=[other])]]
    if gdn:
        runner.attn_groups.append([NS(metadata_builders=[graph_builder('cpu')])])
    runner.initialize_metadata_builders()
    return runner


def dispatch(runner, histories, prompts, *, native=False, counts=None, **kwargs):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    runner.input_batch = NS(num_computed_tokens_cpu=np.array(histories), num_prompt_tokens=np.array(prompts),
                            lora_id_to_lora_request={})
    counts = np.array([1] * len(histories) if counts is None else counts)
    args = (int(counts.sum()), len(counts), counts, int(counts.max()), False)
    method = (GPUModelRunner._determine_batch_execution_and_padding if native
              else type(runner)._determine_batch_execution_and_padding)
    return method(runner, *args, **kwargs)


@unittest.skipUnless(find_spec('vllm'), 'vLLM required')
class TestV1GdnGraph(unittest.TestCase):
    def test_fresh_slot_with_deployed_dispatcher_and_builder(self):
        from vllm.config import CUDAGraphMode as Mode
        builder = graph_builder('cpu')
        runner = graph_runner(Mode.FULL_DECODE_ONLY)
        for module in ('vllm.utils.torch_utils', 'vllm.v1.attention.backends.utils'):
            staging = patch(f'{module}.PIN_MEMORY', False)
            staging.start()
            self.addCleanup(staging.stop)
        capture = builder.build_for_cudagraph_capture(one_token('cpu', 0, 0, False))
        self.assertEqual((capture.num_decodes, capture.num_prefills), (1, 0))
        decode = builder.build(0, one_token('cpu', 17, 7, False))
        self.assertEqual(decode.non_spec_state_indices_tensor.data_ptr(), capture.non_spec_state_indices_tensor.data_ptr())
        self.assertEqual(capture.non_spec_state_indices_tensor.tolist(), [7])
        fresh = builder.build(0, one_token('cpu', 0, 9, True))
        self.assertEqual((fresh.num_decodes, fresh.num_prefills), (0, 1))
        self.assertEqual(fresh.prefill_has_initial_state.tolist(), [False])
        self.assertEqual(fresh.prefill_state_indices.tolist(), [9])
        self.assertEqual(capture.non_spec_state_indices_tensor.tolist(), [7])
        self.assertEqual(dispatch(runner, [0], [1], native=True)[0], Mode.FULL)
        self.assertEqual(dispatch(runner, [0], [1])[0], Mode.NONE)

    def test_first_chunks_mixed_rows_and_graph_modes(self):
        from vllm.config import CUDAGraphMode as Mode
        for mode, fresh, decode in ((Mode.FULL_DECODE_ONLY, Mode.NONE, Mode.FULL),
                                    (Mode.FULL_AND_PIECEWISE, Mode.PIECEWISE, Mode.FULL),
                                    (Mode.PIECEWISE, Mode.PIECEWISE, Mode.PIECEWISE),
                                    (Mode.NONE, Mode.NONE, Mode.NONE)):
            runner = graph_runner(mode)
            for enabled in ('0', '1'):
                with self.subTest(mode=mode, enabled=enabled), patch.dict('os.environ', {'BFLA_ENABLED': enabled}):
                    self.assertEqual(dispatch(runner, [0], [1])[0], fresh)
                    self.assertEqual(dispatch(runner, [0], [23])[0], fresh)
                    self.assertEqual(dispatch(runner, [17, 0], [4, 1])[0], fresh)
                    self.assertEqual(dispatch(runner, [17, 0], [4, 63])[0], fresh)
                    self.assertEqual(dispatch(runner, [1], [2])[0], decode)
                    self.assertEqual(dispatch(runner, [17], [4])[0], decode)
                    self.assertEqual(dispatch(runner, [0], [1], force_uniform_decode=True)[0], decode)
                    self.assertEqual(dispatch(runner, [0], [1], force_uniform_decode=False)[0], fresh)
                    self.assertEqual(dispatch(runner, [0], [1], force_eager=True)[0], Mode.NONE)
            other = graph_runner(mode, gdn=False)
            self.assertEqual(dispatch(other, [0], [1])[0], decode)

    def test_native_full_configuration_resolves_to_separate_decode_graph(self):
        from vllm.config import CompilationConfig, CUDAGraphMode as Mode
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
        support = GDNAttentionMetadataBuilder.get_cudagraph_support(None, None)
        for split, resolved in ((False, Mode.FULL_DECODE_ONLY), (True, Mode.FULL_AND_PIECEWISE)):
            ops = list(CompilationConfig._attention_ops) if split else []
            config = CompilationConfig(cudagraph_mode=Mode.FULL, splitting_ops=ops)
            self.assertEqual(config.resolve_cudagraph_mode_and_sizes(support, 'GDN_ATTN'), resolved)
            runner = graph_runner(resolved)
            self.assertEqual(dispatch(runner, [0], [1])[0], resolved.mixed_mode())
            self.assertEqual(dispatch(runner, [17], [4])[0], Mode.FULL)

    def test_tp_without_sparse_collective_and_speculative_graph_shapes(self):
        from vllm.config import CUDAGraphMode as Mode
        runner = graph_runner(Mode.FULL_DECODE_ONLY, tp=2)
        runner.compilation_config.pass_config.enable_sp = True
        with patch('vllm.distributed.get_tp_group', side_effect=AssertionError('unexpected TP collective')):
            mode, desc, *_ = dispatch(runner, [0], [1])
            self.assertEqual((mode, desc.num_tokens), (Mode.NONE, 2))
            self.assertEqual(dispatch(runner, [17], [4])[0], Mode.FULL)
        speculative = graph_runner(Mode.FULL_DECODE_ONLY, speculative_tokens=1)
        self.assertEqual(dispatch(speculative, [0], [1])[0], Mode.NONE)
        self.assertEqual(dispatch(speculative, [0], [4], counts=[2])[0], Mode.NONE)
        self.assertEqual(dispatch(speculative, [2], [4], counts=[2])[0], Mode.NONE)
        self.assertEqual(dispatch(speculative, [17], [4], counts=[2])[0], Mode.FULL)

    def test_pipeline_peer_gdn_and_sparse_eligibility_reach_attention_free_stage(self):
        from bfla_vllm.requests import _share_pipeline_graph_requirements
        from vllm.config import CUDAGraphMode as Mode

        runner = graph_runner(Mode.FULL_AND_PIECEWISE, gdn=False)
        runner.attn_groups = []
        pp = NS(world_size=2, cpu_group=object())

        def gather(stages, local, *, group):
            self.assertIs(group, pp.cpu_group)
            self.assertEqual(local, (False, (), (Mode.FULL_AND_PIECEWISE, (1, 2)), None, frozenset()))
            stages[:] = [(True, (6,), peer_graph, None, frozenset()), local]

        for peer_graph, expected, decode in (
            ((Mode.FULL_AND_PIECEWISE, (1, 2)), Mode.PIECEWISE, Mode.FULL),
            ((Mode.FULL_DECODE_ONLY, (1, 2)), Mode.NONE, Mode.FULL),
            ((Mode.FULL_AND_PIECEWISE, (2,)), Mode.NONE, Mode.NONE),
            ((Mode.NONE, ()), Mode.NONE, Mode.NONE),
        ):
            with self.subTest(peer_graph=peer_graph), patch('vllm.distributed.get_pp_group', return_value=pp), \
                    patch('torch.distributed.all_gather_object', side_effect=gather) as collective:
                runner._bfla_has_gdn = False
                _share_pipeline_graph_requirements(runner)
                self.assertEqual(dispatch(runner, [0], [1])[0], expected)
                self.assertEqual(dispatch(runner, [8192], [16384])[0], expected)
                self.assertEqual(dispatch(runner, [17], [4])[0], decode)
                collective.assert_called_once()

    def test_speculative_pipeline_decode_with_equal_buckets_but_unequal_modes(self):
        from bfla_vllm.requests import _share_pipeline_graph_requirements
        from vllm.config import CUDAGraphMode as Mode

        sizes = (2, 4, 8)
        runner = graph_runner(Mode.FULL_DECODE_ONLY, speculative_tokens=1, capture_sizes=sizes, max_seqs=8)
        pp = NS(world_size=2, cpu_group=object())
        def gather(stages, local, *, group):
            gdn, ratios, _, ordering, cascades = local
            stages[:] = [(gdn, ratios, (Mode.NONE, sizes), ordering, cascades), local]
        with patch('vllm.distributed.get_pp_group', return_value=pp), \
             patch('torch.distributed.all_gather_object', side_effect=gather) as collective:
            _share_pipeline_graph_requirements(runner)
            args = (runner, [32] * 3, [16] * 3)
            mode, desc, *_ = dispatch(*args, counts=[2, 2, 2])
            self.assertEqual((mode, desc.num_tokens), (Mode.NONE, 6))
            mode, desc, *_ = dispatch(*args, counts=[2, 2, 2], force_uniform_decode=True)
            self.assertEqual((mode, desc.num_tokens), (Mode.FULL, 8))
            collective.assert_called_once()

    def test_dp_votes_before_redispatch(self):
        from vllm.config import CUDAGraphMode as Mode
        for configured, fallback in ((Mode.FULL_DECODE_ONLY, Mode.NONE), (Mode.FULL_AND_PIECEWISE, Mode.PIECEWISE)):
            runner = graph_runner(configured, dp=2)
            for history, prompt, local, peer in ((0, 1, fallback, Mode.FULL), (17, 4, Mode.FULL, fallback)):
                def all_reduce(*, cudagraph_mode, **kwargs):
                    self.assertEqual(cudagraph_mode, local.value)
                    return torch.tensor([[1, 1], [1, 1], [0, 0], [cudagraph_mode, peer.value]])
                with self.subTest(configured=configured, history=history), patch(
                        'vllm.v1.worker.dp_utils._run_ar', side_effect=all_reduce) as vote:
                    mode, desc, _, sizes, _ = dispatch(runner, [history], [prompt], allow_microbatching=False)
                    self.assertEqual(mode, fallback)
                    self.assertEqual(desc.num_tokens, 1)
                    self.assertEqual(sizes.tolist(), [1, 1])
                    vote.assert_called_once()
